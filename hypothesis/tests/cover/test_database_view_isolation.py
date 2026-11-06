# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Same-process acceptance tests for host-level example database corpus
commit and listener view isolation (``database_view_isolation``).

Everything here runs against the same observable database bindings in one
process. The tests contrast, on the same machinery:

* the three corpus commit planes (primary-key save, secondary-key downgrade
  move, pareto-key save/delete),
* synchronous in-memory broadcasts vs. post-hoc directory listening,
* read-only write swallowing vs. multiplexed fan-out writes,
* background-write fetch-vs-listener commit timing,

and then assert that advancing isolation freezes already bound databases,
listener views, and running engine sessions against later swaps, wrapper
stacking, and listener-set changes.
"""

import contextlib
import threading
import time as time_module
from collections import Counter
from random import Random

import pytest

from hypothesis import HealthCheck, Phase, settings
from hypothesis.database import (
    BackgroundWriteDatabase,
    DatabaseView,
    DirectoryBasedExampleDatabase,
    InMemoryExampleDatabase,
    MultiplexedDatabase,
    ReadOnlyDatabase,
    bind_database_view,
    choices_to_bytes,
    database_view_isolation,
)
from hypothesis.internal.conjecture.data import Status
from hypothesis.internal.conjecture.engine import ConjectureRunner, RunIsComplete

from tests.common.utils import wait_for
from tests.conjecture.common import interesting_origin

# conftest monkeypatches time.sleep; capture the real sleep for observer
# startup, just like test_database_backend.py does.
time_sleep = time_module.sleep


class RecordingDatabase(InMemoryExampleDatabase):
    """Records every corpus plane operation without changing semantics."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def fetch(self, key):
        self.calls.append(("fetch", key))
        yield from super().fetch(key)

    def save(self, key, value):
        self.calls.append(("save", key, value))
        super().save(key, value)

    def delete(self, key, value):
        self.calls.append(("delete", key, value))
        super().delete(key, value)

    def move(self, src, dest, value):
        self.calls.append(("move", src, dest, value))
        super().move(src, dest, value)


def _runner(db, *, phases=None, max_examples=10, database_key=b"test-key"):
    def test(data):
        if data.draw_integer(0, 2**8 - 1) == 10:
            data.mark_interesting(interesting_origin())
        else:
            data.mark_invalid()

    return ConjectureRunner(
        test,
        settings=settings(
            max_examples=max_examples,
            database=db,
            phases=list(Phase) if phases is None else phases,
            suppress_health_check=list(HealthCheck),
        ),
        database_key=database_key,
        random=Random(0),
    )


# ---------------------------------------------------------------------------
# Binding: disabled is pin-identical; enabled pins one view per db/generation
# ---------------------------------------------------------------------------


def test_without_isolation_engine_uses_the_live_settings_database():
    db = InMemoryExampleDatabase()
    runner = _runner(db)
    assert runner._database_view is None
    assert runner.database is db
    assert bind_database_view(db) is None
    assert bind_database_view(None) is None


def test_isolation_binds_one_pinned_view_per_database_and_generation():
    db = InMemoryExampleDatabase()
    with database_view_isolation() as scope:
        runner = _runner(db)
        view = runner.database
        assert isinstance(view, DatabaseView)
        assert view._db is db
        assert view._generation == 0
        # repeated binds in one generation share the same pinned view
        assert scope.bind(db) is view
        assert bind_database_view(db) is view
        assert scope.bind(None) is None

        scope.advance()
        assert view._generation == 0  # the old view is not rewritten
        new_view = scope.bind(db)
        assert new_view is not view
        assert new_view._db is db
        assert new_view._generation == 1

    # leaving the context restores pin behavior
    assert bind_database_view(db) is None


# ---------------------------------------------------------------------------
# Corpus key fork: primary save / secondary move / pareto save+delete stay
# three distinct commit exits, with per-plane cleanup rules
# ---------------------------------------------------------------------------


def test_engine_corpus_commit_chain_keeps_three_distinct_planes():
    db = RecordingDatabase()
    with database_view_isolation():
        runner = _runner(db)
        data = runner.cached_test_function((10,))
        assert data.status == Status.INTERESTING
        buf = choices_to_bytes(data.choices)

        runner.save_choices(data.choices)
        runner.downgrade_choices(data.choices)
        runner.save_choices(data.choices, sub_key=b"pareto")
        runner.on_pareto_evict(data.as_result())

    # the three corpus keys are distinct bytes and the operations on them
    # never collapsed into a single save exit
    assert len({runner.database_key, runner.secondary_key, runner.pareto_key}) == 3
    assert ("save", runner.database_key, buf) in db.calls
    assert ("move", runner.database_key, runner.secondary_key, buf) in db.calls
    assert ("save", runner.pareto_key, buf) in db.calls
    assert ("delete", runner.pareto_key, buf) in db.calls

    # observable end state: the downgrade move relocated primary corpus onto
    # the secondary plane, and the pareto eviction removed the pareto entry
    assert buf not in db.data.get(runner.database_key, ())
    assert buf in db.data[runner.secondary_key]
    assert buf not in db.data.get(runner.pareto_key, ())


def test_reuse_phase_mixes_primary_and_secondary_with_per_key_cleanup():
    db = RecordingDatabase()
    with database_view_isolation():
        runner = _runner(db, phases=[Phase.reuse])
        junk = b"\xff\xff not decodable choices"
        db.save(runner.database_key, junk)
        db.save(runner.secondary_key, choices_to_bytes((10,)))  # interesting
        db.save(runner.secondary_key, choices_to_bytes((5,)))  # invalid

        with contextlib.suppress(RunIsComplete):
            runner.reuse_existing_test_cases()

    fetched = {key for op, key, *rest in db.calls if op == "fetch"}
    assert runner.database_key in fetched
    assert runner.secondary_key in fetched

    delete_keys = {key for op, key, *rest in db.calls if op == "delete"}
    # undecodable primary entry is cleared off the primary plane...
    assert runner.database_key in delete_keys
    # ...and invalid secondary entries are cleared off both planes
    assert runner.secondary_key in delete_keys
    # the interesting secondary entry survives its own plane
    assert choices_to_bytes((10,)) in db.data[runner.secondary_key]


def test_reuse_prunes_pareto_plane_without_crossing_into_other_planes():
    db = RecordingDatabase()

    def test(data):
        data.target_observations[""] = 1
        data.draw_integer(0, 2**8 - 1)
        data.draw_integer(0, 2**8 - 1)

    with database_view_isolation():
        runner = ConjectureRunner(
            test,
            settings=settings(
                max_examples=10000,
                database=db,
                suppress_health_check=list(HealthCheck),
                phases=[Phase.reuse],
            ),
            database_key=b"pareto-stuff",
            random=Random(0),
        )
        for i in range(256):
            db.save(runner.pareto_key, choices_to_bytes((i, 0)))
        db.calls.clear()

        runner.run()

    assert len(list(db.fetch(runner.pareto_key))) == 1
    delete_ops = [call for call in db.calls if call[0] == "delete"]
    assert delete_ops
    assert all(key == runner.pareto_key for _, key, _ in delete_ops)
    fetched = {key for op, key, *rest in db.calls if op == "fetch"}
    assert runner.pareto_key in fetched
    # pareto cleanup never deletes from the primary or secondary planes
    assert runner.database_key not in {key for _, key, _ in delete_ops}
    assert runner.secondary_key not in {key for _, key, _ in delete_ops}


def test_full_run_under_isolation_routes_all_commits_through_the_view():
    db = InMemoryExampleDatabase()
    with database_view_isolation():
        runner = _runner(db, max_examples=100)
        view = runner.database
        db.save(runner.database_key, choices_to_bytes((10,)))
        runner.run()
        assert runner.interesting_test_cases
        assert runner.database is view  # the binding never moved mid-session
        assert list(db.fetch(runner.database_key))


# ---------------------------------------------------------------------------
# Timing fork: in-memory broadcast is synchronous; directory writes are only
# observed after the fact via the filesystem listener
# ---------------------------------------------------------------------------


def test_inmemory_view_listener_receives_events_synchronously():
    events = []
    with database_view_isolation() as scope:
        db = InMemoryExampleDatabase()
        view = scope.bind(db)
        view.add_listener(events.append)
        view.save(b"k", b"v")
        # the broadcast already happened before save() returned
        assert events == [("save", (b"k", b"v"))]
        view.delete(b"k", b"v")
        assert events[-1] == ("delete", (b"k", b"v"))


def test_directory_writes_are_never_broadcast_synchronously(tmp_path):
    with database_view_isolation() as scope:
        db = DirectoryBasedExampleDatabase(tmp_path)
        view = scope.bind(db)
        synchronous = []
        db._broadcast_change = synchronous.append
        view.save(b"k", b"v")
        view.move(b"k", b"k2", b"v")
        view.delete(b"k2", b"v")
        # unlike the in-memory database, the write path itself is silent;
        # directory events arrive only through the post-hoc file watcher
        assert synchronous == []


def test_directory_view_delivers_post_hoc_events_and_survives_clear(tmp_path):
    pytest.importorskip("watchdog")
    events = []
    with database_view_isolation() as scope:
        db = DirectoryBasedExampleDatabase(tmp_path)
        view = scope.bind(db)
        view.add_listener(events.append)
        time_sleep(0.1)

        # public listener management must not tear down the view's relay
        db.clear_listeners()
        assert db._observer is not None
        assert db._observer.is_alive()

        view.save(b"k", b"v")
        wait_for(lambda: ("save", (b"k", b"v")) in events, timeout=30)

        view.remove_listener(events.append)
        assert db._observer is None


# ---------------------------------------------------------------------------
# Wrapper fork: read-only silently swallows writes; multiplex fans them out
# ---------------------------------------------------------------------------


def test_readonly_view_swallows_writes_and_forges_no_events():
    inner = InMemoryExampleDatabase()
    events = []
    with database_view_isolation() as scope:
        view = scope.bind(ReadOnlyDatabase(inner))
        view.add_listener(events.append)
        view.save(b"k", b"v")
        view.delete(b"k", b"v")
        view.move(b"k", b"k2", b"v")
        assert events == []  # no forged successful-write event
        assert not inner.data  # nothing landed
        assert list(view.fetch(b"k")) == []  # fetch still passes through


def test_multiplexed_view_fans_out_writes_and_rebroadcasts_leg_changes():
    a = InMemoryExampleDatabase()
    b = InMemoryExampleDatabase()
    events = []
    with database_view_isolation() as scope:
        view = scope.bind(MultiplexedDatabase(a, b))
        view.add_listener(events.append)

        view.save(b"k", b"v")
        assert set(a.fetch(b"k")) == {b"v"}
        assert set(b.fetch(b"k")) == {b"v"}
        # each leg's change is rebroadcast upward through the view
        wait_for(lambda: len(events) == 2)
        assert Counter(events) == {("save", (b"k", b"v")): 2}

        # fetch deduplicates the legs
        assert set(view.fetch(b"k")) == {b"v"}

        view.delete(b"k", b"v")
        wait_for(lambda: ("delete", (b"k", b"v")) in events)
        assert set(a.fetch(b"k")) == set()
        assert set(b.fetch(b"k")) == set()


class _GatedDatabase(InMemoryExampleDatabase):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release_gate = threading.Event()

    def save(self, key, value):
        self.entered.set()
        assert self.release_gate.wait(timeout=30)
        super().save(key, value)


def test_background_write_fetch_waits_but_listener_follows_inner_commit():
    gated = _GatedDatabase()
    bg = BackgroundWriteDatabase(gated)
    events = []
    with database_view_isolation() as scope:
        view = scope.bind(bg)
        view.add_listener(events.append)

        view.save(b"k", b"v")  # enqueued, not yet committed
        wait_for(gated.entered.is_set)
        # a queued write is not committed corpus: no event, nothing landed
        assert events == []
        assert set(gated.fetch(b"k")) == set()

        gated.release_gate.set()
        # fetch must wait for the enqueued write to commit
        assert set(view.fetch(b"k")) == {b"v"}
        # the listener only observes the inner post-commit event, exactly once
        assert events == [("save", (b"k", b"v"))]


# ---------------------------------------------------------------------------
# Advancing isolation: old bindings, listener views, and engine sessions must
# not be written through by later swaps, rewrapping, or listener-set changes
# ---------------------------------------------------------------------------


def test_advance_freezes_old_views_while_new_binds_see_current_databases():
    db1 = InMemoryExampleDatabase()
    db2 = InMemoryExampleDatabase()
    with database_view_isolation() as scope:
        view1 = scope.bind(db1)
        scope.advance()
        view1b = scope.bind(db1)
        assert view1b is not view1
        assert view1._db is db1
        assert view1b._db is db1

        view2 = scope.bind(db2)
        view1.save(b"k", b"old")
        view2.save(b"k", b"new")
        assert set(db1.fetch(b"k")) == {b"old"}
        assert set(db2.fetch(b"k")) == {b"new"}


def test_public_listener_changes_cannot_write_through_a_bound_view():
    db = InMemoryExampleDatabase()
    view_events = []
    public_events = []
    with database_view_isolation() as scope:
        view = scope.bind(db)
        view.add_listener(view_events.append)
        scope.advance()

        # the host rewires the public listener set after the view was bound
        db.clear_listeners()
        db.add_listener(public_events.append)
        db.save(b"k", b"v")
        assert view_events == [("save", (b"k", b"v"))]
        assert public_events == [("save", (b"k", b"v"))]

        db.clear_listeners()
        db.save(b"k", b"w")
        assert view_events[-1] == ("save", (b"k", b"w"))
        # the detached public listener saw nothing new
        assert public_events == [("save", (b"k", b"v"))]

        view.remove_listener(view_events.append)
        db.save(b"k", b"z")
        assert len(view_events) == 2


class _CountingDatabase(InMemoryExampleDatabase):
    def __init__(self):
        super().__init__()
        self.starts = 0
        self.stops = 0

    def _start_listening(self):
        self.starts += 1

    def _stop_listening(self):
        self.stops += 1


def test_view_and_public_listeners_share_one_start_stop_cycle():
    db = _CountingDatabase()
    with database_view_isolation() as scope:
        view = scope.bind(db)
        view.add_listener(lambda event: None)
        assert (db.starts, db.stops) == (1, 0)

        db.add_listener(lambda event: None)
        assert (db.starts, db.stops) == (1, 0)

        db.clear_listeners()  # the view relay keeps the carrier alive
        assert (db.starts, db.stops) == (1, 0)

        # a second view over the same database attaches (and detaches) while
        # another relay is already present, so the carrier is not cycled
        scope.advance()
        view2 = scope.bind(db)
        view2.add_listener(lambda event: None)
        assert (db.starts, db.stops) == (1, 0)
        view2.clear_listeners()
        assert (db.starts, db.stops) == (1, 0)

        view.clear_listeners()
        assert (db.starts, db.stops) == (1, 1)

        # detaching an unknown relay is a silent no-op
        db._detach_view_listener(lambda event: None)
        assert (db.starts, db.stops) == (1, 1)


def test_rewrapping_after_bind_does_not_write_through_the_old_view():
    db = InMemoryExampleDatabase()
    with database_view_isolation() as scope:
        view = scope.bind(db)
        scope.advance()
        # the host stacks fresh wrappers around the same inner database
        ReadOnlyDatabase(db)
        MultiplexedDatabase(db, InMemoryExampleDatabase())

        view.save(b"k", b"v")
        # the old view still writes the inner db directly: not swallowed,
        # not duplicated by the new multiplexer
        assert set(db.fetch(b"k")) == {b"v"}
        view.delete(b"k", b"v")
        assert set(db.fetch(b"k")) == set()


def test_engine_sessions_are_not_written_through_after_advance():
    db1 = InMemoryExampleDatabase()
    db2 = InMemoryExampleDatabase()
    with database_view_isolation() as scope:
        runner1 = _runner(db1)
        scope.advance()
        runner2 = _runner(db2, database_key=b"other-key")  # new session, new view
        assert runner1.database is not runner2.database

        data1 = runner1.cached_test_function((10,))
        data2 = runner2.cached_test_function((10,))
        runner1.save_choices(data1.choices)
        runner2.save_choices(data2.choices)

        buf1 = choices_to_bytes(data1.choices)
        buf2 = choices_to_bytes(data2.choices)
        assert buf1 in db1.data[runner1.database_key]
        assert runner1.database_key not in db2.data
        assert buf2 in db2.data[runner2.database_key]
        assert runner2.database_key not in db1.data
