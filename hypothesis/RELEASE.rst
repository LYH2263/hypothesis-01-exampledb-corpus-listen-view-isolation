RELEASE_TYPE: minor

This release adds :func:`~hypothesis.database.database_view_isolation`, a
host-level context for example database corpus commit and listener view
isolation.

Within an active scope, engine sessions pin their own
:class:`~hypothesis.database.DatabaseView` of the configured database at
construction. The three corpus commit planes stay distinct - primary-key
``save``, secondary-key downgrade ``move``, and pareto-key ``save``/
``delete`` - and the write and listening semantics of the pinned database
(synchronous in-memory broadcast, post-hoc directory watching, read-only
swallowing, multiplexed fan-out, or backgrounded commits) are preserved.

Advancing the :class:`~hypothesis.database.DatabaseViewScope` freezes
already bound databases, listener views, and running engine sessions against
later database swaps, wrapper stacking, or listener-set changes; only new
binds and new sessions take effect under the advanced view. Without an
active scope, behavior is unchanged.
