"""Physical planner: translates an (analyzed, optimized) LogicalPlan into a PhysicalPlan.

Scan, Filter, and Project translate 1:1, one physical node per logical
node, because each has exactly one execution strategy today.

Aggregate is the first place this stops being a 1:1 copy: `group_by(...).
agg(...)` becomes a partial (map-side) `HashAggregateExec`, an
`ExchangeExec` (a shuffle boundary; `execution/stages.py` splits the plan
into stages here), and a final (reduce-side) `HashAggregateExec` that
merges the partial results per group. This is the same "local partial
aggregate, then shuffle, then final aggregate" shape the build spec asks
for, so `group_by(country).sum(revenue)` shuffles one partial sum per
(source partition, country) pair instead of every raw row.
"""

from __future__ import annotations

from minispark.core.schema import Field, Schema
from minispark.core.types import STRING, DataType
from minispark.expressions.base import Expression
from minispark.expressions.column import Column
from minispark.logical.nodes import Aggregate, Filter, LogicalPlan, Project, Scan, output_name
from minispark.physical.plan import (
    ExchangeExec,
    FilterExec,
    HashAggregateExec,
    PhysicalPlan,
    ProjectExec,
    ScanExec,
)

DEFAULT_SHUFFLE_PARTITIONS = 4


def plan_physical(
    logical_plan: LogicalPlan, shuffle_partitions: int = DEFAULT_SHUFFLE_PARTITIONS
) -> PhysicalPlan:
    if isinstance(logical_plan, Scan):
        return ScanExec(logical_plan.dataset, logical_plan.source_name)
    if isinstance(logical_plan, Filter):
        child = plan_physical(logical_plan.child, shuffle_partitions)
        return FilterExec(child, logical_plan.condition)
    if isinstance(logical_plan, Project):
        child = plan_physical(logical_plan.child, shuffle_partitions)
        return ProjectExec(child, logical_plan.columns, logical_plan.schema)
    if isinstance(logical_plan, Aggregate):
        return _plan_aggregate(logical_plan, shuffle_partitions)
    raise NotImplementedError(
        f"No physical strategy implemented for logical node {type(logical_plan).__name__}"
    )


def _plan_aggregate(agg: Aggregate, shuffle_partitions: int) -> PhysicalPlan:
    child_physical = plan_physical(agg.child, shuffle_partitions)
    partial = HashAggregateExec(
        child_physical,
        agg.group_by,
        agg.aggregates,
        schema=_partial_aggregate_schema(agg.group_by, agg.aggregates, agg.child.schema),
        is_partial=True,
    )
    exchange = ExchangeExec(partial, shuffle_partitions, agg.group_by)
    return HashAggregateExec(
        exchange, agg.group_by, agg.aggregates, schema=agg.schema, is_partial=False
    )


def _partial_aggregate_schema(
    group_by: list[Expression], aggregates: list[Expression], child_schema: Schema
) -> Schema:
    group_fields = [
        Field(output_name(g), _column_type(g, child_schema), nullable=True) for g in group_by
    ]
    # Partial-state columns are opaque internal values (e.g. Avg's
    # (sum, count) tuple state, see expressions/aggregate.py); STRING is a
    # placeholder type here, never validated or shown to a user. The
    # user-facing types come from the *final* HashAggregateExec's schema,
    # which is the logical Aggregate node's already-computed `.schema`
    # (see logical/nodes.py's `_aggregate_output_field`), passed straight
    # through above rather than re-derived.
    state_fields = [
        Field(f"__agg_state_{i}", STRING, nullable=True) for i in range(len(aggregates))
    ]
    return Schema(group_fields + state_fields)


def _column_type(expr: Expression, schema: Schema) -> DataType:
    if isinstance(expr, Column) and schema.has_field(expr.name):
        return schema.get_field(expr.name).data_type
    return STRING
