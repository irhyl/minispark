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

Join is the second: `left.join(right, on=..., broadcast=...)` becomes one
`HashJoinExec`, whose two children are wrapped differently depending on
`broadcast`. Not broadcast (the default, a shuffle hash join): both sides
get their own `ExchangeExec`, hash-partitioned by the join key, so
matching keys from either side land in the same target partition.
Broadcast: only the (smaller, hinted) side gets an `ExchangeExec`, with
`num_partitions=1` and `is_broadcast=True`; the other side is left exactly
as `plan_physical()` produced it, unshuffled, so its original partition
count is what the resulting join stage runs with.

Sort is the third, and the odd one out: `order_by(...)` becomes a local
`SortExec`, a *range* `ExchangeExec` (see physical/plan.py), and a final
`SortExec`, per the build spec's "local sort, range partition, shuffle,
final local sort" shape. Choosing where to cut the sort key's range into
`shuffle_partitions` buckets needs to know something about the data
*before* the shuffle that needs those cut points runs; `_sort_range_
boundaries()` gets that by eagerly executing the child plan and scanning
the sort key's min/max right here, at planning time. This is a real,
deliberate exception to "building a plan never touches data" (true of
every other node here), scoped and documented in that function.
"""

from __future__ import annotations

from minispark.core.schema import Field, Schema
from minispark.core.types import FLOAT, INT, STRING, DataType
from minispark.expressions.base import Expression
from minispark.expressions.binary import Multiply
from minispark.expressions.column import Column
from minispark.expressions.literal import Literal
from minispark.logical.nodes import (
    Aggregate,
    Filter,
    Join,
    LogicalPlan,
    Project,
    Scan,
    Sort,
    output_name,
)
from minispark.optimizer.statistics import compute_statistics
from minispark.physical.operators import execute as execute_whole_dataset
from minispark.physical.plan import (
    ExchangeExec,
    FilterExec,
    HashAggregateExec,
    HashJoinExec,
    PhysicalPlan,
    ProjectExec,
    ScanExec,
    SortExec,
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
    if isinstance(logical_plan, Join):
        return _plan_join(logical_plan, shuffle_partitions)
    if isinstance(logical_plan, Sort):
        return _plan_sort(logical_plan, shuffle_partitions)
    raise NotImplementedError(
        f"No physical strategy implemented for logical node {type(logical_plan).__name__}"
    )


def _plan_join(join: Join, shuffle_partitions: int) -> PhysicalPlan:
    left_physical = plan_physical(join.left, shuffle_partitions)
    right_physical = plan_physical(join.right, shuffle_partitions)
    left_keys: list[Expression] = [Column(name) for name in join.on]
    right_keys: list[Expression] = [Column(name) for name in join.on]

    if join.broadcast:
        left_side = left_physical
        right_side: PhysicalPlan = ExchangeExec(
            right_physical, num_partitions=1, partition_exprs=right_keys, is_broadcast=True
        )
    else:
        left_side = ExchangeExec(left_physical, shuffle_partitions, left_keys)
        right_side = ExchangeExec(right_physical, shuffle_partitions, right_keys)

    return HashJoinExec(left_side, right_side, left_keys, right_keys, join.on, join.schema)


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


def _plan_sort(sort: Sort, shuffle_partitions: int) -> PhysicalPlan:
    child_physical = plan_physical(sort.child, shuffle_partitions)
    primary_key = sort.sort_exprs[0]
    primary_ascending = sort.ascending[0]
    # The analyzer (logical/analyzer.py's _analyze_sort) already rejected
    # anything but a plain Column here; this is a defensive check of that
    # invariant, not new validation.
    assert isinstance(primary_key, Column)
    boundaries, num_partitions, partition_key = _sort_range_boundaries(
        child_physical, primary_key, primary_ascending, sort.child.schema, shuffle_partitions
    )
    local_sort = SortExec(child_physical, sort.sort_exprs, sort.ascending, sort.child.schema)
    exchange = ExchangeExec(
        local_sort, num_partitions, [partition_key], range_boundaries=boundaries
    )
    return SortExec(exchange, sort.sort_exprs, sort.ascending, sort.schema)


def _sort_range_boundaries(
    child_physical: PhysicalPlan,
    primary_key: Column,
    ascending: bool,
    child_schema: Schema,
    shuffle_partitions: int,
) -> tuple[list | None, int, Expression]:
    """Compute `order_by()`'s shuffle boundaries by eagerly executing
    `child_physical` and scanning the sort key column
    (optimizer/statistics.py's `compute_statistics()`), right now, at
    physical planning time.

    This is a deliberate, documented exception to "building a plan never
    touches data" (true of every other node in this module): there is no
    distributed sampling stage (see docs/shuffle.md), so the only way to
    know where to cut the sort key's range into `shuffle_partitions`
    buckets is to look at the data before the shuffle that needs those
    cut points runs. `child_physical` must therefore be something
    `physical/operators.py`'s whole-Dataset `execute()` can run directly
    (Scan/Filter/Project): sorting the output of a Join or an Aggregate
    is not supported, `execute()` cannot run those, they need a real
    shuffle to mean anything, which does not exist yet at planning time.

    Also returns the expression the shuffle should actually partition on
    (`partition_key`): `RangePartitioner` (shuffle/partitioner.py) always
    assigns *ascending* target partitions (partition 0 gets the smallest
    keys), and the final merge (execution/scheduler.py) always reads
    partitions back in id order. For a descending sort that combination
    would put the smallest-keyed, internally-descending partition first,
    a locally sorted but globally wrong result. Rather than teach the
    (deliberately Sort-agnostic) scheduler to reverse partition order,
    `descending` negates the partitioning key (`-value`) and computes
    boundaries over the negated range instead: negation is exact for the
    numeric types range partitioning is used for at all (see the
    fallback paragraph below), and it keeps every downstream piece
    (RangePartitioner, the scheduler's merge order) completely unaware
    that direction is even a concept.

    Falls back to a single shuffle partition (`(None, 1, primary_key)`,
    meaning a plain, boundary-less exchange, equivalent to
    `HashPartitioner(1)` where every row lands in the one and only target
    partition anyway) whenever a real multi-bucket range split is not
    meaningful: `shuffle_partitions <= 1`, the sort key's type is not
    numeric (equal-*width* bucketing of, say, a string's range is not
    implemented), or the observed data has no non-null values to compute
    a range from. A single partition is always correct, just not
    parallel: there is only one target, so no row needs to land in a
    specific range relative to any other partition's rows for the final
    per-partition sort (and single-partition read order) to be globally
    sorted.
    """
    key_name = primary_key.name
    key_type = (
        child_schema.get_field(key_name).data_type if child_schema.has_field(key_name) else None
    )
    if shuffle_partitions <= 1 or key_type not in (INT, FLOAT):
        return None, 1, primary_key

    dataset = execute_whole_dataset(child_physical)
    stats = compute_statistics(dataset, columns=[key_name])
    lo = stats.columns[key_name].min_value
    hi = stats.columns[key_name].max_value
    if lo is None or hi is None:
        return None, 1, primary_key

    sign = 1 if ascending else -1
    partition_key: Expression = primary_key if ascending else Multiply(primary_key, Literal(-1))
    signed_lo, signed_hi = sorted([sign * lo, sign * hi])
    if signed_lo == signed_hi:
        boundaries = [signed_hi] * (shuffle_partitions - 1)
        return boundaries, shuffle_partitions, partition_key
    step = (signed_hi - signed_lo) / shuffle_partitions
    boundaries = [signed_lo + step * i for i in range(1, shuffle_partitions)]
    return boundaries, shuffle_partitions, partition_key
