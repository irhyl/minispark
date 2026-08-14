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

Scan pushdown (Milestone 7) is a second such exception, and runs before
any of the above translation happens: `_pushdown_scan_reads()` walks the
already-optimized logical plan once, and for every `Scan` whose `source`
is set (see logical/nodes.py's `Scan`), re-reads it with whatever
columns/filter hints the surrounding Project/Filter chain implies,
substituting the freshly (and, for a source that honors the hints, more
narrowly) read `Dataset` in place of the one `Scan` was originally built
with. `ProjectionPruning` and `PredicatePushdown` (optimizer/rules.py)
already computed exactly this information, at the *logical plan shape*
level, without touching data; this function is what turns that shape
into an actual, smaller read, for sources that can honor it (Parquet:
real column pruning and row-group-level predicate pushdown; CSV/Memory/
Checkpoint: real column pruning only, see storage/*.py). The Project and
Filter physical nodes are still built normally afterward, on top of
whatever Scan resulted: pushdown here is always an optimization
underneath them, never a substitute for their own row-level correctness
(see storage/datasource.py's `DataSource.read()` docstring).
"""

from __future__ import annotations

from minispark.core.schema import Field, Schema
from minispark.core.types import FLOAT, INT, STRING, DataType
from minispark.expressions.base import Expression
from minispark.expressions.binary import And, Multiply
from minispark.expressions.column import Column
from minispark.expressions.literal import Literal
from minispark.logical.analyzer import referenced_columns
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
    """Public entry point: run the scan-pushdown pass exactly once, over
    the whole plan, then translate.

    `_translate()` is the actual recursive translator and does *not*
    re-run `_pushdown_scan_reads()` on every recursive call: doing that
    (calling it once per node instead of once per plan) would still be
    correct, since re-reading an already-pruned Scan with no further
    hints is a safe no-op, but for a Scan that a *deeper* recursive call
    reaches with fresh hints still attached, e.g. `plan_physical(agg.
    child, ...)` inside `_plan_aggregate`, seeing `Filter(Scan)` for the
    first time from that call's perspective, it would call `DataSource.
    read()` a second time for the same data, real, wasted I/O, not just
    a wasted tree-walk. Running the pass once, here, and having every
    internal recursive site call `_translate()` instead of
    `plan_physical()`, is what avoids that.
    """
    return _translate(_pushdown_scan_reads(logical_plan), shuffle_partitions)


def _translate(logical_plan: LogicalPlan, shuffle_partitions: int) -> PhysicalPlan:
    if isinstance(logical_plan, Scan):
        return ScanExec(logical_plan.dataset, logical_plan.source_name)
    if isinstance(logical_plan, Filter):
        child = _translate(logical_plan.child, shuffle_partitions)
        return FilterExec(child, logical_plan.condition)
    if isinstance(logical_plan, Project):
        child = _translate(logical_plan.child, shuffle_partitions)
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
    left_physical = _translate(join.left, shuffle_partitions)
    right_physical = _translate(join.right, shuffle_partitions)
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
    child_physical = _translate(agg.child, shuffle_partitions)
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
    child_physical = _translate(sort.child, shuffle_partitions)
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


def _pushdown_scan_reads(plan: LogicalPlan) -> LogicalPlan:
    """Entry point for the scan-pushdown pass; see this module's
    docstring. `columns`/`filter_expr` both start `None` (no hint yet)."""
    return _walk_for_pushdown(plan, columns=None, filter_expr=None)


def _walk_for_pushdown(
    plan: LogicalPlan, columns: set[str] | None, filter_expr: Expression | None
) -> LogicalPlan:
    if isinstance(plan, Scan):
        if plan.source is None or (columns is None and filter_expr is None):
            return plan
        return _reread_scan(plan, columns, filter_expr)
    if isinstance(plan, Filter):
        # A Filter's condition is safe to carry further down (even past
        # more Filters) because it does not change column names; it is
        # combined (AND) with whatever filter_expr was already
        # accumulated, and `columns`, if a Project above already
        # restricted it, is widened to also cover this condition's own
        # referenced columns, so a narrower read never starves the
        # Filter of something it needs.
        cond_cols = referenced_columns(plan.condition)
        merged_columns = None if columns is None else (columns | cond_cols)
        combined_filter = (
            plan.condition if filter_expr is None else And(filter_expr, plan.condition)
        )
        new_child = _walk_for_pushdown(plan.child, merged_columns, combined_filter)
        return plan if new_child is plan.child else Filter(new_child, plan.condition)
    if isinstance(plan, Project):
        # `columns` is recomputed fresh at every Project, from that
        # Project's own expressions (walking into Alias/computed
        # sub-expressions via referenced_columns, not just plain Column
        # refs), so it stays correct across renames at every level.
        # `filter_expr` is deliberately *not* carried through a Project,
        # even a plain-column one: it is a whole Expression object tied
        # to whatever namespace was valid where it was written, and a
        # Project is exactly a namespace boundary (its output names need
        # not match its input names). Filter pushdown to storage
        # therefore only ever fires along a Filter-directly-on-Scan (or
        # Filter-on-Filter-on-Scan) chain, never past a Project; this
        # matches PredicatePushdown's own logical-level rule, which only
        # ever pushes a Filter below a Project when doing so is already
        # namespace-safe, so a Filter that still sits above a Project
        # after optimization is, correctly, one that could not be pushed
        # that far in the first place.
        proj_cols: set[str] = set()
        for expr in plan.columns:
            proj_cols |= referenced_columns(expr)
        new_child = _walk_for_pushdown(plan.child, proj_cols, None)
        return plan if new_child is plan.child else Project(new_child, plan.columns)
    if isinstance(plan, Aggregate):
        new_child = _walk_for_pushdown(plan.child, None, None)
        return (
            plan
            if new_child is plan.child
            else Aggregate(new_child, plan.group_by, plan.aggregates)
        )
    if isinstance(plan, Join):
        new_left = _walk_for_pushdown(plan.left, None, None)
        new_right = _walk_for_pushdown(plan.right, None, None)
        if new_left is plan.left and new_right is plan.right:
            return plan
        return Join(new_left, new_right, plan.on, plan.how, plan.broadcast)
    if isinstance(plan, Sort):
        new_child = _walk_for_pushdown(plan.child, None, None)
        return plan if new_child is plan.child else Sort(new_child, plan.sort_exprs, plan.ascending)
    raise NotImplementedError(f"scan pushdown has no rule for logical node {type(plan).__name__}")


def _reread_scan(scan: Scan, columns: set[str] | None, filter_expr: Expression | None) -> Scan:
    assert scan.source is not None
    # Defensive union, not strictly required by construction (every
    # _walk_for_pushdown Filter branch already merges its own condition's
    # columns into `columns` before recursing), but cheap and makes the
    # invariant "columns always covers whatever filter_expr needs"
    # explicit and impossible to violate by a future change here.
    if columns is not None and filter_expr is not None:
        columns = columns | referenced_columns(filter_expr)
    column_list = sorted(columns) if columns is not None else None
    dataset = scan.source.read(columns=column_list, filter=filter_expr)
    return Scan(dataset, scan.source_name, source=scan.source)
