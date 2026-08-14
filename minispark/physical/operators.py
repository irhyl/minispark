"""Physical operator execution: turns a PhysicalPlan into a Dataset.

Deliberately mirrors execution/executor.py's per-node logic almost exactly
for Scan/Filter/Project: there is only one way to execute those, so "the
physical strategy" and "the naive logical interpretation" happen to
compute the same thing. HashAggregateExec, HashJoinExec, and
ShuffleReadExec have no equivalent in execution/executor.py at all (the
naive executor only knows Scan/Filter/Project): grouping and joining
fundamentally need a shuffle to be correct across partitions, which is
exactly what Milestone 3's single-process, single-stage naive executor
cannot do.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.expressions.aggregate import AggregateFunction
from minispark.expressions.base import Alias, Expression
from minispark.logical.nodes import output_name
from minispark.physical.plan import (
    FilterExec,
    HashAggregateExec,
    HashJoinExec,
    PhysicalPlan,
    ProjectExec,
    ScanExec,
    ShuffleReadExec,
    SortExec,
)
from minispark.shuffle.reader import (
    MissingShuffleDataError,
    ShuffleChecksumError,
    read_shuffle_blocks,
)
from minispark.shuffle.writer import ShuffleBlockMeta

# Keyed by source stage_id: a Task whose plan reads from more than one
# prior stage (a HashJoinExec-rooted stage reads from two) needs each
# ShuffleReadExec leaf to find only its own stage's blocks, not the
# other leaf's. See execution/scheduler.py for how this dict is built.
ShuffleBlocksByStage = dict[int, list[ShuffleBlockMeta]]


def execute(plan: PhysicalPlan) -> Dataset:
    if isinstance(plan, ScanExec):
        return plan.dataset
    if isinstance(plan, FilterExec):
        return _execute_filter(plan)
    if isinstance(plan, ProjectExec):
        return _execute_project(plan)
    raise NotImplementedError(
        f"No physical operator implemented for {type(plan).__name__}"
    )


def _execute_filter(plan: FilterExec) -> Dataset:
    child_dataset = execute(plan.child)
    condition = plan.condition

    def filtered_partition(parent: Partition) -> Partition:
        def records_fn() -> Iterator[Record]:
            for record in parent:
                if condition.evaluate(record):
                    yield record

        return Partition(parent.partition_id, parent.schema, records_fn, PartitionMetadata())

    new_partitions = [filtered_partition(p) for p in child_dataset.partitions()]
    return Dataset(child_dataset.schema, new_partitions)


def _execute_project(plan: ProjectExec) -> Dataset:
    child_dataset = execute(plan.child)
    columns = plan.columns
    output_schema = plan.schema
    output_names = output_schema.field_names()

    def projected_partition(parent: Partition) -> Partition:
        def records_fn() -> Iterator[Record]:
            for record in parent:
                yield {
                    name: expr.evaluate(record)
                    for name, expr in zip(output_names, columns, strict=True)
                }

        return Partition(
            parent.partition_id,
            output_schema,
            records_fn,
            PartitionMetadata(row_count=parent.row_count()),
        )

    new_partitions = [projected_partition(p) for p in child_dataset.partitions()]
    return Dataset(output_schema, new_partitions)


def execute_partition(
    plan: PhysicalPlan,
    partition_id: int,
    shuffle_blocks: ShuffleBlocksByStage | None = None,
) -> Partition:
    """Execute physical operators for exactly one partition.

    This, not `execute()`, is what a Task actually runs (see
    execution/tasks.py, execution/worker.py). A Task should ship "run this
    plan for partition i" to a worker, not "run this plan for every
    partition and send all of it back": only one partition's rows need to
    cross the process boundary as a result. `plan` itself is safe to send
    whole (every physical node and expression is plain, picklable data,
    and Milestone 3's fix to storage/memory.py and storage/csv.py made
    Dataset/Partition picklable too); `execute_partition` picks out a
    single source partition at each `ScanExec` leaf instead of reading
    `plan.dataset` in full.

    `shuffle_blocks` is only meaningful when `plan` contains one or more
    `ShuffleReadExec` leaves (a stage that reads a prior stage's shuffle
    output; a `HashJoinExec`-rooted stage can have two, one per side): the
    caller (execution/worker.py) already knows exactly which blocks
    belong to this partition for each upstream stage (filtered driver-side
    by `shuffle/manager.py`'s `ShuffleManager.blocks_for()`) and passes
    that fixed, per-stage mapping in, rather than this function looking
    anything up itself.
    """
    if isinstance(plan, ScanExec):
        return plan.dataset.partition(partition_id)
    if isinstance(plan, FilterExec):
        return _execute_filter_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, ProjectExec):
        return _execute_project_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, HashAggregateExec):
        return _execute_hash_aggregate_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, HashJoinExec):
        return _execute_hash_join_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, SortExec):
        return _execute_sort_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, ShuffleReadExec):
        return _execute_shuffle_read_partition(plan, partition_id, shuffle_blocks)
    raise NotImplementedError(
        f"No per-partition physical operator implemented for {type(plan).__name__}"
    )


def _execute_filter_partition(
    plan: FilterExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    condition = plan.condition

    def records_fn() -> Iterator[Record]:
        for record in parent:
            if condition.evaluate(record):
                yield record

    return Partition(parent.partition_id, parent.schema, records_fn, PartitionMetadata())


def _execute_project_partition(
    plan: ProjectExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    columns = plan.columns
    output_schema = plan.schema
    output_names = output_schema.field_names()

    def records_fn() -> Iterator[Record]:
        for record in parent:
            yield {
                name: expr.evaluate(record)
                for name, expr in zip(output_names, columns, strict=True)
            }

    return Partition(
        parent.partition_id,
        output_schema,
        records_fn,
        PartitionMetadata(row_count=parent.row_count()),
    )


def _unwrap_aggregate(expr: Expression) -> AggregateFunction:
    inner = expr.child if isinstance(expr, Alias) else expr
    assert isinstance(inner, AggregateFunction)
    return inner


def _execute_hash_aggregate_partition(
    plan: HashAggregateExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    """Group this one partition's rows by `plan.group_by` and combine
    `plan.aggregates` per group.

    Unlike Filter/Project, this cannot stream row-by-row: a group's final
    state is only known once every row for that key (within this
    partition) has been seen, so the `groups` table below is built up
    front, not lazily inside `records_fn`. This is the same reason a real
    hash-based group-by needs a hash table sized to the *distinct key*
    count, not the row count; spilling that table to disk under memory
    pressure (build spec section 22) is not implemented (Milestone 9).
    """
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    group_by = plan.group_by
    aggregates = [_unwrap_aggregate(a) for a in plan.aggregates]
    group_names = [output_name(g) for g in group_by]

    groups: dict[tuple, list] = {}
    for record in parent:
        key = tuple(g.evaluate(record) for g in group_by)
        if plan.is_partial:
            state = groups.get(key) or [agg.initialize() for agg in aggregates]
            groups[key] = [
                agg.update(s, record) for agg, s in zip(aggregates, state, strict=True)
            ]
        else:
            incoming = [record[f"__agg_state_{i}"] for i in range(len(aggregates))]
            existing = groups.get(key)
            groups[key] = (
                incoming
                if existing is None
                else [
                    agg.merge(s, inc)
                    for agg, s, inc in zip(aggregates, existing, incoming, strict=True)
                ]
            )

    output_schema = plan.schema
    agg_output_names = [output_name(a) for a in plan.aggregates]

    def records_fn() -> Iterator[Record]:
        for key, state in groups.items():
            row: Record = dict(zip(group_names, key, strict=True))
            if plan.is_partial:
                for i, s in enumerate(state):
                    row[f"__agg_state_{i}"] = s
            else:
                for name, agg, s in zip(agg_output_names, aggregates, state, strict=True):
                    row[name] = agg.finalize(s)
            yield row

    return Partition(
        partition_id, output_schema, records_fn, PartitionMetadata(row_count=len(groups))
    )


def _execute_shuffle_read_partition(
    plan: ShuffleReadExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    if shuffle_blocks is None or plan.from_stage_id not in shuffle_blocks:
        raise ValueError(
            f"ShuffleReadExec for stage {plan.from_stage_id} partition {partition_id} "
            "was executed with no blocks for that stage; the caller must supply a "
            "shuffle_blocks[stage_id] entry for every ShuffleReadExec in the plan "
            "(see execution/worker.py, execution/scheduler.py)."
        )
    blocks = shuffle_blocks[plan.from_stage_id]
    try:
        records = list(read_shuffle_blocks(blocks))
    except (FileNotFoundError, ShuffleChecksumError) as exc:
        # A block file this partition needs is gone or corrupted: retrying
        # this same read can never succeed, since the data simply is not
        # there any more. Re-raised as a distinct error type so
        # execution/worker.py and execution/scheduler.py can tell "the
        # upstream stage needs to be recomputed" apart from an ordinary,
        # possibly-transient task failure.
        raise MissingShuffleDataError(plan.from_stage_id, partition_id, str(exc)) from exc
    return Partition(
        partition_id,
        plan.schema,
        functools.partial(iter, records),
        PartitionMetadata(row_count=len(records)),
    )


def _execute_hash_join_partition(
    plan: HashJoinExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    """Build a hash table on `left`'s rows (keyed by `left_keys`), then
    probe it with `right`'s rows (keyed by `right_keys`), for this one
    partition. By the time this runs, `left` and `right` already produce
    exactly the rows this partition's join needs to see, whether that is
    because both sides were shuffle-partitioned by the same key (shuffle
    hash join) or because `right` is a full broadcast copy read
    identically by every partition (broadcast join); see
    physical/planner.py for which case built this node's children.

    Building on `left` (rather than choosing the smaller side) is a fixed
    choice, not a cost-based one: nothing here has a reliable, cheap
    byte-size estimate for either side to choose from (see
    optimizer/statistics.py's documented caveats and logical/nodes.py's
    `Join` docstring on why broadcast side selection is an explicit hint,
    not automatic).
    """
    left_partition = execute_partition(plan.left, partition_id, shuffle_blocks)
    right_partition = execute_partition(plan.right, partition_id, shuffle_blocks)
    left_keys = plan.left_keys
    right_keys = plan.right_keys
    on_set = set(plan.on)

    table: dict[tuple, list[Record]] = {}
    for record in left_partition:
        key = tuple(e.evaluate(record) for e in left_keys)
        table.setdefault(key, []).append(record)

    right_rows = list(right_partition)
    output_schema = plan.schema

    def records_fn() -> Iterator[Record]:
        for right_row in right_rows:
            key = tuple(e.evaluate(right_row) for e in right_keys)
            for left_row in table.get(key, ()):
                merged = dict(left_row)
                for name, value in right_row.items():
                    if name not in on_set:
                        merged[name] = value
                yield merged

    return Partition(partition_id, output_schema, records_fn, PartitionMetadata())


def _execute_sort_partition(
    plan: SortExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    """Sort this one partition's rows by `plan.sort_exprs`/`plan.ascending`.

    Like HashAggregateExec, this cannot stream: the whole partition must
    be seen before any row's final position is known, so rows are
    materialized up front. Multi-key, mixed ascending/descending order is
    achieved with repeated stable sorts, last key first (Python's `sort`
    is guaranteed stable, so an earlier pass's relative order survives
    for rows that tie on a later, higher-priority key): a single
    `sorted(key=...)` call sorting on a tuple of keys cannot vary
    direction per key without either negating values (which breaks for
    non-numeric types like strings) or a custom comparator (removed from
    Python 3's `sorted`).

    Nulls always sort last, regardless of ascending/descending: a
    deliberate, documented simplification rather than implementing
    per-column NULLS FIRST/LAST placement.
    """
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    rows = list(parent)
    for expr, ascending in reversed(list(zip(plan.sort_exprs, plan.ascending, strict=True))):
        key_fn = functools.partial(_null_last_sort_key, expr, ascending)
        rows.sort(key=key_fn, reverse=not ascending)
    return Partition(
        partition_id,
        plan.schema,
        functools.partial(iter, rows),
        PartitionMetadata(row_count=len(rows)),
    )


def _null_last_sort_key(expr: Expression, ascending: bool, record: Record) -> tuple[bool, object]:
    value = expr.evaluate(record)
    is_null = value is None
    # `rows.sort(..., reverse=not ascending)` flips the *whole* key tuple,
    # including whichever boolean marks a null, not just the value part.
    # Pre-flipping the sentinel here (rather than always using `is_null`)
    # is what keeps nulls sorting last in the final output for a
    # descending column too, not only for an ascending one.
    null_sentinel = is_null if ascending else not is_null
    return (null_sentinel, value if not is_null else 0)
