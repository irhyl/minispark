"""Physical operator execution: turns a PhysicalPlan into a Dataset.

Deliberately mirrors execution/executor.py's per-node logic almost exactly
for Scan/Filter/Project: there is only one way to execute those, so "the
physical strategy" and "the naive logical interpretation" happen to
compute the same thing. HashAggregateExec and ShuffleReadExec have no
equivalent in execution/executor.py at all (the naive executor only knows
Scan/Filter/Project): grouping fundamentally needs a shuffle to be
correct across partitions, which is exactly what Milestone 3's
single-process, single-stage naive executor cannot do.
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
    PhysicalPlan,
    ProjectExec,
    ScanExec,
    ShuffleReadExec,
)
from minispark.shuffle.reader import read_shuffle_blocks
from minispark.shuffle.writer import ShuffleBlockMeta


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
    shuffle_blocks: list[ShuffleBlockMeta] | None = None,
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
    single source partition at the ScanExec leaf instead of reading
    `plan.dataset` in full.

    `shuffle_blocks` is only meaningful when `plan` contains a
    `ShuffleReadExec` (a stage that reads a prior stage's shuffle output):
    the caller (execution/worker.py) already knows exactly which blocks
    belong to this partition (filtered driver-side by
    `shuffle/manager.py`'s `ShuffleManager.blocks_for()`) and passes that
    fixed list in, rather than this function looking anything up itself.
    """
    if isinstance(plan, ScanExec):
        return plan.dataset.partition(partition_id)
    if isinstance(plan, FilterExec):
        return _execute_filter_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, ProjectExec):
        return _execute_project_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, HashAggregateExec):
        return _execute_hash_aggregate_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, ShuffleReadExec):
        return _execute_shuffle_read_partition(plan, partition_id, shuffle_blocks)
    raise NotImplementedError(
        f"No per-partition physical operator implemented for {type(plan).__name__}"
    )


def _execute_filter_partition(
    plan: FilterExec, partition_id: int, shuffle_blocks: list[ShuffleBlockMeta] | None
) -> Partition:
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    condition = plan.condition

    def records_fn() -> Iterator[Record]:
        for record in parent:
            if condition.evaluate(record):
                yield record

    return Partition(parent.partition_id, parent.schema, records_fn, PartitionMetadata())


def _execute_project_partition(
    plan: ProjectExec, partition_id: int, shuffle_blocks: list[ShuffleBlockMeta] | None
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
    plan: HashAggregateExec, partition_id: int, shuffle_blocks: list[ShuffleBlockMeta] | None
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
    plan: ShuffleReadExec, partition_id: int, shuffle_blocks: list[ShuffleBlockMeta] | None
) -> Partition:
    if shuffle_blocks is None:
        raise ValueError(
            f"ShuffleReadExec for stage {plan.from_stage_id} partition {partition_id} "
            "was executed with no shuffle_blocks; the caller must supply the block "
            "list for this partition (see execution/worker.py)."
        )
    records = list(read_shuffle_blocks(shuffle_blocks))
    return Partition(
        partition_id,
        plan.schema,
        functools.partial(iter, records),
        PartitionMetadata(row_count=len(records)),
    )
