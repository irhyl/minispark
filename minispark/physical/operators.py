"""Physical operator execution: turns a PhysicalPlan into a Dataset.

Deliberately mirrors execution/executor.py's per-node logic almost exactly.
That duplication is the honest state of things right now: there is only
one way to execute a Scan, a Filter, or a Project, so "the physical
strategy" and "the naive logical interpretation" happen to compute the
same thing. The duplication is expected to stop being a duplication once
Milestone 4/5 add operators with more than one strategy (HashAggregate vs
SortAggregate, HashJoin vs BroadcastJoin) that execution/executor.py's
logical-node interpreter has no way to choose between. Until then,
execution/executor.py stays as-is, used directly only by tests that check
this module against it.
"""

from __future__ import annotations

from collections.abc import Iterator

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.physical.plan import FilterExec, PhysicalPlan, ProjectExec, ScanExec


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


def execute_partition(plan: PhysicalPlan, partition_id: int) -> Partition:
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
    """
    if isinstance(plan, ScanExec):
        return plan.dataset.partition(partition_id)
    if isinstance(plan, FilterExec):
        return _execute_filter_partition(plan, partition_id)
    if isinstance(plan, ProjectExec):
        return _execute_project_partition(plan, partition_id)
    raise NotImplementedError(
        f"No per-partition physical operator implemented for {type(plan).__name__}"
    )


def _execute_filter_partition(plan: FilterExec, partition_id: int) -> Partition:
    parent = execute_partition(plan.child, partition_id)
    condition = plan.condition

    def records_fn() -> Iterator[Record]:
        for record in parent:
            if condition.evaluate(record):
                yield record

    return Partition(parent.partition_id, parent.schema, records_fn, PartitionMetadata())


def _execute_project_partition(plan: ProjectExec, partition_id: int) -> Partition:
    parent = execute_partition(plan.child, partition_id)
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
