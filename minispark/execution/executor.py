"""NaiveExecutor: directly interprets a logical plan into a Dataset.

There is deliberately no optimizer, physical plan, DAG, or scheduler
between the logical plan and this executor yet — those are Milestones 2
and 3. This is the smallest thing that can make
`df.filter(...).select(...).collect()` produce correct results, so that
later milestones have a correctness baseline to compare against (e.g. the
optimizer's rewritten plan must still produce the same rows as this naive
walk would).
"""

from __future__ import annotations

from collections.abc import Iterator

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.logical.nodes import Filter, LogicalPlan, Project, Scan


def execute(plan: LogicalPlan) -> Dataset:
    if isinstance(plan, Scan):
        return plan.dataset
    if isinstance(plan, Filter):
        return _execute_filter(plan)
    if isinstance(plan, Project):
        return _execute_project(plan)
    raise NotImplementedError(
        f"NaiveExecutor cannot execute logical node of type {type(plan).__name__}"
    )


def _execute_filter(plan: Filter) -> Dataset:
    child_dataset = execute(plan.child)
    condition = plan.condition

    def filtered_partition(parent: Partition) -> Partition:
        def records_fn() -> Iterator[Record]:
            for record in parent:
                if condition.evaluate(record):
                    yield record

        # Row count is unknown without scanning (filtering is selective),
        # so metadata deliberately omits row_count rather than guessing.
        return Partition(parent.partition_id, parent.schema, records_fn, PartitionMetadata())

    new_partitions = [filtered_partition(p) for p in child_dataset.partitions()]
    return Dataset(child_dataset.schema, new_partitions)


def _execute_project(plan: Project) -> Dataset:
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
