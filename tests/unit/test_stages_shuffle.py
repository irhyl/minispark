from minispark.api.functions import col, count
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.execution.stages import build_stages
from minispark.logical.nodes import Aggregate
from minispark.logical.nodes import Filter as LogicalFilter
from minispark.logical.nodes import Scan as LogicalScan
from minispark.physical.plan import HashAggregateExec, ShuffleReadExec, ShuffleWriteExec
from minispark.physical.planner import plan_physical
from minispark.storage.memory import MemoryDataSource


def make_aggregate_plan(num_source_partitions=3, shuffle_partitions=5):
    schema = Schema([Field("country", STRING), Field("revenue", INT)])
    rows = [{"country": "US", "revenue": 1}]
    partitions = [
        Partition(i, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
        for i in range(num_source_partitions)
    ]
    scan = LogicalScan(Dataset(schema, partitions), "sales")
    agg = Aggregate(scan, [col("country")], [count("*").alias("n")])
    return plan_physical(agg, shuffle_partitions=shuffle_partitions)


def test_aggregate_plan_splits_into_two_stages():
    physical = make_aggregate_plan()
    stages = build_stages(physical)
    assert len(stages) == 2


def test_first_stage_ends_in_shuffle_write_with_source_partition_count():
    physical = make_aggregate_plan(num_source_partitions=3, shuffle_partitions=5)
    stages = build_stages(physical)
    write_stage = stages[0]
    assert isinstance(write_stage.plan, ShuffleWriteExec)
    assert write_stage.num_partitions == 3
    assert write_stage.plan.num_partitions == 5
    assert isinstance(write_stage.plan.child, HashAggregateExec)
    assert write_stage.plan.child.is_partial is True


def test_second_stage_starts_with_shuffle_read_at_target_partition_count():
    physical = make_aggregate_plan(num_source_partitions=3, shuffle_partitions=5)
    stages = build_stages(physical)
    read_stage = stages[1]
    assert isinstance(read_stage.plan, HashAggregateExec)
    assert read_stage.plan.is_partial is False
    assert isinstance(read_stage.plan.child, ShuffleReadExec)
    assert read_stage.plan.child.from_stage_id == 0
    assert read_stage.num_partitions == 5


def test_stage_ids_are_sequential():
    physical = make_aggregate_plan()
    stages = build_stages(physical)
    assert [s.stage_id for s in stages] == [0, 1]


def test_exchange_free_plan_is_returned_without_rebuilding_nodes():
    """Regression test: build_stages() used to unconditionally rebuild
    every node on the way back up the recursion, even when there was no
    Exchange anywhere below (an unnecessary allocation, and it broke
    identity-based assertions). A Scan/Filter/Project-only plan must come
    back out as the exact same object, not a value-equal copy."""
    dataset = MemoryDataSource([{"a": 1}], num_partitions=1).read()
    logical = LogicalFilter(LogicalScan(dataset, "mem"), col("a") > 0)
    physical = plan_physical(logical)
    (stage,) = build_stages(physical)
    assert stage.plan is physical
