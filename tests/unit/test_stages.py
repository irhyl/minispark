from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.execution.stages import build_stages
from minispark.logical.nodes import Filter, Project, Scan
from minispark.physical.planner import plan_physical


def make_physical_plan(num_partitions=3):
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [{"name": "a", "age": 20}]
    partitions = [
        Partition(i, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
        for i in range(num_partitions)
    ]
    scan = Scan(Dataset(schema, partitions), "users")
    logical = Project(Filter(scan, col("age") > 18), [col("name")])
    return plan_physical(logical)


def test_build_stages_returns_exactly_one_stage():
    plan = make_physical_plan()
    stages = build_stages(plan)
    assert len(stages) == 1


def test_stage_holds_the_whole_plan():
    plan = make_physical_plan()
    (stage,) = build_stages(plan)
    assert stage.plan is plan


def test_stage_partition_count_matches_scan_partitions():
    plan = make_physical_plan(num_partitions=5)
    (stage,) = build_stages(plan)
    assert stage.num_partitions == 5


def test_stage_id_starts_at_zero():
    plan = make_physical_plan()
    (stage,) = build_stages(plan)
    assert stage.stage_id == 0
