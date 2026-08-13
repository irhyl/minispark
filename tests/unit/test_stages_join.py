from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.execution.stages import build_stages
from minispark.logical.nodes import Join, Scan
from minispark.physical.plan import HashJoinExec, ShuffleReadExec, ShuffleWriteExec
from minispark.physical.planner import plan_physical


def make_scans(left_partitions=3, right_partitions=2):
    left_schema = Schema([Field("id", INT), Field("name", STRING)])
    right_schema = Schema([Field("id", INT), Field("amount", INT)])
    left = Scan(
        Dataset(
            left_schema,
            [
                Partition(i, left_schema, lambda: iter([]), PartitionMetadata())
                for i in range(left_partitions)
            ],
        ),
        "left",
    )
    right = Scan(
        Dataset(
            right_schema,
            [
                Partition(i, right_schema, lambda: iter([]), PartitionMetadata())
                for i in range(right_partitions)
            ],
        ),
        "right",
    )
    return left, right


def test_shuffle_hash_join_produces_three_stages():
    left, right = make_scans()
    physical = plan_physical(Join(left, right, on=["id"]), shuffle_partitions=4)
    stages = build_stages(physical)
    assert len(stages) == 3
    assert isinstance(stages[0].plan, ShuffleWriteExec)
    assert isinstance(stages[1].plan, ShuffleWriteExec)
    assert isinstance(stages[2].plan, HashJoinExec)
    assert stages[0].num_partitions == 3  # left's source partitions
    assert stages[1].num_partitions == 2  # right's source partitions
    assert stages[2].num_partitions == 4  # the shuffle target count
    left_read, right_read = stages[2].plan.left, stages[2].plan.right
    assert isinstance(left_read, ShuffleReadExec) and left_read.from_stage_id == 0
    assert isinstance(right_read, ShuffleReadExec) and right_read.from_stage_id == 1
    assert left_read.is_broadcast is False
    assert right_read.is_broadcast is False


def test_broadcast_join_produces_two_stages():
    left, right = make_scans(left_partitions=3, right_partitions=2)
    physical = plan_physical(Join(left, right, on=["id"], broadcast=True), shuffle_partitions=4)
    stages = build_stages(physical)
    assert len(stages) == 2
    assert isinstance(stages[0].plan, ShuffleWriteExec)
    assert stages[0].plan.num_partitions == 1
    assert isinstance(stages[1].plan, HashJoinExec)
    # the join stage runs with the large (left) side's partition count,
    # not the broadcast side's (which is always 1)
    assert stages[1].num_partitions == 3
    right_read = stages[1].plan.right
    assert isinstance(right_read, ShuffleReadExec)
    assert right_read.is_broadcast is True
    assert right_read.from_stage_id == 0


def test_join_stage_ids_are_sequential():
    left, right = make_scans()
    physical = plan_physical(Join(left, right, on=["id"]), shuffle_partitions=4)
    stages = build_stages(physical)
    assert [s.stage_id for s in stages] == [0, 1, 2]
