from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.logical.nodes import Join, Scan
from minispark.physical.operators import execute_partition
from minispark.physical.plan import ExchangeExec, HashJoinExec, ScanExec
from minispark.physical.planner import plan_physical


def make_scans(left_rows, right_rows):
    left_schema = Schema([Field("id", INT), Field("name", STRING)])
    right_schema = Schema([Field("id", INT), Field("amount", INT)])
    left = Scan(
        Dataset(
            left_schema, [Partition(0, left_schema, lambda: iter(left_rows), PartitionMetadata())]
        ),
        "left",
    )
    right = Scan(
        Dataset(
            right_schema,
            [Partition(0, right_schema, lambda: iter(right_rows), PartitionMetadata())],
        ),
        "right",
    )
    return left, right


def test_shuffle_hash_join_wraps_both_sides_in_an_exchange():
    left, right = make_scans([], [])
    physical = plan_physical(Join(left, right, on=["id"]), shuffle_partitions=3)
    assert isinstance(physical, HashJoinExec)
    assert isinstance(physical.left, ExchangeExec)
    assert physical.left.num_partitions == 3
    assert physical.left.is_broadcast is False
    assert isinstance(physical.right, ExchangeExec)
    assert physical.right.num_partitions == 3


def test_broadcast_join_leaves_left_bare_and_broadcasts_right():
    left, right = make_scans([], [])
    physical = plan_physical(Join(left, right, on=["id"], broadcast=True), shuffle_partitions=3)
    assert isinstance(physical, HashJoinExec)
    assert isinstance(physical.left, ScanExec)  # unshuffled
    assert isinstance(physical.right, ExchangeExec)
    assert physical.right.num_partitions == 1
    assert physical.right.is_broadcast is True


def test_physical_schema_matches_logical_schema():
    left, right = make_scans([], [])
    join = Join(left, right, on=["id"])
    physical = plan_physical(join)
    assert physical.schema.field_names() == join.schema.field_names()


def test_hash_join_execution_is_inner_and_fans_out_duplicate_keys():
    left_rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}, {"id": 3, "name": "carol"}]
    right_rows = [
        {"id": 1, "amount": 10},
        {"id": 1, "amount": 20},
        {"id": 2, "amount": 5},
        {"id": 4, "amount": 99},
    ]
    left, right = make_scans(left_rows, right_rows)
    physical = plan_physical(Join(left, right, on=["id"]), shuffle_partitions=1)
    # Bypass the exchanges for a direct single-partition execution test
    # (execution/scheduler.py's tests cover the real shuffle path).
    direct = HashJoinExec(
        physical.left.child,
        physical.right.child,
        physical.left_keys,
        physical.right_keys,
        physical.on,
        physical.schema,
    )
    rows = execute_partition(direct, 0).to_list()
    assert sorted(rows, key=lambda r: (r["id"], r["amount"])) == [
        {"id": 1, "name": "alice", "amount": 10},
        {"id": 1, "name": "alice", "amount": 20},
        {"id": 2, "name": "bob", "amount": 5},
    ]


def test_hash_join_execution_with_no_matches_returns_no_rows():
    left, right = make_scans([{"id": 1, "name": "alice"}], [{"id": 2, "amount": 5}])
    physical = plan_physical(Join(left, right, on=["id"]), shuffle_partitions=1)
    direct = HashJoinExec(
        physical.left.child,
        physical.right.child,
        physical.left_keys,
        physical.right_keys,
        physical.on,
        physical.schema,
    )
    assert execute_partition(direct, 0).to_list() == []
