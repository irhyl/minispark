from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.logical.nodes import Scan, Sort
from minispark.physical.operators import execute_partition
from minispark.physical.plan import ExchangeExec, ScanExec, SortExec
from minispark.physical.planner import plan_physical


def make_scan(rows, partitions=1):
    schema = Schema([Field("name", STRING), Field("age", INT)])
    chunk = -(-len(rows) // partitions) if rows else 1
    parts = []
    for i in range(partitions):
        chunk_rows = rows[i * chunk : (i + 1) * chunk]
        parts.append(Partition(i, schema, lambda r=chunk_rows: iter(r), PartitionMetadata()))
    return Scan(Dataset(schema, parts), "users"), schema


def test_plan_shape_is_local_sort_range_exchange_final_sort():
    scan, _ = make_scan([{"name": "a", "age": 1}])
    physical = plan_physical(Sort(scan, [col("age")], [True]), shuffle_partitions=3)
    assert isinstance(physical, SortExec)
    exchange = physical.child
    assert isinstance(exchange, ExchangeExec)
    local_sort = exchange.child
    assert isinstance(local_sort, SortExec)
    assert isinstance(local_sort.child, ScanExec)


def test_boundaries_are_equal_width_over_observed_range():
    rows = [{"name": "a", "age": 0}, {"name": "b", "age": 100}]
    scan, _ = make_scan(rows, partitions=1)
    physical = plan_physical(Sort(scan, [col("age")], [True]), shuffle_partitions=4)
    exchange = physical.child
    assert exchange.range_boundaries == [25.0, 50.0, 75.0]
    assert exchange.num_partitions == 4


def test_descending_sort_negates_the_partition_key_and_boundaries():
    """Regression test: RangePartitioner always assigns ascending target
    partitions (partition 0 = smallest keys) and the scheduler always
    merges partitions back in id order, so a naive (non-negated) boundary
    computation would put the smallest-keyed, internally-descending
    partition first: locally sorted, globally wrong. Negating the key
    for a descending sort is what fixes that (see physical/planner.py's
    _sort_range_boundaries)."""
    rows = [{"name": "a", "age": 0}, {"name": "b", "age": 100}]
    scan, _ = make_scan(rows, partitions=1)
    physical = plan_physical(Sort(scan, [col("age")], [False]), shuffle_partitions=4)
    exchange = physical.child
    # boundaries computed over the negated range [-100, 0]
    assert exchange.range_boundaries == [-75.0, -50.0, -25.0]
    (partition_expr,) = exchange.partition_exprs
    from minispark.expressions.binary import Multiply

    assert isinstance(partition_expr, Multiply)


def test_single_partition_fallback_for_non_numeric_key():
    scan, _ = make_scan([{"name": "b", "age": 1}, {"name": "a", "age": 2}])
    physical = plan_physical(Sort(scan, [col("name")], [True]), shuffle_partitions=4)
    exchange = physical.child
    assert exchange.range_boundaries is None
    assert exchange.num_partitions == 1


def test_single_partition_fallback_when_shuffle_partitions_is_one():
    scan, _ = make_scan([{"name": "a", "age": 1}])
    physical = plan_physical(Sort(scan, [col("age")], [True]), shuffle_partitions=1)
    exchange = physical.child
    assert exchange.range_boundaries is None
    assert exchange.num_partitions == 1


def test_local_sort_execution_ascending_with_nulls_last():
    schema = Schema([Field("age", INT)])
    rows = [{"age": 3}, {"age": None}, {"age": 1}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata())
    scan_exec = ScanExec(Dataset(schema, [partition]), "test")
    sort_exec = SortExec(scan_exec, [col("age")], [True], schema)
    result = execute_partition(sort_exec, 0).to_list()
    assert result == [{"age": 1}, {"age": 3}, {"age": None}]


def test_local_sort_execution_descending_with_nulls_last():
    schema = Schema([Field("age", INT)])
    rows = [{"age": 3}, {"age": None}, {"age": 1}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata())
    scan_exec = ScanExec(Dataset(schema, [partition]), "test")
    sort_exec = SortExec(scan_exec, [col("age")], [False], schema)
    result = execute_partition(sort_exec, 0).to_list()
    assert result == [{"age": 3}, {"age": 1}, {"age": None}]


def test_local_sort_execution_multi_key_mixed_direction():
    schema = Schema([Field("age", INT), Field("name", STRING)])
    rows = [
        {"age": 30, "name": "b"},
        {"age": 20, "name": "z"},
        {"age": 30, "name": "a"},
        {"age": 20, "name": "y"},
    ]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata())
    scan_exec = ScanExec(Dataset(schema, [partition]), "test")
    sort_exec = SortExec(scan_exec, [col("age"), col("name")], [True, False], schema)
    result = execute_partition(sort_exec, 0).to_list()
    assert result == [
        {"age": 20, "name": "z"},
        {"age": 20, "name": "y"},
        {"age": 30, "name": "b"},
        {"age": 30, "name": "a"},
    ]
