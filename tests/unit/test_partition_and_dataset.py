from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT


def make_partition(pid, rows):
    schema = Schema([Field("n", INT)])
    return Partition(pid, schema, lambda: iter(rows), PartitionMetadata(row_count=len(rows)))


def test_partition_reiterable():
    p = make_partition(0, [{"n": 1}, {"n": 2}])
    assert list(p) == [{"n": 1}, {"n": 2}]
    # calling iter twice must re-derive rows from the factory, not exhaust a
    # single stored iterator
    assert list(p) == [{"n": 1}, {"n": 2}]


def test_dataset_num_partitions_and_row_count():
    p0 = make_partition(0, [{"n": 1}, {"n": 2}])
    p1 = make_partition(1, [{"n": 3}])
    schema = Schema([Field("n", INT)])
    ds = Dataset(schema, [p0, p1])

    assert ds.num_partitions() == 2
    assert ds.row_count() == 3
    assert [r["n"] for r in ds.iter_records()] == [1, 2, 3]


def test_dataset_repartition_preserves_all_records():
    schema = Schema([Field("n", INT)])
    partitions = [make_partition(i, [{"n": i}]) for i in range(6)]
    ds = Dataset(schema, partitions)

    repartitioned = ds.repartition(2)

    assert repartitioned.num_partitions() == 2
    all_values = sorted(r["n"] for r in repartitioned.iter_records())
    assert all_values == [0, 1, 2, 3, 4, 5]
