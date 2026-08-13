from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.optimizer.statistics import compute_statistics


def make_dataset():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [
        {"name": "alice", "age": 30},
        {"name": "bob", "age": None},
        {"name": "alice", "age": 45},
    ]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=len(rows)))
    return Dataset(schema, [partition])


def test_row_count_is_exact():
    stats = compute_statistics(make_dataset())
    assert stats.row_count == 3


def test_null_count_is_exact():
    stats = compute_statistics(make_dataset())
    assert stats.columns["age"].null_count == 1
    assert stats.columns["name"].null_count == 0


def test_min_max_ignore_nulls():
    stats = compute_statistics(make_dataset())
    assert stats.columns["age"].min_value == 30
    assert stats.columns["age"].max_value == 45


def test_distinct_count_is_exact():
    stats = compute_statistics(make_dataset())
    assert stats.columns["name"].distinct_count == 2  # "alice" appears twice


def test_estimated_size_is_positive_and_not_precise():
    stats = compute_statistics(make_dataset())
    assert stats.estimated_size_bytes > 0


def test_columns_argument_restricts_which_columns_get_statistics():
    stats = compute_statistics(make_dataset(), columns=["name"])
    assert list(stats.columns.keys()) == ["name"]
