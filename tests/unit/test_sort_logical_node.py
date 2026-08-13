import pytest

from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.logical.analyzer import AnalysisException, analyze
from minispark.logical.nodes import Scan, Sort


def make_scan():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [{"name": "a", "age": 20}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
    return Scan(Dataset(schema, [partition]), "users")


def test_schema_is_unchanged_from_child():
    scan = make_scan()
    sort = Sort(scan, [col("age")], [True])
    assert sort.schema.field_names() == scan.schema.field_names()


def test_rejects_mismatched_sort_exprs_and_ascending_lengths():
    scan = make_scan()
    with pytest.raises(ValueError, match="same length"):
        Sort(scan, [col("age"), col("name")], [True])


def test_node_label_shows_direction_per_column():
    scan = make_scan()
    sort = Sort(scan, [col("age"), col("name")], [True, False])
    assert sort.node_label == "Sort[age ASC, name DESC]"


def test_analyze_accepts_valid_sort():
    scan = make_scan()
    sort = Sort(scan, [col("age")], [True])
    assert analyze(sort) is sort


def test_analyze_rejects_non_column_sort_expression():
    scan = make_scan()
    sort = Sort(scan, [col("age") > 0], [True])
    with pytest.raises(AnalysisException, match="order_by\\(\\) only accepts column names"):
        analyze(sort)


def test_analyze_rejects_missing_sort_column():
    scan = make_scan()
    sort = Sort(scan, [col("does_not_exist")], [True])
    with pytest.raises(AnalysisException, match="does_not_exist"):
        analyze(sort)
