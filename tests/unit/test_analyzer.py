import pytest

from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.logical.analyzer import AnalysisException, analyze
from minispark.logical.nodes import Filter, Project, Scan


def make_scan():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [{"name": "a", "age": 20}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
    dataset = Dataset(schema, [partition])
    return Scan(dataset, "users")


def test_analyze_valid_plan_returns_it_unchanged():
    scan = make_scan()
    plan = Project(Filter(scan, col("age") > 18), [col("name")])
    assert analyze(plan) is plan


def test_analyze_rejects_missing_column_in_select():
    scan = make_scan()
    plan = Project(scan, [col("does_not_exist")])
    with pytest.raises(AnalysisException, match="does_not_exist"):
        analyze(plan)


def test_analyze_rejects_missing_column_in_filter():
    scan = make_scan()
    plan = Filter(scan, col("does_not_exist") > 1)
    with pytest.raises(AnalysisException, match="does_not_exist"):
        analyze(plan)


def test_analyze_rejects_missing_column_nested_in_expression():
    scan = make_scan()
    plan = Filter(scan, (col("age") > 18) & (col("missing") == 1))
    with pytest.raises(AnalysisException, match="missing"):
        analyze(plan)


def test_analyze_rejects_duplicate_alias():
    scan = make_scan()
    plan = Project(scan, [col("age").alias("x"), col("name").alias("x")])
    with pytest.raises(AnalysisException, match="Duplicate output column 'x'"):
        analyze(plan)


def test_analyze_allows_same_column_selected_twice_with_different_aliases():
    scan = make_scan()
    plan = Project(scan, [col("age").alias("a1"), col("age").alias("a2")])
    assert analyze(plan) is plan
