import pytest

from minispark.api.functions import avg, col, count
from minispark.api.functions import sum as ssum
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import FLOAT, INT, STRING
from minispark.logical.analyzer import AnalysisException, analyze
from minispark.logical.nodes import Aggregate, Scan


def make_scan():
    schema = Schema([Field("country", STRING), Field("revenue", INT)])
    rows = [{"country": "US", "revenue": 10}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
    return Scan(Dataset(schema, [partition]), "sales")


def test_schema_has_group_by_and_aggregate_output_fields():
    scan = make_scan()
    agg = Aggregate(scan, [col("country")], [count("*").alias("n"), ssum("revenue").alias("total")])
    assert agg.schema.field_names() == ["country", "n", "total"]
    assert agg.schema.get_field("country").data_type == STRING
    assert agg.schema.get_field("n").data_type == INT
    assert agg.schema.get_field("total").data_type == INT


def test_avg_output_type_is_float():
    scan = make_scan()
    agg = Aggregate(scan, [col("country")], [avg("revenue").alias("avg_rev")])
    assert agg.schema.get_field("avg_rev").data_type == FLOAT


def test_node_label_shows_group_by_and_aggregates():
    scan = make_scan()
    agg = Aggregate(scan, [col("country")], [count("*").alias("n")])
    assert agg.node_label == "Aggregate[groupBy=(country), aggregates=(n)]"


def test_analyze_accepts_valid_aggregate():
    scan = make_scan()
    agg = Aggregate(scan, [col("country")], [count("*").alias("n")])
    assert analyze(agg) is agg


def test_analyze_rejects_group_by_on_missing_column():
    scan = make_scan()
    agg = Aggregate(scan, [col("does_not_exist")], [count("*").alias("n")])
    with pytest.raises(AnalysisException, match="does_not_exist"):
        analyze(agg)


def test_analyze_rejects_group_by_on_computed_expression():
    scan = make_scan()
    agg = Aggregate(scan, [col("revenue") > 0], [count("*").alias("n")])
    with pytest.raises(AnalysisException, match="group_by"):
        analyze(agg)


def test_analyze_rejects_aggregate_over_missing_column():
    scan = make_scan()
    agg = Aggregate(scan, [col("country")], [ssum("does_not_exist").alias("n")])
    with pytest.raises(AnalysisException, match="does_not_exist"):
        analyze(agg)


def test_analyze_rejects_non_aggregate_expression_in_agg():
    scan = make_scan()
    agg = Aggregate(scan, [col("country")], [col("revenue")])
    with pytest.raises(AnalysisException, match="aggregate expressions"):
        analyze(agg)


def test_analyze_rejects_duplicate_output_names():
    scan = make_scan()
    # group_by("country") outputs a column named "country"; aliasing the
    # count to the same name collides with it.
    agg = Aggregate(scan, [col("country")], [count("*").alias("country")])
    with pytest.raises(AnalysisException, match="Duplicate output column 'country'"):
        analyze(agg)


def test_analyze_rejects_aliased_group_by_expression():
    scan = make_scan()
    agg = Aggregate(scan, [col("country").alias("x")], [count("*").alias("n")])
    with pytest.raises(AnalysisException, match="group_by"):
        analyze(agg)
