import pytest

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.logical.analyzer import AnalysisException, analyze
from minispark.logical.nodes import Join, Scan


def make_scans():
    left_schema = Schema([Field("id", INT), Field("name", STRING)])
    right_schema = Schema([Field("id", INT), Field("amount", INT)])
    left = Scan(
        Dataset(left_schema, [Partition(0, left_schema, lambda: iter([]), PartitionMetadata())]),
        "left",
    )
    right = Scan(
        Dataset(right_schema, [Partition(0, right_schema, lambda: iter([]), PartitionMetadata())]),
        "right",
    )
    return left, right


def test_schema_merges_both_sides_dropping_the_right_on_column():
    left, right = make_scans()
    join = Join(left, right, on=["id"])
    assert join.schema.field_names() == ["id", "name", "amount"]
    assert join.schema.get_field("id").data_type == INT


def test_children_are_left_and_right():
    left, right = make_scans()
    join = Join(left, right, on=["id"])
    assert join.children == [left, right]


def test_node_label_shows_join_type_and_on_columns():
    left, right = make_scans()
    join = Join(left, right, on=["id"])
    assert join.node_label == "Join[inner, on=(id)]"
    broadcast_join = Join(left, right, on=["id"], broadcast=True)
    assert broadcast_join.node_label == "Join[inner, on=(id)] broadcast"


def test_analyze_accepts_valid_join():
    left, right = make_scans()
    join = Join(left, right, on=["id"])
    assert analyze(join) is join


def test_analyze_rejects_missing_on_column_on_either_side():
    left, right = make_scans()
    with pytest.raises(AnalysisException, match="not found on the left side"):
        analyze(Join(left, right, on=["missing"]))


def test_analyze_rejects_empty_on():
    left, right = make_scans()
    with pytest.raises(AnalysisException, match="at least one column"):
        analyze(Join(left, right, on=[]))


def test_analyze_rejects_unsupported_how():
    left, right = make_scans()
    with pytest.raises(AnalysisException, match="Unsupported join type"):
        analyze(Join(left, right, on=["id"], how="left"))


def test_analyze_rejects_non_on_name_collision():
    left_schema = Schema([Field("id", INT), Field("name", STRING)])
    right_schema = Schema([Field("id", INT), Field("name", STRING)])
    left = Scan(
        Dataset(left_schema, [Partition(0, left_schema, lambda: iter([]), PartitionMetadata())]),
        "left",
    )
    right = Scan(
        Dataset(right_schema, [Partition(0, right_schema, lambda: iter([]), PartitionMetadata())]),
        "right",
    )
    with pytest.raises(AnalysisException, match="exist on both sides"):
        analyze(Join(left, right, on=["id"]))
