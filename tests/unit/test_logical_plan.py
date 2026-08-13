from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.logical.nodes import Filter, Project, Scan
from minispark.logical.plan import explain_string


def make_scan():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [{"name": "a", "age": 20}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
    dataset = Dataset(schema, [partition])
    return Scan(dataset, "users")


def test_scan_schema():
    scan = make_scan()
    assert scan.schema.field_names() == ["name", "age"]


def test_filter_preserves_schema():
    scan = make_scan()
    f = Filter(scan, col("age") > 18)
    assert f.schema.field_names() == ["name", "age"]
    assert f.children == [scan]


def test_project_narrows_schema_and_types():
    scan = make_scan()
    p = Project(scan, [col("name")])
    assert p.schema.field_names() == ["name"]
    assert p.schema.get_field("name").data_type == STRING


def test_explain_string_shows_tree_shape():
    scan = make_scan()
    plan = Project(Filter(scan, col("age") > 18), [col("name")])
    text = explain_string(plan)
    lines = text.splitlines()
    assert lines[0].startswith("Project[")
    assert lines[1].strip().startswith("Filter[")
    assert lines[2].strip().startswith("Scan[")
    # indentation increases with depth
    assert lines[1].startswith("  ")
    assert lines[2].startswith("    ")
