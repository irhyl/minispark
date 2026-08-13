from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.execution.executor import execute as execute_naive
from minispark.logical.nodes import Filter, Project, Scan
from minispark.logical.plan import explain_string
from minispark.physical.operators import execute as execute_physical
from minispark.physical.plan import FilterExec, ProjectExec, ScanExec
from minispark.physical.planner import plan_physical


def make_scan():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [
        {"name": "alice", "age": 30},
        {"name": "bob", "age": 17},
        {"name": "carol", "age": 45},
    ]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=len(rows)))
    dataset = Dataset(schema, [partition])
    return Scan(dataset, "users"), dataset


def test_plan_physical_translates_each_node_1_to_1():
    scan, _ = make_scan()
    logical = Project(Filter(scan, col("age") > 18), [col("name")])
    physical = plan_physical(logical)
    assert isinstance(physical, ProjectExec)
    assert isinstance(physical.child, FilterExec)
    assert isinstance(physical.child.child, ScanExec)


def test_physical_plan_schema_matches_logical_schema():
    scan, _ = make_scan()
    logical = Project(Filter(scan, col("age") > 18), [col("name")])
    physical = plan_physical(logical)
    assert physical.schema.field_names() == logical.schema.field_names()


def test_explain_string_renders_physical_plan():
    scan, _ = make_scan()
    logical = Project(Filter(scan, col("age") > 18), [col("name")])
    physical = plan_physical(logical)
    text = explain_string(physical)
    lines = text.splitlines()
    assert lines[0].startswith("ProjectExec[")
    assert lines[1].strip().startswith("FilterExec[")
    assert lines[2].strip().startswith("ScanExec[")


def test_physical_execution_matches_naive_executor():
    """physical/operators.py must agree with execution/executor.py (the
    Milestone 1 oracle) on the same logical plan: they are two
    implementations of exactly the same Scan/Filter/Project semantics."""
    scan, _ = make_scan()
    logical = Project(Filter(scan, col("age") > 18), [col("name")])

    naive_rows = sorted(execute_naive(logical).iter_records(), key=lambda r: r["name"])

    physical = plan_physical(logical)
    physical_rows = sorted(execute_physical(physical).iter_records(), key=lambda r: r["name"])

    assert naive_rows == physical_rows == [{"name": "alice"}, {"name": "carol"}]
