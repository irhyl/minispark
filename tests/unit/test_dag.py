from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.execution.dag import DependencyKind, build_dag, has_wide_dependency
from minispark.logical.nodes import Filter, Project, Scan
from minispark.physical.planner import plan_physical


def make_physical_plan():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [{"name": "a", "age": 20}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
    scan = Scan(Dataset(schema, [partition]), "users")
    logical = Project(Filter(scan, col("age") > 18), [col("name")])
    return plan_physical(logical)


def test_scan_filter_project_are_all_narrow():
    dag = build_dag(make_physical_plan())
    assert dag.dependency is DependencyKind.NARROW
    assert dag.children[0].dependency is DependencyKind.NARROW
    assert dag.children[0].children[0].dependency is DependencyKind.NARROW


def test_dag_shape_matches_physical_plan_shape():
    dag = build_dag(make_physical_plan())
    assert type(dag.plan).__name__ == "ProjectExec"
    assert type(dag.children[0].plan).__name__ == "FilterExec"
    assert type(dag.children[0].children[0].plan).__name__ == "ScanExec"


def test_no_wide_dependency_exists_yet():
    dag = build_dag(make_physical_plan())
    assert has_wide_dependency(dag) is False
