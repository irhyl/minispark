from minispark.api.functions import col, count
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.execution.dag import DependencyKind, build_dag, dependency_kind, has_wide_dependency
from minispark.logical.nodes import Aggregate, Filter, Project, Scan
from minispark.physical.planner import plan_physical


def make_physical_plan():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [{"name": "a", "age": 20}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
    scan = Scan(Dataset(schema, [partition]), "users")
    logical = Project(Filter(scan, col("age") > 18), [col("name")])
    return plan_physical(logical)


def make_aggregate_physical_plan():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [{"name": "a", "age": 20}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
    scan = Scan(Dataset(schema, [partition]), "users")
    logical = Aggregate(scan, [col("name")], [count("*").alias("n")])
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


def test_scan_filter_project_have_no_wide_dependency():
    dag = build_dag(make_physical_plan())
    assert has_wide_dependency(dag) is False


def test_exchange_is_wide():
    dag = build_dag(make_aggregate_physical_plan())
    exchange_node = dag.children[0]  # HashAggregateExec(final) -> Exchange
    assert dependency_kind(exchange_node.plan).name == "WIDE"
    assert has_wide_dependency(dag) is True


def test_hash_aggregate_itself_is_narrow():
    """HashAggregateExec only groups rows within the one partition it is
    given; the wide dependency is specifically the Exchange around it, not
    the aggregation logic itself."""
    dag = build_dag(make_aggregate_physical_plan())
    assert dag.dependency is DependencyKind.NARROW  # the final HashAggregateExec
    partial_node = dag.children[0].children[0]
    assert partial_node.dependency is DependencyKind.NARROW  # the partial HashAggregateExec
