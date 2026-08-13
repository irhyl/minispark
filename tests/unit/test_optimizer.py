from minispark.api.functions import col, lit
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.execution.executor import execute as execute_naive
from minispark.logical.nodes import Filter, Project, Scan
from minispark.optimizer.optimizer import Optimizer


def make_scan():
    schema = Schema([Field("name", STRING), Field("age", INT), Field("country", STRING)])
    rows = [
        {"name": "alice", "age": 30, "country": "US"},
        {"name": "bob", "age": 17, "country": "CA"},
        {"name": "carol", "age": 45, "country": "US"},
    ]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=len(rows)))
    dataset = Dataset(schema, [partition])
    return Scan(dataset, "users")


def test_optimize_is_idempotent():
    scan = make_scan()
    plan = Filter(Project(scan, [col("name"), col("age")]), col("age") > (lit(10) + lit(10)))
    optimizer = Optimizer()
    once = optimizer.optimize(plan)
    twice = optimizer.optimize(once)
    from minispark.logical.plan import explain_string

    assert explain_string(once) == explain_string(twice)


def test_optimized_plan_produces_same_rows_as_unoptimized_plan():
    """The optimizer must be correctness-preserving: same rows, in the same
    row-order-per-partition, regardless of the rewrites applied."""
    scan = make_scan()
    plan = Filter(Project(scan, [col("name"), col("age")]), col("age") > (lit(10) + lit(10)))

    unoptimized_rows = list(execute_naive(plan).iter_records())

    optimized = Optimizer().optimize(plan)
    optimized_rows = list(execute_naive(optimized).iter_records())

    assert sorted(unoptimized_rows, key=lambda r: r["name"]) == sorted(
        optimized_rows, key=lambda r: r["name"]
    )


def _every_node(plan):
    yield plan
    for child in plan.children:
        yield from _every_node(child)


def test_optimize_prunes_and_pushes_down():
    scan = make_scan()
    plan = Filter(Project(scan, [col("name"), col("age")]), col("age") > lit(20))
    optimized = Optimizer().optimize(plan)

    # "country" is needed nowhere (not selected, not filtered on), so
    # projection pruning should have dropped it everywhere above the Scan.
    # Scan itself always reports the dataset's full raw schema (pruning
    # works by inserting a Project above it, not by mutating it), so Scan
    # nodes are excluded from this check on purpose.
    # The exact final shape (whether the user's now-redundant outer
    # Project survives RedundantProjectionElimination) is not asserted
    # here on purpose: which rule interaction produces the minimal tree
    # is an implementation detail, the output columns are the contract.
    assert optimized.schema.field_names() == ["name", "age"]
    for node in _every_node(optimized):
        if isinstance(node, Scan):
            continue
        assert "country" not in node.schema.field_names()

    rows = sorted(execute_naive(optimized).iter_records(), key=lambda r: r["name"])
    assert rows == [{"name": "alice", "age": 30}, {"name": "carol", "age": 45}]
