from minispark.api.functions import col, lit
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.expressions.literal import Literal
from minispark.logical.nodes import Filter, Project, Scan
from minispark.logical.plan import explain_string
from minispark.optimizer.rules import (
    ConstantFolding,
    FilterSimplification,
    PredicatePushdown,
    ProjectionPruning,
    RedundantProjectionElimination,
)


def make_scan():
    schema = Schema([Field("name", STRING), Field("age", INT), Field("country", STRING)])
    rows = [{"name": "a", "age": 20, "country": "US"}]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=1))
    dataset = Dataset(schema, [partition])
    return Scan(dataset, "users")


# ---- ConstantFolding --------------------------------------------------------


def test_constant_folding_evaluates_literal_arithmetic():
    # `Expression.__eq__` builds an Equal expression node rather than
    # returning bool (see expressions/base.py), so assertions here compare
    # `.value` / `repr()` directly instead of using `==` on expressions.
    scan = make_scan()
    before = Filter(scan, col("age") > (lit(10) + lit(10)))
    after = ConstantFolding().apply(before)
    assert isinstance(after, Filter)
    assert isinstance(after.condition.right, Literal)
    assert after.condition.right.value == 20


def test_constant_folding_leaves_column_references_alone():
    scan = make_scan()
    before = Filter(scan, col("age") > lit(18))
    after = ConstantFolding().apply(before)
    assert "Column('age')" in repr(after.condition)


# ---- FilterSimplification ---------------------------------------------------


def test_filter_simplification_removes_and_true():
    scan = make_scan()
    before = Filter(scan, (col("age") > lit(18)) & lit(True))
    after = FilterSimplification().apply(before)
    assert repr(after.condition) == repr(col("age") > lit(18))


def test_filter_simplification_collapses_and_false_to_false():
    scan = make_scan()
    before = Filter(scan, (col("age") > lit(18)) & lit(False))
    after = FilterSimplification().apply(before)
    assert after.condition.value is False


def test_filter_simplification_removes_or_false():
    scan = make_scan()
    before = Filter(scan, (col("age") > lit(18)) | lit(False))
    after = FilterSimplification().apply(before)
    assert repr(after.condition) == repr(col("age") > lit(18))


def test_filter_simplification_removes_double_negation():
    scan = make_scan()
    before = Filter(scan, ~(~(col("age") > lit(18))))
    after = FilterSimplification().apply(before)
    assert repr(after.condition) == repr(col("age") > lit(18))


# ---- PredicatePushdown -------------------------------------------------------


def test_predicate_pushdown_swaps_filter_below_project():
    scan = make_scan()
    before = Filter(Project(scan, [col("name"), col("age")]), col("age") > lit(18))
    after = PredicatePushdown().apply(before)
    assert isinstance(after, Project)
    assert isinstance(after.child, Filter)
    assert after.child.child is scan


def test_predicate_pushdown_still_moves_filter_on_a_column_the_project_drops():
    """A Project only narrows its own *output*; the raw records it reads from
    still have every source column. Filtering on "country" after a Project
    that only selects name/age is still safe to push below that Project,
    since the pushed-down Filter runs against the pre-projection records."""
    scan = make_scan()
    before = Filter(Project(scan, [col("name"), col("age")]), col("country") == lit("US"))
    after = PredicatePushdown().apply(before)
    assert isinstance(after, Project)
    assert isinstance(after.child, Filter)
    assert after.child.child is scan


def test_predicate_pushdown_does_not_move_filter_needing_an_aliased_column():
    """"years" only exists after the Project computes it (as an alias of
    "age"); the Project's child schema has no "years" column, so the
    Filter cannot move below the Project without breaking."""
    scan = make_scan()
    before = Filter(Project(scan, [col("age").alias("years")]), col("years") > lit(18))
    after = PredicatePushdown().apply(before)
    assert isinstance(after, Filter)
    assert isinstance(after.child, Project)


# ---- ProjectionPruning -------------------------------------------------------


def test_projection_pruning_inserts_minimal_project_above_scan():
    scan = make_scan()
    plan = Project(Filter(scan, col("age") > lit(18)), [col("name")])
    pruned = ProjectionPruning().apply(plan)
    text = explain_string(pruned)
    # "country" is referenced nowhere (not in the final select, not in the
    # filter condition) so it should be dropped right after the scan.
    lines = text.splitlines()
    assert lines[0].startswith("Project[name]")
    assert "country" not in lines[2]  # the inserted Project's node_label
    assert "age" in lines[2] and "name" in lines[2]


def test_projection_pruning_leaves_top_level_filter_untouched():
    """Regression test: a bare filter() with no select() must not drop columns.

    ProjectionPruning previously treated "nothing above restricts the
    output" the same as "the filter's own condition is all that's needed",
    which silently dropped every column the top-level Filter did not
    reference itself.
    """
    scan = make_scan()
    plan = Filter(scan, col("age") > lit(18))
    pruned = ProjectionPruning().apply(plan)
    assert pruned.schema.field_names() == ["name", "age", "country"]


# ---- RedundantProjectionElimination ------------------------------------------


def test_redundant_projection_elimination_removes_identity_project():
    scan = make_scan()
    plan = Project(scan, [col("name"), col("age"), col("country")])
    eliminated = RedundantProjectionElimination().apply(plan)
    assert eliminated is scan


def test_redundant_projection_elimination_collapses_nested_plain_projects():
    scan = make_scan()
    inner = Project(scan, [col("name"), col("age")])
    outer = Project(inner, [col("name")])
    eliminated = RedundantProjectionElimination().apply(outer)
    assert isinstance(eliminated, Project)
    assert eliminated.child is scan
    assert eliminated.schema.field_names() == ["name"]
