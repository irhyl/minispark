"""Unit tests for sql/parser.py: does SQL text produce the right shape of
LogicalPlan, built from the same nodes the DataFrame API uses? Structural
checks only (isinstance/attribute assertions and explain_string() text),
matching tests/unit/test_physical_plan.py's style; real end-to-end
execution correctness is tests/integration/test_sql_e2e.py's job.
"""

from __future__ import annotations

import pytest

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.expressions.aggregate import Count, Sum
from minispark.expressions.base import Alias
from minispark.expressions.binary import And, GreaterEqual, GreaterThan
from minispark.expressions.column import Column
from minispark.expressions.literal import Literal
from minispark.logical.nodes import Aggregate, Filter, Join, Project, Scan, Sort
from minispark.logical.plan import explain_string
from minispark.sql.parser import SqlParseError, parse_sql

USERS_SCHEMA = Schema([Field("name", STRING), Field("age", INT), Field("country", STRING)])
REGIONS_SCHEMA = Schema([Field("country", STRING), Field("region", STRING)])


def _table(schema: Schema, rows: list[dict]) -> Scan:
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=len(rows)))
    return Scan(Dataset(schema, [partition]), "test")


def users_table() -> Scan:
    return _table(
        USERS_SCHEMA,
        [
            {"name": "alice", "age": 30, "country": "US"},
            {"name": "bob", "age": 17, "country": "CA"},
        ],
    )


def regions_table() -> Scan:
    return _table(REGIONS_SCHEMA, [{"country": "US", "region": "Americas"}])


def test_select_star_is_a_bare_scan_no_project():
    plan = parse_sql("SELECT * FROM users", {"users": users_table()})
    assert isinstance(plan, Scan)


def test_select_columns_builds_project():
    plan = parse_sql("SELECT name, age FROM users", {"users": users_table()})
    assert isinstance(plan, Project)
    assert [c.name for c in plan.columns] == ["name", "age"]
    assert isinstance(plan.child, Scan)


def test_select_with_alias():
    plan = parse_sql("SELECT age AS a FROM users", {"users": users_table()})
    assert isinstance(plan, Project)
    (col,) = plan.columns
    assert isinstance(col, Alias)
    assert col.name == "a"
    assert isinstance(col.child, Column)
    assert col.child.name == "age"


def test_where_builds_filter_with_correct_condition():
    plan = parse_sql("SELECT name FROM users WHERE age >= 18", {"users": users_table()})
    assert isinstance(plan, Project)
    assert isinstance(plan.child, Filter)
    condition = plan.child.condition
    assert isinstance(condition, GreaterEqual)
    assert isinstance(condition.left, Column)
    assert condition.left.name == "age"
    assert isinstance(condition.right, Literal)
    assert condition.right.value == 18


def test_where_and_or_not_precedence():
    plan = parse_sql(
        "SELECT name FROM users WHERE age >= 18 AND country = 'US'",
        {"users": users_table()},
    )
    condition = plan.child.condition
    assert isinstance(condition, And)


def test_string_literal_value():
    plan = parse_sql("SELECT name FROM users WHERE country = 'US'", {"users": users_table()})
    assert plan.child.condition.right.value == "US"


def test_group_by_and_aggregate_builds_aggregate_node():
    plan = parse_sql(
        "SELECT country, COUNT(*) AS n, SUM(age) AS total FROM users GROUP BY country",
        {"users": users_table()},
    )
    assert isinstance(plan, Aggregate)
    assert [c.name for c in plan.group_by] == ["country"]
    assert len(plan.aggregates) == 2
    n_alias, total_alias = plan.aggregates
    assert n_alias.name == "n"
    assert isinstance(n_alias.child, Count)
    assert total_alias.name == "total"
    assert isinstance(total_alias.child, Sum)


def test_global_aggregate_with_no_group_by():
    plan = parse_sql("SELECT COUNT(*) AS n FROM users", {"users": users_table()})
    assert isinstance(plan, Aggregate)
    assert plan.group_by == []


def test_having_resolves_to_aggregate_output_column():
    plan = parse_sql(
        "SELECT country, COUNT(*) AS n FROM users GROUP BY country HAVING n >= 2",
        {"users": users_table()},
    )
    assert isinstance(plan, Filter)
    assert isinstance(plan.child, Aggregate)
    condition = plan.condition
    assert isinstance(condition, GreaterEqual)
    assert isinstance(condition.left, Column)
    assert condition.left.name == "n"


def test_having_with_raw_aggregate_call_resolves_by_structural_match():
    plan = parse_sql(
        "SELECT country, COUNT(*) AS n FROM users GROUP BY country HAVING COUNT(*) >= 2",
        {"users": users_table()},
    )
    assert isinstance(plan, Filter)
    assert isinstance(plan.condition.left, Column)
    assert plan.condition.left.name == "n"


def test_having_referencing_unselected_aggregate_raises():
    with pytest.raises(SqlParseError, match="not in the SELECT list"):
        parse_sql(
            "SELECT country, COUNT(*) AS n FROM users GROUP BY country HAVING AVG(age) > 10",
            {"users": users_table()},
        )


def test_order_by_default_ascending():
    plan = parse_sql("SELECT * FROM users ORDER BY age", {"users": users_table()})
    assert isinstance(plan, Sort)
    assert plan.ascending == [True]


def test_order_by_desc():
    plan = parse_sql("SELECT * FROM users ORDER BY age DESC", {"users": users_table()})
    assert plan.ascending == [False]


def test_order_by_multiple_columns_mixed_direction():
    plan = parse_sql(
        "SELECT * FROM users ORDER BY country ASC, age DESC", {"users": users_table()}
    )
    assert [c.name for c in plan.sort_exprs] == ["country", "age"]
    assert plan.ascending == [True, False]


def test_join_builds_join_node_with_matching_column_names():
    plan = parse_sql(
        "SELECT name, region FROM users JOIN regions ON users.country = regions.country",
        {"users": users_table(), "regions": regions_table()},
    )
    assert isinstance(plan, Project)
    join = plan.child
    assert isinstance(join, Join)
    assert join.on == ["country"]
    assert join.how == "inner"


def test_join_with_differently_named_columns_raises_clear_error():
    with pytest.raises(SqlParseError, match="same name"):
        parse_sql(
            "SELECT name FROM users JOIN regions ON users.country = regions.other",
            {"users": users_table(), "regions": regions_table()},
        )


def test_unknown_table_raises_clear_error():
    with pytest.raises(SqlParseError, match="Unknown table"):
        parse_sql("SELECT * FROM nope", {"users": users_table()})


def test_unknown_function_raises_clear_error():
    with pytest.raises(SqlParseError, match="Unknown function"):
        parse_sql("SELECT UPPER(name) FROM users", {"users": users_table()})


def test_select_list_item_not_aggregated_or_grouped_raises():
    with pytest.raises(SqlParseError, match="neither aggregated nor a GROUP BY column"):
        parse_sql(
            "SELECT name, COUNT(*) FROM users GROUP BY country", {"users": users_table()}
        )


def test_mixing_aggregate_and_plain_column_without_group_by_raises():
    with pytest.raises(SqlParseError, match="no GROUP BY"):
        parse_sql("SELECT name, COUNT(*) FROM users", {"users": users_table()})


def test_arithmetic_and_comparison_expression():
    plan = parse_sql("SELECT name FROM users WHERE age + 1 > 18", {"users": users_table()})
    condition = plan.child.condition
    assert isinstance(condition, GreaterThan)


def test_is_null_and_is_not_null():
    from minispark.expressions.predicates import IsNotNull, IsNull

    plan1 = parse_sql("SELECT name FROM users WHERE age IS NULL", {"users": users_table()})
    assert isinstance(plan1.child.condition, IsNull)
    plan2 = parse_sql("SELECT name FROM users WHERE age IS NOT NULL", {"users": users_table()})
    assert isinstance(plan2.child.condition, IsNotNull)


def test_null_true_false_literals():
    plan = parse_sql(
        "SELECT name FROM users WHERE age > NULL OR (TRUE AND NOT FALSE)",
        {"users": users_table()},
    )
    assert isinstance(plan.child, Filter)


def test_explain_string_renders_full_plan():
    plan = parse_sql(
        "SELECT name FROM users WHERE age >= 18 ORDER BY name",
        {"users": users_table()},
    )
    text = explain_string(plan)
    assert "Sort[" in text
    assert "Filter[" in text
    assert "Scan[" in text


def test_syntax_error_reports_position():
    with pytest.raises(SqlParseError):
        parse_sql("SELECT FROM users", {"users": users_table()})


def test_trailing_garbage_raises():
    with pytest.raises(SqlParseError, match="Unexpected"):
        parse_sql("SELECT * FROM users EXTRA", {"users": users_table()})
