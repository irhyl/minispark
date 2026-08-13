from minispark.api.functions import col, lit
from minispark.expressions.binary import And, GreaterThan
from minispark.expressions.column import Column


def test_comparison_builds_tree_without_evaluating():
    expr = col("age") > 18
    assert isinstance(expr, GreaterThan)
    assert isinstance(expr.left, Column)
    assert expr.left.name == "age"
    assert expr.right.value == 18


def test_expression_evaluate_on_record():
    expr = col("age") > 18
    assert expr.evaluate({"age": 21}) is True
    assert expr.evaluate({"age": 10}) is False


def test_and_or_combination():
    expr = (col("age") > 18) & (col("country") == lit("US"))
    assert isinstance(expr, And)
    assert expr.evaluate({"age": 21, "country": "US"}) is True
    assert expr.evaluate({"age": 21, "country": "CA"}) is False


def test_not_and_is_null():
    assert (~(col("active") == lit(True))).evaluate({"active": False}) is True
    assert col("x").is_null().evaluate({"x": None}) is True
    assert col("x").is_not_null().evaluate({"x": 1}) is True


def test_arithmetic_expression():
    expr = (col("a") + col("b")) * lit(2)
    assert expr.evaluate({"a": 3, "b": 4}) == 14


def test_alias():
    expr = col("age").alias("years")
    assert expr.name == "years"
    assert expr.evaluate({"age": 30}) == 30


def test_missing_column_raises_keyerror():
    import pytest

    with pytest.raises(KeyError):
        col("missing").evaluate({"age": 1})
