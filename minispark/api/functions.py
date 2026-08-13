"""Free functions for building expressions: col(), lit(), and the
aggregate functions (count, sum, avg, min, max).

`sum`, `min`, and `max` shadow the Python builtins of the same name inside
this module, on purpose: this matches the DSL convention users already
know from PySpark (`from minispark.api.functions import sum, count`), and
the shadowing is local to this module's namespace, it does not affect any
other module that does not explicitly import these names.
"""

from __future__ import annotations

from typing import Any

from minispark.expressions.aggregate import Avg, Count, Max, Min, Sum
from minispark.expressions.base import Expression
from minispark.expressions.column import Column
from minispark.expressions.literal import Literal


def col(name: str) -> Column:
    return Column(name)


def lit(value: Any) -> Literal:
    return Literal(value)


def _to_column(name_or_expr: str | Expression) -> Expression:
    if isinstance(name_or_expr, str):
        return Column(name_or_expr)
    return name_or_expr


def count(column: str | Expression | None = "*") -> Count:
    """`count("*")` (the default) counts rows; `count("col")` counts
    non-null values of `col`."""
    if column is None or column == "*":
        return Count(None)
    return Count(_to_column(column))


def sum(column: str | Expression) -> Sum:
    return Sum(_to_column(column))


def avg(column: str | Expression) -> Avg:
    return Avg(_to_column(column))


def min(column: str | Expression) -> Min:
    return Min(_to_column(column))


def max(column: str | Expression) -> Max:
    return Max(_to_column(column))
