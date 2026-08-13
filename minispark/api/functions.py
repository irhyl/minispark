"""Free functions for building expressions: col(), lit().

Aggregate functions (count, sum, avg, min, max) are added in Milestone 4
alongside the Aggregate logical node they only make sense next to.
"""

from __future__ import annotations

from typing import Any

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
