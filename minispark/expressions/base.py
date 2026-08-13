"""Expression: the base class for every node in an expression tree.

Operator overloads (`__gt__`, `__add__`, ...) live here rather than only on
`Column`, so that composite expressions (e.g. `(col("a") + col("b")) > 10`)
can be built too. The binary-expression classes are imported lazily inside
each method to avoid a base.py <-> binary.py import cycle: binary.py's
classes subclass Expression, so base.py cannot import binary.py at module
scope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from minispark.core.record import Record


class Expression(ABC):
    @abstractmethod
    def evaluate(self, record: Record) -> Any:
        """Evaluate this expression against a single row.

        This is the naive, row-at-a-time evaluation strategy used by
        Milestone 1's tree-walking executor. It is O(expression depth) per
        row with no vectorization — Milestone 7's columnar execution path
        replaces this with batch evaluation over Arrow arrays. Kept here
        because expressions must be evaluable *somehow* for
        filter()/select() to produce real results before the physical
        planner exists.
        """

    @property
    def children(self) -> list[Expression]:
        """Direct child expressions, for generic tree walking.

        Defaults to no children (Column, Literal). Subclasses with children
        (BinaryExpression, Alias, Not, IsNull, IsNotNull) override this.
        Added in Milestone 2 so the analyzer (column existence checks) and
        the optimizer (constant folding, predicate pushdown) can walk any
        expression tree without an isinstance chain over every node type.
        """
        return []

    def alias(self, name: str) -> Alias:
        return Alias(self, name)

    # ---- comparison operators -------------------------------------------------
    def __gt__(self, other: object) -> Expression:
        from minispark.expressions.binary import GreaterThan

        return GreaterThan(self, _to_expr(other))

    def __ge__(self, other: object) -> Expression:
        from minispark.expressions.binary import GreaterEqual

        return GreaterEqual(self, _to_expr(other))

    def __lt__(self, other: object) -> Expression:
        from minispark.expressions.binary import LessThan

        return LessThan(self, _to_expr(other))

    def __le__(self, other: object) -> Expression:
        from minispark.expressions.binary import LessEqual

        return LessEqual(self, _to_expr(other))

    def __eq__(self, other: object) -> Expression:  # type: ignore[override]
        from minispark.expressions.binary import Equal

        return Equal(self, _to_expr(other))

    def __ne__(self, other: object) -> Expression:  # type: ignore[override]
        from minispark.expressions.binary import NotEqual

        return NotEqual(self, _to_expr(other))

    def __hash__(self) -> int:
        # Overriding __eq__ to build expressions (Spark-style DSL) forfeits
        # the default identity hash. Fall back to id(); Expression trees
        # are not intended to be used as dict/set keys for value equality.
        return id(self)

    # ---- boolean operators ------------------------------------------------
    def __and__(self, other: object) -> Expression:
        from minispark.expressions.binary import And

        return And(self, _to_expr(other))

    def __or__(self, other: object) -> Expression:
        from minispark.expressions.binary import Or

        return Or(self, _to_expr(other))

    def __invert__(self) -> Expression:
        from minispark.expressions.predicates import Not

        return Not(self)

    # ---- arithmetic operators ----------------------------------------------
    def __add__(self, other: object) -> Expression:
        from minispark.expressions.binary import Add

        return Add(self, _to_expr(other))

    def __sub__(self, other: object) -> Expression:
        from minispark.expressions.binary import Subtract

        return Subtract(self, _to_expr(other))

    def __mul__(self, other: object) -> Expression:
        from minispark.expressions.binary import Multiply

        return Multiply(self, _to_expr(other))

    def __truediv__(self, other: object) -> Expression:
        from minispark.expressions.binary import Divide

        return Divide(self, _to_expr(other))

    def is_null(self) -> Expression:
        from minispark.expressions.predicates import IsNull

        return IsNull(self)

    def is_not_null(self) -> Expression:
        from minispark.expressions.predicates import IsNotNull

        return IsNotNull(self)


def _to_expr(value: object) -> Expression:
    """Wrap a plain Python value in a Literal; pass Expressions through."""
    if isinstance(value, Expression):
        return value
    from minispark.expressions.literal import Literal

    return Literal(value)


class Alias(Expression):
    """Renames the output of an expression, e.g. `count("*").alias("users")`."""

    def __init__(self, child: Expression, name: str):
        self.child = child
        self.name = name

    def evaluate(self, record: Record) -> Any:
        return self.child.evaluate(record)

    @property
    def children(self) -> list[Expression]:
        return [self.child]

    def __repr__(self) -> str:
        return f"Alias({self.child!r}, {self.name!r})"
