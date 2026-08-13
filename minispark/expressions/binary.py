"""Binary expressions: two child expressions combined by an operator.

Each concrete class is a thin wrapper that names an operator explicitly
(GreaterThan, Add, And, ...) rather than a single generic `BinaryOp(op_str)`
node. This keeps the tree self-describing for `explain()` and gives the
future optimizer (Milestone 2, constant folding / filter simplification)
concrete node types to pattern-match on instead of string comparison.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

from minispark.core.record import Record
from minispark.expressions.base import Expression


class BinaryExpression(Expression):
    symbol: str = "?"
    op: Callable[[Any, Any], Any]

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, record: Record) -> Any:
        return self.op(self.left.evaluate(record), self.right.evaluate(record))

    @property
    def children(self) -> list[Expression]:
        return [self.left, self.right]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.left!r}, {self.right!r})"


class GreaterThan(BinaryExpression):
    symbol = ">"
    op = staticmethod(operator.gt)


class GreaterEqual(BinaryExpression):
    symbol = ">="
    op = staticmethod(operator.ge)


class LessThan(BinaryExpression):
    symbol = "<"
    op = staticmethod(operator.lt)


class LessEqual(BinaryExpression):
    symbol = "<="
    op = staticmethod(operator.le)


class Equal(BinaryExpression):
    symbol = "=="
    op = staticmethod(operator.eq)


class NotEqual(BinaryExpression):
    symbol = "!="
    op = staticmethod(operator.ne)


class And(BinaryExpression):
    symbol = "AND"
    op = staticmethod(lambda a, b: bool(a) and bool(b))


class Or(BinaryExpression):
    symbol = "OR"
    op = staticmethod(lambda a, b: bool(a) or bool(b))


class Add(BinaryExpression):
    symbol = "+"
    op = staticmethod(operator.add)


class Subtract(BinaryExpression):
    symbol = "-"
    op = staticmethod(operator.sub)


class Multiply(BinaryExpression):
    symbol = "*"
    op = staticmethod(operator.mul)


class Divide(BinaryExpression):
    symbol = "/"
    op = staticmethod(operator.truediv)
