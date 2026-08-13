"""Unary predicate expressions: Not, IsNull, IsNotNull."""

from __future__ import annotations

from typing import Any

from minispark.core.record import Record
from minispark.expressions.base import Expression


class Not(Expression):
    def __init__(self, child: Expression):
        self.child = child

    def evaluate(self, record: Record) -> Any:
        return not bool(self.child.evaluate(record))

    @property
    def children(self) -> list[Expression]:
        return [self.child]

    def __repr__(self) -> str:
        return f"Not({self.child!r})"


class IsNull(Expression):
    def __init__(self, child: Expression):
        self.child = child

    def evaluate(self, record: Record) -> Any:
        return self.child.evaluate(record) is None

    @property
    def children(self) -> list[Expression]:
        return [self.child]

    def __repr__(self) -> str:
        return f"IsNull({self.child!r})"


class IsNotNull(Expression):
    def __init__(self, child: Expression):
        self.child = child

    def evaluate(self, record: Record) -> Any:
        return self.child.evaluate(record) is not None

    @property
    def children(self) -> list[Expression]:
        return [self.child]

    def __repr__(self) -> str:
        return f"IsNotNull({self.child!r})"
