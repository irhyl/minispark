"""Aggregate expressions: count, sum, avg, min, max.

An aggregate has no value for a single row; it only produces one after
combining many rows. So instead of `evaluate()` (which raises here, see
below), an `AggregateFunction` is driven through four steps that mirror
exactly how Milestone 4's shuffle-based grouping works:

    initialize()            -> state       one group's state, before any row
    update(state, record)   -> state       fold one *raw* row into state
    merge(state_a, state_b) -> state       combine two states
    finalize(state)         -> value       state -> the value shown to the user

`update` only ever runs on raw rows: that is map-side partial aggregation,
run once per source partition, before any shuffle. `merge` only ever runs
on states that already came out of `initialize`/`update`/`merge`: that is
what the reduce side does after a shuffle, combining partial states that
originally came from different source partitions but landed in the same
hash bucket. This split (rather than just running `update` on every row
after a shuffle) is what lets partial aggregation shrink shuffle volume,
`group_by(country).sum(revenue)` shuffles one partial sum per
(source partition, country) pair, not every raw row.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from minispark.core.record import Record
from minispark.core.types import FLOAT, INT, DataType
from minispark.expressions.base import Expression


class AggregateFunction(Expression):
    def __init__(self, child: Expression | None):
        self.child = child

    def evaluate(self, record: Record) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} is an aggregate expression; it has no "
            "per-row value. It is driven through initialize()/update()/"
            "merge()/finalize() by HashAggregateExec, not evaluate()."
        )

    @property
    def children(self) -> list[Expression]:
        return [self.child] if self.child is not None else []

    @abstractmethod
    def initialize(self) -> Any:
        """The state a fresh group starts with, before any row is seen."""

    @abstractmethod
    def update(self, state: Any, record: Record) -> Any:
        """Fold one raw row into `state`."""

    @abstractmethod
    def merge(self, state_a: Any, state_b: Any) -> Any:
        """Combine two states produced by (subsets of) the same group's rows."""

    def finalize(self, state: Any) -> Any:
        """`state` -> the value the user sees. Identity unless overridden."""
        return state

    @abstractmethod
    def result_type(self, child_type: DataType) -> DataType:
        """The output DataType, given the input column's DataType."""


class Count(AggregateFunction):
    """`count(*)` when `child` is None (counts rows); `count(col)` otherwise
    (counts non-null values of `col`)."""

    def initialize(self) -> int:
        return 0

    def update(self, state: int, record: Record) -> int:
        if self.child is None:
            return state + 1
        return state + (0 if self.child.evaluate(record) is None else 1)

    def merge(self, state_a: int, state_b: int) -> int:
        return state_a + state_b

    def result_type(self, child_type: DataType) -> DataType:
        return INT

    def __repr__(self) -> str:
        return f"Count({self.child!r})" if self.child is not None else "Count(*)"


class Sum(AggregateFunction):
    def __init__(self, child: Expression):
        super().__init__(child)

    def initialize(self) -> Any:
        return None

    def update(self, state: Any, record: Record) -> Any:
        assert self.child is not None
        value = self.child.evaluate(record)
        if value is None:
            return state
        return value if state is None else state + value

    def merge(self, state_a: Any, state_b: Any) -> Any:
        if state_a is None:
            return state_b
        if state_b is None:
            return state_a
        return state_a + state_b

    def result_type(self, child_type: DataType) -> DataType:
        return child_type

    def __repr__(self) -> str:
        return f"Sum({self.child!r})"


class Avg(AggregateFunction):
    def __init__(self, child: Expression):
        super().__init__(child)

    def initialize(self) -> tuple[Any, int]:
        return (0, 0)

    def update(self, state: tuple[Any, int], record: Record) -> tuple[Any, int]:
        assert self.child is not None
        value = self.child.evaluate(record)
        if value is None:
            return state
        total, count = state
        return (total + value, count + 1)

    def merge(self, state_a: tuple[Any, int], state_b: tuple[Any, int]) -> tuple[Any, int]:
        return (state_a[0] + state_b[0], state_a[1] + state_b[1])

    def finalize(self, state: tuple[Any, int]) -> Any:
        total, count = state
        return (total / count) if count else None

    def result_type(self, child_type: DataType) -> DataType:
        return FLOAT

    def __repr__(self) -> str:
        return f"Avg({self.child!r})"


class Min(AggregateFunction):
    def __init__(self, child: Expression):
        super().__init__(child)

    def initialize(self) -> Any:
        return None

    def update(self, state: Any, record: Record) -> Any:
        assert self.child is not None
        value = self.child.evaluate(record)
        if value is None:
            return state
        return value if state is None or value < state else state

    def merge(self, state_a: Any, state_b: Any) -> Any:
        if state_a is None:
            return state_b
        if state_b is None:
            return state_a
        return state_a if state_a < state_b else state_b

    def result_type(self, child_type: DataType) -> DataType:
        return child_type

    def __repr__(self) -> str:
        return f"Min({self.child!r})"


class Max(AggregateFunction):
    def __init__(self, child: Expression):
        super().__init__(child)

    def initialize(self) -> Any:
        return None

    def update(self, state: Any, record: Record) -> Any:
        assert self.child is not None
        value = self.child.evaluate(record)
        if value is None:
            return state
        return value if state is None or value > state else state

    def merge(self, state_a: Any, state_b: Any) -> Any:
        if state_a is None:
            return state_b
        if state_b is None:
            return state_a
        return state_a if state_a > state_b else state_b

    def result_type(self, child_type: DataType) -> DataType:
        return child_type

    def __repr__(self) -> str:
        return f"Max({self.child!r})"
