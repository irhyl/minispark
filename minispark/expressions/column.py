from __future__ import annotations

from typing import Any

from minispark.core.record import Record
from minispark.expressions.base import Expression


class Column(Expression):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, record: Record) -> Any:
        # No analyzer exists yet (Milestone 2) to catch a missing column
        # before execution, so a bad column name surfaces here as a KeyError
        # with the offending name and the row it happened on.
        try:
            return record[self.name]
        except KeyError:
            raise KeyError(
                f"Column '{self.name}' not found in record with keys {list(record.keys())}"
            ) from None

    def __repr__(self) -> str:
        return f"Column({self.name!r})"
