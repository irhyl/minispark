from __future__ import annotations

from typing import Any

from minispark.core.record import Record
from minispark.expressions.base import Expression


class Literal(Expression):
    def __init__(self, value: Any):
        self.value = value

    def evaluate(self, record: Record) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"
