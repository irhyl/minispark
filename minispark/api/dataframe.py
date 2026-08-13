"""DataFrame: the lazy, user-facing query-building API.

`filter()`/`select()` only build logical plan nodes — see `logical/nodes.py`
— and never touch data. Only the action methods at the bottom of this file
(`collect`, `show`, `count`, `explain`) trigger execution, currently via
the naive tree-walking executor (`execution/executor.py`); Milestone 2/3
will retarget these actions at an optimizer + physical plan + DAG/scheduler
without changing this class's public surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minispark.core.record import Record
from minispark.core.schema import Schema
from minispark.execution.executor import execute
from minispark.expressions.base import Expression
from minispark.expressions.column import Column
from minispark.logical.nodes import Filter, LogicalPlan, Project
from minispark.logical.plan import explain_string

if TYPE_CHECKING:
    from minispark.api.session import MiniSparkSession


class DataFrame:
    def __init__(self, session: MiniSparkSession, plan: LogicalPlan):
        self._session = session
        self._plan = plan

    @property
    def schema(self) -> Schema:
        return self._plan.schema

    @property
    def plan(self) -> LogicalPlan:
        return self._plan

    # ---- transformations (lazy: build plan nodes only) --------------------
    def filter(self, condition: Expression) -> DataFrame:
        return DataFrame(self._session, Filter(self._plan, condition))

    # `where` is the common SQL-flavored alias for `filter`.
    where = filter

    def select(self, *columns: str | Expression) -> DataFrame:
        if not columns:
            raise ValueError("select() requires at least one column")
        exprs = [Column(c) if isinstance(c, str) else c for c in columns]
        return DataFrame(self._session, Project(self._plan, exprs))

    # ---- actions (trigger execution) ---------------------------------------
    def collect(self) -> list[Record]:
        dataset = execute(self._plan)
        return list(dataset.iter_records())

    def count(self) -> int:
        dataset = execute(self._plan)
        return dataset.row_count()

    def show(self, n: int = 20) -> None:
        rows = self.collect()[:n]
        columns = self.schema.field_names()
        if not rows:
            print(f"({', '.join(columns)})\n(no rows)")
            return
        widths = [
            max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in columns
        ]
        header = " | ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))
        separator = "-+-".join("-" * w for w in widths)
        print(header)
        print(separator)
        for r in rows:
            cells = (str(r.get(c, "")).ljust(w) for c, w in zip(columns, widths, strict=True))
            print(" | ".join(cells))

    def explain(self) -> None:
        print("== Logical Plan ==")
        print(explain_string(self._plan))
