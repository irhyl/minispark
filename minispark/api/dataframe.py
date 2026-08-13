"""DataFrame: the lazy, user-facing query-building API.

`filter()`/`select()` only build logical plan nodes — see `logical/nodes.py`
— and never touch data. Only the action methods at the bottom of this file
(`collect`, `show`, `count`, `explain`) trigger anything: analysis
(`logical/analyzer.py`), optimization (`optimizer/optimizer.py`), physical
planning (`physical/planner.py`), and physical execution
(`physical/operators.py`), in that order. Milestone 1's naive executor
(`execution/executor.py`) is no longer on this path; it remains as the
correctness oracle physical execution is tested against. Milestone 3 will
retarget the last step at a DAG/scheduler without changing this class's
public surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minispark.core.record import Record
from minispark.core.schema import Schema
from minispark.expressions.base import Expression
from minispark.expressions.column import Column
from minispark.logical.analyzer import analyze
from minispark.logical.nodes import Filter, LogicalPlan, Project
from minispark.logical.plan import explain_string
from minispark.optimizer.optimizer import Optimizer, default_rules
from minispark.physical import operators as physical_operators
from minispark.physical.planner import plan_physical

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

    # ---- actions (trigger analysis, optimization, and execution) ----------
    def _optimized_plan(self) -> LogicalPlan:
        analyzed = analyze(self._plan)
        optimizer = Optimizer(default_rules(self._session.config.optimizer))
        return optimizer.optimize(analyzed)

    def collect(self) -> list[Record]:
        physical = plan_physical(self._optimized_plan())
        dataset = physical_operators.execute(physical)
        return list(dataset.iter_records())

    def count(self) -> int:
        physical = plan_physical(self._optimized_plan())
        dataset = physical_operators.execute(physical)
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

    def explain(self, optimized: bool = False) -> None:
        if not optimized:
            print("== Logical Plan ==")
            print(explain_string(self._plan))
            return
        analyzed = analyze(self._plan)
        print("== Analyzed Logical Plan ==")
        print(explain_string(analyzed))
        optimizer = Optimizer(default_rules(self._session.config.optimizer))
        optimized_plan = optimizer.optimize(analyzed)
        print()
        print("== Optimized Logical Plan ==")
        print(explain_string(optimized_plan))
        physical = plan_physical(optimized_plan)
        print()
        print("== Physical Plan ==")
        print(explain_string(physical))
