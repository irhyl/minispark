"""DataFrame: the lazy, user-facing query-building API.

`filter()`/`select()`/`group_by()` only build logical plan nodes — see
`logical/nodes.py` — and never touch data. Only the action methods at the
bottom of this file (`collect`, `show`, `count`, `explain`) trigger
anything: analysis (`logical/analyzer.py`), optimization
(`optimizer/optimizer.py`), physical planning (`physical/planner.py`),
stage splitting (`execution/stages.py`), and scheduled execution
(`execution/scheduler.py`'s `LocalScheduler`), in that order. Milestone 1's
naive executor (`execution/executor.py`) and `physical/operators.py`'s
whole-Dataset `execute()` are no longer on this path; both remain as
correctness oracles other tests check the real path against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minispark.api.grouped import GroupedData
from minispark.core.record import Record
from minispark.core.schema import Schema
from minispark.execution.scheduler import LocalScheduler
from minispark.execution.stages import Stage, build_stages
from minispark.expressions.base import Expression
from minispark.expressions.column import Column
from minispark.logical.analyzer import analyze
from minispark.logical.nodes import Filter, Join, LogicalPlan, Project, Sort
from minispark.logical.plan import explain_string
from minispark.optimizer.optimizer import Optimizer, default_rules
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

    def group_by(self, *columns: str | Expression) -> GroupedData:
        if not columns:
            raise ValueError("group_by() requires at least one column")
        exprs = [Column(c) if isinstance(c, str) else c for c in columns]
        return GroupedData(self._session, self._plan, exprs)

    def join(
        self,
        other: DataFrame,
        on: str | list[str],
        how: str = "inner",
        broadcast: bool = False,
    ) -> DataFrame:
        """Inner equi-join on column name(s) present on both sides.

        See logical/nodes.py's `Join` docstring for exactly what this does
        and does not support (inner only, common-name `on=` only).
        `broadcast=True` is an explicit hint (see physical/planner.py);
        there is no automatic broadcast-vs-shuffle selection.
        """
        on_list = [on] if isinstance(on, str) else list(on)
        if not on_list:
            raise ValueError("join() requires at least one column in on=")
        return DataFrame(
            self._session, Join(self._plan, other._plan, on_list, how=how, broadcast=broadcast)
        )

    def order_by(
        self, *columns: str | Expression, ascending: bool | list[bool] = True
    ) -> DataFrame:
        if not columns:
            raise ValueError("order_by() requires at least one column")
        exprs = [Column(c) if isinstance(c, str) else c for c in columns]
        if isinstance(ascending, bool):
            ascending_list = [ascending] * len(exprs)
        else:
            ascending_list = list(ascending)
            if len(ascending_list) != len(exprs):
                raise ValueError("ascending must be a single bool or one bool per column")
        return DataFrame(self._session, Sort(self._plan, exprs, ascending_list))

    # `sort` is the common alias for `order_by` (pandas/SQL-flavored naming).
    sort = order_by

    # ---- actions (trigger analysis, optimization, and execution) ----------
    def _optimized_plan(self) -> LogicalPlan:
        analyzed = analyze(self._plan)
        optimizer = Optimizer(default_rules(self._session.config.optimizer))
        return optimizer.optimize(analyzed)

    def _stages(self) -> list[Stage]:
        physical = plan_physical(
            self._optimized_plan(),
            shuffle_partitions=self._session.config.execution.shuffle_partitions,
        )
        return build_stages(physical)

    def _scheduler(self) -> LocalScheduler:
        engine = self._session.config.engine
        return LocalScheduler(num_workers=engine.num_workers, max_retries=engine.max_task_retries)

    def collect(self) -> list[Record]:
        dataset = self._scheduler().run_plan(self._stages())
        return list(dataset.iter_records())

    def count(self) -> int:
        dataset = self._scheduler().run_plan(self._stages())
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
        physical = plan_physical(
            optimized_plan, shuffle_partitions=self._session.config.execution.shuffle_partitions
        )
        print()
        print("== Physical Plan ==")
        print(explain_string(physical))
        stages = build_stages(physical)
        print()
        print("== Stages ==")
        for i, stage in enumerate(stages):
            if i:
                print()
            print(f"Stage {stage.stage_id} ({stage.num_partitions} partitions):")
            print(explain_string(stage.plan))
