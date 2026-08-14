"""Aggregated observability: per-stage and per-query summaries built from
TaskMetrics (execution/tasks.py). This is Milestone 8's answer to the
build spec's "Metrics" and (together with worker.py's psutil-based
fields) "Profiling" bullets.

Per-task metrics have existed since Milestone 3 (`TaskMetrics`), but were
only ever attached to an individual `TaskResult` and, past
`_results_to_dataset()` pulling out just `output_records`, discarded:
nothing aggregated them across a stage or a whole query, and
`DataFrame.collect()`/`show()`/`count()` never exposed them to a caller
at all. `StageMetrics`/`QueryMetrics` close that gap: `LocalScheduler.
run_plan()` builds one `QueryMetrics` per call, stored on `LocalScheduler.
last_metrics`, and `DataFrame` exposes the most recently collected one
via `DataFrame.last_run_metrics` (see api/dataframe.py).

Deliberately *not* threaded into `DataFrame.explain()`: `explain()` has
never executed anything (Milestone 1's behavior, still true today), and
conflating "show me the plan" with "run the query and show me what
happened" would be a real, unwanted change to an already-stable, highly
visible method's contract. Metrics are only ever available after an
action has actually run, mirroring how real Spark's execution metrics
come from the Spark UI/status tracker after a job runs, not from
`df.explain()`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from minispark.execution.tasks import TaskMetrics


def _sum_known_int(values: Iterable[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _sum_known_float(values: Iterable[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _max_known(values: Iterable[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return max(known) if known else None


@dataclass
class StageMetrics:
    stage_id: int
    num_tasks: int
    # Wall-clock time this stage's tasks actually took (from the
    # scheduler's perspective, spanning every retry and every batch);
    # under real parallelism this is smaller than
    # total_execution_time_seconds, the sum of each task's own time.
    wall_clock_seconds: float
    # How many distinct tasks needed at least one ordinary retry (not
    # counting a lineage-recovery-triggered rerun, see `recomputed`).
    retried_tasks: int
    # True if this StageMetrics represents a lineage-based recomputation
    # of the stage (execution/scheduler.py's _try_recover_missing_shuffle),
    # not its normal, once-per-query run. A recomputed stage produces a
    # *second* StageMetrics entry for the same stage_id in QueryMetrics.stages,
    # not a merge with the first: both runs did real, separately
    # measurable work.
    recomputed: bool
    total_execution_time_seconds: float
    total_input_records: int | None
    total_output_records: int
    total_output_bytes: int
    total_shuffle_bytes: int
    # None whenever psutil is not installed (an optional dependency, see
    # execution/worker.py); summed/maxed only over tasks that did report
    # a value, silently undercounting if some tasks are unknown and
    # others are not, a known imprecision rather than a promise every
    # task contributed.
    total_cpu_time_seconds: float | None
    max_peak_memory_bytes: int | None

    @staticmethod
    def from_task_metrics(
        stage_id: int,
        metrics: list[TaskMetrics],
        wall_clock_seconds: float,
        retried_tasks: int,
        recomputed: bool,
    ) -> StageMetrics:
        return StageMetrics(
            stage_id=stage_id,
            num_tasks=len(metrics),
            wall_clock_seconds=wall_clock_seconds,
            retried_tasks=retried_tasks,
            recomputed=recomputed,
            total_execution_time_seconds=sum(m.execution_time_seconds for m in metrics),
            total_input_records=_sum_known_int(m.input_records for m in metrics),
            total_output_records=sum(m.output_records for m in metrics),
            total_output_bytes=sum(m.output_bytes for m in metrics),
            total_shuffle_bytes=sum(m.shuffle_bytes for m in metrics),
            total_cpu_time_seconds=_sum_known_float(m.cpu_time_seconds for m in metrics),
            max_peak_memory_bytes=_max_known(m.peak_memory_bytes for m in metrics),
        )


@dataclass
class QueryMetrics:
    stages: list[StageMetrics] = field(default_factory=list)
    total_wall_clock_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Query: {self.total_wall_clock_seconds:.3f}s wall clock, "
            f"{len(self.stages)} stage run(s)"
        ]
        for s in self.stages:
            tag = " (lineage recompute)" if s.recomputed else ""
            cpu = (
                f"{s.total_cpu_time_seconds:.3f}s"
                if s.total_cpu_time_seconds is not None
                else "n/a"
            )
            mem = (
                f"{s.max_peak_memory_bytes / 1e6:.1f}MB"
                if s.max_peak_memory_bytes is not None
                else "n/a"
            )
            in_records = s.total_input_records if s.total_input_records is not None else "n/a"
            lines.append(
                f"  Stage {s.stage_id}{tag}: {s.num_tasks} task(s), "
                f"{s.wall_clock_seconds:.3f}s wall / {s.total_execution_time_seconds:.3f}s "
                f"task-sum, in={in_records} out={s.total_output_records} "
                f"shuffle_bytes={s.total_shuffle_bytes} retries={s.retried_tasks} "
                f"cpu={cpu} peak_mem={mem}"
            )
        return "\n".join(lines)
