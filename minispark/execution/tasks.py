"""Task: the unit of work a Worker executes, one partition of one stage.

`Task.plan` is the whole stage's PhysicalPlan, shared by every task in
that stage; only `partition_id` differs between them (see
physical/operators.py's `execute_partition()`, which is what actually
runs it). `Task` and everything it carries must be picklable: a Task is
sent whole to a worker process when `local[N]` with `N > 1` is in use
(see execution/scheduler.py). Task ids are unique across the whole run
(not just within a stage): execution/scheduler.py numbers them
sequentially as stages run, since a task in a later stage needs an id
distinct from every task in every earlier stage.

Two fields only matter for a task whose stage reads shuffle input (its
plan contains a `ShuffleReadExec`, see execution/stages.py):
`shuffle_root_dir` (where every block for this query was written) and
`shuffle_blocks` (exactly the blocks this task's partition needs to read,
already filtered driver-side by `shuffle/manager.py`'s `ShuffleManager`).
Both are `None` for a task whose stage does not read shuffle input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from minispark.core.record import Record
from minispark.physical.plan import PhysicalPlan
from minispark.shuffle.writer import ShuffleBlockMeta


class TaskState(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()


@dataclass(frozen=True)
class Task:
    task_id: int
    stage_id: int
    partition_id: int
    plan: PhysicalPlan
    shuffle_root_dir: str | None = None
    shuffle_blocks: list[ShuffleBlockMeta] | None = None


@dataclass(frozen=True)
class TaskContext:
    """Per-attempt identity, threaded into a task's execution for logging.

    Not a general-purpose side-channel (no accumulators, no broadcast
    variable access): nothing in the physical operators reads from a
    TaskContext yet, this exists so Worker.execute_task can log/report
    "which task, which attempt" without smuggling that information through
    extra positional arguments.
    """

    task_id: int
    stage_id: int
    partition_id: int
    attempt_number: int = 0


@dataclass
class TaskMetrics:
    execution_time_seconds: float = 0.0
    input_records: int | None = None
    output_records: int = 0
    output_bytes: int = 0
    # Not implemented: would need either byte-offset tracking in the
    # storage layer (input_bytes) or the `psutil` optional dependency
    # (cpu_time_seconds, peak_memory_bytes), neither of which exists yet.
    input_bytes: int | None = None
    cpu_time_seconds: float | None = None
    peak_memory_bytes: int | None = None
    # 0 for a task with no shuffle input or output. For a shuffle-write
    # task, the total bytes written across all target partitions; for a
    # shuffle-read task, the total bytes read for its one target
    # partition (see shuffle/manager.py's ShufflePartitionMetrics).
    shuffle_bytes: int = 0


@dataclass
class TaskResult:
    task_id: int
    state: TaskState
    rows: list[Record] = field(default_factory=list)
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    error: str | None = None
    # Populated only for a shuffle-write task: the blocks it wrote, which
    # execution/scheduler.py registers into the ShuffleManager so a
    # downstream stage's tasks know what to read.
    shuffle_blocks: list[ShuffleBlockMeta] = field(default_factory=list)
