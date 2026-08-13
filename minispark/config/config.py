"""Engine configuration.

Mirrors the shape sketched in the build prompt (engine/execution/memory/
optimizer sections) so later milestones can add fields without restructuring.
Reality check, as of Milestone 4: `engine.master` (via `num_workers`) and
`engine.max_task_retries` are read by `execution/scheduler.py`'s
`LocalScheduler`; `optimizer.predicate_pushdown` and
`optimizer.projection_pruning` are read by `optimizer/optimizer.py`'s
`default_rules()`; `execution.shuffle_partitions` is read by
`physical/planner.py` when translating an Aggregate (how many reduce-side
partitions a shuffle fans out to). `execution.partition_size_mb`,
`execution.shuffle_compression`, and `memory` are still unread.

YAML loading (the build prompt's example config file) is intentionally not
implemented yet: it would be a config *format* with no config *consumers*
behind it. Added when the first subsystem needs to be configured from a
file rather than constructed in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    master: str = "local[4]"
    max_task_retries: int = 3

    @property
    def num_workers(self) -> int:
        """Parse 'local[N]' / 'local' into a worker count.

        Read by `execution/scheduler.py`'s `LocalScheduler`: `N == 1`
        runs tasks sequentially in-process, `N > 1` runs them across a
        real `ProcessPoolExecutor`.
        """
        if self.master == "local":
            return 1
        if self.master.startswith("local[") and self.master.endswith("]"):
            return int(self.master[len("local[") : -1])
        raise ValueError(f"Unsupported master: {self.master!r} (only 'local[N]' is supported)")


@dataclass
class ExecutionConfig:
    partition_size_mb: int = 128
    shuffle_compression: bool = True
    shuffle_partitions: int = 4


@dataclass
class MemoryConfig:
    execution_limit_mb: int = 4096
    spill_threshold: float = 0.8


@dataclass
class OptimizerConfig:
    predicate_pushdown: bool = True
    projection_pruning: bool = True
    broadcast_join_threshold_mb: int = 50


@dataclass
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
