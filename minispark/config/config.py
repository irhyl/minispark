"""Engine configuration.

Mirrors the shape sketched in the build prompt (engine/execution/memory/
optimizer sections) so later milestones can add fields without restructuring.
Milestone-1 reality check: only `engine.master` is read anywhere right now
(by MiniSparkSession, purely for display/storage), because there is no
scheduler, optimizer, or memory manager yet to consume the rest of these
values. They exist now so each subsystem introduced later has a config
home from day one instead of hardcoded constants.

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
        """Parse 'local[N]' / 'local' into a worker count. Not yet consumed
        by anything — no local scheduler exists until Milestone 3."""
        if self.master == "local":
            return 1
        if self.master.startswith("local[") and self.master.endswith("]"):
            return int(self.master[len("local[") : -1])
        raise ValueError(f"Unsupported master: {self.master!r} (only 'local[N]' in Milestone 1)")


@dataclass
class ExecutionConfig:
    partition_size_mb: int = 128
    shuffle_compression: bool = True


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
