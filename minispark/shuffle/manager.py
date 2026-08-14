"""ShuffleManager: driver-side bookkeeping for one query's shuffles.

Owns the scratch directory every shuffle block for this query is written
under, and tracks which blocks exist for which (stage, target partition)
once map tasks report them. This bookkeeping is driver-only, in-memory: a
worker process does not have access to a live `ShuffleManager` (worker
processes do not share memory with the driver), so a reduce task is
handed the exact block list it needs (via `blocks_for`) as plain, picklable
data attached to its `Task` (see execution/tasks.py, execution/
scheduler.py), not a reference to this object.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass

from minispark.shuffle.writer import ShuffleBlockMeta


@dataclass
class ShufflePartitionMetrics:
    """Observability for one target partition's shuffle read: how many
    blocks it was assembled from, and their combined size. Threaded into
    TaskMetrics.shuffle_bytes for a reduce task (execution/worker.py)."""

    block_count: int
    record_count: int
    byte_length: int


class ShuffleManager:
    def __init__(self, root_dir: str | None = None):
        self.root_dir = root_dir or tempfile.mkdtemp(prefix="minispark-shuffle-")
        self._blocks: dict[int, list[ShuffleBlockMeta]] = {}

    def register_blocks(self, stage_id: int, blocks: list[ShuffleBlockMeta]) -> None:
        """Overwrite, not append: a stage is normally registered exactly
        once, but lineage-based recomputation (execution/scheduler.py) can
        re-run a whole stage and call this a second time for the same
        stage_id, and its fresh blocks must fully replace the stale ones,
        not sit alongside them (the old block files that recomputation was
        triggered by are gone or corrupt; keeping their metadata around
        would let a later reader pick them again)."""
        self._blocks[stage_id] = list(blocks)

    def blocks_for(self, stage_id: int, target_partition: int) -> list[ShuffleBlockMeta]:
        return [
            b for b in self._blocks.get(stage_id, []) if b.target_partition == target_partition
        ]

    def metrics_for(self, stage_id: int, target_partition: int) -> ShufflePartitionMetrics:
        blocks = self.blocks_for(stage_id, target_partition)
        return ShufflePartitionMetrics(
            block_count=len(blocks),
            record_count=sum(b.record_count for b in blocks),
            byte_length=sum(b.byte_length for b in blocks),
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.root_dir, ignore_errors=True)
