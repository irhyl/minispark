"""DataSource: the abstract interface every storage backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from minispark.core.dataset import Dataset
from minispark.expressions.base import Expression


class DataSource(ABC):
    @abstractmethod
    def read(
        self, columns: list[str] | None = None, filter: Expression | None = None
    ) -> Dataset:
        """Produce a Dataset. Partition row-data must stay lazy (see Partition);
        only schema and partition boundaries may be computed eagerly here.

        `columns` and `filter` are optional pushdown hints, not a contract
        every source must honor: a source that ignores one (or both) must
        still return fully correct, unrestricted data, since the physical
        plan always keeps its own Project/Filter operators above a Scan
        regardless of whether pushdown happened underneath them (see
        physical/planner.py's scan-pushdown pass). `filter`, when honored,
        is always treated as a hint for skipping work, never the sole
        source of correctness: a source may return a superset of the
        matching rows (never a subset), and the row-level Filter above it
        is what guarantees the final result is exactly right.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source identifier, shown in Scan's explain() label."""
