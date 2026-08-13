"""DataSource: the abstract interface every storage backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from minispark.core.dataset import Dataset


class DataSource(ABC):
    @abstractmethod
    def read(self) -> Dataset:
        """Produce a Dataset. Partition row-data must stay lazy (see Partition);
        only schema and partition boundaries may be computed eagerly here."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source identifier, shown in Scan's explain() label."""
