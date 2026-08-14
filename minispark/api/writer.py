"""DataFrameWriter: `df.write.parquet(path)`.

Mirrors `DataFrameReader` (api/session.py) on the write side: a thin
companion object holding a reference to the `DataFrame` it writes,
exposing one method per output format. Only Parquet exists (Milestone 7);
`DataFrame.checkpoint()` (api/dataframe.py) is the only other thing that
writes a DataFrame's result to disk, and uses a different, simpler,
internal-only pickle-based format, not this class, since a checkpoint is
not meant to be a portable, externally-readable file format the way
`write.parquet()`'s output is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minispark.api.dataframe import DataFrame


class DataFrameWriter:
    def __init__(self, df: DataFrame):
        self._df = df

    def parquet(self, path: str) -> None:
        """Run this DataFrame now and write its result to `path` as one
        `.parquet` file per partition (`part-00000.parquet`, ...).

        Local import: `pyarrow` is an optional extra, see
        storage/parquet.py's module docstring and `DataFrameReader.
        parquet()` (api/session.py) for why the import has to live inside
        the method body, not at this module's top.
        """
        from minispark.storage.parquet import write_parquet_dataset

        dataset = self._df._collect_dataset()
        write_parquet_dataset(dataset, path)
