"""Proof that DataFrame.checkpoint() (api/dataframe.py) does what it
claims: materializes the current result to disk, and returns a DataFrame
whose plan is a fresh Scan over that checkpoint, not the original plan.

Two things are checked that a plan-shape assertion alone would not catch:
the source feeding the pre-checkpoint plan is read exactly once by
checkpoint() and never again by anything built on the returned DataFrame
(proving lineage is actually cut, not just relabeled), and the final
result is still correct under real `local[2]` multiprocessing.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from pathlib import Path

from minispark.api.dataframe import DataFrame
from minispark.api.functions import col
from minispark.api.session import MiniSparkSession
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.core.schema import Field, Schema
from minispark.core.types import INT
from minispark.logical.nodes import Scan

SCHEMA = Schema([Field("x", INT)])


def _counting_records_fn(counter_path: str, rows: list[dict]) -> Iterator[Record]:
    with open(counter_path, "a", encoding="utf-8") as f:
        f.write("1\n")
    return iter(rows)


def _count_reads(counter_path: str) -> int:
    return len(Path(counter_path).read_text(encoding="utf-8").splitlines())


def make_counting_dataset(counter_path: str, rows_per_partition: list[list[dict]]) -> Dataset:
    partitions = [
        Partition(
            i,
            SCHEMA,
            functools.partial(_counting_records_fn, counter_path, rows),
            PartitionMetadata(row_count=len(rows)),
        )
        for i, rows in enumerate(rows_per_partition)
    ]
    return Dataset(SCHEMA, partitions)


def test_checkpoint_truncates_lineage_under_real_multiprocessing(tmp_path):
    counter_path = str(tmp_path / "reads.count")
    Path(counter_path).write_text("", encoding="utf-8")
    dataset = make_counting_dataset(counter_path, [[{"x": 1}, {"x": 2}], [{"x": 3}]])

    session = (
        MiniSparkSession.builder.master("local[2]").app_name("checkpoint_test").get_or_create()
    )
    df = DataFrame(session, Scan(dataset, source_name="counting"))
    filtered = df.filter(col("x") > 0)

    checkpoint_dir = str(tmp_path / "cp")
    checkpointed = filtered.checkpoint(checkpoint_dir)

    # One read per source partition, from running the pre-checkpoint plan
    # exactly once inside checkpoint().
    assert _count_reads(counter_path) == 2

    # The new DataFrame's plan is a bare Scan over the checkpoint, not the
    # Filter-over-"counting" plan that produced it.
    assert isinstance(checkpointed.plan, Scan)
    assert checkpointed.plan.source_name.startswith("checkpoint:")

    result = checkpointed.filter(col("x") >= 2).collect()
    assert sorted(r["x"] for r in result) == [2, 3]

    # Collecting the checkpointed DataFrame (even through a further
    # transformation) must not have gone back to the original source.
    assert _count_reads(counter_path) == 2


def test_checkpoint_preserves_row_data_exactly(tmp_path):
    counter_path = str(tmp_path / "reads.count")
    Path(counter_path).write_text("", encoding="utf-8")
    dataset = make_counting_dataset(counter_path, [[{"x": 5}], [{"x": 6}, {"x": 7}]])

    session = (
        MiniSparkSession.builder.master("local[1]").app_name("checkpoint_test").get_or_create()
    )
    df = DataFrame(session, Scan(dataset, source_name="counting"))

    checkpointed = df.checkpoint(str(tmp_path / "cp"))
    assert sorted(r["x"] for r in checkpointed.collect()) == [5, 6, 7]
