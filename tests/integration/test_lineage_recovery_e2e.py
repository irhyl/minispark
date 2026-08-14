"""Proof that lineage-based recomputation (Milestone 6) works end to end,
through the real DataFrame path, real disk-backed shuffle, and real OS
worker processes (`local[2]`), not just the stub-based orchestration
tests in tests/unit/test_scheduler_lineage_recovery.py.

The shuffle-write stage (group_by's map side) runs normally and writes
real blocks to disk. Before the shuffle-read stage's tasks run,
`_run_task_deleting_shuffle_once` deletes exactly one target partition's
block files, real files, on real disk, simulating that partition's data
being lost after it was already produced (e.g. a disk failure). Plain
task retry could never recover from this: the file is gone, retrying the
same read would just fail again. If the query still produces the correct
result, the map stage was genuinely recomputed, not just retried.

`local[2]` also exercises something the unit-level stub tests cannot:
`TaskResult.missing_shuffle_stage_id` surviving a real trip through
`pickle` across a `ProcessPoolExecutor` boundary.
"""

from __future__ import annotations

import collections
import functools
import os

from minispark.api.functions import sum as ssum
from minispark.api.session import MiniSparkSession
from minispark.execution.scheduler import LocalScheduler
from minispark.execution.stages import build_stages
from minispark.execution.tasks import Task, TaskResult
from minispark.execution.worker import execute_task
from minispark.logical.analyzer import analyze
from minispark.optimizer.optimizer import Optimizer, default_rules
from minispark.physical.planner import plan_physical


def _run_task_deleting_shuffle_once(marker_path: str, task: Task, attempt: int) -> TaskResult:
    if task.shuffle_blocks and not os.path.exists(marker_path):
        try:
            open(marker_path, "x").close()  # atomic create: only one task wins this race
        except FileExistsError:
            pass
        else:
            for blocks in task.shuffle_blocks.values():
                for block in blocks:
                    try:
                        os.remove(block.path)
                    except FileNotFoundError:
                        pass
    return execute_task(task, attempt)


def test_lineage_recovers_a_deleted_shuffle_block_under_real_multiprocessing(tmp_path):
    session = (
        MiniSparkSession.builder.master("local[2]").app_name("lineage_test").get_or_create()
    )
    countries = ["US", "CA", "UK", "DE"]
    records = [{"country": countries[i % len(countries)], "revenue": i} for i in range(20)]
    df = session.create_dataframe(records, num_partitions=4)
    grouped = df.group_by("country").agg(ssum("revenue").alias("total"))

    # Mirrors DataFrame._stages(): build the real physical plan and real
    # stages, then drive them through a LocalScheduler whose run_task is
    # wrapped to inject the deletion, since DataFrame.collect() does not
    # expose a way to intercept task execution.
    analyzed = analyze(grouped.plan)
    optimized = Optimizer(default_rules(session.config.optimizer)).optimize(analyzed)
    physical = plan_physical(
        optimized, shuffle_partitions=session.config.execution.shuffle_partitions
    )
    stages = build_stages(physical)

    marker_path = str(tmp_path / "deleted_once.marker")
    run_task = functools.partial(_run_task_deleting_shuffle_once, marker_path)
    scheduler = LocalScheduler(num_workers=2, max_retries=1, run_task=run_task)

    dataset = scheduler.run_plan(stages)
    rows = {r["country"]: r["total"] for r in dataset.iter_records()}

    reference: dict[str, int] = collections.defaultdict(int)
    for r in records:
        reference[r["country"]] += r["revenue"]

    assert rows == dict(reference)
    assert os.path.exists(marker_path)  # sanity: the injected loss actually fired
