"""Proof that `local[N]` with N > 1 uses real OS processes, not a simulated
or in-process stand-in for parallelism.

The helper task functions below are module-level (not nested inside a test
function) on purpose: `ProcessPoolExecutor` sends callables to worker
processes with `pickle`, and a closure or a locally-defined function is
not picklable no matter what it captures (see storage/memory.py's
`_make_records_fn` docstring for the same constraint applied to Partition
row data).
"""

from __future__ import annotations

import os

from minispark.api.functions import col
from minispark.api.session import MiniSparkSession
from minispark.core.dataset import Dataset
from minispark.core.schema import Field, Schema
from minispark.core.types import INT
from minispark.execution.scheduler import LocalScheduler
from minispark.execution.stages import Stage
from minispark.execution.tasks import Task, TaskMetrics, TaskResult, TaskState
from minispark.physical.plan import ScanExec


def _report_pid(task: Task, attempt: int) -> TaskResult:
    return TaskResult(
        task_id=task.task_id,
        state=TaskState.SUCCESS,
        rows=[{"partition_id": task.partition_id, "pid": os.getpid()}],
        metrics=TaskMetrics(output_records=1),
    )


def test_local_n_runs_tasks_in_real_child_processes():
    """Every task must run with a PID different from this (the driver)
    process's PID: if `local[N]` with N > 1 quietly ran everything
    in-process, every row would report this test's own PID instead."""
    # _report_pid never reads `task.plan`; still a real, fully-formed
    # ScanExec (not a placeholder/uninitialized object) so pickling this
    # Task exercises the genuine object graph a real Task carries.
    empty_dataset = Dataset(Schema([Field("x", INT)]), [])
    scan_plan = ScanExec(empty_dataset, "unused")
    stage = Stage(stage_id=0, plan=scan_plan, num_partitions=4)
    scheduler = LocalScheduler(num_workers=2, run_task=_report_pid)

    dataset = scheduler.run_stage(stage)
    rows = list(dataset.iter_records())

    assert len(rows) == 4
    main_pid = os.getpid()
    worker_pids = {row["pid"] for row in rows}
    assert main_pid not in worker_pids
    assert 1 <= len(worker_pids) <= 2


def test_real_end_to_end_collect_under_local_n():
    """The full DataFrame path (analyze, optimize, physical plan, stage,
    LocalScheduler, real ProcessPoolExecutor, merge) must produce correct
    results, not just avoid crashing."""
    session = (
        MiniSparkSession.builder.master("local[2]").app_name("mp_test").get_or_create()
    )
    records = [
        {"name": "alice", "age": 30},
        {"name": "bob", "age": 17},
        {"name": "carol", "age": 45},
        {"name": "dave", "age": 19},
    ]
    df = session.create_dataframe(records, num_partitions=4)
    result = df.filter(col("age") > 18).select("name")

    rows = sorted(result.collect(), key=lambda r: r["name"])
    assert rows == [{"name": "alice"}, {"name": "carol"}, {"name": "dave"}]


def test_csv_byte_offset_seeking_is_correct_across_real_worker_processes(tmp_path):
    """Milestone 9: each CSV partition's records_fn (storage/csv.py's
    `_read_csv_range`) seeks to a byte offset computed once by the driver
    (`_locate_partition_offsets`) and pickled into the Task sent to a
    worker process. This proves that offset survives the real pickle/
    unpickle/seek round-trip through a real ProcessPoolExecutor worker,
    not just in-process, and that every partition reads its own rows
    exactly once (no row dropped or duplicated by an off-by-one in the
    offset math) with enough partitions that every worker handles more
    than one."""
    csv_path = tmp_path / "wide.csv"
    lines = ["id,value"] + [f"{i},{i * 7}" for i in range(200)]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    session = (
        MiniSparkSession.builder.master("local[3]").app_name("csv_offset_mp_test").get_or_create()
    )
    df = session.read.csv(str(csv_path), num_partitions=9)
    rows = sorted(df.collect(), key=lambda r: r["id"])

    assert rows == [{"id": i, "value": i * 7} for i in range(200)]
