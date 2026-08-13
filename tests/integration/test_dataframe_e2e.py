import pytest

from minispark.api.functions import col, lit
from minispark.api.session import MiniSparkSession
from minispark.execution.tasks import TaskResult, TaskState
from minispark.logical.analyzer import AnalysisException


def make_session():
    # local[1]: exercises the full Task/Stage/LocalScheduler path added in
    # Milestone 3, but sequentially in this process, not through a real
    # ProcessPoolExecutor. Keeps this file's routine correctness tests fast
    # and free of OS-process-spawn overhead; real multiprocessing (local[N]
    # with N > 1) gets its own dedicated tests in
    # tests/integration/test_scheduler_multiprocessing.py.
    return MiniSparkSession.builder.master("local[1]").app_name("test").get_or_create()


def test_filter_select_collect_on_memory_data():
    session = make_session()
    records = [
        {"name": "alice", "age": 30, "country": "US"},
        {"name": "bob", "age": 17, "country": "CA"},
        {"name": "carol", "age": 45, "country": "US"},
    ]
    df = session.create_dataframe(records, num_partitions=2)

    result = df.filter(col("age") > 18).select("name", "age")

    rows = result.collect()
    assert sorted(rows, key=lambda r: r["name"]) == [
        {"name": "alice", "age": 30},
        {"name": "carol", "age": 45},
    ]
    assert result.count() == 2
    assert result.schema.field_names() == ["name", "age"]


def test_filter_select_collect_on_csv(tmp_path):
    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "name,age,country\n"
        "alice,30,US\n"
        "bob,17,CA\n"
        "carol,45,US\n"
        "dave,19,UK\n",
        encoding="utf-8",
    )
    session = make_session()
    df = session.read.csv(str(csv_path), num_partitions=2)

    result = df.filter(col("age") >= 18).select("name", "country")

    rows = sorted(result.collect(), key=lambda r: r["name"])
    assert rows == [
        {"name": "alice", "country": "US"},
        {"name": "carol", "country": "US"},
        {"name": "dave", "country": "UK"},
    ]


def test_explain_shows_plan_tree(capsys):
    session = make_session()
    df = session.create_dataframe([{"a": 1}], num_partitions=1)
    df.filter(col("a") > 0).select("a").explain()
    captured = capsys.readouterr()
    assert "Project[" in captured.out
    assert "Filter[" in captured.out
    assert "Scan[" in captured.out


def test_show_prints_table(capsys):
    session = make_session()
    df = session.create_dataframe(
        [{"name": "alice", "age": 30}, {"name": "bob", "age": 17}], num_partitions=1
    )
    df.filter(col("age") > 18).show()
    out = capsys.readouterr().out
    assert "alice" in out
    assert "bob" not in out


def test_no_execution_before_action(monkeypatch):
    """filter()/select() must not analyze, optimize, plan, schedule, or execute
    anything. Patches the `execute_task` name inside execution/scheduler.py
    (not execution/worker.py, where it is defined): scheduler.py imports it
    with `from ... import execute_task`, which copies the reference into
    its own module namespace, so that is the binding LocalScheduler
    actually calls and the one that needs patching."""
    import minispark.execution.scheduler as scheduler_module

    def _fail(task, attempt_number=0):
        raise AssertionError("execute_task() should not run during plan construction")

    monkeypatch.setattr(scheduler_module, "execute_task", _fail)
    session = make_session()
    df = session.create_dataframe([{"a": 1}], num_partitions=1)
    df.filter(col("a") > 0).select("a")  # no .collect()/.show()/.count()


def test_select_unknown_column_does_not_fail_until_an_action_runs():
    session = make_session()
    df = session.create_dataframe([{"name": "alice", "age": 30}], num_partitions=1)
    built = df.select("does_not_exist")  # building the plan alone must not raise
    with pytest.raises(AnalysisException, match="does_not_exist"):
        built.collect()


def test_filter_unknown_column_raises_analysis_exception_on_collect():
    session = make_session()
    df = session.create_dataframe([{"name": "alice", "age": 30}], num_partitions=1)
    with pytest.raises(AnalysisException, match="does_not_exist"):
        df.filter(col("does_not_exist") > 0).collect()


def test_explain_optimized_shows_analyzed_optimized_and_physical_sections(capsys):
    session = make_session()
    df = session.create_dataframe(
        [{"name": "alice", "age": 30, "country": "US"}], num_partitions=1
    )
    df.select("name", "age").filter(col("age") > (lit(10) + lit(10))).explain(optimized=True)
    out = capsys.readouterr().out
    assert "== Analyzed Logical Plan ==" in out
    assert "== Optimized Logical Plan ==" in out
    assert "== Physical Plan ==" in out
    assert "== Stages ==" in out
    assert "ScanExec[" in out
    assert "Stage 0 (1 partitions)" in out
    # constant folding should have collapsed "10 + 10" down to 20 by the
    # time the optimized plan is printed.
    assert "Literal(20)" in out


def test_task_retry_recovers_from_a_transient_failure(monkeypatch):
    """A task that fails once and succeeds on retry must still produce a
    correct final result: proves LocalScheduler's retry loop end to end
    through the real DataFrame path, not just that retry logic exists in
    isolation (see tests/unit/test_scheduler.py for that)."""
    import minispark.execution.scheduler as scheduler_module

    real_execute_task = scheduler_module.execute_task
    failed_once = False

    def flaky(task, attempt_number=0):
        nonlocal failed_once
        if task.partition_id == 0 and not failed_once:
            failed_once = True
            return TaskResult(task_id=task.task_id, state=TaskState.FAILED, error="injected")
        return real_execute_task(task, attempt_number)

    monkeypatch.setattr(scheduler_module, "execute_task", flaky)

    session = make_session()
    df = session.create_dataframe(
        [{"name": "alice", "age": 30}, {"name": "bob", "age": 17}], num_partitions=1
    )
    rows = df.filter(col("age") > 18).collect()

    assert rows == [{"name": "alice", "age": 30}]
    assert failed_once is True


def test_optimization_does_not_change_query_results():
    session = make_session()
    records = [
        {"name": "alice", "age": 30, "country": "US"},
        {"name": "bob", "age": 17, "country": "CA"},
        {"name": "carol", "age": 45, "country": "US"},
    ]
    df = session.create_dataframe(records, num_partitions=2)
    result = df.select("name", "age").filter(col("age") > (lit(10) + lit(10)))
    rows = sorted(result.collect(), key=lambda r: r["name"])
    assert rows == [{"name": "alice", "age": 30}, {"name": "carol", "age": 45}]
