import pytest

from minispark.api.functions import col, lit
from minispark.api.session import MiniSparkSession
from minispark.logical.analyzer import AnalysisException


def make_session():
    return MiniSparkSession.builder.master("local[2]").app_name("test").get_or_create()


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
    """filter()/select() must not analyze, optimize, plan, or execute anything."""
    import minispark.physical.operators as physical_operators

    def _fail(plan):
        raise AssertionError("execute() should not run during plan construction")

    monkeypatch.setattr(physical_operators, "execute", _fail)
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
    assert "ScanExec[" in out
    # constant folding should have collapsed "10 + 10" down to 20 by the
    # time the optimized plan is printed.
    assert "Literal(20)" in out


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
