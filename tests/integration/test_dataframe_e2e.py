from minispark.api.functions import col
from minispark.api.session import MiniSparkSession


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
    """filter()/select() must not call the executor at all."""
    import minispark.api.dataframe as dataframe_module

    calls = []
    monkeypatch.setattr(
        dataframe_module, "execute", lambda plan: calls.append(plan) or (_ for _ in ()).throw(
            AssertionError("execute() should not run during plan construction")
        )
    )
    session = make_session()
    df = session.create_dataframe([{"a": 1}], num_partitions=1)
    df.filter(col("a") > 0).select("a")  # no .collect()/.show()/.count()
    assert calls == []
