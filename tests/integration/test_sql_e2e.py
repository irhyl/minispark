"""End-to-end tests for session.sql() through the real path: analyzer,
optimizer, physical planner (including real scan pushdown), stage
splitting, and LocalScheduler, including real OS-process parallelism
(`local[3]`) for the shuffle-heavy queries. Correctness is checked
against a plain Python computation over the same data, independent of
MiniSpark, per the build spec's "MiniSpark result == reference result"
rule, and against the equivalent DataFrame API call, since both must
produce byte-identical logical plans, not just byte-identical rows.
"""

from __future__ import annotations

import collections

from minispark.api.functions import col, count
from minispark.api.functions import sum as ssum
from minispark.api.session import MiniSparkSession
from minispark.logical.analyzer import AnalysisException
from minispark.logical.plan import explain_string
from minispark.optimizer.optimizer import Optimizer, default_rules
from minispark.physical.planner import plan_physical
from minispark.sql.parser import SqlParseError

USERS = [
    {"name": "alice", "age": 30, "country": "US"},
    {"name": "bob", "age": 17, "country": "CA"},
    {"name": "carol", "age": 45, "country": "US"},
    {"name": "dave", "age": 19, "country": "UK"},
    {"name": "erin", "age": 15, "country": "UK"},
]

REGIONS = [
    {"country": "US", "region": "Americas"},
    {"country": "CA", "region": "Americas"},
    {"country": "UK", "region": "Europe"},
]


def make_session(master: str = "local[1]") -> MiniSparkSession:
    return MiniSparkSession.builder.master(master).app_name("sql_test").get_or_create()


def test_select_where_order_by_matches_reference():
    session = make_session("local[1]")
    users = session.create_dataframe(USERS, num_partitions=3)
    session.create_or_replace_temp_view("users", users)

    result = session.sql("SELECT name, age FROM users WHERE age >= 18 ORDER BY age DESC")
    rows = result.collect()

    reference = sorted(
        ({"name": u["name"], "age": u["age"]} for u in USERS if u["age"] >= 18),
        key=lambda r: -r["age"],
    )
    assert rows == reference


def test_group_by_having_matches_reference_under_real_multiprocessing():
    session = make_session("local[3]")
    users = session.create_dataframe(USERS, num_partitions=4)
    session.create_or_replace_temp_view("users", users)

    result = session.sql(
        "SELECT country, COUNT(*) AS n, SUM(age) AS total_age FROM users "
        "GROUP BY country HAVING COUNT(*) >= 2 ORDER BY country"
    )
    rows = result.collect()

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for u in USERS:
        groups[u["country"]].append(u)
    reference = sorted(
        (
            {"country": c, "n": len(g), "total_age": sum(u["age"] for u in g)}
            for c, g in groups.items()
            if len(g) >= 2
        ),
        key=lambda r: r["country"],
    )
    assert rows == reference


def test_join_matches_reference_under_real_multiprocessing():
    session = make_session("local[3]")
    users = session.create_dataframe(USERS, num_partitions=3)
    regions = session.create_dataframe(REGIONS, num_partitions=2)
    session.create_or_replace_temp_view("users", users)
    session.create_or_replace_temp_view("regions", regions)

    result = session.sql(
        "SELECT users.name, regions.region FROM users JOIN regions "
        "ON users.country = regions.country ORDER BY name"
    )
    rows = result.collect()

    region_by_country = {r["country"]: r["region"] for r in REGIONS}
    reference = sorted(
        (
            {"name": u["name"], "region": region_by_country[u["country"]]}
            for u in USERS
            if u["country"] in region_by_country
        ),
        key=lambda r: r["name"],
    )
    assert rows == reference


def test_select_star_matches_full_row_reference():
    session = make_session("local[1]")
    users = session.create_dataframe(USERS, num_partitions=2)
    session.create_or_replace_temp_view("users", users)

    result = session.sql("SELECT * FROM users WHERE country = 'US'")
    rows = sorted(result.collect(), key=lambda r: r["name"])
    reference = sorted((u for u in USERS if u["country"] == "US"), key=lambda r: r["name"])
    assert rows == reference


def test_global_aggregate_matches_reference():
    session = make_session("local[2]")
    users = session.create_dataframe(USERS, num_partitions=3)
    session.create_or_replace_temp_view("users", users)

    result = session.sql("SELECT COUNT(*) AS n, AVG(age) AS avg_age FROM users")
    (row,) = result.collect()
    assert row["n"] == len(USERS)
    assert row["avg_age"] == sum(u["age"] for u in USERS) / len(USERS)


def test_sql_and_dataframe_api_produce_the_same_optimized_plan():
    """A DataFrame built from SQL must be indistinguishable, downstream
    of parsing, from one built by chaining DataFrame API calls: this is
    the build spec's "no separate SQL execution engine" rule, checked
    directly by comparing the two paths' optimized-plan text."""
    session = make_session("local[1]")
    users = session.create_dataframe(USERS, num_partitions=2)
    session.create_or_replace_temp_view("users", users)

    sql_df = session.sql("SELECT name FROM users WHERE age >= 18")
    api_df = users.filter(col("age") >= 18).select("name")

    def optimized_text(df):
        from minispark.logical.analyzer import analyze

        analyzed = analyze(df.plan)
        optimized = Optimizer(default_rules(session.config.optimizer)).optimize(analyzed)
        return explain_string(optimized)

    assert optimized_text(sql_df) == optimized_text(api_df)


def test_sql_scan_pushdown_matches_dataframe_api_pushdown():
    """The physical plan's Scan (after the scan-pushdown pass) must show
    the same narrowed column set whether the query came from SQL or the
    DataFrame API: SQL is a translator into the same logical plan, so it
    gets exactly the same physical-planning-time pushdown for free."""
    session = make_session("local[1]")
    users = session.create_dataframe(USERS, num_partitions=2)
    session.create_or_replace_temp_view("users", users)

    sql_df = session.sql("SELECT name FROM users WHERE age >= 18")
    api_df = users.filter(col("age") >= 18).select("name")

    sql_physical = plan_physical(sql_df._optimized_plan())
    api_physical = plan_physical(api_df._optimized_plan())
    assert explain_string(sql_physical) == explain_string(api_physical)


def test_explain_works_on_a_sql_built_dataframe(capsys):
    session = make_session("local[1]")
    users = session.create_dataframe(USERS, num_partitions=2)
    session.create_or_replace_temp_view("users", users)

    result = session.sql("SELECT name FROM users WHERE age >= 18 ORDER BY name")
    result.explain(optimized=True)
    out = capsys.readouterr().out
    assert "== Physical Plan ==" in out
    assert "== Stages ==" in out


def test_last_run_metrics_works_on_a_sql_built_dataframe():
    session = make_session("local[2]")
    users = session.create_dataframe(USERS, num_partitions=3)
    session.create_or_replace_temp_view("users", users)

    result = session.sql(
        "SELECT country, COUNT(*) AS n FROM users GROUP BY country"
    )
    result.collect()
    assert result.last_run_metrics is not None
    assert len(result.last_run_metrics.stages) == 2


def test_unknown_column_raises_analysis_exception_not_a_silent_wrong_answer():
    session = make_session("local[1]")
    users = session.create_dataframe(USERS, num_partitions=1)
    session.create_or_replace_temp_view("users", users)

    df = session.sql("SELECT does_not_exist FROM users")
    try:
        df.collect()
        raise AssertionError("expected AnalysisException")
    except AnalysisException:
        pass


def test_unknown_table_raises_before_touching_the_scheduler():
    session = make_session("local[1]")
    try:
        session.sql("SELECT * FROM nope")
        raise AssertionError("expected SqlParseError")
    except SqlParseError:
        pass


def test_checkpoint_works_on_a_sql_built_dataframe(tmp_path):
    session = make_session("local[1]")
    users = session.create_dataframe(USERS, num_partitions=2)
    session.create_or_replace_temp_view("users", users)

    result = session.sql("SELECT name, age FROM users WHERE age >= 18")
    checkpointed = result.checkpoint(str(tmp_path / "cp"))
    rows = sorted(checkpointed.collect(), key=lambda r: r["name"])
    reference = sorted(
        ({"name": u["name"], "age": u["age"]} for u in USERS if u["age"] >= 18),
        key=lambda r: r["name"],
    )
    assert rows == reference


def test_create_or_replace_temp_view_replaces_a_previous_registration():
    session = make_session("local[1]")
    first = session.create_dataframe([{"x": 1}], num_partitions=1)
    second = session.create_dataframe([{"x": 2}], num_partitions=1)
    session.create_or_replace_temp_view("t", first)
    session.create_or_replace_temp_view("t", second)
    rows = session.sql("SELECT * FROM t").collect()
    assert rows == [{"x": 2}]


def test_count_aggregate_and_sum_functions_wired_through_functions_module():
    """Sanity: session.sql()'s COUNT/SUM produce the same result as the
    api/functions.py DataFrame equivalents on the same data."""
    session = make_session("local[1]")
    users = session.create_dataframe(USERS, num_partitions=2)
    session.create_or_replace_temp_view("users", users)

    sql_rows = session.sql(
        "SELECT country, COUNT(*) AS n, SUM(age) AS total FROM users GROUP BY country "
        "ORDER BY country"
    ).collect()
    api_rows = (
        users.group_by("country")
        .agg(count("*").alias("n"), ssum("age").alias("total"))
        .order_by("country")
        .collect()
    )
    assert sql_rows == api_rows
