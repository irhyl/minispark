# SQL

How `session.sql(...)` works, as of Milestone 8. See `minispark.sql.
parser`'s module docstring for the exact grammar; this document is the
"why," not a grammar reference.

## No separate SQL execution engine

The build spec is explicit: "there must not be a separate SQL execution
engine." `sql/parser.py`'s `parse_sql()` does not interpret SQL itself;
it translates SQL text into exactly the same `LogicalPlan` nodes
(`Scan`, `Filter`, `Project`, `Aggregate`, `Join`, `Sort`) and
expressions the DataFrame API builds by chaining `.filter()`/`.select()`/
`.group_by().agg()`/`.join()`/`.order_by()`. `MiniSparkSession.sql()`
hands the resulting `LogicalPlan` to a plain `DataFrame`, which then goes
through the exact same `analyze()` -> `Optimizer.optimize()` ->
`plan_physical()` -> `build_stages()` -> `LocalScheduler.run_plan()`
path any other `DataFrame` does. `tests/integration/test_sql_e2e.py`'s
`test_sql_and_dataframe_api_produce_the_same_optimized_plan` and
`test_sql_scan_pushdown_matches_dataframe_api_pushdown` check this
directly, by comparing `explain_string()` output between a SQL-built and
an API-built `DataFrame` for the equivalent query, not just comparing
final rows.

One consequence: a SQL query gets Milestone 7's real scan pushdown
(column pruning, and for Parquet, row-group-level predicate pushdown)
for free, with no SQL-specific code needed for it. The scan-pushdown
pass runs in `physical/planner.py`, downstream of where SQL parsing
ends; it has no idea whether the logical plan it is looking at came from
SQL or from `.filter().select()`, and does not need to.

## Why a hand-written parser

The build spec's allowed-dependencies line explicitly permits "a
lightweight parser if needed for SQL." `sql/tokenizer.py` and `sql/
parser.py` are both hand-written: a tokenizer (character stream ->
`Token`s) and a recursive-descent parser with precedence climbing for
expressions (`sql/parser.py`'s module docstring spells out the full
grammar). No parser-generator library, no grammar file: the supported
SQL subset (below) is small enough that a generated parser would be
more machinery than the problem needs, the same reasoning `optimizer/
rules.py` gives for not having a generic visitor abstraction over six
logical node types.

## Scope: a translator, not a capability expansion

SQL support is scoped to mirror what the DataFrame API can already
express, deliberately no more:

* `SELECT` list: `*`, or a comma-separated list of expressions
  (columns, arithmetic, aggregate function calls), each optionally
  `AS`-aliased.
* `FROM <table>`: `<table>` must be a name already registered with
  `session.create_or_replace_temp_view(name, df)`. There is no notion
  of reading a file path directly in a `FROM` clause; register a
  `DataFrame` (built however, `session.read.csv(...)`, `session.read.
  parquet(...)`, `session.create_dataframe(...)`) under a name first.
* `WHERE <expr>`: comparisons, `AND`/`OR`/`NOT`, `IS [NOT] NULL`,
  arithmetic, parenthesized sub-expressions.
* `[INNER] JOIN <table> ON <a> = <b>`: exactly one join, inner only,
  and `<a>`/`<b>` must be the *same* bare column name on both sides
  (`users.country = regions.country` works; `users.id = orders.
  user_id` does not), matching `logical/nodes.py`'s `Join` docstring
  exactly. A qualified name's table prefix (`users.country`) is
  accepted for readability and then discarded: MiniSpark's expression
  tree resolves a `Column` by bare name only, the same as `col("x")`
  built through the DataFrame API, so `users.country` and `regions.
  country` both just become `Column("country")`.
* `GROUP BY <expr, ...>` and aggregate function calls `COUNT`/`SUM`/
  `AVG`/`MIN`/`MAX` (`COUNT(*)` and `COUNT(col)` both work, matching
  `api/functions.py`'s `count()`), including a global aggregate with no
  `GROUP BY` at all (`SELECT COUNT(*) FROM users`). Every non-aggregated
  `SELECT` list item must be a `GROUP BY` column, exactly standard SQL's
  own rule, checked directly by the parser (`SqlParseError`, not a
  silently wrong query) rather than left for the analyzer to catch
  later less clearly.
* `HAVING <expr>`: applied after the `Aggregate`. An aggregate function
  call written directly in `HAVING` (`HAVING COUNT(*) >= 1`) is resolved
  against the matching `SELECT`-list aggregate's output column (by
  structural match, not by literally re-embedding the raw
  `AggregateFunction` expression, which has no per-row value and would
  raise if evaluated directly, see `expressions/aggregate.py`); an
  aggregate not present in the `SELECT` list raises `SqlParseError`
  rather than silently adding a second, hidden aggregate the way some
  SQL engines do.
* `ORDER BY <expr> [ASC|DESC], ...`: multiple keys, mixed direction,
  matching `.order_by()` exactly.

**Deliberately not supported**, because the DataFrame API cannot express
it either, and SQL support is a front-end, not a new capability:
subqueries, `UNION`, window functions, `LIMIT`, common table
expressions (`WITH`), user-defined functions, and any join type other
than inner (`LEFT`/`RIGHT`/`FULL OUTER`, semi/anti). Adding any of these
to SQL without first adding the underlying `LogicalPlan`/execution
support would violate "no separate SQL execution engine" as much as
writing a second interpreter would.

## `parse_sql()`'s table-resolution seam

`parse_sql(query: str, tables: dict[str, LogicalPlan])` takes a plain
`dict`, not a `MiniSparkSession`: `sql/parser.py` never imports from
`api/`, and never touches session state. `MiniSparkSession.sql()`
(`api/session.py`) is what resolves table names against its own
`_temp_views` registry (`{name: df.plan for name, df in self.
_temp_views.items()}`) before calling `parse_sql()`. This keeps the
parser fully testable with a plain dict of hand-built `LogicalPlan`s
(`tests/unit/test_sql_parser.py`), independent of a real session, and
avoids a dependency cycle: `api/session.py` importing from `sql/` is
fine (a "user API" layer depending on a lower one), the reverse would
not be.

`create_or_replace_temp_view(name, df)` is a plain, session-scoped, in-
memory `dict[str, DataFrame]`, not a real catalog: nothing persists, and
registering the same name twice silently replaces the first (matching
PySpark's own `createOrReplaceTempView` naming and behavior exactly).
`df` itself is never executed by registering it; only its still-fully-
lazy `.plan` is kept, read only once `sql()` actually parses a query
that references `name`.

## What this is not

No query string caching or prepared statements (`sql()` tokenizes and
parses the string fresh every call; the resulting `LogicalPlan` is
already cheap enough to build that memoizing this was not worth the
added state). No `EXPLAIN <sql>` shorthand: get a `DataFrame` from
`session.sql(...)` first, then call `.explain(optimized=True)` on it,
exactly like any other `DataFrame`. No SQL-level error recovery or
partial-query suggestions: a malformed query raises `SqlParseError`
(or its parent, `SqlLexError`'s sibling `SqlSyntaxError`, for a
tokenizer-level problem) naming the offending token and its position,
and that is the whole error-reporting story.
