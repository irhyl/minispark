# Query Planning

How a `DataFrame` call becomes rows, as of Milestone 8. As of this
milestone there are two ways to build the logical plan this document
covers, `DataFrame` API chaining, and `session.sql(...)` (`sql/parser.py`,
see `docs/sql.md`), which parses SQL text into exactly the same logical
plan nodes rather than getting its own execution path, per the build
spec's "there must not be a separate SQL execution engine" rule.
Everything below this point (analysis, optimization, physical planning,
scan pushdown, stage splitting, scheduled execution) applies identically
regardless of which path built the plan.

```
DataFrame API (filter/select/group_by/join/order_by)  session.sql("SELECT ...")
                        \                                    /
                         \                                  /
Logical Plan          (logical/nodes.py: Scan, Filter, Project, Aggregate, Join, Sort)
        |
Analyzer               (logical/analyzer.py: analyze())
        |
Optimizer              (optimizer/optimizer.py: Optimizer.optimize())
        |
Physical Plan          (physical/planner.py: plan_physical())
        |
Stage Splitting        (execution/stages.py: build_stages())
        |
Scheduled Execution    (execution/scheduler.py: LocalScheduler.run_plan())
        |
Dataset (rows)
```

## Building the logical plan

`df.filter(cond)`, `df.select(*cols)`, `df.group_by(*cols)`/`.agg(...)`,
`df.join(other, on=...)`, and `df.order_by(*cols)` each wrap the current
plan in one new node (`Filter`, `Project`, `Aggregate`, `Join`, `Sort`) and
return a new `DataFrame`. No data moves and no validation happens here:
`df.select("does_not_exist")` builds a plan successfully, it just fails
later. This is what makes the API lazy.

## Analysis

Only an action (`collect`/`show`/`count`/`explain(optimized=True)`) calls
`analyze()`. It walks the plan top-down and, per node type: for `Filter`
and `Aggregate`'s per-group expressions, checks that every referenced
`Column` exists in the relevant child schema; for `Project`/`Aggregate`,
rejects duplicate output names; for `Aggregate`, additionally rejects a
`group_by` on anything but a plain column and a non-aggregate expression
in `agg()`; for `Join`, rejects an unsupported `how` (only `"inner"` is
implemented), an empty `on=`, an `on` column missing from either side, or
any other column name that collides between the two sides; for `Sort`,
rejects an `order_by()` argument that is not a plain column. The first
problem found raises `AnalysisException` with a clear message. Analysis
does not rewrite the plan; it only decides whether to raise.

What analysis does not check yet: result types of computed expressions
(there is no type-inference engine), and anything about join types other
than inner (left/right/full outer, semi/anti are not implemented, see
`logical/nodes.py`'s `Join` docstring).

## Optimization

`Optimizer.optimize()` runs a fixed list of rules over the analyzed plan,
repeating the whole list until a pass produces no change (compared by
`explain_string()` text, capped at 10 iterations: see
`docs/architecture.md`'s Milestone-2 design decisions for why text
comparison instead of `==`). The default rules, in order:

1. **ConstantFolding**: evaluates sub-expressions made entirely of
   literals once, at plan time, instead of once per row.
2. **FilterSimplification**: simplifies boolean expressions like
   `x AND True` to `x`, or `Not(Not(x))` to `x`.
3. **PredicatePushdown** *(config: `optimizer.predicate_pushdown`)*: moves
   a `Filter` below a `Project` when the filter does not need a column the
   Project computed or renamed, so rows are dropped before being projected
   instead of after; also moves a `Filter` below a `Join` into whichever
   side it exclusively references (valid only because `Join.how` is
   always `"inner"`, see `optimizer/rules.py`'s `PredicatePushdown`
   docstring).
4. **ProjectionPruning** *(config: `optimizer.projection_pruning`)*:
   inserts a column-only `Project` directly above `Scan`, keeping only
   what is referenced anywhere above it; `Aggregate` and `Join` fully
   determine what they need from their child(ren) (like `Project` does),
   so `needed` is replaced, not merged, at those nodes; `Sort` behaves
   like `Filter` (adds its sort keys to whatever was already needed).
   Through Milestone 6 this only narrowed plan *shape* (smaller Record
   dicts flowing through Filter/Project below `Scan`), not bytes read
   from disk; Milestone 7's physical-planning-time scan-pushdown pass
   (below) is what turns this rule's already-computed column set into an
   actual, narrower read for a source that can honor it.
5. **RedundantProjectionElimination**: removes a `Project` that turned
   out to be a no-op (its columns exactly match its child's schema), and
   collapses a plain-column `Project` sitting directly below another one.

Every rule has a case for every logical node type (`Scan`/`Filter`/
`Project`/`Aggregate`/`Join`/`Sort`); adding one is what made
`optimizer/rules.py` immediately break the first time an `Aggregate`
appeared in a plan (the tree walkers raised `NotImplementedError` on a
node type they had no case for), a good early signal that the walker
functions need a case added for every new logical node, not just the
node's own analyzer/schema code. Each rule is still a pure function
(`LogicalPlan -> LogicalPlan`); none of them consult
`optimizer/statistics.py`, there is no cost-based decision to make yet
(broadcast-vs-shuffle join selection is an explicit hint, not automatic,
see `logical/nodes.py`'s `Join` docstring). See
`tests/unit/test_optimizer_rules.py` for one before/after test per rule,
and `tests/unit/test_optimizer.py` for end-to-end fixed-point and
correctness-preservation tests.

## Physical planning and execution

`plan_physical()` translates the optimized logical plan into a
`PhysicalPlan` tree. `Scan`/`Filter`/`Project` translate 1:1, one physical
node per logical node, because each has exactly one execution strategy.
`Aggregate`, `Join`, and `Sort` do not: see `docs/shuffle.md` for
`Aggregate`'s partial-aggregate/exchange/final-aggregate shape and
`Join`'s shuffle-hash-join/broadcast-join shape, and the note below for
`Sort`'s.

`physical/operators.py`'s `execute()` (whole-Dataset, single-process) is
what Milestone 1-2 used directly; as of Milestone 3 it is an oracle only,
used in tests to check that later, more complex execution paths agree
with it. The real path is `execute_partition()` (one partition at a time)
run by `execution/scheduler.py`'s `LocalScheduler`, across a real
`ProcessPoolExecutor` when `local[N]` has `N > 1`: see
`docs/execution-model.md` for the DAG/Stage/Task/Worker picture that sits
between physical planning and rows.

**`Sort` and scan pushdown are the two places physical planning touches
data.** Every other node above only rearranges plan *shape*. Choosing
where to cut `order_by`'s sort key into `shuffle_partitions` ranges needs
to know something about the data's range *before* the shuffle that needs
those cut points runs, and there is no distributed sampling stage to
compute that without touching data (see `docs/shuffle.md`). `physical/
planner.py`'s `_sort_range_boundaries()` gets it by eagerly executing the
child plan (via `physical/operators.py`'s whole-Dataset `execute()`, so
only a Scan/Filter/Project child chain is supported, not one ending in
`Aggregate` or `Join`) and scanning the sort key's min/max with
`optimizer/statistics.py`'s `compute_statistics()`, right there in the
planner.

The second, added in Milestone 7: `_pushdown_scan_reads()` runs once,
before any node translation, and re-reads any `Scan` whose `source` is
set (see `logical/nodes.py`'s `ScanSource` Protocol) with the columns/
filter its surrounding Project/Filter chain implies, so a source that
can honor them (Parquet: real column pruning and row-group-level
predicate pushdown; CSV/Memory/Checkpoint: real column pruning only)
actually reads less. See `docs/columnar-storage.md` for exactly how the
hints are computed and propagated, and for the "run once, not once per
recursive call" fix a first version of this needed. Both of these are
flagged loudly, in both code and here, because each is a real, deliberate
exception to "building a plan never touches data," true of literally
everything else in this file.

## `explain()`

`df.explain()` (the default) prints only `"== Logical Plan =="`, the raw,
unanalyzed plan, matching Milestone 1's behavior exactly.
`df.explain(optimized=True)` prints four sections: `"== Analyzed Logical
Plan =="`, `"== Optimized Logical Plan =="`, `"== Physical Plan =="`, and
`"== Stages =="` (one subsection per stage, from
`execution/stages.py`'s `build_stages()`), so the effect of each step,
including how many stages a shuffle-heavy query actually splits into, is
visible. See `examples/basic_dataframe.py` for a Milestone 1/2 example
(constant folding, pushdown, and pruning all fire on one small query),
`examples/aggregations.py` / `examples/joins.py` for Milestone 4/5
examples that produce real multi-stage, multi-shuffle plans, and
`examples/parquet.py` for a Milestone 7 example where the "Physical
Plan" section's `ScanExec` visibly reads fewer columns than the source
file has, the scan-pushdown pass's effect made directly visible (the
"Optimized Logical Plan" section above it does not show this: pushdown
happens at physical-planning time, one step later, see this document's
"Physical planning and execution" section), and `examples/sql.py` for a
Milestone 8 example proving a SQL-built `DataFrame` explains exactly
like any other, join, group by, having, and order by all included in
one query. `explain()` never executes anything, before or after
Milestone 8: `DataFrame.last_run_metrics` (see docs/execution-model.md's
"Metrics and profiling") is the separate, only-after-an-action-actually-
runs mechanism for seeing what a query *did*, not what it *would* do.
