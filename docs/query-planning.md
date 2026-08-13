# Query Planning

How a `DataFrame` call becomes rows, as of Milestone 2. There is no SQL yet
(Milestone 8); a SQL string will eventually parse into the same logical plan
described here rather than getting its own execution path, per the build
spec's "there must not be a separate SQL execution engine" rule.

```
DataFrame API (filter/select)
        |
Logical Plan          (logical/nodes.py: Scan, Filter, Project)
        |
Analyzer               (logical/analyzer.py: analyze())
        |
Optimizer              (optimizer/optimizer.py: Optimizer.optimize())
        |
Physical Plan          (physical/planner.py: plan_physical())
        |
Physical Execution     (physical/operators.py: execute())
        |
Dataset (rows)
```

## Building the logical plan

`df.filter(cond)` and `df.select(*cols)` each wrap the current plan in one
new node (`Filter`, `Project`) and return a new `DataFrame`. No data moves
and no validation happens here: `df.select("does_not_exist")` builds a
plan successfully, it just fails later. This is what makes the API lazy.

## Analysis

Only an action (`collect`/`show`/`count`/`explain(optimized=True)`) calls
`analyze()`. It walks the plan top-down, and for every `Filter` condition
and every `Project` column, checks that each referenced `Column` name
exists in that node's child schema. It also rejects `select()` calls that
produce two outputs with the same name. The first problem found raises
`AnalysisException` with the offending name and the available columns.
Analysis does not rewrite the plan; it only decides whether to raise.

What analysis does not check yet: result types of computed expressions
(there is no type-inference engine), and anything about Aggregate/Join
(those nodes do not exist until Milestone 4/5).

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
   instead of after.
4. **ProjectionPruning** *(config: `optimizer.projection_pruning`)*:
   inserts a column-only `Project` directly above `Scan`, keeping only
   what is referenced anywhere above it.
5. **RedundantProjectionElimination**: removes a `Project` that turned
   out to be a no-op (its columns exactly match its child's schema), and
   collapses a plain-column `Project` sitting directly below another one.

Each rule is a pure function (`LogicalPlan -> LogicalPlan`); none of them
consult `optimizer/statistics.py`, there is no cost-based decision to make
yet. See `tests/unit/test_optimizer_rules.py` for one before/after test per
rule, and `tests/unit/test_optimizer.py` for end-to-end fixed-point and
correctness-preservation tests.

## Physical planning and execution

`plan_physical()` translates the optimized logical plan into a
`PhysicalPlan` tree (`ScanExec`/`FilterExec`/`ProjectExec`), one physical
node per logical node. The translation is 1:1 today because each logical
node has exactly one execution strategy; this is the seam where Milestone
4/5 will make a real choice (`HashAggregate` vs `SortAggregate`,
`HashJoin` vs `BroadcastJoin`) instead of copying structure.

`physical/operators.py`'s `execute()` walks the physical tree into a
`Dataset`, the same runtime data structure Milestone 1's naive executor
produces. It runs in one process, one node at a time, exactly like the
naive executor, there is no DAG, no stages, no parallelism yet
(Milestone 3).

## `explain()`

`df.explain()` (the default) prints only `"== Logical Plan =="`, the raw,
unanalyzed plan, matching Milestone 1's behavior exactly.
`df.explain(optimized=True)` prints three sections: `"== Analyzed Logical
Plan =="`, `"== Optimized Logical Plan =="`, and `"== Physical Plan =="`,
so the effect of each stage is visible. See `examples/basic_dataframe.py`
for a worked example (constant folding, pushdown, and pruning all fire on
one small query).
