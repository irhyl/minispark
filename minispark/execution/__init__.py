"""Execution.

`executor.py` (Milestone 1) is a naive, single-process, tree-walking
interpreter of a *logical* plan. It is not used by `DataFrame` as of
Milestone 2 (superseded by the analyzer/optimizer/physical plan) and is
kept only as the correctness oracle other execution paths are tested
against.

Milestone 3 adds the rest of this package: `dag.py` (narrow/wide
dependency classification), `stages.py` (splits a physical plan into
stages at shuffle boundaries; today there is exactly one stage per query,
no physical node is wide yet), `tasks.py` (`Task`/`TaskContext`/
`TaskResult`/`TaskMetrics`), `worker.py` (`execute_task`, what actually
runs a Task's physical operators for one partition), and `scheduler.py`
(`LocalScheduler`, which turns a Stage into Tasks, runs them either
sequentially or across real OS processes depending on `local[N]`, retries
failures, and merges results back into a Dataset). `DataFrame` actions
route through this scheduler as of Milestone 3.
"""
