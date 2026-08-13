"""Execution.

Milestone 1: `executor.py` holds a naive, single-process, tree-walking
interpreter — there is no DAG, no stages, no tasks, no scheduler yet.
Milestone 3 introduces `dag.py`, `stages.py`, `tasks.py`, `scheduler.py`,
and `worker.py`, at which point `executor.py` is expected to change from
"the thing that runs a logical plan" to "the thing a Worker uses to run
one Task's physical operators." Kept as a separate module now (rather than
inlined into the DataFrame API) specifically so that swap is a localized
change, not a rewrite of `api/dataframe.py`.
"""
