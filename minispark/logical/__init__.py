"""The logical plan: a relational tree describing *what* to compute, not *how*.

`df.filter(...).select(...)` builds this tree; no data moves until an
action (`collect()`, `show()`, `count()`) triggers execution.
"""
