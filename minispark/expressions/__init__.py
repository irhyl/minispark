"""The expression tree used inside logical plans (Filter conditions, Project columns).

`col("age") > 18` builds an Expression tree; it does not touch any data.
Evaluation happens later, per-record, when the executor runs a Filter or
Project node.
"""
