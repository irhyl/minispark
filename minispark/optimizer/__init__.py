"""The rule-based optimizer: rewrites an analyzed logical plan into an
equivalent, cheaper-to-execute one.

Milestone 2 scope: constant folding, filter simplification, predicate
pushdown, projection pruning, and redundant projection elimination. All five
operate purely on plan/expression shape (no statistics are consulted yet);
`optimizer/statistics.py` exists as infrastructure for later, cost-based
decisions (join strategy selection in Milestone 5), not as an input to any
rule here.
"""
