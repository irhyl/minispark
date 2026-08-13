"""Optimizer: applies a list of rules to a logical plan until it stops changing.

Fixed-point detection compares plans by their explain() text rather than by
`==`. `Expression.__eq__` is overloaded to build an `Equal` expression node
(see expressions/base.py), not to return a bool, so `plan_a == plan_b` is
meaningless for logical plans that hold expressions. Rendering both sides
with `explain_string()` and comparing the strings sidesteps that without
adding a separate structural-equality method that would only ever be used
here.
"""

from __future__ import annotations

from minispark.config.config import OptimizerConfig
from minispark.logical.nodes import LogicalPlan
from minispark.logical.plan import explain_string
from minispark.optimizer.rules import (
    ConstantFolding,
    FilterSimplification,
    PredicatePushdown,
    ProjectionPruning,
    RedundantProjectionElimination,
    Rule,
)

_DEFAULT_MAX_ITERATIONS = 10


def default_rules(config: OptimizerConfig | None = None) -> list[Rule]:
    """The rule set `Optimizer` runs by default, gated by `OptimizerConfig`.

    ConstantFolding, FilterSimplification, and RedundantProjectionElimination
    have no corresponding config flag (the build spec's OptimizerConfig only
    defines `predicate_pushdown` and `projection_pruning`) and always run:
    they are correctness-preserving cleanups with no plausible reason to
    disable them. PredicatePushdown and ProjectionPruning are individually
    toggleable because they are the two the config already had a field for.
    """
    config = config or OptimizerConfig()
    rules: list[Rule] = [ConstantFolding(), FilterSimplification()]
    if config.predicate_pushdown:
        rules.append(PredicatePushdown())
    if config.projection_pruning:
        rules.append(ProjectionPruning())
    rules.append(RedundantProjectionElimination())
    return rules


class Optimizer:
    def __init__(
        self,
        rules: list[Rule] | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ):
        self.rules = rules if rules is not None else default_rules()
        self.max_iterations = max_iterations

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        current = plan
        for _ in range(self.max_iterations):
            before = explain_string(current)
            for rule in self.rules:
                current = rule.apply(current)
            if explain_string(current) == before:
                break
        return current
