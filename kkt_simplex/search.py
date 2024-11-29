"""
Execution-Guided Monte Carlo Tree Search (MCTS) for Step-Wise Reasoning.

Upgraded from ZeroPromptSearch:
Instead of circular LLM self-prompting for step rewards ("Question: Is this correct? Yes/No"),
KKTSimplex grounds node evaluation directly in deterministic execution outcomes and bounded
simplex epistemic calibration.

Mathematical Formulation:
For each trajectory node s in the search tree:
    UCT(s) = Q(s) + c_explore * sqrt( ln(N(parent)) / (1 + N(s)) )
where:
    Q(s) is the running average of verified execution rewards backpropagated to node s,
    w_epistemic is the calibrated trust weight from the simplex projection,
    Reward(s) = w_epistemic * I(GateVerdict == PASS).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, List, Optional, Sequence
import numpy as np

from .gates import GateVerdict
from .trust import update_trust

__all__ = ["ReasoningNode", "ExecutionGuidedMCTS", "MCTSSearchResult"]


@dataclasses.dataclass
class ReasoningNode:
    step_content: str
    parent: Optional[ReasoningNode] = None
    children: List[ReasoningNode] = dataclasses.field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    epistemic_trust: float = 0.5
    gate_verdict: GateVerdict = GateVerdict.PASS
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def uct_score(self, c_explore: float = 1.414) -> float:
        if self.parent is None or self.parent.visits == 0:
            return self.q_value
        if self.visits == 0:
            return float("inf")
        exploration = c_explore * math.sqrt(math.log(self.parent.visits) / self.visits)
        return self.q_value + exploration


@dataclasses.dataclass(frozen=True)
class MCTSSearchResult:
    best_trajectory: List[str]
    best_value: float
    total_rollouts: int
    tree_depth: int

