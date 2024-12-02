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


class ExecutionGuidedMCTS:
    """
    Monte Carlo Tree Search over step-wise reasoning trajectories, guided by
    deterministic execution gates and calibrated epistemic trust.
    """

    def __init__(
        self,
        step_generator: Callable[[List[str]], List[str]],
        execution_verifier: Callable[[List[str]], tuple[GateVerdict, float]],
        c_explore: float = 1.414,
        max_depth: int = 5,
        num_simulations: int = 15,
    ):
        self.step_generator = step_generator
        self.execution_verifier = execution_verifier
        self.c_explore = c_explore
        self.max_depth = max_depth
        self.num_simulations = num_simulations

    def search(self, initial_problem: str, num_simulations: int | None = None) -> MCTSSearchResult:
        sims = num_simulations if num_simulations is not None else self.num_simulations
        root = ReasoningNode(step_content=initial_problem)

        for _ in range(sims):
            # 1. Selection
            node = root
            trajectory = [node.step_content]

            while node.children and len(trajectory) < self.max_depth:
                # Select child with highest UCT score
                node = max(node.children, key=lambda c: c.uct_score(self.c_explore))
                trajectory.append(node.step_content)

            # 2. Expansion
            if not node.is_terminal and len(trajectory) < self.max_depth:
                candidate_steps = self.step_generator(trajectory)
                for step in candidate_steps:
                    child = ReasoningNode(step_content=step, parent=node)
                    node.children.append(child)

                if node.children:
                    node = node.children[0]
                    trajectory.append(node.step_content)
                else:
                    # No children generated; mark node terminal to prevent infinite spin
                    node.is_terminal = True

            # 3. Deterministic Evaluation
            gate_verdict, raw_confidence = self.execution_verifier(trajectory)

            # Epistemic calibration: Scale Bayesian evidence by execution confidence
            conf = float(raw_confidence) if (isinstance(raw_confidence, (int, float)) and math.isfinite(raw_confidence) and raw_confidence > 0) else 1.0
            if gate_verdict == GateVerdict.PASS:
                support = np.array([conf, 0.0])
                harm = np.array([0.0, conf])
            else:
                support = np.array([0.0, conf])
                harm = np.array([conf, 0.0])

            weights = update_trust(
                scores=[0.5, 0.5],
                support=support,
                harm=harm,
                n_claims=[1.0, 1.0],
                floor=0.1,
                ceiling=0.9,
            )
            calibrated_trust = float(weights[0])
            node.epistemic_trust = calibrated_trust
            node.gate_verdict = gate_verdict

            # Immediate reward: calibrated trust if passed, 0 if regression
            reward = calibrated_trust if gate_verdict == GateVerdict.PASS else 0.0

            # 4. Backpropagation
            curr: Optional[ReasoningNode] = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += reward
                curr = curr.parent

        # Find best terminal or deepest trajectory that passed execution gates
        best_path: List[str] = []
        curr = root
        while curr.children:
            # Consider children that passed verification and have positive value
            passing_children = [c for c in curr.children if c.gate_verdict == GateVerdict.PASS and c.visits > 0]
            if passing_children:
                best_child = max(passing_children, key=lambda c: c.q_value)
                best_path.append(best_child.step_content)
                curr = best_child
            else:
                # If even root has no passing children, trajectory is empty
                if curr is root:
                    best_path = []
                    curr = None
                break

        return MCTSSearchResult(
            best_trajectory=best_path,
            best_value=curr.q_value if curr else 0.0,
            total_rollouts=sims,
            tree_depth=len(best_path),
        )
