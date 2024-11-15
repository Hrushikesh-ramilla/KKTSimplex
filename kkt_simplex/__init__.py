"""
KKTSimplex: Verifiable Epistemic Calibration & Autonomous Execution Governance.
"""

from .polytope import project_simplex_with_bounds, effective_ceiling, assert_simplex_invariants
from .gates import GateVerdict, GateExecutionResult, run_gate_command, evaluate_gate_diff
from .governance import (
    Action, OutcomeState, ContaminationScope, GovernanceDecision,
    evaluate_task_completion, revert_git_diff,
)

__version__ = "0.1.0"
