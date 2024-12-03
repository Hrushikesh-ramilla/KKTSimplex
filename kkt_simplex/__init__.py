"""
KKTSimplex: Verifiable Epistemic Calibration & Autonomous Execution Governance for Agentic AI.
"""

from .polytope import project_simplex_with_bounds, effective_ceiling, assert_simplex_invariants
from .trust import update_trust, evaluate_hostile_audit, AuditDecision, AuditResult
from .gates import GateVerdict, GateExecutionResult, run_gate_command, evaluate_gate_diff
from .governance import (
    Action,
    OutcomeState,
    ContaminationScope,
    GovernanceDecision,
    evaluate_task_completion,
    revert_git_diff,
)
from .anchor import SpecInvariant, SpecAnchor, DriftCheckResult, validate_against_anchor
from .search import ReasoningNode, ExecutionGuidedMCTS, MCTSSearchResult

__version__ = "0.1.0"
