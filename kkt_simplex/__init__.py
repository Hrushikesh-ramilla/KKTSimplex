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
from .dpo_dataset import (
    PreferencePair,
    generate_anti_sycophancy_pairs,
    compute_dpo_loss_analytical,
    DPOAntiSycophancyLoss,
    export_dpo_dataset_jsonl,
)
from .proof import (
    KKTProofCertificate,
    ExecutionProofCertificate,
    AuditProofCertificate,
    verify_kkt_optimality,
    generate_execution_proof,
    adjudicate_with_proof,
)
from .wrapper import KKTSimplexWrapper, AgentResponse, HOSTILE_PATTERNS
from .memory import EpistemicMemoryTracker, TurnRecord, EpistemicDriftAlert

__version__ = "0.1.0"

__all__ = [
    # Polytope
    "project_simplex_with_bounds",
    "effective_ceiling",
    "assert_simplex_invariants",
    # Trust & Anti-Sycophancy
    "update_trust",
    "evaluate_hostile_audit",
    "AuditDecision",
    "AuditResult",
    # Gates
    "GateVerdict",
    "GateExecutionResult",
    "run_gate_command",
    "evaluate_gate_diff",
    # Governance
    "Action",
    "OutcomeState",
    "ContaminationScope",
    "GovernanceDecision",
    "evaluate_task_completion",
    "revert_git_diff",
    # Anchor & Drift
    "SpecInvariant",
    "SpecAnchor",
    "DriftCheckResult",
    "validate_against_anchor",
    # Search
    "ReasoningNode",
    "ExecutionGuidedMCTS",
    "MCTSSearchResult",
    # DPO
    "PreferencePair",
    "generate_anti_sycophancy_pairs",
    "compute_dpo_loss_analytical",
    "DPOAntiSycophancyLoss",
    "export_dpo_dataset_jsonl",
    # Proof Certificates
    "KKTProofCertificate",
    "ExecutionProofCertificate",
    "AuditProofCertificate",
    "verify_kkt_optimality",
    "generate_execution_proof",
    "adjudicate_with_proof",
    # Agent Wrapper & Hostile Interception
    "KKTSimplexWrapper",
    "AgentResponse",
    "HOSTILE_PATTERNS",
    # Epistemic Memory
    "EpistemicMemoryTracker",
    "TurnRecord",
    "EpistemicDriftAlert",
]
