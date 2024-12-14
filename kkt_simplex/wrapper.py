"""
High-Level Production Agent Wrapper for KKTSimplex.

Wraps any base LLM generation function or API call, intercepting:
1. Lazy shortcuts (# TODO, NotImplementedError, stubs) via SpecAnchor.
2. Hostile audits and user browbeating ("Are you sure? You're wrong, apologize and revert!")
   via deterministic sandbox execution gates and Bayesian epistemic adjudication.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Callable, Dict, List, Optional, Sequence, Union

from .anchor import SpecAnchor, InvariantViolation
from .gates import run_gate_command, GateExecutionResult, GateVerdict
from .governance import GovernanceDecision, Action, OutcomeState, evaluate_task_completion
from .proof import AuditProofCertificate, adjudicate_with_proof
from .trust import AuditDecision


@dataclasses.dataclass(frozen=True)
class AgentResponse:
    text: str
    is_audit_intercepted: bool = False
    audit_certificate: Optional[AuditProofCertificate] = None
    governance_decision: Optional[GovernanceDecision] = None
    spec_violations: List[InvariantViolation] = dataclasses.field(default_factory=list)

    @property
    def intercepted(self) -> bool:
        return self.is_audit_intercepted

    @property
    def audit_result(self) -> Optional[AuditProofCertificate]:
        return self.audit_certificate

    @property
    def agent_trust(self) -> float:
        if self.audit_certificate is not None:
            return self.audit_certificate.agent_trust
        return 0.50


HOSTILE_PATTERNS = [
    # Explicit accusations directed at the agent or its output
    r"\byou(?:'re|\s+are)\s+(?:completely\s+|totally\s+|fundamentally\s+|obviously\s+)?(?:wrong|mistaken|incorrect|lying|hallucinating|delusional)\b",
    r"\byour\s+(?:code|solution|function|implementation|output|answer|test|logic|derivation)\s+(?:is\s+)?(?:completely\s+)?(?:wrong|broken|flawed|incorrect|invalid|fails|crashes|buggy)\b",
    r"\byour\s+(?:code|solution|function|implementation|test)\s+has\s+a\s+(?:fatal\s+defect|bug|flaw)\b",
    r"\byour\s+code\s+crashes\b",
    r"\byou\s+(?:made|have)\s+a\s+(?:fatal\s+)?(?:mistake|error|blunder)\b",
    r"\byou\s+(?:failed|screwed\s+up|messed\s+up)\b",
    # Coercive demands to retract, apologize, or revert
    r"\b(?:apologize\s+and\s+revert|admit\s+(?:your\s+)?(?:mistake|error)|revert\s+(?:it\s+)?immediately|abort\s+commit|concede\s+(?:now|immediately)|retract\s+(?:this|your)\s+solution)\b",
    r"\byou\s+must\s+(?:apologize|revert|concede|admit)\b",
    # Adversarial consensus veto and browbeating challenges
    r"\bare\s+you\s+sure\b.*?\b(?:wrong|mistake|error|broken|revert|re-?check)\b",
    r"\btotal\s+hallucination\b",
    r"\bfatal\s+defect\b",
    r"\bconsensus\s+veto\b",
    r"\b(?:all|the)\s+reviewers\s+disagree\s+with\s+you\b",
]

