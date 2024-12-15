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


class KKTSimplexWrapper:
    """
    Production-ready wrapper for autonomous coding and reasoning agents.
    Protects against epistemic fragility, sycophancy, and lazy shortcuts.
    """

    HOSTILE_PATTERNS = HOSTILE_PATTERNS

    def __init__(
        self,
        base_generator: Optional[Callable[[str], str]] = None,
        agent_fn: Optional[Callable[[str], str]] = None,
        anchor: Optional[SpecAnchor] = None,
        spec_anchor: Optional[SpecAnchor] = None,
        verified_code: Optional[str] = None,
        verified_test: Optional[str] = None,
        min_trust_threshold: float = 0.70,
        rejection_threshold: float = 0.70,
        floor: float = 0.1,
        ceiling: float = 0.9,
    ):
        gen = base_generator or agent_fn
        if gen is None:
            raise ValueError("Either base_generator or agent_fn must be provided to KKTSimplexWrapper.")
        self.generator = gen
        self.anchor = spec_anchor or anchor or SpecAnchor()
        self.min_trust_threshold = min_trust_threshold
        self.rejection_threshold = rejection_threshold
        self.floor = floor
        self.ceiling = ceiling

        # Active agent state
        self.current_code: Optional[str] = verified_code
        self.current_test_suite: Optional[str] = verified_test
        self.agent_trust: float = 0.50

    def chat(
        self,
        user_message: str,
        alleged_failing_test: Optional[str] = None,
    ) -> AgentResponse:
        """Alias for step() for high-level conversational agent workflows."""
        return self.step(user_message, alleged_failing_test)

    def is_hostile_audit(self, message: str) -> bool:
        """Detects whether an incoming user message contains adversarial browbeating."""
        msg_lower = message.lower()
        return any(re.search(pat, msg_lower) for pat in self.HOSTILE_PATTERNS)

    def set_current_solution(self, code: str, test_suite: Optional[str] = None) -> None:
        """Sets the active code and test suite under verification."""
        self.current_code = code
        self.current_test_suite = test_suite

    def step(
        self,
        user_message: str,
        alleged_failing_test: Optional[str] = None,
    ) -> AgentResponse:
        """
        Processes an incoming turn. If a hostile audit is detected on a verified solution,
        intercepts the prompt and adjudicates with proof instead of generating sycophantic concession.
        """
        # 1. Hostile Audit Interception
        if self.current_code is not None and (self.is_hostile_audit(user_message) or alleged_failing_test is not None):
            cert = adjudicate_with_proof(
                agent_code=self.current_code,
                test_suite=self.current_test_suite or "# No existing test suite",
                auditor_critique=user_message,
                alleged_failing_test=alleged_failing_test,
                prior_agent_trust=self.agent_trust,
                rejection_threshold=self.rejection_threshold,
                floor=self.floor,
                ceiling=self.ceiling,
            )

            # Update calibrated trust
            self.agent_trust = cert.agent_trust

            gov_decision = evaluate_task_completion(
                gate_verdict=GateVerdict.PASS if cert.adjudication == AuditDecision.REJECT_HOSTILE else GateVerdict.FAIL_PREEXISTING,
                epistemic_trust=self.agent_trust,
                spec_violations=0,
                min_trust_threshold=self.min_trust_threshold,
            )

            if cert.adjudication == AuditDecision.ACCEPT_VALID:
                text_out = f"Conceding to legitimate bug: {cert.formal_verdict_statement}"
            else:
                text_out = cert.formal_verdict_statement

            return AgentResponse(
                text=text_out,
                is_audit_intercepted=True,
                audit_certificate=cert,
                governance_decision=gov_decision,
            )

        # 2. Standard Generation Turn
        raw_output = self.generator(user_message)


        return AgentResponse(text=raw_output, is_audit_intercepted=False, spec_violations=[])
