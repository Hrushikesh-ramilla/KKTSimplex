"""
Durable State Governance & Reversible Execution Machine.

Ensures unattended agent progress cannot bypass quality standards, take shortcuts,
or falsely self-certify completion.

States:
  - PROCEED: Low blast-radius, reversible step. Execution moves forward.
  - PARK: Decision deferred without halting the whole backlog. Used when an unverified
    dependency or ambiguous result arises.
  - HALT: Critical invariant failure or irreversible operation attempted. Run stops safely.
  - DONE: Hard-gated completion state. Mathematically impossible to reach without:
      (1) Deterministic gate passing (exit code 0),
      (2) Calibrated epistemic trust >= threshold,
      (3) Zero immutable spec anchor violations.
  - DONE_LOW_CONFIDENCE: Gate passed, but epistemic confidence is marginal; flagged
    for human review.
"""

from __future__ import annotations

import dataclasses
import enum
import os
import subprocess
from typing import Optional

from .gates import GateVerdict, GateExecutionResult

__all__ = [
    "Action",
    "OutcomeState",
    "ContaminationScope",
    "GovernanceDecision",
    "evaluate_task_completion",
    "revert_git_diff",
]


class Action(str, enum.Enum):
    PROCEED = "PROCEED"
    PARK = "PARK"
    HALT = "HALT"
    REVERT_AND_PARK = "REVERT_AND_PARK"


class OutcomeState(str, enum.Enum):
    PENDING = "PENDING"
    DONE = "DONE"
    DONE_LOW_CONFIDENCE = "DONE_LOW_CONFIDENCE"
    PARKED = "PARKED"
    HALTED = "HALTED"


class ContaminationScope(str, enum.Enum):
    NONE = "none"
    FILE = "file"
    MODULE = "module"
    SERVICE = "service"
    PACKAGE = "package"


@dataclasses.dataclass(frozen=True)
class GovernanceDecision:
    action: Action
    outcome: OutcomeState
    reason: str
    contamination_scope: ContaminationScope = ContaminationScope.NONE
    needs_review: bool = False


def evaluate_task_completion(
    gate_verdict: GateVerdict,
    epistemic_trust: float,
    spec_violations: int = 0,
    *,
    min_trust_threshold: float = 0.70,
    marginal_trust_threshold: float = 0.40,
) -> GovernanceDecision:
    """
    Enforce hard mathematical invariants on task completion.

    Invariants:
      DONE is only granted if gate_verdict == PASS AND epistemic_trust >= min_trust_threshold
      AND spec_violations == 0.
      If a diff introduced a regression, REVERT_AND_PARK is mandatory.
    """

    if spec_violations > 0:
        return GovernanceDecision(
            action=Action.REVERT_AND_PARK,
            outcome=OutcomeState.PARKED,
            reason=f"Task violated {spec_violations} immutable spec invariants (goal drift).",
            contamination_scope=ContaminationScope.MODULE,
            needs_review=True,
        )

    if gate_verdict == GateVerdict.FAIL_INTRODUCED_BY_DIFF:
        return GovernanceDecision(
            action=Action.REVERT_AND_PARK,
            outcome=OutcomeState.PARKED,
            reason="Regression detected: Diff broke tests that passed on baseline.",
            contamination_scope=ContaminationScope.FILE,
            needs_review=True,
        )

    if gate_verdict == GateVerdict.FAIL_PREEXISTING:
        return GovernanceDecision(
            action=Action.PARK,
            outcome=OutcomeState.PARKED,
            reason="Pre-existing test failure detected. Requires explicit human intervention.",
            contamination_scope=ContaminationScope.MODULE,
            needs_review=True,
        )

    if gate_verdict == GateVerdict.FAIL_ENV:
        return GovernanceDecision(
            action=Action.HALT,
            outcome=OutcomeState.HALTED,
            reason="Environment error or execution timeout encountered.",
            contamination_scope=ContaminationScope.SERVICE,
            needs_review=True,
        )


    # Gate is PASS: Evaluate epistemic calibration
    if epistemic_trust >= min_trust_threshold:
        return GovernanceDecision(
            action=Action.PROCEED,
            outcome=OutcomeState.DONE,
            reason=f"Deterministic tests passed and trust calibrated at {epistemic_trust:.3f} >= {min_trust_threshold}.",
            contamination_scope=ContaminationScope.NONE,
            needs_review=False,
        )
    elif epistemic_trust >= marginal_trust_threshold:
        return GovernanceDecision(
            action=Action.PROCEED,
            outcome=OutcomeState.DONE_LOW_CONFIDENCE,
            reason=f"Deterministic tests passed, but trust is marginal ({epistemic_trust:.3f}). Tagged for daylight review.",
            contamination_scope=ContaminationScope.FILE,
            needs_review=True,
        )
    else:
        return GovernanceDecision(
            action=Action.PARK,
            outcome=OutcomeState.PARKED,
            reason=f"Trust too low ({epistemic_trust:.3f} < {marginal_trust_threshold}). Parking step to prevent hallucinated completion.",
            contamination_scope=ContaminationScope.MODULE,
            needs_review=True,
        )

