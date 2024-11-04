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

