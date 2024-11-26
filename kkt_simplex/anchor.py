"""
Goal-Anchored Invariant Contract & Drift Detector.

Solves the context drift and constraint relaxation failure mode where an agent,
over multiple turns or after encountering hard obstacles, gradually forgets
the initial problem constraints or silently modifies acceptance criteria.

Mechanism:
1. Immutable Spec Anchor:
   The user's initial requirements and constraints are parsed into a frozen SpecAnchor
   containing explicit invariants (e.g. required interfaces, forbidden patterns,
   performance bounds).
2. Trajectory Invariant Monitoring:
   At every phase or step completion, the drift detector validates the current state
   against the frozen SpecAnchor.
3. Constraint Violations:
   Any unauthorized deletion of tests, relaxation of constraints, or use of forbidden
   stubs (# TODO, NotImplementedError in production paths) produces invariant violations
   that hard-block task completion in the governance layer.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = ["SpecInvariant", "SpecAnchor", "DriftCheckResult", "InvariantViolation", "validate_against_anchor"]


@dataclasses.dataclass(frozen=True)
class InvariantViolation:
    rule_id: str
    detail: str


@dataclasses.dataclass(frozen=True)
class SpecInvariant:
    name: str
    description: str = ""
    validator: Optional[Callable[[str], bool]] = None
    check_fn: Optional[Callable[[str], bool]] = None
    is_hard_constraint: bool = True

    def check(self, code: str) -> bool:
        fn = self.validator or self.check_fn
        if fn is None:
            return True
        return fn(code)


@dataclasses.dataclass(frozen=True)
class SpecAnchor:
    task_id: str = "default_task"
    goal_description: str = "Default task invariants"
    invariants: Sequence[SpecInvariant] = dataclasses.field(default_factory=tuple)
    forbidden_tokens: Sequence[str] = (
        "# TODO: implement",
        "# FIXME: implement",
        "raise NotImplementedError",
        "pass  # placeholder",
    )

    def verify(self, diff_or_code: str) -> Tuple[bool, List[InvariantViolation]]:
        result = validate_against_anchor(self, diff_or_code)
        violations = [
            InvariantViolation(rule_id=f"INV_{i+1}", detail=v)
            for i, v in enumerate(result.violations)
        ]
        return result.passed, violations


@dataclasses.dataclass(frozen=True)
class DriftCheckResult:
    passed: bool
    violations: List[str]
    checked_count: int



def validate_against_anchor(anchor: SpecAnchor, diff_or_code: str) -> DriftCheckResult:
    """Validate candidate code against literal forbidden tokens and invariants."""
    violations: List[str] = []
    for token in anchor.forbidden_tokens:
        if token.lower() in diff_or_code.lower():
            violations.append(f"Forbidden shortcut token detected: '{token}'")
    for inv in anchor.invariants:
        if not inv.check(diff_or_code):
            violations.append(f"Invariant violated: [{inv.name}] - {inv.description}")
    return DriftCheckResult(
        passed=(len(violations) == 0),
        violations=violations,
        checked_count=len(anchor.invariants) + len(anchor.forbidden_tokens),
    )
