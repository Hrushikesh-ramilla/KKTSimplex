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

# Robust regex patterns for shortcut and stub detection
_DEFAULT_FORBIDDEN_PATTERNS = (
    re.compile(r"#\s*(todo|fixme)\b", re.IGNORECASE),
    re.compile(r"\bnotimplementederror\b", re.IGNORECASE),
    re.compile(r"pass\s*#\s*placeholder", re.IGNORECASE),
    re.compile(r":\s*\.{3}\s*(?:\n|$)", re.IGNORECASE),  # Ellipsis stub body
    re.compile(r"raise\s+\w+\s*\(\s*['\"][^'\"]*not\s+implemented", re.IGNORECASE),
)

# Common homoglyph mappings (Cyrillic, Greek, etc. to ASCII)
_HOMOGLYPH_MAP = str.maketrans({
    "Т": "T", "т": "t", "О": "O", "о": "o", "Е": "E", "е": "e", "А": "A", "а": "a",
    "Р": "P", "р": "p", "С": "C", "с": "c", "Х": "X", "х": "x", "В": "B", "М": "M",
})


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


def validate_against_anchor(
    anchor: SpecAnchor,
    diff_or_code: str,
) -> DriftCheckResult:
    """
    Validate an agent's candidate code or diff against the frozen SpecAnchor.
    Includes Unicode homoglyph normalization and regex pattern matching.
    """
    violations: List[str] = []

    # Normalize Unicode homoglyphs and compatibility forms
    normalized_code = unicodedata.normalize("NFKD", diff_or_code).translate(_HOMOGLYPH_MAP)

    # 1. Check robust forbidden patterns
    for pattern in _DEFAULT_FORBIDDEN_PATTERNS:
        match = pattern.search(normalized_code)
        if match:
            violations.append(f"Forbidden shortcut pattern detected: '{match.group(0).strip()}'")

    # Also check user-configured literal forbidden tokens
    for token in anchor.forbidden_tokens:
        if token.lower() in normalized_code.lower():
            violations.append(f"Forbidden shortcut token detected: '{token}'")

    # 2. Check each explicit invariant predicate
    for inv in anchor.invariants:
        try:
            holds = inv.check(diff_or_code)
            if not holds:
                violations.append(f"Invariant violated: [{inv.name}] - {inv.description}")
        except Exception as exc:
            violations.append(f"Invariant check failed with error: [{inv.name}] - {exc}")

    return DriftCheckResult(
        passed=(len(violations) == 0),
        violations=violations,
        checked_count=len(anchor.invariants) + len(anchor.forbidden_tokens) + len(_DEFAULT_FORBIDDEN_PATTERNS),
    )
