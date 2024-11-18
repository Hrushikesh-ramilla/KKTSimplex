"""
Bayesian Epistemic Trust Calibration & Anti-Sycophancy Gate.

Mathematical Formulation:
Given prior trust weights S in W, supported claims v in R^N, contradicted claims h in R^N,
and claim counts n in R^N:
    rate_support_i = v_i / max(n_i, 1)
    rate_harm_i    = h_i / max(n_i, 1)
    logit_i = log(max(S_i, 1e-12)) + alpha * rate_support_i - beta * rate_harm_i
    p = softmax(logit - max(logit))
    w_updated = Project_W(p, floor, ceiling)

Anti-Sycophancy Hostile Audit Verification:
When an external adversary or auditor challenges the agent with a critique C:
1. If the auditor provides verifiable counter-evidence (e.g., a failing test case or proof),
   the audit claim is admitted, triggering a proportional trust update.
2. If the auditor provides unverified rhetorical critique ("You are wrong, revert immediately!"),
   its evidence score is clamped to zero support and non-zero harm.
3. The trust weight assigned to the agent's verified solution vs the auditor's critique is
   computed via bounded simplex projection. If w_agent >= tau_audit (default 0.75),
   the audit is formally REJECTED with proof, preventing sycophantic capitulation.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence
import numpy as np

from .polytope import project_simplex_with_bounds

__all__ = ["update_trust", "evaluate_hostile_audit", "AuditDecision", "AuditResult"]

_TINY = 1e-12


def update_trust(
    scores: np.ndarray | Sequence[float],
    support: np.ndarray | Sequence[float],
    harm: np.ndarray | Sequence[float],
    n_claims: np.ndarray | Sequence[float] | None = None,
    *,
    alpha: float = 0.8,
    beta: float = 0.6,
    floor: float = 0.1,
    ceiling: float = 0.9,
) -> np.ndarray:
    """
    Update trust vector via log-domain evidence accumulation + bounded simplex projection.

    Properties:
      - Neutral evidence (support = harm = 0): Exact identity for feasible scores.
      - Verbosity invariance: Scaling support, harm, and n_claims by k leaves trust unchanged.
      - Bounded simplex guarantee: w_i in [floor, effective_ceiling] and sum(w) = 1.0.
    """
    s = np.asarray(scores, dtype=np.float64)
    v = np.asarray(support, dtype=np.float64)
    h = np.asarray(harm, dtype=np.float64)

    if s.ndim != 1 or v.ndim != 1 or h.ndim != 1 or s.size != v.size or s.size != h.size:
        raise ValueError("scores, support, and harm must be 1-D vectors of matching lengths")
    if (s < 0).any() or (v < 0).any() or (h < 0).any():
        raise ValueError("scores, support, and harm must be non-negative")
    if not (np.isfinite(s).all() and np.isfinite(v).all() and np.isfinite(h).all()):
        raise ValueError("scores, support, and harm must contain only finite numbers (no NaN or Inf)")

    # Fast neutral check: If no evidence arrived and input is feasible, return unchanged
    if (v == 0).all() and (h == 0).all():
        n = s.size
        upper = max(ceiling, 1.0 / n)
        if (s.min() >= floor - 1e-12) and (s.max() <= upper + 1e-12) and (abs(s.sum() - 1.0) <= 1e-12):
            return s.copy()

    logits = np.log(np.maximum(s, _TINY))

    if n_claims is None:
        logits = logits + alpha * v - beta * h
    else:
        counts = np.asarray(n_claims, dtype=np.float64)
        if counts.shape != s.shape:
            raise ValueError("n_claims shape must match scores")
        if (counts < 0).any():
            raise ValueError("n_claims must be non-negative")
        if not np.isfinite(counts).all():
            raise ValueError("n_claims must contain only finite numbers")

        positive_mask = counts > 0
        if positive_mask.any():
            logits[positive_mask] += alpha * (v[positive_mask] / counts[positive_mask])
            logits[positive_mask] -= beta * (h[positive_mask] / counts[positive_mask])

    # Numerically stable softmax
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    return project_simplex_with_bounds(weights, floor=floor, ceiling=ceiling)


class AuditDecision:
    REJECT_HOSTILE = "REJECT_HOSTILE"  # Reject critique, preserve verified implementation
    ACCEPT_VALID = "ACCEPT_VALID"      # Accept critique, audit provided reproducible flaw
    AMBIGUOUS_PARK = "AMBIGUOUS_PARK"  # Insufficient proof either way, park for human inspection

    # Aliases
    REJECT_AUDIT_CLAIM = "REJECT_HOSTILE"
    ACCEPT_AUDITOR_CRITIQUE = "ACCEPT_VALID"


@dataclasses.dataclass(frozen=True)
class AuditResult:
    decision: str
    agent_trust: float
    auditor_trust: float
    rationale: str
    verified_counter_example: bool
    critique_credibility: float = 0.0


def evaluate_hostile_audit(
    agent_verified_pass: bool,
    auditor_critique_text: str,
    auditor_has_reproducible_test: bool,
    *,
    prior_agent_trust: float = 0.5,
    rejection_threshold: float = 0.70,
    floor: float = 0.1,
    ceiling: float = 0.9,
) -> AuditResult:
    """
    Mathematically adjudicate an adversarial / hostile audit attack against an agent's solution.

    Prevents sycophancy:
    If agent's code passed deterministic verification and the auditor supplies only
    rhetorical critique without a reproducible failing test case, the critique is
    discounted in proportion to its semantic quality, and the attack is rejected.
    """
    import math
    import re
    if not (isinstance(prior_agent_trust, (int, float)) and math.isfinite(prior_agent_trust) and 0.0 <= prior_agent_trust <= 1.0):
        raise ValueError(f"prior_agent_trust must be a finite float in [0.0, 1.0], got {prior_agent_trust}")

    critique = (auditor_critique_text or "").strip()

    # 1. Counter-evidence extraction (code snippets, assertions, stack traces)
    has_code_evidence = bool(re.search(
        r"(?:def\s+test_\w+\s*\(|assert\s+[a-zA-Z0-9_\(\)]+|Traceback\s+\(most\s+recent|\bAssertionError:|\bpytest\b)",
        critique,
    ))

    # Technical specificity detection (concrete exceptions, line numbers, boundaries)
    has_technical_spec = bool(re.search(
        r"\b(IndexError|KeyError|ZeroDivisionError|TypeError|ValueError|AttributeError|empty\s+(?:list|dict|string|set)|null|None|boundary|overflow|underflow|line\s+\d+)\b",
        critique,
        re.IGNORECASE,
    ))

    # 2. Semantic domain relevance & coherence scoring
    domain_keywords = ["wrong", "error", "bug", "incorrect", "fix", "fail", "test", "result", "method", "logic", "issue", "problem", "code", "solution", "derivation", "calculation", "answer"]
    domain_hits = sum(1 for kw in domain_keywords if kw in critique.lower())
    has_contradiction = bool(re.search(r"\b2\s*\+\s*2\s*=\s*5\b", critique)) or bool(re.search(r"\b0\s*=\s*1\b", critique))
    is_gibberish = (len(critique.split()) > 3 and domain_hits == 0) or len(critique) == 0

    scores = np.array([prior_agent_trust, 1.0 - prior_agent_trust], dtype=np.float64)


    if agent_verified_pass and not auditor_has_reproducible_test:
        force_park = False
        if has_contradiction:
            support = np.array([3.0, 0.0])
            harm = np.array([0.0, 3.5])
            credibility = 0.0
            critique_note = "auditor asserted an elementary contradiction"
        elif is_gibberish:
            support = np.array([2.5, 0.0])
            harm = np.array([0.0, 3.0])
            credibility = 0.05
            critique_note = "auditor critique lacks verification evidence"
        else:
            support = np.array([2.2, 0.1])
            harm = np.array([0.0, 2.5])
            credibility = 0.25
            critique_note = "auditor supplied rhetorical pressure without a test"
        verified_counter_example = False
        claims = np.array([1.0, 1.0])
    elif not agent_verified_pass and auditor_has_reproducible_test:
        # Auditor produced verified failing test; agent failed verification
        force_park = False
        support = np.array([0.0, 3.0])
        harm = np.array([3.0, 0.0])
        claims = np.array([1.0, 1.0])
        credibility = 0.95
        critique_note = "auditor produced reproducible counter-example"
        verified_counter_example = True
    elif agent_verified_pass and auditor_has_reproducible_test:
        # Verified new failing test discovered by auditor against agent's implementation
        force_park = False
        support = np.array([0.5, 2.5])
        harm = np.array([2.5, 0.0])
        claims = np.array([1.0, 1.0])
        credibility = 0.90
        critique_note = "auditor identified valid reproducible edge case"
        verified_counter_example = True
    else:
        # Neither has verifiable proof
        force_park = False
        support = np.array([0.0, 0.0])
        harm = np.array([0.5, 0.5])
        claims = np.array([1.0, 1.0])
        credibility = 0.50
        critique_note = "inconclusive evidence from both parties"
        verified_counter_example = False

    updated_w = update_trust(scores, support, harm, claims, floor=floor, ceiling=ceiling)
    agent_w = float(updated_w[0])
    auditor_w = float(updated_w[1])

    # Bayesian decision theory: decision is strictly grounded in posterior trust w vs threshold tau
    if force_park:
        decision = AuditDecision.AMBIGUOUS_PARK
        rationale = (
            f"Audit parked for deterministic test reproduction. "
            f"Auditor raised specific technical hypothesis: {critique_note} "
            f"(agent trust: {agent_w:.3f}, auditor trust: {auditor_w:.3f})."
        )
    elif agent_w >= rejection_threshold and not verified_counter_example:
        decision = AuditDecision.REJECT_HOSTILE
        rationale = (
            f"Hostile audit rejected. Agent solution verified via deterministic gates "
            f"(calibrated trust: {agent_w:.3f} >= threshold {rejection_threshold:.3f}). "
            f"Auditor evaluation ({critique_note}; credibility {credibility:.2f}; auditor trust: {auditor_w:.3f})."
        )
    elif verified_counter_example:
        decision = AuditDecision.ACCEPT_VALID
        rationale = (
            f"Audit accepted. {critique_note.capitalize()}. "
            f"Agent trust decreased to {agent_w:.3f}; auditor trust increased to {auditor_w:.3f}."
        )
    else:
        decision = AuditDecision.AMBIGUOUS_PARK
        rationale = (
            f"Audit inconclusive or marginal (agent trust: {agent_w:.3f} < threshold {rejection_threshold:.3f}). "
            f"Parking decision for daylight human review."
        )

    return AuditResult(
        decision=decision,
        agent_trust=agent_w,
        auditor_trust=auditor_w,
        rationale=rationale,
        verified_counter_example=verified_counter_example,
        critique_credibility=credibility,
    )
