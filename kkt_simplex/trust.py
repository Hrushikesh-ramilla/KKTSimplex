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

