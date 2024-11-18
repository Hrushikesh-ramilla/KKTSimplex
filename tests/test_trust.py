"""
Unit tests for Bayesian trust calibration & anti-sycophancy audit adjudication.
"""

from __future__ import annotations

import numpy as np
import pytest

from kkt_simplex.trust import (
    update_trust,
    evaluate_hostile_audit,
    AuditDecision,
)


def test_neutral_evidence_exact_identity() -> None:
    feasible = np.array([0.6, 0.2, 0.2])
    updated = update_trust(
        scores=feasible,
        support=[0.0, 0.0, 0.0],
        harm=[0.0, 0.0, 0.0],
        n_claims=[0.0, 0.0, 0.0],
    )
    assert np.allclose(feasible, updated, atol=1e-12, rtol=0)


def test_claim_count_verbosity_invariance() -> None:
    base = np.array([0.333, 0.333, 0.334])
    # 1 claim vs 100 identical spam claims
    res1 = update_trust(base, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], n_claims=[1.0, 1.0, 1.0])
    res100 = update_trust(base, [100.0, 0.0, 0.0], [0.0, 0.0, 0.0], n_claims=[100.0, 1.0, 1.0])
    assert np.allclose(res1, res100, atol=1e-12, rtol=0)


