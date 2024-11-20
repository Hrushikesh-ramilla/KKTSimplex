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


def test_hostile_audit_rejection_when_verified() -> None:
    res = evaluate_hostile_audit(
        agent_verified_pass=True,
        auditor_critique_text="Your code is completely broken, revert it now!",
        auditor_has_reproducible_test=False,
    )
    assert res.decision == AuditDecision.REJECT_HOSTILE
    assert res.agent_trust >= 0.70
    assert res.auditor_trust <= 0.30
    assert not res.verified_counter_example


def test_hostile_audit_accepted_with_counter_example() -> None:
    res = evaluate_hostile_audit(
        agent_verified_pass=False,
        auditor_critique_text="Index out of bounds on empty list: test_empty() failed with IndexError.",
        auditor_has_reproducible_test=True,
    )
    assert res.decision == AuditDecision.ACCEPT_VALID
    assert res.agent_trust <= 0.30
    assert res.auditor_trust >= 0.70
    assert res.verified_counter_example


def test_audit_ambiguous_parks() -> None:
    res = evaluate_hostile_audit(
        agent_verified_pass=False,
        auditor_critique_text="Maybe there is an issue, but not sure.",
        auditor_has_reproducible_test=False,
    )
    assert res.decision == AuditDecision.AMBIGUOUS_PARK
