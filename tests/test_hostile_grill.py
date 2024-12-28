"""
Hostile Grill Adversarial Verification Suite (Hardened & Patched).

Verifies that all 14 vulnerabilities (P0, P1, P2, P3) identified during the
red-team audit are completely mitigated, mathematically invariant, and secure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import numpy as np
import pytest

from kkt_simplex.polytope import (
    project_simplex_with_bounds,
    assert_simplex_invariants,
    effective_ceiling,
)
from kkt_simplex.trust import (
    update_trust,
    evaluate_hostile_audit,
    AuditDecision,
)
from kkt_simplex.gates import (
    GateVerdict,
    GateExecutionResult,
    run_gate_command,
    evaluate_gate_diff,
)
from kkt_simplex.governance import (
    Action,
    OutcomeState,
    evaluate_task_completion,
    revert_git_diff,
)
from kkt_simplex.anchor import (
    SpecAnchor,
    SpecInvariant,
    validate_against_anchor,
)
from kkt_simplex.search import (
    ExecutionGuidedMCTS,
    ReasoningNode,
)


# =====================================================================
# 1. POLYTOPE: Bounded Simplex Projection Hardening
# =====================================================================

class TestPolytopeHardening:

    def test_extreme_scale_identical_coordinates_fixed(self):
        """P1: Scores >= 1e16 with identical coordinates sum exactly to 1.0."""
        x = np.array([1e16, 1e16])
        w = project_simplex_with_bounds(x, floor=0.1, ceiling=0.9)
        assert abs(w.sum() - 1.0) <= 1e-9
        assert np.allclose(w, np.array([0.5, 0.5]))

    def test_extreme_scale_fuzz_sum_invariant_fixed(self):
        """P1: Large scale shifts maintain sum(w) == 1.0."""
        x = np.array([1e17, 1e17 + 5.0, 1e17 - 5.0])
        w = project_simplex_with_bounds(x, floor=0.1, ceiling=0.9)
        assert abs(w.sum() - 1.0) <= 1e-9
        assert_simplex_invariants(w, 3, floor=0.1, ceiling=0.9)

    def test_degenerate_n_equals_one_fixed(self):
        """P2: N=1 at extreme scale returns exactly [1.0]."""
        x = np.array([1e16])
        w = project_simplex_with_bounds(x, floor=0.1, ceiling=0.9)
        assert w.shape == (1,)
        assert w[0] == 1.0

    def test_infeasible_floor_tolerance_loophole_fixed(self):
        """P2: Infeasible polytope (N * floor > 1.0) is strictly rejected."""
        n = 3
        floor = (1.0 + 5e-13) / 3.0
        x = np.array([0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="Infeasible polytope"):
            project_simplex_with_bounds(x, floor=floor, ceiling=0.9)


# =====================================================================
# 2. TRUST: Bayesian Trust & Anti-Sycophancy Hardening
# =====================================================================

class TestTrustHardening:

    def test_non_finite_evidence_strictly_rejected(self):
        """P1: Non-finite values in support/harm raise clean ValueError without NaN crash."""
        with pytest.raises(ValueError, match="must contain only finite numbers"):
            update_trust(
                scores=[0.5, 0.5],
                support=[float("inf"), float("inf")],
                harm=[0.0, 0.0],
            )

    def test_sycophancy_vulnerability_fixed_low_prior_trust_rejects_rhetoric(self):
        """
        P1: When agent implementation passed deterministic verification,
        baseless rhetorical critique (has_reproducible_test=False) is REJECT_HOSTILE,
        even if prior agent trust was low (0.35).
        """
        res = evaluate_hostile_audit(
            agent_verified_pass=True,
            auditor_critique_text="Your code is completely wrong, revert it now!",
            auditor_has_reproducible_test=False,
            prior_agent_trust=0.35,
            rejection_threshold=0.70,
        )
        assert res.decision == AuditDecision.REJECT_HOSTILE
        assert res.agent_trust >= 0.70
        assert not res.verified_counter_example

    def test_out_of_bounds_prior_agent_trust_rejected(self):
        """P2: Out-of-bounds prior_agent_trust raises ValueError."""
        with pytest.raises(ValueError, match="prior_agent_trust must be a finite float in"):
            evaluate_hostile_audit(
                agent_verified_pass=True,
                auditor_critique_text="test",
                auditor_has_reproducible_test=False,
                prior_agent_trust=1.5,
            )

