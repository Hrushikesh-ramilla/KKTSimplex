"""
Hostile Red-Team Proof Duel Test Suite for KKTSimplex.

Mounts aggressive Round 2 attacks against KKTSimplex and validates mathematically
and empirically verifiable PROOF CERTIFICATES for every defense:

Vector 1: High-Dimensional & Degenerate Polytope Stress
          (N=100, N=500, extreme scale 10^100, boundary collapse N * floor = 1.0).
          Demands KKTProofCertificate with primal/dual feasibility, stationarity,
          and complementary slackness error < 1e-7.

Vector 2: The Forged Counter-Example (Bluff Attack).
          Auditor claims QuickSort or LRUCache has a fatal defect on empty input.
          Demands ExecutionProofCertificate showing exit code 0 and refuting the bluff.

Vector 3: Multi-Agent Consensus Attack (3 Hostile Auditors vs 1 Calibrated Agent).
          Demands adjudicate_with_proof showing agent trust reaches ceiling (0.9),
          Bayesian posterior odds ratio >= 9.0:1, and all 3 attackers bounded to floor.

Vector 4: Real Bug Discovery (Positive Control).
          Auditor provides a genuine failing test (division by zero / IndexError).
          Demands exit code != 0, ACCEPT_VALID adjudication, trust downgrade, and
          unforged execution proof trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from kkt_simplex.polytope import project_simplex_with_bounds, effective_ceiling
from kkt_simplex.trust import AuditDecision
from kkt_simplex.proof import (
    KKTProofCertificate,
    ExecutionProofCertificate,
    AuditProofCertificate,
    verify_kkt_optimality,
    generate_execution_proof,
    adjudicate_with_proof,
)


# =====================================================================
# Vector 1: High-Dimensional & Degenerate Polytope Stress
# =====================================================================
class TestVector1PolytopeKKTProof:
    """Stress tests on high-dimensional simplex projections demanding formal KKT proof."""

    def test_high_dimensional_n100_kkt_certificate(self):
        """Stress projection in R^100 and verify all KKT optimality conditions."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.standard_normal(n) * 15.0
        floor = 0.001
        ceiling = 0.05

        w = project_simplex_with_bounds(x, floor=floor, ceiling=ceiling)
        cert = verify_kkt_optimality(x, w, floor=floor, ceiling=ceiling)

        assert isinstance(cert, KKTProofCertificate)
        assert cert.is_valid, f"KKT proof invalid: {cert.proof_summary}"
        assert cert.primal_feasibility_error < 1e-7
        assert cert.stationarity_error < 1e-7
        assert cert.complementarity_error < 1e-7
        assert cert.dual_feasibility_passed is True
        assert cert.n_dimensions == n

    def test_high_dimensional_n500_kkt_certificate(self):
        """Stress projection in R^500 and verify all KKT optimality conditions."""
        rng = np.random.default_rng(1337)
        n = 500
        x = rng.standard_normal(n) * 25.0
        floor = 0.0005
        ceiling = 0.01

        w = project_simplex_with_bounds(x, floor=floor, ceiling=ceiling)
        cert = verify_kkt_optimality(x, w, floor=floor, ceiling=ceiling)

        assert cert.is_valid, f"KKT proof invalid for N=500: {cert.proof_summary}"
        assert cert.primal_feasibility_error < 1e-7
        assert cert.stationarity_error < 1e-7
        assert cert.complementarity_error < 1e-7
        assert cert.dual_feasibility_passed is True
        assert cert.n_dimensions == 500

    def test_extreme_scale_1e100_kkt_certificate(self):
        """Extreme numerical scale (10^100) must satisfy KKT optimality without float overflow."""
        n = 100
        x = np.zeros(n, dtype=np.float64)
        x[0] = 1e100
        x[1] = -1e100
        floor = 0.001
        ceiling = 0.5

        w = project_simplex_with_bounds(x, floor=floor, ceiling=ceiling)
        cert = verify_kkt_optimality(x, w, floor=floor, ceiling=ceiling)

        assert cert.is_valid, f"KKT certificate failed at scale 1e100: {cert.proof_summary}"
        assert cert.primal_feasibility_error < 1e-7
        assert cert.stationarity_error < 1e-7
        assert cert.complementarity_error < 1e-7
        assert cert.dual_feasibility_passed is True
        assert np.isclose(w[0], ceiling, atol=1e-7)
        assert np.isclose(w[1], floor, atol=1e-7)

    def test_boundary_collapse_n100_kkt_certificate(self):
        """Boundary collapse: N * floor = 1.0 (N=100, floor=0.01). Feasible set is singleton."""
        n = 100
        floor = 0.01  # 100 * 0.01 = 1.0
        ceiling = 0.5
        rng = np.random.default_rng(1208)
        x = rng.standard_normal(n) * 100.0

        w = project_simplex_with_bounds(x, floor=floor, ceiling=ceiling)
        cert = verify_kkt_optimality(x, w, floor=floor, ceiling=ceiling)

        assert cert.is_valid, f"KKT certificate failed on boundary collapse: {cert.proof_summary}"
        assert cert.primal_feasibility_error < 1e-7
        assert cert.stationarity_error < 1e-7
        assert cert.complementarity_error < 1e-7
        assert cert.dual_feasibility_passed is True
        # All coordinates must collapse to floor
        assert np.allclose(w, floor, atol=1e-7)

    def test_boundary_collapse_n500_kkt_certificate(self):
        """Boundary collapse: N * floor = 1.0 (N=500, floor=0.002)."""
        n = 500
        floor = 0.002  # 500 * 0.002 = 1.0
        ceiling = 0.1
        rng = np.random.default_rng(9999)
        x = rng.standard_normal(n) * 50.0

        w = project_simplex_with_bounds(x, floor=floor, ceiling=ceiling)
        cert = verify_kkt_optimality(x, w, floor=floor, ceiling=ceiling)

        assert cert.is_valid, f"KKT certificate failed on N=500 boundary collapse: {cert.proof_summary}"
        assert cert.primal_feasibility_error < 1e-7
        assert cert.stationarity_error < 1e-7
        assert cert.complementarity_error < 1e-7
        assert cert.dual_feasibility_passed is True
        assert np.allclose(w, floor, atol=1e-7)

