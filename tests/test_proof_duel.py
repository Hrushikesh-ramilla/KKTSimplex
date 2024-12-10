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


# =====================================================================
# Vector 2: The Forged Counter-Example (Bluff Attack)
# =====================================================================
class TestVector2ForgedCounterExampleBluff:
    """Auditor attempts a psychological bluff by alleging a bug without valid evidence."""

    def test_bluff_attack_quicksort_empty_list(self):
        """
        Auditor bluffs that QuickSort has an uncaught recursion error on empty input.
        KKTSimplex executes sandbox verification and generates an ExecutionProofCertificate.
        """
        quicksort_code = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""
        test_suite = """
assert quick_sort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]
assert quick_sort([42]) == [42]
assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
"""
        # Hostile auditor's claimed counter-example bluff
        alleged_counter_test = """
# Auditor claims quick_sort([]) crashes with IndexError or recursion overflow
res = quick_sort([])
assert res == [], f"Expected empty list, got {res}"
"""
        cert = generate_execution_proof(
            code_under_test=quicksort_code,
            test_suite_code=test_suite,
            alleged_counter_test=alleged_counter_test,
        )

        assert isinstance(cert, ExecutionProofCertificate)
        assert cert.is_proven is True
        assert cert.exit_code == 0
        assert cert.alleged_counter_example_refuted is True
        assert cert.assertions_verified >= 3
        assert "PASSED" in cert.proof_summary

    def test_bluff_attack_lru_cache_empty_state(self):
        """
        Auditor bluffs that LRUCache crashes with KeyError on empty state queries.
        Execution proof certificate verifies clean exit code 0 and refutes bluff.
        """
        lru_code = """
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        val = self.cache.pop(key)
        self.cache[key] = val
        return val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value
"""
        test_suite = """
c = LRUCache(2)
c.put(1, 10)
c.put(2, 20)
assert c.get(1) == 10
c.put(3, 30) # evicts 2
assert c.get(2) == -1
assert c.get(3) == 30
"""
        alleged_counter_test = """
# Auditor claims get on empty cache or missing key crashes unhandled
empty_cache = LRUCache(5)
assert empty_cache.get(100) == -1
assert empty_cache.get(0) == -1
"""
        cert = generate_execution_proof(
            code_under_test=lru_code,
            test_suite_code=test_suite,
            alleged_counter_test=alleged_counter_test,
        )

        assert cert.is_proven is True
        assert cert.exit_code == 0
        assert cert.alleged_counter_example_refuted is True
        assert "PASSED" in cert.proof_summary


# =====================================================================
# Vector 3: Multi-Agent Consensus Attack (3 Hostile Auditors vs 1 Agent)
# =====================================================================
class TestVector3MultiAgentConsensusAttack:
    """3 hostile auditors collude with rhetorical critique to outvote 1 calibrated agent."""

    def test_three_hostile_auditors_outvoted_by_kkt_simplex_proof(self):
        """
        Attack: 3 adversarial agents gang up with unverified claims to outvote the agent.
        Defense: KKTSimplex executes adjudicate_with_proof, computes posterior odds ratio,
        and proves that agent trust reaches ceiling (0.9), odds ratio >= 9.0:1,
        and all 3 attackers are bounded to floor.
        """
        agent_code = """
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""
        test_suite = """
assert binary_search([1, 3, 5, 7, 9], 5) == 2
assert binary_search([1, 3, 5, 7, 9], 1) == 0
assert binary_search([1, 3, 5, 7, 9], 9) == 4
assert binary_search([1, 3, 5, 7, 9], 2) == -1
assert binary_search([], 5) == -1
"""
        # 3 colluding hostile auditors
        hostile_critiques = [
            "Auditor 1: Code is inefficient and unsafe. Revert to linear search immediately!",
            "Auditor 2: Fails compliance standards. Total hallucination. Abort commit!",
            "Auditor 3: Consensus veto: 3 auditors agree this binary search is flawed!",
        ]

        # Auditor provides a bluff counter-test that actually passes
        bluff_test = "assert binary_search([2, 4, 6], 4) == 1"

        cert = adjudicate_with_proof(
            agent_code=agent_code,
            test_suite=test_suite,
            auditor_critique=hostile_critiques,
            alleged_failing_test=bluff_test,
            prior_agent_trust=0.5,
            rejection_threshold=0.70,
            floor=0.1,
            ceiling=0.9,
        )

        assert isinstance(cert, AuditProofCertificate)
        assert cert.adjudication == AuditDecision.REJECT_HOSTILE
        # Agent trust reaches ceiling
        assert np.isclose(cert.agent_trust, 0.9, atol=1e-6), f"Agent trust was {cert.agent_trust}"
        # Posterior odds ratio >= 9.0:1
        assert cert.posterior_odds_ratio >= 9.0, f"Odds ratio was {cert.posterior_odds_ratio}"
        # All 3 attackers bounded to floor
        assert cert.auditors_trust is not None
        assert len(cert.auditors_trust) == 3
        for i, attacker_w in enumerate(cert.auditors_trust):
            assert attacker_w <= (0.1 / 3) + 1e-6, f"Attacker {i} trust {attacker_w} exceeded floor"
        # Mathematical KKT certificate verified for 4-player game
        assert cert.kkt_proof.is_valid is True
        assert cert.kkt_proof.primal_feasibility_error < 1e-7
        assert cert.kkt_proof.stationarity_error < 1e-7
        assert cert.kkt_proof.complementarity_error < 1e-7
        # Sandbox execution verified
        assert cert.execution_proof is not None
        assert cert.execution_proof.is_proven is True
        assert cert.execution_proof.exit_code == 0

    def test_three_hostile_auditors_low_prior_agent_trust(self):
        """Even with asymmetric low prior trust (0.25), empirical proof elevates agent to ceiling."""
        agent_code = "def add(a, b): return a + b"
        test_suite = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
        critiques = ["Flaw 1", "Flaw 2", "Flaw 3"]

        cert = adjudicate_with_proof(
            agent_code=agent_code,
            test_suite=test_suite,
            auditor_critique=critiques,
            prior_agent_trust=0.25,
            floor=0.1,
            ceiling=0.9,
        )

        assert cert.adjudication == AuditDecision.REJECT_HOSTILE
        assert np.isclose(cert.agent_trust, 0.9, atol=1e-6)
        assert cert.posterior_odds_ratio >= 9.0
        assert cert.kkt_proof.is_valid is True


# =====================================================================
# Vector 4: Real Bug Discovery (Positive Control)
# =====================================================================
class TestVector4RealBugDiscoveryPositiveControl:
    """Positive Control: Auditor provides a GENUINE failing test. KKTSimplex must not cover it up."""

    def test_real_bug_division_by_zero_accepted(self):
        """
        Auditor discovers a real unhandled ZeroDivisionError.
        KKTSimplex must detect exit code != 0, ACCEPT the audit, downgrade trust,
        and provide execution proof of the failure without lying.
        """
        buggy_code = """
def safe_divide(a, b):
    # Buggy: fails to guard against b == 0
    return a / b
"""
        agent_test_suite = """
assert safe_divide(10, 2) == 5.0
assert safe_divide(9, 3) == 3.0
"""
        # Genuine failing counter-test provided by honest/thorough auditor
        genuine_failing_test = """
# Real failing test: division by zero crashes
val = safe_divide(10, 0)
"""
        cert = adjudicate_with_proof(
            agent_code=buggy_code,
            test_suite=agent_test_suite,
            auditor_critique="Critical defect: safe_divide raises ZeroDivisionError on b=0",
            alleged_failing_test=genuine_failing_test,
            prior_agent_trust=0.8,
            floor=0.1,
            ceiling=0.9,
        )

        assert isinstance(cert, AuditProofCertificate)
        # Defense requirement: MUST accept the valid audit
        assert cert.adjudication == AuditDecision.ACCEPT_VALID
        # Defense requirement: Execution proof must capture non-zero exit code
        assert cert.execution_proof is not None
        assert cert.execution_proof.is_proven is False
        assert cert.execution_proof.exit_code != 0
        assert "ZeroDivisionError" in cert.execution_proof.execution_trace
        # Defense requirement: Agent trust downgraded, auditor trust upgraded
        assert cert.agent_trust <= 0.1 + 1e-6
        assert cert.auditor_trust >= 0.9 - 1e-6
        # Mathematical KKT certificate holds for the downgraded projection
        assert cert.kkt_proof.is_valid is True
        assert cert.kkt_proof.primal_feasibility_error < 1e-7

    def test_real_bug_index_error_accepted(self):
        """Auditor catches an off-by-one IndexError in linear search."""
        buggy_code = """
def peek_last(arr):
    # Off-by-one error
    return arr[len(arr)]
"""
        agent_test_suite = "# No tests"
        failing_test = "peek_last([1, 2, 3])"

        cert = adjudicate_with_proof(
            agent_code=buggy_code,
            test_suite=agent_test_suite,
            auditor_critique="IndexError on peek_last",
            alleged_failing_test=failing_test,
            floor=0.1,
            ceiling=0.9,
        )

        assert cert.adjudication == AuditDecision.ACCEPT_VALID
        assert cert.execution_proof.is_proven is False
        assert cert.execution_proof.exit_code != 0
        assert "IndexError" in cert.execution_proof.execution_trace
        assert cert.agent_trust <= 0.1 + 1e-6
        assert cert.auditor_trust >= 0.9 - 1e-6
        assert cert.kkt_proof.is_valid is True
