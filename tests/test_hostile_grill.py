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


# =====================================================================
# 3. GATES: Deterministic Execution Gates Hardening
# =====================================================================

class TestGatesHardening:

    def test_timeout_handles_str_without_attribute_error(self):
        """P0: Timeout on text=True does not crash on .decode(). Returns timed_out=True."""
        cmd = [sys.executable, "-c", "import time, sys; sys.stdout.write(chr(65)); sys.stdout.flush(); time.sleep(3)"]
        res = run_gate_command(cmd, timeout_seconds=1)
        assert res.timed_out is True
        assert res.is_success is False
        assert res.returncode == -1

    def test_non_utf8_output_handled_cleanly_with_replacement(self):
        """P1: Non-UTF8 bytes are replaced cleanly without dropping output."""
        cmd = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([255, 254, 170])); sys.stdout.flush()"]
        res = run_gate_command(cmd)
        assert res.is_success is True
        assert len(res.stdout) > 0  # Replaced characters preserved

    def test_shell_injection_prevented_on_string_command(self):
        """P1: String commands execute with shell=False; shell metacharacters are safely tokenized."""
        # Chaining command is treated as arguments to python, not executed as secondary shell command
        res = run_gate_command(f"{sys.executable} -c \"print('first')\" & {sys.executable} -c \"print('second')\"")
        assert "first" in res.stdout
        assert "second" not in res.stdout  # Secondary shell command prevented!


# =====================================================================
# 4. GOVERNANCE: State Machine Permutations Hardening
# =====================================================================

class TestGovernanceHardening:

    def test_negative_spec_violations_rejected(self):
        """P1: Negative spec violations raise ValueError rather than granting DONE."""
        with pytest.raises(ValueError, match="spec_violations must be a non-negative integer"):
            evaluate_task_completion(
                gate_verdict=GateVerdict.PASS,
                epistemic_trust=0.85,
                spec_violations=-1,
            )

    def test_unknown_verdict_rejected(self):
        """P1: Unrecognized GateVerdict raises ValueError rather than falling through to DONE."""
        with pytest.raises(ValueError, match="Invalid or unrecognized GateVerdict"):
            evaluate_task_completion(
                gate_verdict="CORRUPT_UNKNOWN_VERDICT",  # type: ignore
                epistemic_trust=0.85,
                spec_violations=0,
            )

    def test_infinite_trust_rejected(self):
        """P2: Non-finite epistemic_trust raises ValueError."""
        with pytest.raises(ValueError, match="epistemic_trust must be a finite float"):
            evaluate_task_completion(
                gate_verdict=GateVerdict.PASS,
                epistemic_trust=float("inf"),
                spec_violations=0,
            )

    def test_revert_git_diff_reverts_staged_changes(self):
        """P0: revert_git_diff uses git reset --hard to completely revert staged changes."""
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["git", "init"], cwd=td, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=td, check=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=td, check=True)

            fpath = os.path.join(td, "main.py")
            with open(fpath, "w") as f:
                f.write("def safe(): return True\n")
            subprocess.run(["git", "add", "."], cwd=td, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=td, check=True)

            # Agent edits and stages broken code
            with open(fpath, "w") as f:
                f.write("MALICIOUS / BROKEN CODE\n")
            subprocess.run(["git", "add", "main.py"], cwd=td, check=True)

            # Revert is triggered
            reverted = revert_git_diff(cwd=td)
            assert reverted is True

            # Staged broken code must be completely gone!
            with open(fpath, "r") as f:
                content = f.read()
            assert "MALICIOUS / BROKEN CODE" not in content
            assert "def safe(): return True" in content


# =====================================================================
# 5. ANCHOR: Spec Contract Hardening
# =====================================================================

class TestAnchorHardening:

    def test_whitespace_variation_intercepted(self):
        """P2: SpecAnchor catches whitespace-padded and omitted variations of # TODO / FIXME."""
        anchor = SpecAnchor(task_id="t1", goal_description="goal", invariants=[])

        evasions = [
            "#  TODO: implement this",
            "# TODO:  implement this",
            "pass # placeholder",
            "# TODO",
            "# FIXME",
        ]
        for code in evasions:
            res = validate_against_anchor(anchor, code)
            assert res.passed is False, f"Failed to catch: {code}"
            assert len(res.violations) >= 1

    def test_dynamic_execution_and_ellipsis_intercepted(self):
        """P2: SpecAnchor catches Ellipsis stubbing and dynamic NotImplementedError."""
        anchor = SpecAnchor(task_id="t1", goal_description="goal", invariants=[])

        evasions = [
            "def solve():\n    ...",
            "def solve():\n    raise eval('NotImplementedError')",
            "def solve():\n    raise RuntimeError('Not implemented')",
        ]
        for code in evasions:
            res = validate_against_anchor(anchor, code)
            assert res.passed is False, f"Failed to catch: {code}"

    def test_unicode_homoglyphs_intercepted(self):
        """P2: Replacing characters with Unicode homoglyphs is caught via normalization."""
        anchor = SpecAnchor(task_id="t1", goal_description="goal", invariants=[])
        # Cyrillic capital Т: \u0422
        code = "# \u0422ODO: implement"
        res = validate_against_anchor(anchor, code)
        assert res.passed is False


# =====================================================================
# 6. SEARCH: Execution-Guided MCTS Hardening
# =====================================================================

class TestSearchHardening:

    def test_empty_candidate_generator_sets_terminal_and_correct_rollouts(self):
        """P1: Empty step_generator sets node.is_terminal=True and returns non-None rollouts."""
        mcts = ExecutionGuidedMCTS(
            step_generator=lambda traj: [],
            execution_verifier=lambda traj: (GateVerdict.PASS, 1.0),
            num_simulations=5,
        )
        res = mcts.search("test problem")
        assert res.best_trajectory == []
        assert res.total_rollouts == 5  # Fixed: returns 5, not None!

    def test_all_candidates_fail_returns_empty_trajectory(self):
        """P1: When all candidates fail execution gates, MCTS does not return a broken trajectory."""
        steps = {"root": ["regressed_step_1", "regressed_step_2"]}

        def step_gen(traj):
            return steps.get(traj[-1], [])

        def verifier(traj):
            return GateVerdict.FAIL_INTRODUCED_BY_DIFF, 0.0

        mcts = ExecutionGuidedMCTS(
            step_generator=step_gen,
            execution_verifier=verifier,
            num_simulations=10,
        )
        res = mcts.search("root")
        # Fixed: refutes returning a regressed candidate as verified!
        assert res.best_trajectory == []
        assert res.best_value == 0.0
