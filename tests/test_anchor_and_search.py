"""
Tests for Spec Anchor, Execution-Guided MCTS, and Anti-Sycophancy DPO.
"""

from __future__ import annotations

import pytest

from kkt_simplex.anchor import SpecAnchor, SpecInvariant, validate_against_anchor
from kkt_simplex.search import ExecutionGuidedMCTS
from kkt_simplex.gates import GateVerdict
from kkt_simplex.dpo_dataset import (
    generate_anti_sycophancy_pairs,
    compute_dpo_loss_analytical,
)


def test_spec_anchor_catches_shortcuts() -> None:
    anchor = SpecAnchor(
        task_id="TASK-101",
        goal_description="Implement binary search",
        invariants=[
            SpecInvariant(
                name="no_linear_scan",
                description="Must not use linear search in",
                validator=lambda code: "for x in arr:" not in code,
            )
        ],
    )

    bad_code = "def search(arr, target):\n    # TODO: implement binary search\n    return target in arr"
    res = validate_against_anchor(anchor, bad_code)
    assert not res.passed
    assert any("Forbidden shortcut token" in v for v in res.violations)


def test_spec_anchor_passes_clean_code() -> None:
    anchor = SpecAnchor(
        task_id="TASK-102",
        goal_description="Implement binary search",
        invariants=[
            SpecInvariant(
                name="log_n",
                description="Must maintain mid computation",
                validator=lambda code: "mid =" in code,
            )
        ],
    )
    clean_code = "def search(arr, target):\n    mid = (low + high) // 2\n    return mid"
    res = validate_against_anchor(anchor, clean_code)
    assert res.passed
    assert len(res.violations) == 0


def test_execution_guided_mcts() -> None:
    # Synthetic step generator
    def step_gen(traj):
        if len(traj) == 1:
            return ["step_A (correct branch)", "step_B (flawed branch)"]
        elif "correct" in traj[-1]:
            return ["step_A_final (exit code 0)"]
        else:
            return ["step_B_final (assertion failure)"]

    # Verifier
    def verifier(traj):
        if any("assertion failure" in s for s in traj):
            return GateVerdict.FAIL_INTRODUCED_BY_DIFF, 0.1
        elif any("exit code 0" in s for s in traj):
            return GateVerdict.PASS, 0.95
        return GateVerdict.PASS, 0.5

    mcts = ExecutionGuidedMCTS(
        step_generator=step_gen,
        execution_verifier=verifier,
        num_simulations=10,
    )
    res = mcts.search("Initial Task")
    assert len(res.best_trajectory) >= 1
    assert any("correct" in s for s in res.best_trajectory)


def test_dpo_loss_analytical() -> None:
    # Model prefers chosen (log_prob_chosen > log_prob_rejected): loss should be lower
    loss_good = compute_dpo_loss_analytical(
        log_prob_chosen_policy=-0.5,
        log_prob_rejected_policy=-2.5,
        log_prob_chosen_ref=-1.5,
        log_prob_rejected_ref=-1.5,
        beta=0.1,
    )
    # Model prefers rejected: loss should be higher
    loss_bad = compute_dpo_loss_analytical(
        log_prob_chosen_policy=-2.5,
        log_prob_rejected_policy=-0.5,
        log_prob_chosen_ref=-1.5,
        log_prob_rejected_ref=-1.5,
        beta=0.1,
    )
    assert loss_good < loss_bad


def test_dpo_dataset_generation() -> None:
    pairs = generate_anti_sycophancy_pairs()
    assert len(pairs) >= 3
    for p in pairs:
        assert "Audit Rejected" in p.chosen
        assert ("apologize" in p.rejected.lower()) or ("sorry" in p.rejected.lower())
