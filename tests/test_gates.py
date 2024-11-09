"""
Tests for deterministic execution gates and governance state machine.
"""

from __future__ import annotations

import sys
import pytest

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
)


def test_gate_command_execution_success() -> None:
    # Cross-platform python inline success
    res = run_gate_command([sys.executable, "-c", "import sys; sys.exit(0)"])
    assert res.is_success
    assert res.returncode == 0
    assert not res.timed_out


def test_gate_command_execution_failure() -> None:
    res = run_gate_command([sys.executable, "-c", "import sys; sys.exit(1)"])
    assert not res.is_success
    assert res.returncode == 1


def test_gate_differential_evaluation() -> None:
    pass_res = GateExecutionResult(["test"], 0, "", "", False, True)
    fail_res = GateExecutionResult(["test"], 1, "", "", False, False)

    # Baseline PASS, Post PASS -> PASS
    assert evaluate_gate_diff(pass_res, pass_res) == GateVerdict.PASS
    # Baseline PASS, Post FAIL -> FAIL_INTRODUCED_BY_DIFF
    assert evaluate_gate_diff(pass_res, fail_res) == GateVerdict.FAIL_INTRODUCED_BY_DIFF
    # Baseline FAIL, Post FAIL -> FAIL_PREEXISTING
    assert evaluate_gate_diff(fail_res, fail_res) == GateVerdict.FAIL_PREEXISTING

