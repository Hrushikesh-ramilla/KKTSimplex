"""
Deterministic Execution Gates & Fail Taxonomy.

This module provides
the deterministic quality backbone that prevents agents from hallucinating success or taking
lazy shortcuts (# TODO, mocked returns, incomplete implementations).

Key Invariants:
1. Baseline vs Post-Diff Differential Evaluation:
   We evaluate the gate before the agent's edit (baseline) and after the edit:
     baseline GREEN, post GREEN -> PASS
     baseline GREEN, post RED   -> FAIL_INTRODUCED_BY_DIFF (hard-block, must revert)
     baseline RED,   post RED   -> FAIL_PREEXISTING        (downgrade confidence, park)
     timeout / crash            -> FAIL_ENV                (environmental failure)
2. Non-Interactive Isolation:
   The command runs in a headless environment without TTY access to prevent hanging prompts.
3. Subprocess Timeout:
   Every execution is bounded by a strict wall-clock timeout.
"""

from __future__ import annotations

import dataclasses
import enum
import os
import shlex
import subprocess
import sys
from typing import Sequence

__all__ = ["GateVerdict", "GateExecutionResult", "run_gate_command", "evaluate_gate_diff"]


class GateVerdict(str, enum.Enum):
    PASS = "PASS"
    FAIL_INTRODUCED_BY_DIFF = "FAIL_INTRODUCED_BY_DIFF"
    FAIL_PREEXISTING = "FAIL_PREEXISTING"
    FAIL_ENV = "FAIL_ENV"


@dataclasses.dataclass(frozen=True)
class GateExecutionResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    is_success: bool


_NON_INTERACTIVE_ENV = {
    "CI": "1",
    "DEBIAN_FRONTEND": "noninteractive",
    "PIP_NO_INPUT": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "NO_COLOR": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}



def run_gate_command(
    command: Sequence[str] | str,
    cwd: str | None = None,
    timeout_seconds: int = 30,
) -> GateExecutionResult:
    """Run a deterministic command in a bounded environment."""
    env = dict(os.environ)
    env.update(_NON_INTERACTIVE_ENV)
    use_shell = isinstance(command, str)
    cmd_list = command.split() if isinstance(command, str) else list(command)
    try:
        proc = subprocess.run(
            command if use_shell else cmd_list,
            cwd=cwd,
            env=env,
            shell=use_shell,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return GateExecutionResult(
            command=cmd_list,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            timed_out=False,
            is_success=(proc.returncode == 0),
        )
    except subprocess.TimeoutExpired as exc:
        return GateExecutionResult(
            command=cmd_list,
            returncode=-1,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or "") + f"\nCommand timed out after {timeout_seconds} seconds",
            timed_out=True,
            is_success=False,
        )


def evaluate_gate_diff(
    baseline_result: GateExecutionResult,
    post_diff_result: GateExecutionResult,
) -> GateVerdict:
    """
    Differential evaluation of baseline vs post-diff execution results.
    """
    if post_diff_result.timed_out or baseline_result.timed_out:
        return GateVerdict.FAIL_ENV

    if baseline_result.is_success and post_diff_result.is_success:
        return GateVerdict.PASS
    elif baseline_result.is_success and not post_diff_result.is_success:
        return GateVerdict.FAIL_INTRODUCED_BY_DIFF
    elif not baseline_result.is_success and not post_diff_result.is_success:
        return GateVerdict.FAIL_PREEXISTING
    else:
        # Baseline failed but post succeeded: fixed a pre-existing issue!
        return GateVerdict.PASS
