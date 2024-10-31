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


def _decode_output(val: str | bytes | None) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _parse_command_args(command: Sequence[str] | str) -> list[str]:
    """Tokenize a command securely across platforms without shell execution or backslash corruption."""
    if not isinstance(command, str):
        return list(command)
    if sys.platform == "win32":
        raw_args = shlex.split(command, posix=False)
        return [
            a[1:-1] if (len(a) >= 2 and a[0] == a[-1] and a[0] in ('"', "'")) else a
            for a in raw_args
        ]
    return shlex.split(command, posix=True)


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
    """Run a deterministic command in a non-interactive, bounded environment."""
    env = dict(os.environ)
    env.update(_NON_INTERACTIVE_ENV)

    cmd_args = _parse_command_args(command)

    try:
        proc = subprocess.run(
            cmd_args,
            cwd=cwd,
            env=env,
            shell=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
        )
        return GateExecutionResult(
            command=cmd_args,
            returncode=proc.returncode,
            stdout=_decode_output(proc.stdout),
            stderr=_decode_output(proc.stderr),
            timed_out=False,
            is_success=(proc.returncode == 0),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        return GateExecutionResult(
            command=cmd_args,
            returncode=-1,
            stdout=stdout,
            stderr=stderr + f"\nCommand timed out after {timeout_seconds} seconds",
            timed_out=True,
            is_success=False,
        )
    except Exception as exc:
        return GateExecutionResult(
            command=cmd_args,
            returncode=-2,
            stdout="",
            stderr=str(exc),
            timed_out=False,
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
