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

