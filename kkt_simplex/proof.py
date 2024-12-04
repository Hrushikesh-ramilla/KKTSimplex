"""
Formal Mathematical and Empirical Proof Certification Engine for KKTSimplex.

Every defense against an adversarial attack must produce a verifiable Proof Certificate:
1. Mathematical Proof Certificate (KKT Optimality & Convex Invariants):
   Given input x and projected vector w:
   Proves Karush-Kuhn-Tucker (KKT) optimality for:
     min (1/2) ||w - x||_2^2 subject to sum(w) = 1, floor <= w_i <= upper.
   Certifies:
     - Primal feasibility: min(w) >= floor - 1e-9, max(w) <= upper + 1e-9, |sum(w) - 1| <= 1e-9
     - Stationarity: w_i - x_i + theta - lambda_low_i + lambda_high_i = 0
     - Complementary slackness: lambda_low_i * (w_i - floor) = 0, lambda_high_i * (upper - w_i) = 0
     - Dual feasibility: lambda_low_i >= 0, lambda_high_i >= 0

2. Empirical Execution Proof Certificate:
   Given an adversarial claim and candidate code:
   Executes the code and counter-test in an isolated deterministic gate.
   Proves that:
     - All assertion boundaries hold
     - Exit code is 0
     - Auditor's alleged counter-example does NOT fail on the implementation

3. Bayesian Odds Ratio & Audit Proof Certificate:
   Computes exact posterior odds ratio:
     Odds(Verified Solution : Rhetorical Critique) >= (1 - floor) / floor
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional
import numpy as np

from .polytope import effective_ceiling
from .gates import run_gate_command, GateExecutionResult

__all__ = [
    "KKTProofCertificate",
    "ExecutionProofCertificate",
    "AuditProofCertificate",
    "verify_kkt_optimality",
    "generate_execution_proof",
    "adjudicate_with_proof",
]


@dataclasses.dataclass(frozen=True)
class KKTProofCertificate:
    is_valid: bool
    primal_feasibility_error: float
    stationarity_error: float
    complementarity_error: float
    dual_feasibility_passed: bool
    theta: float
    n_dimensions: int
    proof_summary: str

