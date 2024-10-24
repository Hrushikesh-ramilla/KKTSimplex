"""
KKTSimplex: Verifiable Epistemic Calibration & Autonomous Execution Governance.
"""

from .polytope import project_simplex_with_bounds, effective_ceiling, assert_simplex_invariants

__version__ = "0.1.0"
__all__ = ["project_simplex_with_bounds", "effective_ceiling", "assert_simplex_invariants"]
