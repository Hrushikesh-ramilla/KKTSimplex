"""
Mathematical invariant tests for bounded simplex water-filling projection.
"""

from __future__ import annotations

import numpy as np
import pytest

from kkt_simplex.polytope import (
    project_simplex_with_bounds,
    effective_ceiling,
    assert_simplex_invariants,
)


def bisection_reference(x: np.ndarray, floor: float = 0.1, ceiling: float = 0.9) -> np.ndarray:
    """Reference numerical bisection solver for convex projection onto {w: floor <= w <= upper, sum(w) = 1}."""
    upper = max(ceiling, 1.0 / x.size)
    lo, hi = x.min() - upper - 1.0, x.max() - floor + 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if np.clip(x - mid, floor, upper).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(x - 0.5 * (lo + hi), floor, upper)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_polytope_invariants_random_distribution(n: int) -> None:
    rng = np.random.default_rng(1017 + n)
    for _ in range(1_000):
        # Spanning multiple orders of magnitude
        raw = rng.normal(0.0, 10.0, n) * (10.0 ** rng.integers(-3, 4))
        w = project_simplex_with_bounds(raw, floor=0.1, ceiling=0.85)
        assert_simplex_invariants(w, n, floor=0.1, ceiling=0.85)


def test_closed_form_matches_bisection_reference() -> None:
    rng = np.random.default_rng(42)
    for _ in range(250):
        x = rng.normal(0.0, 5.0, 4)
        ours = project_simplex_with_bounds(x, floor=0.1, ceiling=0.8)
        ref = bisection_reference(x, floor=0.1, ceiling=0.8)
        assert np.allclose(ours, ref, atol=1e-9, rtol=0), f"Mismatch: ours={ours}, ref={ref}"


def test_projection_idempotence() -> None:
    rng = np.random.default_rng(999)
    for _ in range(500):
        x = rng.normal(0.0, 10.0, 3)
        w1 = project_simplex_with_bounds(x, floor=0.1, ceiling=0.9)
        w2 = project_simplex_with_bounds(w1, floor=0.1, ceiling=0.9)
        assert np.allclose(w1, w2, atol=1e-12, rtol=0)


def test_exact_identity_on_feasible_vector() -> None:
    feasible = np.array([0.7, 0.15, 0.15])
    w = project_simplex_with_bounds(feasible, floor=0.1, ceiling=0.9)
    assert np.allclose(feasible, w, atol=1e-14, rtol=0)


def test_infeasible_floor_raises() -> None:
    with pytest.raises(ValueError, match="Infeasible polytope"):
        # 3 dimensions * 0.4 = 1.2 > 1.0
        project_simplex_with_bounds(np.ones(3), floor=0.4)


def test_invalid_bounds_raise() -> None:
    with pytest.raises(ValueError):
        project_simplex_with_bounds(np.ones(3), floor=-0.1)
    with pytest.raises(ValueError):
        project_simplex_with_bounds(np.ones(3), floor=0.8, ceiling=0.2)
    with pytest.raises(ValueError):
        project_simplex_with_bounds(np.array([np.nan, 1.0]))
