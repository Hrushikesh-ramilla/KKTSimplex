"""
Bounded Probability Simplex Projection via Exact Water-Filling.

Mathematical Formulation:
Given an arbitrary score vector x in R^N and bounds [floor, ceiling] with 0 <= floor < ceiling <= 1,
solve the convex optimization problem:
    min_{w in W} (1/2) ||w - x||_2^2
subject to:
    w in W = { w in R^N : floor <= w_i <= effective_ceiling, sum_{i=1}^N w_i = 1 }
where effective_ceiling = max(ceiling, 1/N).

The solution has the closed-form structure:
    w_i(theta) = clip(x_i - theta, floor, effective_ceiling)
where theta is the unique Lagrange multiplier satisfying:
    phi(theta) = sum_{i=1}^N clip(x_i - theta, floor, effective_ceiling) = 1.

Since phi(theta) is piecewise linear, continuous, and strictly decreasing over the active range,
the exact root is found by evaluating phi at all breakpoints {x_i - effective_ceiling, x_i - floor},
identifying the bounding interval, and computing theta analytically in O(N log N) time without
numerical bisection error.
"""

from __future__ import annotations

import numpy as np

__all__ = ["project_simplex_with_bounds", "effective_ceiling", "assert_simplex_invariants"]

_TOL = 1e-12


def effective_ceiling(n: int, ceiling: float) -> float:
    """Return effective ceiling relaxed to uniform share if n * ceiling < 1."""
    return max(ceiling, 1.0 / n)


def _validate_vector(name: str, x: np.ndarray, n: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} length mismatch: expected {n}, got {arr.size}")
    return arr


def _validate_bounds(floor: float, ceiling: float) -> tuple[float, float]:
    floor = float(floor)
    ceiling = float(ceiling)
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must be in [0, 1), got {floor}")
    if not 0.0 < ceiling <= 1.0:
        raise ValueError(f"ceiling must be in (0, 1], got {ceiling}")
    if floor > ceiling:
        raise ValueError(f"floor {floor} exceeds ceiling {ceiling}")
    return floor, ceiling


