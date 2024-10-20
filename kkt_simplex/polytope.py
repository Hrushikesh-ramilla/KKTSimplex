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




def project_simplex_with_bounds(
    scores: np.ndarray,
    floor: float = 0.1,
    ceiling: float = 0.9,
) -> np.ndarray:
    """Project scores onto the bounded probability simplex."""
    x = _validate_vector("scores", scores)
    floor, ceiling = _validate_bounds(floor, ceiling)
    n = x.size
    upper = effective_ceiling(n, ceiling)
    if x.min() >= floor - _TOL and x.max() <= upper + _TOL and abs(x.sum() - 1.0) <= _TOL:
        return x.copy()
    breaks = np.unique(np.concatenate((x - upper, x - floor)))
    sums = np.clip(x[None, :] - breaks[:, None], floor, upper).sum(axis=1)
    crossed = np.nonzero(sums <= 1.0 + _TOL)[0]
    j = int(crossed[0]) if crossed.size else int(breaks.size - 1)
    if j == 0:
        theta = float(breaks[0])
    else:
        mid = 0.5 * float(breaks[j - 1] + breaks[j])
        delta = x - mid
        free = (delta >= floor) & (delta <= upper)
        capped = delta > upper
        floored = delta < floor
        m = int(free.sum())
        theta = mid if m == 0 else (float(x[free].sum()) + upper * int(capped.sum()) + floor * int(floored.sum()) - 1.0) / m
    return np.clip(x - theta, floor, upper)


def assert_simplex_invariants(w: np.ndarray, n: int, floor: float = 0.1, ceiling: float = 0.9, tol: float = 1e-9) -> None:
    """Formal verification of all mathematical simplex invariants."""
    upper = effective_ceiling(n, ceiling)
    assert w.shape == (n,), f"Expected shape ({n},), got {w.shape}"
    assert np.all(np.isfinite(w)), f"Found non-finite values in {w}"
    assert w.min() >= floor - tol, f"Floor invariant violated: min={w.min()} < floor={floor}"
    assert w.max() <= upper + tol, f"Ceiling invariant violated: max={w.max()} > upper={upper}"
    assert abs(w.sum() - 1.0) <= tol, f"Sum invariant violated: sum={w.sum()} != 1.0"
