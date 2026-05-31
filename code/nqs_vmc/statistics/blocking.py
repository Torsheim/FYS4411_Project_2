"""Blocking analysis for correlated Monte Carlo data."""

from __future__ import annotations

import math

import numpy as np

Array = np.ndarray


def _crop_to_power_of_two(data: Array) -> Array:
    data = np.asarray(data, dtype=float).reshape(-1)
    if data.size < 2:
        raise ValueError("Blocking analysis needs at least two data points.")
    power = 2 ** int(math.floor(math.log2(data.size)))
    return data[:power].copy()


def blocking_analysis(data: Array) -> list[dict[str, float]]:
    """Return blocking statistics for all possible blocking levels.

    Each row contains the block level, number of blocked data points, block
    size, mean, variance of the blocked data, and estimated standard error of
    the mean at that blocking level.
    """
    x = _crop_to_power_of_two(data)
    rows: list[dict[str, float]] = []
    block_size = 1

    while x.size >= 2:
        variance = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
        error = math.sqrt(variance / x.size) if x.size > 1 else float("nan")
        rows.append(
            {
                "level": float(len(rows)),
                "n_blocks": float(x.size),
                "block_size": float(block_size),
                "mean": float(np.mean(x)),
                "variance": variance,
                "error": error,
            }
        )
        x = 0.5 * (x[0::2] + x[1::2])
        block_size *= 2

    return rows


def blocking_error(data: Array, method: str = "max") -> float:
    """Return one conservative error estimate from blocking analysis.

    ``method='max'`` returns the largest finite blocking error. This is a
    conservative default for project work. ``method='last'`` returns the final
    finite blocking level.
    """
    rows = blocking_analysis(data)
    errors = np.asarray([row["error"] for row in rows], dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return float("nan")
    if method == "max":
        return float(np.max(errors))
    if method == "last":
        return float(errors[-1])
    raise ValueError("method must be 'max' or 'last'.")
