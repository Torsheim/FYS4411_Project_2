"""Basic statistical error estimates."""

from __future__ import annotations

import math

import numpy as np

Array = np.ndarray


def standard_error(data: Array) -> float:
    data = np.asarray(data, dtype=float).reshape(-1)
    if data.size < 2:
        return float("nan")
    return float(np.std(data, ddof=1) / math.sqrt(data.size))


def confidence_interval(data: Array, z: float = 1.96) -> tuple[float, float]:
    """Approximate normal confidence interval for the mean."""
    data = np.asarray(data, dtype=float).reshape(-1)
    mean = float(np.mean(data))
    err = standard_error(data)
    return mean - z * err, mean + z * err
