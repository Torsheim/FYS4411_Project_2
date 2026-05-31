"""Observable and summary-statistic helpers."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def mean(values: Array) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def variance(values: Array, ddof: int = 1) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= ddof:
        return 0.0
    return float(np.var(values, ddof=ddof))


def standard_error(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def autocorrelation(values: Array, max_lag: int | None = None) -> Array:
    """Estimate normalized autocorrelation up to ``max_lag``."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x - np.mean(x)
    n = x.size
    if max_lag is None:
        max_lag = min(n - 1, 1000)
    corr = np.empty(max_lag + 1)
    denom = np.dot(x, x)
    if denom == 0:
        corr.fill(0.0)
        corr[0] = 1.0
        return corr
    for lag in range(max_lag + 1):
        corr[lag] = np.dot(x[: n - lag], x[lag:]) / denom
    return corr
