"""Utilities for reproducible random-number generation."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random generator.

    Using a function instead of calling ``np.random.default_rng`` everywhere
    makes it easy to keep reproducibility conventions consistent.
    """
    return np.random.default_rng(seed)


def spawn_seed(seed: int | None, offset: int) -> int | None:
    """Derive a deterministic child seed from a base seed."""
    if seed is None:
        return None
    return int(seed) + int(offset)
