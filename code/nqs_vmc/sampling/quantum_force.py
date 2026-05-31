"""Quantum force for importance sampling."""

from __future__ import annotations

import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM

Array = np.ndarray


def quantum_force(model: GaussianBinaryRBM, x: Array) -> Array:
    """Return the quantum force F = 2 grad log(Psi)."""
    return 2.0 * model.gradient_log_psi_position(x)


def drift_step(model: GaussianBinaryRBM, x: Array, time_step: float, diffusion: float = 0.5) -> Array:
    """Return the deterministic Langevin drift D * F * dt."""
    return diffusion * quantum_force(model, x) * time_step
