"""Analytical local energy for the Gaussian-binary RBM ansatz."""

from __future__ import annotations

import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from .potentials import harmonic_oscillator_potential, inverse_distance_interaction

Array = np.ndarray


def kinetic_energy(model: GaussianBinaryRBM, x: Array) -> float:
    """Return the kinetic local energy ``-0.5 * nabla^2 Psi / Psi``."""
    return float(-0.5 * model.laplacian_over_psi(x))


def local_energy(
    model: GaussianBinaryRBM,
    x: Array,
    num_particles: int,
    dimensions: int,
    omega: float = 1.0,
    include_interaction: bool = False,
    coulomb_epsilon: float = 1.0e-12,
) -> float:
    """Return the local energy for the oscillator Hamiltonian.

    Parameters match the project Hamiltonian

        H = sum_i [-1/2 nabla_i^2 + 1/2 omega^2 r_i^2] + sum_{i<j} 1/r_ij.
    """
    kinetic = kinetic_energy(model, x)
    potential = harmonic_oscillator_potential(x, omega=omega)
    if include_interaction:
        potential += inverse_distance_interaction(
            x,
            num_particles=num_particles,
            dimensions=dimensions,
            epsilon=coulomb_epsilon,
        )
    return float(kinetic + potential)
