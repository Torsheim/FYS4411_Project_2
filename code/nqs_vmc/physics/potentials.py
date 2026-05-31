"""Potential-energy functions for the project Hamiltonian."""

from __future__ import annotations

import itertools

import numpy as np

Array = np.ndarray


def reshape_positions(x: Array, num_particles: int, dimensions: int) -> Array:
    """Return positions with shape ``(num_particles, dimensions)``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    expected = int(num_particles) * int(dimensions)
    if x.size != expected:
        raise ValueError(f"Expected {expected} coordinates, got {x.size}.")
    return x.reshape(int(num_particles), int(dimensions))


def harmonic_oscillator_potential(x: Array, omega: float = 1.0) -> float:
    """Return 0.5 * omega^2 * sum_i r_i^2."""
    x = np.asarray(x, dtype=float)
    return float(0.5 * omega * omega * np.sum(x * x))


def pair_distances(x: Array, num_particles: int, dimensions: int) -> Array:
    """Return all pair distances r_ij."""
    positions = reshape_positions(x, num_particles, dimensions)
    distances: list[float] = []
    for i, j in itertools.combinations(range(num_particles), 2):
        distances.append(float(np.linalg.norm(positions[i] - positions[j])))
    return np.asarray(distances, dtype=float)


def inverse_distance_interaction(
    x: Array,
    num_particles: int,
    dimensions: int,
    epsilon: float = 1.0e-12,
) -> float:
    """Return the Coulomb-like repulsive interaction sum_{i<j} 1/r_ij."""
    if num_particles < 2:
        return 0.0
    distances = pair_distances(x, num_particles, dimensions)
    if np.any(distances <= epsilon):
        return float(np.inf)
    return float(np.sum(1.0 / distances))
