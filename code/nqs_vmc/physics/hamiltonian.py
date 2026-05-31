"""Hamiltonian object for the oscillator systems in Project 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from .local_energy import local_energy
from .potentials import harmonic_oscillator_potential, inverse_distance_interaction

Array = np.ndarray


@dataclass(frozen=True)
class HarmonicOscillatorHamiltonian:
    """Harmonic oscillator Hamiltonian with optional pair interaction."""

    num_particles: int
    dimensions: int
    omega: float = 1.0
    include_interaction: bool = False
    coulomb_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive.")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        if self.omega <= 0:
            raise ValueError("omega must be positive.")

    @property
    def num_visible(self) -> int:
        return self.num_particles * self.dimensions

    def potential_energy(self, x: Array) -> float:
        potential = harmonic_oscillator_potential(x, omega=self.omega)
        if self.include_interaction:
            potential += inverse_distance_interaction(
                x,
                num_particles=self.num_particles,
                dimensions=self.dimensions,
                epsilon=self.coulomb_epsilon,
            )
        return float(potential)

    def local_energy(self, model: GaussianBinaryRBM, x: Array) -> float:
        return local_energy(
            model=model,
            x=x,
            num_particles=self.num_particles,
            dimensions=self.dimensions,
            omega=self.omega,
            include_interaction=self.include_interaction,
            coulomb_epsilon=self.coulomb_epsilon,
        )

    def exact_non_interacting_energy(self) -> float:
        """Exact energy for the non-interacting ground state."""
        return 0.5 * self.num_particles * self.dimensions * self.omega

    def exact_interacting_2p2d_energy(self) -> float | None:
        """Known benchmark used in the project: two particles in 2D at omega=1."""
        if self.num_particles == 2 and self.dimensions == 2 and abs(self.omega - 1.0) < 1.0e-14:
            return 3.0
        return None
