"""Brute-force Metropolis sampler for |Psi|^2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.physics.hamiltonian import HarmonicOscillatorHamiltonian
from nqs_vmc.random_utils import make_rng
from .sampler_result import SamplerResult

Array = np.ndarray


@dataclass
class MetropolisSampler:
    model: GaussianBinaryRBM
    hamiltonian: HarmonicOscillatorHamiltonian
    step_size: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")
        if self.model.num_visible != self.hamiltonian.num_visible:
            raise ValueError("Model and Hamiltonian disagree about number of coordinates.")
        self.rng = make_rng(self.seed)

    def _initial_position(self, initial_position: Array | None) -> Array:
        if initial_position is None:
            return self.rng.normal(scale=self.model.sigma, size=self.model.num_visible)
        x = np.asarray(initial_position, dtype=float).reshape(-1)
        if x.size != self.model.num_visible:
            raise ValueError(f"Expected {self.model.num_visible} initial coordinates, got {x.size}.")
        return x.copy()

    def sample(
        self,
        n_samples: int,
        n_burn_in: int = 0,
        initial_position: Array | None = None,
        thin: int = 1,
    ) -> SamplerResult:
        """Generate samples from |Psi|^2 using uniform proposals."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if n_burn_in < 0:
            raise ValueError("n_burn_in cannot be negative.")
        if thin <= 0:
            raise ValueError("thin must be positive.")

        x = self._initial_position(initial_position)
        log_psi_x = self.model.log_psi(x)
        positions = np.empty((n_samples, self.model.num_visible), dtype=float)
        energies = np.empty(n_samples, dtype=float)

        accepted = 0
        total = 0
        stored = 0
        total_steps = n_burn_in + n_samples * thin

        for step in range(total_steps):
            proposal = x + self.rng.uniform(-self.step_size, self.step_size, size=x.size)
            log_psi_new = self.model.log_psi(proposal)
            log_ratio = 2.0 * (log_psi_new - log_psi_x)
            if np.log(self.rng.random()) < min(0.0, log_ratio):
                x = proposal
                log_psi_x = log_psi_new
                accepted += 1
            total += 1

            if step >= n_burn_in and (step - n_burn_in) % thin == 0:
                positions[stored] = x
                energies[stored] = self.hamiltonian.local_energy(self.model, x)
                stored += 1

        return SamplerResult(
            positions=positions,
            local_energies=energies,
            accepted_moves=accepted,
            total_moves=total,
            final_position=x.copy(),
        )
