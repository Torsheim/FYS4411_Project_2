"""Training loop for variational Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.optimization.gradients import energy_gradient, gradient_norm

Array = np.ndarray


@dataclass
class VMCTrainer:
    """Optimize an RBM wave function by minimizing sampled local energy."""

    model: GaussianBinaryRBM
    sampler: Any
    optimizer: Any

    def train(
        self,
        n_iterations: int,
        n_samples: int,
        n_burn_in: int = 0,
        thin: int = 1,
        initial_position: Array | None = None,
        verbose: bool = False,
    ) -> tuple[list[dict[str, float]], Array, Array]:
        """Run VMC optimization.

        Returns
        -------
        history:
            List of dictionaries, one row per optimization iteration.
        final_position:
            Last Markov-chain position.
        final_local_energies:
            Local-energy samples from the last iteration. These are useful for
            blocking analysis.
        """
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")

        x = initial_position
        history: list[dict[str, float]] = []
        final_energies = np.empty(0)

        for iteration in range(1, n_iterations + 1):
            result = self.sampler.sample(
                n_samples=n_samples,
                n_burn_in=n_burn_in if iteration == 1 else 0,
                initial_position=x,
                thin=thin,
            )
            x = result.final_position
            final_energies = result.local_energies.copy()

            gradients = energy_gradient(self.model, result.positions, result.local_energies)
            gnorm = gradient_norm(gradients)
            self.optimizer.step(self.model, gradients)

            row = {
                "iteration": float(iteration),
                "energy_mean": result.energy_mean,
                "energy_std": result.energy_std,
                "energy_error": result.energy_error,
                "acceptance_rate": result.acceptance_rate,
                "gradient_norm": gnorm,
            }
            history.append(row)

            if verbose and (iteration == 1 or iteration % max(1, n_iterations // 10) == 0):
                print(
                    f"iter={iteration:5d} "
                    f"E={row['energy_mean']:.8f} ± {row['energy_error']:.2e} "
                    f"acc={row['acceptance_rate']:.3f} |g|={gnorm:.3e}"
                )

        assert x is not None
        return history, np.asarray(x, dtype=float), final_energies
