"""Gaussian-binary restricted Boltzmann machine wave function.

The visible variables are continuous particle coordinates and the hidden
variables are binary.  The unnormalized wave function is

    Psi(X) = exp[-sum_i (X_i-a_i)^2/(2 sigma^2)]
             prod_j [1 + exp(b_j + sum_i X_i W_ij / sigma^2)].

The normalization constant is irrelevant for VMC because all ratios and
log-derivatives cancel it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray


def _sigmoid(x: Array) -> Array:
    """Numerically stable logistic sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


@dataclass
class GaussianBinaryRBM:
    """Gaussian-binary RBM used as a real-valued NQS ansatz.

    Parameters
    ----------
    num_visible:
        Number of visible variables. For this project this is ``P * D``.
    num_hidden:
        Number of hidden binary nodes.
    sigma:
        Width of the Gaussian visible-node factor. The project formulas assume
        a common value of sigma for all visible coordinates.
    a, b, W:
        Visible biases, hidden biases and visible-hidden weights.
    """

    num_visible: int
    num_hidden: int
    sigma: float = 1.0
    a: Array | None = None
    b: Array | None = None
    W: Array | None = None

    def __post_init__(self) -> None:
        if self.num_visible <= 0:
            raise ValueError("num_visible must be positive.")
        if self.num_hidden <= 0:
            raise ValueError("num_hidden must be positive.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

        if self.a is None:
            self.a = np.zeros(self.num_visible, dtype=float)
        else:
            self.a = np.asarray(self.a, dtype=float).reshape(self.num_visible)

        if self.b is None:
            self.b = np.zeros(self.num_hidden, dtype=float)
        else:
            self.b = np.asarray(self.b, dtype=float).reshape(self.num_hidden)

        if self.W is None:
            self.W = np.zeros((self.num_visible, self.num_hidden), dtype=float)
        else:
            self.W = np.asarray(self.W, dtype=float).reshape(self.num_visible, self.num_hidden)

    @classmethod
    def random_initialization(
        cls,
        num_visible: int,
        num_hidden: int,
        sigma: float = 1.0,
        scale: float = 0.01,
        seed: int | None = None,
    ) -> "GaussianBinaryRBM":
        """Create an RBM with small random parameters."""
        rng = np.random.default_rng(seed)
        return cls(
            num_visible=num_visible,
            num_hidden=num_hidden,
            sigma=sigma,
            a=scale * rng.normal(size=num_visible),
            b=scale * rng.normal(size=num_hidden),
            W=scale * rng.normal(size=(num_visible, num_hidden)),
        )

    @property
    def sigma2(self) -> float:
        return self.sigma * self.sigma

    @property
    def sigma4(self) -> float:
        return self.sigma2 * self.sigma2

    @property
    def num_parameters(self) -> int:
        return self.num_visible + self.num_hidden + self.num_visible * self.num_hidden

    def _as_flat_position(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.num_visible:
            raise ValueError(f"Expected {self.num_visible} visible values, got {x.size}.")
        return x

    def theta(self, x: Array) -> Array:
        """Hidden-node pre-activation vector theta_j."""
        x = self._as_flat_position(x)
        return self.b + x @ self.W / self.sigma2

    def hidden_probabilities(self, x: Array) -> Array:
        """Return sigmoid(theta_j), the conditional hidden probabilities."""
        return _sigmoid(self.theta(x))

    def log_psi(self, x: Array) -> float:
        """Return log of the unnormalized RBM wave function."""
        x = self._as_flat_position(x)
        gaussian = -0.5 * np.sum((x - self.a) ** 2) / self.sigma2
        hidden = np.sum(np.logaddexp(0.0, self.theta(x)))
        return float(gaussian + hidden)

    def log_psi_batch(self, positions: Array) -> Array:
        """Vectorized log wave function for an array of positions."""
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.shape[1] != self.num_visible:
            raise ValueError(f"Expected shape (n,{self.num_visible}), got {positions.shape}.")
        gaussian = -0.5 * np.sum((positions - self.a) ** 2, axis=1) / self.sigma2
        theta = self.b[None, :] + positions @ self.W / self.sigma2
        hidden = np.sum(np.logaddexp(0.0, theta), axis=1)
        return gaussian + hidden

    def psi(self, x: Array) -> float:
        """Return the unnormalized wave function."""
        return float(np.exp(self.log_psi(x)))

    def psi_ratio(self, x_new: Array, x_old: Array) -> float:
        """Return Psi(x_new) / Psi(x_old)."""
        return float(np.exp(self.log_psi(x_new) - self.log_psi(x_old)))

    def probability_ratio(self, x_new: Array, x_old: Array) -> float:
        """Return |Psi(x_new)|^2 / |Psi(x_old)|^2."""
        exponent = 2.0 * (self.log_psi(x_new) - self.log_psi(x_old))
        return float(np.exp(min(700.0, exponent)))

    def gradient_log_psi_position(self, x: Array) -> Array:
        """Gradient of log(Psi) with respect to visible coordinates."""
        x = self._as_flat_position(x)
        q = self.hidden_probabilities(x)
        return -(x - self.a) / self.sigma2 + (self.W @ q) / self.sigma2

    def second_derivative_log_psi_position(self, x: Array) -> Array:
        """Coordinate-wise second derivatives of log(Psi).

        The returned vector contains d^2 log(Psi) / dX_i^2 for every visible
        coordinate i.
        """
        x = self._as_flat_position(x)
        q = self.hidden_probabilities(x)
        hidden_curvature = (self.W ** 2) @ (q * (1.0 - q)) / self.sigma4
        return -np.ones(self.num_visible) / self.sigma2 + hidden_curvature

    def laplacian_log_psi_position(self, x: Array) -> float:
        """Laplacian of log(Psi)."""
        return float(np.sum(self.second_derivative_log_psi_position(x)))

    def laplacian_over_psi(self, x: Array) -> float:
        """Return (nabla^2 Psi) / Psi.

        Since nabla^2 Psi / Psi = nabla^2 log(Psi) + |nabla log(Psi)|^2.
        """
        grad = self.gradient_log_psi_position(x)
        second = self.second_derivative_log_psi_position(x)
        return float(np.sum(second + grad * grad))

    def parameter_derivatives(self, x: Array) -> dict[str, Array]:
        """Return derivatives of log(Psi) with respect to RBM parameters.

        These are the observables O_k = (1/Psi) dPsi/dalpha_k = dlog(Psi)/dalpha_k
        used in the VMC gradient estimator.
        """
        x = self._as_flat_position(x)
        q = self.hidden_probabilities(x)
        return {
            "a": (x - self.a) / self.sigma2,
            "b": q.copy(),
            "W": np.outer(x / self.sigma2, q),
        }

    def parameter_vector(self) -> Array:
        """Return all variational parameters as one flat vector."""
        return np.concatenate([self.a.ravel(), self.b.ravel(), self.W.ravel()])

    def set_parameter_vector(self, vector: Array) -> None:
        """Set all variational parameters from one flat vector."""
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if vector.size != self.num_parameters:
            raise ValueError(f"Expected {self.num_parameters} parameters, got {vector.size}.")
        n_a = self.num_visible
        n_b = self.num_hidden
        self.a = vector[:n_a].copy()
        self.b = vector[n_a:n_a + n_b].copy()
        self.W = vector[n_a + n_b:].reshape(self.num_visible, self.num_hidden).copy()

    def add_parameter_update(self, update: dict[str, Array], scale: float = 1.0) -> None:
        """Add a structured update to parameters."""
        self.a = self.a + scale * np.asarray(update["a"], dtype=float)
        self.b = self.b + scale * np.asarray(update["b"], dtype=float)
        self.W = self.W + scale * np.asarray(update["W"], dtype=float)

    def copy(self) -> "GaussianBinaryRBM":
        """Return a deep copy of the model."""
        return GaussianBinaryRBM(
            num_visible=self.num_visible,
            num_hidden=self.num_hidden,
            sigma=self.sigma,
            a=self.a.copy(),
            b=self.b.copy(),
            W=self.W.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_visible": self.num_visible,
            "num_hidden": self.num_hidden,
            "sigma": self.sigma,
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "W": self.W.tolist(),
        }
