"""Optional neural-Jastrow ansatz placeholder.

Project 2f is optional.  This module gives a small, real-valued neural network
that can be used as a multiplicative correlation factor on top of a harmonic
oscillator one-body factor.  It is intentionally separate from the main RBM
solution so that Parts b--e remain focused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class NeuralJastrow:
    """One-hidden-layer neural correlation factor.

    The log wave function is

        log Psi(X) = -0.5 * omega * |X|^2 + sum_j v_j tanh(c_j + X @ U_j).

    This file is not used by the main RBM scripts, but it provides a starting
    point for the optional neural-network replacement task.
    """

    num_visible: int
    num_hidden: int
    omega: float = 1.0
    U: Array | None = None
    c: Array | None = None
    v: Array | None = None

    def __post_init__(self) -> None:
        if self.U is None:
            self.U = np.zeros((self.num_visible, self.num_hidden))
        if self.c is None:
            self.c = np.zeros(self.num_hidden)
        if self.v is None:
            self.v = np.zeros(self.num_hidden)
        self.U = np.asarray(self.U, dtype=float).reshape(self.num_visible, self.num_hidden)
        self.c = np.asarray(self.c, dtype=float).reshape(self.num_hidden)
        self.v = np.asarray(self.v, dtype=float).reshape(self.num_hidden)

    @classmethod
    def random_initialization(
        cls,
        num_visible: int,
        num_hidden: int,
        omega: float = 1.0,
        scale: float = 0.01,
        seed: int | None = None,
    ) -> "NeuralJastrow":
        rng = np.random.default_rng(seed)
        return cls(
            num_visible=num_visible,
            num_hidden=num_hidden,
            omega=omega,
            U=scale * rng.normal(size=(num_visible, num_hidden)),
            c=scale * rng.normal(size=num_hidden),
            v=scale * rng.normal(size=num_hidden),
        )

    def _x(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.num_visible:
            raise ValueError(f"Expected {self.num_visible} visible values, got {x.size}.")
        return x

    def log_psi(self, x: Array) -> float:
        x = self._x(x)
        z = self.c + x @ self.U
        return float(-0.5 * self.omega * np.sum(x * x) + np.sum(self.v * np.tanh(z)))

    def psi(self, x: Array) -> float:
        return float(np.exp(self.log_psi(x)))
