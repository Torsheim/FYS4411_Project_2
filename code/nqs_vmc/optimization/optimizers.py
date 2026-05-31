"""Simple optimizers for RBM parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM

Array = np.ndarray
GradientDict = dict[str, Array]


def _zeros_like_parameters(model: GaussianBinaryRBM) -> GradientDict:
    return {
        "a": np.zeros_like(model.a),
        "b": np.zeros_like(model.b),
        "W": np.zeros_like(model.W),
    }


@dataclass
class SGD:
    """Plain stochastic gradient descent."""

    learning_rate: float = 0.01

    def step(self, model: GaussianBinaryRBM, gradients: GradientDict) -> None:
        model.add_parameter_update(gradients, scale=-self.learning_rate)


@dataclass
class Adam:
    """Adam optimizer for structured RBM parameters."""

    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    t: int = 0
    m: GradientDict | None = field(default=None, init=False)
    v: GradientDict | None = field(default=None, init=False)

    def step(self, model: GaussianBinaryRBM, gradients: GradientDict) -> None:
        if self.m is None or self.v is None:
            self.m = _zeros_like_parameters(model)
            self.v = _zeros_like_parameters(model)

        self.t += 1
        updates: GradientDict = {}
        for name, grad in gradients.items():
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            updates[name] = -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        model.add_parameter_update(updates, scale=1.0)


def create_optimizer(
    name: str,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
):
    """Factory used by scripts."""
    name = name.lower()
    if name == "sgd":
        return SGD(learning_rate=learning_rate)
    if name == "adam":
        return Adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    raise ValueError(f"Unknown optimizer '{name}'. Choose 'sgd' or 'adam'.")
