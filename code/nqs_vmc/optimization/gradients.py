"""Gradient estimators for VMC parameter optimization."""

from __future__ import annotations

import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM

Array = np.ndarray
GradientDict = dict[str, Array]


def collect_log_derivatives(model: GaussianBinaryRBM, positions: Array) -> GradientDict:
    """Evaluate O_k = d log(Psi) / d alpha_k for all samples."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)

    n = positions.shape[0]
    Oa = np.empty((n, model.num_visible), dtype=float)
    Ob = np.empty((n, model.num_hidden), dtype=float)
    OW = np.empty((n, model.num_visible, model.num_hidden), dtype=float)

    for i, x in enumerate(positions):
        derivatives = model.parameter_derivatives(x)
        Oa[i] = derivatives["a"]
        Ob[i] = derivatives["b"]
        OW[i] = derivatives["W"]

    return {"a": Oa, "b": Ob, "W": OW}


def energy_gradient(model: GaussianBinaryRBM, positions: Array, local_energies: Array) -> GradientDict:
    """Return stochastic gradient of <E> with respect to RBM parameters.

    The estimator is

        G_k = 2 ( <E_L O_k> - <E_L><O_k> ),

    where O_k = d log(Psi) / d alpha_k.
    """
    energies = np.asarray(local_energies, dtype=float).reshape(-1)
    if energies.size == 0:
        raise ValueError("Need at least one local energy to estimate gradients.")
    derivatives = collect_log_derivatives(model, positions)
    e_mean = np.mean(energies)

    gradients: GradientDict = {}
    for name, values in derivatives.items():
        shape = (energies.size,) + (1,) * (values.ndim - 1)
        weighted = np.mean(energies.reshape(shape) * values, axis=0)
        average = np.mean(values, axis=0)
        gradients[name] = 2.0 * (weighted - e_mean * average)
    return gradients


def gradient_norm(gradients: GradientDict) -> float:
    """Euclidean norm of a structured gradient dictionary."""
    return float(np.sqrt(sum(np.sum(g * g) for g in gradients.values())))
