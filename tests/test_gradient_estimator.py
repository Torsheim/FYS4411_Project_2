import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.optimization.gradients import energy_gradient


def test_energy_gradient_shapes():
    model = GaussianBinaryRBM.random_initialization(2, 3, seed=10)
    positions = np.random.default_rng(11).normal(size=(5, 2))
    energies = np.linspace(0.1, 0.5, 5)
    grad = energy_gradient(model, positions, energies)
    assert grad["a"].shape == model.a.shape
    assert grad["b"].shape == model.b.shape
    assert grad["W"].shape == model.W.shape
