import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.sampling.quantum_force import quantum_force


def test_quantum_force_is_twice_log_gradient():
    model = GaussianBinaryRBM.random_initialization(2, 2, seed=5)
    x = np.array([0.1, -0.2])
    np.testing.assert_allclose(quantum_force(model, x), 2.0 * model.gradient_log_psi_position(x))
