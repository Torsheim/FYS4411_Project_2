import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM


def test_rbm_shapes_and_finite_logpsi():
    model = GaussianBinaryRBM.random_initialization(4, 3, seed=1)
    x = np.array([0.1, -0.2, 0.3, -0.4])
    assert np.isfinite(model.log_psi(x))
    assert model.hidden_probabilities(x).shape == (3,)
    assert model.gradient_log_psi_position(x).shape == (4,)
    assert model.second_derivative_log_psi_position(x).shape == (4,)


def test_parameter_vector_roundtrip():
    model = GaussianBinaryRBM.random_initialization(2, 2, seed=2)
    vector = model.parameter_vector()
    clone = model.copy()
    clone.set_parameter_vector(vector)
    np.testing.assert_allclose(clone.parameter_vector(), vector)
