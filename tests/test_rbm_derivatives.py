import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM


def central_difference(f, x, i, h=1e-6):
    xp = x.copy(); xm = x.copy()
    xp[i] += h; xm[i] -= h
    return (f(xp) - f(xm)) / (2 * h)


def test_position_gradient_matches_finite_difference():
    model = GaussianBinaryRBM.random_initialization(3, 4, seed=3, scale=0.1)
    x = np.array([0.2, -0.3, 0.4])
    analytic = model.gradient_log_psi_position(x)
    numeric = np.array([central_difference(model.log_psi, x, i) for i in range(x.size)])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_parameter_derivative_a_matches_finite_difference():
    model = GaussianBinaryRBM.random_initialization(2, 3, seed=4, scale=0.1)
    x = np.array([0.25, -0.15])
    analytic = model.parameter_derivatives(x)["a"]
    numeric = []
    h = 1e-6
    for i in range(model.num_visible):
        plus = model.copy(); minus = model.copy()
        plus.a[i] += h; minus.a[i] -= h
        numeric.append((plus.log_psi(x) - minus.log_psi(x)) / (2 * h))
    np.testing.assert_allclose(analytic, np.array(numeric), rtol=1e-5, atol=1e-6)
