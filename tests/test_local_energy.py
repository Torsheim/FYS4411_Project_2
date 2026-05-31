import numpy as np

from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.physics.hamiltonian import HarmonicOscillatorHamiltonian


def test_ground_state_local_energy_for_one_dimensional_oscillator():
    # With no hidden coupling and sigma=1, a=0, the RBM is proportional to exp(-x^2/2),
    # the exact ground-state wave function for omega=1 in one dimension.
    model = GaussianBinaryRBM(num_visible=1, num_hidden=1, sigma=1.0)
    hamiltonian = HarmonicOscillatorHamiltonian(num_particles=1, dimensions=1, omega=1.0)
    for x in np.linspace(-2, 2, 9):
        assert abs(hamiltonian.local_energy(model, np.array([x])) - 0.5) < 1e-12
