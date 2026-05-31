from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.physics.hamiltonian import HarmonicOscillatorHamiltonian
from nqs_vmc.sampling.metropolis import MetropolisSampler


def test_metropolis_sampler_smoke():
    model = GaussianBinaryRBM.random_initialization(1, 2, seed=6)
    hamiltonian = HarmonicOscillatorHamiltonian(num_particles=1, dimensions=1)
    sampler = MetropolisSampler(model, hamiltonian, step_size=1.0, seed=7)
    result = sampler.sample(n_samples=20, n_burn_in=5)
    assert result.positions.shape == (20, 1)
    assert result.local_energies.shape == (20,)
    assert 0.0 <= result.acceptance_rate <= 1.0
