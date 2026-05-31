from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.physics.hamiltonian import HarmonicOscillatorHamiltonian
from nqs_vmc.sampling.importance_sampling import ImportanceSampler


def test_importance_sampler_smoke():
    model = GaussianBinaryRBM.random_initialization(1, 2, seed=8)
    hamiltonian = HarmonicOscillatorHamiltonian(num_particles=1, dimensions=1)
    sampler = ImportanceSampler(model, hamiltonian, time_step=0.05, seed=9)
    result = sampler.sample(n_samples=20, n_burn_in=5)
    assert result.positions.shape == (20, 1)
    assert result.local_energies.shape == (20,)
    assert 0.0 <= result.acceptance_rate <= 1.0
