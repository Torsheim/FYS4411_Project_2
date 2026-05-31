from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.physics.hamiltonian import HarmonicOscillatorHamiltonian
from nqs_vmc.sampling.metropolis import MetropolisSampler


def test_same_seed_gives_same_samples():
    model1 = GaussianBinaryRBM.random_initialization(1, 2, seed=14)
    model2 = GaussianBinaryRBM.random_initialization(1, 2, seed=14)
    hamiltonian = HarmonicOscillatorHamiltonian(num_particles=1, dimensions=1)
    sampler1 = MetropolisSampler(model1, hamiltonian, step_size=1.0, seed=15)
    sampler2 = MetropolisSampler(model2, hamiltonian, step_size=1.0, seed=15)
    r1 = sampler1.sample(n_samples=10, n_burn_in=2)
    r2 = sampler2.sample(n_samples=10, n_burn_in=2)
    assert (r1.positions == r2.positions).all()
    assert (r1.local_energies == r2.local_energies).all()
