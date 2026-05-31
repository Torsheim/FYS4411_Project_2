from nqs_vmc.models.gaussian_binary_rbm import GaussianBinaryRBM
from nqs_vmc.optimization.optimizers import SGD
from nqs_vmc.optimization.trainer import VMCTrainer
from nqs_vmc.physics.hamiltonian import HarmonicOscillatorHamiltonian
from nqs_vmc.sampling.metropolis import MetropolisSampler


def test_trainer_runs_one_iteration():
    model = GaussianBinaryRBM.random_initialization(1, 2, seed=12)
    hamiltonian = HarmonicOscillatorHamiltonian(num_particles=1, dimensions=1)
    sampler = MetropolisSampler(model, hamiltonian, step_size=1.0, seed=13)
    trainer = VMCTrainer(model=model, sampler=sampler, optimizer=SGD(learning_rate=0.01))
    history, final_position, final_energies = trainer.train(n_iterations=1, n_samples=20, n_burn_in=5)
    assert len(history) == 1
    assert final_position.shape == (1,)
    assert final_energies.shape == (20,)
