from nqs_vmc.physics.hamiltonian import HarmonicOscillatorHamiltonian


def test_non_interacting_exact_energy_formula():
    h = HarmonicOscillatorHamiltonian(num_particles=2, dimensions=2, omega=1.0)
    assert h.exact_non_interacting_energy() == 2.0


def test_interacting_benchmark_value():
    h = HarmonicOscillatorHamiltonian(num_particles=2, dimensions=2, omega=1.0, include_interaction=True)
    assert h.exact_interacting_2p2d_energy() == 3.0
