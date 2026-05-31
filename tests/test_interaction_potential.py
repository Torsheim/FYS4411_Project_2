import numpy as np

from nqs_vmc.physics.potentials import inverse_distance_interaction, pair_distances


def test_pair_distance_two_particles_2d():
    x = np.array([0.0, 0.0, 3.0, 4.0])
    distances = pair_distances(x, num_particles=2, dimensions=2)
    assert distances.shape == (1,)
    assert distances[0] == 5.0
    assert inverse_distance_interaction(x, num_particles=2, dimensions=2) == 0.2
