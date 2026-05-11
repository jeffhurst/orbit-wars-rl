import numpy as np

from orbit_wars_rl.features.observation_encoder import encode_observation, observation_size


def test_observation_encoder_returns_fixed_shape_float32():
    obs = {
        "player": 0,
        "step": 10,
        "planets": [[2, -1, 50, 50, 2, 15, 3], [1, 0, 10, 10, 2, 20, 2]],
        "fleets": [[0, 0, 15, 15, 0.0, 1, 5]],
    }
    encoded = encode_observation(obs, max_planets=4, max_fleets=3)
    assert encoded.shape == (observation_size(4, 3),)
    assert encoded.dtype == np.float32
