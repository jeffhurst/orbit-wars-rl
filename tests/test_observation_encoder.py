from types import SimpleNamespace

import numpy as np

from orbit_wars_rl.features.candidate_generator import generate_candidates
from orbit_wars_rl.features.observation_encoder import (
    CANDIDATE_FEATURES,
    encode_observation,
    observation_size,
)


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


def test_observation_encoder_appends_candidate_features():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 30, 2],
            [1, -1, 50.0, 10.0, 2.0, 12, 3],
        ],
        "fleets": [],
    }
    candidates = generate_candidates(obs, max_candidates=4)

    encoded = encode_observation(obs, 2, 0, candidates, max_candidates=4)

    assert encoded.shape == (observation_size(2, 0, 4),)
    candidate_start = observation_size(2, 0)
    assert np.allclose(
        encoded[
            candidate_start
            + CANDIDATE_FEATURES : candidate_start
            + 2 * CANDIDATE_FEATURES
        ],
        candidates[1]["candidate_features"],
    )


def test_observation_encoder_accepts_object_observations_and_rows():
    obs = SimpleNamespace(
        player=0,
        planets=[
            SimpleNamespace(
                id=0, owner=0, x=10.0, y=10.0, radius=2.0, ships=30, production=2
            ),
            {
                "id": 1,
                "owner": -1,
                "x": 50.0,
                "y": 10.0,
                "radius": 2.0,
                "ships": 12,
                "production": 3,
            },
        ],
        fleets=[],
    )

    encoded = encode_observation(obs, max_planets=2, max_fleets=0)

    assert encoded.shape == (observation_size(2, 0),)
    assert encoded.dtype == np.float32
