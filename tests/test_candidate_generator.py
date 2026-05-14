import math

from orbit_wars_rl.features.candidate_generator import generate_candidates
from orbit_wars_rl.features.metrics import make_action_metrics
from orbit_wars_rl.features.observation_encoder import CANDIDATE_FEATURES


def sample_obs():
    return {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 30, 2],
            [1, -1, 50.0, 10.0, 2.0, 12, 3],
            [2, 1, 90.0, 90.0, 2.0, 20, 2],
            [3, 0, 20.0, 20.0, 2.0, 8, 1],
        ],
        "fleets": [],
    }


def many_owned_and_target_obs():
    return {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 50, 2],
            [1, 0, 20.0, 10.0, 2.0, 45, 2],
            [2, 0, 30.0, 10.0, 2.0, 40, 2],
            [3, 0, 40.0, 10.0, 2.0, 35, 2],
            [4, 0, 50.0, 10.0, 2.0, 30, 2],
            [5, -1, 20.0, 60.0, 2.0, 8, 1],
            [6, -1, 30.0, 60.0, 2.0, 10, 1],
            [7, -1, 40.0, 60.0, 2.0, 12, 1],
            [8, 1, 70.0, 70.0, 2.0, 15, 2],
            [9, 1, 80.0, 70.0, 2.0, 18, 2],
            [10, 1, 90.0, 70.0, 2.0, 20, 2],
        ],
        "fleets": [],
    }


def test_candidate_generator_balances_sources_with_many_options():
    candidates = generate_candidates(many_owned_and_target_obs(), max_candidates=6)
    send_candidates = [
        candidate for candidate in candidates if candidate.get("type") == "send"
    ]

    assert candidates[0] == {"type": "noop"}
    assert len(send_candidates) == 5
    assert {candidate["from_planet_id"] for candidate in send_candidates} == {
        0,
        1,
        2,
        3,
        4,
    }


def test_candidate_generator_reserves_enemy_attacks_with_many_options():
    candidates = generate_candidates(many_owned_and_target_obs(), max_candidates=6)
    send_candidates = [
        candidate for candidate in candidates if candidate.get("type") == "send"
    ]

    assert len({candidate["from_planet_id"] for candidate in send_candidates}) > 1
    assert any(
        candidate["purpose"] == "attack_enemy" for candidate in send_candidates
    )


def test_candidate_generator_always_includes_noop():
    candidates = generate_candidates(sample_obs(), max_candidates=8)
    assert candidates[0] == {"type": "noop"}


def test_candidate_generator_never_exceeds_max_candidates():
    assert len(generate_candidates(sample_obs(), max_candidates=5)) <= 5


def test_candidate_generator_labels_target_owner_and_purpose():
    candidates = generate_candidates(sample_obs(), max_candidates=8)
    send_candidates = [
        candidate for candidate in candidates if candidate.get("type") == "send"
    ]

    assert send_candidates
    assert {candidate["target_owner"] for candidate in send_candidates} <= {-1, 0, 1}
    assert {candidate["purpose"] for candidate in send_candidates} <= {
        "capture_neutral",
        "attack_enemy",
        "reinforce",
    }
    assert send_candidates[0]["target_owner"] == -1
    assert send_candidates[0]["purpose"] == "capture_neutral"


def test_candidate_generator_emits_rich_candidate_features():
    candidates = generate_candidates(sample_obs(), max_candidates=32)
    send_candidates = [
        candidate for candidate in candidates if candidate.get("type") == "send"
    ]

    assert send_candidates
    assert all(
        len(candidate["candidate_features"]) == CANDIDATE_FEATURES
        for candidate in send_candidates
    )
    # The first feature marks real send candidates so NOOP/padding rows remain
    # distinguishable from legal launches with very small numeric values.
    assert all(
        candidate["candidate_features"][0] == 1.0
        for candidate in send_candidates
    )
    # Target type is explicit instead of requiring PPO to infer it from ordering.
    assert send_candidates[0]["candidate_features"][5:8] == [1.0, 0.0, 0.0]


def moving_target_obs():
    return {
        "player": 0,
        "angular_velocity": 0.2,
        "planets": [
            [0, 0, 50.0, 50.0, 2.0, 30, 2],
            [1, -1, 60.0, 50.0, 2.0, 12, 3],
        ],
        "fleets": [],
    }


def test_candidate_generator_uses_intercept_angle_for_orbiting_target():
    config = {"shipSpeed": 5.0}

    candidates = generate_candidates(
        moving_target_obs(), max_candidates=4, config=config
    )
    send_candidate = next(
        candidate for candidate in candidates if candidate.get("type") == "send"
    )

    naive_angle = math.atan2(50.0 - 50.0, 60.0 - 50.0)
    assert send_candidate["intercept_angle_used"] is True
    assert not math.isclose(send_candidate["angle"], naive_angle)
    assert send_candidate["angle"] > naive_angle


def test_candidate_generator_falls_back_to_direct_angle_without_motion_metadata():
    candidates = generate_candidates(sample_obs(), max_candidates=4)
    send_candidate = next(
        candidate for candidate in candidates if candidate.get("type") == "send"
    )

    assert send_candidate["intercept_angle_used"] is False
    assert math.isclose(send_candidate["angle"], 0.0)


def test_action_metrics_report_intercept_angle_usage():
    candidates = generate_candidates(
        moving_target_obs(), max_candidates=4, config={"shipSpeed": 5.0}
    )

    metrics = make_action_metrics(1, candidates, max_candidates=4)

    assert metrics["intercept_angle_rate"] == 1.0
    assert metrics["candidate_pool_intercept_angle_rate"] > 0.0


def test_candidate_generator_uses_logarithmic_speed_for_intercept_angle():
    obs = {
        "player": 0,
        "angular_velocity": 0.05,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 1002, 2],
            [1, -1, 60.0, 20.0, 2.0, 12, 3],
        ],
        "fleets": [],
    }

    candidates = generate_candidates(obs, max_candidates=4, config={"maxSpeed": 6.0})
    quarter = next(
        candidate
        for candidate in candidates
        if candidate.get("type") == "send" and candidate["ship_fraction"] == 0.25
    )
    half = next(
        candidate
        for candidate in candidates
        if candidate.get("type") == "send" and candidate["ship_fraction"] == 0.5
    )

    assert quarter["intercept_angle_used"] is True
    assert half["intercept_angle_used"] is True
    assert half["fleet_speed"] > quarter["fleet_speed"]
    assert half["travel_time"] < quarter["travel_time"]
    assert not math.isclose(half["angle"], quarter["angle"])


def test_candidate_generator_uses_initial_planets_and_step_for_orbit_prediction():
    obs = {
        "player": 0,
        "step": 10,
        "angular_velocity": {"1": 0.05},
        "initial_planets": [
            [0, 0, 20.0, 50.0, 2.0, 100, 2],
            [1, -1, 60.0, 50.0, 2.0, 12, 3],
        ],
        "planets": [
            [0, 0, 20.0, 50.0, 2.0, 100, 2],
            [1, -1, 60.0, 50.0, 2.0, 12, 3],
        ],
        "fleets": [],
    }

    candidates = generate_candidates(obs, max_candidates=4, config={"maxSpeed": 6.0})
    send_candidate = next(
        candidate for candidate in candidates if candidate.get("type") == "send"
    )

    assert send_candidate["intercept_angle_used"] is True
    assert send_candidate["angle"] > 0.0


def test_candidate_generator_filters_routes_through_sun():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 20.0, 50.0, 2.0, 100, 2],
            [1, -1, 80.0, 50.0, 2.0, 12, 3],
        ],
        "fleets": [],
    }

    candidates = generate_candidates(
        obs, max_candidates=8, config={"maxSpeed": 6.0, "sunRadius": 6.0}
    )

    assert candidates == [{"type": "noop"}]


def test_candidate_generator_filters_routes_into_moving_planets():
    obs = {
        "player": 0,
        "angular_velocity": {"2": 0.0},
        "initial_planets": [
            [0, 0, 20.0, 20.0, 2.0, 100, 2],
            [1, -1, 80.0, 20.0, 2.0, 12, 3],
            [2, -1, 50.0, 20.0, 3.0, 12, 3],
        ],
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 100, 2],
            [1, -1, 80.0, 20.0, 2.0, 12, 3],
            [2, -1, 50.0, 20.0, 3.0, 12, 3],
        ],
        "fleets": [],
    }

    candidates = generate_candidates(obs, max_candidates=16, config={"maxSpeed": 6.0})

    assert all(
        candidate.get("target_planet_id") != 1
        for candidate in candidates
        if candidate.get("type") == "send"
    )


class AbcHostileMapping:
    """Mapping-like object whose __class__ breaks collections.abc isinstance checks."""

    def __init__(self, values):
        self._values = values

    @property
    def __class__(self):
        raise RuntimeError("ABC instance checks should not inspect __class__")

    def get(self, key, default=None):
        return self._values.get(key, default)


class AbcHostileSequence:
    """Sequence-like row whose __class__ breaks collections.abc isinstance checks."""

    def __init__(self, values):
        self._values = values

    @property
    def __class__(self):
        raise RuntimeError("ABC instance checks should not inspect __class__")

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


def test_candidate_generator_handles_abc_hostile_observation_rows():
    obs = AbcHostileMapping(
        {
            "player": 0,
            "step": 10,
            "angular_velocity": {"1": 0.05},
            "initial_planets": [
                AbcHostileSequence([0, 0, 20.0, 50.0, 2.0, 100, 2]),
                AbcHostileSequence([1, -1, 60.0, 50.0, 2.0, 12, 3]),
            ],
            "planets": [
                AbcHostileSequence([0, 0, 20.0, 50.0, 2.0, 100, 2]),
                AbcHostileSequence([1, -1, 60.0, 50.0, 2.0, 12, 3]),
            ],
            "fleets": [],
        }
    )

    candidates = generate_candidates(obs, max_candidates=4, config={"maxSpeed": 6.0})
    send_candidate = next(
        candidate for candidate in candidates if candidate.get("type") == "send"
    )

    assert send_candidate["intercept_angle_used"] is True
