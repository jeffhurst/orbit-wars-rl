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
