from orbit_wars_rl.features.candidate_generator import generate_candidates


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
