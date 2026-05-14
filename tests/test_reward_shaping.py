from types import SimpleNamespace

import pytest

from orbit_wars_rl.features.reward_shaping import (
    compute_reward,
    compute_reward_components,
)


def make_obs(owner_by_planet, ships_by_planet=None, production_by_planet=None):
    ships_by_planet = ships_by_planet or {}
    production_by_planet = production_by_planet or {}
    return {
        "player": 0,
        "planets": [
            [
                pid,
                owner,
                float(pid * 10),
                float(pid * 10),
                2.0,
                ships_by_planet.get(pid, 10.0),
                production_by_planet.get(pid, 2.0),
            ]
            for pid, owner in owner_by_planet.items()
        ],
        "fleets": [],
    }


def terminal_state(raw_reward):
    return [SimpleNamespace(reward=raw_reward), SimpleNamespace(reward=-raw_reward)]


def assert_bounded(components):
    for key, value in components.items():
        if key == "reward_terminal_raw":
            continue
        assert -1.0 <= value <= 1.0, key


def test_capture_transition_has_positive_bounded_reward():
    previous_obs = make_obs({0: 0, 1: -1})
    current_obs = make_obs({0: 0, 1: 0})

    components = compute_reward_components(previous_obs, current_obs, None, False)

    assert_bounded(components)
    assert components["reward_capture"] > 0.0
    assert components["reward_loss"] == 0.0
    assert 0.0 < components["reward_total"] <= 1.0
    assert compute_reward(previous_obs, current_obs, None, False) == pytest.approx(
        components["reward_total"]
    )


def test_loss_transition_has_negative_bounded_reward():
    previous_obs = make_obs({0: 0, 1: 0})
    current_obs = make_obs({0: 0, 1: 1})

    components = compute_reward_components(previous_obs, current_obs, None, False)

    assert_bounded(components)
    assert components["reward_capture"] == 0.0
    assert components["reward_loss"] < 0.0
    assert -1.0 <= components["reward_total"] < 0.0


def test_terminal_win_uses_bounded_bonus_and_logs_raw_score():
    obs = make_obs({0: 0, 1: 1})

    components = compute_reward_components(obs, obs, terminal_state(50.0), True)

    assert_bounded(components)
    assert components["reward_terminal_raw"] == 50.0
    assert components["reward_terminal"] > 0.0
    assert 0.0 < components["reward_total"] <= 1.0


def test_terminal_loss_uses_bounded_penalty_and_logs_raw_score():
    obs = make_obs({0: 0, 1: 1})

    components = compute_reward_components(obs, obs, terminal_state(-50.0), True)

    assert_bounded(components)
    assert components["reward_terminal_raw"] == -50.0
    assert components["reward_terminal"] < 0.0
    assert -1.0 <= components["reward_total"] < 0.0


def test_large_advantage_swing_is_clipped():
    previous_obs = make_obs(
        {0: 0, 1: 1},
        ships_by_planet={0: 1_000.0, 1: 1_000.0},
        production_by_planet={0: 50.0, 1: 50.0},
    )
    current_obs = make_obs(
        {0: 0, 1: 0},
        ships_by_planet={0: 10_000.0, 1: 10_000.0},
        production_by_planet={0: 500.0, 1: 500.0},
    )

    components = compute_reward_components(previous_obs, current_obs, None, False)

    assert_bounded(components)
    assert components["reward_delta_advantage"] == 0.4
    assert components["reward_total"] <= 1.0
