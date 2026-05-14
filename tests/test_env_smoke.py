import numpy as np
import pytest

from orbit_wars_rl.envs.orbit_wars_gym import OrbitWarsGym


def test_orbit_wars_gym_reset_works():
    env = OrbitWarsGym(max_candidates=8, max_planets=8, max_fleets=8)
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_orbit_wars_gym_step_noop_works():
    env = OrbitWarsGym(max_candidates=8, max_planets=8, max_fleets=8)
    env.reset(seed=123)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_masks_shape_and_true():
    env = OrbitWarsGym(max_candidates=8, max_planets=8, max_fleets=8)
    env.reset(seed=123)
    mask = env.action_masks()
    assert mask.shape == (8,)
    assert mask.dtype == np.bool_
    assert mask.any()
    assert mask[0]


def test_reset_info_reports_synthetic_fallback_when_active(monkeypatch):
    env = OrbitWarsGym(max_candidates=8, max_planets=8, max_fleets=8)

    def fake_make_env():
        env._fallback_reason = "test fallback"
        return None

    monkeypatch.setattr(env, "_make_env", fake_make_env)
    _, info = env.reset(seed=123)

    assert info["fallback_env"] is True
    assert info["fallback_reason"] == "test fallback"


def test_reset_passes_seed_to_supported_kaggle_reset(monkeypatch):
    class AgentState:
        def __init__(self, observation):
            self.observation = observation
            self.reward = 0.0
            self.status = "ACTIVE"

    class FakeKaggleEnv:
        def __init__(self):
            self.configuration = None
            self.steps = []
            self.reset_num_players = None
            self.reset_seed = None
            self.reset_options = None
            self.state = [
                AgentState({"player": 0, "planets": [], "fleets": []}),
                AgentState({"player": 1, "planets": [], "fleets": []}),
            ]

        def reset(self, num_players=None, seed=None, options=None):
            self.reset_num_players = num_players
            self.reset_seed = seed
            self.reset_options = options
            return self.state

    fake_kaggle_env = FakeKaggleEnv()
    env = OrbitWarsGym(max_candidates=8, max_planets=8, max_fleets=8)

    def fake_make_env(seed=None, options=None):
        return fake_kaggle_env

    monkeypatch.setattr(env, "_make_env", fake_make_env)
    _, info = env.reset(seed=987, options={"map": "regression-map"})

    assert fake_kaggle_env.reset_num_players == 2
    assert fake_kaggle_env.reset_seed == 987
    assert fake_kaggle_env.reset_options == {"map": "regression-map"}
    assert info["seed"] == 987
    assert info["kaggle_seed_applied"] is True


def test_require_real_kaggle_env_rejects_synthetic_fallback(monkeypatch):
    env = OrbitWarsGym(max_candidates=8, max_planets=8, max_fleets=8)

    def fake_make_env():
        env._fallback_reason = "test fallback"
        return None

    monkeypatch.setattr(env, "_make_env", fake_make_env)
    env.reset(seed=123)

    with pytest.raises(RuntimeError, match="synthetic smoke-test fallback"):
        env.require_real_kaggle_env()


def test_real_env_observations_are_snapshotted_for_reward_deltas(monkeypatch):
    shared_obs = {
        "player": 0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 20, 2],
            [1, -1, 50.0, 20.0, 2.0, 5, 1],
        ],
        "fleets": [],
    }

    class AgentState:
        def __init__(self, observation, reward=0.0):
            self.observation = observation
            self.reward = reward
            self.status = "ACTIVE"

    class MutableObservationEnv:
        def __init__(self):
            self.configuration = None
            self.steps = []
            self.state = [
                AgentState(shared_obs),
                AgentState(
                    {"player": 1, "planets": shared_obs["planets"], "fleets": []}
                ),
            ]

        def reset(self, num_players=None):
            return self.state

        def step(self, actions):
            shared_obs["planets"][1][1] = 0
            self.steps.append(actions)
            return self.state

    env = OrbitWarsGym(
        opponent_agent=lambda obs, config: [],
        max_candidates=8,
        max_planets=8,
        max_fleets=8,
    )
    monkeypatch.setattr(env, "_make_env", MutableObservationEnv)

    env.reset(seed=123)
    _, reward, _, _, info = env.step(0)

    assert info["custom_metrics"]["captures"] == 1.0
    assert reward > 0.25
