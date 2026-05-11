import numpy as np

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
