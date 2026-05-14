import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecMonitor

from orbit_wars_rl import train


class OneStepMaskedEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.steps = 0

    def action_masks(self) -> np.ndarray:
        return np.array([True, False], dtype=bool)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(1, dtype=np.float32), 1.5, True, False, {}


def test_build_vec_env_adds_episode_monitoring_and_preserves_action_masks(monkeypatch):
    def fake_make_orbit_wars_env(**kwargs):
        return OneStepMaskedEnv

    monkeypatch.setattr(train, "make_orbit_wars_env", fake_make_orbit_wars_env)
    args = argparse.Namespace(opponent="starter", seed=123, n_envs=1)

    env = train.build_vec_env(args)
    try:
        assert isinstance(env, VecMonitor)
        np.testing.assert_array_equal(
            env.env_method("action_masks")[0], np.array([True, False], dtype=bool)
        )

        env.reset()
        _, _, dones, infos = env.step([0])

        assert dones.tolist() == [True]
        assert infos[0]["episode"]["r"] == 1.5
        assert infos[0]["episode"]["l"] == 1
    finally:
        env.close()
