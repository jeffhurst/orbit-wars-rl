"""Gymnasium wrapper for Kaggle Orbit Wars with discrete candidate actions."""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orbit_wars_rl.features.action_decoder import decode_candidate
from orbit_wars_rl.features.candidate_generator import generate_candidates
from orbit_wars_rl.features.metrics import (
    make_action_metrics,
    summarize_planets,
    count_captures_and_losses,
)
from orbit_wars_rl.features.observation_encoder import (
    encode_observation,
    observation_size,
)
from orbit_wars_rl.features.reward_shaping import compute_reward_components
from orbit_wars_rl.opponents.starter_bot import agent as starter_agent

OpponentAgent = Callable[[Any, Any], list]


class OrbitWarsGym(gym.Env):
    """Train one player in a 2-player Kaggle Orbit Wars match.

    The policy action is a discrete index into a freshly generated candidate list.
    Invalid padded indices, NOOP, and malformed candidates decode to [] safely.
    """

    metadata = {"render_modes": ["ansi", "html", "json"]}

    def __init__(
        self,
        opponent_agent: OpponentAgent | None = None,
        player_id: int = 0,
        max_candidates: int = 64,
        max_planets: int = 48,
        max_fleets: int = 128,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.player_id = int(player_id)
        self.num_players = 2
        self.opponent_agent = opponent_agent or starter_agent
        self.max_candidates = int(max_candidates)
        self.max_planets = int(max_planets)
        self.max_fleets = int(max_fleets)
        self.debug = debug

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(
                observation_size(
                    self.max_planets, self.max_fleets, self.max_candidates
                ),
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_candidates)
        self.current_candidates: list[dict] = [{"type": "noop"}]
        self._env: Any | None = None
        self._last_obs: Any | None = None
        self.previous_obs: Any | None = None
        self.episode_metrics = {
            "captures_per_episode": 0.0,
            "planets_lost_per_episode": 0.0,
        }
        self._fallback_step = 0
        self._fallback_reason: str | None = None
        self._last_seed: int | None = None
        self._kaggle_seed_applied = False

    @property
    def using_fallback_env(self) -> bool:
        """Whether the wrapper is currently backed by the synthetic smoke-test fallback."""
        return self._env is None

    @property
    def fallback_reason(self) -> str | None:
        """Reason the synthetic fallback is active, if known."""
        return self._fallback_reason

    def require_real_kaggle_env(self) -> None:
        """Raise if reset() selected the synthetic fallback instead of Kaggle orbit_wars."""
        if not self.using_fallback_env:
            return

        reason = f" Reason: {self._fallback_reason}." if self._fallback_reason else ""
        raise RuntimeError(
            "OrbitWarsGym is using the synthetic smoke-test fallback instead of Kaggle's "
            "real orbit_wars environment. Refusing to train because this would produce "
            "a checkpoint from fallback observations and rewards."
            f"{reason} Install/configure kaggle-environments with orbit_wars support, "
            "then rerun training."
        )

    def _make_env(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Any | None:
        self._fallback_reason = None
        self._kaggle_seed_applied = False
        if importlib.util.find_spec("kaggle_environments") is None:
            self._fallback_reason = "kaggle_environments is not installed"
            if self.debug:
                print(
                    f"Falling back to synthetic Orbit Wars smoke env: {self._fallback_reason}"
                )
            return None
        from kaggle_environments import make

        seeded_configuration = self._seeded_configuration(seed, options)
        seeded_make_error: Exception | None = None
        if seeded_configuration is not None:
            try:
                env = make(
                    "orbit_wars",
                    configuration=seeded_configuration,
                    debug=self.debug,
                )
                if seed is not None:
                    self._kaggle_seed_applied = True
                return env
            except Exception as exc:
                seeded_make_error = exc
                if self.debug:
                    print(
                        "Kaggle make('orbit_wars') did not accept seeded "
                        f"configuration; retrying without it: {exc}"
                    )

        try:
            return make("orbit_wars", debug=self.debug)
        except Exception as exc:
            self._fallback_reason = f"make('orbit_wars') failed: {exc}"
            if seeded_make_error is not None:
                self._fallback_reason += (
                    f"; seeded configuration failed: {seeded_make_error}"
                )
            if self.debug:
                print(
                    f"Falling back to synthetic Orbit Wars smoke env: {self._fallback_reason}"
                )
            return None

    def _seeded_configuration(
        self, seed: int | None, options: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        configuration: dict[str, Any] = {}
        if isinstance(options, dict) and isinstance(options.get("configuration"), dict):
            configuration.update(options["configuration"])
        if seed is not None:
            configuration["seed"] = int(seed)
        return configuration or None

    def _reset_kaggle_env(
        self, seed: int | None, options: dict[str, Any] | None
    ) -> bool:
        if self._env is None:
            return False

        reset_options = dict(options or {})
        if seed is not None:
            reset_options.setdefault("seed", int(seed))

        seeded_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        if seed is not None:
            seeded_attempts.extend(
                [
                    ((self.num_players,), {"seed": int(seed), "options": options}),
                    (
                        (),
                        {
                            "num_players": self.num_players,
                            "seed": int(seed),
                            "options": options,
                        },
                    ),
                    ((self.num_players,), {"options": reset_options}),
                    ((), {"num_players": self.num_players, "options": reset_options}),
                    ((), {"seed": int(seed), "options": options}),
                    ((), {"options": reset_options}),
                ]
            )
        elif options is not None:
            seeded_attempts.extend(
                [
                    ((self.num_players,), {"options": options}),
                    ((), {"num_players": self.num_players, "options": options}),
                    ((), {"options": options}),
                ]
            )

        for args, kwargs in seeded_attempts:
            try:
                self._env.reset(*args, **kwargs)
                return seed is not None
            except Exception as exc:
                if self.debug:
                    print(f"Kaggle env.reset did not accept seeded/options call: {exc}")

        try:
            self._env.reset(self.num_players)
        except TypeError:
            self._env.reset()
        return False

    def _player_obs(self) -> Any:
        if self._env is None:
            return self._fallback_obs()
        obs = copy.deepcopy(self._env.state[self.player_id].observation)
        try:
            obs["step"] = len(getattr(self._env, "steps", []))
        except Exception:
            pass
        return obs

    def _opponent_obs(self, opponent_id: int) -> Any:
        if self._env is None:
            return self._fallback_obs(player=opponent_id)
        return copy.deepcopy(self._env.state[opponent_id].observation)

    def _fallback_obs(self, player: int | None = None) -> dict:
        player = self.player_id if player is None else player
        return {
            "player": player,
            "step": self._fallback_step,
            "episodeSteps": 500,
            "planets": [
                [0, 0, 20.0, 20.0, 2.0, 10 + self._fallback_step, 2],
                [1, 1, 80.0, 80.0, 2.0, 10 + self._fallback_step, 2],
                [2, -1, 50.0, 30.0, 2.0, 15, 3],
            ],
            "fleets": [],
        }

    def _refresh_candidates(self) -> None:
        self.current_candidates = generate_candidates(
            self._last_obs or {},
            self.max_candidates,
            getattr(self._env, "configuration", None),
        )
        if not self.current_candidates:
            self.current_candidates = [{"type": "noop"}]

    def _collect_metrics(
        self,
        action: int,
        candidates: list[dict],
        previous_obs: Any | None,
        current_obs: Any,
        terminated: bool,
        reward_components: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        ownership_changes = count_captures_and_losses(previous_obs, current_obs)
        self.episode_metrics["captures_per_episode"] += ownership_changes["captures"]
        self.episode_metrics["planets_lost_per_episode"] += ownership_changes[
            "planets_lost"
        ]

        custom_metrics = {
            **make_action_metrics(int(action), candidates, self.max_candidates),
            **summarize_planets(current_obs),
            **ownership_changes,
        }
        if reward_components is not None:
            custom_metrics.update(reward_components)
        metrics = {"custom_metrics": custom_metrics}
        if terminated:
            metrics["episode_metrics"] = dict(self.episode_metrics)
        return metrics

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.max_candidates, dtype=bool)
        valid_count = min(len(self.current_candidates), self.max_candidates)
        if valid_count > 0:
            mask[:valid_count] = True
        mask[0] = True
        return mask

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._last_seed = int(seed) if seed is not None else None
        self._fallback_step = 0
        self.previous_obs = None
        self.episode_metrics = {
            "captures_per_episode": 0.0,
            "planets_lost_per_episode": 0.0,
        }
        try:
            self._env = self._make_env(seed=self._last_seed, options=options)
        except TypeError:
            self._env = self._make_env()
        reset_seed_applied = self._reset_kaggle_env(self._last_seed, options)
        self._kaggle_seed_applied = self._kaggle_seed_applied or reset_seed_applied
        self._last_obs = self._player_obs()
        self._refresh_candidates()
        info: dict[str, Any] = {
            "fallback_env": self.using_fallback_env,
            "seed": self._last_seed,
            "kaggle_seed_applied": (
                self._kaggle_seed_applied and not self.using_fallback_env
            ),
        }
        if self._fallback_reason:
            info["fallback_reason"] = self._fallback_reason
        return (
            encode_observation(
                self._last_obs,
                self.max_planets,
                self.max_fleets,
                self.current_candidates,
                self.max_candidates,
            ),
            info,
        )

    def step(self, action: int):
        previous_obs = self._last_obs
        action_candidates = list(self.current_candidates)
        rl_action = decode_candidate(action, action_candidates)

        if self._env is None:
            self._fallback_step += 1
            current_obs = self._fallback_obs()
            terminated = self._fallback_step >= 500
            reward_components = compute_reward_components(
                previous_obs, current_obs, None, terminated
            )
            reward = reward_components["reward_total"]
            final_score = (
                reward_components["reward_terminal_raw"] if terminated else None
            )
            info = {
                "kaggle_action": rl_action,
                "fallback_env": True,
                "final_score": final_score,
                **self._collect_metrics(
                    action,
                    action_candidates,
                    previous_obs,
                    current_obs,
                    terminated,
                    reward_components,
                ),
            }
            self.previous_obs = current_obs
            self._last_obs = current_obs
            self._refresh_candidates()
            encoded = encode_observation(
                current_obs,
                self.max_planets,
                self.max_fleets,
                self.current_candidates,
                self.max_candidates,
            )
            return encoded, reward, terminated, False, info

        actions: list[Any] = [[] for _ in range(self.num_players)]
        actions[self.player_id] = rl_action
        for pid in range(self.num_players):
            if pid == self.player_id:
                continue
            try:
                actions[pid] = self.opponent_agent(
                    self._opponent_obs(pid), getattr(self._env, "configuration", None)
                )
            except Exception:
                actions[pid] = []

        try:
            state = self._env.step(actions)
        except Exception as exc:
            if self.debug:
                print(f"Kaggle env.step failed; treating action as NOOP: {exc}")
            state = self._env.step([[] for _ in range(self.num_players)])

        current_obs = self._player_obs()
        terminated = all(
            getattr(agent_state, "status", "DONE") != "ACTIVE"
            for agent_state in self._env.state
        )
        reward_components = compute_reward_components(
            previous_obs, current_obs, state, terminated
        )
        reward = reward_components["reward_total"]
        raw_reward = getattr(self._env.state[self.player_id], "reward", None)
        info = {
            "kaggle_action": rl_action,
            "raw_reward": raw_reward,
            "final_score": raw_reward if terminated else None,
            **self._collect_metrics(
                action,
                action_candidates,
                previous_obs,
                current_obs,
                terminated,
                reward_components,
            ),
        }
        self.previous_obs = current_obs
        self._last_obs = current_obs
        self._refresh_candidates()
        encoded = encode_observation(
            current_obs,
            self.max_planets,
            self.max_fleets,
            self.current_candidates,
            self.max_candidates,
        )
        return encoded, reward, terminated, False, info

    def render(self):
        if self._env is None:
            return str(self._fallback_obs())
        return self._env.render(mode="ansi")

    def close(self) -> None:
        self._env = None
