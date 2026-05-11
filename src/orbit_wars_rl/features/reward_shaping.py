"""Small, conservative reward shaping for Orbit Wars."""

from __future__ import annotations

from typing import Any

from orbit_wars_rl.features.observation_encoder import get_fleets, get_planets, get_player


def _state_reward(env_state: Any, player: int) -> float:
    try:
        state = env_state[player]
        reward = getattr(state, "reward", None)
        if reward is None and isinstance(state, dict):
            reward = state.get("reward")
        return 0.0 if reward is None else float(reward)
    except Exception:
        return 0.0


def _advantages(obs: Any) -> tuple[float, float]:
    if obs is None:
        return 0.0, 0.0
    player = get_player(obs)
    ship_adv = 0.0
    prod_adv = 0.0
    for p in get_planets(obs):
        sign = 1.0 if int(p["owner"]) == player else (-1.0 if int(p["owner"]) >= 0 else 0.0)
        ship_adv += sign * float(p["ships"])
        prod_adv += sign * float(p["production"])
    for f in get_fleets(obs):
        sign = 1.0 if int(f["owner"]) == player else (-1.0 if int(f["owner"]) >= 0 else 0.0)
        ship_adv += sign * float(f["ships"])
    return ship_adv, prod_adv


def compute_reward(previous_obs, current_obs, env_state, done: bool) -> float:
    """Combine terminal Kaggle reward with small advantage deltas."""

    player = get_player(current_obs if current_obs is not None else previous_obs)
    prev_ships, prev_prod = _advantages(previous_obs)
    cur_ships, cur_prod = _advantages(current_obs)
    reward = 0.002 * (cur_ships - prev_ships) + 0.02 * (cur_prod - prev_prod)
    if done:
        reward += _state_reward(env_state, player)
    return float(reward)
