"""Conservative reward shaping for Orbit Wars."""

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


def _summary(obs: Any) -> dict[str, float]:
    if obs is None:
        return {
            "ship_adv": 0.0,
            "prod_adv": 0.0,
            "planet_adv": 0.0,
            "my_planets": 0.0,
            "enemy_planets": 0.0,
        }

    player = get_player(obs)

    ship_adv = 0.0
    prod_adv = 0.0
    my_planets = 0.0
    enemy_planets = 0.0

    for p in get_planets(obs):
        owner = int(p["owner"])
        ships = float(p["ships"])
        production = float(p["production"])

        if owner == player:
            ship_adv += ships
            prod_adv += production
            my_planets += 1.0
        elif owner >= 0:
            ship_adv -= ships
            prod_adv -= production
            enemy_planets += 1.0

    for f in get_fleets(obs):
        owner = int(f["owner"])
        ships = float(f["ships"])

        if owner == player:
            ship_adv += ships
        elif owner >= 0:
            ship_adv -= ships

    return {
        "ship_adv": ship_adv,
        "prod_adv": prod_adv,
        "planet_adv": my_planets - enemy_planets,
        "my_planets": my_planets,
        "enemy_planets": enemy_planets,
    }


def _ownership_changes(previous_obs: Any, current_obs: Any) -> tuple[float, float]:
    """
    Returns:
        captures, losses
    """
    if previous_obs is None or current_obs is None:
        return 0.0, 0.0

    player = get_player(current_obs)

    prev_owners = {
        int(p["id"]): int(p["owner"])
        for p in get_planets(previous_obs)
    }

    captures = 0.0
    losses = 0.0

    for p in get_planets(current_obs):
        pid = int(p["id"])
        old_owner = prev_owners.get(pid)
        new_owner = int(p["owner"])

        if old_owner is None:
            continue

        if old_owner != player and new_owner == player:
            captures += 1.0

        if old_owner == player and new_owner != player:
            losses += 1.0

    return captures, losses


def compute_reward(previous_obs, current_obs, env_state, done: bool) -> float:
    """
    Reward philosophy:
    - Reward changes in advantage.
    - Also give small ongoing pressure for current map control.
    - Explicitly reward captures and punish losses.
    - Terminal reward still matters, but should not be the only useful signal.
    """

    player = get_player(current_obs if current_obs is not None else previous_obs)

    prev = _summary(previous_obs)
    cur = _summary(current_obs)

    captures, losses = _ownership_changes(previous_obs, current_obs)

    delta_ship_adv = cur["ship_adv"] - prev["ship_adv"]
    delta_prod_adv = cur["prod_adv"] - prev["prod_adv"]
    delta_planet_adv = cur["planet_adv"] - prev["planet_adv"]

    reward = 0.0

    # Immediate advantage deltas.
    reward += 0.002 * delta_ship_adv
    reward += 0.04 * delta_prod_adv
    reward += 0.05 * delta_planet_adv

    # Explicit ownership events.
    reward += 0.25 * captures
    reward -= 0.35 * losses

    # Ongoing strategic pressure.
    # These are intentionally small, but they tell the agent:
    # "being behind in economy every turn is bad."
    reward += 0.001 * cur["ship_adv"]
    reward += 0.005 * cur["prod_adv"]
    reward += 0.01 * cur["planet_adv"]

    if done:
        reward += _state_reward(env_state, player)

    return float(reward)
