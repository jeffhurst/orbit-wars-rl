from __future__ import annotations

from collections import defaultdict
from typing import Any


def get_player_id(obs: dict[str, Any]) -> int:
    """
    Orbit Wars observations may use slightly different names depending on wrapper/version.
    Default to player 0 if unavailable.
    """
    return int(obs.get("player", obs.get("player_id", 0)))


def get_planets(obs: dict[str, Any]) -> list[list]:
    return list(obs.get("planets", []))


def planet_id(planet: list) -> int:
    return int(planet[0])


def planet_owner(planet: list) -> int:
    return int(planet[1])


def planet_ships(planet: list) -> float:
    # Expected format:
    # [id, owner, x, y, radius, ships, production]
    try:
        return float(planet[5])
    except Exception:
        return 0.0


def planet_production(planet: list) -> float:
    try:
        return float(planet[6])
    except Exception:
        return 0.0


def summarize_planets(obs: dict[str, Any]) -> dict[str, float]:
    player_id = get_player_id(obs)

    my_planets = 0
    enemy_planets = 0
    my_ships = 0.0
    enemy_ships = 0.0
    my_production = 0.0
    enemy_production = 0.0

    for p in get_planets(obs):
        owner = planet_owner(p)
        ships = planet_ships(p)
        production = planet_production(p)

        if owner == player_id:
            my_planets += 1
            my_ships += ships
            my_production += production
        elif owner != -1:
            enemy_planets += 1
            enemy_ships += ships
            enemy_production += production

    return {
        "planet_count_mine": float(my_planets),
        "planet_count_enemy": float(enemy_planets),
        "ship_advantage": float(my_ships - enemy_ships),
        "production_advantage": float(my_production - enemy_production),
    }


def count_captures_and_losses(
    previous_obs: dict[str, Any] | None,
    current_obs: dict[str, Any],
) -> dict[str, float]:
    if previous_obs is None:
        return {
            "captures": 0.0,
            "planets_lost": 0.0,
        }

    player_id = get_player_id(current_obs)

    prev_owners = {
        planet_id(p): planet_owner(p)
        for p in get_planets(previous_obs)
    }

    captures = 0
    planets_lost = 0

    for p in get_planets(current_obs):
        pid = planet_id(p)
        old_owner = prev_owners.get(pid)
        new_owner = planet_owner(p)

        if old_owner is None:
            continue

        if old_owner != player_id and new_owner == player_id:
            captures += 1

        if old_owner == player_id and new_owner != player_id:
            planets_lost += 1

    return {
        "captures": float(captures),
        "planets_lost": float(planets_lost),
    }


def make_action_metrics(
    action_index: int,
    candidates: list[dict[str, Any]],
    max_candidates: int,
) -> dict[str, float]:
    """
    Per-step action diagnostics.
    Each boolean metric is represented as 0.0 or 1.0 so it can be averaged.
    """
    metrics = {
        "noop_rate": 0.0,
        "invalid_action_rate": 0.0,
        "send_rate": 0.0,
        "avg_candidates": float(len(candidates)),
        "avg_ships_sent": 0.0,
        "avg_ship_fraction": 0.0,
        "target_owner_neutral_rate": 0.0,
        "target_owner_enemy_rate": 0.0,
        "target_owner_self_rate": 0.0,
        "source_planet_id": -1.0,
        "chosen_candidate_index": float(action_index),
    }

    if action_index < 0 or action_index >= len(candidates):
        metrics["invalid_action_rate"] = 1.0
        return metrics

    candidate = candidates[action_index]
    ctype = candidate.get("type", "invalid")

    if ctype == "noop":
        metrics["noop_rate"] = 1.0
        return metrics

    if ctype != "send":
        metrics["invalid_action_rate"] = 1.0
        return metrics

    metrics["send_rate"] = 1.0
    metrics["avg_ships_sent"] = float(candidate.get("ships", 0))
    metrics["avg_ship_fraction"] = float(candidate.get("ship_fraction", 0.0))
    metrics["source_planet_id"] = float(candidate.get("from_planet_id", -1))

    purpose = candidate.get("purpose", "")

    if purpose == "capture_neutral":
        metrics["target_owner_neutral_rate"] = 1.0
    elif purpose == "attack_enemy":
        metrics["target_owner_enemy_rate"] = 1.0
    elif purpose == "reinforce":
        metrics["target_owner_self_rate"] = 1.0

    return metrics
