"""Candidate action generation for the discrete RL policy."""

from __future__ import annotations

import math
from typing import Any

from orbit_wars_rl.features.observation_encoder import get_planets, get_player

SHIP_FRACTIONS = (0.25, 0.5, 0.75)
MIN_SOURCE_SHIPS = 6
RESERVE_SHIPS = 2


def _angle_between(source: dict[str, Any], target: dict[str, Any]) -> float:
    """Aim directly at the target's current position.

    This small function is the seam where future intercept/orbital prediction logic
    can be added without changing the candidate schema.
    """

    return float(math.atan2(float(target["y"]) - float(source["y"]), float(target["x"]) - float(source["x"])))


def _target_priority(planet: dict[str, Any], player: int) -> tuple[int, float, int]:
    owner = int(planet["owner"])
    if owner == -1:
        group = 0
    elif owner != player:
        group = 1
    else:
        group = 2
    return group, float(planet["ships"]), int(planet["id"])


def generate_candidates(obs: dict, max_candidates: int) -> list[dict]:
    """Generate legal/plausible candidate actions.

    Candidate 0 is always NOOP. Non-NOOP candidates launch one fleet from an
    owned planet toward a neutral, enemy, or friendly reinforcement target.
    """

    if max_candidates <= 0:
        return []

    candidates: list[dict] = [{"type": "noop"}]
    planets = get_planets(obs)
    player = get_player(obs)
    owned = [p for p in planets if int(p["owner"]) == player and float(p["ships"]) >= MIN_SOURCE_SHIPS]
    owned.sort(key=lambda p: (-float(p["ships"]), int(p["id"])))
    targets = sorted(planets, key=lambda p: _target_priority(p, player))

    for source in owned:
        available = max(int(float(source["ships"])) - RESERVE_SHIPS, 0)
        if available <= 0:
            continue
        for target in targets:
            if int(target["id"]) == int(source["id"]):
                continue
            for fraction in SHIP_FRACTIONS:
                ships = int(math.floor(available * fraction))
                if ships <= 0:
                    continue
                distance = math.hypot(float(target["x"]) - float(source["x"]), float(target["y"]) - float(source["y"]))
                candidates.append({
                    "type": "send",
                    "from_planet_id": int(source["id"]),
                    "target_planet_id": int(target["id"]),
                    "ship_fraction": float(fraction),
                    "ships": int(ships),
                    "angle": _angle_between(source, target),
                    "candidate_features": [
                        float(fraction),
                        min(float(source["ships"]) / 500.0, 5.0),
                        min(float(target["ships"]) / 500.0, 5.0),
                        min(distance / 150.0, 2.0),
                        1.0 if int(target["owner"]) == player else 0.0,
                        1.0 if int(target["owner"]) == -1 else 0.0,
                    ],
                })
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates
