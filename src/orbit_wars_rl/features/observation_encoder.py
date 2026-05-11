"""Fixed-size observation encoding for Orbit Wars.

Orbit Wars observations are dictionaries whose core arrays are documented as:
planets: [id, owner, x, y, radius, ships, production]
fleets:  [id, owner, x, y, angle, from_planet_id, ships]
The helpers here are intentionally defensive because Kaggle may pass a struct-like
object and future environment versions may add fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

BOARD_SIZE = 100.0
DEFAULT_EPISODE_STEPS = 500.0
PLANET_FEATURES = 9
FLEET_FEATURES = 8
GLOBAL_FEATURES = 6


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rows(obs: Any, key: str) -> list[Any]:
    rows = _get(obs, key, [])
    if rows is None:
        return []
    if isinstance(rows, np.ndarray):
        rows = rows.tolist()
    if not isinstance(rows, list | tuple):
        return []
    return list(rows)


def _row_value(row: Any, index: int, name: str, default: Any = 0) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    if hasattr(row, name):
        return getattr(row, name)
    try:
        return row[index]
    except (TypeError, IndexError, KeyError):
        return default


def get_player(obs: Any) -> int:
    """Return the active player id, defaulting to player 0."""

    return _as_int(_get(obs, "player", _get(obs, "mark", 0)), 0)


def get_planets(obs: Any) -> list[dict[str, float | int]]:
    """Parse planets into dictionaries sorted by id when possible."""

    planets: list[dict[str, float | int]] = []
    for row in _rows(obs, "planets"):
        planet = {
            "id": _as_int(_row_value(row, 0, "id", len(planets))),
            "owner": _as_int(_row_value(row, 1, "owner", -1), -1),
            "x": _as_float(_row_value(row, 2, "x", 0.0)),
            "y": _as_float(_row_value(row, 3, "y", 0.0)),
            "radius": _as_float(_row_value(row, 4, "radius", 0.0)),
            "ships": _as_float(_row_value(row, 5, "ships", 0.0)),
            "production": _as_float(_row_value(row, 6, "production", 0.0)),
        }
        planets.append(planet)
    return sorted(planets, key=lambda p: int(p["id"]))


def get_fleets(obs: Any) -> list[dict[str, float | int]]:
    """Parse fleets into dictionaries sorted by id when possible."""

    fleets: list[dict[str, float | int]] = []
    for row in _rows(obs, "fleets"):
        fleet = {
            "id": _as_int(_row_value(row, 0, "id", len(fleets))),
            "owner": _as_int(_row_value(row, 1, "owner", -1), -1),
            "x": _as_float(_row_value(row, 2, "x", 0.0)),
            "y": _as_float(_row_value(row, 3, "y", 0.0)),
            "angle": _as_float(_row_value(row, 4, "angle", 0.0)),
            "from_planet_id": _as_int(_row_value(row, 5, "from_planet_id", -1), -1),
            "ships": _as_float(_row_value(row, 6, "ships", 0.0)),
        }
        fleets.append(fleet)
    return sorted(fleets, key=lambda f: int(f["id"]))


def observation_size(max_planets: int, max_fleets: int) -> int:
    return GLOBAL_FEATURES + max_planets * PLANET_FEATURES + max_fleets * FLEET_FEATURES


def _owner_features(owner: int, player: int) -> tuple[float, float]:
    if owner == player:
        return 1.0, 0.0
    if owner < 0:
        return 0.0, 0.0
    return 0.0, 1.0


def encode_observation(obs: dict, max_planets: int, max_fleets: int) -> np.ndarray:
    """Encode an Orbit Wars observation as a fixed-size float32 vector."""

    player = get_player(obs)
    step = _as_float(_get(obs, "step", _get(obs, "turn", 0.0)))
    episode_steps = max(_as_float(_get(obs, "episodeSteps", DEFAULT_EPISODE_STEPS), DEFAULT_EPISODE_STEPS), 1.0)
    planets = get_planets(obs)
    fleets = get_fleets(obs)

    owned_planets = sum(1 for p in planets if int(p["owner"]) == player)
    enemy_planets = sum(1 for p in planets if int(p["owner"]) >= 0 and int(p["owner"]) != player)

    features: list[float] = [
        np.clip(step / episode_steps, 0.0, 1.0),
        np.clip(player / 3.0, 0.0, 1.0),
        np.clip(len(planets) / max(float(max_planets), 1.0), 0.0, 2.0),
        np.clip(len(fleets) / max(float(max_fleets), 1.0), 0.0, 2.0),
        np.clip(owned_planets / max(len(planets), 1), 0.0, 1.0),
        np.clip(enemy_planets / max(len(planets), 1), 0.0, 1.0),
    ]

    for i in range(max_planets):
        if i < len(planets):
            p = planets[i]
            owner = int(p["owner"])
            is_mine, is_enemy = _owner_features(owner, player)
            features.extend([
                np.clip(float(p["id"]) / 128.0, 0.0, 4.0),
                is_mine,
                is_enemy,
                np.clip(float(p["x"]) / BOARD_SIZE, -1.0, 2.0),
                np.clip(float(p["y"]) / BOARD_SIZE, -1.0, 2.0),
                np.clip(float(p["radius"]) / 10.0, 0.0, 2.0),
                np.clip(float(p["ships"]) / 500.0, 0.0, 5.0),
                np.clip(float(p["production"]) / 5.0, 0.0, 2.0),
                1.0,
            ])
        else:
            features.extend([0.0] * PLANET_FEATURES)

    for i in range(max_fleets):
        if i < len(fleets):
            f = fleets[i]
            owner = int(f["owner"])
            is_mine, is_enemy = _owner_features(owner, player)
            angle = _as_float(f["angle"])
            features.extend([
                np.clip(float(f["id"]) / 512.0, 0.0, 8.0),
                is_mine,
                is_enemy,
                np.clip(float(f["x"]) / BOARD_SIZE, -1.0, 2.0),
                np.clip(float(f["y"]) / BOARD_SIZE, -1.0, 2.0),
                float(np.sin(angle)),
                float(np.cos(angle)),
                np.clip(float(f["ships"]) / 500.0, 0.0, 5.0),
            ])
        else:
            features.extend([0.0] * FLEET_FEATURES)

    return np.asarray(features, dtype=np.float32)
