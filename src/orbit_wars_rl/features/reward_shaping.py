"""Conservative reward shaping for Orbit Wars."""

from __future__ import annotations

from math import isfinite
from typing import Any

REWARD_MIN = -1.0
REWARD_MAX = 1.0
DELTA_ADVANTAGE_MIN = -0.4
DELTA_ADVANTAGE_MAX = 0.4
CAPTURE_REWARD_MAX = 0.5
LOSS_REWARD_MIN = -0.6
TERMINAL_OUTCOME_SCALE = 0.5
_MISSING = object()


def _is_dict(obj: Any) -> bool:
    return issubclass(type(obj), dict)


def _is_non_string_sequence(obj: Any) -> bool:
    obj_type = type(obj)
    if issubclass(obj_type, (str, bytes, bytearray)):
        return False
    if issubclass(obj_type, (list, tuple)):
        return True
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__")


def _mapping_value(obj: Any, key: str, default: Any = None) -> Any:
    if _is_dict(obj):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        sentinel = object()
        value = getter(key, sentinel)
        return default if value is sentinel else value
    return default


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    value = _mapping_value(obj, key, default=_MISSING)
    if value is not _MISSING:
        return value
    return getattr(obj, key, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if isfinite(out) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rows(obs: Any, key: str) -> list[Any]:
    rows = _get(obs, key, [])
    if rows is None:
        return []
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    if not _is_non_string_sequence(rows):
        return []
    return list(rows)


def _row_value(row: Any, index: int, name: str, default: Any = 0) -> Any:
    value = _mapping_value(row, name, default=_MISSING)
    if value is not _MISSING:
        return value
    if hasattr(row, name):
        return getattr(row, name)
    try:
        return row[index]
    except (TypeError, IndexError, KeyError):
        return default


def get_player(obs: Any) -> int:
    return _as_int(
        _get(obs, "player", _get(obs, "player_id", _get(obs, "mark", 0))), 0
    )


def get_planets(obs: Any) -> list[dict[str, float | int]]:
    planets: list[dict[str, float | int]] = []
    for row in _rows(obs, "planets"):
        planet = {
            "id": _as_int(_row_value(row, 0, "id", len(planets))),
            "owner": _as_int(_row_value(row, 1, "owner", -1), -1),
            "ships": _as_float(_row_value(row, 5, "ships", 0.0)),
            "production": _as_float(_row_value(row, 6, "production", 0.0)),
        }
        planets.append(planet)
    return sorted(planets, key=lambda planet: int(planet["id"]))


def get_fleets(obs: Any) -> list[dict[str, float | int]]:
    fleets: list[dict[str, float | int]] = []
    for row in _rows(obs, "fleets"):
        fleet = {
            "id": _as_int(_row_value(row, 0, "id", len(fleets))),
            "owner": _as_int(_row_value(row, 1, "owner", -1), -1),
            "ships": _as_float(_row_value(row, 6, "ships", 0.0)),
        }
        fleets.append(fleet)
    return sorted(fleets, key=lambda fleet: int(fleet["id"]))


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


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


def compute_reward_components(
    previous_obs, current_obs, env_state, done: bool
) -> dict[str, float]:
    """
    Return bounded PPO reward components and raw terminal diagnostics.

    The PPO reward is intentionally based on bounded one-step deltas and discrete
    ownership events. The raw Kaggle terminal score is exposed as diagnostics only
    in ``reward_terminal_raw``; a small signed terminal outcome bonus is used for
    training instead of adding the raw score directly.
    """

    player = get_player(current_obs if current_obs is not None else previous_obs)

    prev = _summary(previous_obs)
    cur = _summary(current_obs)

    captures, losses = _ownership_changes(previous_obs, current_obs)

    delta_ship_adv = cur["ship_adv"] - prev["ship_adv"]
    delta_prod_adv = cur["prod_adv"] - prev["prod_adv"]
    delta_planet_adv = cur["planet_adv"] - prev["planet_adv"]

    reward_delta_advantage = _clip(
        (0.002 * delta_ship_adv)
        + (0.04 * delta_prod_adv)
        + (0.05 * delta_planet_adv),
        DELTA_ADVANTAGE_MIN,
        DELTA_ADVANTAGE_MAX,
    )
    reward_capture = _clip(0.30 * captures, 0.0, CAPTURE_REWARD_MAX)
    reward_loss = _clip(-0.40 * losses, LOSS_REWARD_MIN, 0.0)

    reward_terminal_raw = _state_reward(env_state, player) if done else 0.0
    reward_terminal = (
        TERMINAL_OUTCOME_SCALE * _sign(reward_terminal_raw) if done else 0.0
    )

    reward_total = _clip(
        reward_delta_advantage + reward_capture + reward_loss + reward_terminal,
        REWARD_MIN,
        REWARD_MAX,
    )

    return {
        "reward_delta_advantage": reward_delta_advantage,
        "reward_capture": reward_capture,
        "reward_loss": reward_loss,
        "reward_terminal": reward_terminal,
        "reward_terminal_raw": float(reward_terminal_raw),
        "reward_total": reward_total,
    }


def compute_reward(previous_obs, current_obs, env_state, done: bool) -> float:
    """Return the clipped PPO reward for a transition."""

    return compute_reward_components(previous_obs, current_obs, env_state, done)[
        "reward_total"
    ]
