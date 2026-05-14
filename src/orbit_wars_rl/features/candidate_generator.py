"""Candidate action generation for the discrete RL policy."""

from __future__ import annotations

import math
from typing import Any

from orbit_wars_rl.features.observation_encoder import get_planets, get_player

SHIP_FRACTIONS = (0.25, 0.5, 0.75)
MIN_SOURCE_SHIPS = 6
RESERVE_SHIPS = 2
PURPOSE_ORDER = ("capture_neutral", "attack_enemy", "reinforce")
DEFAULT_BOARD_SIZE = 100.0
DEFAULT_MAX_FLEET_SPEED = 6.0
DEFAULT_SUN_RADIUS = 5.0
TRAJECTORY_SAMPLE_STEP = 0.5
MAX_INTERCEPT_TIME = 500.0
_MISSING = object()


def _is_dict(obj: Any) -> bool:
    return issubclass(type(obj), dict)


def _direct_attr(obj: Any, name: str, default: Any = _MISSING) -> Any:
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return default


def _is_non_string_sequence(obj: Any) -> bool:
    obj_type = type(obj)
    if issubclass(obj_type, (str, bytes, bytearray)):
        return False
    if issubclass(obj_type, (list, tuple)):
        return True
    return (
        _direct_attr(obj, "__len__") is not _MISSING
        and _direct_attr(obj, "__getitem__") is not _MISSING
    )


def _mapping_value(obj: Any, name: str, default: Any = None) -> Any:
    if _is_dict(obj):
        return obj[name] if name in obj else default
    getter = _direct_attr(obj, "get", default=None)
    if callable(getter):
        sentinel = object()
        value = getter(name, sentinel)
        return default if value is sentinel else value
    return default


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            continue
        value = _mapping_value(obj, name, default=_MISSING)
        if value is not _MISSING:
            return value
        if callable(_direct_attr(obj, "get", default=None)):
            continue
        value = _direct_attr(obj, name, default=_MISSING)
        if value is not _MISSING:
            return value
    return default


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _direct_angle(source: dict[str, Any], target: dict[str, Any]) -> float:
    return float(
        math.atan2(
            float(target["y"]) - float(source["y"]),
            float(target["x"]) - float(source["x"]),
        )
    )


def _estimate_orbit_center(obs: Any = None, config: Any = None) -> tuple[float, float]:
    """Return the known/assumed orbit center used to rotate planets."""

    for container in (config, obs):
        x = _finite_float(
            _field(container, "orbitCenterX", "orbit_center_x", "center_x")
        )
        y = _finite_float(
            _field(container, "orbitCenterY", "orbit_center_y", "center_y")
        )
        if x is not None and y is not None:
            return x, y
    return 50.0, 50.0


def _board_size(obs: Any = None, config: Any = None) -> float:
    for container in (config, obs):
        for names in (("boardSize", "board_size"), ("width",), ("height",)):
            value = _finite_float(_field(container, *names))
            if value is not None and value > 0.0:
                return value
    return DEFAULT_BOARD_SIZE


def _sun_radius(obs: Any = None, config: Any = None) -> float:
    for container in (config, obs):
        value = _finite_float(
            _field(container, "sunRadius", "sun_radius", "starRadius", "star_radius")
        )
        if value is not None and value >= 0.0:
            return value
    return DEFAULT_SUN_RADIUS


def _max_fleet_speed(obs: Any = None, config: Any = None) -> float:
    for container in (config, obs):
        value = _finite_float(
            _field(
                container,
                "maxSpeed",
                "max_speed",
                "maxFleetSpeed",
                "max_fleet_speed",
            )
        )
        if value is not None and value > 0.0:
            return value
    # Older tests/configs may expose a constant ship speed. Treat it as the max
    # speed so one-ship fleets still use the documented 1.0 floor.
    for container in (config, obs):
        value = _finite_float(
            _field(container, "shipSpeed", "ship_speed", "fleetSpeed", "fleet_speed")
        )
        if value is not None and value > 0.0:
            return value
    return DEFAULT_MAX_FLEET_SPEED


def _fleet_speed(ships: int, obs: Any = None, config: Any = None) -> float:
    """Return fleet speed from the Orbit Wars logarithmic size curve."""

    max_speed = _max_fleet_speed(obs, config)
    if ships <= 1:
        return 1.0
    scale = min(max(math.log(float(ships)) / math.log(1000.0), 0.0), 1.0)
    return 1.0 + (max_speed - 1.0) * (scale**1.5)


def _row_id(row: Any, default: int) -> int:
    value = _field(row, "id", default=None)
    if value is None and _is_non_string_sequence(row):
        try:
            value = row[0]
        except (IndexError, TypeError):
            value = default
    row_id = _finite_int(value)
    return row_id if row_id is not None else default


def _initial_planet_by_id(obs: Any, planet_id: int) -> Any | None:
    initial_rows = _field(obs, "initial_planets", "initialPlanets", default=None)
    if initial_rows is None:
        return None
    if _is_dict(initial_rows) or callable(_direct_attr(initial_rows, "get", None)):
        value = _mapping_value(initial_rows, planet_id, default=None)
        return (
            _mapping_value(initial_rows, str(planet_id), default=None)
            if value is None
            else value
        )
    if _is_non_string_sequence(initial_rows):
        for index, row in enumerate(initial_rows):
            if _row_id(row, index) == planet_id:
                return row
    return None


def _row_xy_radius(row: Any, fallback: dict[str, Any]) -> tuple[float, float, float]:
    x = _finite_float(_field(row, "x", default=None))
    y = _finite_float(_field(row, "y", default=None))
    radius = _finite_float(_field(row, "radius", default=None))
    if _is_non_string_sequence(row):
        if x is None and len(row) > 2:
            x = _finite_float(row[2])
        if y is None and len(row) > 3:
            y = _finite_float(row[3])
        if radius is None and len(row) > 4:
            radius = _finite_float(row[4])
    return (
        float(fallback["x"]) if x is None else x,
        float(fallback["y"]) if y is None else y,
        float(fallback.get("radius", 0.0)) if radius is None else radius,
    )


def _angular_velocity_from_collection(collection: Any, planet_id: int) -> float | None:
    if collection is None:
        return None
    scalar = _finite_float(collection)
    if scalar is not None:
        return scalar
    if _is_dict(collection) or callable(_direct_attr(collection, "get", None)):
        value = _mapping_value(collection, planet_id, default=None)
        return _finite_float(
            _mapping_value(collection, str(planet_id), default=None)
            if value is None
            else value
        )
    if _is_non_string_sequence(collection):
        if planet_id < len(collection):
            return _finite_float(collection[planet_id])
        for index, row in enumerate(collection):
            row_id = _row_id(row, index)
            if row_id == planet_id:
                value = _finite_float(
                    _field(row, "angular_velocity", "angularVelocity", default=None)
                )
                if value is not None:
                    return value
                if _is_non_string_sequence(row):
                    for item in row[1:]:
                        value = _finite_float(item)
                        if value is not None:
                            return value
    return None


def _estimate_angular_motion(
    planet: dict[str, Any],
    obs: Any = None,
    config: Any = None,
) -> float | None:
    """Estimate a planet's angular velocity in radians/turn if metadata exists."""

    planet_id = int(planet["id"])
    initial_planet = _initial_planet_by_id(obs, planet_id)
    for container in (planet, initial_planet, obs, config):
        velocity = _finite_float(
            _field(
                container,
                "angular_velocity",
                "angularVelocity",
                "orbit_angular_velocity",
                "orbitAngularVelocity",
                default=None,
            )
        )
        if velocity is not None:
            return velocity
    for container in (obs, config):
        velocity = _angular_velocity_from_collection(
            _field(container, "angular_velocity", "angularVelocity", default=None),
            planet_id,
        )
        if velocity is not None:
            return velocity
    return None


def _current_step(obs: Any = None) -> float:
    return _finite_float(_field(obs, "step", "turn", default=0.0),) or 0.0


def _is_orbiting_planet(
    planet: dict[str, Any], obs: Any = None, config: Any = None
) -> bool:
    center_x, center_y = _estimate_orbit_center(obs, config)
    initial = _initial_planet_by_id(obs, int(planet["id"]))
    x, y, radius = _row_xy_radius(initial, planet) if initial is not None else (
        float(planet["x"]),
        float(planet["y"]),
        float(planet.get("radius", 0.0)),
    )
    orbital_radius = math.hypot(x - center_x, y - center_y)
    return orbital_radius + radius < 50.0


def _planet_position_at(
    planet: dict[str, Any],
    turns_from_now: float,
    obs: Any = None,
    config: Any = None,
) -> tuple[float, float]:
    velocity = _estimate_angular_motion(planet, obs, config)
    if velocity is None or not _is_orbiting_planet(planet, obs, config):
        return float(planet["x"]), float(planet["y"])

    center_x, center_y = _estimate_orbit_center(obs, config)
    initial = _initial_planet_by_id(obs, int(planet["id"]))
    if initial is not None:
        base_x, base_y, _ = _row_xy_radius(initial, planet)
        delta = velocity * (_current_step(obs) + turns_from_now)
    else:
        base_x = float(planet["x"])
        base_y = float(planet["y"])
        delta = velocity * turns_from_now
    radius_x = base_x - center_x
    radius_y = base_y - center_y
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)
    return (
        center_x + radius_x * cos_delta - radius_y * sin_delta,
        center_y + radius_x * sin_delta + radius_y * cos_delta,
    )


def _intercept_solution(
    source: dict[str, Any],
    target: dict[str, Any],
    ships: int,
    obs: Any = None,
    config: Any = None,
) -> tuple[float, float, float, bool] | None:
    """Return (angle, travel_time, distance, used_intercept) for a valid shot."""

    speed = _fleet_speed(ships, obs, config)
    if speed <= 0.0:
        return None

    source_x = float(source["x"])
    source_y = float(source["y"])
    moving_target = (
        _estimate_angular_motion(target, obs, config) is not None
        and _is_orbiting_planet(target, obs, config)
    )

    if not moving_target:
        target_x, target_y = float(target["x"]), float(target["y"])
        distance = math.hypot(target_x - source_x, target_y - source_y)
        if distance <= 0.0:
            return None
        return (
            math.atan2(target_y - source_y, target_x - source_x),
            distance / speed,
            distance,
            False,
        )

    def residual(t: float) -> float:
        target_x, target_y = _planet_position_at(target, t, obs, config)
        return math.hypot(target_x - source_x, target_y - source_y) / speed - t

    low = 0.0
    high = max(residual(0.0), 1.0)
    horizon = min(
        MAX_INTERCEPT_TIME,
        max(float(_field(obs, "episodeSteps", default=500.0) or 500.0), 1.0),
    )
    while high < horizon and residual(high) > 0.0:
        high *= 2.0
    if high > horizon and residual(horizon) > 0.0:
        return None
    high = min(high, horizon)

    for _ in range(48):
        mid = (low + high) / 2.0
        if residual(mid) > 0.0:
            low = mid
        else:
            high = mid
    travel_time = high
    target_x, target_y = _planet_position_at(target, travel_time, obs, config)
    distance = math.hypot(target_x - source_x, target_y - source_y)
    if distance <= 0.0 or abs(distance / speed - travel_time) > 0.05:
        return None
    return (
        math.atan2(target_y - source_y, target_x - source_x),
        travel_time,
        distance,
        True,
    )


def _segment_intersects_circle(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    if radius <= 0.0:
        return False
    sx, sy = start
    ex, ey = end
    cx, cy = center
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.hypot(cx - sx, cy - sy) <= radius
    projection = ((cx - sx) * dx + (cy - sy) * dy) / length_sq
    projection = max(0.0, min(1.0, projection))
    nearest_x = sx + projection * dx
    nearest_y = sy + projection * dy
    return math.hypot(cx - nearest_x, cy - nearest_y) <= radius


def _trajectory_is_safe(
    source: dict[str, Any],
    target: dict[str, Any],
    angle: float,
    travel_time: float,
    distance: float,
    speed: float,
    planets: list[dict[str, Any]],
    obs: Any = None,
    config: Any = None,
) -> bool:
    source_x = float(source["x"])
    source_y = float(source["y"])
    end_x = source_x + math.cos(angle) * distance
    end_y = source_y + math.sin(angle) * distance
    board_size = _board_size(obs, config)
    if not (0.0 <= end_x <= board_size and 0.0 <= end_y <= board_size):
        return False
    sun_center = _estimate_orbit_center(obs, config)
    sun_radius = _sun_radius(obs, config)
    source_inside_sun = (
        math.hypot(source_x - sun_center[0], source_y - sun_center[1]) <= sun_radius
    )
    target_inside_sun = (
        math.hypot(end_x - sun_center[0], end_y - sun_center[1]) <= sun_radius
    )
    if (
        not source_inside_sun
        and not target_inside_sun
        and _segment_intersects_circle(
            (source_x, source_y),
            (end_x, end_y),
            sun_center,
            sun_radius,
        )
    ):
        return False

    source_id = int(source["id"])
    target_id = int(target["id"])
    sample_count = max(1, int(math.ceil(travel_time / TRAJECTORY_SAMPLE_STEP)))
    source_clear_time = float(source.get("radius", 0.0)) / max(speed, 1.0e-9)
    for i in range(sample_count + 1):
        t = min(travel_time, i * travel_time / sample_count)
        fleet_x = source_x + math.cos(angle) * speed * t
        fleet_y = source_y + math.sin(angle) * speed * t
        if not (0.0 <= fleet_x <= board_size and 0.0 <= fleet_y <= board_size):
            return False
        for planet in planets:
            planet_id = int(planet["id"])
            if planet_id == source_id and t <= source_clear_time + 0.25:
                continue
            if planet_id == target_id:
                continue
            px, py = _planet_position_at(planet, t, obs, config)
            radius = float(planet.get("radius", 0.0))
            if math.hypot(px - fleet_x, py - fleet_y) <= radius:
                return False
    return True


def _target_priority(planet: dict[str, Any], player: int) -> tuple[int, float, int]:
    owner = int(planet["owner"])
    if owner == -1:
        group = 0
    elif owner != player:
        group = 1
    else:
        group = 2
    return group, float(planet["ships"]), int(planet["id"])


def _target_purpose(target: dict[str, Any], player: int) -> str:
    target_owner = int(target["owner"])
    if target_owner == -1:
        return "capture_neutral"
    if target_owner != player:
        return "attack_enemy"
    return "reinforce"


def _build_send_candidate(
    source: dict[str, Any],
    target: dict[str, Any],
    fraction: float,
    available: int,
    player: int,
    obs: dict,
    config: Any = None,
    planets: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    ships = int(math.floor(available * fraction))
    if ships <= 0:
        return None

    solution = _intercept_solution(source, target, ships, obs, config)
    if solution is None:
        return None
    angle, travel_time, distance, intercept_angle_used = solution
    speed = _fleet_speed(ships, obs, config)
    planet_rows = planets or get_planets(obs)
    if not _trajectory_is_safe(
        source, target, angle, travel_time, distance, speed, planet_rows, obs, config
    ):
        return None

    target_owner = int(target["owner"])
    purpose = _target_purpose(target, player)
    return {
        "type": "send",
        "from_planet_id": int(source["id"]),
        "target_planet_id": int(target["id"]),
        "ship_fraction": float(fraction),
        "ships": int(ships),
        "fleet_speed": float(speed),
        "travel_time": float(travel_time),
        "angle": float(angle),
        "intercept_angle_used": bool(intercept_angle_used),
        "purpose": purpose,
        "target_owner": target_owner,
        "candidate_features": [
            1.0,
            float(fraction),
            min(float(source["ships"]) / 500.0, 5.0),
            min(float(target["ships"]) / 500.0, 5.0),
            min(distance / 150.0, 2.0),
            1.0 if target_owner == -1 else 0.0,
            1.0 if target_owner >= 0 and target_owner != player else 0.0,
            1.0 if target_owner == player else 0.0,
            float(source["x"]) / 100.0,
            float(source["y"]) / 100.0,
            float(target["x"]) / 100.0,
            float(target["y"]) / 100.0,
        ],
    }


def _purpose_slot_counts(
    available_by_purpose: dict[str, list[dict[str, Any]]], slots: int
) -> dict[str, int]:
    """Reserve candidate capacity across target purposes that have options."""

    quotas = {purpose: 0 for purpose in PURPOSE_ORDER}
    if slots <= 0:
        return quotas

    active = [purpose for purpose in PURPOSE_ORDER if available_by_purpose[purpose]]
    remaining = slots
    for purpose in active:
        if remaining <= 0:
            break
        quotas[purpose] += 1
        remaining -= 1

    purpose_index = 0
    while remaining > 0 and active:
        purpose = active[purpose_index % len(active)]
        quotas[purpose] += 1
        remaining -= 1
        purpose_index += 1

    return quotas


def generate_candidates(
    obs: dict, max_candidates: int, config: Any = None
) -> list[dict]:
    """Generate legal/plausible candidate actions.

    Candidate 0 is always NOOP. Non-NOOP candidates launch one fleet from an
    owned planet toward a neutral, enemy, or friendly reinforcement target.
    Unsafe launches that would hit the sun, leave the board, or collide with an
    intervening planet are filtered out before the policy can select them.
    """

    if max_candidates <= 0:
        return []

    candidates: list[dict] = [{"type": "noop"}]
    send_slots = max_candidates - 1
    if send_slots <= 0:
        return candidates

    planets = get_planets(obs)
    player = get_player(obs)
    owned = [
        p
        for p in planets
        if int(p["owner"]) == player and float(p["ships"]) >= MIN_SOURCE_SHIPS
    ]
    owned.sort(key=lambda p: (-float(p["ships"]), int(p["id"])))
    targets = sorted(planets, key=lambda p: _target_priority(p, player))

    available_by_purpose: dict[str, list[dict[str, Any]]] = {
        purpose: [] for purpose in PURPOSE_ORDER
    }
    available_by_source: dict[int, list[dict[str, Any]]] = {}
    for source in owned:
        source_id = int(source["id"])
        available = max(int(float(source["ships"])) - RESERVE_SHIPS, 0)
        if available <= 0:
            continue

        for target in targets:
            if int(target["id"]) == source_id:
                continue
            for fraction in SHIP_FRACTIONS:
                candidate = _build_send_candidate(
                    source, target, fraction, available, player, obs, config, planets
                )
                if candidate is None:
                    continue
                purpose = str(candidate["purpose"])
                available_by_purpose[purpose].append(candidate)
                available_by_source.setdefault(source_id, []).append(candidate)

    purpose_quotas = _purpose_slot_counts(available_by_purpose, send_slots)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int, float]] = set()
    purpose_counts = {purpose: 0 for purpose in PURPOSE_ORDER}
    source_counts = {source_id: 0 for source_id in available_by_source}

    def add_candidate(candidate: dict[str, Any]) -> bool:
        if len(selected) >= send_slots:
            return False
        key = (
            int(candidate["from_planet_id"]),
            int(candidate["target_planet_id"]),
            float(candidate["ship_fraction"]),
        )
        if key in selected_keys:
            return False
        selected.append(candidate)
        selected_keys.add(key)
        purpose_counts[str(candidate["purpose"])] += 1
        source_counts[int(candidate["from_planet_id"])] += 1
        return True

    def first_unused(
        pool: list[dict[str, Any]],
        respect_purpose_quota: bool = False,
        prefer_uncovered_source: bool = False,
    ) -> dict[str, Any] | None:
        fallback = None
        for candidate in pool:
            key = (
                int(candidate["from_planet_id"]),
                int(candidate["target_planet_id"]),
                float(candidate["ship_fraction"]),
            )
            if key in selected_keys:
                continue
            purpose = str(candidate["purpose"])
            if (
                respect_purpose_quota
                and purpose_counts[purpose] >= purpose_quotas[purpose]
            ):
                continue
            source_id = int(candidate["from_planet_id"])
            if prefer_uncovered_source and source_counts[source_id] > 0:
                fallback = fallback or candidate
                continue
            return candidate
        return fallback

    for purpose in PURPOSE_ORDER:
        if purpose_quotas[purpose] <= 0:
            continue
        candidate = first_unused(
            available_by_purpose[purpose], prefer_uncovered_source=True
        )
        if candidate is not None:
            add_candidate(candidate)

    for source in owned:
        if len(selected) >= send_slots:
            break
        source_id = int(source["id"])
        if source_counts.get(source_id, 0) > 0:
            continue
        candidate = first_unused(
            available_by_source.get(source_id, []), respect_purpose_quota=True
        ) or first_unused(available_by_source.get(source_id, []))
        if candidate is not None:
            add_candidate(candidate)

    for purpose in PURPOSE_ORDER:
        while (
            len(selected) < send_slots
            and purpose_counts[purpose] < purpose_quotas[purpose]
        ):
            candidate = first_unused(available_by_purpose[purpose])
            if candidate is None:
                break
            add_candidate(candidate)

    all_candidates = [
        candidate
        for purpose in PURPOSE_ORDER
        for candidate in available_by_purpose[purpose]
    ]
    while len(selected) < send_slots:
        candidate = first_unused(all_candidates)
        if candidate is None:
            break
        add_candidate(candidate)

    candidates.extend(selected)
    return candidates
