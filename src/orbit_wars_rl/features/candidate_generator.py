"""Candidate action generation for the discrete RL policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from orbit_wars_rl.features.observation_encoder import get_planets, get_player

SHIP_FRACTIONS = (0.25, 0.5, 0.75)
MIN_SOURCE_SHIPS = 6
RESERVE_SHIPS = 2
PURPOSE_ORDER = ("capture_neutral", "attack_enemy", "reinforce")


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            continue
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
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


def _estimate_angular_motion(
    planet: dict[str, Any],
    obs: Any = None,
    config: Any = None,
) -> float | None:
    """Estimate a planet's angular velocity in radians/turn if metadata exists."""

    for container in (planet, obs, config):
        velocity = _finite_float(
            _field(
                container,
                "angular_velocity",
                "angularVelocity",
                "orbit_angular_velocity",
                "orbitAngularVelocity",
            )
        )
        if velocity is not None:
            return velocity
    return None


def _estimate_fleet_travel_time(
    source: dict[str, Any],
    target: dict[str, Any],
    obs: Any = None,
    config: Any = None,
) -> float | None:
    """Estimate fleet arrival time from source-target distance and fleet speed."""

    speed = None
    for container in (config, obs):
        speed = _finite_float(
            _field(
                container,
                "shipSpeed",
                "ship_speed",
                "fleetSpeed",
                "fleet_speed",
            )
        )
        if speed is not None:
            break

    if speed is None or speed <= 0.0:
        return None

    distance = math.hypot(
        float(target["x"]) - float(source["x"]),
        float(target["y"]) - float(source["y"]),
    )
    return distance / speed


def _predicted_target_position(
    target: dict[str, Any],
    travel_time: float,
    angular_velocity: float,
    obs: Any = None,
    config: Any = None,
) -> tuple[float, float]:
    """Project a target's orbital position at fleet arrival time."""

    center_x, center_y = _estimate_orbit_center(obs, config)
    current_x = float(target["x"])
    current_y = float(target["y"])
    radius_x = current_x - center_x
    radius_y = current_y - center_y
    delta = angular_velocity * travel_time
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)
    return (
        center_x + radius_x * cos_delta - radius_y * sin_delta,
        center_y + radius_x * sin_delta + radius_y * cos_delta,
    )


def _angle_between(
    source: dict[str, Any],
    target: dict[str, Any],
    obs: Any = None,
    config: Any = None,
) -> float:
    """Aim at the target's predicted arrival-time position when possible."""

    travel_time = _estimate_fleet_travel_time(source, target, obs, config)
    angular_velocity = _estimate_angular_motion(target, obs, config)
    if travel_time is None or angular_velocity is None:
        return _direct_angle(source, target)

    predicted_x, predicted_y = _predicted_target_position(
        target, travel_time, angular_velocity, obs, config
    )
    return float(
        math.atan2(predicted_y - float(source["y"]), predicted_x - float(source["x"]))
    )


def _uses_intercept_angle(
    source: dict[str, Any],
    target: dict[str, Any],
    obs: Any = None,
    config: Any = None,
) -> bool:
    return _estimate_fleet_travel_time(source, target, obs, config) is not None and (
        _estimate_angular_motion(target, obs, config) is not None
    )


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
) -> dict[str, Any] | None:
    ships = int(math.floor(available * fraction))
    if ships <= 0:
        return None

    distance = math.hypot(
        float(target["x"]) - float(source["x"]),
        float(target["y"]) - float(source["y"]),
    )
    target_owner = int(target["owner"])
    purpose = _target_purpose(target, player)
    angle = _angle_between(source, target, obs, config)
    intercept_angle_used = _uses_intercept_angle(source, target, obs, config)
    return {
        "type": "send",
        "from_planet_id": int(source["id"]),
        "target_planet_id": int(target["id"]),
        "ship_fraction": float(fraction),
        "ships": int(ships),
        "angle": angle,
        "intercept_angle_used": bool(intercept_angle_used),
        "purpose": purpose,
        "target_owner": target_owner,
        "candidate_features": [
            1.0,  # send action; NOOP/padding encode as all zeros
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
                    source, target, fraction, available, player, obs, config
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

    # Reserve at least one slot for each available target purpose before source
    # balancing, so small candidate lists still expose enemy attacks and captures.
    for purpose in PURPOSE_ORDER:
        if purpose_quotas[purpose] <= 0:
            continue
        candidate = first_unused(
            available_by_purpose[purpose], prefer_uncovered_source=True
        )
        if candidate is not None:
            add_candidate(candidate)

    # Balance across sources next: each owned source with legal sends contributes
    # once when the candidate budget allows it, before any source is repeated for
    # extra target/fraction variants.
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

    # Fill reserved purpose capacity with the strongest remaining source options,
    # then use any legal leftovers if source balancing consumed a purpose quota.
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
