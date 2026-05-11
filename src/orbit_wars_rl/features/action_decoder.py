"""Decode a selected candidate index into a Kaggle Orbit Wars action."""

from __future__ import annotations

import math
from typing import Any


def decode_candidate(action_index: int, candidates: list[dict]) -> list[list]:
    """Return [[from_planet_id, angle, ships]] or [] for invalid/NOOP choices."""

    try:
        idx = int(action_index)
    except (TypeError, ValueError):
        return []
    if idx < 0 or idx >= len(candidates):
        return []

    candidate = candidates[idx] or {}
    if candidate.get("type") != "send":
        return []

    try:
        from_planet_id = int(candidate["from_planet_id"])
        angle = float(candidate["angle"])
        ships = int(candidate["ships"])
    except (KeyError, TypeError, ValueError):
        return []

    if from_planet_id < 0 or ships <= 0 or not math.isfinite(angle):
        return []
    return [[from_planet_id, angle, ships]]
