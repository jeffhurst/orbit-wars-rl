"""Greedy deterministic baseline that prefers cheap neutral planets."""

from __future__ import annotations

from orbit_wars_rl.features.action_decoder import decode_candidate
from orbit_wars_rl.features.candidate_generator import generate_candidates
from orbit_wars_rl.features.observation_encoder import get_planets, get_player


def agent(obs, config=None):
    player = get_player(obs)
    planets = {int(p["id"]): p for p in get_planets(obs)}
    candidates = generate_candidates(obs, max_candidates=64)
    best_idx = 0
    best_key = None
    for idx, candidate in enumerate(candidates):
        if candidate.get("type") != "send":
            continue
        target = planets.get(int(candidate.get("target_planet_id", -1)))
        if target is None:
            continue
        owner = int(target["owner"])
        group = 0 if owner == -1 else (1 if owner != player else 2)
        key = (group, float(target["ships"]), -int(candidate.get("ships", 0)))
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx
    return decode_candidate(best_idx, candidates)
