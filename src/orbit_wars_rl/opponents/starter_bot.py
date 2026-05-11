"""Simple deterministic starter baseline for Orbit Wars."""

from __future__ import annotations

from orbit_wars_rl.features.action_decoder import decode_candidate
from orbit_wars_rl.features.candidate_generator import generate_candidates


def agent(obs, config=None):
    candidates = generate_candidates(obs, max_candidates=48)
    for idx, candidate in enumerate(candidates):
        if candidate.get("type") == "send" and candidate.get("ship_fraction") == 0.5:
            return decode_candidate(idx, candidates)
    return []
