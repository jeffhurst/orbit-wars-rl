"""Random lightweight Orbit Wars opponent."""

from __future__ import annotations

import random

from orbit_wars_rl.features.action_decoder import decode_candidate
from orbit_wars_rl.features.candidate_generator import generate_candidates


def agent(obs, config=None):
    if random.random() < 0.35:
        return []
    candidates = generate_candidates(obs, max_candidates=24)
    if len(candidates) <= 1:
        return []
    return decode_candidate(random.randrange(1, len(candidates)), candidates)
