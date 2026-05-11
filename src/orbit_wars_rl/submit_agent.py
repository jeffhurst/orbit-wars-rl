"""Kaggle-compatible Orbit Wars submission agent."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from orbit_wars_rl.features.action_decoder import decode_candidate
from orbit_wars_rl.features.candidate_generator import generate_candidates
from orbit_wars_rl.features.observation_encoder import encode_observation
from orbit_wars_rl.opponents.starter_bot import agent as starter_agent

MAX_CANDIDATES = 64
MAX_PLANETS = 48
MAX_FLEETS = 128
_MODEL = None
_MODEL_LOAD_ATTEMPTED = False


def _model_path() -> Path:
    return Path(os.environ.get("ORBIT_WARS_MODEL", "models/orbit_wars_maskableppo_v1.zip"))


def _load_model():
    global _MODEL, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL
    _MODEL_LOAD_ATTEMPTED = True
    if importlib.util.find_spec("sb3_contrib") is None:
        return None
    from sb3_contrib import MaskablePPO

    path = _model_path()
    if path.exists():
        try:
            _MODEL = MaskablePPO.load(str(path))
        except Exception:
            _MODEL = None
    return _MODEL


def agent(obs, config=None):
    """Return a Kaggle action list, falling back safely on any failure."""

    try:
        model = _load_model()
        if model is None:
            return starter_agent(obs, config)
        candidates = generate_candidates(obs, MAX_CANDIDATES)
        mask = np.zeros(MAX_CANDIDATES, dtype=bool)
        mask[: min(len(candidates), MAX_CANDIDATES)] = True
        mask[0] = True
        encoded = encode_observation(obs, MAX_PLANETS, MAX_FLEETS)
        action, _ = model.predict(encoded, deterministic=True, action_masks=mask)
        return decode_candidate(int(action), candidates)
    except Exception:
        return []
