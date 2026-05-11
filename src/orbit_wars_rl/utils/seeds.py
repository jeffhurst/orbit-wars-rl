"""Random seed helpers."""

from __future__ import annotations

import random

import numpy as np


def set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
