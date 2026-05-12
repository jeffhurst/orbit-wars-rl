from __future__ import annotations

from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomTensorboardCallback(BaseCallback):
    """
    Reads metrics from env info dicts and records rolling averages to TensorBoard.

    The env should return:
        info["custom_metrics"] = {... per-step metrics ...}
        info["episode_metrics"] = {... per-episode metrics ...} when episode ends
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self.step_buffers = defaultdict(list)
        self.episode_buffers = defaultdict(list)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            custom_metrics = info.get("custom_metrics", {})
            for key, value in custom_metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.step_buffers[key].append(float(value))

            episode_metrics = info.get("episode_metrics", {})
            for key, value in episode_metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.episode_buffers[key].append(float(value))

        if self.n_calls % self.log_freq == 0:
            self._dump_step_metrics()
            self._dump_episode_metrics()

        return True

    def _dump_step_metrics(self) -> None:
        for key, values in self.step_buffers.items():
            if not values:
                continue

            self.logger.record(f"custom/{key}", float(np.mean(values)))

        self.step_buffers.clear()

    def _dump_episode_metrics(self) -> None:
        for key, values in self.episode_buffers.items():
            if not values:
                continue

            self.logger.record(f"custom/{key}", float(np.mean(values)))

        self.episode_buffers.clear()
