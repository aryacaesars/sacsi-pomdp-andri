"""Causal sequence wrapper for the eight-dimensional SAC environment."""

from __future__ import annotations

from collections import deque

import numpy as np

from sac_basic.environment import SACIrrigationEnv


class SACLSTMEnv(SACIrrigationEnv):
    def __init__(self, *args, sequence_length: int = 24, **kwargs) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self._history = deque(maxlen=sequence_length)
        super().__init__(*args, **kwargs)

    def reset(self, start_index: int | None = None):
        current = super().reset(start_index)
        self._history.clear()
        self._history.extend(np.zeros_like(current) for _ in range(self.sequence_length - 1))
        self._history.append(current.copy())
        return current, np.stack(self._history)

    def step(self, action):
        current, reward, done, info = super().step(action)
        self._history.append(current.copy())
        return (current, np.stack(self._history)), reward, done, info
