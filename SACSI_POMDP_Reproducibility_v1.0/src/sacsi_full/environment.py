"""Current 8-D + causal history 24×8 + SF-20 forecast 3-D environment."""

from __future__ import annotations

import numpy as np

from sac_lstm.environment import SACLSTMEnv


class SACSIEnv(SACLSTMEnv):
    forecast_protocol = "SF-20_h1_controlled_proxy"

    def _forecast(self) -> np.ndarray:
        row = self.data.iloc[self.index]
        return np.clip(np.asarray([
            self._scale("precipitation_mm", row.forecast_precipitation_mm),
            self._scale("et0_mm", row.forecast_et0_mm),
            self._scale("temperature_c", row.forecast_temperature_c),
        ], np.float32), -5, 5)

    def reset(self, start_index: int | None = None):
        current, history = super().reset(start_index)
        return current, history, self._forecast()

    def step(self, action):
        (current, history), reward, done, info = super().step(action)
        return (current, history, self._forecast()), reward, done, info
