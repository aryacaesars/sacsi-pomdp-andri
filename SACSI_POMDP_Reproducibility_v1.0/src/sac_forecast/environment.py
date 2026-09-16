"""SAC environment using current observation plus SF-20 h+1 context."""

from __future__ import annotations

import numpy as np

from sac_basic.environment import SACIrrigationEnv


class SACForecastEnv(SACIrrigationEnv):
    observation_dim = 11
    forecast_protocol = "SF-20_h1_controlled_proxy"

    def _observation(self) -> np.ndarray:
        current = super()._observation()
        row = self.data.iloc[self.index]
        forecast = np.asarray([
            self._scale("precipitation_mm", row.forecast_precipitation_mm),
            self._scale("et0_mm", row.forecast_et0_mm),
            self._scale("temperature_c", row.forecast_temperature_c),
        ], dtype=np.float32)
        return np.clip(np.concatenate((current, forecast)), -5.0, 5.0)
