"""Eight-dimensional SAC Basic environment around VirtualGardenCore."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from virtual_garden import VirtualGardenCore


WEATHER_FEATURES = [
    "precipitation_mm", "temperature_c", "relative_humidity_pct", "et0_mm",
    "vpd_kpa", "shortwave_radiation_w_m2",
]


@dataclass(frozen=True)
class RewardConfig:
    """Scalarized physical costs; defaults exactly reproduce reward_v2."""

    name: str = "reward_v2"
    tracking_weight: float = 100.0
    deficit_ratio: float = 2.0
    water_weight: float = 0.02
    smoothness_weight: float = 0.01
    violation_weight: float = 1.0

    def __post_init__(self) -> None:
        weights = (
            self.tracking_weight,
            self.deficit_ratio,
            self.water_weight,
            self.smoothness_weight,
            self.violation_weight,
        )
        if any(value < 0 or not np.isfinite(value) for value in weights):
            raise ValueError("reward weights must be finite and non-negative")


REWARD_V2_CONFIG = RewardConfig()
LOCKED_REWARD_CONFIG = RewardConfig(
    name="reward_v4",
    water_weight=0.01,
    violation_weight=2.0,
)


class SACIrrigationEnv:
    observation_dim = 8
    action_dim = 1
    reward_version = "reward_v4"

    def __init__(
        self,
        data: pd.DataFrame,
        normalizer: dict[str, dict[str, float]] | str | Path,
        episode_length: int = 336,
        seed: int = 11,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.normalizer = self._load_normalizer(normalizer)
        self.episode_length = min(episode_length, len(self.data))
        self.rng = np.random.default_rng(seed)
        self.reward_config = reward_config or LOCKED_REWARD_CONFIG
        self.reward_version = self.reward_config.name
        self.garden = VirtualGardenCore()
        self.reset(start_index=0)

    @staticmethod
    def _load_normalizer(value):
        if isinstance(value, (str, Path)):
            return json.loads(Path(value).read_text(encoding="utf-8"))
        return value

    def reset(self, start_index: int | None = None) -> np.ndarray:
        maximum_start = len(self.data) - self.episode_length
        self.start_index = (
            int(self.rng.integers(0, maximum_start + 1)) if start_index is None else int(start_index)
        )
        if not 0 <= self.start_index <= maximum_start:
            raise ValueError("episode start would cross the dataset boundary")
        self.index = self.start_index
        self.steps = 0
        self.previous_irrigation = 0.0
        self.garden.reset()
        return self._observation()

    def _scale(self, column: str, value: float) -> float:
        stats = self.normalizer[column]
        return (value - stats["mean"]) / max(stats["std"], 1e-8)

    def _observation(self) -> np.ndarray:
        row = self.data.iloc[self.index]
        values = [
            (self.garden.theta - 0.27) / 0.10,
            self._scale("precipitation_mm", row.precipitation_mm),
            self._scale("temperature_c", row.temperature_c),
            self._scale("relative_humidity_pct", row.relative_humidity_pct),
            self._scale("et0_mm", row.et0_mm),
            self._scale("vpd_kpa", row.vpd_kpa),
            self._scale("shortwave_radiation_w_m2", row.shortwave_radiation_w_m2),
            self.previous_irrigation / self.garden.config.max_irrigation_mm_h,
        ]
        return np.clip(np.asarray(values, dtype=np.float32), -5.0, 5.0)

    def step(self, action: float | np.ndarray):
        action = float(np.asarray(action).reshape(-1)[0])
        row = self.data.iloc[self.index]
        previous_action = self.previous_irrigation
        result = self.garden.step(row.precipitation_mm, row.et0_mm, action)
        action = result.irrigation_mm
        cfg = self.garden.config
        deficit_distance = max(cfg.target_min - result.theta, 0.0)
        surplus_distance = max(result.theta - cfg.target_max, 0.0)
        weights = self.reward_config
        violation = float(deficit_distance > 0 or surplus_distance > 0)
        reward_costs = {
            "tracking": weights.tracking_weight
            * (weights.deficit_ratio * deficit_distance + surplus_distance),
            "water": weights.water_weight * action,
            "smoothness": weights.smoothness_weight * abs(action - previous_action),
            "violation": weights.violation_weight * violation,
        }
        # The constant offset makes the default exactly equivalent to reward_v2.
        reward_offset = weights.violation_weight
        reward = reward_offset - sum(reward_costs.values())

        self.previous_irrigation = action
        self.steps += 1
        done = self.steps >= self.episode_length or self.index >= len(self.data) - 1
        if not done:
            self.index += 1
        next_observation = self._observation()
        info = {
            **result.as_dict(),
            "reward_version": self.reward_version,
            "reward_offset": reward_offset,
            **{f"reward_{name}_cost": value for name, value in reward_costs.items()},
        }
        return next_observation, float(reward), done, info
