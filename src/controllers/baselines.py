"""Deterministic non-RL irrigation baselines with one shared interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    theta: float
    precipitation_mm: float
    et0_mm: float
    forecast_precipitation_h1_mm: float = 0.0


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def select_action(self, observation: Observation) -> float: ...


class NoIrrigationController:
    name = "No Irrigation"

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> float:
        return 0.0


class FixedScheduleController:
    name = "Fixed Schedule"

    def __init__(self, hour: int = 6, amount_mm: float = 3.0) -> None:
        self.hour = hour
        self.amount_mm = amount_mm

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> float:
        return self.amount_mm if observation.timestamp.hour == self.hour else 0.0


class ThresholdController:
    name = "Threshold-Based"

    def __init__(self, trigger_theta: float = 0.25, amount_mm: float = 5.0) -> None:
        self.trigger_theta = trigger_theta
        self.amount_mm = amount_mm

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> float:
        return self.amount_mm if observation.theta < self.trigger_theta else 0.0


class RuleBasedForecastController:
    name = "Rule-Based Forecast-Aware"

    def __init__(
        self,
        trigger_theta: float = 0.25,
        rain_skip_mm: float = 1.0,
        amount_mm: float = 5.0,
    ) -> None:
        self.trigger_theta = trigger_theta
        self.rain_skip_mm = rain_skip_mm
        self.amount_mm = amount_mm

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> float:
        needs_water = observation.theta < self.trigger_theta
        rain_expected = observation.forecast_precipitation_h1_mm >= self.rain_skip_mm
        return self.amount_mm if needs_water and not rain_expected else 0.0


class FuzzyController:
    name = "Fuzzy Controller"

    def __init__(self, dry_theta: float = 0.22, wet_theta: float = 0.28) -> None:
        self.dry_theta = dry_theta
        self.wet_theta = wet_theta

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> float:
        dryness = min(max((self.wet_theta - observation.theta) / (
            self.wet_theta - self.dry_theta
        ), 0.0), 1.0)
        rain_suppression = min(max(observation.forecast_precipitation_h1_mm / 2.0, 0.0), 1.0)
        return 5.0 * dryness * (1.0 - rain_suppression)
