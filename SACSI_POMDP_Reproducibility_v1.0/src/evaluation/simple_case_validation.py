"""Deterministic Module 8C simple cases and locked 2024 raw episodes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from controllers import Observation, RuleBasedForecastController, ThresholdController
from virtual_garden import VirtualGardenCore


RAW_EPISODES = {
    "dry": ("2024-04-16 00:00:00", "2024-04-29 23:00:00", 0.3),
    "wet": ("2024-11-27 00:00:00", "2024-12-10 23:00:00", 419.4),
    "mixed": ("2024-12-17 00:00:00", "2024-12-30 23:00:00", 77.4),
}


def _simulate(
    case_id: str,
    initial_theta: float,
    rain: np.ndarray,
    et0: np.ndarray,
    action: Callable[[int, Observation], float],
    forecast_rain: np.ndarray | None = None,
) -> pd.DataFrame:
    garden = VirtualGardenCore()
    garden.reset(initial_theta)
    timestamps = pd.date_range("2024-01-01", periods=len(rain), freq="h")
    forecasts = np.zeros(len(rain)) if forecast_rain is None else forecast_rain
    records = []
    for hour, timestamp in enumerate(timestamps):
        theta_before = garden.theta
        observation = Observation(
            timestamp.to_pydatetime(), theta_before, float(rain[hour]),
            float(et0[hour]), float(forecasts[hour]),
        )
        result = garden.step(rain[hour], et0[hour], action(hour, observation))
        records.append({
            "case_id": case_id,
            "timestamp": timestamp,
            "theta_before": theta_before,
            **result.as_dict(),
        })
    return pd.DataFrame(records)


def run_simple_cases() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    threshold = ThresholdController()
    forecast_guard = RuleBasedForecastController()
    zeros_24, et_24 = np.zeros(24), np.full(24, 0.4)
    rain_pulse = zeros_24.copy()
    rain_pulse[12] = 10.0

    logs = {
        "C1": _simulate("C1", 0.27, np.zeros(48), np.full(48, 0.4), lambda *_: 0.0),
        "C2": _simulate("C2", 0.27, rain_pulse, et_24, lambda *_: 0.0),
        "C3": _simulate("C3", 0.27, zeros_24, et_24, lambda hour, _: 5.0 if hour == 12 else 0.0),
        "C4": _simulate("C4", 0.319, np.zeros(12), np.full(12, 0.2), lambda _, obs: threshold.select_action(obs)),
        "C5": _simulate("C5", 0.20, np.zeros(12), np.full(12, 0.2), lambda _, obs: threshold.select_action(obs)),
        "C6": _simulate(
            "C6", 0.20, np.array([25.0, 0.0, 0.0]), np.full(3, 0.2),
            lambda _, obs: forecast_guard.select_action(obs), np.array([25.0, 0.0, 0.0]),
        ),
    }
    expected = {
        "C1": "theta decreases without rain or irrigation",
        "C2": "rain pulse increases theta",
        "C3": "bounded irrigation pulse increases theta",
        "C4": "near-upper-band state receives no irrigation",
        "C5": "below-target state recovers into target band",
        "C6": "heavy-rain forecast suppresses irrigation",
    }
    responses = {
        "C1": lambda log: log.iloc[-1].theta < log.iloc[0].theta_before,
        "C2": lambda log: log.iloc[12].theta > log.iloc[12].theta_before,
        "C3": lambda log: log.iloc[12].theta > log.iloc[12].theta_before and log.iloc[12].irrigation_mm == 5.0,
        "C4": lambda log: log.irrigation_mm.sum() == 0.0,
        "C5": lambda log: log.iloc[-1].theta >= 0.22 and log.irrigation_mm.sum() > 0.0,
        "C6": lambda log: log.iloc[0].irrigation_mm == 0.0 and log.iloc[0].runoff_mm > 0.0,
    }
    rows = []
    for case_id, log in logs.items():
        finite = bool(np.isfinite(log.select_dtypes(include="number").to_numpy()).all())
        bounded = bool(log["irrigation_mm"].between(0.0, 5.0).all())
        balanced = float(log["mass_balance_error_mm"].abs().max()) <= 1e-8
        response_passed = bool(responses[case_id](log))
        rows.append({
            "case_id": case_id,
            "expected_response": expected[case_id],
            "hours": len(log),
            "initial_theta": log.iloc[0].theta_before,
            "final_theta": log.iloc[-1].theta,
            "min_theta": log.theta.min(),
            "max_theta": log.theta.max(),
            "total_rain_mm": log.precipitation_mm.sum(),
            "total_irrigation_mm": log.irrigation_mm.sum(),
            "total_runoff_mm": log.runoff_mm.sum(),
            "max_abs_mass_balance_error_mm": log.mass_balance_error_mm.abs().max(),
            "action_bounded": bounded,
            "all_finite": finite,
            "response_passed": response_passed,
            "passed": finite and bounded and balanced and response_passed,
        })
    return pd.DataFrame(rows), logs


def select_raw_episodes(validation_2024: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = validation_2024.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    episodes = {}
    for name, (start, end, expected_rain) in RAW_EPISODES.items():
        episode = data.loc[data["timestamp"].between(start, end)].copy()
        if len(episode) != 336:
            raise ValueError(f"{name} episode must contain exactly 336 hourly rows")
        rain = float(episode["precipitation_mm"].sum())
        if not np.isclose(rain, expected_rain, atol=1e-9):
            raise ValueError(f"{name} rain total changed: expected {expected_rain}, got {rain}")
        episodes[name] = episode
    return episodes
