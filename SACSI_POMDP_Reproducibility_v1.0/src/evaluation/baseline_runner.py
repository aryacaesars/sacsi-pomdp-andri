"""Shared simulation, logging, and metrics for deterministic controllers."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from controllers.baselines import Controller, Observation
from virtual_garden import VirtualGardenConfig, VirtualGardenCore


def run_controller(
    weather: pd.DataFrame,
    controller: Controller,
    forecast: pd.DataFrame | None = None,
    config: VirtualGardenConfig | None = None,
) -> pd.DataFrame:
    garden = VirtualGardenCore(config)
    controller.reset()
    forecast_by_time = {} if forecast is None else forecast.set_index("timestamp")[
        "forecast_precipitation_h1_mm"
    ].to_dict()
    records = []

    for row in weather.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        observation = Observation(
            timestamp=timestamp.to_pydatetime(),
            theta=garden.theta,
            precipitation_mm=float(row.precipitation_mm),
            et0_mm=float(row.et0_mm),
            forecast_precipitation_h1_mm=float(forecast_by_time.get(timestamp, 0.0)),
        )
        started = perf_counter()
        action = float(controller.select_action(observation))
        latency_ms = (perf_counter() - started) * 1_000
        result = garden.step(observation.precipitation_mm, observation.et0_mm, action)
        records.append({
            "timestamp": timestamp,
            "controller": controller.name,
            "theta_before": observation.theta,
            "theta": result.theta,
            "precipitation_mm": result.precipitation_mm,
            "forecast_precipitation_h1_mm": observation.forecast_precipitation_h1_mm,
            "irrigation_mm": result.irrigation_mm,
            "evapotranspiration_mm": result.evapotranspiration_mm,
            "drainage_mm": result.drainage_mm,
            "runoff_mm": result.runoff_mm,
            "mass_balance_error_mm": result.mass_balance_error_mm,
            "decision_latency_ms": latency_ms,
        })
    return pd.DataFrame(records)


def compute_metrics(log: pd.DataFrame, config: VirtualGardenConfig | None = None) -> dict[str, float]:
    cfg = config or VirtualGardenConfig()
    theta = log["theta"].to_numpy()
    action = log["irrigation_mm"].to_numpy()
    deficit = theta < cfg.target_min
    surplus = theta > cfg.target_max
    in_target = ~(deficit | surplus)
    band_distance = np.where(deficit, cfg.target_min - theta, np.where(surplus, theta - cfg.target_max, 0))
    total_water = float(action.sum())
    return {
        "total_irrigation_mm": total_water,
        "time_in_target_pct": float(in_target.mean() * 100),
        "violation_rate_pct": float((~in_target).mean() * 100),
        "deficit_rate_pct": float(deficit.mean() * 100),
        "surplus_rate_pct": float(surplus.mean() * 100),
        "rmse_band": float(np.sqrt(np.mean(band_distance ** 2))),
        "mean_soil_moisture": float(theta.mean()),
        "runoff_total_mm": float(log["runoff_mm"].sum()),
        "drainage_total_mm": float(log["drainage_mm"].sum()),
        "action_smoothness": float(np.abs(np.diff(action)).mean()) if len(action) > 1 else 0.0,
        "decision_latency_ms": float(log["decision_latency_ms"].mean()),
        "water_use_efficiency": float(in_target.sum() / total_water) if total_water else 0.0,
        "max_abs_mass_balance_error_mm": float(log["mass_balance_error_mm"].abs().max()),
    }
