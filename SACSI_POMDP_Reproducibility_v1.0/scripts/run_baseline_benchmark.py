"""Run all deterministic baselines on the locked temporal splits."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers import (
    FixedScheduleController,
    FuzzyController,
    NoIrrigationController,
    RuleBasedForecastController,
    ThresholdController,
)
from evaluation import compute_metrics, run_controller


DATA_DIR = ROOT / "00_Dataset" / "Processed"
RESULTS_DIR = ROOT / "Results"
LOGS_DIR = ROOT / "Logs" / "Baselines"


def load_inputs(split_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather = pd.read_csv(DATA_DIR / split_file, parse_dates=["timestamp"])
    forecast = pd.read_csv(DATA_DIR / "forecast_clean.csv", parse_dates=["timestamp"])
    # Continuous historical forecast is used only as an explicitly labelled h+1 proxy.
    forecast["timestamp"] = forecast["timestamp"] - pd.Timedelta(hours=1)
    forecast = forecast.rename(columns={
        "forecast_precipitation_mm": "forecast_precipitation_h1_mm"
    })[["timestamp", "forecast_precipitation_h1_mm"]]
    return weather, forecast


def main() -> None:
    splits = {
        "train_2021_2023": "train_2021_2023.csv",
        "validation_2024": "validation_2024.csv",
        "benchmark_2025": "benchmark_2025.csv",
    }
    controllers = [
        NoIrrigationController(),
        FixedScheduleController(),
        ThresholdController(),
        RuleBasedForecastController(),
        FuzzyController(),
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    for split, filename in splits.items():
        weather, forecast = load_inputs(filename)
        for controller in controllers:
            log = run_controller(weather, controller, forecast)
            log.to_csv(LOGS_DIR / f"{split}_{controller.name.lower().replace(' ', '_')}.csv", index=False)
            metrics_rows.append({
                "split": split,
                "controller": controller.name,
                "forecast_protocol": "historical_continuous_h1_proxy",
                **compute_metrics(log),
            })

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    print(metrics[["split", "controller", "total_irrigation_mm", "time_in_target_pct", "violation_rate_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
