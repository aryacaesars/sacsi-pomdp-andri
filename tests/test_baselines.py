import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers import (
    FixedScheduleController,
    FuzzyController,
    NoIrrigationController,
    Observation,
    RuleBasedForecastController,
    ThresholdController,
)
from evaluation import compute_metrics, run_controller


def observation(theta=0.24, forecast=0.0, hour=6):
    return Observation(datetime(2025, 1, 1, hour), theta, 0.0, 0.2, forecast)


def test_controller_decisions_and_bounds() -> None:
    assert NoIrrigationController().select_action(observation()) == 0
    assert FixedScheduleController().select_action(observation(hour=6)) == 3
    assert FixedScheduleController().select_action(observation(hour=7)) == 0
    assert ThresholdController().select_action(observation(theta=0.24)) == 5
    assert ThresholdController().select_action(observation(theta=0.26)) == 0
    assert RuleBasedForecastController().select_action(observation(forecast=2)) == 0
    assert 0 <= FuzzyController().select_action(observation()) <= 5


def test_common_runner_and_metrics() -> None:
    weather = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=48, freq="h"),
        "precipitation_mm": 0.0,
        "et0_mm": 0.2,
    })
    log = run_controller(weather, ThresholdController())
    metrics = compute_metrics(log)
    assert len(log) == 48
    assert log["irrigation_mm"].between(0, 5).all()
    assert metrics["time_in_target_pct"] + metrics["violation_rate_pct"] == pytest.approx(100)
    assert metrics["max_abs_mass_balance_error_mm"] <= 1e-8
