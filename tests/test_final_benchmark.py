import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.final_benchmark import FORMAL_METRICS, summarize_runs, validate_common_support


def test_common_support_requires_continuous_2025_and_matching_forecast():
    timestamps = pd.date_range("2025-01-01", periods=3, freq="h")
    weather = pd.DataFrame({"timestamp": timestamps})
    forecast = pd.DataFrame({"timestamp": timestamps})
    assert validate_common_support(weather, forecast, expected_hours=3).equals(timestamps)

    forecast.loc[2, "timestamp"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="identical timestamp support"):
        validate_common_support(weather, forecast, expected_hours=3)

    weather.loc[0, "timestamp"] = pd.Timestamp("2024-12-31 23:00:00")
    with pytest.raises(ValueError, match="2025 only"):
        validate_common_support(weather, expected_hours=3)


def test_summary_ranks_target_occupancy_then_water_and_keeps_failed_seed():
    rows = []
    for method, seed, target, water, gate in (
        ("SAC Basic", 11, 55.0, 500.0, True),
        ("SAC Basic", 22, 45.0, 400.0, False),
        ("Rule-Based Forecast-Aware", None, 50.0, 300.0, None),
    ):
        row = {
            "method": method,
            "method_type": "rl" if seed else "baseline",
            "seed": seed,
            "validation_gate": gate,
        }
        row.update({metric: 0.0 for metric in FORMAL_METRICS})
        row["time_in_target_pct"], row["total_irrigation_mm"] = target, water
        rows.append(row)
    summary = summarize_runs(pd.DataFrame(rows))
    assert summary["method"].tolist() == ["Rule-Based Forecast-Aware", "SAC Basic"]
    basic = summary.loc[summary["method"] == "SAC Basic"].iloc[0]
    assert basic["n_runs"] == 2
    assert basic["time_in_target_pct_mean"] == 50.0


def test_summary_assigns_shared_rank_to_exact_ties():
    rows = []
    for method in ("Threshold-Based", "Rule-Based Forecast-Aware"):
        row = {"method": method, "method_type": "baseline"}
        row.update({metric: 1.0 for metric in FORMAL_METRICS})
        rows.append(row)
    assert summarize_runs(pd.DataFrame(rows))["rank"].tolist() == [1, 1]
