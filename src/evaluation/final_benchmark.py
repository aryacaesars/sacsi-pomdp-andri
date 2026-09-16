"""Shared integrity checks and aggregation for the final 2025 benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
METHODS = (
    "No Irrigation",
    "Fixed Schedule",
    "Threshold-Based",
    "Rule-Based Forecast-Aware",
    "Fuzzy Controller",
    "SAC Basic",
    "SAC + Forecast",
    "SAC + LSTM",
    "SACSI Full",
)
FORMAL_METRICS = (
    "total_irrigation_mm",
    "time_in_target_pct",
    "violation_rate_pct",
    "rmse_band",
    "action_smoothness",
    "deficit_rate_pct",
    "surplus_rate_pct",
    "drainage_total_mm",
    "runoff_total_mm",
    "mean_soil_moisture",
    "max_abs_mass_balance_error_mm",
)


def validate_common_support(
    weather: pd.DataFrame,
    forecast: pd.DataFrame | None = None,
    expected_hours: int = 8_760,
) -> pd.DatetimeIndex:
    """Require one unique, continuous, 2025-only hourly support."""
    if "timestamp" not in weather:
        raise ValueError("Weather is missing timestamp")
    timestamps = pd.DatetimeIndex(pd.to_datetime(weather["timestamp"]))
    if len(timestamps) != expected_hours or not timestamps.is_unique:
        raise ValueError(f"Benchmark must contain {expected_hours} unique hourly timestamps")
    if set(timestamps.year) != {2025}:
        raise ValueError("Final benchmark must use 2025 only")
    if len(timestamps) > 1 and not (timestamps[1:] - timestamps[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("Benchmark timestamps must be continuous hourly data")
    if forecast is not None:
        forecast_timestamps = pd.DatetimeIndex(pd.to_datetime(forecast["timestamp"]))
        if not timestamps.equals(forecast_timestamps):
            raise ValueError("Weather and forecast must have identical timestamp support")
    return timestamps


def validate_log_support(log: pd.DataFrame, timestamps: pd.DatetimeIndex) -> None:
    if not timestamps.equals(pd.DatetimeIndex(pd.to_datetime(log["timestamp"]))):
        raise ValueError("Controller log does not match common benchmark support")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_registry(registry: pd.DataFrame, root: Path) -> pd.DataFrame:
    seeds = tuple(sorted(registry["seed"].astype(int)))
    if len(registry) != len(SEEDS) or seeds != tuple(sorted(SEEDS)):
        raise ValueError("Registry must contain exactly the 10 locked seeds")
    for row in registry.itertuples(index=False):
        checkpoint = root / Path(row.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if sha256_file(checkpoint) != row.checkpoint_sha256:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    return registry.sort_values("seed", key=lambda values: values.astype(int)).reset_index(drop=True)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    missing = set(FORMAL_METRICS).difference(runs.columns)
    if missing:
        raise ValueError(f"Missing formal metrics: {sorted(missing)}")
    rows = []
    for method, group in runs.groupby("method", sort=False):
        row: dict[str, float | int | str] = {
            "method": method,
            "method_type": group["method_type"].iloc[0],
            "n_runs": len(group),
        }
        for metric in FORMAL_METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_method_order"] = summary["method"].map({name: i for i, name in enumerate(METHODS)})
    summary = summary.sort_values(
        ["time_in_target_pct_mean", "total_irrigation_mm_mean", "_method_order"],
        ascending=[False, True, True],
    ).drop(columns="_method_order").reset_index(drop=True)
    ranks, current_rank, previous = [], 1, None
    for index, row in summary.iterrows():
        key = (row["time_in_target_pct_mean"], row["total_irrigation_mm_mean"])
        if previous is not None and not all(np.isclose(a, b) for a, b in zip(key, previous)):
            current_rank = index + 1
        ranks.append(current_rank)
        previous = key
    summary.insert(0, "rank", ranks)
    return summary
