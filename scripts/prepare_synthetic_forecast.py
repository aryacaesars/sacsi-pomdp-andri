"""Create the explicitly labelled causal SF-20 h+1 forecast proxy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "00_Dataset" / "Processed"


def build_synthetic_forecast(
    weather: pd.DataFrame,
    error_fraction: float = 0.20,
    seed: int = 2020,
) -> pd.DataFrame:
    if error_fraction < 0:
        raise ValueError("error_fraction must be non-negative")
    weather = weather.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    actual_columns = ["precipitation_mm", "et0_mm", "temperature_c"]
    future = weather.groupby(weather["timestamp"].dt.year)[actual_columns].shift(-1)
    future = future.fillna(weather[actual_columns])  # no forecast crosses a split-year boundary
    rng = np.random.default_rng(seed)
    precipitation = np.maximum(
        future["precipitation_mm"] * (1 + rng.normal(0, error_fraction, len(weather))), 0
    )
    et0 = np.maximum(future["et0_mm"] * (1 + rng.normal(0, error_fraction, len(weather))), 0)
    temperature = future["temperature_c"] + rng.normal(
        0, error_fraction * 1.3772097352568569, len(weather)
    )
    target_timestamp = weather["timestamp"] + pd.Timedelta(hours=1)
    year_end = target_timestamp.dt.year != weather["timestamp"].dt.year
    target_timestamp.loc[year_end] = weather.loc[year_end, "timestamp"]
    return pd.DataFrame({
        "timestamp": weather["timestamp"],
        "forecast_target_timestamp": target_timestamp,
        "forecast_precipitation_mm": precipitation,
        "forecast_et0_mm": et0,
        "forecast_temperature_c": temperature,
        "forecast_protocol": f"SF-{round(error_fraction * 100):02d}_h1_controlled_proxy",
    })


def build_sf20(weather: pd.DataFrame, seed: int = 2020) -> pd.DataFrame:
    return build_synthetic_forecast(weather, 0.20, seed)


def main() -> None:
    weather = pd.read_csv(DATA_DIR / "data_clean.csv", parse_dates=["timestamp"])
    forecast = build_sf20(weather)
    output = DATA_DIR / "synthetic_forecast_sf20.csv"
    forecast.to_csv(output, index=False)
    print(f"rows={len(forecast):,} missing={int(forecast.isna().sum().sum())} output={output}")


if __name__ == "__main__":
    main()
