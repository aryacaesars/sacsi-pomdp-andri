"""Audit and prepare the SACSI 2021-2025 hourly datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "00_Dataset"
OUT_DIR = RAW_DIR / "Processed"

WEATHER_COLUMNS = {
    "time": "timestamp",
    "temperature_2m (°C)": "temperature_c",
    "relative_humidity_2m (%)": "relative_humidity_pct",
    "precipitation (mm)": "precipitation_mm",
    "rain (mm)": "rain_mm",
    "et0_fao_evapotranspiration (mm)": "et0_mm",
    "vapour_pressure_deficit (kPa)": "vpd_kpa",
    "shortwave_radiation (W/m²)": "shortwave_radiation_w_m2",
}

FORECAST_COLUMNS = {
    "time": "timestamp",
    "temperature_2m (°C)": "forecast_temperature_c",
    "precipitation (mm)": "forecast_precipitation_mm",
    "et0_fao_evapotranspiration (mm)": "forecast_et0_mm",
}


def load_open_meteo(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    """Load an Open-Meteo CSV containing two metadata rows and a blank row."""
    frame = pd.read_csv(path, skiprows=3)
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {sorted(missing)}")
    frame = frame[list(columns)].rename(columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame.sort_values("timestamp").reset_index(drop=True)


def audit(name: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    timestamps = frame["timestamp"]
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq="h")
    rows = [{
        "dataset": name,
        "scope": "dataset",
        "column": "",
        "row_count": len(frame),
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "missing_hours": len(expected.difference(timestamps)),
        "missing_values": int(frame.isna().sum().sum()),
    }]
    rows.extend({
        "dataset": name,
        "scope": "column",
        "column": column,
        "row_count": len(frame),
        "start": "",
        "end": "",
        "duplicate_timestamps": "",
        "missing_hours": "",
        "missing_values": int(frame[column].isna().sum()),
    } for column in frame.columns)
    return rows


def fit_normalizer(training: pd.DataFrame) -> dict[str, dict[str, float]]:
    features = training.columns.drop("timestamp")
    return {
        column: {
            "mean": float(training[column].mean()),
            "std": float(training[column].std(ddof=0)),
        }
        for column in features
    }


def main() -> None:
    weather = load_open_meteo(RAW_DIR / "Historical Weather 2021-2025.csv", WEATHER_COLUMNS)
    forecast = load_open_meteo(RAW_DIR / "Historical Forecast 2021-2025.csv", FORECAST_COLUMNS)

    if weather["timestamp"].duplicated().any() or forecast["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")
    if weather.isna().any().any() or forecast.isna().any().any():
        raise ValueError("Missing values detected in required features")

    train = weather[weather["timestamp"].dt.year <= 2023].copy()
    validation = weather[weather["timestamp"].dt.year == 2024].copy()
    benchmark = weather[weather["timestamp"].dt.year == 2025].copy()
    if set(weather["timestamp"].dt.year.unique()) != {2021, 2022, 2023, 2024, 2025}:
        raise ValueError("Weather data must cover exactly 2021-2025")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weather.to_csv(OUT_DIR / "data_clean.csv", index=False)
    forecast.to_csv(OUT_DIR / "forecast_clean.csv", index=False)
    train.to_csv(OUT_DIR / "train_2021_2023.csv", index=False)
    validation.to_csv(OUT_DIR / "validation_2024.csv", index=False)
    benchmark.to_csv(OUT_DIR / "benchmark_2025.csv", index=False)
    pd.DataFrame(audit("weather", weather) + audit("forecast", forecast)).to_csv(
        OUT_DIR / "data_audit_report.csv", index=False
    )
    (OUT_DIR / "normalizer.json").write_text(
        json.dumps(fit_normalizer(train), indent=2), encoding="utf-8"
    )

    print(f"Processed {len(weather):,} weather rows and {len(forecast):,} forecast rows")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
