import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.process_data import FORECAST_COLUMNS, WEATHER_COLUMNS, RAW_DIR, load_open_meteo


def test_required_data_is_hourly_complete_and_causal() -> None:
    weather = load_open_meteo(RAW_DIR / "Historical Weather 2021-2025.csv", WEATHER_COLUMNS)
    forecast = load_open_meteo(RAW_DIR / "Historical Forecast 2021-2025.csv", FORECAST_COLUMNS)

    assert len(weather) == 43_824
    assert weather["timestamp"].min() == pd.Timestamp("2021-01-01 00:00:00")
    assert weather["timestamp"].max() == pd.Timestamp("2025-12-31 23:00:00")
    assert not weather.isna().any().any()
    assert not forecast.isna().any().any()
    assert not weather["timestamp"].duplicated().any()
    assert not forecast["timestamp"].duplicated().any()
