import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sac_forecast import SACForecastEnv
from scripts.prepare_synthetic_forecast import build_sf20


def test_sf20_is_complete_and_does_not_cross_year_boundary() -> None:
    weather = pd.read_csv(ROOT / "00_Dataset" / "Processed" / "data_clean.csv", parse_dates=["timestamp"])
    forecast = build_sf20(weather)
    assert len(forecast) == len(weather)
    assert not forecast.isna().any().any()
    year_end = forecast["timestamp"].dt.year != (forecast["timestamp"] + pd.Timedelta(hours=1)).dt.year
    assert (forecast.loc[year_end, "forecast_target_timestamp"] == forecast.loc[year_end, "timestamp"]).all()


def test_forecast_environment_has_11d_observation() -> None:
    weather = pd.read_csv(ROOT / "00_Dataset" / "Processed" / "train_2021_2023.csv", nrows=48)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    forecast = build_sf20(weather)
    data = weather.merge(forecast, on="timestamp", validate="one_to_one")
    normalizer = json.loads((ROOT / "00_Dataset" / "Processed" / "normalizer.json").read_text())
    env = SACForecastEnv(data, normalizer, episode_length=48, seed=11)
    observation = env.reset(start_index=0)
    assert observation.shape == (11,)
    assert np.isfinite(observation).all()
    _, _, _, info = env.step([1.0])
    assert abs(info["mass_balance_error_mm"]) <= 1e-8
