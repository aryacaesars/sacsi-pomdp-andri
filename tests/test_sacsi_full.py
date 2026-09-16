import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sac_basic.agent import Actor, SACConfig
from sacsi_full import SACSIConfig, SACSIEnv, SACSIRecurrentAgent


def test_sacsi_state_shapes_and_zero_context_warm_start() -> None:
    data_dir = ROOT / "00_Dataset" / "Processed"
    weather = pd.read_csv(data_dir / "train_2021_2023.csv", nrows=49, parse_dates=["timestamp"])
    forecast = pd.read_csv(data_dir / "synthetic_forecast_sf20.csv", nrows=49, parse_dates=["timestamp"])
    data = weather.merge(forecast, on="timestamp", validate="one_to_one")
    normalizer = json.loads((data_dir / "normalizer.json").read_text())
    env = SACSIEnv(data, normalizer, episode_length=48, seed=11, sequence_length=24)
    current, history, forecast_context = env.reset(start_index=0)
    assert current.shape == (8,)
    assert history.shape == (24, 8)
    assert forecast_context.shape == (3,)
    assert np.isfinite(forecast_context).all()

    checkpoint = ROOT / "Checkpoints" / "SAC_Basic" / "sac_basic_seed11_reward_v2_training_ep100.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    basic = Actor(SACConfig())
    basic.load_state_dict(state["actor"])
    agent = SACSIRecurrentAgent(SACSIConfig(), device="cpu")
    agent.warm_start(checkpoint)
    zeros = (np.zeros(8, np.float32), np.zeros((24, 8), np.float32), np.zeros(3, np.float32))
    with torch.inference_mode():
        expected, _ = basic(torch.zeros((1, 8)), True, False)
    actual = agent.select_action(zeros, deterministic=True)
    assert actual[0] == pytest.approx(float(expected[0, 0]), abs=1e-7)
    assert agent.context_norms() == {"history_residual_norm": 0.0, "forecast_residual_norm": 0.0}
