import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sac_basic.agent import Actor, SACConfig
from sac_lstm import RecurrentSACAgent, RecurrentSACConfig, SACLSTMEnv


def make_env(length=49):
    data = pd.read_csv(ROOT / "00_Dataset" / "Processed" / "train_2021_2023.csv", nrows=length)
    normalizer = json.loads((ROOT / "00_Dataset" / "Processed" / "normalizer.json").read_text())
    return SACLSTMEnv(data, normalizer, episode_length=48, seed=11, sequence_length=24)


def test_sequence_is_causal_and_reset_clears_history() -> None:
    env = make_env()
    current, history = env.reset(start_index=0)
    assert history.shape == (24, 8)
    assert np.count_nonzero(history[:-1]) == 0
    assert np.array_equal(history[-1], current)
    (next_current, next_history), _, _, _ = env.step([0])
    assert np.array_equal(next_history[-1], next_current)
    _, reset_history = env.reset(start_index=1)
    assert np.count_nonzero(reset_history[:-1]) == 0


def test_rrws_zero_residual_matches_basic_actor() -> None:
    checkpoint = ROOT / "Checkpoints" / "SAC_Basic" / "sac_basic_seed11_reward_v2_training_ep100.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    basic = Actor(SACConfig())
    basic.load_state_dict(state["actor"])
    recurrent = RecurrentSACAgent(RecurrentSACConfig(), device="cpu")
    recurrent.warm_start(checkpoint)
    current = np.zeros(8, np.float32)
    history = np.zeros((24, 8), np.float32)
    with torch.inference_mode():
        expected, _ = basic(torch.as_tensor(current).unsqueeze(0), True, False)
    actual = recurrent.select_action((current, history), deterministic=True)
    assert actual[0] == pytest.approx(float(expected[0, 0]), abs=1e-7)


import pytest
