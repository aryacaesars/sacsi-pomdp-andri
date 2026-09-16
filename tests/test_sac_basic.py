import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sac_basic import ReplayBuffer, SACAgent, SACConfig, SACIrrigationEnv


def make_env(length=48):
    data = pd.read_csv(ROOT / "00_Dataset" / "Processed" / "train_2021_2023.csv", nrows=length)
    normalizer = json.loads((ROOT / "00_Dataset" / "Processed" / "normalizer.json").read_text())
    return SACIrrigationEnv(data, normalizer, episode_length=length, seed=11)


def test_sac_environment_has_locked_observation_and_mass_balance() -> None:
    env = make_env()
    observation = env.reset(start_index=0)
    assert observation.shape == (8,)
    next_observation, reward, done, info = env.step(np.array([9.0]))
    assert next_observation.shape == (8,)
    assert np.isfinite(reward)
    assert info["irrigation_mm"] == 5.0
    assert abs(info["mass_balance_error_mm"]) <= 1e-8
    assert not done


def test_sac_agent_action_and_single_update_are_finite() -> None:
    config = SACConfig(hidden_dim=16, batch_size=8, warmup=8)
    agent = SACAgent(config, device="cpu")
    replay = ReplayBuffer(32, config.observation_dim)
    observation = np.zeros(config.observation_dim, dtype=np.float32)
    for _ in range(8):
        action = agent.select_action(observation)
        assert 0 <= action[0] <= 5
        replay.add(observation, action, 0.0, observation, False)
    losses = agent.update(replay)
    assert all(np.isfinite(float(value.cpu())) for value in losses.values())
