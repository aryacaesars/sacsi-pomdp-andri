import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpg import DDPGAgent, DDPGConfig, ReplayBuffer
from sac_basic import SACIrrigationEnv


def make_env(length=48):
    data = pd.read_csv(ROOT / "00_Dataset" / "Processed" / "train_2021_2023.csv", nrows=length)
    normalizer = json.loads(
        (ROOT / "00_Dataset" / "Processed" / "normalizer.json").read_text(encoding="utf-8")
    )
    return SACIrrigationEnv(data, normalizer, length, 11)


def test_ddpg_action_update_and_target_soft_update_are_valid():
    config = DDPGConfig(hidden_dim=16, batch_size=8, warmup=8)
    agent = DDPGAgent(config, device="cpu")
    replay = ReplayBuffer(32, config.observation_dim, seed=11)
    observation = np.zeros(config.observation_dim, dtype=np.float32)
    for _ in range(8):
        action = agent.select_action(observation)
        assert 0 <= action[0] <= 5
        replay.add(observation, action, 1.0, observation, False)
    before = [parameter.detach().clone() for parameter in agent.target_actor.parameters()]
    losses = agent.update(replay)
    assert all(np.isfinite(float(value)) for value in losses.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.target_actor.parameters()))


def test_ddpg_uses_locked_current_observation_only():
    env = make_env()
    observation = env.reset(start_index=0)
    config = DDPGConfig()
    assert observation.shape == (config.observation_dim,) == (8,)
    assert env.reward_version == "reward_v4"
    assert not hasattr(config, "forecast_dim") and not hasattr(config, "history_length")


def test_ddpg_checkpoint_round_trip():
    agent = DDPGAgent(DDPGConfig(hidden_dim=16), device="cpu")
    observation = np.zeros(8, dtype=np.float32)
    expected = agent.select_action(observation)
    path = ROOT / "Checkpoints" / "DDPG" / f".round_trip_{uuid4().hex}.pt"
    try:
        agent.save(path, {"seed": 11})
        loaded, metadata = DDPGAgent.load(path, device="cpu")
        assert metadata == {"seed": 11}
        assert np.allclose(loaded.select_action(observation), expected)
    finally:
        path.unlink(missing_ok=True)


def test_ddpg_production_artifacts_are_complete():
    result_dir = ROOT / "Results" / "DDPG"
    results = pd.read_csv(result_dir / "ddpg_validation_results.csv")
    training = pd.read_csv(result_dir / "ddpg_training_log.csv")
    selection = pd.read_csv(result_dir / "ddpg_checkpoint_selection.csv")
    assert len(results) == 3 and set(results["seed"]) == {11, 22, 33}
    assert len(training) == 3 * 20
    assert selection.groupby("seed")["episode"].count().eq(4).all()
    assert set(results["reward_version"]) == {"reward_v4"}
    assert set(results["observation_dim"]) == {8}
    assert not results["forecast"].any() and not results["history"].any()
    assert results["losses_finite"].all()
    assert results["max_abs_mass_balance_error_mm"].max() <= 1e-8
    for seed in (11, 22, 33):
        checkpoint = ROOT / "Checkpoints" / "DDPG" / f"ddpg_seed{seed}_best.pt"
        agent, metadata = DDPGAgent.load(checkpoint, device="cpu")
        assert metadata["seed"] == seed and metadata["reward_version"] == "reward_v4"
        assert 0 <= agent.select_action(np.zeros(8, dtype=np.float32))[0] <= 5
