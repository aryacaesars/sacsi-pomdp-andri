import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sac_basic import SACIrrigationEnv
from td3 import ReplayBuffer, TD3Agent, TD3Config


def make_env(length=48):
    data = pd.read_csv(ROOT / "00_Dataset" / "Processed" / "train_2021_2023.csv", nrows=length)
    normalizer = json.loads(
        (ROOT / "00_Dataset" / "Processed" / "normalizer.json").read_text(encoding="utf-8")
    )
    return SACIrrigationEnv(data, normalizer, length, 11)


def test_td3_twin_critics_clipped_noise_and_delayed_update_are_valid():
    config = TD3Config(
        hidden_dim=16, batch_size=8, warmup=8,
        target_noise_std=100.0, target_noise_clip=0.1, policy_delay=2,
    )
    agent = TD3Agent(config, device="cpu")
    replay = ReplayBuffer(32, config.observation_dim, seed=11)
    observation = np.zeros(config.observation_dim, dtype=np.float32)
    for _ in range(8):
        action = agent.select_action(observation)
        replay.add(observation, action, 1.0, observation, False)

    assert any(
        not torch.equal(first, second)
        for first, second in zip(agent.critic1.parameters(), agent.critic2.parameters())
    )
    before_actor = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    first = agent.update(replay)
    assert not first["actor_updated"]
    assert all(
        torch.equal(old, new) for old, new in zip(before_actor, agent.actor.parameters())
    )
    second = agent.update(replay)
    assert second["actor_updated"]
    assert any(
        not torch.equal(old, new) for old, new in zip(before_actor, agent.actor.parameters())
    )
    assert float(second["target_noise_abs_max"]) <= config.target_noise_clip + 1e-6
    assert all(np.isfinite(float(second[key])) for key in ("actor_loss", "critic_loss"))
    action = agent.select_action(observation)
    assert np.isfinite(action).all() and 0 <= action[0] <= config.action_max


def test_td3_uses_locked_current_observation_only():
    env = make_env()
    observation = env.reset(start_index=0)
    config = TD3Config()
    assert observation.shape == (config.observation_dim,) == (8,)
    assert env.reward_version == "reward_v4"
    assert not hasattr(config, "forecast_dim") and not hasattr(config, "history_length")


def test_td3_checkpoint_round_trip():
    agent = TD3Agent(TD3Config(hidden_dim=16), device="cpu")
    observation = np.zeros(8, dtype=np.float32)
    expected = agent.select_action(observation)
    path = ROOT / "Checkpoints" / "TD3" / f".round_trip_{uuid4().hex}.pt"
    try:
        agent.save(path, {"seed": 11})
        loaded, metadata = TD3Agent.load(path, device="cpu")
        assert metadata == {"seed": 11}
        assert np.allclose(loaded.select_action(observation), expected)
    finally:
        path.unlink(missing_ok=True)


def test_td3_production_artifacts_are_complete():
    result_dir = ROOT / "Results" / "TD3"
    results = pd.read_csv(result_dir / "td3_validation_results.csv")
    training = pd.read_csv(result_dir / "td3_training_log.csv")
    selection = pd.read_csv(result_dir / "td3_checkpoint_selection.csv")
    assert len(results) == 3 and set(results["seed"]) == {11, 22, 33}
    assert len(training) == 3 * 20
    assert selection.groupby("seed")["episode"].count().eq(4).all()
    assert set(results["reward_version"]) == {"reward_v4"}
    assert set(results["observation_dim"]) == {8}
    assert set(results["policy_delay"]) == {2}
    assert (results["policy_updates"] == results["critic_updates"] // 2).all()
    assert not results["forecast"].any() and not results["history"].any()
    assert results["losses_finite"].all()
    assert results["max_abs_mass_balance_error_mm"].max() <= 1e-8
    for seed in (11, 22, 33):
        checkpoint = ROOT / "Checkpoints" / "TD3" / f"td3_seed{seed}_best.pt"
        agent, metadata = TD3Agent.load(checkpoint, device="cpu")
        assert metadata["seed"] == seed and metadata["reward_version"] == "reward_v4"
        assert 0 <= agent.select_action(np.zeros(8, dtype=np.float32))[0] <= 5
