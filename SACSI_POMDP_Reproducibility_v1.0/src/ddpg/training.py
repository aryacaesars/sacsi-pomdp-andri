"""DDPG training with validation-2024-only checkpoint selection."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from evaluation import compute_metrics
from sac_basic import ReplayBuffer, SACIrrigationEnv
from sac_basic.training import set_seed, validation_gate

from .agent import DDPGAgent


def evaluate(agent: DDPGAgent, env: SACIrrigationEnv):
    actor = deepcopy(agent.actor).cpu().eval()
    observation = env.reset(start_index=0)
    records, done = [], False
    while not done:
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            action = actor(tensor).numpy()[0]
        next_observation, reward, done, info = env.step(action)
        row = env.data.iloc[env.index if done else env.index - 1]
        records.append({
            "timestamp": row.timestamp,
            "controller": "DDPG",
            "theta": info["theta"],
            "irrigation_mm": info["irrigation_mm"],
            "runoff_mm": info["runoff_mm"],
            "drainage_mm": info["drainage_mm"],
            "mass_balance_error_mm": info["mass_balance_error_mm"],
            "decision_latency_ms": 0.0,
            "reward": reward,
        })
        observation = next_observation
    log = pd.DataFrame(records)
    metrics = compute_metrics(log)
    metrics["cumulative_reward"] = float(log["reward"].sum())
    return log, metrics


def _snapshot(agent: DDPGAgent) -> dict:
    return {
        name: deepcopy(getattr(agent, name).state_dict())
        for name in ("actor", "critic", "target_actor", "target_critic")
    }


def _restore(agent: DDPGAgent, state: dict) -> None:
    for name, parameters in state.items():
        getattr(agent, name).load_state_dict(parameters)


def _selection_key(metrics: dict[str, float]) -> tuple:
    return (
        validation_gate(metrics),
        metrics["time_in_target_pct"],
        -metrics["total_irrigation_mm"],
        -metrics["rmse_band"],
    )


def train(
    agent: DDPGAgent,
    env: SACIrrigationEnv,
    episodes: int,
    validation_env: SACIrrigationEnv,
    validation_interval: int = 5,
):
    seed = int(env.rng.bit_generator._seed_seq.entropy)
    set_seed(seed)
    noise_rng = np.random.default_rng(seed + 80_000)
    replay = ReplayBuffer(100_000, env.observation_dim, seed)
    history, selections, total_steps, updates = [], [], 0, 0
    best_key, best_state, best_episode = None, None, None
    for episode in range(1, episodes + 1):
        observation, done, episode_reward, losses = env.reset(), False, 0.0, {}
        while not done:
            if total_steps < agent.config.warmup:
                action = np.array([env.rng.uniform(0, agent.config.action_max)], dtype=np.float32)
            else:
                action = np.clip(
                    agent.select_action(observation)
                    + noise_rng.normal(0, agent.config.exploration_noise_std, 1),
                    0, agent.config.action_max,
                ).astype(np.float32)
            next_observation, reward, done, _ = env.step(action)
            replay.add(observation, action, reward, next_observation, done)
            observation, episode_reward = next_observation, episode_reward + reward
            total_steps += 1
            if len(replay) >= agent.config.batch_size and total_steps >= agent.config.warmup:
                losses = agent.update(replay)
                updates += 1
        history.append({
            "seed": seed,
            "episode": episode,
            "reward": episode_reward,
            "steps": total_steps,
            "updates": updates,
            **{key: float(value.cpu()) for key, value in losses.items()},
        })
        if episode % validation_interval == 0 or episode == episodes:
            _, metrics = evaluate(agent, validation_env)
            key = _selection_key(metrics)
            selections.append({
                "seed": seed, "episode": episode,
                "validation_gate": validation_gate(metrics), **metrics,
            })
            if best_key is None or key > best_key:
                best_key, best_state, best_episode = key, _snapshot(agent), episode
    _restore(agent, best_state)
    return pd.DataFrame(history), pd.DataFrame(selections), replay, best_episode
