"""Training and deterministic validation helpers for SAC Basic."""

from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import pandas as pd
import torch

from evaluation import compute_metrics

from .agent import ReplayBuffer, SACAgent
from .environment import SACIrrigationEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _snapshot(agent: SACAgent) -> dict:
    return {
        name: deepcopy(getattr(agent, name).state_dict())
        for name in ("actor", "critic1", "critic2", "target1", "target2")
    } | {"log_alpha": agent.log_alpha.detach().clone()}


def _restore(agent: SACAgent, state: dict) -> None:
    for name in ("actor", "critic1", "critic2", "target1", "target2"):
        getattr(agent, name).load_state_dict(state[name])
    agent.log_alpha.data.copy_(state["log_alpha"])


def _selection_score(metrics: dict[str, float]) -> float:
    water_penalty = max(metrics["total_irrigation_mm"] - 750, 0) / 10
    deficit_penalty = max(metrics["deficit_rate_pct"] - 20, 0)
    return metrics["time_in_target_pct"] - water_penalty - deficit_penalty


def _selection_key(metrics: dict[str, float]) -> tuple:
    return (
        validation_gate(metrics),
        metrics["time_in_target_pct"],
        -metrics["total_irrigation_mm"],
        -metrics["rmse_band"],
    )


def train(
    agent: SACAgent,
    env: SACIrrigationEnv,
    episodes: int,
    buffer_capacity: int = 100_000,
    validation_env: SACIrrigationEnv | None = None,
    validation_interval: int = 10,
):
    seed = int(env.rng.bit_generator._seed_seq.entropy)
    set_seed(seed)
    replay = ReplayBuffer(buffer_capacity, env.observation_dim, seed)
    history, validation_history, total_steps, updates = [], [], 0, 0
    best_key, best_state, best_episode = None, None, None
    for episode in range(episodes):
        observation, done, episode_reward = env.reset(), False, 0.0
        losses = {}
        while not done:
            if total_steps < agent.config.warmup:
                action = np.asarray([env.rng.uniform(0, agent.config.action_max)], dtype=np.float32)
            else:
                action = agent.select_action(observation)
            next_observation, reward, done, _ = env.step(action)
            replay.add(observation, action, reward, next_observation, done)
            observation, episode_reward = next_observation, episode_reward + reward
            total_steps += 1
            if len(replay) >= agent.config.batch_size and total_steps >= agent.config.warmup:
                losses = agent.update(replay)
                updates += 1
        scalar_losses = {key: float(value.cpu()) for key, value in losses.items()}
        history.append({
            "episode": episode + 1,
            "reward": episode_reward,
            "steps": total_steps,
            "updates": updates,
            **scalar_losses,
        })
        should_validate = validation_env is not None and (
            (episode + 1) % validation_interval == 0 or episode + 1 == episodes
        )
        if should_validate:
            _, metrics = evaluate(agent, validation_env)
            score = _selection_score(metrics)
            key = _selection_key(metrics)
            validation_history.append({
                "episode": episode + 1,
                "validation_gate": key[0],
                "selection_score": score,
                **metrics,
            })
            if best_key is None or key > best_key:
                best_key, best_state, best_episode = key, _snapshot(agent), episode + 1
    if best_state is not None:
        _restore(agent, best_state)
    return pd.DataFrame(history), replay, pd.DataFrame(validation_history), best_episode


def evaluate(agent: SACAgent, env: SACIrrigationEnv):
    evaluation_actor = deepcopy(agent.actor).cpu().eval()
    observation = env.reset(start_index=0)
    records, done = [], False
    while not done:
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            action, _ = evaluation_actor(tensor, deterministic=True, with_log_prob=False)
        action = action.numpy()[0]
        next_observation, reward, done, info = env.step(action)
        row = env.data.iloc[env.index if done else env.index - 1]
        records.append({
            "timestamp": row.timestamp,
            "controller": "SAC Basic",
            "theta": info["theta"],
            "irrigation_mm": info["irrigation_mm"],
            "runoff_mm": info["runoff_mm"],
            "drainage_mm": info["drainage_mm"],
            "mass_balance_error_mm": info["mass_balance_error_mm"],
            "decision_latency_ms": 0.0,
            "reward": reward,
            **{
                key: info[key]
                for key in (
                    "reward_offset",
                    "reward_tracking_cost",
                    "reward_water_cost",
                    "reward_smoothness_cost",
                    "reward_violation_cost",
                )
            },
        })
        observation = next_observation
    log = pd.DataFrame(records)
    metrics = compute_metrics(log)
    metrics["cumulative_reward"] = float(log["reward"].sum())
    return log, metrics


def validation_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics["time_in_target_pct"] >= 50
        and metrics["violation_rate_pct"] <= 50
        and metrics["total_irrigation_mm"] <= 750
        and 0.22 <= metrics["mean_soil_moisture"] <= 0.32
        and metrics["deficit_rate_pct"] <= 20
        and metrics["max_abs_mass_balance_error_mm"] <= 1e-8
        and all(np.isfinite(value) for value in metrics.values())
    )
