"""Training, selection, and context diagnostics for SACSI Full."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from evaluation import compute_metrics
from sac_basic.training import _selection_key, _selection_score, set_seed

from .agent import SACSIRecurrentAgent, SACSIReplayBuffer
from .environment import SACSIEnv


def evaluate(agent: SACSIRecurrentAgent, env: SACSIEnv):
    actor = deepcopy(agent.actor).cpu().eval()
    state, done, records = env.reset(start_index=0), False, []
    while not done:
        tensors = [torch.as_tensor(value, dtype=torch.float32).unsqueeze(0) for value in state]
        with torch.inference_mode():
            action, _ = actor(*tensors, True, False)
        next_state, reward, done, info = env.step(action.numpy()[0])
        row = env.data.iloc[env.index if done else env.index - 1]
        records.append({
            "timestamp": row.timestamp, "controller": "SACSI Full",
            "theta": info["theta"], "irrigation_mm": info["irrigation_mm"],
            "runoff_mm": info["runoff_mm"], "drainage_mm": info["drainage_mm"],
            "mass_balance_error_mm": info["mass_balance_error_mm"],
            "decision_latency_ms": 0.0, "reward": reward,
        })
        state = next_state
    log = pd.DataFrame(records)
    metrics = compute_metrics(log)
    metrics["cumulative_reward"] = float(log["reward"].sum())
    return log, metrics


def _snapshot(agent):
    return {
        name: deepcopy(getattr(agent, name).state_dict())
        for name in ("actor", "critic1", "critic2", "target1", "target2")
    } | {"log_alpha": agent.log_alpha.detach().clone()}


def _restore(agent, state):
    for name in ("actor", "critic1", "critic2", "target1", "target2"):
        getattr(agent, name).load_state_dict(state[name])
    agent.log_alpha.data.copy_(state["log_alpha"])


def train(
    agent: SACSIRecurrentAgent,
    env: SACSIEnv,
    episodes: int,
    validation_env: SACSIEnv,
    validation_interval: int = 2,
    capacity: int = 100_000,
):
    seed = int(env.rng.bit_generator._seed_seq.entropy)
    set_seed(seed)
    replay = SACSIReplayBuffer(capacity, agent.config, seed)
    _, initial_metrics = evaluate(agent, validation_env)
    best_key, best_state, best_episode = _selection_key(initial_metrics), _snapshot(agent), 0
    history_rows, selection_rows, steps, updates = [], [{
        "episode": 0,
        "validation_gate": best_key[0],
        "selection_score": _selection_score(initial_metrics),
        **initial_metrics,
    }], 0, 0
    for episode in range(episodes):
        state, done, episode_reward, losses = env.reset(), False, 0.0, {}
        while not done:
            action = (
                np.asarray([env.rng.uniform(0, agent.config.action_max)], np.float32)
                if steps < agent.config.warmup else agent.select_action(state)
            )
            next_state, reward, done, _ = env.step(action)
            replay.add(state, action, reward, next_state, done)
            state, episode_reward, steps = next_state, episode_reward + reward, steps + 1
            if len(replay) >= agent.config.batch_size and steps >= agent.config.warmup:
                losses = agent.update(replay)
                updates += 1
        history_rows.append({
            "episode": episode + 1, "reward": episode_reward, "steps": steps, "updates": updates,
            **{key: float(value.cpu()) for key, value in losses.items()},
        })
        if (episode + 1) % validation_interval == 0 or episode + 1 == episodes:
            _, metrics = evaluate(agent, validation_env)
            score = _selection_score(metrics)
            key = _selection_key(metrics)
            selection_rows.append({
                "episode": episode + 1,
                "validation_gate": key[0],
                "selection_score": score,
                **metrics,
            })
            if key > best_key:
                best_key, best_state, best_episode = key, _snapshot(agent), episode + 1
    _restore(agent, best_state)
    return pd.DataFrame(history_rows), replay, pd.DataFrame(selection_rows), best_episode


def context_diagnostics(agent: SACSIRecurrentAgent, env: SACSIEnv, hours: int = 336):
    actor = deepcopy(agent.actor).cpu().eval()
    state = env.reset(start_index=0)
    deltas = {"zero_history": [], "reverse_history": [], "zero_forecast": [], "zero_context": []}
    for _ in range(min(hours, env.episode_length)):
        current, history, forecast = state
        current_tensor = torch.as_tensor(current, dtype=torch.float32).unsqueeze(0)

        def act(history_value, forecast_value):
            history_tensor = torch.as_tensor(history_value, dtype=torch.float32).unsqueeze(0)
            forecast_tensor = torch.as_tensor(forecast_value, dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                action, _ = actor(current_tensor, history_tensor, forecast_tensor, True, False)
            return float(action[0, 0])

        actual = act(history, forecast)
        zero_history = np.zeros_like(history)
        zero_forecast = np.zeros_like(forecast)
        deltas["zero_history"].append(abs(actual - act(zero_history, forecast)))
        deltas["reverse_history"].append(abs(actual - act(history[::-1].copy(), forecast)))
        deltas["zero_forecast"].append(abs(actual - act(history, zero_forecast)))
        deltas["zero_context"].append(abs(actual - act(zero_history, zero_forecast)))
        state, _, done, _ = env.step([actual])
        if done:
            break
    return {f"{name}_action_delta_mm": float(np.mean(values)) for name, values in deltas.items()}
