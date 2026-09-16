"""Training, validation selection, and memory diagnostics for RRWS SAC."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from evaluation import compute_metrics
from sac_basic.training import _selection_key, _selection_score, set_seed

from .agent import RecurrentReplayBuffer, RecurrentSACAgent
from .environment import SACLSTMEnv


def evaluate(agent: RecurrentSACAgent, env: SACLSTMEnv):
    actor = deepcopy(agent.actor).cpu().eval()
    state, done, records = env.reset(start_index=0), False, []
    while not done:
        current = torch.as_tensor(state[0], dtype=torch.float32).unsqueeze(0)
        history = torch.as_tensor(state[1], dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            action, _ = actor(current, history, True, False)
        next_state, reward, done, info = env.step(action.numpy()[0])
        row = env.data.iloc[env.index if done else env.index - 1]
        records.append({
            "timestamp": row.timestamp, "controller": "SAC + LSTM",
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
    agent: RecurrentSACAgent,
    env: SACLSTMEnv,
    episodes: int,
    validation_env: SACLSTMEnv | None = None,
    validation_interval: int = 10,
    capacity: int = 100_000,
):
    seed = int(env.rng.bit_generator._seed_seq.entropy)
    set_seed(seed)
    replay = RecurrentReplayBuffer(capacity, agent.config, seed)
    history_rows, selection_rows, steps, updates = [], [], 0, 0
    best_key, best_state, best_episode = None, None, None
    if validation_env is not None:
        _, initial_metrics = evaluate(agent, validation_env)
        best_key, best_state, best_episode = _selection_key(initial_metrics), _snapshot(agent), 0
        selection_rows.append({
            "episode": 0,
            "validation_gate": best_key[0],
            "selection_score": _selection_score(initial_metrics),
            **initial_metrics,
        })
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
        scalar_losses = {key: float(value.cpu()) for key, value in losses.items()}
        history_rows.append({
            "episode": episode + 1, "reward": episode_reward, "steps": steps,
            "updates": updates, **scalar_losses,
        })
        if validation_env is not None and (
            (episode + 1) % validation_interval == 0 or episode + 1 == episodes
        ):
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
    if best_state is not None:
        _restore(agent, best_state)
    return pd.DataFrame(history_rows), replay, pd.DataFrame(selection_rows), best_episode


def memory_diagnostics(agent: RecurrentSACAgent, env: SACLSTMEnv, hours: int = 336):
    actor = deepcopy(agent.actor).cpu().eval()
    state, deltas = env.reset(start_index=0), {"zero": [], "reverse": [], "shuffle": []}
    rng = np.random.default_rng(2024)
    for _ in range(min(hours, env.episode_length)):
        current, history = state
        current_tensor = torch.as_tensor(current, dtype=torch.float32).unsqueeze(0)

        def act(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                action, _ = actor(current_tensor, sequence, True, False)
            return float(action[0, 0])

        actual = act(history)
        deltas["zero"].append(abs(actual - act(np.zeros_like(history))))
        deltas["reverse"].append(abs(actual - act(history[::-1].copy())))
        deltas["shuffle"].append(abs(actual - act(history[rng.permutation(len(history))])))
        state, _, done, _ = env.step([actual])
        if done:
            break
    return {f"{name}_history_action_delta_mm": float(np.mean(values)) for name, values in deltas.items()}
