"""Controlled SACSI inference interventions and factorial interaction helpers."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from evaluation import compute_metrics


CONTEXT_CONDITIONS = (
    "Full",
    "No History",
    "No Forecast",
    "No Context",
    "Shuffled History",
    "Reversed History",
    "Zero History",
    "Shuffled Forecast",
    "Zero Forecast",
)


def controlled_action(actor, state, condition: str = "Full", forecast_override=None) -> float:
    """Deterministic SACSI action with an explicit branch/input intervention."""
    if condition not in CONTEXT_CONDITIONS:
        raise ValueError(f"Unknown context condition: {condition}")
    current, history, forecast = state
    if condition == "Reversed History":
        history = history[::-1].copy()
    elif condition == "Zero History":
        history = np.zeros_like(history)
    if condition == "Zero Forecast":
        forecast = np.zeros_like(forecast)
    elif forecast_override is not None:
        forecast = forecast_override

    current = torch.as_tensor(current, dtype=torch.float32).unsqueeze(0)
    history = torch.as_tensor(history, dtype=torch.float32).unsqueeze(0)
    forecast = torch.as_tensor(forecast, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        base_hidden = actor.base.body(current)
        _, (history_hidden, _) = actor.history_lstm(history)
        history_mean = actor.history_mean(history_hidden[-1])
        forecast_mean = actor.forecast_mean(actor.forecast_encoder(forecast))
        if condition in {"No History", "No Context"}:
            history_mean.zero_()
        if condition in {"No Forecast", "No Context"}:
            forecast_mean.zero_()
        mean = actor.base.mean(base_hidden) + history_mean + forecast_mean
        return float(((torch.tanh(mean) + 1) * actor.action_scale)[0, 0])


def evaluate_context(
    agent,
    env,
    condition: str = "Full",
    shuffle_seed: int = 2025,
    return_actions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    actor = deepcopy(agent.actor).cpu().eval()
    state, done, step, records, actions = env.reset(start_index=0), False, 0, [], []
    rng = np.random.default_rng(shuffle_seed)
    forecast_order = rng.permutation(len(env.data)) if condition == "Shuffled Forecast" else None
    while not done:
        current, history, forecast = state
        if condition == "Shuffled History":
            history = history[rng.permutation(len(history))]
        forecast_override = None
        if forecast_order is not None:
            row = env.data.iloc[int(forecast_order[step])]
            forecast_override = np.asarray([
                env._scale("precipitation_mm", row.forecast_precipitation_mm),
                env._scale("et0_mm", row.forecast_et0_mm),
                env._scale("temperature_c", row.forecast_temperature_c),
            ], dtype=np.float32)
            forecast_override = np.clip(forecast_override, -5, 5)
        action = controlled_action(
            actor, (current, history, forecast), condition, forecast_override
        )
        actions.append(action)
        state, reward, done, info = env.step([action])
        records.append({
            "theta": info["theta"],
            "irrigation_mm": info["irrigation_mm"],
            "runoff_mm": info["runoff_mm"],
            "drainage_mm": info["drainage_mm"],
            "mass_balance_error_mm": info["mass_balance_error_mm"],
            "decision_latency_ms": 0.0,
            "reward": reward,
        })
        step += 1
    log = pd.DataFrame(records)
    metrics = compute_metrics(log)
    metrics["cumulative_reward"] = float(log["reward"].sum())
    if return_actions:
        return metrics, np.asarray(actions, dtype=np.float32)
    return metrics


def factorial_interactions(runs: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    rl = runs.loc[runs["method_type"] == "rl", ["seed", "method", *metrics]].copy()
    if rl.groupby("seed")["method"].nunique().ne(4).any():
        raise ValueError("Each seed must contain all four factorial RL methods")
    rows = []
    for seed, group in rl.groupby("seed"):
        indexed = group.set_index("method")
        row = {"seed": int(seed)}
        for metric in metrics:
            row[f"{metric}_interaction"] = float(
                indexed.loc["SACSI Full", metric]
                - indexed.loc["SAC + Forecast", metric]
                - indexed.loc["SAC + LSTM", metric]
                + indexed.loc["SAC Basic", metric]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
