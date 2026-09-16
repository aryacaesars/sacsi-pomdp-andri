"""Train SAC + Forecast using the controlled SF-20 h+1 proxy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sac_basic import SACAgent, SACConfig
from sac_basic.training import evaluate, set_seed, train, validation_gate
from sac_forecast import SACForecastEnv


def load_split(data_dir: Path, filename: str) -> pd.DataFrame:
    weather = pd.read_csv(data_dir / filename, parse_dates=["timestamp"])
    forecast = pd.read_csv(data_dir / "synthetic_forecast_sf20.csv", parse_dates=["timestamp"])
    merged = weather.merge(forecast, on="timestamp", how="left", validate="one_to_one")
    if merged.filter(like="forecast_").isna().any().any():
        raise ValueError("Missing SF-20 forecast context after timestamp merge")
    return merged


def forecast_intervention_delta(agent: SACAgent, env: SACForecastEnv, hours: int = 336) -> float:
    observation = env.reset(start_index=0)
    deltas = []
    for _ in range(min(hours, env.episode_length)):
        action = agent.select_action(observation, deterministic=True)
        no_forecast = observation.copy()
        no_forecast[-3:] = 0
        ablated_action = agent.select_action(no_forecast, deterministic=True)
        deltas.append(abs(float(action[0]) - float(ablated_action[0])))
        observation, _, done, _ = env.step(action)
        if done:
            break
    return float(np.mean(deltas))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-length", type=int, default=336)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    episodes = args.episodes or (4 if args.smoke else 100)

    data_dir = ROOT / "00_Dataset" / "Processed"
    forecast_path = data_dir / "synthetic_forecast_sf20.csv"
    if not forecast_path.exists():
        raise FileNotFoundError("Run scripts/prepare_synthetic_forecast.py first")
    training_data = load_split(data_dir, "train_2021_2023.csv")
    validation_data = load_split(data_dir, "validation_2024.csv")
    normalizer = data_dir / "normalizer.json"

    config = SACConfig(observation_dim=11)
    set_seed(args.seed)
    agent = SACAgent(config)
    training_env = SACForecastEnv(training_data, normalizer, args.episode_length, args.seed)
    validation_env = SACForecastEnv(validation_data, normalizer, len(validation_data), args.seed)
    history, replay, selection, selected_episode = train(
        agent, training_env, episodes,
        validation_env=validation_env,
        validation_interval=args.validation_interval,
    )
    validation_log, metrics = evaluate(agent, validation_env)
    intervention_delta = forecast_intervention_delta(agent, validation_env)
    gate = validation_gate(metrics)
    run_name = (
        f"sac_forecast_seed{args.seed}_{training_env.reward_version}_sf20_"
        f"{'smoke' if args.smoke else 'training'}_ep{episodes}"
    )
    log_dir = ROOT / "Logs" / "SAC_Forecast"
    result_dir = ROOT / "Results" / "SAC_Forecast"
    checkpoint = ROOT / "Checkpoints" / "SAC_Forecast" / f"{run_name}.pt"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(log_dir / f"{run_name}_history.csv", index=False)
    selection.to_csv(log_dir / f"{run_name}_selection.csv", index=False)
    validation_log.to_csv(log_dir / f"{run_name}_validation.csv", index=False)
    metadata = {
        "model": "SAC + Forecast",
        "device": str(agent.device),
        "seed": args.seed,
        "training_period": "2021-2023",
        "validation_period": "2024",
        "forecast_protocol": training_env.forecast_protocol,
        "forecast_horizon": 1,
        "forecast_error": "20%",
        "reward_version": training_env.reward_version,
        "virtual_garden_version": "field_capacity_0.35",
        "episodes": episodes,
        "episode_length": args.episode_length,
        "selected_episode": selected_episode,
        "transitions": len(replay),
        "forecast_intervention_action_delta_mm": intervention_delta,
        "validation_gate": gate,
        "validation_metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    agent.save(checkpoint, metadata)
    (result_dir / f"{run_name}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(history.tail().to_string(index=False))
    print(json.dumps(metadata, indent=2))
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
