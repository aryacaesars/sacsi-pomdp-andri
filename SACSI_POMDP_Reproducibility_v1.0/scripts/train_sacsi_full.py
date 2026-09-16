"""Train SACSI Full with matched SAC Basic RRWS and SF-20 h1 context."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sac_basic.training import set_seed, validation_gate
from sacsi_full import SACSIConfig, SACSIEnv, SACSIRecurrentAgent
from sacsi_full.training import context_diagnostics, evaluate, train


def load_split(data_dir: Path, filename: str) -> pd.DataFrame:
    weather = pd.read_csv(data_dir / filename, parse_dates=["timestamp"])
    forecast = pd.read_csv(data_dir / "synthetic_forecast_sf20.csv", parse_dates=["timestamp"])
    merged = weather.merge(forecast, on="timestamp", how="left", validate="one_to_one")
    if merged.filter(like="forecast_").isna().any().any():
        raise ValueError("Missing SF-20 context")
    return merged


def basic_checkpoint(seed: int) -> Path:
    registry = pd.read_csv(ROOT / "Results" / "SAC_Basic" / "validation_registry.csv")
    row = registry.loc[registry["seed"] == seed]
    if len(row) != 1:
        raise ValueError(f"No unique SAC Basic checkpoint for seed {seed}")
    return ROOT / Path(row.iloc[0]["checkpoint"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-length", type=int, default=336)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    episodes = args.episodes or (4 if args.smoke else 10)
    data_dir = ROOT / "00_Dataset" / "Processed"
    training_data = load_split(data_dir, "train_2021_2023.csv")
    validation_data = load_split(data_dir, "validation_2024.csv")
    normalizer = data_dir / "normalizer.json"
    config = SACSIConfig(sequence_length=args.sequence_length)
    set_seed(args.seed)
    agent = SACSIRecurrentAgent(config)
    source_checkpoint = basic_checkpoint(args.seed)
    agent.warm_start(source_checkpoint)
    training_env = SACSIEnv(
        training_data, normalizer, args.episode_length, args.seed,
        sequence_length=args.sequence_length,
    )
    validation_env = SACSIEnv(
        validation_data, normalizer, len(validation_data), args.seed,
        sequence_length=args.sequence_length,
    )
    history, replay, selection, selected_episode = train(
        agent, training_env, episodes, validation_env, args.validation_interval
    )
    validation_log, metrics = evaluate(agent, validation_env)
    diagnostics = context_diagnostics(agent, validation_env)
    norms = agent.context_norms()
    gate = validation_gate(metrics)
    run_name = (
        f"sacsi_full_seed{args.seed}_{training_env.reward_version}_sf20_rrws_k{args.sequence_length}_"
        f"{'smoke' if args.smoke else 'training'}_ep{episodes}"
    )
    log_dir = ROOT / "Logs" / "SACSI_Full"
    result_dir = ROOT / "Results" / "SACSI_Full"
    checkpoint = ROOT / "Checkpoints" / "SACSI_Full" / f"{run_name}.pt"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(log_dir / f"{run_name}_history.csv", index=False)
    selection.to_csv(log_dir / f"{run_name}_selection.csv", index=False)
    validation_log.to_csv(log_dir / f"{run_name}_validation.csv", index=False)
    metadata = {
        "model": "SACSI Full", "device": str(agent.device), "seed": args.seed,
        "training_period": "2021-2023", "validation_period": "2024",
        "sequence_length": args.sequence_length, "forecast_horizon": 1,
        "forecast_protocol": training_env.forecast_protocol,
        "training_strategy": "Multi-Context Residual Recurrent Warm-Start",
        "warm_start_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "reward_version": training_env.reward_version,
        "virtual_garden_version": "field_capacity_0.35",
        "episodes": episodes, "episode_length": args.episode_length,
        "selected_episode": selected_episode, "transitions": len(replay),
        **norms, **diagnostics,
        "validation_gate": gate, "validation_metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    agent.save(checkpoint, metadata)
    (result_dir / f"{run_name}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(history.tail().to_string(index=False))
    print(json.dumps(metadata, indent=2))
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
