"""Train SAC Basic; smoke mode is the default safe local check."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sac_basic import SACAgent, SACConfig, SACIrrigationEnv
from sac_basic.training import evaluate, set_seed, train, validation_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-length", type=int, default=336)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    episodes = args.episodes or (4 if args.smoke else 200)

    data_dir = ROOT / "00_Dataset" / "Processed"
    normalizer = data_dir / "normalizer.json"
    training_data = pd.read_csv(data_dir / "train_2021_2023.csv", parse_dates=["timestamp"])
    validation_data = pd.read_csv(data_dir / "validation_2024.csv", parse_dates=["timestamp"])
    config = SACConfig()
    set_seed(args.seed)
    agent = SACAgent(config)
    training_env = SACIrrigationEnv(training_data, normalizer, args.episode_length, args.seed)
    validation_env = SACIrrigationEnv(validation_data, normalizer, len(validation_data), args.seed)
    history, replay, validation_history, selected_episode = train(
        agent, training_env, episodes,
        validation_env=validation_env,
        validation_interval=args.validation_interval,
    )
    validation_log, metrics = evaluate(agent, validation_env)
    gate = validation_gate(metrics)
    run_name = (
        f"sac_basic_seed{args.seed}_{training_env.reward_version}_"
        f"{'smoke' if args.smoke else 'training'}_ep{episodes}"
    )
    log_dir = ROOT / "Logs" / "SAC_Basic"
    result_dir = ROOT / "Results" / "SAC_Basic"
    checkpoint = ROOT / "Checkpoints" / "SAC_Basic" / f"{run_name}.pt"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(log_dir / f"{run_name}_history.csv", index=False)
    validation_history.to_csv(log_dir / f"{run_name}_selection.csv", index=False)
    validation_log.to_csv(log_dir / f"{run_name}_validation.csv", index=False)
    metadata = {
        "model": "SAC Basic",
        "device": str(agent.device),
        "seed": args.seed,
        "training_period": "2021-2023",
        "validation_period": "2024",
        "reward_version": training_env.reward_version,
        "virtual_garden_version": "field_capacity_0.35",
        "episodes": episodes,
        "episode_length": args.episode_length,
        "selected_episode": selected_episode,
        "transitions": len(replay),
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
