"""Train residual recurrent SAC from the matched SAC Basic checkpoint."""

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
from sac_lstm import RecurrentSACAgent, RecurrentSACConfig, SACLSTMEnv
from sac_lstm.training import evaluate, memory_diagnostics, train


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
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    episodes = args.episodes or (4 if args.smoke else 100)

    data_dir = ROOT / "00_Dataset" / "Processed"
    training_data = pd.read_csv(data_dir / "train_2021_2023.csv", parse_dates=["timestamp"])
    validation_data = pd.read_csv(data_dir / "validation_2024.csv", parse_dates=["timestamp"])
    normalizer = data_dir / "normalizer.json"
    config = RecurrentSACConfig(sequence_length=args.sequence_length)
    set_seed(args.seed)
    agent = RecurrentSACAgent(config)
    source_checkpoint = basic_checkpoint(args.seed)
    agent.warm_start(source_checkpoint)
    training_env = SACLSTMEnv(
        training_data, normalizer, args.episode_length, args.seed,
        sequence_length=args.sequence_length,
    )
    validation_env = SACLSTMEnv(
        validation_data, normalizer, len(validation_data), args.seed,
        sequence_length=args.sequence_length,
    )
    history, replay, selection, selected_episode = train(
        agent, training_env, episodes, validation_env, args.validation_interval
    )
    validation_log, metrics = evaluate(agent, validation_env)
    diagnostics = memory_diagnostics(agent, validation_env)
    gate = validation_gate(metrics)
    run_name = (
        f"sac_lstm_seed{args.seed}_{training_env.reward_version}_rrws_k{args.sequence_length}_"
        f"{'smoke' if args.smoke else 'training'}_ep{episodes}"
    )
    log_dir = ROOT / "Logs" / "SAC_LSTM"
    result_dir = ROOT / "Results" / "SAC_LSTM"
    checkpoint = ROOT / "Checkpoints" / "SAC_LSTM" / f"{run_name}.pt"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(log_dir / f"{run_name}_history.csv", index=False)
    selection.to_csv(log_dir / f"{run_name}_selection.csv", index=False)
    validation_log.to_csv(log_dir / f"{run_name}_validation.csv", index=False)
    metadata = {
        "model": "SAC + LSTM", "device": str(agent.device), "seed": args.seed,
        "training_period": "2021-2023", "validation_period": "2024",
        "sequence_length": args.sequence_length, "lstm_hidden_dim": config.lstm_hidden_dim,
        "training_strategy": "Residual Recurrent Warm-Start (RRWS)",
        "warm_start_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "reward_version": training_env.reward_version,
        "virtual_garden_version": "field_capacity_0.35",
        "episodes": episodes, "episode_length": args.episode_length,
        "selected_episode": selected_episode, "transitions": len(replay),
        "context_residual_norm": agent.residual_norm(), **diagnostics,
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
