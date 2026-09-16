"""Train the three initial TD3 seeds on the locked fair protocol."""

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

from sac_basic import SACIrrigationEnv
from sac_basic.training import set_seed, validation_gate
from td3 import TD3Agent, TD3Config, evaluate, train


def run_seed(seed: int, episodes: int, episode_length: int, validation_interval: int, smoke: bool):
    data_dir = ROOT / "00_Dataset" / "Processed"
    training_data = pd.read_csv(data_dir / "train_2021_2023.csv", parse_dates=["timestamp"])
    validation_data = pd.read_csv(data_dir / "validation_2024.csv", parse_dates=["timestamp"])
    if smoke:
        training_data, validation_data = training_data.head(96), validation_data.head(48)
    set_seed(seed)
    agent = TD3Agent(TD3Config())
    training_env = SACIrrigationEnv(
        training_data, data_dir / "normalizer.json", episode_length, seed
    )
    validation_env = SACIrrigationEnv(
        validation_data, data_dir / "normalizer.json", len(validation_data), seed
    )
    history, selections, replay, selected_episode = train(
        agent, training_env, episodes, validation_env, validation_interval
    )
    validation_log, metrics = evaluate(agent, validation_env)
    observed_losses = history[["actor_loss", "critic_loss"]].dropna().to_numpy()
    row = {
        "model": "TD3",
        "seed": seed,
        "device": str(agent.device),
        "reward_version": training_env.reward_version,
        "observation_dim": training_env.observation_dim,
        "action_min_mm_h": 0.0,
        "action_max_mm_h": agent.config.action_max,
        "forecast": False,
        "history": False,
        "training_period": "2021-2023",
        "validation_period": "2024",
        "episodes": episodes,
        "episode_length": episode_length,
        "environment_interactions": episodes * episode_length,
        "selected_episode": selected_episode,
        "replay_size": len(replay),
        "critic_updates": agent.total_updates,
        "policy_updates": agent.total_updates // agent.config.policy_delay,
        "policy_delay": agent.config.policy_delay,
        "losses_finite": bool(np.isfinite(observed_losses).all()),
        "validation_gate": validation_gate(metrics),
        **metrics,
    }
    if smoke:
        print(json.dumps(row, indent=2))
        return row, history, selections

    result_dir = ROOT / "Results" / "TD3"
    log_dir = ROOT / "Logs" / "TD3"
    checkpoint = ROOT / "Checkpoints" / "TD3" / f"td3_seed{seed}_best.pt"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **row,
        "virtual_garden_version": "field_capacity_0.35",
        "checkpoint_selection": "validation_2024_gate_target_water_rmse",
        "benchmark_2025_accessed_for_training_or_selection": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    agent.save(checkpoint, metadata)
    (result_dir / f"td3_seed{seed}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    history.to_csv(log_dir / f"td3_seed{seed}_training.csv", index=False)
    selections.to_csv(log_dir / f"td3_seed{seed}_selection.csv", index=False)
    validation_log.to_csv(log_dir / f"td3_seed{seed}_validation.csv", index=False)
    return row, history, selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=336)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_seed(args.seeds[0], 12, 48, 6, True)
        return

    results, histories, selections = [], [], []
    for seed in args.seeds:
        row, history, selection = run_seed(
            seed, args.episodes, args.episode_length, args.validation_interval, False
        )
        results.append(row)
        histories.append(history)
        selections.append(selection)
        print(
            f"seed={seed} device={row['device']} selected={row['selected_episode']} "
            f"target={row['time_in_target_pct']:.2f}% water={row['total_irrigation_mm']:.2f}mm",
            flush=True,
        )
    result_dir = ROOT / "Results" / "TD3"
    pd.DataFrame(results).to_csv(result_dir / "td3_validation_results.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        result_dir / "td3_training_log.csv", index=False
    )
    pd.concat(selections, ignore_index=True).to_csv(
        result_dir / "td3_checkpoint_selection.csv", index=False
    )


if __name__ == "__main__":
    main()
