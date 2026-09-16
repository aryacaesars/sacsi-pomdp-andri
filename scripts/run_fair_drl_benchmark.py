"""Run the locked three-seed DDPG-TD3-SAC fairness benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpg import DDPGAgent, evaluate as evaluate_ddpg
from evaluation.final_benchmark import FORMAL_METRICS, sha256_file, validate_common_support
from sac_basic import SACAgent, SACConfig, SACIrrigationEnv
from sac_basic.training import evaluate as evaluate_sac
from sac_basic.training import set_seed, train, validation_gate
from td3 import TD3Agent, evaluate as evaluate_td3


DATA = ROOT / "00_Dataset" / "Processed"
RESULTS = ROOT / "Results" / "Fair_DRL"
LOGS = ROOT / "Logs" / "Fair_DRL"
SAC_CHECKPOINTS = ROOT / "Checkpoints" / "Fair_DRL" / "SAC"
SEEDS = (11, 22, 33)
ALGORITHM_SPECS = {
    "DDPG": (DDPGAgent, evaluate_ddpg),
    "TD3": (TD3Agent, evaluate_td3),
    "SAC": (SACAgent, evaluate_sac),
}


def checkpoint_path(algorithm: str, seed: int) -> Path:
    if algorithm == "SAC":
        return SAC_CHECKPOINTS / f"sac_seed{seed}_best.pt"
    return ROOT / "Checkpoints" / algorithm / f"{algorithm.lower()}_seed{seed}_best.pt"


def actual_common_fields(config, metadata: dict) -> dict:
    return {
        "virtual_garden_version": metadata["virtual_garden_version"],
        "observation_dim": int(config.observation_dim),
        "action_dim": int(config.action_dim),
        "action_min_mm_h": float(metadata["action_min_mm_h"]),
        "action_max_mm_h": float(config.action_max),
        "hidden_dim": int(config.hidden_dim),
        "batch_size": int(config.batch_size),
        "warmup_steps": int(config.warmup),
        "actor_learning_rate": float(config.actor_lr),
        "critic_learning_rate": float(config.critic_lr),
        "gamma": float(config.gamma),
        "tau": float(config.tau),
        "reward_version": metadata["reward_version"],
        "training_period": metadata["training_period"],
        "validation_period": metadata["validation_period"],
        "episodes_per_seed": int(metadata["episodes"]),
        "episode_length_hours": int(metadata["episode_length"]),
        "environment_interactions_per_seed": int(metadata["environment_interactions"]),
        "validation_interval_episodes": int(metadata.get("validation_interval", 5)),
        "checkpoint_selection": metadata["checkpoint_selection"],
        "metric_engine": "evaluation.compute_metrics",
        "forecast": bool(metadata["forecast"]),
        "history": bool(metadata["history"]),
    }


def config_hash(fields: dict) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def train_sac_seed(
    seed: int,
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    episodes: int,
    episode_length: int,
    validation_interval: int,
    smoke: bool,
):
    set_seed(seed)
    agent = SACAgent(SACConfig())
    training_env = SACIrrigationEnv(
        training_data, DATA / "normalizer.json", episode_length, seed
    )
    validation_env = SACIrrigationEnv(
        validation_data, DATA / "normalizer.json", len(validation_data), seed
    )
    history, replay, selection, selected_episode = train(
        agent,
        training_env,
        episodes,
        validation_env=validation_env,
        validation_interval=validation_interval,
    )
    validation_log, metrics = evaluate_sac(agent, validation_env)
    observed_losses = history[["actor_loss", "critic_loss", "alpha_loss"]].dropna().to_numpy()
    row = {
        "model": "SAC",
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
        "validation_interval": validation_interval,
        "selected_episode": selected_episode,
        "replay_size": len(replay),
        "losses_finite": bool(np.isfinite(observed_losses).all()),
        "validation_gate": validation_gate(metrics),
        **metrics,
    }
    if smoke:
        print(json.dumps(row, indent=2))
        return row, history, selection

    metadata = {
        **row,
        "virtual_garden_version": "field_capacity_0.35",
        "checkpoint_selection": "validation_2024_gate_target_water_rmse",
        "benchmark_2025_accessed_for_training_or_selection": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint = checkpoint_path("SAC", seed)
    agent.save(checkpoint, metadata)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"sac_seed{seed}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    history.insert(0, "seed", seed)
    selection.insert(0, "seed", seed)
    history.to_csv(LOGS / f"sac_seed{seed}_training.csv", index=False)
    selection.to_csv(LOGS / f"sac_seed{seed}_selection.csv", index=False)
    validation_log.to_csv(LOGS / f"sac_seed{seed}_validation.csv", index=False)
    return row, history, selection


def build_entries() -> list[dict]:
    entries = []
    for algorithm, (agent_class, _) in ALGORITHM_SPECS.items():
        for seed in SEEDS:
            checkpoint = checkpoint_path(algorithm, seed)
            agent, metadata = agent_class.load(checkpoint, device="cpu")
            common = actual_common_fields(agent.config, metadata)
            entries.append({
                "algorithm_family": algorithm,
                "seed": seed,
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "selected_episode": int(metadata["selected_episode"]),
                "checkpoint_validation_gate": bool(metadata["validation_gate"]),
                "training_device": metadata["device"],
                "reward_version": metadata["reward_version"],
                "common_config_hash": config_hash(common),
                "result_status": "VALIDATION",
                "common_fields": common,
            })
    return entries


def build_fairness_audit(entries: list[dict]) -> dict:
    hashes = {
        algorithm: sorted({
            entry["common_config_hash"]
            for entry in entries if entry["algorithm_family"] == algorithm
        })
        for algorithm in ALGORITHM_SPECS
    }
    checks = {
        "common_fields_match": len({value for values in hashes.values() for value in values}) == 1,
        "one_common_hash_per_algorithm": all(len(values) == 1 for values in hashes.values()),
        "matched_seeds_complete": all(
            sorted(entry["seed"] for entry in entries if entry["algorithm_family"] == algorithm)
            == list(SEEDS)
            for algorithm in ALGORITHM_SPECS
        ),
        "nine_checkpoints_registered": len(entries) == 9,
        "failed_seeds_retained": len(entries) == 9,
        "reward_v4_locked": {entry["reward_version"] for entry in entries} == {"reward_v4"},
        "validation_only_checkpoint_selection": all(
            entry["common_fields"]["checkpoint_selection"]
            == "validation_2024_gate_target_water_rmse"
            for entry in entries
        ),
        "matched_interaction_budget": {
            entry["common_fields"]["environment_interactions_per_seed"] for entry in entries
        } == {6720},
        "no_forecast_or_history": all(
            not entry["common_fields"]["forecast"] and not entry["common_fields"]["history"]
            for entry in entries
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Fairness audit failed: {checks}")
    return {
        "module": "8F",
        "status": "PASS",
        "benchmark_classification": "retrospective final benchmark 2025",
        "common_config_hash_by_algorithm": {
            algorithm: values[0] for algorithm, values in hashes.items()
        },
        "common_fields": entries[0]["common_fields"],
        "checks": checks,
        "algorithm_specific_mechanisms_preserved": {
            "DDPG": ["single critic", "deterministic actor", "Gaussian exploration"],
            "TD3": ["twin critics", "target smoothing", "policy delay 2"],
            "SAC": ["twin critics", "stochastic actor", "automatic entropy tuning"],
        },
    }


def result_rows(entries: list[dict], data: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for entry in entries:
        algorithm = entry["algorithm_family"]
        agent_class, evaluator = ALGORITHM_SPECS[algorithm]
        agent, metadata = agent_class.load(ROOT / entry["checkpoint"], device="cpu")
        env = SACIrrigationEnv(data, DATA / "normalizer.json", len(data), entry["seed"])
        _, metrics = evaluator(agent, env)
        rows.append({
            "model": algorithm,
            "algorithm_family": algorithm,
            "method_type": "rl",
            "seed": entry["seed"],
            "evaluation_split": split,
            "evaluation_period": "2024" if split == "validation" else "2025",
            "evaluation_hours": len(data),
            "result_status": "VALIDATION" if split == "validation" else "RETROSPECTIVE_BENCHMARK",
            "training_device": entry["training_device"],
            "inference_device": "cpu",
            "reward_version": metadata["reward_version"],
            "observation_dim": metadata["observation_dim"],
            "action_min_mm_h": metadata["action_min_mm_h"],
            "action_max_mm_h": metadata["action_max_mm_h"],
            "forecast": metadata["forecast"],
            "history": metadata["history"],
            "training_period": metadata["training_period"],
            "episodes": metadata["episodes"],
            "episode_length": metadata["episode_length"],
            "environment_interactions": metadata["environment_interactions"],
            "selected_episode": metadata["selected_episode"],
            "checkpoint_validation_gate": entry["checkpoint_validation_gate"],
            "evaluation_gate": validation_gate(metrics) if split == "validation" else None,
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "common_config_hash": entry["common_config_hash"],
            **{name: metrics[name] for name in FORMAL_METRICS},
            "cumulative_reward": metrics["cumulative_reward"],
        })
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[list(FORMAL_METRICS) + ["cumulative_reward"]].to_numpy()).all():
        raise RuntimeError(f"Non-finite {split} metrics")
    if frame["max_abs_mass_balance_error_mm"].max() > 1e-8:
        raise RuntimeError(f"Mass-balance gate failed on {split}")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ("time_in_target_pct", "total_irrigation_mm", "rmse_band", "violation_rate_pct")
    summary = frame.groupby("algorithm_family", sort=False)[list(metrics)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(
        ["time_in_target_pct_mean", "total_irrigation_mm_mean"], ascending=[False, True]
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.insert(2, "n_seeds", len(SEEDS))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    training_data = pd.read_csv(DATA / "train_2021_2023.csv", parse_dates=["timestamp"])
    validation_data = pd.read_csv(DATA / "validation_2024.csv", parse_dates=["timestamp"])
    if args.smoke:
        train_sac_seed(11, training_data.head(96), validation_data.head(48), 12, 48, 6, True)
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    histories, selections = [], []
    for seed in SEEDS:
        row, history, selection = train_sac_seed(
            seed, training_data, validation_data, 20, 336, 5, False
        )
        histories.append(history)
        selections.append(selection)
        print(
            f"SAC seed={seed} device={row['device']} selected={row['selected_episode']} "
            f"target={row['time_in_target_pct']:.2f}% water={row['total_irrigation_mm']:.2f}mm",
            flush=True,
        )
    pd.concat(histories, ignore_index=True).to_csv(
        RESULTS / "sac_fair_training_log.csv", index=False
    )
    pd.concat(selections, ignore_index=True).to_csv(
        RESULTS / "sac_fair_checkpoint_selection.csv", index=False
    )

    entries = build_entries()
    audit = build_fairness_audit(entries)
    registry = pd.DataFrame([
        {key: value for key, value in entry.items() if key != "common_fields"}
        for entry in entries
    ])
    registry.to_csv(RESULTS / "fair_drl_checkpoint_registry.csv", index=False)

    validation_results = result_rows(entries, validation_data, "validation")
    validation_results.to_csv(RESULTS / "fair_drl_results_validation.csv", index=False)
    summarize(validation_results).to_csv(RESULTS / "fair_drl_summary_validation.csv", index=False)

    benchmark_data = pd.read_csv(DATA / "benchmark_2025.csv", parse_dates=["timestamp"])
    validate_common_support(benchmark_data)
    benchmark_results = result_rows(entries, benchmark_data, "retrospective_benchmark")
    benchmark_results.to_csv(RESULTS / "fair_drl_results_2025.csv", index=False)
    summarize(benchmark_results).to_csv(RESULTS / "fair_drl_summary_2025.csv", index=False)
    audit.update({
        "benchmark_2025_loaded_after_training_and_fairness_lock": True,
        "no_retraining_or_checkpoint_reselection_after_2025_opening": True,
        "validation_rows": len(validation_results),
        "benchmark_2025_rows": len(benchmark_results),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    })
    (RESULTS / "fairness_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summarize(benchmark_results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
