"""Run resumable Module 8B reward ablation and local weight sensitivity."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.reward_validation import (
    CONFIRMATION_SEEDS,
    SEEDS,
    RewardExperimentSpec,
    confirmation_decision,
    confirmation_specs,
    experiment_specs,
    reconcile_confirmation,
    reward_decision,
    summarize_candidates,
)
from sac_basic import SACAgent, SACConfig, SACIrrigationEnv
from sac_basic.training import evaluate, set_seed, train


def run_one(
    spec: RewardExperimentSpec,
    seed: int,
    episodes: int,
    episode_length: int,
    run_dir: Path,
) -> Path:
    output = run_dir / f"{spec.config_id}_seed{seed}.json"
    if output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached["episodes"] == episodes and cached["episode_length"] == episode_length:
            numeric_values = [
                value for key, value in cached.items()
                if key != "training_finite" and isinstance(value, (int, float))
            ]
            cached["training_finite"] = bool(np.isfinite(numeric_values).all())
            output.write_text(json.dumps(cached, indent=2), encoding="utf-8")
            return output

    data_dir = ROOT / "00_Dataset" / "Processed"
    training = pd.read_csv(data_dir / "train_2021_2023.csv", parse_dates=["timestamp"])
    validation = pd.read_csv(data_dir / "validation_2024.csv", parse_dates=["timestamp"])
    set_seed(seed)
    agent = SACAgent(SACConfig())
    reward_config = spec.reward_config()
    training_env = SACIrrigationEnv(
        training, data_dir / "normalizer.json", episode_length, seed, reward_config
    )
    validation_env = SACIrrigationEnv(
        validation, data_dir / "normalizer.json", len(validation), seed, reward_config
    )
    history, replay, _, _ = train(agent, training_env, episodes)
    log, metrics = evaluate(agent, validation_env)
    numeric_history = history.select_dtypes(include="number")
    observed_training_values = np.concatenate([
        numeric_history[column].dropna().to_numpy()
        for column in numeric_history
    ])
    row = {
        "config_id": spec.config_id,
        "seed": seed,
        "device": str(agent.device),
        "episodes": episodes,
        "episode_length": episode_length,
        "environment_interactions": episodes * episode_length,
        "training_finite": bool(np.isfinite(observed_training_values).all()),
        "water_multiplier": spec.water_multiplier,
        "violation_multiplier": spec.violation_multiplier,
        "tracking_weight": reward_config.tracking_weight,
        "deficit_ratio": reward_config.deficit_ratio,
        "water_weight": reward_config.water_weight,
        "smoothness_weight": reward_config.smoothness_weight,
        "violation_weight": reward_config.violation_weight,
        "ablation_label": spec.ablation_label,
        "sensitivity_label": spec.sensitivity_label,
        **metrics,
    }
    for term in ("tracking", "water", "smoothness", "violation"):
        values = log[f"reward_{term}_cost"]
        row[f"reward_{term}_cost_total"] = float(values.sum())
        row[f"reward_{term}_cost_mean"] = float(values.mean())
        row[f"reward_{term}_cost_max"] = float(values.max())
    row["reward_offset_total"] = float(log["reward_offset"].sum())
    row["replay_size"] = len(replay)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def build_outputs(run_files: list[Path], output_dir: Path) -> dict:
    raw = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")) for path in run_files)
    ablation = raw.loc[raw["ablation_label"].notna()].copy()
    ablation.insert(0, "experiment", "ablation")
    ablation.insert(1, "candidate_id", ablation.pop("ablation_label"))
    sensitivity = raw.loc[raw["sensitivity_label"].notna()].copy()
    sensitivity.insert(0, "experiment", "sensitivity")
    sensitivity.insert(1, "candidate_id", sensitivity.pop("sensitivity_label"))
    for frame in (ablation, sensitivity):
        frame.insert(2, "data_split", "validation_2024")
        frame.sort_values(["candidate_id", "seed"], inplace=True)

    ablation.to_csv(output_dir / "reward_ablation_results.csv", index=False)
    sensitivity.to_csv(output_dir / "reward_weight_sensitivity.csv", index=False)
    ablation_summary = summarize_candidates(ablation, "ablation")
    sensitivity_summary = summarize_candidates(sensitivity, "sensitivity")
    pareto = pd.concat((ablation_summary, sensitivity_summary), ignore_index=True)
    pareto.to_csv(output_dir / "reward_pareto.csv", index=False)

    decision = reward_decision(sensitivity_summary)
    confirmation_path = output_dir / "reward_confirmation_decision.json"
    if confirmation_path.exists():
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        decision = reconcile_confirmation(decision, confirmation)
    (output_dir / "reward_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    figure_dir = ROOT / "Figures" / "Reward_Validation"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure = px.scatter(
        sensitivity_summary,
        x="total_irrigation_mm_mean",
        y="time_in_target_pct_mean",
        color="pareto_non_dominated",
        hover_name="candidate_id",
        error_x="total_irrigation_mm_std",
        error_y="time_in_target_pct_std",
        title="Module 8B — Validation 2024 Reward Pareto",
        labels={
            "total_irrigation_mm_mean": "Mean irrigation (mm)",
            "time_in_target_pct_mean": "Mean Time in Target (%)",
        },
    )
    figure.write_html(
        figure_dir / "reward_pareto_validation_2024.html",
        include_plotlyjs=True,
    )
    return decision


def run_confirmation(episodes: int, episode_length: int) -> dict:
    output_dir = ROOT / "Results" / "Reward_Validation"
    development_dir = output_dir / "runs"
    confirmation_dir = output_dir / "confirmation_runs"
    specs = confirmation_specs()
    rows = []
    jobs = [(spec, seed) for spec in specs for seed in CONFIRMATION_SEEDS]
    for completed, (spec, seed) in enumerate(jobs, 1):
        cached = development_dir / f"{spec.config_id}_seed{seed}.json"
        path = cached if cached.exists() else run_one(
            spec, seed, episodes, episode_length, confirmation_dir
        )
        row = json.loads(path.read_text(encoding="utf-8"))
        row.update({
            "experiment": "confirmation",
            "candidate_id": spec.sensitivity_label,
            "data_split": "validation_2024",
            "reused_development_run": cached.exists(),
        })
        rows.append(row)
        print(f"[{completed:02d}/{len(jobs)}] {spec.config_id} seed={seed}", flush=True)

    results = pd.DataFrame(rows).sort_values(["candidate_id", "seed"])
    results.to_csv(output_dir / "reward_confirmation_results.csv", index=False)
    summary = summarize_candidates(results, "confirmation")
    summary.to_csv(output_dir / "reward_confirmation_summary.csv", index=False)
    decision = confirmation_decision(summary)
    (output_dir / "reward_confirmation_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    base_path = output_dir / "reward_decision.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_path.write_text(
        json.dumps(reconcile_confirmation(base, decision), indent=2), encoding="utf-8"
    )
    figure = px.scatter(
        summary,
        x="total_irrigation_mm_mean",
        y="time_in_target_pct_mean",
        color="pareto_non_dominated",
        hover_name="candidate_id",
        error_x="total_irrigation_mm_std",
        error_y="time_in_target_pct_std",
        title="Module 8B — 10-Seed Reward Confirmation (Validation 2024)",
    )
    figure.write_html(
        ROOT / "Figures" / "Reward_Validation" / "reward_confirmation_10seed_2024.html",
        include_plotlyjs=True,
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=336)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    specs = experiment_specs()

    if args.confirm:
        if args.workers != 1:
            raise ValueError("reward confirmation uses one GPU worker to avoid VRAM contention")
        decision = run_confirmation(args.episodes, args.episode_length)
        print(json.dumps(decision, indent=2))
        return

    if args.smoke:
        smoke_dir = ROOT / "Results" / "Reward_Validation_Smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        path = run_one(specs[0], SEEDS[0], 1, 48, smoke_dir)
        result = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({key: result[key] for key in (
            "config_id", "seed", "device", "training_finite", "time_in_target_pct"
        )}, indent=2))
        return

    output_dir = ROOT / "Results" / "Reward_Validation"
    run_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(spec, seed) for spec in specs for seed in SEEDS]
    run_files = []
    if args.workers == 1:
        for completed, (spec, seed) in enumerate(jobs, 1):
            run_files.append(run_one(spec, seed, args.episodes, args.episode_length, run_dir))
            print(f"[{completed:02d}/{len(jobs)}] {spec.config_id} seed={seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one, spec, seed, args.episodes, args.episode_length, run_dir):
                (spec.config_id, seed)
                for spec, seed in jobs
            }
            for completed, future in enumerate(as_completed(futures), 1):
                path = future.result()
                run_files.append(path)
                config_id, seed = futures[future]
                print(f"[{completed:02d}/{len(jobs)}] {config_id} seed={seed}", flush=True)
    decision = build_outputs(run_files, output_dir)
    print(json.dumps({"runs": len(run_files), "decision": decision["decision"]}, indent=2))


if __name__ == "__main__":
    main()
