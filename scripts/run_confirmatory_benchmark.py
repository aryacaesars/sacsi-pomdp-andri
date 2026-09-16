"""Run Module 8H's locked 10-seed confirmatory benchmark and statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ddpg import DDPGAgent, evaluate as evaluate_ddpg  # noqa: E402
from evaluation.final_benchmark import FORMAL_METRICS, sha256_file, validate_common_support  # noqa: E402
from evaluation.statistics import (  # noqa: E402
    factorial_contrasts,
    holm_adjust,
    one_df_repeated_effect,
    paired_comparisons,
    repeated_measures_omnibus,
)
from sac_basic import SACAgent, SACIrrigationEnv  # noqa: E402
from sac_basic.training import evaluate as evaluate_sac  # noqa: E402
from sac_forecast import SACForecastEnv  # noqa: E402
from sac_lstm import SACLSTMEnv  # noqa: E402
from sac_lstm.training import evaluate as evaluate_lstm  # noqa: E402
from sacsi_full import SACSIEnv  # noqa: E402
from sacsi_full.training import evaluate as evaluate_sacsi  # noqa: E402
from scripts.prepare_synthetic_forecast import build_synthetic_forecast  # noqa: E402
from scripts.run_fair_drl_benchmark import train_sac_seed  # noqa: E402
from scripts.run_pomdp_ablation import (  # noqa: E402
    evaluate_factorial,
    load_context_agent,
    merge_forecast,
    train_forecast_variant,
    train_full_variant,
    train_lstm_variant,
)
from scripts.train_ddpg import run_seed as train_ddpg_seed  # noqa: E402
from scripts.train_td3 import run_seed as train_td3_seed  # noqa: E402
from td3 import TD3Agent, evaluate as evaluate_td3  # noqa: E402


DATA = ROOT / "00_Dataset" / "Processed"
RESULTS = ROOT / "Results" / "Confirmatory_10Seed"
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
MAIN_MODELS = ("DDPG", "TD3", "SAC", "SACSI-POMDP")
FACTORIAL_VARIANTS = ("SAC Basic", "SAC + Forecast", "SAC + LSTM", "SACSI Full")
TRAINING_MODELS = ("DDPG", "TD3", "SAC Basic", "SAC + Forecast", "SAC + LSTM", "SACSI Full")
EPISODES = 20
EPISODE_LENGTH = 336
VALIDATION_INTERVAL = 5
BOOTSTRAP_RESAMPLES = 20_000
PRIMARY_ENDPOINT = "time_in_target_pct"


def slug(model: str) -> str:
    return model.lower().replace(" + ", "_").replace("-pomdp", "").replace(" ", "_")


def checkpoint_path(model: str, seed: int) -> Path:
    if model == "DDPG":
        return ROOT / "Checkpoints" / "DDPG" / f"ddpg_seed{seed}_best.pt"
    if model == "TD3":
        return ROOT / "Checkpoints" / "TD3" / f"td3_seed{seed}_best.pt"
    if model in {"SAC", "SAC Basic"}:
        return ROOT / "Checkpoints" / "Fair_DRL" / "SAC" / f"sac_seed{seed}_best.pt"
    folder = slug("SACSI Full" if model == "SACSI-POMDP" else model)
    return ROOT / "Checkpoints" / "POMDP_Ablation" / folder / f"{folder}_seed{seed}_best.pt"


def run_artifacts(model: str, seed: int) -> tuple[Path, ...]:
    if model in {"DDPG", "TD3"}:
        lower = model.lower()
        return (
            ROOT / "Results" / model / f"{lower}_seed{seed}_metadata.json",
            ROOT / "Logs" / model / f"{lower}_seed{seed}_training.csv",
            ROOT / "Logs" / model / f"{lower}_seed{seed}_selection.csv",
            ROOT / "Logs" / model / f"{lower}_seed{seed}_validation.csv",
        )
    if model in {"SAC", "SAC Basic"}:
        return (
            ROOT / "Results" / "Fair_DRL" / f"sac_seed{seed}_metadata.json",
            ROOT / "Logs" / "Fair_DRL" / f"sac_seed{seed}_training.csv",
            ROOT / "Logs" / "Fair_DRL" / f"sac_seed{seed}_selection.csv",
            ROOT / "Logs" / "Fair_DRL" / f"sac_seed{seed}_validation.csv",
        )
    name = slug("SACSI Full" if model == "SACSI-POMDP" else model)
    return (
        ROOT / "Results" / "POMDP_Ablation" / f"{name}_seed{seed}_metadata.json",
        ROOT / "Logs" / "POMDP_Ablation" / f"{name}_seed{seed}_training.csv",
        ROOT / "Logs" / "POMDP_Ablation" / f"{name}_seed{seed}_selection.csv",
        ROOT / "Logs" / "POMDP_Ablation" / f"{name}_seed{seed}_validation.csv",
    )


def checkpoint_metadata(model: str, seed: int) -> dict | None:
    checkpoint = checkpoint_path(model, seed)
    if not checkpoint.is_file() or not all(path.is_file() for path in run_artifacts(model, seed)):
        return None
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    expected_model = "SAC" if model in {"SAC", "SAC Basic"} else (
        "SACSI Full" if model == "SACSI-POMDP" else model
    )
    expected = {
        "model": expected_model,
        "seed": seed,
        "reward_version": "reward_v4",
        "episodes": EPISODES,
        "episode_length": EPISODE_LENGTH,
        "environment_interactions": EPISODES * EPISODE_LENGTH,
        "checkpoint_selection": "validation_2024_gate_target_water_rmse",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    if not metadata.get("losses_finite") or metadata.get(
        "benchmark_2025_accessed_for_training_or_selection", False
    ):
        return None
    return metadata


def common_support_hash() -> str:
    locked = {
        "virtual_garden_version": "field_capacity_0.35",
        "reward_version": "reward_v4",
        "training_period": "2021-2023",
        "validation_period": "2024",
        "benchmark_period": "2025",
        "action_min_mm_h": 0.0,
        "action_max_mm_h": 5.0,
        "new_interactions_per_seed": EPISODES * EPISODE_LENGTH,
        "checkpoint_selection": "validation_2024_gate_target_water_rmse",
        "metric_engine": "evaluation.compute_metrics",
    }
    payload = json.dumps(locked, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def train_missing() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Module 8H requires CUDA because GPU training was requested")
    training = pd.read_csv(DATA / "train_2021_2023.csv", parse_dates=["timestamp"])
    validation = pd.read_csv(DATA / "validation_2024.csv", parse_dates=["timestamp"])
    forecast = pd.read_csv(DATA / "synthetic_forecast_sf20.csv", parse_dates=["timestamp"])
    training_sf20 = merge_forecast(training, forecast)
    validation_sf20 = merge_forecast(validation, forecast)

    trainers = {
        "DDPG": lambda seed: train_ddpg_seed(
            seed, EPISODES, EPISODE_LENGTH, VALIDATION_INTERVAL, False
        ),
        "TD3": lambda seed: train_td3_seed(
            seed, EPISODES, EPISODE_LENGTH, VALIDATION_INTERVAL, False
        ),
        "SAC Basic": lambda seed: train_sac_seed(
            seed, training, validation, EPISODES, EPISODE_LENGTH, VALIDATION_INTERVAL, False
        ),
        "SAC + Forecast": lambda seed: train_forecast_variant(
            seed, training_sf20, validation_sf20,
            EPISODES, EPISODE_LENGTH, VALIDATION_INTERVAL, False,
        ),
        "SAC + LSTM": lambda seed: train_lstm_variant(
            seed, training, validation,
            EPISODES, EPISODE_LENGTH, VALIDATION_INTERVAL, False,
        ),
        "SACSI Full": lambda seed: train_full_variant(
            seed, training_sf20, validation_sf20,
            EPISODES, EPISODE_LENGTH, VALIDATION_INTERVAL, False,
        ),
    }
    for model in TRAINING_MODELS:
        for seed in SEEDS:
            metadata = checkpoint_metadata(model, seed)
            if metadata:
                print(f"SKIP {model} seed={seed} completed checkpoint", flush=True)
                continue
            print(f"TRAIN {model} seed={seed} on CUDA", flush=True)
            trainers[model](seed)
            metadata = checkpoint_metadata(model, seed)
            if metadata is None:
                raise RuntimeError(f"Invalid checkpoint after training: {model} seed={seed}")
            print(
                f"DONE {model} seed={seed} selected={metadata['selected_episode']} "
                f"gate={metadata['validation_gate']}",
                flush=True,
            )


def build_registry(models: tuple[str, ...], factorial: bool) -> pd.DataFrame:
    rows = []
    for model in models:
        source_model = "SACSI Full" if model == "SACSI-POMDP" else model
        for seed in SEEDS:
            metadata = checkpoint_metadata(source_model, seed)
            if metadata is None:
                raise RuntimeError(f"Missing locked checkpoint: {model} seed={seed}")
            checkpoint = checkpoint_path(source_model, seed)
            base_interactions = int(metadata.get("base_anchor_interactions", 0))
            rows.append({
                ("variant" if factorial else "model"): model,
                "seed": seed,
                "forecast": source_model in {"SAC + Forecast", "SACSI Full"},
                "memory": source_model in {"SAC + LSTM", "SACSI Full"},
                "reward_version": metadata["reward_version"],
                "training_device": metadata["device"],
                "selected_episode": metadata["selected_episode"],
                "validation_gate": metadata["validation_gate"],
                "losses_finite": metadata["losses_finite"],
                "new_interactions": metadata["environment_interactions"],
                "base_anchor_interactions": base_interactions,
                "effective_total_interactions": metadata["environment_interactions"] + base_interactions,
                "checkpoint_selection": metadata["checkpoint_selection"],
                "benchmark_used_for_selection": metadata.get(
                    "benchmark_2025_accessed_for_training_or_selection", False
                ),
                "history_residual_norm": metadata.get("history_residual_norm"),
                "forecast_residual_norm": metadata.get("forecast_residual_norm"),
                "zero_history_action_delta_mm": metadata.get("zero_history_action_delta_mm"),
                "zero_forecast_action_delta_mm": metadata.get("zero_forecast_action_delta_mm"),
                "common_support_hash": common_support_hash(),
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
            })
    frame = pd.DataFrame(rows)
    label = "variant" if factorial else "model"
    if len(frame) != 40 or frame.duplicated([label, "seed"]).any():
        raise RuntimeError(f"Incomplete 40-row {'factorial' if factorial else 'main'} registry")
    if set(frame["seed"]) != set(SEEDS) or set(frame["reward_version"]) != {"reward_v4"}:
        raise RuntimeError("Registry seed/reward lock failed")
    if not frame["losses_finite"].all() or frame["benchmark_used_for_selection"].any():
        raise RuntimeError("Registry loss/leakage audit failed")
    return frame


def benchmark_forecast(benchmark: pd.DataFrame) -> pd.DataFrame:
    all_weather = pd.read_csv(DATA / "data_clean.csv", parse_dates=["timestamp"])
    forecast = build_synthetic_forecast(all_weather, 0.20)
    forecast = forecast.loc[forecast["timestamp"].dt.year == 2025].reset_index(drop=True)
    validate_common_support(benchmark, forecast)
    return merge_forecast(benchmark, forecast)


def evaluate_tables(
    main_registry: pd.DataFrame,
    factorial_registry: pd.DataFrame,
    benchmark: pd.DataFrame,
    sf20: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factorial = evaluate_factorial(factorial_registry, benchmark, sf20)
    factorial["evaluation_split"] = "retrospective_final_benchmark_2025"
    factorial["result_status"] = "CONFIRMATORY"
    registry_index = factorial_registry.set_index(["variant", "seed"])
    for column in (
        "reward_version", "training_device", "new_interactions",
        "base_anchor_interactions", "effective_total_interactions",
    ):
        factorial[column] = [
            registry_index.loc[(row.variant, row.seed), column]
            for row in factorial.itertuples(index=False)
        ]

    factorial_full = factorial.loc[factorial["variant"] == "SACSI Full"].set_index("seed")
    rows = []
    specs = {
        "DDPG": (DDPGAgent, evaluate_ddpg),
        "TD3": (TD3Agent, evaluate_td3),
        "SAC": (SACAgent, evaluate_sac),
    }
    for registry_row in main_registry.itertuples(index=False):
        if registry_row.model == "SACSI-POMDP":
            metrics = factorial_full.loc[registry_row.seed]
        else:
            agent_class, evaluator = specs[registry_row.model]
            agent, _ = agent_class.load(ROOT / registry_row.checkpoint, device="cpu")
            env = SACIrrigationEnv(
                benchmark, DATA / "normalizer.json", len(benchmark), registry_row.seed
            )
            _, metrics = evaluator(agent, env)
        rows.append({
            "model": registry_row.model,
            "seed": registry_row.seed,
            "evaluation_split": "retrospective_final_benchmark_2025",
            "result_status": "CONFIRMATORY",
            "reward_version": registry_row.reward_version,
            "training_device": registry_row.training_device,
            "inference_device": "cpu",
            "checkpoint_validation_gate": registry_row.validation_gate,
            "new_interactions": registry_row.new_interactions,
            "base_anchor_interactions": registry_row.base_anchor_interactions,
            "effective_total_interactions": registry_row.effective_total_interactions,
            "checkpoint": registry_row.checkpoint,
            "checkpoint_sha256": registry_row.checkpoint_sha256,
            **{metric: float(metrics[metric]) for metric in FORMAL_METRICS},
            "cumulative_reward": float(metrics["cumulative_reward"]),
        })
    main = pd.DataFrame(rows)
    validate_result_tables(main, factorial)
    return main, factorial


def validate_result_tables(main: pd.DataFrame, factorial: pd.DataFrame) -> None:
    for frame, column, expected in (
        (main, "model", MAIN_MODELS),
        (factorial, "variant", FACTORIAL_VARIANTS),
    ):
        if len(frame) != 40 or frame.duplicated([column, "seed"]).any():
            raise RuntimeError(f"Incomplete or duplicate 40-row {column} table")
        if set(frame[column]) != set(expected) or set(frame["seed"]) != set(SEEDS):
            raise RuntimeError(f"Unmatched {column}-seed design")
        if frame[PRIMARY_ENDPOINT].isna().any():
            raise RuntimeError("Missing primary endpoint")
        if not np.isfinite(frame[list(FORMAL_METRICS)].to_numpy()).all():
            raise RuntimeError("Non-finite confirmatory metrics")
        if frame["max_abs_mass_balance_error_mm"].max() > 1e-8:
            raise RuntimeError("Mass-balance gate failed")


def inferential_outputs(main: pd.DataFrame, factorial: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    main_wide = main.pivot(index="seed", columns="model", values=PRIMARY_ENDPOINT)
    main_wide = main_wide.loc[list(SEEDS), list(MAIN_MODELS)]
    friedman = pd.DataFrame([{
        "endpoint": "Time in Target (%)",
        **repeated_measures_omnibus(main_wide.to_numpy()),
    }])
    main_pairs = paired_comparisons(
        main, PRIMARY_ENDPOINT, "SACSI-POMDP", ("SAC", "TD3", "DDPG"),
        method_column="model", resamples=BOOTSTRAP_RESAMPLES, seed=8100,
    )
    main_pairs.insert(0, "analysis_family", "main_benchmark")

    factorial_for_stats = factorial.rename(columns={"variant": "method"})
    family_pairs = paired_comparisons(
        factorial_for_stats, PRIMARY_ENDPOINT, "SACSI Full",
        ("SAC Basic", "SAC + Forecast", "SAC + LSTM"),
        resamples=BOOTSTRAP_RESAMPLES, seed=8200,
    )
    family_pairs.insert(0, "analysis_family", "factorial_pairwise")
    planned = pd.concat((main_pairs, family_pairs), ignore_index=True)
    planned["primary_test"] = "exact paired sign-flip"
    planned["primary_p_raw"] = planned["exact_sign_flip_p"]
    planned["primary_p_holm"] = planned["exact_sign_flip_p_holm"]
    planned["primary_significant_0_05"] = planned["exact_significant_holm_0_05"]

    contrasts = factorial_contrasts(factorial_for_stats, PRIMARY_ENDPOINT)
    labels = {
        "forecast_main_effect": "Forecast main effect",
        "memory_main_effect": "Memory main effect",
        "forecast_x_memory_interaction": "Forecast x Memory interaction",
    }
    factorial_inference = pd.DataFrame([
        one_df_repeated_effect(
            contrasts[column], labels[column], "percentage points",
            BOOTSTRAP_RESAMPLES, seed=8300 + index,
        )
        for index, column in enumerate(contrasts.columns[1:])
    ])
    factorial_inference["exact_sign_flip_p_holm"] = holm_adjust(
        factorial_inference["exact_sign_flip_p"]
    )
    factorial_inference["exact_significant_holm_0_05"] = (
        factorial_inference["exact_sign_flip_p_holm"] < 0.05
    )

    holm_rows = planned[[
        "analysis_family", "comparison", "primary_p_raw", "primary_p_holm",
        "primary_significant_0_05",
    ]].copy()
    effect_holm = factorial_inference[[
        "effect", "exact_sign_flip_p", "exact_sign_flip_p_holm",
        "exact_significant_holm_0_05",
    ]].rename(columns={
        "effect": "comparison",
        "exact_sign_flip_p": "primary_p_raw",
        "exact_sign_flip_p_holm": "primary_p_holm",
        "exact_significant_holm_0_05": "primary_significant_0_05",
    })
    effect_holm.insert(0, "analysis_family", "factorial_effect")
    holm = pd.concat((holm_rows, effect_holm), ignore_index=True)

    bootstrap = planned[[
        "analysis_family", "comparison", "mean_difference_pp",
        "bootstrap_ci95_low_pp", "bootstrap_ci95_high_pp",
    ]].rename(columns={
        "mean_difference_pp": "mean_effect_pp",
        "bootstrap_ci95_low_pp": "ci95_low_pp",
        "bootstrap_ci95_high_pp": "ci95_high_pp",
    })
    effect_bootstrap = factorial_inference[[
        "effect", "mean_effect", "bootstrap_ci95_low", "bootstrap_ci95_high",
    ]].rename(columns={
        "effect": "comparison", "mean_effect": "mean_effect_pp",
        "bootstrap_ci95_low": "ci95_low_pp",
        "bootstrap_ci95_high": "ci95_high_pp",
    })
    effect_bootstrap.insert(0, "analysis_family", "factorial_effect")
    bootstrap = pd.concat((bootstrap, effect_bootstrap), ignore_index=True)
    bootstrap["resamples"] = BOOTSTRAP_RESAMPLES

    main_claim = main_pairs[
        ["mean_difference_pp", "bootstrap_ci95_low_pp", "exact_significant_holm_0_05"]
    ]
    summary = {
        "module": "8H",
        "status": "COMPLETE_CONFIRMATORY" ,
        "primary_endpoint": "Time in Target (%)",
        "matched_seeds": list(SEEDS),
        "friedman_significant_0_05": bool(friedman.loc[0, "friedman_p"] < 0.05),
        "main_locked_pipeline_superiority_supported": bool(
            (main_claim["mean_difference_pp"] > 0).all()
            and (main_claim["bootstrap_ci95_low_pp"] > 0).all()
            and main_claim["exact_significant_holm_0_05"].all()
        ),
        "released_main_pipeline_contrasts": main_pairs.loc[
            (main_pairs["mean_difference_pp"] > 0)
            & main_pairs["exact_significant_holm_0_05"], "comparison"
        ].tolist(),
        "released_factorial_pipeline_effects": factorial_inference.loc[
            (factorial_inference["mean_effect"] > 0)
            & factorial_inference["exact_significant_holm_0_05"], "effect"
        ].tolist(),
        "unqualified_equal_total_budget_superiority_claim_released": False,
        "claim_scope": "locked warm-start training pipelines only",
        "failed_validation_seeds_retained": {
            "main_rows": int((~main["checkpoint_validation_gate"].astype(bool)).sum()),
            "factorial_rows": int((~factorial["checkpoint_validation_gate"].astype(bool)).sum()),
        },
        "inference_rule": (
            "Friedman is the omnibus test. Planned comparisons use exact paired sign-flip "
            "tests with Holm correction; Cohen's dz, 20,000-resample percentile bootstrap "
            "CIs, paired t tests, and repeated-measures ANOVA are reported sensitivities."
        ),
        "protocol_limitation": (
            "Context variants use 6,720-interaction SAC anchors plus 6,720 adaptation "
            "interactions; non-context algorithms use 6,720 total interactions. Statistical "
            "results confirm the locked training pipelines, not a pure equal-total-budget "
            "architecture effect."
        ),
        "negative_and_null_results_retained": True,
    }
    return {
        "friedman_results.csv": friedman,
        "planned_contrasts.csv": planned,
        "holm_adjusted_results.csv": holm,
        "bootstrap_ci.csv": bootstrap,
        "factorial_inference.csv": factorial_inference,
        "summary": summary,
    }


def write_outputs(
    main_registry: pd.DataFrame,
    factorial_registry: pd.DataFrame,
    main: pd.DataFrame,
    factorial: pd.DataFrame,
    inference: dict[str, pd.DataFrame | dict],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    main_registry.to_csv(RESULTS / "main_checkpoint_registry.csv", index=False)
    factorial_registry.to_csv(RESULTS / "sac_family_checkpoint_registry.csv", index=False)
    main.to_csv(RESULTS / "main_10seed_results_2025.csv", index=False)
    factorial.to_csv(RESULTS / "sac_family_10seed_factorial.csv", index=False)
    for name in (
        "friedman_results.csv", "planned_contrasts.csv", "holm_adjusted_results.csv",
        "bootstrap_ci.csv", "factorial_inference.csv",
    ):
        inference[name].to_csv(RESULTS / name, index=False)
    main.groupby("model")[list(FORMAL_METRICS)].agg(["mean", "std"]).to_csv(
        RESULTS / "main_10seed_summary.csv"
    )
    factorial.groupby("variant")[list(FORMAL_METRICS)].agg(["mean", "std"]).to_csv(
        RESULTS / "sac_family_10seed_summary.csv"
    )
    summary = inference["summary"] | {
        "benchmark_wording": "retrospective final benchmark 2025",
        "reward_version": "reward_v4",
        "training_device_required": "cuda",
        "checkpoint_selection": "validation 2024 only",
        "benchmark_2025_loaded_after_all_checkpoint_registries_locked": True,
        "no_retraining_or_reselection_after_2025_opening": True,
        "main_rows": len(main),
        "factorial_rows": len(factorial),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / "final_statistics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    artifacts = [
        "main_checkpoint_registry.csv", "sac_family_checkpoint_registry.csv",
        "main_10seed_results_2025.csv", "sac_family_10seed_factorial.csv",
        "friedman_results.csv", "planned_contrasts.csv", "holm_adjusted_results.csv",
        "bootstrap_ci.csv", "factorial_inference.csv", "final_statistics_summary.json",
    ]
    manifest = {
        "module": "8H",
        "status": "COMPLETE_CONFIRMATORY",
        "artifacts": {
            name: {"sha256": sha256_file(RESULTS / name), "size_bytes": (RESULTS / name).stat().st_size}
            for name in artifacts
        },
    }
    (RESULTS / "confirmatory_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def smoke_test() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested smoke training")
    training = pd.read_csv(DATA / "train_2021_2023.csv", parse_dates=["timestamp"]).head(96)
    validation = pd.read_csv(DATA / "validation_2024.csv", parse_dates=["timestamp"]).head(48)
    forecast = pd.read_csv(DATA / "synthetic_forecast_sf20.csv", parse_dates=["timestamp"])
    training_sf20, validation_sf20 = merge_forecast(training, forecast), merge_forecast(validation, forecast)
    train_ddpg_seed(11, 12, 48, 6, True)
    train_td3_seed(11, 12, 48, 6, True)
    train_sac_seed(11, training, validation, 12, 48, 6, True)
    train_forecast_variant(11, training_sf20, validation_sf20, 12, 48, 6, True)
    train_lstm_variant(11, training, validation, 12, 48, 6, True)
    train_full_variant(11, training_sf20, validation_sf20, 12, 48, 6, True)
    print("8H SMOKE PASS: six training paths completed on CUDA without writing checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--statistics-only", action="store_true")
    args = parser.parse_args()
    if sum((args.train_only, args.evaluate_only, args.statistics_only)) > 1:
        parser.error("--train-only, --evaluate-only, and --statistics-only are mutually exclusive")
    if args.smoke:
        smoke_test()
        return
    if args.statistics_only:
        main_registry = pd.read_csv(RESULTS / "main_checkpoint_registry.csv")
        factorial_registry = pd.read_csv(RESULTS / "sac_family_checkpoint_registry.csv")
        main = pd.read_csv(RESULTS / "main_10seed_results_2025.csv")
        factorial = pd.read_csv(RESULTS / "sac_family_10seed_factorial.csv")
        validate_result_tables(main, factorial)
        inference = inferential_outputs(main, factorial)
        write_outputs(main_registry, factorial_registry, main, factorial, inference)
        print(json.dumps(inference["summary"], indent=2), flush=True)
        return
    if not args.evaluate_only:
        train_missing()
    if args.train_only:
        return

    main_registry = build_registry(MAIN_MODELS, factorial=False)
    factorial_registry = build_registry(FACTORIAL_VARIANTS, factorial=True)
    benchmark = pd.read_csv(DATA / "benchmark_2025.csv", parse_dates=["timestamp"])
    validate_common_support(benchmark)
    main, factorial = evaluate_tables(
        main_registry, factorial_registry, benchmark, benchmark_forecast(benchmark)
    )
    inference = inferential_outputs(main, factorial)
    write_outputs(main_registry, factorial_registry, main, factorial, inference)
    print(main.groupby("model")[PRIMARY_ENDPOINT].agg(["mean", "std"]).to_string(), flush=True)
    print(json.dumps(inference["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
