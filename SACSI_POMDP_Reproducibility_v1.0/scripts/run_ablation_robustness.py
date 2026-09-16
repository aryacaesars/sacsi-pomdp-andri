"""Run Sprint 11 factorial, context-ablation, and robustness experiments."""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.ablation_robustness import (  # noqa: E402
    CONTEXT_CONDITIONS,
    evaluate_context,
    factorial_interactions,
)
from evaluation.final_benchmark import FORMAL_METRICS, validate_common_support, validate_registry  # noqa: E402
from sacsi_full import SACSIConfig, SACSIEnv, SACSIRecurrentAgent  # noqa: E402
from scripts.prepare_synthetic_forecast import build_synthetic_forecast  # noqa: E402


DATA = ROOT / "00_Dataset" / "Processed"
RESULTS = ROOT / "Results" / "Ablation_Robustness"
TABLES = ROOT / "Tables"
FORECAST_LEVELS = (10, 20, 30)
SEQUENCE_LENGTHS = (6, 12, 24, 48)
REFERENCE_CONDITIONS = {
    "context_ablation": "Full",
    "forecast_robustness": "SF20",
    "sequence_sensitivity": "k24",
}


def merge_forecast(weather: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    forecast = forecast.loc[forecast["timestamp"].dt.year == 2025].reset_index(drop=True)
    validate_common_support(weather, forecast)
    merged = weather.merge(forecast, on="timestamp", how="left", validate="one_to_one")
    if merged.filter(like="forecast_").isna().any().any():
        raise ValueError("Missing forecast context after merge")
    return merged


def load_agent(checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = SACSIRecurrentAgent(SACSIConfig(**state["config"]), device="cpu")
    agent.actor.load_state_dict(state["actor"])
    agent.actor.eval()
    return agent


def metric_row(experiment: str, condition: str, seed: int, metrics: dict) -> dict:
    return {
        "experiment": experiment,
        "condition": condition,
        "seed": seed,
        **{name: metrics[name] for name in FORMAL_METRICS},
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (experiment, condition), group in rows.groupby(["experiment", "condition"], sort=False):
        row = {"experiment": experiment, "condition": condition, "n_seeds": len(group)}
        for metric in FORMAL_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        output.append(row)
    return pd.DataFrame(output)


def reference_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "time_in_target_pct_mean", "total_irrigation_mm_mean",
        "violation_rate_pct_mean", "rmse_band_mean",
    )
    output = summary[["experiment", "condition", "n_seeds", *columns]].copy()
    for experiment, reference in REFERENCE_CONDITIONS.items():
        mask = output["experiment"] == experiment
        reference_row = output.loc[mask & (output["condition"] == reference)].iloc[0]
        output.loc[mask, "reference_condition"] = reference
        for column in columns:
            output.loc[mask, f"{column}_delta"] = output.loc[mask, column] - reference_row[column]
    return output


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    weather = pd.read_csv(DATA / "benchmark_2025.csv", parse_dates=["timestamp"])
    validate_common_support(weather)
    all_weather = pd.read_csv(DATA / "data_clean.csv", parse_dates=["timestamp"])
    forecasts = {
        level: merge_forecast(weather, build_synthetic_forecast(all_weather, level / 100))
        for level in FORECAST_LEVELS
    }
    stored_sf20 = pd.read_csv(DATA / "synthetic_forecast_sf20.csv", parse_dates=["timestamp"])
    stored_sf20 = stored_sf20.loc[stored_sf20["timestamp"].dt.year == 2025].reset_index(drop=True)
    generated_columns = [
        "forecast_precipitation_mm", "forecast_et0_mm", "forecast_temperature_c",
    ]
    if not np.allclose(
        forecasts[20][generated_columns].to_numpy(),
        stored_sf20[generated_columns].to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("Generated SF20 does not reproduce the locked training proxy")

    registry = validate_registry(
        pd.read_csv(ROOT / "Results" / "SACSI_Full" / "validation_registry.csv"), ROOT
    )
    rows = []
    for registry_row in registry.itertuples(index=False):
        seed = int(registry_row.seed)
        print(f"LOAD SACSI seed={seed}", flush=True)
        agent = load_agent(ROOT / Path(registry_row.checkpoint))
        full_metrics = None
        for condition in CONTEXT_CONDITIONS:
            print(f"RUN context seed={seed} condition={condition}", flush=True)
            env = SACSIEnv(forecasts[20], DATA / "normalizer.json", len(weather), seed, sequence_length=24)
            metrics = evaluate_context(agent, env, condition)
            rows.append(metric_row("context_ablation", condition, seed, metrics))
            if condition == "Full":
                full_metrics = metrics
        for level in FORECAST_LEVELS:
            condition = f"SF{level}"
            if level == 20:
                metrics = full_metrics
            else:
                print(f"RUN forecast seed={seed} condition={condition}", flush=True)
                env = SACSIEnv(
                    forecasts[level], DATA / "normalizer.json", len(weather), seed, sequence_length=24
                )
                metrics = evaluate_context(agent, env)
            rows.append(metric_row("forecast_robustness", condition, seed, metrics))
        for length in SEQUENCE_LENGTHS:
            condition = f"k{length}"
            if length == 24:
                metrics = full_metrics
            else:
                print(f"RUN sequence seed={seed} condition={condition}", flush=True)
                env = SACSIEnv(
                    forecasts[20], DATA / "normalizer.json", len(weather), seed, sequence_length=length
                )
                metrics = evaluate_context(agent, env)
            rows.append(metric_row("sequence_sensitivity", condition, seed, metrics))
        del agent
        gc.collect()

    runs = pd.DataFrame(rows)
    expected = len(registry) * (
        len(CONTEXT_CONDITIONS) + len(FORECAST_LEVELS) + len(SEQUENCE_LENGTHS)
    )
    if len(runs) != expected or runs.groupby(["experiment", "condition"]).size().ne(10).any():
        raise RuntimeError("Ablation/robustness matrix is incomplete")
    if runs["max_abs_mass_balance_error_mm"].max() > 1e-8:
        raise RuntimeError("Mass-balance acceptance gate failed")

    final_runs = pd.read_csv(ROOT / "Results" / "Final_Experiment" / "benchmark_2025_runs.csv")
    factorial = final_runs.loc[final_runs["method_type"] == "rl"].copy()
    interactions = factorial_interactions(final_runs, FORMAL_METRICS)
    interaction_summary = pd.DataFrame({
        "metric": FORMAL_METRICS,
        "interaction_mean": [interactions[f"{m}_interaction"].mean() for m in FORMAL_METRICS],
        "interaction_std": [interactions[f"{m}_interaction"].std(ddof=1) for m in FORMAL_METRICS],
        "beneficial_direction": [
            "positive" if m == "time_in_target_pct"
            else "context-dependent" if m == "mean_soil_moisture"
            else "closer-to-zero" if m == "max_abs_mass_balance_error_mm"
            else "negative"
            for m in FORMAL_METRICS
        ],
    })
    summary = summarize(runs)
    deltas = reference_deltas(summary)
    runs.to_csv(RESULTS / "ablation_robustness_runs.csv", index=False)
    summary.to_csv(RESULTS / "ablation_robustness_summary.csv", index=False)
    deltas.to_csv(RESULTS / "ablation_reference_deltas.csv", index=False)
    factorial.to_csv(RESULTS / "factorial_rl_runs.csv", index=False)
    interactions.to_csv(RESULTS / "factorial_interactions.csv", index=False)
    interaction_summary.to_csv(RESULTS / "factorial_interaction_summary.csv", index=False)
    summary.to_csv(TABLES / "ablation_robustness_2025.csv", index=False)
    manifest = {
        "benchmark_type": "retrospective 2025 controlled ablation and robustness",
        "hours": len(weather),
        "seeds": registry["seed"].astype(int).tolist(),
        "context_conditions": list(CONTEXT_CONDITIONS),
        "forecast_levels_pct": list(FORECAST_LEVELS),
        "sequence_lengths_h": list(SEQUENCE_LENGTHS),
        "reference_conditions": REFERENCE_CONDITIONS,
        "context_intervention_definitions": {
            "No History": "history residual contribution disabled",
            "No Forecast": "forecast residual contribution disabled",
            "No Context": "both residual contributions disabled",
            "Zero History": "history branch active with all-zero sequence input",
            "Zero Forecast": "forecast branch active with all-zero forecast input",
        },
        "checkpoint_selection": "locked validation-2024 checkpoints; no retraining/reselection",
        "sequence_design": "inference-only window sensitivity using the same k24-trained checkpoint",
        "sf_noise_design": "matched standard-noise realization scaled to 10/20/30 percent",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[[
        "experiment", "condition", "time_in_target_pct_mean", "time_in_target_pct_std",
        "total_irrigation_mm_mean",
    ]].to_string(index=False))
    print("ABLATION ROBUSTNESS FINISHED", flush=True)


if __name__ == "__main__":
    main()
