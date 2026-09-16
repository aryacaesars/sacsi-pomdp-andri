"""Run the locked nine-method retrospective benchmark on common-support 2025 data."""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers import (  # noqa: E402
    FixedScheduleController,
    FuzzyController,
    NoIrrigationController,
    RuleBasedForecastController,
    ThresholdController,
)
from evaluation import compute_metrics, run_controller  # noqa: E402
from evaluation.final_benchmark import (  # noqa: E402
    FORMAL_METRICS,
    METHODS,
    validate_common_support,
    validate_log_support,
    validate_registry,
    summarize_runs,
)
from sac_basic import SACAgent, SACConfig, SACIrrigationEnv  # noqa: E402
from sac_basic.training import evaluate as evaluate_basic  # noqa: E402
from sac_forecast import SACForecastEnv  # noqa: E402
from sac_lstm import RecurrentSACAgent, RecurrentSACConfig, SACLSTMEnv  # noqa: E402
from sac_lstm.training import evaluate as evaluate_lstm  # noqa: E402
from sacsi_full import SACSIConfig, SACSIEnv, SACSIRecurrentAgent  # noqa: E402
from sacsi_full.training import evaluate as evaluate_sacsi  # noqa: E402


DATA = ROOT / "00_Dataset" / "Processed"
RESULTS = ROOT / "Results" / "Final_Experiment"
LOGS = ROOT / "Logs" / "Final_Experiment"
TABLES = ROOT / "Tables"


def load_sf20(weather: pd.DataFrame) -> pd.DataFrame:
    forecast = pd.read_csv(DATA / "synthetic_forecast_sf20.csv", parse_dates=["timestamp"])
    forecast = forecast.loc[forecast["timestamp"].dt.year == 2025].reset_index(drop=True)
    validate_common_support(weather, forecast)
    merged = weather.merge(forecast, on="timestamp", how="left", validate="one_to_one")
    if merged.filter(like="forecast_").isna().any().any():
        raise ValueError("Missing SF-20 context on benchmark support")
    return merged


def load_baseline_forecast() -> pd.DataFrame:
    forecast = pd.read_csv(DATA / "forecast_clean.csv", parse_dates=["timestamp"])
    forecast["timestamp"] -= pd.Timedelta(hours=1)
    return forecast.rename(columns={
        "forecast_precipitation_mm": "forecast_precipitation_h1_mm",
    })[["timestamp", "forecast_precipitation_h1_mm"]]


def slug(value: str) -> str:
    return value.lower().replace(" + ", "_plus_").replace("-", "_").replace(" ", "_")


def load_agent(agent_class, config_class, checkpoint_path: Path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = agent_class(config_class(**state["config"]), device="cpu")
    agent.actor.load_state_dict(state["actor"])
    agent.actor.eval()
    return agent, state.get("metadata", {})


def metric_row(method: str, method_type: str, metrics: dict, **extra) -> dict:
    return {
        "benchmark_period": "2025",
        "method": method,
        "method_type": method_type,
        **extra,
        **{name: metrics[name] for name in FORMAL_METRICS},
    }


def run_baselines(weather: pd.DataFrame, timestamps) -> list[dict]:
    forecast = load_baseline_forecast()
    controllers = (
        NoIrrigationController(),
        FixedScheduleController(),
        ThresholdController(),
        RuleBasedForecastController(),
        FuzzyController(),
    )
    rows = []
    for controller in controllers:
        print(f"RUN baseline {controller.name}", flush=True)
        log = run_controller(weather, controller, forecast)
        validate_log_support(log, timestamps)
        log.to_csv(LOGS / f"benchmark_2025_{slug(controller.name)}.csv", index=False)
        rows.append(metric_row(
            controller.name,
            "baseline",
            compute_metrics(log),
            seed=pd.NA,
            validation_gate=pd.NA,
            checkpoint="",
            checkpoint_sha256="",
            training_device="",
            inference_device="cpu",
            forecast_protocol="historical_continuous_h1_proxy",
        ))
    return rows


def run_rl_family(
    method: str,
    registry_path: Path,
    data: pd.DataFrame,
    agent_class,
    config_class,
    env_class,
    evaluator,
    forecast_protocol: str,
    timestamps,
) -> tuple[list[dict], list[dict]]:
    registry = validate_registry(pd.read_csv(registry_path), ROOT)
    metric_rows, audit_rows = [], []
    for row in registry.itertuples(index=False):
        seed = int(row.seed)
        checkpoint_path = ROOT / Path(row.checkpoint)
        print(f"RUN {method} seed={seed}", flush=True)
        agent, metadata = load_agent(agent_class, config_class, checkpoint_path)
        env = env_class(data, DATA / "normalizer.json", len(data), seed)
        log, metrics = evaluator(agent, env)
        log["controller"] = method
        log["seed"] = seed
        validate_log_support(log, timestamps)
        log.to_csv(LOGS / f"benchmark_2025_{slug(method)}_seed{seed}.csv", index=False)
        metric_rows.append(metric_row(
            method,
            "rl",
            metrics,
            seed=seed,
            validation_gate=str(row.validation_gate).lower() == "true",
            checkpoint=row.checkpoint,
            checkpoint_sha256=row.checkpoint_sha256,
            training_device=row.device,
            inference_device="cpu",
            forecast_protocol=forecast_protocol,
        ))
        audit_rows.append({
            **row._asdict(),
            "benchmark_period": "2025",
            **{f"benchmark_2025_{name}": metrics[name] for name in FORMAL_METRICS},
            "checkpoint_training_period": metadata.get("training_period", ""),
            "checkpoint_validation_period": metadata.get("validation_period", ""),
        })
        del agent, env
        gc.collect()
    return metric_rows, audit_rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    weather = pd.read_csv(DATA / "benchmark_2025.csv", parse_dates=["timestamp"])
    timestamps = validate_common_support(weather)
    sf20 = load_sf20(weather)

    rows = run_baselines(weather, timestamps)
    audit_rows = []
    families = (
        (
            "SAC Basic", ROOT / "Results" / "SAC_Basic" / "validation_registry.csv",
            weather, SACAgent, SACConfig, SACIrrigationEnv, evaluate_basic, "none",
        ),
        (
            "SAC + Forecast", ROOT / "Results" / "SAC_Forecast" / "validation_registry.csv",
            sf20, SACAgent, SACConfig, SACForecastEnv, evaluate_basic, "SF-20_h1_controlled_proxy",
        ),
        (
            "SAC + LSTM", ROOT / "Results" / "SAC_LSTM" / "validation_registry.csv",
            weather, RecurrentSACAgent, RecurrentSACConfig, SACLSTMEnv, evaluate_lstm, "none",
        ),
        (
            "SACSI Full", ROOT / "Results" / "SACSI_Full" / "validation_registry.csv",
            sf20, SACSIRecurrentAgent, SACSIConfig, SACSIEnv, evaluate_sacsi,
            "SF-20_h1_controlled_proxy",
        ),
    )
    for family in families:
        family_rows, family_audit = run_rl_family(*family, timestamps=timestamps)
        rows.extend(family_rows)
        audit_rows.extend(family_audit)

    runs = pd.DataFrame(rows)
    expected_counts = {method: (10 if method.startswith("SAC") else 1) for method in METHODS}
    if runs.groupby("method").size().to_dict() != expected_counts:
        raise RuntimeError("Final benchmark must contain 5 baseline and 40 RL runs")
    if not np.isfinite(runs[list(FORMAL_METRICS)].to_numpy(dtype=float)).all():
        raise RuntimeError("Final benchmark contains non-finite formal metrics")
    if runs["max_abs_mass_balance_error_mm"].max() > 1e-8:
        raise RuntimeError("Mass-balance acceptance gate failed")
    summary = summarize_runs(runs)
    runs.to_csv(RESULTS / "benchmark_2025_runs.csv", index=False)
    summary.to_csv(RESULTS / "benchmark_2025_summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(RESULTS / "rl_checkpoint_registry_2025.csv", index=False)
    summary.to_csv(TABLES / "final_benchmark_2025.csv", index=False)
    manifest = {
        "benchmark_type": "retrospective final benchmark",
        "period": "2025-01-01 00:00:00/2025-12-31 23:00:00",
        "hours": len(weather),
        "methods": list(METHODS),
        "locked_rl_seeds": [11, 22, 33, 44, 55, 66, 77, 88, 99, 110],
        "run_count": len(runs),
        "ranking_rule": "time_in_target_pct descending, then total_irrigation_mm ascending; exact ties share rank",
        "checkpoint_selection": "validation 2024 only; no 2025 tuning or reselection",
        "forecast_protocols": ["historical_continuous_h1_proxy", "SF-20_h1_controlled_proxy"],
        "inference_device": "cpu (deterministic sequential inference)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / "benchmark_2025_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary[[
        "rank", "method", "n_runs", "time_in_target_pct_mean", "time_in_target_pct_std",
        "total_irrigation_mm_mean", "total_irrigation_mm_std",
    ]].to_string(index=False))
    print("FINAL BENCHMARK FINISHED", flush=True)


if __name__ == "__main__":
    main()
