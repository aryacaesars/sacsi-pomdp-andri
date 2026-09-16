"""Run the locked three-seed incremental POMDP ablation and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.ablation_robustness import CONTEXT_CONDITIONS, evaluate_context
from evaluation.final_benchmark import FORMAL_METRICS, sha256_file, validate_common_support
from sac_basic import SACAgent, SACConfig, SACIrrigationEnv
from sac_basic.training import _selection_key, _selection_score
from sac_basic.training import evaluate as evaluate_sac
from sac_basic.training import set_seed, train as train_sac, validation_gate
from sac_forecast import SACForecastEnv
from sac_lstm import RecurrentSACAgent, RecurrentSACConfig, SACLSTMEnv
from sac_lstm.training import evaluate as evaluate_lstm
from sac_lstm.training import memory_diagnostics, train as train_lstm
from sacsi_full import SACSIConfig, SACSIEnv, SACSIRecurrentAgent
from sacsi_full.training import context_diagnostics, evaluate as evaluate_sacsi
from sacsi_full.training import train as train_sacsi
from scripts.prepare_synthetic_forecast import build_synthetic_forecast


DATA = ROOT / "00_Dataset" / "Processed"
RESULTS = ROOT / "Results" / "POMDP_Ablation"
LOGS = ROOT / "Logs" / "POMDP_Ablation"
CHECKPOINTS = ROOT / "Checkpoints" / "POMDP_Ablation"
SEEDS = (11, 22, 33)
VARIANTS = ("SAC Basic", "SAC + Forecast", "SAC + LSTM", "SACSI Full")
FORECAST_LEVELS = (10, 20, 30)
SEQUENCE_LENGTHS = (6, 12, 24, 48)


def merge_forecast(weather: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    merged = weather.merge(forecast, on="timestamp", how="left", validate="one_to_one")
    if merged.filter(like="forecast_").isna().any().any():
        raise ValueError("Missing forecast context after timestamp merge")
    return merged


def load_sf20(weather: pd.DataFrame) -> pd.DataFrame:
    forecast = pd.read_csv(DATA / "synthetic_forecast_sf20.csv", parse_dates=["timestamp"])
    return merge_forecast(weather, forecast)


def base_checkpoint(seed: int) -> Path:
    return ROOT / "Checkpoints" / "Fair_DRL" / "SAC" / f"sac_seed{seed}_best.pt"


def variant_checkpoint(variant: str, seed: int) -> Path:
    if variant == "SAC Basic":
        return base_checkpoint(seed)
    folder = variant.lower().replace(" + ", "_").replace(" ", "_")
    return CHECKPOINTS / folder / f"{folder}_seed{seed}_best.pt"


def _copy_expanded_forecast_network(target, source: dict, actor: bool) -> None:
    target_state = target.state_dict()
    expanded_key = "body.0.weight" if actor else "network.0.weight"
    for name, source_value in source.items():
        if target_state[name].shape == source_value.shape:
            target_state[name].copy_(source_value)
        elif name == expanded_key:
            target_state[name].zero_()
            target_state[name][:, :8].copy_(source_value[:, :8])
            if not actor:
                target_state[name][:, -1].copy_(source_value[:, -1])
        else:
            raise ValueError(f"Unexpected forecast warm-start shape for {name}")
    target.load_state_dict(target_state)


def warm_start_forecast(agent: SACAgent, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location=agent.device, weights_only=False)
    _copy_expanded_forecast_network(agent.actor, state["actor"], actor=True)
    for target, name in (
        (agent.critic1, "critic1"),
        (agent.critic2, "critic2"),
        (agent.target1, "target1"),
        (agent.target2, "target2"),
    ):
        _copy_expanded_forecast_network(target, state[name], actor=False)
    agent.log_alpha.data.copy_(state["log_alpha"].to(agent.device))


def _sac_snapshot(agent: SACAgent) -> dict:
    return {
        name: deepcopy(getattr(agent, name).state_dict())
        for name in ("actor", "critic1", "critic2", "target1", "target2")
    } | {"log_alpha": agent.log_alpha.detach().clone()}


def _sac_restore(agent: SACAgent, state: dict) -> None:
    for name in ("actor", "critic1", "critic2", "target1", "target2"):
        getattr(agent, name).load_state_dict(state[name])
    agent.log_alpha.data.copy_(state["log_alpha"])


def load_context_agent(variant: str, checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if variant == "SAC + LSTM":
        agent = RecurrentSACAgent(RecurrentSACConfig(**state["config"]), device="cpu")
    elif variant == "SACSI Full":
        agent = SACSIRecurrentAgent(SACSIConfig(**state["config"]), device="cpu")
    else:
        raise ValueError(variant)
    for name in ("actor", "critic1", "critic2", "target1", "target2"):
        getattr(agent, name).load_state_dict(state[name])
    agent.log_alpha.data.copy_(state["log_alpha"])
    return agent, state["metadata"]


def save_run(variant: str, seed: int, agent, metadata: dict, history, selection, log) -> None:
    checkpoint = variant_checkpoint(variant, seed)
    agent.save(checkpoint, metadata)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    slug = variant.lower().replace(" + ", "_").replace(" ", "_")
    (RESULTS / f"{slug}_seed{seed}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    history.to_csv(LOGS / f"{slug}_seed{seed}_training.csv", index=False)
    selection.to_csv(LOGS / f"{slug}_seed{seed}_selection.csv", index=False)
    log.to_csv(LOGS / f"{slug}_seed{seed}_validation.csv", index=False)


def completed_run(variant: str, seed: int, episodes: int, episode_length: int):
    checkpoint = variant_checkpoint(variant, seed)
    slug = variant.lower().replace(" + ", "_").replace(" ", "_")
    required = (
        checkpoint,
        RESULTS / f"{slug}_seed{seed}_metadata.json",
        LOGS / f"{slug}_seed{seed}_training.csv",
        LOGS / f"{slug}_seed{seed}_selection.csv",
        LOGS / f"{slug}_seed{seed}_validation.csv",
    )
    if not all(path.is_file() for path in required):
        return None
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    expected = {
        "model": variant,
        "seed": seed,
        "reward_version": "reward_v4",
        "episodes": episodes,
        "episode_length": episode_length,
        "environment_interactions": episodes * episode_length,
        "checkpoint_selection": "validation_2024_gate_target_water_rmse",
        "benchmark_2025_accessed_for_training_or_selection": False,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    return metadata if metadata.get("losses_finite") else None


def common_metadata(
    variant: str,
    seed: int,
    agent,
    env,
    episodes: int,
    episode_length: int,
    selected_episode: int,
    replay,
    history: pd.DataFrame,
    metrics: dict,
) -> dict:
    loss_columns = [name for name in ("actor_loss", "critic_loss", "alpha_loss") if name in history]
    losses = history[loss_columns].dropna().to_numpy()
    return {
        "model": variant,
        "seed": seed,
        "device": str(agent.device),
        "reward_version": env.reward_version,
        "virtual_garden_version": "field_capacity_0.35",
        "training_period": "2021-2023",
        "validation_period": "2024",
        "episodes": episodes,
        "episode_length": episode_length,
        "environment_interactions": episodes * episode_length,
        "base_anchor_interactions": 6720,
        "selected_episode": selected_episode,
        "replay_size": len(replay),
        "checkpoint_selection": "validation_2024_gate_target_water_rmse",
        "warm_start_checkpoint": base_checkpoint(seed).relative_to(ROOT).as_posix(),
        "losses_finite": bool(np.isfinite(losses).all()),
        "validation_gate": validation_gate(metrics),
        "validation_metrics": metrics,
        "benchmark_2025_accessed_for_training_or_selection": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def train_forecast_variant(
    seed: int,
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    episodes: int,
    episode_length: int,
    validation_interval: int,
    smoke: bool,
):
    set_seed(seed)
    agent = SACAgent(SACConfig(observation_dim=11))
    warm_start_forecast(agent, base_checkpoint(seed))
    train_env = SACForecastEnv(training_data, DATA / "normalizer.json", episode_length, seed)
    validation_env = SACForecastEnv(
        validation_data, DATA / "normalizer.json", len(validation_data), seed
    )
    initial_state = _sac_snapshot(agent)
    _, initial_metrics = evaluate_sac(agent, validation_env)
    history, replay, selection, selected_episode = train_sac(
        agent,
        train_env,
        episodes,
        validation_env=validation_env,
        validation_interval=validation_interval,
    )
    _, selected_metrics = evaluate_sac(agent, validation_env)
    initial_row = pd.DataFrame([{
        "episode": 0,
        "validation_gate": validation_gate(initial_metrics),
        "selection_score": _selection_score(initial_metrics),
        **initial_metrics,
    }])
    selection = pd.concat((initial_row, selection), ignore_index=True)
    if _selection_key(initial_metrics) > _selection_key(selected_metrics):
        _sac_restore(agent, initial_state)
        selected_episode = 0
    validation_log, metrics = evaluate_sac(agent, validation_env)
    metadata = common_metadata(
        "SAC + Forecast", seed, agent, train_env, episodes, episode_length,
        selected_episode, replay, history, metrics,
    ) | {
        "forecast_protocol": "SF-20_h1_controlled_proxy",
        "forecast_horizon": 1,
        "forecast_dim": 3,
        "training_strategy": "expanded-input warm-start from fair SAC",
    }
    if smoke:
        print(json.dumps(metadata, indent=2))
    else:
        save_run("SAC + Forecast", seed, agent, metadata, history, selection, validation_log)
    return metadata


def train_lstm_variant(
    seed: int,
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    episodes: int,
    episode_length: int,
    validation_interval: int,
    smoke: bool,
):
    set_seed(seed)
    config = RecurrentSACConfig(sequence_length=24)
    agent = RecurrentSACAgent(config)
    agent.warm_start(base_checkpoint(seed))
    train_env = SACLSTMEnv(
        training_data, DATA / "normalizer.json", episode_length, seed, sequence_length=24
    )
    validation_env = SACLSTMEnv(
        validation_data, DATA / "normalizer.json", len(validation_data), seed, sequence_length=24
    )
    history, replay, selection, selected_episode = train_lstm(
        agent, train_env, episodes, validation_env, validation_interval
    )
    validation_log, metrics = evaluate_lstm(agent, validation_env)
    diagnostics = memory_diagnostics(agent, validation_env)
    metadata = common_metadata(
        "SAC + LSTM", seed, agent, train_env, episodes, episode_length,
        selected_episode, replay, history, metrics,
    ) | {
        "sequence_length": 24,
        "lstm_hidden_dim": config.lstm_hidden_dim,
        "training_strategy": "Residual Recurrent Warm-Start",
        "history_residual_norm": agent.residual_norm(),
        **diagnostics,
    }
    if smoke:
        print(json.dumps(metadata, indent=2))
    else:
        save_run("SAC + LSTM", seed, agent, metadata, history, selection, validation_log)
    return metadata


def train_full_variant(
    seed: int,
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    episodes: int,
    episode_length: int,
    validation_interval: int,
    smoke: bool,
):
    set_seed(seed)
    config = SACSIConfig(sequence_length=24)
    agent = SACSIRecurrentAgent(config)
    agent.warm_start(base_checkpoint(seed))
    train_env = SACSIEnv(
        training_data, DATA / "normalizer.json", episode_length, seed, sequence_length=24
    )
    validation_env = SACSIEnv(
        validation_data, DATA / "normalizer.json", len(validation_data), seed, sequence_length=24
    )
    history, replay, selection, selected_episode = train_sacsi(
        agent, train_env, episodes, validation_env, validation_interval
    )
    validation_log, metrics = evaluate_sacsi(agent, validation_env)
    diagnostics = context_diagnostics(agent, validation_env)
    metadata = common_metadata(
        "SACSI Full", seed, agent, train_env, episodes, episode_length,
        selected_episode, replay, history, metrics,
    ) | {
        "sequence_length": 24,
        "lstm_hidden_dim": config.lstm_hidden_dim,
        "forecast_protocol": "SF-20_h1_controlled_proxy",
        "forecast_horizon": 1,
        "forecast_dim": 3,
        "training_strategy": "Multi-Context Residual Recurrent Warm-Start",
        **agent.context_norms(),
        **diagnostics,
    }
    if smoke:
        print(json.dumps(metadata, indent=2))
    else:
        save_run("SACSI Full", seed, agent, metadata, history, selection, validation_log)
    return metadata


def build_registry() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            checkpoint = variant_checkpoint(variant, seed)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metadata = state["metadata"]
            rows.append({
                "variant": variant,
                "seed": seed,
                "forecast": variant in {"SAC + Forecast", "SACSI Full"},
                "memory": variant in {"SAC + LSTM", "SACSI Full"},
                "reward_version": metadata["reward_version"],
                "selected_episode": metadata["selected_episode"],
                "validation_gate": metadata["validation_gate"],
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
            })
    registry = pd.DataFrame(rows)
    if len(registry) != 12 or set(registry["reward_version"]) != {"reward_v4"}:
        raise RuntimeError("Incomplete or unlocked SAC-family registry")
    return registry


def evaluate_factorial(registry: pd.DataFrame, weather, sf20) -> pd.DataFrame:
    rows = []
    for row in registry.itertuples(index=False):
        checkpoint = ROOT / Path(row.checkpoint)
        if row.variant == "SAC Basic":
            agent, metadata = SACAgent.load(checkpoint, device="cpu")
            env = SACIrrigationEnv(weather, DATA / "normalizer.json", len(weather), row.seed)
            _, metrics = evaluate_sac(agent, env)
        elif row.variant == "SAC + Forecast":
            agent, metadata = SACAgent.load(checkpoint, device="cpu")
            env = SACForecastEnv(sf20, DATA / "normalizer.json", len(weather), row.seed)
            _, metrics = evaluate_sac(agent, env)
        elif row.variant == "SAC + LSTM":
            agent, metadata = load_context_agent(row.variant, checkpoint)
            env = SACLSTMEnv(
                weather, DATA / "normalizer.json", len(weather), row.seed, sequence_length=24
            )
            _, metrics = evaluate_lstm(agent, env)
        else:
            agent, metadata = load_context_agent(row.variant, checkpoint)
            env = SACSIEnv(sf20, DATA / "normalizer.json", len(weather), row.seed, sequence_length=24)
            _, metrics = evaluate_sacsi(agent, env)
        rows.append({
            "variant": row.variant,
            "seed": row.seed,
            "forecast": row.forecast,
            "memory": row.memory,
            "evaluation_split": "retrospective_benchmark_2025",
            "result_status": "EXPLORATORY_REVALIDATION",
            "checkpoint": row.checkpoint,
            "checkpoint_sha256": row.checkpoint_sha256,
            "checkpoint_validation_gate": row.validation_gate,
            **{metric: metrics[metric] for metric in FORMAL_METRICS},
            "cumulative_reward": metrics["cumulative_reward"],
        })
    return pd.DataFrame(rows)


def factorial_effects(factorial: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in factorial.groupby("seed"):
        indexed = group.set_index("variant")
        for metric in FORMAL_METRICS:
            base = float(indexed.loc["SAC Basic", metric])
            forecast = float(indexed.loc["SAC + Forecast", metric])
            memory = float(indexed.loc["SAC + LSTM", metric])
            full = float(indexed.loc["SACSI Full", metric])
            rows.append({
                "seed": int(seed),
                "metric": metric,
                "base": base,
                "forecast_only": forecast,
                "memory_only": memory,
                "full": full,
                "forecast_main_effect": ((forecast - base) + (full - memory)) / 2,
                "memory_main_effect": ((memory - base) + (full - forecast)) / 2,
                "interaction": full - forecast - memory + base,
            })
    return pd.DataFrame(rows)


def run_diagnostics(registry: pd.DataFrame, weather, forecasts: dict[int, pd.DataFrame]):
    context_rows, robustness_rows, sequence_rows = [], [], []
    full_registry = registry.loc[registry["variant"] == "SACSI Full"]
    for row in full_registry.itertuples(index=False):
        agent, metadata = load_context_agent(row.variant, ROOT / Path(row.checkpoint))
        actions_by_condition = {}
        metrics_by_condition = {}
        for condition in CONTEXT_CONDITIONS:
            env = SACSIEnv(
                forecasts[20], DATA / "normalizer.json", len(weather), row.seed, sequence_length=24
            )
            metrics, actions = evaluate_context(agent, env, condition, return_actions=True)
            actions_by_condition[condition] = actions
            metrics_by_condition[condition] = metrics
        full_actions = actions_by_condition["Full"]
        for condition in CONTEXT_CONDITIONS:
            context_rows.append({
                "seed": row.seed,
                "condition": condition,
                "mean_abs_action_delta_vs_full_mm": float(np.mean(np.abs(
                    actions_by_condition[condition] - full_actions
                ))),
                "history_residual_norm": metadata["history_residual_norm"],
                "forecast_residual_norm": metadata["forecast_residual_norm"],
                "history_branch_active": metadata["history_residual_norm"] > 1e-8,
                "forecast_branch_active": metadata["forecast_residual_norm"] > 1e-8,
                **{metric: metrics_by_condition[condition][metric] for metric in FORMAL_METRICS},
            })
        for level in FORECAST_LEVELS:
            metrics = metrics_by_condition["Full"] if level == 20 else evaluate_context(
                agent,
                SACSIEnv(
                    forecasts[level], DATA / "normalizer.json", len(weather), row.seed,
                    sequence_length=24,
                ),
            )
            robustness_rows.append({
                "seed": row.seed,
                "forecast_level": f"SF{level}",
                **{metric: metrics[metric] for metric in FORMAL_METRICS},
            })
        for length in SEQUENCE_LENGTHS:
            metrics = metrics_by_condition["Full"] if length == 24 else evaluate_context(
                agent,
                SACSIEnv(
                    forecasts[20], DATA / "normalizer.json", len(weather), row.seed,
                    sequence_length=length,
                ),
            )
            sequence_rows.append({
                "seed": row.seed,
                "sequence_length": length,
                **{metric: metrics[metric] for metric in FORMAL_METRICS},
            })
    return pd.DataFrame(context_rows), pd.DataFrame(robustness_rows), pd.DataFrame(sequence_rows)


def validate_outputs(factorial, context, robustness, sequence, effects) -> None:
    if len(factorial) != 12 or factorial.groupby("seed")["variant"].nunique().ne(4).any():
        raise RuntimeError("Incomplete 2x2 factorial")
    if len(context) != 27 or context.groupby("seed")["condition"].nunique().ne(9).any():
        raise RuntimeError("Incomplete context interventions")
    if len(robustness) != 9 or robustness.groupby("seed")["forecast_level"].nunique().ne(3).any():
        raise RuntimeError("Incomplete forecast robustness")
    if len(sequence) != 12 or sequence.groupby("seed")["sequence_length"].nunique().ne(4).any():
        raise RuntimeError("Incomplete sequence sensitivity")
    if len(effects) != len(SEEDS) * len(FORMAL_METRICS):
        raise RuntimeError("Incomplete factorial effects")
    for frame in (factorial, context, robustness, sequence):
        if not np.isfinite(frame[list(FORMAL_METRICS)].to_numpy()).all():
            raise RuntimeError("Non-finite POMDP ablation metrics")
        if frame["max_abs_mass_balance_error_mm"].max() > 1e-8:
            raise RuntimeError("Mass-balance gate failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    training_weather = pd.read_csv(DATA / "train_2021_2023.csv", parse_dates=["timestamp"])
    validation_weather = pd.read_csv(DATA / "validation_2024.csv", parse_dates=["timestamp"])
    training_sf20, validation_sf20 = load_sf20(training_weather), load_sf20(validation_weather)
    episodes, episode_length, validation_interval = (12, 48, 6) if args.smoke else (20, 336, 5)
    seeds = (11,) if args.smoke else SEEDS
    for seed in seeds:
        forecast_meta = None if args.smoke else completed_run(
            "SAC + Forecast", seed, episodes, episode_length
        )
        if forecast_meta:
            print(f"SKIP Forecast seed={seed} completed checkpoint", flush=True)
        else:
            forecast_meta = train_forecast_variant(
                seed,
                training_sf20.head(96) if args.smoke else training_sf20,
                validation_sf20.head(48) if args.smoke else validation_sf20,
                episodes, episode_length, validation_interval, args.smoke,
            )
        print(f"Forecast seed={seed} selected={forecast_meta['selected_episode']}", flush=True)
        lstm_meta = None if args.smoke else completed_run(
            "SAC + LSTM", seed, episodes, episode_length
        )
        if lstm_meta:
            print(f"SKIP LSTM seed={seed} completed checkpoint", flush=True)
        else:
            lstm_meta = train_lstm_variant(
                seed,
                training_weather.head(96) if args.smoke else training_weather,
                validation_weather.head(48) if args.smoke else validation_weather,
                episodes, episode_length, validation_interval, args.smoke,
            )
        print(f"LSTM seed={seed} selected={lstm_meta['selected_episode']}", flush=True)
        full_meta = None if args.smoke else completed_run(
            "SACSI Full", seed, episodes, episode_length
        )
        if full_meta:
            print(f"SKIP SACSI seed={seed} completed checkpoint", flush=True)
        else:
            full_meta = train_full_variant(
                seed,
                training_sf20.head(96) if args.smoke else training_sf20,
                validation_sf20.head(48) if args.smoke else validation_sf20,
                episodes, episode_length, validation_interval, args.smoke,
            )
        print(f"SACSI seed={seed} selected={full_meta['selected_episode']}", flush=True)
    if args.smoke:
        return

    registry = build_registry()
    weather = pd.read_csv(DATA / "benchmark_2025.csv", parse_dates=["timestamp"])
    validate_common_support(weather)
    all_weather = pd.read_csv(DATA / "data_clean.csv", parse_dates=["timestamp"])
    forecasts = {}
    for level in FORECAST_LEVELS:
        generated = build_synthetic_forecast(all_weather, level / 100)
        generated = generated.loc[generated["timestamp"].dt.year == 2025].reset_index(drop=True)
        validate_common_support(weather, generated)
        forecasts[level] = merge_forecast(weather, generated)

    factorial = evaluate_factorial(registry, weather, forecasts[20])
    effects = factorial_effects(factorial)
    context, robustness, sequence = run_diagnostics(registry, weather, forecasts)
    validate_outputs(factorial, context, robustness, sequence, effects)
    RESULTS.mkdir(parents=True, exist_ok=True)
    registry.to_csv(RESULTS / "sac_family_checkpoint_registry.csv", index=False)
    factorial.to_csv(RESULTS / "sac_family_factorial_results.csv", index=False)
    context.to_csv(RESULTS / "context_intervention_results.csv", index=False)
    robustness.to_csv(RESULTS / "forecast_robustness.csv", index=False)
    sequence.to_csv(RESULTS / "sequence_sensitivity.csv", index=False)
    effects.to_csv(RESULTS / "factorial_effects.csv", index=False)
    manifest = {
        "module": "8G",
        "status": "COMPLETE_EXPLORATORY_REVALIDATION",
        "seeds": list(SEEDS),
        "reward_version": "reward_v4",
        "base_anchor_budget_per_seed": 6720,
        "context_adaptation_budget_per_seed": 6720,
        "checkpoint_selection": "validation_2024_gate_target_water_rmse",
        "evaluation": "retrospective benchmark 2025; opened only after training lock",
        "factorial_variants": list(VARIANTS),
        "context_conditions": list(CONTEXT_CONDITIONS),
        "forecast_levels": [f"SF{level}" for level in FORECAST_LEVELS],
        "sequence_lengths": list(SEQUENCE_LENGTHS),
        "claim_guard": {
            "branch_activation_is_not_performance_benefit": True,
            "aggregate_means_are_not_statistical_significance": True,
            "performance_superiority_claim_released": False,
            "statistical_significance_claim_released": False,
        },
        "protocol_limitation": (
            "Context variants use the locked SAC anchor plus an equal adaptation budget; "
            "factorial performance effects are exploratory and include adaptation effects."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / "pomdp_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        factorial.groupby("variant")[["time_in_target_pct", "total_irrigation_mm"]]
        .agg(["mean", "std"]).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
