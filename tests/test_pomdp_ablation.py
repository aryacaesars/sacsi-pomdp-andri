import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.ablation_robustness import CONTEXT_CONDITIONS, evaluate_context
from evaluation.final_benchmark import FORMAL_METRICS, sha256_file
from run_pomdp_ablation import completed_run, factorial_effects, load_context_agent
from sacsi_full import SACSIEnv


def test_factorial_effect_formulas_are_exact():
    rows = []
    for variant, value in (
        ("SAC Basic", 1.0),
        ("SAC + Forecast", 3.0),
        ("SAC + LSTM", 4.0),
        ("SACSI Full", 10.0),
    ):
        row = {"seed": 11, "variant": variant}
        row.update({metric: value for metric in FORMAL_METRICS})
        rows.append(row)
    effects = factorial_effects(pd.DataFrame(rows))
    first = effects.iloc[0]
    assert first["forecast_main_effect"] == 4.0
    assert first["memory_main_effect"] == 5.0
    assert first["interaction"] == 4.0


def test_context_evaluator_can_return_aligned_actions():
    checkpoint = (
        ROOT / "Checkpoints" / "SACSI_Full"
        / "sacsi_full_seed11_reward_v2_sf20_rrws_k24_training_ep10.pt"
    )
    agent, _ = load_context_agent("SACSI Full", checkpoint)
    weather = pd.read_csv(
        ROOT / "00_Dataset" / "Processed" / "validation_2024.csv",
        parse_dates=["timestamp"], nrows=48,
    )
    forecast = pd.read_csv(
        ROOT / "00_Dataset" / "Processed" / "synthetic_forecast_sf20.csv",
        parse_dates=["timestamp"],
    )
    data = weather.merge(forecast, on="timestamp", how="left", validate="one_to_one")
    env = SACSIEnv(data, ROOT / "00_Dataset" / "Processed" / "normalizer.json", 48, 11)
    metrics, actions = evaluate_context(agent, env, "Full", return_actions=True)
    assert len(actions) == 48 and np.isfinite(actions).all()
    assert 0 <= actions.min() <= actions.max() <= 5
    assert metrics["max_abs_mass_balance_error_mm"] <= 1e-8


def test_resume_guard_accepts_only_complete_locked_run():
    metadata = completed_run("SAC + Forecast", 11, 20, 336)
    assert metadata is not None
    assert metadata["reward_version"] == "reward_v4"
    assert completed_run("SAC + Forecast", 11, 19, 336) is None


def test_pomdp_ablation_production_artifacts_are_complete():
    result_dir = ROOT / "Results" / "POMDP_Ablation"
    factorial = pd.read_csv(result_dir / "sac_family_factorial_results.csv")
    context = pd.read_csv(result_dir / "context_intervention_results.csv")
    robustness = pd.read_csv(result_dir / "forecast_robustness.csv")
    sequence = pd.read_csv(result_dir / "sequence_sensitivity.csv")
    effects = pd.read_csv(result_dir / "factorial_effects.csv")
    registry = pd.read_csv(result_dir / "sac_family_checkpoint_registry.csv")
    manifest = json.loads(
        (result_dir / "pomdp_ablation_manifest.json").read_text(encoding="utf-8")
    )
    pairs = {(variant, seed) for variant in (
        "SAC Basic", "SAC + Forecast", "SAC + LSTM", "SACSI Full"
    ) for seed in (11, 22, 33)}
    assert set(zip(factorial["variant"], factorial["seed"])) == pairs
    assert set(zip(registry["variant"], registry["seed"])) == pairs
    assert set(registry["reward_version"]) == {"reward_v4"}
    assert len(context) == 27 and set(context["condition"]) == set(CONTEXT_CONDITIONS)
    assert len(robustness) == 9 and set(robustness["forecast_level"]) == {"SF10", "SF20", "SF30"}
    assert len(sequence) == 12 and set(sequence["sequence_length"]) == {6, 12, 24, 48}
    assert len(effects) == 3 * len(FORMAL_METRICS)
    for frame in (factorial, context, robustness, sequence):
        assert np.isfinite(frame[list(FORMAL_METRICS)].to_numpy()).all()
        assert frame["max_abs_mass_balance_error_mm"].max() <= 1e-8
    assert manifest["reward_version"] == "reward_v4"
    assert manifest["claim_guard"]["performance_superiority_claim_released"] is False
    assert manifest["claim_guard"]["statistical_significance_claim_released"] is False
    for row in registry.itertuples(index=False):
        checkpoint = ROOT / Path(row.checkpoint)
        assert checkpoint.is_file() and sha256_file(checkpoint) == row.checkpoint_sha256
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)["metadata"]
        if row.variant != "SAC Basic":
            assert metadata["losses_finite"]
            assert not metadata["benchmark_2025_accessed_for_training_or_selection"]
