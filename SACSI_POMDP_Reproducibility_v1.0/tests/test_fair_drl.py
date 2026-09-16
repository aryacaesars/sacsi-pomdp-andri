import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ddpg import DDPGConfig
from evaluation.final_benchmark import FORMAL_METRICS, sha256_file, validate_common_support
from run_fair_drl_benchmark import actual_common_fields, config_hash
from sac_basic import SACAgent, SACConfig
from sac_basic.training import _selection_key
from td3 import TD3Config


def metadata():
    return {
        "virtual_garden_version": "field_capacity_0.35",
        "action_min_mm_h": 0.0,
        "reward_version": "reward_v4",
        "training_period": "2021-2023",
        "validation_period": "2024",
        "episodes": 20,
        "episode_length": 336,
        "environment_interactions": 6720,
        "validation_interval": 5,
        "checkpoint_selection": "validation_2024_gate_target_water_rmse",
        "forecast": False,
        "history": False,
    }


def test_common_fairness_fields_and_hash_match_all_algorithms():
    fields = [actual_common_fields(config, metadata()) for config in (
        DDPGConfig(), TD3Config(), SACConfig()
    )]
    assert fields[0] == fields[1] == fields[2]
    assert len({config_hash(value) for value in fields}) == 1


def test_locked_selection_key_prioritizes_gate_then_target_water_rmse():
    base = {
        "time_in_target_pct": 60.0,
        "violation_rate_pct": 40.0,
        "total_irrigation_mm": 600.0,
        "mean_soil_moisture": 0.28,
        "deficit_rate_pct": 10.0,
        "max_abs_mass_balance_error_mm": 0.0,
        "rmse_band": 0.02,
    }
    failed = {**base, "time_in_target_pct": 99.0, "mean_soil_moisture": 0.21}
    lower_water = {**base, "total_irrigation_mm": 500.0}
    assert _selection_key(base) > _selection_key(failed)
    assert _selection_key(lower_water) > _selection_key(base)


def test_sac_checkpoint_round_trip():
    agent = SACAgent(SACConfig(hidden_dim=16), device="cpu")
    observation = np.zeros(8, dtype=np.float32)
    expected = agent.select_action(observation, deterministic=True)
    path = ROOT / "Checkpoints" / "Fair_DRL" / "SAC" / f".round_trip_{uuid4().hex}.pt"
    try:
        agent.save(path, {"seed": 11})
        loaded, saved_metadata = SACAgent.load(path, device="cpu")
        assert saved_metadata == {"seed": 11}
        assert np.allclose(loaded.select_action(observation, deterministic=True), expected)
    finally:
        path.unlink(missing_ok=True)


def test_fair_drl_production_artifacts_are_complete():
    result_dir = ROOT / "Results" / "Fair_DRL"
    validation = pd.read_csv(result_dir / "fair_drl_results_validation.csv")
    benchmark = pd.read_csv(result_dir / "fair_drl_results_2025.csv")
    registry = pd.read_csv(result_dir / "fair_drl_checkpoint_registry.csv")
    audit = json.loads((result_dir / "fairness_audit.json").read_text(encoding="utf-8"))
    benchmark_weather = pd.read_csv(
        ROOT / "00_Dataset" / "Processed" / "benchmark_2025.csv", parse_dates=["timestamp"]
    )
    validate_common_support(benchmark_weather)
    expected_pairs = {(algorithm, seed) for algorithm in ("DDPG", "TD3", "SAC") for seed in (11, 22, 33)}
    assert set(zip(validation["algorithm_family"], validation["seed"])) == expected_pairs
    assert set(zip(benchmark["algorithm_family"], benchmark["seed"])) == expected_pairs
    assert set(zip(registry["algorithm_family"], registry["seed"])) == expected_pairs
    assert list(validation.columns) == list(benchmark.columns)
    assert set(validation["evaluation_hours"]) == {8784}
    assert set(benchmark["evaluation_hours"]) == {8760}
    assert set(validation["environment_interactions"]) == {6720}
    assert set(benchmark["result_status"]) == {"RETROSPECTIVE_BENCHMARK"}
    assert not validation["forecast"].any() and not validation["history"].any()
    for frame in (validation, benchmark):
        assert np.isfinite(frame[list(FORMAL_METRICS) + ["cumulative_reward"]].to_numpy()).all()
        assert frame["max_abs_mass_balance_error_mm"].max() <= 1e-8
    assert audit["status"] == "PASS" and all(audit["checks"].values())
    assert len(set(audit["common_config_hash_by_algorithm"].values())) == 1
    assert audit["no_retraining_or_checkpoint_reselection_after_2025_opening"]
    for row in registry.itertuples(index=False):
        checkpoint = ROOT / Path(row.checkpoint)
        assert checkpoint.is_file()
        assert sha256_file(checkpoint) == row.checkpoint_sha256
