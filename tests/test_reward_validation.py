import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.reward_validation import confirmation_specs, experiment_specs, mark_pareto
from sac_basic import LOCKED_REWARD_CONFIG, REWARD_V2_CONFIG, RewardConfig, SACIrrigationEnv


def make_env(reward_config=None, length=48):
    data = pd.read_csv(ROOT / "00_Dataset" / "Processed" / "train_2021_2023.csv", nrows=length)
    normalizer = json.loads(
        (ROOT / "00_Dataset" / "Processed" / "normalizer.json").read_text(encoding="utf-8")
    )
    return SACIrrigationEnv(data, normalizer, length, 11, reward_config)


def test_formal_default_reward_exactly_matches_locked_reward_v2():
    env = make_env(REWARD_V2_CONFIG)
    env.reset(start_index=0)
    previous_action = env.previous_irrigation
    _, reward, _, info = env.step([2.5])
    cfg = env.garden.config
    deficit = max(cfg.target_min - info["theta"], 0.0)
    surplus = max(info["theta"] - cfg.target_max, 0.0)
    legacy = (
        1.0 if deficit == surplus == 0 else -200.0 * deficit - 100.0 * surplus
    ) - 0.02 * info["irrigation_mm"] - 0.01 * abs(info["irrigation_mm"] - previous_action)
    assert np.isclose(reward, legacy)
    assert np.isclose(
        reward,
        info["reward_offset"]
        - sum(info[f"reward_{term}_cost"] for term in (
            "tracking", "water", "smoothness", "violation"
        )),
    )


def test_environment_default_is_locked_reward_v4():
    env = make_env()
    assert env.reward_config == LOCKED_REWARD_CONFIG
    assert env.reward_version == "reward_v4"
    assert env.reward_config.water_weight == 0.01
    assert env.reward_config.violation_weight == 2.0


def test_reward_design_has_12_unique_configs_and_reuses_full_default():
    specs = experiment_specs()
    assert len(specs) == len({spec.config_id for spec in specs}) == 12
    assert sum(spec.ablation_label is not None for spec in specs) == 4
    assert sum(spec.sensitivity_label is not None for spec in specs) == 9
    default = next(spec for spec in specs if spec.ablation_label == "R-D")
    assert default.sensitivity_label == "wI=1.0|wV=1.0"
    assert default.reward_config() == RewardConfig(name=default.config_id)
    assert {spec.sensitivity_label for spec in confirmation_specs()} == {
        "wI=1.0|wV=1.0", "wI=0.5|wV=2.0", "wI=1.0|wV=2.0",
    }


def test_pareto_marks_only_non_dominated_candidates():
    frame = pd.DataFrame({
        "candidate_id": ["balanced", "water_saver", "dominated"],
        "total_irrigation_mm_mean": [400.0, 300.0, 450.0],
        "time_in_target_pct_mean": [60.0, 55.0, 50.0],
    })
    marked = mark_pareto(frame).set_index("candidate_id")
    assert bool(marked.loc["balanced", "pareto_non_dominated"])
    assert bool(marked.loc["water_saver", "pareto_non_dominated"])
    assert not bool(marked.loc["dominated", "pareto_non_dominated"])


def test_module_8b_outputs_are_complete_and_validation_only():
    result_dir = ROOT / "Results" / "Reward_Validation"
    ablation = pd.read_csv(result_dir / "reward_ablation_results.csv")
    sensitivity = pd.read_csv(result_dir / "reward_weight_sensitivity.csv")
    pareto = pd.read_csv(result_dir / "reward_pareto.csv")
    decision = json.loads((result_dir / "reward_decision.json").read_text(encoding="utf-8"))

    assert len(ablation) == 4 * 3
    assert len(sensitivity) == 9 * 3
    assert len(pareto) == 4 + 9
    assert set(ablation["seed"]) == set(sensitivity["seed"]) == {11, 22, 33}
    assert set(ablation["data_split"]) == set(sensitivity["data_split"]) == {"validation_2024"}
    assert ablation["training_finite"].all() and sensitivity["training_finite"].all()
    assert decision["decision"] == "REVISE REWARD"
    assert decision["decision_status"] == "FINAL_LOCKED_AFTER_10_SEED_CONFIRMATION"
    assert decision["locked_reward"]["reward_version"] == "reward_v4"
    assert decision["locked_reward"]["candidate_id"] == "wI=0.5|wV=2.0"
    assert decision["benchmark_2025_accessed"] is False
    assert (ROOT / "Figures" / "Reward_Validation" / "reward_pareto_validation_2024.html").exists()


def test_reward_confirmation_has_10_matched_seeds_and_final_lock():
    result_dir = ROOT / "Results" / "Reward_Validation"
    results = pd.read_csv(result_dir / "reward_confirmation_results.csv")
    summary = pd.read_csv(result_dir / "reward_confirmation_summary.csv")
    decision = json.loads(
        (result_dir / "reward_confirmation_decision.json").read_text(encoding="utf-8")
    )

    assert len(results) == 3 * 10
    assert len(summary) == 3
    assert set(results["seed"]) == {11, 22, 33, 44, 55, 66, 77, 88, 99, 110}
    assert results.groupby("candidate_id")["seed"].nunique().eq(10).all()
    assert set(results["data_split"]) == {"validation_2024"}
    assert results["training_finite"].all()
    assert decision["decision_status"] == "FINAL_REWARD_LOCKED"
    assert decision["selected_reward"]["reward_version"] == "reward_v4"
    assert decision["selected_reward"]["candidate_id"] == "wI=0.5|wV=2.0"
    assert decision["benchmark_2025_accessed"] is False
