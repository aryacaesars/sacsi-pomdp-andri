import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.final_benchmark import FORMAL_METRICS, sha256_file
from scripts.run_confirmatory_benchmark import (
    FACTORIAL_VARIANTS,
    MAIN_MODELS,
    RESULTS,
    SEEDS,
    checkpoint_metadata,
    inferential_outputs,
    validate_result_tables,
)


def result_fixture():
    main_values = {"DDPG": 40.0, "TD3": 45.0, "SAC": 50.0, "SACSI-POMDP": 60.0}
    family_values = {
        "SAC Basic": 50.0,
        "SAC + Forecast": 52.0,
        "SAC + LSTM": 53.0,
        "SACSI Full": 60.0,
    }

    def rows(values, label):
        output = []
        for seed in SEEDS:
            for method, time_in_target in values.items():
                row = {
                    label: method,
                    "seed": seed,
                    "checkpoint_validation_gate": not (seed == 22 and method in values),
                }
                row.update({metric: 1.0 for metric in FORMAL_METRICS})
                row["time_in_target_pct"] = time_in_target + seed / 1000
                row["max_abs_mass_balance_error_mm"] = 0.0
                output.append(row)
        return pd.DataFrame(output)

    return rows(main_values, "model"), rows(family_values, "variant")


def test_confirmatory_design_and_statistics_are_matched():
    main, factorial = result_fixture()
    validate_result_tables(main, factorial)
    outputs = inferential_outputs(main, factorial)
    friedman = outputs["friedman_results.csv"]
    planned = outputs["planned_contrasts.csv"]
    assert friedman.loc[0, "n_subjects"] == 10
    assert friedman.loc[0, "friedman_p"] < 0.05
    assert len(planned) == 6
    assert planned["primary_p_holm"].max() < 0.05
    assert outputs["summary"]["main_locked_pipeline_superiority_supported"]
    assert not outputs["summary"]["unqualified_equal_total_budget_superiority_claim_released"]


def test_confirmatory_design_rejects_duplicate_or_missing_cells():
    main, factorial = result_fixture()
    with pytest.raises(RuntimeError, match="40-row"):
        validate_result_tables(pd.concat((main, main.iloc[[0]]), ignore_index=True), factorial)
    with pytest.raises(RuntimeError, match="40-row"):
        validate_result_tables(main, factorial.iloc[:-1])


def test_all_confirmatory_checkpoints_pass_resume_guard():
    models = ("DDPG", "TD3", "SAC Basic", "SAC + Forecast", "SAC + LSTM", "SACSI Full")
    for model in models:
        for seed in SEEDS:
            metadata = checkpoint_metadata(model, seed)
            assert metadata is not None, f"invalid {model} seed={seed}"
            assert metadata["reward_version"] == "reward_v4"
            assert metadata["losses_finite"]


def test_production_confirmatory_artifacts_when_available():
    manifest_path = RESULTS / "confirmatory_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Production 8H training has not completed yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    main = pd.read_csv(RESULTS / "main_10seed_results_2025.csv")
    factorial = pd.read_csv(RESULTS / "sac_family_10seed_factorial.csv")
    validate_result_tables(main, factorial)
    assert set(main["model"]) == set(MAIN_MODELS)
    assert set(factorial["variant"]) == set(FACTORIAL_VARIANTS)
    assert (main["reward_version"] == "reward_v4").all()
    for name, record in manifest["artifacts"].items():
        path = RESULTS / name
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]
