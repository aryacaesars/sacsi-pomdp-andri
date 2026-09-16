import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.statistics import (
    bootstrap_mean_ci,
    exact_sign_flip_pvalue,
    factorial_contrasts,
    holm_adjust,
    one_df_repeated_effect,
    paired_comparisons,
    repeated_measures_omnibus,
    validate_matched_design,
)


def make_master():
    rows = []
    values = {
        "SAC Basic": 10.0,
        "SAC + Forecast": 12.0,
        "SAC + LSTM": 13.0,
        "SACSI Full": 20.0,
    }
    for seed in (11, 22, 33, 44, 55, 66, 77, 88, 99, 110):
        for method, value in values.items():
            rows.append({
                "seed": seed, "method": method, "method_type": "rl",
                "validation_gate": method != "SAC Basic" or seed != 22,
                "metric": value + seed / 100,
            })
    return validate_matched_design(pd.DataFrame(rows))


def test_matched_factorial_contrasts_have_expected_effects():
    contrasts = factorial_contrasts(make_master(), "metric")
    assert contrasts["forecast_main_effect"].tolist() == pytest.approx([4.5] * 10)
    assert contrasts["memory_main_effect"].tolist() == pytest.approx([5.5] * 10)
    assert contrasts["forecast_x_memory_interaction"].tolist() == pytest.approx([5.0] * 10)


def test_holm_adjustment_and_exact_sign_flip_are_deterministic():
    assert holm_adjust([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert exact_sign_flip_pvalue(np.ones(10)) == pytest.approx(2 / 1024)
    first = bootstrap_mean_ci(np.arange(10), resamples=1000, seed=11)
    second = bootstrap_mean_ci(np.arange(10), resamples=1000, seed=11)
    assert first == second


def test_matched_design_rejects_missing_seed_method_cell():
    incomplete = make_master().iloc[:-1]
    with pytest.raises(ValueError, match="Every seed"):
        validate_matched_design(incomplete)


def test_one_df_effect_reports_f_as_t_squared_and_t_ci():
    result = one_df_repeated_effect(np.arange(1, 11), "effect", "unit", resamples=1000)
    expected_t = np.mean(np.arange(1, 11)) / (np.std(np.arange(1, 11), ddof=1) / np.sqrt(10))
    assert result["F"] == pytest.approx(expected_t ** 2)
    assert result["t_ci95_low"] < result["mean_effect"] < result["t_ci95_high"]


def test_generic_confirmatory_statistics_preserve_matched_pairs():
    master = make_master()
    paired = paired_comparisons(
        master, "metric", "SACSI Full", ("SAC Basic",), resamples=1000
    )
    assert paired.loc[0, "mean_difference_pp"] == pytest.approx(10.0)
    assert paired.loc[0, "exact_sign_flip_p"] == pytest.approx(2 / 1024)
    matrix = master.pivot(index="seed", columns="method", values="metric").to_numpy()
    omnibus = repeated_measures_omnibus(matrix)
    assert omnibus["n_subjects"] == 10
    assert omnibus["n_conditions"] == 4
    assert omnibus["friedman_p"] < 0.05
