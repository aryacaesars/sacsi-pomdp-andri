from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.decision_engine import classify_superiority
from evaluation.pareto import pareto_frontier


def test_frozen_main_contrasts_release_three_scoped_superiority_labels() -> None:
    contrasts = pd.read_csv(ROOT / "Results" / "Confirmatory_10Seed" / "planned_contrasts.csv")
    main = contrasts.loc[contrasts["analysis_family"] == "main_benchmark"]

    decisions = classify_superiority(main)

    assert len(decisions) == 3
    assert decisions["decision"].eq("STATISTICALLY SUPERIOR").all()
    assert decisions["method"].eq("SACSI-POMDP").all()


def test_confirmatory_pareto_frontier_retains_tradeoffs_and_marks_dominance() -> None:
    runs = pd.read_csv(ROOT / "Results" / "Confirmatory_10Seed" / "main_10seed_results_2025.csv")
    summary = runs.groupby("model", as_index=False)[
        ["time_in_target_pct", "total_irrigation_mm"]
    ].mean()
    pareto = pareto_frontier(
        summary,
        method_column="model",
        maximize="time_in_target_pct",
        minimize="total_irrigation_mm",
    ).set_index("model")

    assert pareto.loc["SAC", "pareto_status"] == "DOMINATED"
    assert pareto.loc["SAC", "dominated_by"] == "SACSI-POMDP"
    assert pareto.drop(index="SAC")["pareto_non_dominated"].all()


def test_superiority_engine_does_not_promote_non_significant_difference() -> None:
    contrast = pd.DataFrame([{
        "comparison": "A - B",
        "mean_difference_pp": 1.0,
        "bootstrap_ci95_low_pp": -0.5,
        "bootstrap_ci95_high_pp": 2.0,
        "cohens_dz": 0.5,
        "primary_p_holm": 0.2,
    }])

    decision = classify_superiority(contrast).loc[0, "decision"]

    assert decision == "DESCRIPTIVELY BETTER"
