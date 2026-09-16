"""Run the pre-specified final matched-seed statistical analysis."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.final_benchmark import FORMAL_METRICS  # noqa: E402
from evaluation.final_benchmark import sha256_file  # noqa: E402
from evaluation.statistics import (  # noqa: E402
    bootstrap_mean_ci,
    factorial_contrasts,
    one_df_repeated_effect,
    pairwise_full_comparisons,
    validate_matched_design,
)


SOURCE = ROOT / "Results" / "Final_Experiment" / "benchmark_2025_runs.csv"
OUTPUT = ROOT / "Statistics"
TABLES = ROOT / "Tables"
PRIMARY_ENDPOINT = "time_in_target_pct"
BOOTSTRAP_RESAMPLES = 20_000


def descriptive_factorial(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in FORMAL_METRICS:
        contrasts = factorial_contrasts(master, metric)
        for index, column in enumerate(contrasts.columns[1:]):
            values = contrasts[column].to_numpy()
            ci_low, ci_high = bootstrap_mean_ci(
                values, BOOTSTRAP_RESAMPLES, seed=3000 + index
            )
            rows.append({
                "metric": metric,
                "effect": column,
                "n_seeds": len(values),
                "mean_effect": float(values.mean()),
                "sd_effect": float(values.std(ddof=1)),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "inferential_status": (
                    "primary_pre-specified" if metric == PRIMARY_ENDPOINT
                    else "secondary_descriptive_only"
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    runs = pd.read_csv(SOURCE)
    master = validate_matched_design(runs)
    if (~master["validation_gate"].astype(str).str.lower().eq("true")).sum() != 1:
        raise RuntimeError("Expected the retained failed validation seed in the master table")

    contrasts = factorial_contrasts(master, PRIMARY_ENDPOINT)
    labels = {
        "forecast_main_effect": "Forecast main effect",
        "memory_main_effect": "Memory main effect",
        "forecast_x_memory_interaction": "Forecast x Memory interaction",
    }
    factorial = pd.DataFrame([
        one_df_repeated_effect(
            contrasts[column], labels[column], "percentage points",
            BOOTSTRAP_RESAMPLES, seed=2025 + index,
        )
        for index, column in enumerate(contrasts.columns[1:])
    ])
    pairwise = pairwise_full_comparisons(
        master, PRIMARY_ENDPOINT, BOOTSTRAP_RESAMPLES, seed=2025
    )
    factorial["bootstrap_t_inference_discordant"] = (
        (factorial["bootstrap_ci95_low"] * factorial["bootstrap_ci95_high"] > 0)
        != factorial["significant_alpha_0_05"]
    )
    pairwise["bootstrap_t_inference_discordant"] = (
        (pairwise["bootstrap_ci95_low_pp"] * pairwise["bootstrap_ci95_high_pp"] > 0)
        != pairwise["significant_holm_0_05"]
    )
    descriptive = descriptive_factorial(master)
    baselines = runs.loc[runs["method_type"] == "baseline", [
        "method", *FORMAL_METRICS,
    ]].copy()
    baselines.insert(1, "n_trajectories", 1)
    baselines.insert(2, "inferential_eligible", False)
    baselines.insert(3, "reporting_role", "deterministic trajectory reference only")

    master.to_csv(OUTPUT / "master_seed_table.csv", index=False)
    contrasts.to_csv(OUTPUT / "factorial_primary_contrasts.csv", index=False)
    factorial.to_csv(OUTPUT / "factorial_rm_anova_primary.csv", index=False)
    pairwise.to_csv(OUTPUT / "pairwise_sacsi_primary.csv", index=False)
    descriptive.to_csv(OUTPUT / "factorial_all_metrics_descriptive.csv", index=False)
    baselines.to_csv(OUTPUT / "deterministic_baseline_reference.csv", index=False)
    factorial.to_csv(TABLES / "statistics_factorial_primary.csv", index=False)
    pairwise.to_csv(TABLES / "statistics_pairwise_primary.csv", index=False)

    findings = {
        "primary_endpoint": "Time in Target (%)",
        "alpha": 0.05,
        "factorial_significant_effects": factorial.loc[
            factorial["significant_alpha_0_05"], "effect"
        ].tolist(),
        "holm_significant_pairwise": pairwise.loc[
            pairwise["significant_holm_0_05"], "comparison"
        ].tolist(),
        "sacsi_superiority_supported": bool(pairwise["significant_holm_0_05"].all()),
        "bootstrap_t_discordance": factorial.loc[
            factorial["bootstrap_t_inference_discordant"], "effect"
        ].tolist() + pairwise.loc[
            pairwise["bootstrap_t_inference_discordant"], "comparison"
        ].tolist(),
        "inference_rule": (
            "Significance follows the pre-specified parametric p-values and Holm correction; "
            "bootstrap and exact sign-flip results are robustness diagnostics."
        ),
        "interpretation_guard": (
            "Technical/context validity is separate from performance superiority. "
            "Deterministic baselines are trajectory references, not fake repeated seeds."
        ),
    }
    (OUTPUT / "statistical_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "design": "2x2 repeated measures; subject=matched training seed",
        "factor_mapping": {
            "SAC Basic": "F0 M0", "SAC + Forecast": "F1 M0",
            "SAC + LSTM": "F0 M1", "SACSI Full": "F1 M1",
        },
        "primary_endpoint": "Time in Target (%)",
        "n_seeds": 10,
        "failed_validation_seeds_retained": int(
            (~master["validation_gate"].astype(str).str.lower().eq("true")).sum()
        ),
        "factorial_test": "one-df within-seed contrast; F=t^2; df1=1; df2=9",
        "pairwise_test": "paired two-sided t-test",
        "multiplicity": "Holm correction over three pre-specified SACSI comparisons",
        "effect_sizes": ["partial eta-squared", "Cohen's dz"],
        "confidence_interval": f"95% percentile bootstrap; {BOOTSTRAP_RESAMPLES} resamples",
        "parametric_confidence_interval": "95% Student-t CI paired to the primary t/F tests",
        "robust_confirmation": "exact two-sided paired sign-flip; all 2^10 assignments",
        "robust_multiplicity": "Holm correction over the same three exact sign-flip comparisons",
        "secondary_metrics": "descriptive contrasts only",
        "deterministic_baselines": "single trajectory references; excluded from seed inference",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "statistical_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("PRIMARY FACTORIAL")
    print(factorial.to_string(index=False))
    print("\nPAIRWISE SACSI")
    print(pairwise.to_string(index=False))
    print("\nSTATISTICAL ANALYSIS FINISHED")


if __name__ == "__main__":
    main()
