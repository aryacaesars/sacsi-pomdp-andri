"""Locked Module 8B reward experiment definitions and physical selection rules."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sac_basic import RewardConfig


SEEDS = (11, 22, 33)
CONFIRMATION_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)


@dataclass(frozen=True)
class RewardExperimentSpec:
    config_id: str
    water_multiplier: float
    violation_multiplier: float
    include_water: bool = True
    include_smoothness: bool = True
    include_violation: bool = True
    ablation_label: str | None = None
    sensitivity_label: str | None = None

    def reward_config(self) -> RewardConfig:
        return RewardConfig(
            name=self.config_id,
            water_weight=0.02 * self.water_multiplier if self.include_water else 0.0,
            smoothness_weight=0.01 if self.include_smoothness else 0.0,
            violation_weight=self.violation_multiplier if self.include_violation else 0.0,
        )


def experiment_specs() -> list[RewardExperimentSpec]:
    specs = [
        RewardExperimentSpec("r_a_moisture", 0.0, 0.0, False, False, False, "R-A"),
        RewardExperimentSpec("r_b_moisture_water", 1.0, 0.0, True, False, False, "R-B"),
        RewardExperimentSpec("r_c_moisture_water_smooth", 1.0, 0.0, True, True, False, "R-C"),
    ]
    for water in (0.5, 1.0, 2.0):
        for violation in (0.5, 1.0, 2.0):
            label = f"wI={water:.1f}|wV={violation:.1f}"
            specs.append(RewardExperimentSpec(
                f"r_d_wi{water:.1f}_wv{violation:.1f}".replace(".", ""),
                water,
                violation,
                ablation_label="R-D" if water == violation == 1.0 else None,
                sensitivity_label=label,
            ))
    return specs


def confirmation_specs() -> list[RewardExperimentSpec]:
    candidate_ids = {"r_d_wi10_wv10", "r_d_wi05_wv20", "r_d_wi10_wv20"}
    return [spec for spec in experiment_specs() if spec.config_id in candidate_ids]


def mark_pareto(summary: pd.DataFrame) -> pd.DataFrame:
    """Mark candidates not dominated on lower water and higher target occupancy."""
    result = summary.copy()
    flags = []
    for row in result.itertuples():
        dominated = (
            (result["total_irrigation_mm_mean"] <= row.total_irrigation_mm_mean)
            & (result["time_in_target_pct_mean"] >= row.time_in_target_pct_mean)
            & (
                (result["total_irrigation_mm_mean"] < row.total_irrigation_mm_mean)
                | (result["time_in_target_pct_mean"] > row.time_in_target_pct_mean)
            )
        ).any()
        flags.append(not bool(dominated))
    result["pareto_non_dominated"] = flags
    return result


def summarize_candidates(results: pd.DataFrame, experiment: str) -> pd.DataFrame:
    subset = results.loc[results["experiment"] == experiment]
    metrics = [
        "time_in_target_pct", "total_irrigation_mm", "violation_rate_pct",
        "deficit_rate_pct", "rmse_band", "action_smoothness",
        "reward_tracking_cost_total", "reward_water_cost_total",
        "reward_smoothness_cost_total", "reward_violation_cost_total",
    ]
    grouped = subset.groupby("candidate_id", sort=False)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index()
    summary.insert(0, "experiment", experiment)
    summary["n_seeds"] = subset.groupby("candidate_id", sort=False).size().to_numpy()
    return mark_pareto(summary)


def reward_decision(sensitivity_summary: pd.DataFrame) -> dict:
    current_id = "wI=1.0|wV=1.0"
    current = sensitivity_summary.set_index("candidate_id").loc[current_id]
    best = sensitivity_summary.sort_values(
        ["time_in_target_pct_mean", "total_irrigation_mm_mean"],
        ascending=[False, True],
    ).iloc[0]
    target_gap = float(best.time_in_target_pct_mean - current.time_in_target_pct_mean)
    water_ratio = float(current.total_irrigation_mm_mean / best.total_irrigation_mm_mean)
    checks = {
        "pareto_non_dominated": bool(current.pareto_non_dominated),
        "seed_stable_time_in_target_std_le_10pp": bool(current.time_in_target_pct_std <= 10.0),
        "target_gap_from_best_le_2pp": target_gap <= 2.0,
        "water_not_over_110pct_of_best_target_candidate": water_ratio <= 1.10,
    }
    decision = "KEEP CURRENT REWARD" if all(checks.values()) else "REVISE REWARD"
    selected = current if decision.startswith("KEEP") else best
    selected_id = current_id if decision.startswith("KEEP") else str(best.candidate_id)
    multipliers = {
        part.split("=")[0]: float(part.split("=")[1])
        for part in selected_id.split("|")
    }
    return {
        "decision": decision,
        "decision_status": "PROVISIONAL_SELECTED_PENDING_10_SEED_CONFIRMATION",
        "current_reward": "reward_v2 / R-D / wI=1.0 / wV=1.0",
        "provisional_reward": {
            "reward_version": "reward_v2" if decision.startswith("KEEP") else "reward_v4",
            "candidate_id": selected_id,
            "tracking_weight": 100.0,
            "deficit_ratio": 2.0,
            "water_weight": 0.02 * multipliers["wI"],
            "smoothness_weight": 0.01,
            "violation_weight": multipliers["wV"],
            "selection_rule": "Highest mean Time in Target; lowest mean water is the tie-breaker.",
            "validation_summary": selected.to_dict(),
        },
        "selection_split": "validation_2024",
        "benchmark_2025_accessed": False,
        "selection_basis": "Physical metrics and Water-vs-Time-in-Target Pareto; never cumulative reward.",
        "predefined_thresholds": {
            "time_in_target_seed_std_max_pp": 10.0,
            "time_in_target_gap_from_best_max_pp": 2.0,
            "water_ratio_to_best_target_candidate_max": 1.10,
        },
        "checks": checks,
        "current_summary": current.to_dict(),
        "best_target_candidate": best.to_dict(),
        "target_gap_pp": target_gap,
        "water_ratio_to_best_target_candidate": water_ratio,
        "term_scale_evidence": {
            "tracking": "reward_tracking_cost_total/mean/max",
            "water": "reward_water_cost_total/mean/max",
            "smoothness": "reward_smoothness_cost_total/mean/max",
            "violation": "reward_violation_cost_total/mean/max",
        },
        "next_gate": "Confirm reward_v2, provisional reward_v4, and wI=1.0/wV=2.0 on 10 matched seeds before final lock.",
        "claim_guard": "This is validation-only reward selection, not evidence of final controller superiority.",
    }


def confirmation_decision(summary: pd.DataFrame) -> dict:
    best_target = float(summary["time_in_target_pct_mean"].max())
    eligible = summary.loc[
        summary["pareto_non_dominated"]
        & (summary["time_in_target_pct_std"] <= 10.0)
        & (summary["time_in_target_pct_mean"] >= best_target - 0.5)
    ].sort_values(["total_irrigation_mm_mean", "time_in_target_pct_mean"], ascending=[True, False])
    if eligible.empty:
        raise ValueError("no confirmation candidate passed the predefined stability/Pareto gate")
    selected = eligible.iloc[0]
    versions = {
        "wI=1.0|wV=1.0": "reward_v2",
        "wI=0.5|wV=2.0": "reward_v4",
        "wI=1.0|wV=2.0": "reward_v5",
    }
    selected_id = str(selected.candidate_id)
    multipliers = {part.split("=")[0]: float(part.split("=")[1]) for part in selected_id.split("|")}
    return {
        "decision_status": "FINAL_REWARD_LOCKED",
        "selected_reward": {
            "reward_version": versions[selected_id],
            "candidate_id": selected_id,
            "tracking_weight": 100.0,
            "deficit_ratio": 2.0,
            "water_weight": 0.02 * multipliers["wI"],
            "smoothness_weight": 0.01,
            "violation_weight": multipliers["wV"],
            "validation_summary": selected.to_dict(),
        },
        "selection_split": "validation_2024",
        "benchmark_2025_accessed": False,
        "matched_seeds": list(CONFIRMATION_SEEDS),
        "predefined_rule": (
            "Require Pareto non-dominance and Time-in-Target SD <= 10 pp; among candidates within "
            "0.5 pp of the highest mean Time in Target, select the lowest mean irrigation."
        ),
        "practical_equivalence_margin_pp": 0.5,
        "eligible_candidates": eligible["candidate_id"].tolist(),
        "claim_guard": "Selected on validation 2024; this is not a final 2025 superiority result.",
    }


def reconcile_confirmation(base_decision: dict, confirmation: dict) -> dict:
    final = dict(base_decision)
    final.pop("provisional_reward", None)
    final.pop("next_gate", None)
    final.update({
        "decision_status": "FINAL_LOCKED_AFTER_10_SEED_CONFIRMATION",
        "locked_reward": confirmation["selected_reward"],
        "confirmation_evidence": {
            "results": "reward_confirmation_results.csv",
            "summary": "reward_confirmation_summary.csv",
            "decision": "reward_confirmation_decision.json",
            "matched_seeds": confirmation["matched_seeds"],
            "predefined_rule": confirmation["predefined_rule"],
        },
        "retraining_requirement": (
            "Use reward_v4 for Modules 8C-8H and retrain every model included in the new fair benchmark. "
            "Historical reward_v2 checkpoints/results remain provenance artifacts and must not be overwritten."
        ),
    })
    return final
