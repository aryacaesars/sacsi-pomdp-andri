"""Matched-seed statistics for the locked four-model RL factorial design."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


RL_METHODS = ("SAC Basic", "SAC + Forecast", "SAC + LSTM", "SACSI Full")
LOCKED_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
FACTOR_LEVELS = {
    "SAC Basic": (0, 0),
    "SAC + Forecast": (1, 0),
    "SAC + LSTM": (0, 1),
    "SACSI Full": (1, 1),
}


def validate_matched_design(runs: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "method", "method_type", "validation_gate"}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"Missing master-table columns: {sorted(missing)}")
    rl = runs.loc[runs["method_type"] == "rl"].copy()
    rl["seed"] = pd.to_numeric(rl["seed"], errors="raise").astype(int)
    if set(rl["method"]) != set(RL_METHODS):
        raise ValueError("Matched design must contain the four locked RL methods")
    if tuple(sorted(rl["seed"].unique())) != LOCKED_SEEDS:
        raise ValueError("Matched design must contain the 10 locked seeds")
    if rl.duplicated(["seed", "method"]).any():
        raise ValueError("Matched design contains duplicate seed-method rows")
    if not rl.groupby("seed")["method"].nunique().eq(4).all() or len(rl) != 40:
        raise ValueError("Every seed must contain all four RL methods")
    rl[["forecast", "memory"]] = rl["method"].map(FACTOR_LEVELS).apply(pd.Series)
    return rl.sort_values(["seed", "forecast", "memory"]).reset_index(drop=True)


def factorial_contrasts(master: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = master.pivot(index="seed", columns="method", values=metric).astype(float)
    basic = wide["SAC Basic"]
    forecast = wide["SAC + Forecast"]
    memory = wide["SAC + LSTM"]
    full = wide["SACSI Full"]
    return pd.DataFrame({
        "seed": wide.index.astype(int),
        "forecast_main_effect": ((forecast + full) - (basic + memory)) / 2,
        "memory_main_effect": ((memory + full) - (basic + forecast)) / 2,
        "forecast_x_memory_interaction": full - forecast - memory + basic,
    }).reset_index(drop=True)


def bootstrap_mean_ci(
    values,
    resamples: int = 20_000,
    confidence: float = 0.95,
    seed: int = 2025,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite one-dimensional sample")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(low), float(high)


def exact_sign_flip_pvalue(values) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Sign-flip values must be a finite one-dimensional sample")
    observed = abs(float(values.mean()))
    permutations = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    permuted = np.abs((permutations * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def one_df_repeated_effect(
    values,
    effect: str,
    unit: str,
    resamples: int = 20_000,
    seed: int = 2025,
) -> dict[str, float | int | str | bool]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    standard_error = sd / np.sqrt(n)
    t_statistic = mean / standard_error if standard_error else 0.0
    f_statistic = t_statistic ** 2
    p_value = float(2 * stats.t.sf(abs(t_statistic), df=n - 1))
    ci_low, ci_high = bootstrap_mean_ci(values, resamples, seed=seed)
    t_margin = float(stats.t.ppf(0.975, df=n - 1) * standard_error)
    return {
        "effect": effect,
        "n_seeds": n,
        "unit": unit,
        "mean_effect": mean,
        "sd_effect": sd,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "t_ci95_low": mean - t_margin,
        "t_ci95_high": mean + t_margin,
        "F": f_statistic,
        "df1": 1,
        "df2": n - 1,
        "p_value": p_value,
        "partial_eta_squared": f_statistic / (f_statistic + n - 1),
        "exact_sign_flip_p": exact_sign_flip_pvalue(values),
        "significant_alpha_0_05": p_value < 0.05,
    }


def holm_adjust(p_values) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not ((0 <= p_values) & (p_values <= 1)).all():
        raise ValueError("p-values must be one-dimensional values in [0, 1]")
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    ).clip(max=1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def repeated_measures_omnibus(values) -> dict[str, float | int]:
    """Friedman primary test plus one-way repeated-measures ANOVA sensitivity."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("Repeated-measures values must be a finite 2-D matrix")
    n_subjects, n_conditions = values.shape
    friedman_statistic, friedman_p = stats.friedmanchisquare(*values.T)
    grand = values.mean()
    ss_condition = n_subjects * np.square(values.mean(axis=0) - grand).sum()
    ss_subject = n_conditions * np.square(values.mean(axis=1) - grand).sum()
    ss_total = np.square(values - grand).sum()
    ss_error = ss_total - ss_condition - ss_subject
    df_condition = n_conditions - 1
    df_error = (n_subjects - 1) * df_condition
    f_statistic = (ss_condition / df_condition) / (ss_error / df_error) if ss_error else np.inf
    return {
        "n_subjects": n_subjects,
        "n_conditions": n_conditions,
        "friedman_chi_square": float(friedman_statistic),
        "friedman_df": df_condition,
        "friedman_p": float(friedman_p),
        "rm_anova_F": float(f_statistic),
        "rm_anova_df1": df_condition,
        "rm_anova_df2": df_error,
        "rm_anova_p": float(stats.f.sf(f_statistic, df_condition, df_error)),
        "rm_anova_partial_eta_squared": float(ss_condition / (ss_condition + ss_error)),
    }


def paired_comparisons(
    master: pd.DataFrame,
    metric: str,
    reference: str,
    comparators: tuple[str, ...],
    method_column: str = "method",
    resamples: int = 20_000,
    seed: int = 2025,
) -> pd.DataFrame:
    """Matched differences with pre-specified parametric and exact inference."""
    wide = master.pivot(index="seed", columns=method_column, values=metric).astype(float)
    required = {reference, *comparators}
    if not required.issubset(wide.columns) or wide[list(required)].isna().any().any():
        raise ValueError("Every paired comparison requires all matched seed cells")
    rows = []
    for index, comparison in enumerate(comparators):
        differences = (wide[reference] - wide[comparison]).to_numpy()
        mean = float(differences.mean())
        sd = float(differences.std(ddof=1))
        standard_error = sd / np.sqrt(len(differences))
        t_statistic = mean / standard_error if standard_error else 0.0
        p_value = float(2 * stats.t.sf(abs(t_statistic), len(differences) - 1))
        ci_low, ci_high = bootstrap_mean_ci(differences, resamples, seed=seed + index)
        t_margin = float(stats.t.ppf(0.975, df=len(differences) - 1) * standard_error)
        rows.append({
            "comparison": f"{reference} - {comparison}",
            "n_pairs": len(differences),
            "mean_difference_pp": mean,
            "sd_difference_pp": sd,
            "bootstrap_ci95_low_pp": ci_low,
            "bootstrap_ci95_high_pp": ci_high,
            "t_ci95_low_pp": mean - t_margin,
            "t_ci95_high_pp": mean + t_margin,
            "t": t_statistic,
            "df": len(differences) - 1,
            "p_raw": p_value,
            "cohens_dz": mean / sd if sd else 0.0,
            "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
        })
    output = pd.DataFrame(rows)
    output["p_holm"] = holm_adjust(output["p_raw"])
    output["significant_holm_0_05"] = output["p_holm"] < 0.05
    output["exact_sign_flip_p_holm"] = holm_adjust(output["exact_sign_flip_p"])
    output["exact_significant_holm_0_05"] = output["exact_sign_flip_p_holm"] < 0.05
    return output


def pairwise_full_comparisons(
    master: pd.DataFrame,
    metric: str,
    resamples: int = 20_000,
    seed: int = 2025,
) -> pd.DataFrame:
    return paired_comparisons(
        master,
        metric,
        "SACSI Full",
        ("SAC Basic", "SAC + Forecast", "SAC + LSTM"),
        resamples=resamples,
        seed=seed,
    )
