"""Evidence-aware labels for the locked confirmatory comparisons."""

from __future__ import annotations

import pandas as pd


DECISION_COLUMNS = {
    "comparison",
    "mean_difference_pp",
    "bootstrap_ci95_low_pp",
    "bootstrap_ci95_high_pp",
    "cohens_dz",
    "primary_p_holm",
}


def classify_superiority(
    contrasts: pd.DataFrame,
    *,
    alpha: float = 0.05,
    minimum_effect: float = 0.2,
) -> pd.DataFrame:
    """Classify pre-specified contrasts without inventing new inference."""
    missing = DECISION_COLUMNS.difference(contrasts.columns)
    if missing:
        raise ValueError(f"Missing superiority columns: {sorted(missing)}")

    rows = []
    for row in contrasts.itertuples(index=False):
        first, separator, second = row.comparison.partition(" - ")
        if not separator:
            raise ValueError(f"Invalid comparison label: {row.comparison}")

        delta = float(row.mean_difference_pp)
        ci_low = float(row.bootstrap_ci95_low_pp)
        ci_high = float(row.bootstrap_ci95_high_pp)
        p_holm = float(row.primary_p_holm)
        effect = float(row.cohens_dz)
        positive = p_holm < alpha and ci_low > 0 and effect >= minimum_effect
        negative = p_holm < alpha and ci_high < 0 and effect <= -minimum_effect

        if positive:
            decision = "STATISTICALLY SUPERIOR"
        elif negative:
            decision = "INFERIOR"
        elif delta > 0:
            decision = "DESCRIPTIVELY BETTER"
        elif delta == 0:
            decision = "COMPARABLE"
        else:
            decision = "INSUFFICIENT EVIDENCE"

        rows.append({
            "method": first,
            "compared_with": second,
            "mean_difference_pp": delta,
            "bootstrap_ci95_low_pp": ci_low,
            "bootstrap_ci95_high_pp": ci_high,
            "primary_p_holm": p_holm,
            "cohens_dz": effect,
            "decision": decision,
            "decision_scope": "Time in Target; locked warm-start pipelines",
        })
    return pd.DataFrame(rows)
