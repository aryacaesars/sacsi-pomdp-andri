"""Small Pareto helper for controller water-versus-control trade-offs."""

from __future__ import annotations

import pandas as pd


def pareto_frontier(
    frame: pd.DataFrame,
    *,
    method_column: str,
    maximize: str,
    minimize: str,
) -> pd.DataFrame:
    required = {method_column, maximize, minimize}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Pareto columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Pareto inputs cannot contain missing values")

    result = frame.copy()
    dominated_by = []
    # Four confirmatory methods: the direct O(n²) definition is clearest.
    for candidate in result.itertuples(index=False):
        candidate_values = candidate._asdict()
        dominators = []
        for challenger in result.itertuples(index=False):
            challenger_values = challenger._asdict()
            if challenger_values[method_column] == candidate_values[method_column]:
                continue
            no_worse = (
                challenger_values[maximize] >= candidate_values[maximize]
                and challenger_values[minimize] <= candidate_values[minimize]
            )
            strictly_better = (
                challenger_values[maximize] > candidate_values[maximize]
                or challenger_values[minimize] < candidate_values[minimize]
            )
            if no_worse and strictly_better:
                dominators.append(str(challenger_values[method_column]))
        dominated_by.append(" | ".join(dominators))

    result["dominated_by"] = dominated_by
    result["pareto_non_dominated"] = result["dominated_by"].eq("")
    result["pareto_status"] = result["pareto_non_dominated"].map({
        True: "PARETO NON-DOMINATED",
        False: "DOMINATED",
    })
    return result
