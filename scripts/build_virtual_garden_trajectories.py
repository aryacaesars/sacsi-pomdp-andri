"""Build deployment-ready DDPG/TD3 replay trajectories from frozen checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from Dashboard.data import DASHBOARD_TRAJECTORIES, method_slug, sha256_file  # noqa: E402
from ddpg import DDPGAgent, evaluate as evaluate_ddpg  # noqa: E402
from evaluation.final_benchmark import FORMAL_METRICS, validate_log_support  # noqa: E402
from sac_basic import SACIrrigationEnv  # noqa: E402
from td3 import TD3Agent, evaluate as evaluate_td3  # noqa: E402


DATA = ROOT / "00_Dataset" / "Processed"
CONFIRMATORY = ROOT / "Results" / "Confirmatory_10Seed"
MODELS = {
    "DDPG": (DDPGAgent, evaluate_ddpg),
    "TD3": (TD3Agent, evaluate_td3),
}


def build_virtual_garden_trajectories() -> pd.DataFrame:
    benchmark = pd.read_csv(DATA / "benchmark_2025.csv", parse_dates=["timestamp"])
    expected = pd.read_csv(CONFIRMATORY / "main_10seed_results_2025.csv")
    expected = expected.loc[expected["model"].isin(MODELS)].copy()
    if len(expected) != 20 or expected.duplicated(["model", "seed"]).any():
        raise ValueError("Expected ten confirmatory DDPG and TD3 rows")

    DASHBOARD_TRAJECTORIES.mkdir(parents=True, exist_ok=True)
    timestamps = pd.DatetimeIndex(benchmark["timestamp"])
    rows = []
    for result in expected.sort_values(["model", "seed"]).itertuples(index=False):
        agent_class, evaluator = MODELS[result.model]
        checkpoint = ROOT / Path(result.checkpoint)
        if sha256_file(checkpoint) != result.checkpoint_sha256:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")

        agent, _ = agent_class.load(checkpoint, device="cpu")
        environment = SACIrrigationEnv(
            benchmark,
            DATA / "normalizer.json",
            episode_length=len(benchmark),
            seed=int(result.seed),
        )
        trajectory, metrics = evaluator(agent, environment)
        validate_log_support(trajectory, timestamps)
        mismatches = [
            metric for metric in FORMAL_METRICS
            if not np.isclose(
                float(metrics[metric]),
                float(getattr(result, metric)),
                rtol=1e-10,
                atol=1e-10,
            )
        ]
        if mismatches:
            raise ValueError(
                f"{result.model} seed {result.seed} disagrees with confirmatory metrics: {mismatches}"
            )

        trajectory["seed"] = int(result.seed)
        path = DASHBOARD_TRAJECTORIES / (
            f"benchmark_2025_{method_slug(result.model)}_seed{int(result.seed)}.csv"
        )
        trajectory.to_csv(path, index=False)
        rows.append({
            "method": result.model,
            "seed": int(result.seed),
            "trajectory_path": path.relative_to(ROOT).as_posix(),
            "trajectory_rows": len(trajectory),
            "trajectory_sha256": sha256_file(path),
            "checkpoint": result.checkpoint,
            "checkpoint_sha256": result.checkpoint_sha256,
            "reward_version": result.reward_version,
            "evaluation_split": result.evaluation_split,
            "metrics_reconciled": True,
        })

    registry = pd.DataFrame(rows)
    registry.to_csv(
        DASHBOARD_TRAJECTORIES.parent / "virtual_garden_trajectory_registry.csv",
        index=False,
    )
    return registry


if __name__ == "__main__":
    registry = build_virtual_garden_trajectories()
    print(json.dumps({
        "status": "READY",
        "methods": sorted(registry["method"].unique()),
        "trajectories": len(registry),
        "rows_per_trajectory": [
            int(value) for value in sorted(registry["trajectory_rows"].unique())
        ],
        "metrics_reconciled": bool(registry["metrics_reconciled"].all()),
    }, indent=2))
