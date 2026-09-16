"""Build the auditable SAC Basic checkpoint registry from run metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result_dir = ROOT / "Results" / "SAC_Basic"
    checkpoint_dir = ROOT / "Checkpoints" / "SAC_Basic"
    rows = []
    for metadata_path in sorted(result_dir.glob("sac_basic_seed*_reward_v2_training_ep100_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = checkpoint_dir / metadata_path.name.replace("_metadata.json", ".pt")
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        metrics = metadata["validation_metrics"]
        rows.append({
            "model": metadata["model"],
            "seed": metadata["seed"],
            "device": metadata["device"],
            "reward_version": metadata["reward_version"],
            "selected_episode": metadata["selected_episode"],
            "validation_gate": metadata["validation_gate"],
            "total_irrigation_mm": metrics["total_irrigation_mm"],
            "time_in_target_pct": metrics["time_in_target_pct"],
            "violation_rate_pct": metrics["violation_rate_pct"],
            "deficit_rate_pct": metrics["deficit_rate_pct"],
            "surplus_rate_pct": metrics["surplus_rate_pct"],
            "mean_soil_moisture": metrics["mean_soil_moisture"],
            "max_abs_mass_balance_error_mm": metrics["max_abs_mass_balance_error_mm"],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })
    registry = pd.DataFrame(rows).sort_values("seed")
    registry.to_csv(result_dir / "validation_registry.csv", index=False)
    print(registry.to_string(index=False))


if __name__ == "__main__":
    main()
