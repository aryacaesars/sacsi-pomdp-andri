"""Build the SACSI Full validation registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result_dir = ROOT / "Results" / "SACSI_Full"
    checkpoint_dir = ROOT / "Checkpoints" / "SACSI_Full"
    rows = []
    pattern = "sacsi_full_seed*_reward_v2_sf20_rrws_k24_training_ep10_metadata.json"
    for metadata_path in sorted(result_dir.glob(pattern)):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = checkpoint_dir / metadata_path.name.replace("_metadata.json", ".pt")
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        metrics = metadata["validation_metrics"]
        history_active = metadata["history_residual_norm"] > 0 and metadata["zero_history_action_delta_mm"] > 0
        forecast_active = metadata["forecast_residual_norm"] > 0 and metadata["zero_forecast_action_delta_mm"] > 0
        rows.append({
            "model": metadata["model"], "seed": metadata["seed"], "device": metadata["device"],
            "selected_episode": metadata["selected_episode"],
            "validation_gate": metadata["validation_gate"],
            "history_active": history_active, "forecast_active": forecast_active,
            "history_residual_norm": metadata["history_residual_norm"],
            "forecast_residual_norm": metadata["forecast_residual_norm"],
            "zero_history_action_delta_mm": metadata["zero_history_action_delta_mm"],
            "reverse_history_action_delta_mm": metadata["reverse_history_action_delta_mm"],
            "zero_forecast_action_delta_mm": metadata["zero_forecast_action_delta_mm"],
            "zero_context_action_delta_mm": metadata["zero_context_action_delta_mm"],
            "total_irrigation_mm": metrics["total_irrigation_mm"],
            "time_in_target_pct": metrics["time_in_target_pct"],
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
