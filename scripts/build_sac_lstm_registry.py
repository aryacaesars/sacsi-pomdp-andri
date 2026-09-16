"""Build the SAC + LSTM RRWS validation registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result_dir = ROOT / "Results" / "SAC_LSTM"
    checkpoint_dir = ROOT / "Checkpoints" / "SAC_LSTM"
    rows = []
    pattern = "sac_lstm_seed*_reward_v2_rrws_k24_training_ep10_metadata.json"
    for metadata_path in sorted(result_dir.glob(pattern)):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = checkpoint_dir / metadata_path.name.replace("_metadata.json", ".pt")
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        metrics = metadata["validation_metrics"]
        rows.append({
            "model": metadata["model"], "seed": metadata["seed"],
            "device": metadata["device"], "sequence_length": metadata["sequence_length"],
            "selected_episode": metadata["selected_episode"],
            "validation_gate": metadata["validation_gate"],
            "memory_active": metadata["context_residual_norm"] > 0
                and metadata["zero_history_action_delta_mm"] > 0,
            "context_residual_norm": metadata["context_residual_norm"],
            "zero_history_action_delta_mm": metadata["zero_history_action_delta_mm"],
            "reverse_history_action_delta_mm": metadata["reverse_history_action_delta_mm"],
            "shuffle_history_action_delta_mm": metadata["shuffle_history_action_delta_mm"],
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
