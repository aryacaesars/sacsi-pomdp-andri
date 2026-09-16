"""Resumable 10-seed expansion for the four matched RL families."""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
LOG_DIR = ROOT / "Logs" / "Expanded_Training"
STATUS_PATH = ROOT / "Results" / "expanded_training_status.csv"

PHASES = (
    (
        "SAC Basic", "train_sac_basic.py", "build_sac_basic_registry.py",
        lambda seed: ROOT / "Results" / "SAC_Basic" / f"sac_basic_seed{seed}_reward_v2_training_ep100_metadata.json",
        lambda seed: ROOT / "Checkpoints" / "SAC_Basic" / f"sac_basic_seed{seed}_reward_v2_training_ep100.pt",
        ("--episodes", "100", "--validation-interval", "10"),
    ),
    (
        "SAC + Forecast", "train_sac_forecast.py", "build_sac_forecast_registry.py",
        lambda seed: ROOT / "Results" / "SAC_Forecast" / f"sac_forecast_seed{seed}_reward_v2_sf20_training_ep100_metadata.json",
        lambda seed: ROOT / "Checkpoints" / "SAC_Forecast" / f"sac_forecast_seed{seed}_reward_v2_sf20_training_ep100.pt",
        ("--episodes", "100", "--validation-interval", "10"),
    ),
    (
        "SAC + LSTM", "train_sac_lstm.py", "build_sac_lstm_registry.py",
        lambda seed: ROOT / "Results" / "SAC_LSTM" / f"sac_lstm_seed{seed}_reward_v2_rrws_k24_training_ep10_metadata.json",
        lambda seed: ROOT / "Checkpoints" / "SAC_LSTM" / f"sac_lstm_seed{seed}_reward_v2_rrws_k24_training_ep10.pt",
        ("--episodes", "10", "--validation-interval", "2"),
    ),
    (
        "SACSI Full", "train_sacsi_full.py", "build_sacsi_registry.py",
        lambda seed: ROOT / "Results" / "SACSI_Full" / f"sacsi_full_seed{seed}_reward_v2_sf20_rrws_k24_training_ep10_metadata.json",
        lambda seed: ROOT / "Checkpoints" / "SACSI_Full" / f"sacsi_full_seed{seed}_reward_v2_sf20_rrws_k24_training_ep10.pt",
        ("--episodes", "10", "--validation-interval", "2"),
    ),
)


def write_status(rows: list[dict[str, object]]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "seed", "status", "started", "finished", "log"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model, trainer, registry, metadata_for, checkpoint_for, arguments in PHASES:
        for seed in SEEDS:
            metadata, checkpoint = metadata_for(seed), checkpoint_for(seed)
            log_path = LOG_DIR / f"{model.lower().replace(' ', '_').replace('+', 'plus')}_seed{seed}.log"
            if metadata.exists() and checkpoint.exists():
                rows.append({
                    "model": model, "seed": seed, "status": "complete-existing",
                    "started": "", "finished": "", "log": str(log_path.relative_to(ROOT)),
                })
                write_status(rows)
                continue
            started = datetime.now(timezone.utc).isoformat()
            print(f"START {model} seed={seed} {started}", flush=True)
            command = [
                sys.executable, str(ROOT / "scripts" / trainer), "--seed", str(seed), *arguments,
            ]
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            complete = result.returncode == 0 and metadata.exists() and checkpoint.exists()
            finished = datetime.now(timezone.utc).isoformat()
            status = "complete" if complete else f"failed-{result.returncode}"
            rows.append({
                "model": model, "seed": seed, "status": status,
                "started": started, "finished": finished, "log": str(log_path.relative_to(ROOT)),
            })
            write_status(rows)
            print(f"END {model} seed={seed} status={status} {finished}", flush=True)
        registry_log = LOG_DIR / f"{registry}.log"
        with registry_log.open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / registry)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
    print("EXPANDED TRAINING FINISHED", flush=True)


if __name__ == "__main__":
    main()
