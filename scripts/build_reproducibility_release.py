"""Build and verify the Module 9D dissertation reproducibility freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Dashboard.data import load_dashboard_release, sha256_file  # noqa: E402


RESULTS = ROOT / "Results" / "Reproducibility_Freeze"
PACKAGE = ROOT / "SACSI_POMDP_Reproducibility_v1.0"
CONFIRMATORY = ROOT / "Results" / "Confirmatory_10Seed"
INTENDED_TAG = "v1.0-dissertation-freeze"


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _verified_release_metadata(path: Path, expected_module: str) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("module") != expected_module or metadata.get("status") != "READY":
        raise RuntimeError(f"Module {expected_module} release is NOT READY")
    for record in metadata["outputs"].values():
        source = ROOT / record["path"]
        if not source.is_file() or sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"Module {expected_module} hash mismatch: {record['path']}")
    return metadata


def _verify_confirmatory_manifest() -> dict:
    manifest = json.loads((CONFIRMATORY / "confirmatory_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_CONFIRMATORY":
        raise RuntimeError("Module 8H confirmatory manifest is not complete")
    for name, record in manifest["artifacts"].items():
        path = CONFIRMATORY / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Frozen 8H artifact hash mismatch: {name}")
    return manifest


def _checkpoint_registry(source_name: str, label: str, expected_models: set[str]) -> pd.DataFrame:
    source = pd.read_csv(CONFIRMATORY / source_name)
    model_column = "model" if "model" in source else "variant"
    if (
        len(source) != 40
        or set(source[model_column]) != expected_models
        or not source.groupby(model_column)["seed"].nunique().eq(10).all()
        or source.duplicated([model_column, "seed"]).any()
    ):
        raise RuntimeError(f"{label} checkpoint registry is not a complete 40-slot design")

    rows = []
    for row in source.to_dict(orient="records"):
        checkpoint = ROOT / row["checkpoint"]
        exists = checkpoint.is_file()
        actual_hash = sha256_file(checkpoint) if exists else ""
        verified = exists and actual_hash == row["checkpoint_sha256"]
        rows.append({
            "registry": label,
            "slot_id": f"{row[model_column]}|seed={int(row['seed'])}",
            "model": row[model_column],
            "seed": int(row["seed"]),
            "reward_version": row["reward_version"],
            "training_device": row["training_device"],
            "validation_gate": bool(row["validation_gate"]),
            "losses_finite": bool(row["losses_finite"]),
            "effective_total_interactions": int(row["effective_total_interactions"]),
            "checkpoint": row["checkpoint"],
            "expected_sha256": row["checkpoint_sha256"],
            "actual_sha256": actual_hash,
            "checkpoint_exists": exists,
            "hash_verified": verified,
            "freeze_status": "FROZEN" if verified else "NOT_READY",
        })
    registry = pd.DataFrame(rows)
    if (
        registry["reward_version"].ne("reward_v4").any()
        or registry["training_device"].ne("cuda").any()
        or not registry["losses_finite"].all()
        or not registry["hash_verified"].all()
    ):
        raise RuntimeError(f"{label} checkpoint integrity gate failed")
    return registry


def _master_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(CONFIRMATORY / "main_10seed_results_2025.csv")
    factorial = pd.read_csv(CONFIRMATORY / "sac_family_10seed_factorial.csv")
    if (
        len(main) != 40
        or len(factorial) != 40
        or main.duplicated(["model", "seed"]).any()
        or factorial.duplicated(["variant", "seed"]).any()
        or set(main["reward_version"]) != {"reward_v4"}
        or set(factorial["reward_version"]) != {"reward_v4"}
    ):
        raise RuntimeError("Frozen result tables are not valid 40-row matched designs")

    weather = pd.read_csv(ROOT / "00_Dataset" / "Processed" / "benchmark_2025.csv", usecols=["timestamp"])
    period_start = str(pd.to_datetime(weather["timestamp"]).min())
    period_end = str(pd.to_datetime(weather["timestamp"]).max())
    family_map = {
        "DDPG": "DDPG", "TD3": "TD3", "SAC": "SAC", "SACSI-POMDP": "SACSI_POMDP",
        "SAC Basic": "SAC", "SAC + Forecast": "SAC_FORECAST",
        "SAC + LSTM": "SAC_LSTM", "SACSI Full": "SACSI_POMDP",
    }

    rows = []

    def append(source: pd.DataFrame, model_column: str, experiment_family: str) -> None:
        for item in source.to_dict(orient="records"):
            model = item[model_column]
            forecast = bool(item.get("forecast", model in {"SACSI-POMDP"}))
            memory = bool(item.get("memory", model in {"SACSI-POMDP"}))
            interactions = int(item["effective_total_interactions"])
            rows.append({
                "experiment_id": f"8H_{experiment_family}_{model.lower().replace(' ', '_').replace('+', 'plus')}_seed{int(item['seed'])}",
                "experiment_family": experiment_family,
                "module": "8H",
                "algorithm_family": family_map[model],
                "model": model,
                "seed": int(item["seed"]),
                "split": item["evaluation_split"],
                "period_start": period_start,
                "period_end": period_end,
                "reward_version": item["reward_version"],
                "virtual_garden_version": "field_capacity_0.35",
                "observation_version": "current8+history24x8+forecast3" if forecast and memory else "current8+forecast3" if forecast else "current8+history24x8" if memory else "current8",
                "forecast_enabled": forecast,
                "memory_enabled": memory,
                "forecast_protocol": "SF-20_h1_controlled_proxy" if forecast else "none",
                "forecast_error": 0.20 if forecast else None,
                "forecast_horizon": 1 if forecast else None,
                "sequence_length": 24 if memory else None,
                "checkpoint": item["checkpoint"],
                "checkpoint_hash": item["checkpoint_sha256"],
                "validation_gate": bool(item["checkpoint_validation_gate"]),
                "steps": len(weather),
                "environment_interactions": interactions,
                "training_budget_version": "anchor6720+adaptation6720" if interactions > 6720 else "fair6720",
                "total_water_mm": item["total_irrigation_mm"],
                "time_in_target_pct": item["time_in_target_pct"],
                "violation_rate_pct": item["violation_rate_pct"],
                "deficit_rate_pct": item["deficit_rate_pct"],
                "surplus_rate_pct": item["surplus_rate_pct"],
                "rmse_band": item["rmse_band"],
                "action_smoothness": item["action_smoothness"],
                "mean_soil_moisture": item["mean_soil_moisture"],
                "runoff_total_mm": item["runoff_total_mm"],
                "drainage_total_mm": item["drainage_total_mm"],
                "reward_mean": float(item["cumulative_reward"]) / len(weather),
                "mass_balance_error": item["max_abs_mass_balance_error_mm"],
                "git_commit": "not_recorded_at_training",
                "source_result_status": item["result_status"],
                "result_status": "FROZEN",
            })

    append(main, "model", "main_confirmatory")
    append(factorial, "variant", "factorial_confirmatory")
    master = pd.DataFrame(rows)
    if len(master) != 80 or master["experiment_id"].duplicated().any() or master.isna().all(axis=1).any():
        raise RuntimeError("master_results.csv reconciliation failed")
    return master, main, factorial


def _expanded_result_registry(dashboard_registry: pd.DataFrame, releases: tuple[dict, ...]) -> pd.DataFrame:
    rows = dashboard_registry.to_dict(orient="records")
    existing = {row["evidence_path"] for row in rows}

    def add(module: str, evidence_type: str, relative: str, digest: str | None = None) -> None:
        if relative in existing:
            return
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing registered result: {relative}")
        synthetic = "smoke" in path.name.lower() or "fixture" in path.as_posix().lower()
        rows.append({
            "module": module,
            "evidence_type": evidence_type,
            "evidence_path": relative,
            "dashboard_page": "Release package",
            "sha256": digest or sha256_file(path),
            "size_bytes": path.stat().st_size,
            "row_count": len(pd.read_csv(path)) if path.suffix.lower() == ".csv" else 1 if path.suffix.lower() == ".json" else None,
            "readiness_status": "REJECTED_SYNTHETIC" if synthetic else "READY",
            "production_evidence": not synthetic,
            "synthetic_fixture": synthetic,
        })
        existing.add(relative)

    for metadata in releases:
        for name, record in metadata["outputs"].items():
            add(metadata["module"], Path(name).stem, record["path"], record["sha256"])
    manifest = json.loads((CONFIRMATORY / "confirmatory_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["artifacts"].items():
        add("8H", Path(name).stem, f"Results/Confirmatory_10Seed/{name}", record["sha256"])
    for name in ("master_results.csv", "checkpoint_registry.csv", "sac_family_checkpoint_registry.csv"):
        add("9D", Path(name).stem, f"Results/Reproducibility_Freeze/{name}")

    registry = pd.DataFrame(rows)
    if (
        registry["evidence_path"].duplicated().any()
        or registry["readiness_status"].ne("READY").any()
        or registry["synthetic_fixture"].astype(bool).any()
    ):
        raise RuntimeError("Final result registry contains missing, duplicate, or synthetic evidence")
    return registry


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".git"),
    )


def _dataset_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "00_Dataset").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        name = path.name.lower()
        classification = "controlled_synthetic_forecast_proxy" if "synthetic_forecast" in name else "processed_derivative" if "Processed/" in relative else "user_supplied_raw_or_archived"
        rows.append({
            "dataset_path": relative,
            "classification": classification,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "upstream_provider": "NOT_RECORDED",
            "redistribution_license": "NOT_RECORDED",
            "included_in_publication_package": False,
        })
    return pd.DataFrame(rows)


def verify_artifact_manifest(package: Path = PACKAGE) -> list[str]:
    manifest_path = package / "artifact_manifest_sha256.csv"
    if not manifest_path.is_file():
        return ["artifact_manifest_sha256.csv missing"]
    mismatches = []
    for row in pd.read_csv(manifest_path).itertuples(index=False):
        path = package / row.relative_path
        if not path.is_file() or path.stat().st_size != row.size_bytes or sha256_file(path) != row.sha256:
            mismatches.append(row.relative_path)
    return mismatches


def git_tag_gate() -> dict:
    try:
        head = _git("rev-parse", "HEAD")
        tags = set(_git("tag", "--points-at", "HEAD").splitlines())
        clean = not _git("status", "--porcelain", "--untracked-files=normal")
        error = None
    except (OSError, subprocess.CalledProcessError) as exc:
        head, tags, clean, error = "UNAVAILABLE", set(), False, type(exc).__name__
    return {
        "head": head,
        "worktree_clean": clean,
        "tag": INTENDED_TAG,
        "tag_points_at_head": INTENDED_TAG in tags,
        "passed": clean and INTENDED_TAG in tags,
        "inspection_error": error,
    }


def build_reproducibility_release() -> dict:
    dashboard_registry, _, dashboard = load_dashboard_release()
    if dashboard.get("status") != "READY":
        raise RuntimeError("Module 9A dashboard release is NOT READY")
    dissertation = _verified_release_metadata(
        ROOT / "Results" / "Dissertation_Evidence" / "dissertation_release_metadata.json", "9B"
    )
    defense = _verified_release_metadata(
        ROOT / "Results" / "Reviewer_Defense" / "defense_release_metadata.json", "9C"
    )
    _verify_confirmatory_manifest()

    RESULTS.mkdir(parents=True, exist_ok=True)
    main_checkpoints = _checkpoint_registry(
        "main_checkpoint_registry.csv", "main", {"DDPG", "TD3", "SAC", "SACSI-POMDP"}
    )
    family_checkpoints = _checkpoint_registry(
        "sac_family_checkpoint_registry.csv", "sac_family",
        {"SAC Basic", "SAC + Forecast", "SAC + LSTM", "SACSI Full"},
    )
    master, main, factorial = _master_results()
    main_checkpoints.to_csv(RESULTS / "checkpoint_registry.csv", index=False)
    family_checkpoints.to_csv(RESULTS / "sac_family_checkpoint_registry.csv", index=False)
    master.to_csv(RESULTS / "master_results.csv", index=False)

    result_registry = _expanded_result_registry(
        dashboard_registry, (dissertation, defense)
    )
    result_registry.to_csv(RESULTS / "result_registry.csv", index=False)

    claim_release = pd.read_csv(ROOT / "Results" / "Reviewer_Defense" / "claim_to_evidence_matrix.csv")
    claim_release["freeze_version"] = "v1.0"
    claim_release["freeze_status"] = "LOCKED"
    claim_release.to_csv(RESULTS / "claim_release_matrix.csv", index=False)

    package_map = pd.DataFrame((
        ("environment", "README.md|requirements.txt|environment.yml|LICENSE_or_data_notice.md", "package root", "included", "Setup, scope, and rights notice"),
        ("source", "src/|scripts/|configs/|tests/", "src/|scripts/|configs/|tests/", "included", "Executable implementation and tests"),
        ("data", "00_Dataset/", "data_provenance/dataset_manifest_sha256.csv", "hashes_only", "Raw data excluded pending license clearance"),
        ("checkpoints", "Results/Confirmatory_10Seed/*checkpoint_registry.csv", "checkpoint_registry.csv|publication_package/sac_family_checkpoint_registry.csv", "registry_only", "Checkpoint paths and verified hashes; binaries remain in research repository"),
        ("results", "Results/Confirmatory_10Seed/", "master_results.csv|statistics/|tables/", "included", "Frozen seed-level results and statistics"),
        ("dashboard", "Dashboard/|Results/Dashboard/", "dashboard/", "included", "Read-only evidence dashboard"),
        ("dissertation", "Docs/Dissertation/|Results/Dissertation_Evidence/", "dissertation_evidence/", "included", "Reconciled drafts and matrices"),
        ("defense", "Docs/Reviewer_Defense/|Results/Reviewer_Defense/", "publication_package/", "included", "Reviewer responses and defense package"),
    ), columns=("category", "source", "package_destination", "distribution", "purpose"))
    package_map.to_csv(RESULTS / "publication_package_map.csv", index=False)

    PACKAGE.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "LICENSE_or_data_notice.md", "requirements.txt", "environment.yml", "CHANGELOG.md"):
        shutil.copy2(ROOT / name, PACKAGE / name)
    for name in ("src", "scripts", "tests", "configs"):
        _copy_tree(ROOT / name, PACKAGE / name)

    notebooks = PACKAGE / "notebooks"
    notebooks.mkdir(parents=True, exist_ok=True)
    (notebooks / "README.md").write_text(
        "# Notebooks\n\nThe frozen release uses the versioned scripts in `scripts/` as canonical executable workflows. No additional notebook-only logic is required.\n",
        encoding="utf-8",
    )

    provenance = PACKAGE / "data_provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    datasets = _dataset_manifest()
    datasets.to_csv(provenance / "dataset_manifest_sha256.csv", index=False)
    shutil.copy2(ROOT / "00_Dataset" / "Processed" / "data_audit_report.csv", provenance / "data_audit_report.csv")
    shutil.copy2(ROOT / "Docs" / "Reviewer_Alignment" / "scope_and_data_classification.md", provenance / "scope_and_data_classification.md")
    shutil.copy2(ROOT / "LICENSE_or_data_notice.md", provenance / "LICENSE_or_data_notice.md")

    shutil.copy2(RESULTS / "checkpoint_registry.csv", PACKAGE / "checkpoint_registry.csv")
    shutil.copy2(RESULTS / "result_registry.csv", PACKAGE / "result_registry.csv")
    shutil.copy2(RESULTS / "master_results.csv", PACKAGE / "master_results.csv")

    statistics = PACKAGE / "statistics"
    statistics.mkdir(parents=True, exist_ok=True)
    for name in (
        "friedman_results.csv", "planned_contrasts.csv", "holm_adjusted_results.csv",
        "bootstrap_ci.csv", "factorial_inference.csv", "final_statistics_summary.json",
        "confirmatory_manifest.json",
    ):
        shutil.copy2(CONFIRMATORY / name, statistics / name)

    figures = PACKAGE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _copy_tree(ROOT / "Figures", figures)

    tables = PACKAGE / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (CONFIRMATORY / "main_10seed_summary.csv", "main_10seed_summary.csv"),
        (CONFIRMATORY / "sac_family_10seed_summary.csv", "sac_family_10seed_summary.csv"),
        (CONFIRMATORY / "planned_contrasts.csv", "planned_contrasts.csv"),
        (CONFIRMATORY / "factorial_inference.csv", "factorial_inference.csv"),
        (ROOT / "Results" / "Dissertation_Evidence" / "hypothesis_decision_table.csv", "hypothesis_decision_table.csv"),
        (RESULTS / "claim_release_matrix.csv", "claim_release_matrix.csv"),
    ):
        shutil.copy2(source, tables / target)

    dashboard_package = PACKAGE / "dashboard"
    _copy_tree(ROOT / "Dashboard", dashboard_package)
    _copy_tree(ROOT / "Results" / "Dashboard", dashboard_package / "evidence")

    dissertation_package = PACKAGE / "dissertation_evidence"
    _copy_tree(ROOT / "Results" / "Dissertation_Evidence", dissertation_package / "matrices")
    _copy_tree(ROOT / "Docs" / "Dissertation", dissertation_package / "drafts")

    publication = PACKAGE / "publication_package"
    _copy_tree(ROOT / "Results" / "Reviewer_Defense", publication / "matrices")
    _copy_tree(ROOT / "Docs" / "Reviewer_Defense", publication / "defense_docs")
    shutil.copy2(RESULTS / "sac_family_checkpoint_registry.csv", publication / "sac_family_checkpoint_registry.csv")
    shutil.copy2(RESULTS / "publication_package_map.csv", publication / "publication_package_map.csv")
    shutil.copy2(RESULTS / "claim_release_matrix.csv", publication / "claim_release_matrix.csv")

    required = (
        "README.md", "LICENSE_or_data_notice.md", "requirements.txt", "environment.yml",
        "configs", "src", "scripts", "tests", "notebooks", "data_provenance",
        "checkpoint_registry.csv", "result_registry.csv", "master_results.csv", "statistics",
        "figures", "tables", "dashboard", "dissertation_evidence", "publication_package",
        "artifact_manifest_sha256.csv", "CHANGELOG.md", "FREEZE_METADATA.json",
    )
    (PACKAGE / "artifact_manifest_sha256.csv").touch()
    (PACKAGE / "FREEZE_METADATA.json").touch()
    inventory_rows = []
    for relative in required:
        path = PACKAGE / relative
        inventory_rows.append({
            "path": relative,
            "required": True,
            "exists": path.exists(),
            "type": "directory" if path.is_dir() else "file",
            "sha256": sha256_file(path) if path.is_file() and relative not in {"artifact_manifest_sha256.csv", "FREEZE_METADATA.json"} else None,
            "readiness_status": "READY" if path.exists() else "NOT_READY",
        })
    submission_inventory = pd.DataFrame(inventory_rows)
    if submission_inventory["readiness_status"].ne("READY").any():
        raise RuntimeError("Submission inventory is incomplete")
    submission_inventory.to_csv(RESULTS / "submission_inventory.csv", index=False)
    shutil.copy2(RESULTS / "submission_inventory.csv", publication / "submission_inventory.csv")

    prohibited_parts = {"__pycache__", ".pytest_cache", ".git", ".env"}
    prohibited_names = {"Historical Weather 2021-2025.csv", "Historical Forecast 2021-2025.csv"}
    package_files = []
    for path in PACKAGE.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(PACKAGE)
        if prohibited_parts.intersection(relative.parts) or path.suffix.lower() == ".pyc" or path.name in prohibited_names:
            raise RuntimeError(f"Unsanitized package artifact: {relative.as_posix()}")
        if "smoke" in path.name.lower() and path.suffix.lower() in {".csv", ".json", ".pt"}:
            raise RuntimeError(f"Synthetic/smoke production artifact detected: {relative.as_posix()}")
        if relative.as_posix() not in {"artifact_manifest_sha256.csv", "FREEZE_METADATA.json"}:
            package_files.append(path)

    manifest_rows = [{
        "relative_path": path.relative_to(PACKAGE).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in sorted(package_files)]
    artifact_manifest = pd.DataFrame(manifest_rows)
    artifact_manifest.to_csv(PACKAGE / "artifact_manifest_sha256.csv", index=False)
    if verify_artifact_manifest():
        raise RuntimeError("Generated artifact manifest failed verification")
    shutil.copy2(PACKAGE / "artifact_manifest_sha256.csv", RESULTS / "artifact_manifest_sha256.csv")

    previous_path = RESULTS / "FREEZE_METADATA_v1.0.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    manifest_hash = sha256_file(PACKAGE / "artifact_manifest_sha256.csv")
    metadata = {
        "module": "9D",
        "status": "READY_FOR_GIT_TAG",
        "freeze_version": "v1.0",
        "freeze_level": "DISSERTATION_FREEZE",
        "result_freeze": "PASS",
        "dissertation_freeze": "PASS",
        "publication_package": "GENERATED_WITH_RAW_DATA_EXCLUDED",
        "raw_data_redistribution": "BLOCKED_PENDING_LICENSE_CLEARANCE",
        "intended_git_tag": INTENDED_TAG,
        "git_tag_gate": {
            "required": True,
            "verification": "python scripts/build_reproducibility_release.py --verify-tag",
        },
        "main_result_rows": len(main),
        "factorial_result_rows": len(factorial),
        "master_result_rows": len(master),
        "main_checkpoint_slots": len(main_checkpoints),
        "sac_family_checkpoint_slots": len(family_checkpoints),
        "result_registry_rows": len(result_registry),
        "artifact_manifest_rows": len(artifact_manifest),
        "synthetic_production_evidence_count": int(result_registry["synthetic_fixture"].astype(bool).sum()),
        "dashboard_release_status": dashboard["status"],
        "dissertation_release_status": dissertation["status"],
        "defense_release_status": defense["status"],
        "manifest_sha256": manifest_hash,
        "generated_utc": previous.get("generated_utc") if previous.get("manifest_sha256") == manifest_hash else datetime.now(timezone.utc).isoformat(),
    }
    (PACKAGE / "FREEZE_METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    previous_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-tag", action="store_true", help="Require the intended tag at a clean HEAD")
    args = parser.parse_args()
    release = build_reproducibility_release()
    print(json.dumps(release, indent=2))
    if args.verify_tag and not git_tag_gate()["passed"]:
        raise SystemExit(f"Git tag gate failed: commit the frozen snapshot and create {INTENDED_TAG}")
