import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_reproducibility_release import (
    PACKAGE,
    RESULTS,
    build_reproducibility_release,
    verify_artifact_manifest,
)


def test_module_9d_builds_a_complete_hash_verified_freeze_package() -> None:
    metadata = build_reproducibility_release()
    assert metadata["status"] == "READY_FOR_GIT_TAG"
    assert metadata["result_freeze"] == metadata["dissertation_freeze"] == "PASS"
    assert metadata["main_result_rows"] == metadata["factorial_result_rows"] == 40
    assert metadata["master_result_rows"] == 80
    assert metadata["main_checkpoint_slots"] == metadata["sac_family_checkpoint_slots"] == 40
    assert metadata["synthetic_production_evidence_count"] == 0
    assert not verify_artifact_manifest()

    checkpoints = pd.read_csv(RESULTS / "checkpoint_registry.csv")
    results = pd.read_csv(RESULTS / "result_registry.csv")
    master = pd.read_csv(RESULTS / "master_results.csv")
    assert checkpoints["hash_verified"].all()
    assert checkpoints.groupby("model")["seed"].nunique().eq(10).all()
    assert results["readiness_status"].eq("READY").all()
    assert not results["synthetic_fixture"].astype(bool).any()
    assert master["result_status"].eq("FROZEN").all()
    assert not master.duplicated(["experiment_id"]).any()


def test_module_9d_package_excludes_raw_data_and_cache_files() -> None:
    metadata = json.loads((RESULTS / "FREEZE_METADATA_v1.0.json").read_text(encoding="utf-8"))
    paths = [path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*") if path.is_file()]
    assert metadata["raw_data_redistribution"] == "BLOCKED_PENDING_LICENSE_CLEARANCE"
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in paths)
    assert "Historical Weather 2021-2025.csv" not in {Path(path).name for path in paths}
    assert "Historical Forecast 2021-2025.csv" not in {Path(path).name for path in paths}
    assert (PACKAGE / "data_provenance" / "dataset_manifest_sha256.csv").is_file()
    assert (PACKAGE / "LICENSE_or_data_notice.md").is_file()
