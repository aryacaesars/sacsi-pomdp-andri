import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Dashboard.data import sha256_file
from scripts.build_dissertation_release import DOCS, RESULTS, build_dissertation_release


def test_module_9b_release_reconciles_frozen_evidence() -> None:
    metadata = build_dissertation_release()
    assert metadata["status"] == metadata["dashboard_status"] == "READY"
    assert metadata["hypothesis_decisions"] == 4
    assert metadata["unsupported_superiority_claims"] == 0

    insertions = pd.read_csv(RESULTS / "result_insertion_matrix.csv")
    decisions = pd.read_csv(RESULTS / "hypothesis_decision_table.csv").set_index("hypothesis_id")
    claims = pd.read_csv(RESULTS / "claim_matrix.csv")
    main = pd.read_csv(ROOT / "Results" / "Confirmatory_10Seed" / "main_10seed_results_2025.csv")

    sacsi_exact = float(insertions.set_index("metric_or_statement").loc[
        "SACSI-POMDP confirmatory summary", "source_value_exact"
    ])
    assert sacsi_exact == main.loc[main["model"].eq("SACSI-POMDP"), "time_in_target_pct"].mean()
    assert decisions.loc["H2", "decision"] == "INCONCLUSIVE_DIRECT_TEST_NOT_EXPORTED"
    assert decisions.loc["H4", "decision"] == "REJECT_H0_LOCKED_PIPELINE_SCOPE"
    assert claims.loc[claims["claim_topic"].eq("Standalone forecast benefit"), "release_status"].item() == "NOT_RELEASED"
    assert insertions["readiness_status"].eq("READY").all()

    for record in metadata["outputs"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]
    assert (DOCS / "chapter_results_draft.md").read_text(encoding="utf-8").count("## 5.") == 9
    assert (DOCS / "chapter_discussion_draft.md").read_text(encoding="utf-8").count("## 6.") == 9


def test_module_9b_metadata_is_valid_json() -> None:
    metadata = json.loads((RESULTS / "dissertation_release_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "READY"
    assert metadata["result_insertions"] >= 30
