import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Dashboard.data import sha256_file
from scripts.build_defense_package import DOCS, RESULTS, build_defense_package


def test_module_9c_package_has_complete_reviewer_and_defense_coverage() -> None:
    metadata = build_defense_package()
    assert metadata["status"] == "READY"
    assert metadata["reviewer_items_answered"] == 12
    assert metadata["reviewer_coverage_pct"] == 100.0
    assert metadata["defense_questions"] >= 11
    assert metadata["mandatory_topics_covered"] == 11

    responses = pd.read_csv(RESULTS / "reviewer_response_matrix.csv")
    qa = pd.read_csv(RESULTS / "defense_qa_bank.csv")
    claims = pd.read_csv(RESULTS / "claim_to_evidence_matrix.csv")
    assert responses["readiness_status"].eq("READY").all()
    assert qa["readiness_status"].eq("READY").all()
    assert claims["readiness_status"].eq("READY").all()
    assert responses[["short_answer_30_60_sec", "technical_answer", "mathematical_support", "evidence_source", "claim_guard"]].notna().all().all()
    assert claims.loc[claims["claim_topic"].eq("Standalone forecast benefit"), "release_status"].item() == "NOT_RELEASED"
    assert claims.loc[claims["claim_topic"].eq("Field effectiveness"), "release_status"].item() == "NOT_RELEASED"

    for record in metadata["outputs"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]


def test_module_9c_red_flags_and_defense_card_retain_claim_guards() -> None:
    metadata = json.loads((RESULTS / "defense_release_metadata.json").read_text(encoding="utf-8"))
    red_flags = (DOCS / "red_flag_wording.md").read_text(encoding="utf-8")
    card = (DOCS / "one_page_defense_card.md").read_text(encoding="utf-8")
    assert metadata["unsupported_claim_warnings"] >= 13
    assert "Forecast meningkatkan performa SACSI" in red_flags
    assert "DDPG, TD3, dan SAC terbukti berbeda signifikan" in red_flags
    assert "belum ada field validation" in card
    assert "equal-total-budget" in card
