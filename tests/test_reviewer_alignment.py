import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT / "Docs" / "Reviewer_Alignment"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ALIGNMENT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_module_8a_minimum_artifacts_exist_and_are_populated():
    names = {
        "reviewer_alignment_matrix.csv",
        "research_question_objective_map.csv",
        "hypothesis_map.csv",
        "scope_and_data_classification.md",
    }
    assert names <= {path.name for path in ALIGNMENT.iterdir()}
    assert all((ALIGNMENT / name).stat().st_size > 0 for name in names)


def test_research_questions_objectives_and_hypotheses_are_one_to_one():
    mapping = read_csv("research_question_objective_map.csv")
    hypotheses = read_csv("hypothesis_map.csv")

    assert [row["research_question_id"] for row in mapping] == ["RM1", "RM2", "RM3", "RM4"]
    assert [row["objective_id"] for row in mapping] == ["T1", "T2", "T3", "T4"]
    assert all(row["measurement_status"] == "LOCKED" for row in mapping)
    assert [row["hypothesis_id"] for row in hypotheses] == ["H1", "H2", "H3", "H4"]
    assert all(row["decision_status"].startswith("PENDING_") for row in hypotheses)


def test_reviewer_items_are_mapped_and_scope_contains_claim_guards():
    reviewer_items = read_csv("reviewer_alignment_matrix.csv")
    scope = (ALIGNMENT / "scope_and_data_classification.md").read_text(encoding="utf-8")

    assert len(reviewer_items) == 12
    assert all(row["alignment_status"] == "ALIGNED" for row in reviewer_items)
    for required_text in (
        "hortikultura generik",
        "Real/raw meteorological observations",
        "Controlled synthetic forecast proxy",
        "Virtual Garden simulated state/output",
        "Field validation has not been performed",
        "0.22 <= theta <= 0.32",
        "0–5 mm/hour",
        "H4 tidak dipaksa diterima",
    ):
        assert required_text in scope
