from io import BytesIO
from zipfile import ZipFile

from datetime import date
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Dashboard.data import (
    PAGE_NAMES,
    build_result_registry,
    export_csv,
    export_experiment_zip,
    export_json,
    export_json_bundle,
    export_xlsx,
    export_xlsx_bundle,
    filter_dates,
    load_ablation_summary,
    load_confirmatory_evidence,
    load_dashboard_release,
    load_final_logs,
    load_final_registry,
    load_logs,
    load_metrics,
    load_statistics,
    localize_columns,
    png_chart_config,
    summarize_runs,
    translate,
)
from Dashboard.pages import RENDERERS


def test_dashboard_data_filter_and_exports() -> None:
    metrics = load_metrics()
    assert set(metrics["controller"]) == {
        "No Irrigation", "Fixed Schedule", "Threshold-Based",
        "Rule-Based Forecast-Aware", "Fuzzy Controller",
    }
    logs = load_logs("benchmark_2025", ["No Irrigation", "Threshold-Based"])
    filtered = filter_dates(logs, date(2025, 1, 1), date(2025, 1, 2))
    assert len(filtered) == 96
    assert filtered["timestamp"].min() == pd.Timestamp("2025-01-01 00:00:00")
    assert filtered["timestamp"].max() == pd.Timestamp("2025-01-02 23:00:00")
    assert export_csv(filtered).startswith(b"timestamp,")
    assert export_json(filtered).startswith(b"[")
    assert export_xlsx(filtered).startswith(b"PK")


def test_final_dashboard_registry_logs_and_experiment_exports() -> None:
    runs, summary = load_final_registry()
    assert len(runs) == 45
    assert summary["method"].nunique() == 9

    logs = load_final_logs(["Threshold-Based", "SACSI Full"], seed=11)
    filtered = filter_dates(logs, date(2025, 1, 1), date(2025, 1, 2))
    assert len(filtered) == 96
    assert filtered["controller"].nunique() == 2
    assert not filtered[["actual_precipitation_mm", "forecast_precipitation_mm"]].isna().any().any()
    assert filtered.groupby("controller")["cumulative_irrigation_mm"].is_monotonic_increasing.all()

    ablation = load_ablation_summary()
    factorial, pairwise, findings = load_statistics()
    assert set(ablation["experiment"]) == {
        "context_ablation", "forecast_robustness", "sequence_sensitivity",
    }
    assert len(factorial) == len(pairwise) == 3
    assert findings["main_locked_pipeline_superiority_supported"] is True
    assert findings["unqualified_equal_total_budget_superiority_claim_released"] is False

    frames = {"trajectory": filtered, "summary": summary}
    xlsx = export_xlsx_bundle(frames)
    json_data = export_json_bundle(frames, {"seed": 11})
    archive = export_experiment_zip({"experiment.xlsx": xlsx, "experiment.json": json_data})
    assert xlsx.startswith(b"PK") and json_data.startswith(b"{") and archive.startswith(b"PK")
    with ZipFile(BytesIO(archive)) as zipped:
        assert set(zipped.namelist()) == {"experiment.xlsx", "experiment.json"}

    config = png_chart_config("soil_moisture")
    assert config["toImageButtonOptions"]["format"] == "png"
    assert config["toImageButtonOptions"]["filename"] == "soil_moisture"


def test_dashboard_indonesian_translation_and_column_labels() -> None:
    assert translate("Overview", "Bahasa Indonesia") == "Ringkasan"
    assert translate("SACSI Full", "Bahasa Indonesia") == "SACSI Full"
    frame = pd.DataFrame({"method": ["SACSI Full"], "time_in_target_pct": [55.0]})
    assert localize_columns(frame, "Bahasa Indonesia").columns.tolist() == [
        "metode", "waktu_dalam_target_pct",
    ]


def test_module_9a_release_is_ready_and_reconciles_confirmatory_results() -> None:
    registry, matrix, metadata = load_dashboard_release()
    assert metadata["status"] == "READY"
    assert len(registry) == 24
    assert len(matrix) == 12
    assert matrix["readiness_status"].eq("READY").all()
    assert not registry["synthetic_fixture"].astype(bool).any()
    assert set(RENDERERS) == set(PAGE_NAMES)

    main, factorial, friedman, planned, effects, findings = load_confirmatory_evidence()
    main_summary = summarize_runs(main, "model")
    sacsi = main_summary.set_index("model").loc["SACSI-POMDP"]
    assert sacsi["time_in_target_pct_mean"] == main.loc[
        main["model"] == "SACSI-POMDP", "time_in_target_pct"
    ].mean()
    assert len(main) == len(factorial) == 40
    assert len(friedman) == 1 and len(planned) == 6 and len(effects) == 3
    assert findings["claim_scope"] == "locked warm-start training pipelines only"


def test_module_9a_missing_and_synthetic_evidence_are_never_ready() -> None:
    specs = (
        ("X", "fixture", "Results/test_fixture.csv", "Research Design"),
        ("X", "missing", "Results/missing.csv", "Research Design"),
    )
    registry = build_result_registry(specs=specs, root=ROOT).set_index("evidence_type")
    assert registry.loc["fixture", "readiness_status"] == "REJECTED_SYNTHETIC"
    assert not bool(registry.loc["fixture", "production_evidence"])
    assert registry.loc["missing", "readiness_status"] == "NOT READY"
