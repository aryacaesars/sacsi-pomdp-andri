from io import BytesIO
from zipfile import ZipFile

from datetime import date
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Dashboard.data import (
    METHOD_SUPERIORITY_PAGE,
    PAGE_NAMES,
    UI_PAGE_NAMES,
    _evidence_hash_variants,
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
    load_virtual_garden_registry,
    localize_columns,
    png_chart_config,
    summarize_runs,
    translate,
)
from Dashboard.views import RENDERERS
from Dashboard.views.virtual_garden import _moisture_gauge, _ordered_snapshot, _status


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


def test_dashboard_evidence_hash_allows_only_newline_conversion() -> None:
    windows_hashes = _evidence_hash_variants(b"value\r\n1\r\n")
    linux_hashes = _evidence_hash_variants(b"value\n1\n")
    changed_hashes = _evidence_hash_variants(b"value\n2\n")

    assert windows_hashes == linux_hashes
    assert windows_hashes.isdisjoint(changed_hashes)


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


def test_virtual_garden_registry_includes_confirmatory_ddpg_and_td3() -> None:
    registry = load_virtual_garden_registry()
    assert len(registry) == 65
    assert registry["method"].nunique() == 11
    assert set(registry.loc[registry["trajectory_protocol"] == "confirmatory_reward_v4", "method"]) == {
        "DDPG", "TD3",
    }

    logs = load_final_logs(["DDPG", "TD3"], seed=11)
    filtered = filter_dates(logs, date(2025, 1, 1), date(2025, 1, 2))
    assert len(filtered) == 96
    assert set(filtered["controller"]) == {"DDPG", "TD3"}


def test_dashboard_indonesian_translation_and_column_labels() -> None:
    assert translate("Virtual Garden", "Bahasa Indonesia") == "Kebun Virtual"
    assert translate(METHOD_SUPERIORITY_PAGE, "Bahasa Indonesia") == "Analisis Superioritas Metode"
    assert translate("Overview", "Bahasa Indonesia") == "Ringkasan"
    assert translate("SACSI Full", "Bahasa Indonesia") == "SACSI Full"
    frame = pd.DataFrame({"method": ["SACSI Full"], "time_in_target_pct": [55.0]})
    assert localize_columns(frame, "Bahasa Indonesia").columns.tolist() == [
        "metode", "waktu_dalam_target_pct",
    ]


def test_virtual_garden_status_and_gauge_follow_locked_target_band() -> None:
    assert _status(0.21, "English") == ("🥀", "Below target")
    assert _status(0.27, "Bahasa Indonesia") == ("🌱", "Di dalam target")
    assert _status(0.33, "English") == ("🌿", "Above target")
    gauge = _moisture_gauge("SACSI Full", 0.27, "English")
    assert gauge.data[0]["value"] == 0.27
    assert gauge.data[0]["gauge"]["bar"]["color"] == "#2E7D32"
    assert gauge.data[0]["gauge"]["steps"][1]["range"] == (0.22, 0.32)
    assert _moisture_gauge("dry", 0.21, "English").data[0]["gauge"]["bar"]["color"] == "#E65100"
    assert _moisture_gauge("wet", 0.33, "English").data[0]["gauge"]["bar"]["color"] == "#1565C0"


def test_virtual_garden_snapshot_follows_compare_selection_order() -> None:
    timestamp = pd.Timestamp("2025-01-01 00:00:00")
    view = pd.DataFrame({
        "timestamp": [timestamp] * 3,
        "controller": ["TD3", "SACSI Full", "DDPG"],
        "theta": [0.31, 0.27, 0.23],
    })

    snapshot = _ordered_snapshot(view, timestamp, ["SACSI Full", "DDPG", "TD3"])

    assert snapshot["controller"].tolist() == ["SACSI Full", "DDPG", "TD3"]
    assert snapshot["theta"].tolist() == [0.27, 0.23, 0.31]


def test_custom_navigation_has_no_streamlit_auto_page_modules() -> None:
    auto_pages = ROOT / "Dashboard" / "pages"
    assert not auto_pages.exists() or not list(auto_pages.glob("*.py"))


def test_module_9a_release_is_ready_and_reconciles_confirmatory_results() -> None:
    registry, matrix, metadata = load_dashboard_release()
    assert metadata["status"] == "READY"
    assert len(registry) == 24
    assert len(matrix) == 12
    assert matrix["readiness_status"].eq("READY").all()
    assert not registry["synthetic_fixture"].astype(bool).any()
    assert UI_PAGE_NAMES[0] == "Virtual Garden"
    assert METHOD_SUPERIORITY_PAGE in UI_PAGE_NAMES
    assert set(RENDERERS) == set(UI_PAGE_NAMES)

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
