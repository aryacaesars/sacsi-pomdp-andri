"""Testable data access and export helpers for both dashboard versions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "Results" / "baseline_metrics.csv"
LOGS_DIR = ROOT / "Logs" / "Baselines"
FINAL_RUNS = ROOT / "Results" / "Final_Experiment" / "benchmark_2025_runs.csv"
FINAL_SUMMARY = ROOT / "Results" / "Final_Experiment" / "benchmark_2025_summary.csv"
FINAL_LOGS = ROOT / "Logs" / "Final_Experiment"
DASHBOARD_TRAJECTORIES = ROOT / "Results" / "Dashboard" / "Trajectories"
ABLATION_SUMMARY = ROOT / "Results" / "Ablation_Robustness" / "ablation_robustness_summary.csv"
FACTORIAL_STATS = ROOT / "Statistics" / "factorial_rm_anova_primary.csv"
PAIRWISE_STATS = ROOT / "Statistics" / "pairwise_sacsi_primary.csv"
STATISTICAL_FINDINGS = ROOT / "Statistics" / "statistical_findings.json"
WEATHER_2025 = ROOT / "00_Dataset" / "Processed" / "benchmark_2025.csv"
SF20 = ROOT / "00_Dataset" / "Processed" / "synthetic_forecast_sf20.csv"
DASHBOARD_RESULTS = ROOT / "Results" / "Dashboard"
CONFIRMATORY = ROOT / "Results" / "Confirmatory_10Seed"
POMDP_RESULTS = ROOT / "Results" / "POMDP_Ablation"
REWARD_RESULTS = ROOT / "Results" / "Reward_Validation"
SIMPLE_RESULTS = ROOT / "Results" / "Simple_Case_Validation"
REVIEWER_DOCS = ROOT / "Docs" / "Reviewer_Alignment"

VIRTUAL_GARDEN_METHODS = (
    "No Irrigation",
    "Fixed Schedule",
    "Threshold-Based",
    "Rule-Based Forecast-Aware",
    "Fuzzy Controller",
    "DDPG",
    "TD3",
    "SAC Basic",
    "SAC + Forecast",
    "SAC + LSTM",
    "SACSI Full",
)
CONFIRMATORY_REPLAY_METHODS = {"DDPG", "TD3"}

PAGE_NAMES = (
    "Research Design",
    "Reward Lab",
    "Simple-Case & Raw-Data Validation",
    "Fair DRL Benchmark",
    "POMDP Contribution",
    "10-Seed Confirmatory Statistics",
    "Robustness & Context Diagnostics",
    "Reviewer Evidence Matrix",
    "Reproducibility & Provenance",
)

METHOD_SUPERIORITY_PAGE = "Method Superiority Analysis"

# UI-only analytical consumers are not new evidence sources, so the locked
# Module 9A evidence registry remains unchanged.
UI_PAGE_NAMES = (
    "Virtual Garden",
    *PAGE_NAMES[:5],
    METHOD_SUPERIORITY_PAGE,
    *PAGE_NAMES[5:],
)

EVIDENCE_SPECS = (
    ("8A", "research_design", "Docs/Reviewer_Alignment/research_question_objective_map.csv", PAGE_NAMES[0]),
    ("8A", "hypotheses", "Docs/Reviewer_Alignment/hypothesis_map.csv", PAGE_NAMES[0]),
    ("8A", "reviewer_alignment", "Docs/Reviewer_Alignment/reviewer_alignment_matrix.csv", PAGE_NAMES[7]),
    ("8A", "scope", "Docs/Reviewer_Alignment/scope_and_data_classification.md", PAGE_NAMES[0]),
    ("8B", "reward_decision", "Results/Reward_Validation/reward_confirmation_decision.json", PAGE_NAMES[1]),
    ("8B", "reward_confirmation", "Results/Reward_Validation/reward_confirmation_summary.csv", PAGE_NAMES[1]),
    ("8B", "reward_pareto", "Results/Reward_Validation/reward_pareto.csv", PAGE_NAMES[1]),
    ("8C", "simple_cases", "Results/Simple_Case_Validation/simple_case_results.csv", PAGE_NAMES[2]),
    ("8C", "raw_episodes", "Results/Simple_Case_Validation/raw_episode_summary.csv", PAGE_NAMES[2]),
    ("8D", "ddpg_validation", "Results/DDPG/ddpg_validation_results.csv", PAGE_NAMES[3]),
    ("8E", "td3_validation", "Results/TD3/td3_validation_results.csv", PAGE_NAMES[3]),
    ("8F", "fairness_audit", "Results/Fair_DRL/fairness_audit.json", PAGE_NAMES[3]),
    ("8F", "fair_drl_results", "Results/Fair_DRL/fair_drl_results_2025.csv", PAGE_NAMES[3]),
    ("8G", "pomdp_manifest", "Results/POMDP_Ablation/pomdp_ablation_manifest.json", PAGE_NAMES[4]),
    ("8G", "context_interventions", "Results/POMDP_Ablation/context_intervention_results.csv", PAGE_NAMES[6]),
    ("8G", "forecast_robustness", "Results/POMDP_Ablation/forecast_robustness.csv", PAGE_NAMES[6]),
    ("8G", "sequence_sensitivity", "Results/POMDP_Ablation/sequence_sensitivity.csv", PAGE_NAMES[6]),
    ("8H", "main_results", "Results/Confirmatory_10Seed/main_10seed_results_2025.csv", PAGE_NAMES[3]),
    ("8H", "factorial_results", "Results/Confirmatory_10Seed/sac_family_10seed_factorial.csv", PAGE_NAMES[4]),
    ("8H", "friedman", "Results/Confirmatory_10Seed/friedman_results.csv", PAGE_NAMES[5]),
    ("8H", "planned_contrasts", "Results/Confirmatory_10Seed/planned_contrasts.csv", PAGE_NAMES[5]),
    ("8H", "factorial_inference", "Results/Confirmatory_10Seed/factorial_inference.csv", PAGE_NAMES[5]),
    ("8H", "statistics_summary", "Results/Confirmatory_10Seed/final_statistics_summary.json", PAGE_NAMES[5]),
    ("8H", "confirmatory_manifest", "Results/Confirmatory_10Seed/confirmatory_manifest.json", PAGE_NAMES[8]),
)

INDONESIAN = {
    "SACSI-POMDP Final Evidence Dashboard": "Dashboard Evidence Final SACSI-POMDP",
    "Reviewer-oriented evidence from Modules 8A–8H": "Evidence berorientasi reviewer dari Modul 8A–8H",
    "Dashboard Page": "Halaman Dashboard",
    "Virtual Garden": "Kebun Virtual",
    "Research Design": "Desain Penelitian",
    "Reward Lab": "Lab Reward",
    "Simple-Case & Raw-Data Validation": "Validasi Simple-Case & Raw-Data",
    "Fair DRL Benchmark": "Benchmark DRL Fair",
    "POMDP Contribution": "Kontribusi POMDP",
    "Method Superiority Analysis": "Analisis Superioritas Metode",
    "10-Seed Confirmatory Statistics": "Statistik Konfirmatori 10-Seed",
    "Robustness & Context Diagnostics": "Robustness & Diagnostik Konteks",
    "Reviewer Evidence Matrix": "Matriks Evidence Reviewer",
    "Reproducibility & Provenance": "Reproducibility & Provenance",
    "NOT READY": "BELUM SIAP",
    "READY": "SIAP",
    "Locked reward": "Reward terkunci",
    "Matched seeds": "Seed berpasangan",
    "Primary endpoint": "Endpoint primer",
    "Main confirmatory results": "Hasil konfirmatori utama",
    "Locked-pipeline claim only; unequal effective total interaction budget blocks an unqualified architecture claim.": "Klaim hanya untuk pipeline terkunci; perbedaan total budget interaksi efektif melarang klaim arsitektur tanpa kualifikasi.",
    "Exploratory reward-v4 diagnostics from Module 8G; no significance claim is inferred here.": "Diagnostik eksploratori reward-v4 dari Modul 8G; tidak ada klaim signifikansi yang ditarik di sini.",
    "All reviewer items mapped": "Seluruh item reviewer terpetakan",
    "Production evidence registry": "Registry evidence produksi",
    "No test, fixture, or smoke artifact is accepted as production evidence.": "Tidak ada artefak test, fixture, atau smoke yang diterima sebagai evidence produksi.",
    "SACSI-POMDP · Unified Final Dashboard": "SACSI-POMDP · Dashboard Final Terpadu",
    "Retrospective benchmark 2025 · 9 methods · 10 matched RL seeds · target band 0.22–0.32": "Benchmark retrospektif 2025 · 9 metode · 10 seed RL berpasangan · target band 0,22–0,32",
    "Final Experiment View": "Tampilan Eksperimen Final",
    "Single method": "Satu metode",
    "Compare 2–4 methods": "Bandingkan 2–4 metode",
    "Method": "Metode",
    "Methods": "Metode",
    "Select at least two methods.": "Pilih minimal dua metode.",
    "Matched RL seed": "Seed RL berpasangan",
    "The seed selector affects RL methods only; deterministic baselines remain one trajectory.": "Selector seed hanya memengaruhi metode RL; baseline deterministik tetap menggunakan satu trajektori.",
    "Displayed period": "Periode tampilan",
    "Select both start and end dates.": "Pilih tanggal awal dan akhir.",
    "No data in the selected period.": "Tidak ada data pada periode yang dipilih.",
    "Overview": "Ringkasan",
    "Trajectories": "Trajektori",
    "Ablation & Robustness": "Ablasi & Robustness",
    "Statistics": "Statistik",
    "Exports": "Ekspor",
    "Selected methods": "Metode dipilih",
    "Displayed hours": "Jam ditampilkan",
    "Best target occupancy": "Okupansi target terbaik",
    "Lowest irrigation": "Irigasi terendah",
    "Nine-method final registry": "Registry final sembilan metode",
    "Mean Time in Target (%)": "Rerata waktu dalam target (%)",
    "PNG: hover over a chart and click the camera icon in the Plotly toolbar.": "PNG: arahkan kursor ke grafik lalu klik ikon kamera pada toolbar Plotly.",
    "Soil moisture and target band": "Kelembapan tanah dan target band",
    "Volumetric soil moisture (m³/m³)": "Kelembapan tanah volumetrik (m³/m³)",
    "Irrigation (mm/h)": "Irigasi (mm/jam)",
    "Cumulative irrigation (mm)": "Irigasi kumulatif (mm)",
    "Actual": "Aktual",
    "Precipitation (mm/h)": "Presipitasi (mm/jam)",
    "Metrics for displayed period": "Metrik periode tampilan",
    "Context ablation": "Ablasi konteks",
    "Forecast robustness SF10–SF30": "Robustness forecast SF10–SF30",
    "Sequence sensitivity k6–k48": "Sensitivitas sekuens k6–k48",
    "Experiment": "Eksperimen",
    "Condition": "Kondisi",
    "All conditions use the same 10 locked seeds; sequence sensitivity is inference-window sensitivity without retraining.": "Semua kondisi memakai 10 seed terkunci yang sama; sensitivitas sekuens mengubah jendela inferensi tanpa training ulang.",
    "Primary endpoint: Time in Target (%)": "Endpoint primer: Waktu dalam target (%)",
    "Pre-specified paired SACSI comparisons": "Perbandingan berpasangan SACSI yang ditetapkan sebelumnya",
    "SACSI superiority is supported for all pre-specified comparisons.": "Superioritas SACSI didukung pada seluruh perbandingan yang ditetapkan sebelumnya.",
    "SACSI superiority over all variants is not supported. Technical validity remains separate from superiority.": "Superioritas SACSI atas semua varian tidak didukung. Validitas teknis tetap dipisahkan dari superioritas performa.",
    "Deterministic baselines are trajectory references and are not assigned fake stochastic seeds.": "Baseline deterministik adalah referensi trajektori dan tidak diberi seed stokastik palsu.",
    "Export selected experiment view": "Ekspor tampilan eksperimen terpilih",
    "Download CSV": "Unduh CSV",
    "Download XLSX": "Unduh XLSX",
    "Download JSON": "Unduh JSON",
    "Download ZIP": "Unduh ZIP",
    "PNG export is available from the camera icon on every chart toolbar.": "Ekspor PNG tersedia melalui ikon kamera pada toolbar setiap grafik.",
    "Forecast display: SF-20 h+1 controlled proxy, not an archived operational forecast. Checkpoint selection used validation 2024 only; 2025 was retrospective evaluation.": "Tampilan forecast memakai proxy terkontrol SF-20 h+1, bukan forecast operasional terarsip. Pemilihan checkpoint hanya memakai validasi 2024; data 2025 digunakan untuk evaluasi retrospektif.",
    "rank": "peringkat",
    "method": "metode",
    "n_runs": "jumlah_run",
    "time_in_target_pct_mean": "rerata_waktu_dalam_target_pct",
    "time_in_target_pct_std": "sd_waktu_dalam_target_pct",
    "total_irrigation_mm_mean": "rerata_irigasi_mm",
    "total_irrigation_mm_std": "sd_irigasi_mm",
    "rmse_band_mean": "rerata_rmse_band",
    "total_irrigation_mm": "total_irigasi_mm",
    "time_in_target_pct": "waktu_dalam_target_pct",
    "violation_rate_pct": "tingkat_pelanggaran_pct",
    "deficit_rate_pct": "tingkat_defisit_pct",
    "surplus_rate_pct": "tingkat_surplus_pct",
    "action_smoothness": "kehalusan_aksi",
    "runoff_total_mm": "total_runoff_mm",
    "drainage_total_mm": "total_drainase_mm",
    "max_abs_mass_balance_error_mm": "galat_neraca_massa_maks_mm",
    "experiment": "eksperimen",
    "condition": "kondisi",
    "n_seeds": "jumlah_seed",
    "effect": "efek",
    "mean_effect": "rerata_efek",
    "p_value": "nilai_p",
    "significant_alpha_0_05": "signifikan_alpha_0_05",
    "comparison": "perbandingan",
    "compared_with": "dibandingkan_dengan",
    "decision": "keputusan",
    "decision_scope": "ruang_lingkup_keputusan",
    "objective": "objektif",
    "direction": "arah",
    "leader": "pemimpin",
    "value": "nilai",
    "evidence_level": "tingkat_evidence",
    "claim_guard": "batas_klaim",
    "pareto_status": "status_pareto",
    "pareto_role": "peran_pareto",
    "dominated_by": "didominasi_oleh",
    "pareto_non_dominated": "pareto_non_dominated",
    "primary_p_holm": "p_holm_primer",
    "cohens_dz": "cohen_dz",
    "mean_difference_pp": "selisih_rerata_pp",
    "significant_holm_0_05": "signifikan_holm_0_05",
}


def translate(text: str, language: str) -> str:
    return INDONESIAN.get(text, text) if language == "Bahasa Indonesia" else text


def localize_columns(frame: pd.DataFrame, language: str) -> pd.DataFrame:
    return frame.rename(columns=lambda column: translate(str(column), language))


def controller_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def method_slug(name: str) -> str:
    return name.lower().replace(" + ", "_plus_").replace("-", "_").replace(" ", "_")


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


def load_logs(split: str, controllers: list[str]) -> pd.DataFrame:
    frames = []
    for controller in controllers:
        path = LOGS_DIR / f"{split}_{controller_slug(controller)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run baseline benchmark first: missing {path.name}")
        frames.append(pd.read_csv(path, parse_dates=["timestamp"]))
    return pd.concat(frames, ignore_index=True)


def load_final_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(FINAL_RUNS)
    summary = pd.read_csv(FINAL_SUMMARY).sort_values("rank").reset_index(drop=True)
    if len(runs) != 45 or summary["method"].nunique() != 9:
        raise ValueError("Final registry must contain 45 runs and nine methods")
    return runs, summary


def load_selected_registry(methods: list[str], seed: int) -> pd.DataFrame:
    runs, _ = load_final_registry()
    baseline = runs["method_type"].eq("baseline") & runs["method"].isin(methods)
    rl = (
        runs["method_type"].eq("rl")
        & runs["method"].isin(methods)
        & pd.to_numeric(runs["seed"], errors="coerce").eq(seed)
    )
    selected = runs.loc[baseline | rl].copy()
    if set(selected["method"]) != set(methods):
        raise ValueError("Selected method/seed is missing from the final registry")
    return selected


def load_virtual_garden_registry() -> pd.DataFrame:
    historical, _ = load_final_registry()
    historical = historical[["method", "method_type", "seed"]].copy()
    historical["trajectory_protocol"] = "historical_sprint13"

    confirmatory = pd.read_csv(CONFIRMATORY / "main_10seed_results_2025.csv")
    confirmatory = confirmatory.loc[
        confirmatory["model"].isin(CONFIRMATORY_REPLAY_METHODS), ["model", "seed"]
    ].rename(columns={"model": "method"})
    confirmatory["method_type"] = "rl"
    confirmatory["trajectory_protocol"] = "confirmatory_reward_v4"

    registry = pd.concat((historical, confirmatory), ignore_index=True)
    if (
        len(registry) != 65
        or set(registry["method"]) != set(VIRTUAL_GARDEN_METHODS)
        or registry.loc[registry["method_type"] == "rl"].duplicated(["method", "seed"]).any()
    ):
        raise ValueError("Virtual Garden registry must contain 11 methods and all RL seeds")
    return registry


def load_selected_virtual_registry(methods: list[str], seed: int) -> pd.DataFrame:
    runs = load_virtual_garden_registry()
    baseline = runs["method_type"].eq("baseline") & runs["method"].isin(methods)
    rl = (
        runs["method_type"].eq("rl")
        & runs["method"].isin(methods)
        & pd.to_numeric(runs["seed"], errors="coerce").eq(seed)
    )
    selected = runs.loc[baseline | rl].copy()
    if set(selected["method"]) != set(methods):
        raise ValueError("Selected method/seed is missing from the Virtual Garden registry")
    return selected


def _weather_context() -> pd.DataFrame:
    weather = pd.read_csv(WEATHER_2025, usecols=["timestamp", "precipitation_mm"], parse_dates=["timestamp"])
    weather = weather.rename(columns={"precipitation_mm": "actual_precipitation_mm"})
    forecast = pd.read_csv(
        SF20,
        usecols=["timestamp", "forecast_precipitation_mm"],
        parse_dates=["timestamp"],
    )
    forecast = forecast.loc[forecast["timestamp"].dt.year == 2025]
    return weather.merge(forecast, on="timestamp", how="left", validate="one_to_one")


def load_final_logs(methods: list[str], seed: int) -> pd.DataFrame:
    registry = load_selected_virtual_registry(methods, seed)
    frames = []
    for row in registry.itertuples(index=False):
        suffix = f"_seed{seed}" if row.method_type == "rl" else ""
        directory = (
            DASHBOARD_TRAJECTORIES
            if row.method in CONFIRMATORY_REPLAY_METHODS
            else FINAL_LOGS
        )
        path = directory / f"benchmark_2025_{method_slug(row.method)}{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Build Virtual Garden trajectories first: missing {path.name}"
            )
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frame["controller"] = row.method
        frame["display_seed"] = str(seed) if row.method_type == "rl" else "deterministic"
        frames.append(frame)
    logs = pd.concat(frames, ignore_index=True)
    logs = logs.merge(_weather_context(), on="timestamp", how="left", validate="many_to_one")
    if logs[["actual_precipitation_mm", "forecast_precipitation_mm"]].isna().any().any():
        raise ValueError("Weather enrichment failed on final common support")
    logs = logs.sort_values(["controller", "timestamp"]).reset_index(drop=True)
    logs["cumulative_irrigation_mm"] = logs.groupby("controller")["irrigation_mm"].cumsum()
    if not logs.groupby("controller")["timestamp"].nunique().eq(8_760).all():
        raise ValueError("Every selected method must have 8,760 benchmark hours")
    return logs


def load_ablation_summary() -> pd.DataFrame:
    return pd.read_csv(ABLATION_SUMMARY)


def load_statistics() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return (
        pd.read_csv(CONFIRMATORY / "factorial_inference.csv"),
        pd.read_csv(CONFIRMATORY / "planned_contrasts.csv").query(
            "analysis_family == 'main_benchmark'"
        ).reset_index(drop=True),
        json.loads((CONFIRMATORY / "final_statistics_summary.json").read_text(encoding="utf-8")),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_hash_variants(content: bytes) -> set[str]:
    normalized = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    variants = (content, normalized, normalized.replace(b"\n", b"\r\n"))
    return {hashlib.sha256(value).hexdigest() for value in variants}


def _evidence_hash_matches(path: Path, expected: str) -> bool:
    if path.suffix.lower() in {".csv", ".json", ".md"}:
        return expected in _evidence_hash_variants(path.read_bytes())
    return sha256_file(path) == expected


def _synthetic_fixture_path(path: Path) -> bool:
    lowered = path.as_posix().lower()
    return (
        "/tests/" in f"/{lowered}/"
        or "/fixtures/" in f"/{lowered}/"
        or "smoke" in path.name.lower()
        or path.name.lower().startswith("test_")
    )


def build_result_registry(specs=EVIDENCE_SPECS, root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for module, evidence_type, relative, page in specs:
        path = root / relative
        rejected = _synthetic_fixture_path(Path(relative))
        status, digest, size, row_count = "NOT READY", None, None, None
        if rejected:
            status = "REJECTED_SYNTHETIC"
        elif path.is_file():
            try:
                if path.suffix.lower() == ".csv":
                    row_count = len(pd.read_csv(path))
                elif path.suffix.lower() == ".json":
                    json.loads(path.read_text(encoding="utf-8"))
                    row_count = 1
                elif not path.read_text(encoding="utf-8").strip():
                    raise ValueError("empty evidence file")
                digest, size, status = sha256_file(path), path.stat().st_size, "READY"
            except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
                status = "NOT READY"
        rows.append({
            "module": module,
            "evidence_type": evidence_type,
            "evidence_path": Path(relative).as_posix(),
            "dashboard_page": page,
            "sha256": digest,
            "size_bytes": size,
            "row_count": row_count,
            "readiness_status": status,
            "production_evidence": status == "READY",
            "synthetic_fixture": rejected,
        })
    registry = pd.DataFrame(rows)
    required = {
        "module", "evidence_type", "evidence_path", "dashboard_page", "sha256",
        "readiness_status", "production_evidence", "synthetic_fixture",
    }
    if required.difference(registry.columns) or registry["evidence_path"].duplicated().any():
        raise ValueError("Invalid or duplicate dashboard result registry")
    return registry


REVIEWER_EVIDENCE = {
    "RA-01": (PAGE_NAMES[0], ("Docs/Reviewer_Alignment/research_question_objective_map.csv",), "MAPPED"),
    "RA-02": (PAGE_NAMES[0], ("Docs/Reviewer_Alignment/research_question_objective_map.csv",), "MAPPED"),
    "RA-03": (PAGE_NAMES[4], ("Docs/Reviewer_Alignment/scope_and_data_classification.md", "Results/POMDP_Ablation/context_intervention_results.csv"), "SUPPORTED_WITH_GUARD"),
    "RA-04": (PAGE_NAMES[3], ("Docs/Reviewer_Alignment/scope_and_data_classification.md", "Results/Fair_DRL/fairness_audit.json"), "SUPPORTED"),
    "RA-05": (PAGE_NAMES[3], ("Results/DDPG/ddpg_validation_results.csv", "Results/TD3/td3_validation_results.csv", "Results/Confirmatory_10Seed/main_10seed_results_2025.csv"), "SUPPORTED"),
    "RA-06": (PAGE_NAMES[4], ("Results/POMDP_Ablation/context_intervention_results.csv", "Results/Confirmatory_10Seed/factorial_inference.csv"), "PARTIALLY_SUPPORTED"),
    "RA-07": (PAGE_NAMES[2], ("Docs/Reviewer_Alignment/scope_and_data_classification.md", "Results/Simple_Case_Validation/simple_case_results.csv"), "SUPPORTED_WITH_SCOPE_GUARD"),
    "RA-08": (PAGE_NAMES[2], ("Docs/Reviewer_Alignment/scope_and_data_classification.md", "Results/Simple_Case_Validation/raw_episode_summary.csv"), "SUPPORTED_WITH_DATA_CLASSIFICATION"),
    "RA-09": (PAGE_NAMES[1], ("Results/Reward_Validation/reward_pareto.csv", "Results/Confirmatory_10Seed/main_10seed_results_2025.csv"), "SUPPORTED_WITH_TRADEOFF_GUARD"),
    "RA-10": (PAGE_NAMES[3], ("Results/Fair_DRL/fairness_audit.json", "Results/Confirmatory_10Seed/main_10seed_results_2025.csv", "Results/Confirmatory_10Seed/sac_family_10seed_factorial.csv"), "SUPPORTED_WITH_BUDGET_GUARD"),
    "RA-11": (PAGE_NAMES[0], ("Docs/Reviewer_Alignment/scope_and_data_classification.md", "Results/Confirmatory_10Seed/final_statistics_summary.json"), "NO_FIELD_CLAIM"),
    "RA-12": (PAGE_NAMES[5], ("Docs/Reviewer_Alignment/hypothesis_map.csv", "Results/Confirmatory_10Seed/final_statistics_summary.json"), "NULL_RESULTS_RETAINED"),
}


def build_reviewer_evidence_matrix(registry: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(REVIEWER_DOCS / "reviewer_alignment_matrix.csv")
    readiness = registry.set_index("evidence_path")["readiness_status"].to_dict()
    rows = []
    for item in source.itertuples(index=False):
        page, evidence, claim_status = REVIEWER_EVIDENCE[item.reviewer_item_id]
        statuses = [readiness.get(path, "NOT READY") for path in evidence]
        rows.append({
            "reviewer_item_id": item.reviewer_item_id,
            "reviewer_question_or_input": item.alignment_topic,
            "module_source": item.target_modules,
            "evidence_file": "|".join(evidence),
            "dashboard_page": page,
            "claim_status": claim_status,
            "readiness_status": "READY" if all(value == "READY" for value in statuses) else "NOT READY",
        })
    matrix = pd.DataFrame(rows)
    if set(matrix["reviewer_item_id"]) != set(REVIEWER_EVIDENCE) or matrix.isna().any().any():
        raise ValueError("Reviewer evidence matrix is not 100% mapped")
    return matrix


def build_dashboard_release() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    DASHBOARD_RESULTS.mkdir(parents=True, exist_ok=True)
    registry = build_result_registry()
    matrix = build_reviewer_evidence_matrix(registry)
    pages_mapped = set(registry["dashboard_page"]) | set(matrix["dashboard_page"])
    ready = (
        registry["readiness_status"].eq("READY").all()
        and matrix["readiness_status"].eq("READY").all()
        and set(PAGE_NAMES).issubset(pages_mapped)
        and not registry["synthetic_fixture"].any()
    )
    registry_path = DASHBOARD_RESULTS / "result_registry.csv"
    matrix_path = DASHBOARD_RESULTS / "reviewer_evidence_matrix.csv"
    registry.to_csv(registry_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    metadata = {
        "module": "9A",
        "status": "READY" if ready else "NOT READY",
        "page_map": list(PAGE_NAMES),
        "source_of_truth": "Results/Confirmatory_10Seed",
        "result_registry_rows": len(registry),
        "reviewer_items_mapped": len(matrix),
        "reviewer_mapping_pct": float(matrix["readiness_status"].eq("READY").mean() * 100),
        "synthetic_production_evidence_count": int(registry["synthetic_fixture"].sum()),
        "historical_sprint13_excluded_from_final_claims": True,
        "result_registry_sha256": sha256_file(registry_path),
        "reviewer_evidence_matrix_sha256": sha256_file(matrix_path),
        "confirmatory_manifest_sha256": sha256_file(CONFIRMATORY / "confirmatory_manifest.json"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (DASHBOARD_RESULTS / "dashboard_release_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return registry, matrix, metadata


def load_dashboard_release() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = (
        DASHBOARD_RESULTS / "result_registry.csv",
        DASHBOARD_RESULTS / "reviewer_evidence_matrix.csv",
        DASHBOARD_RESULTS / "dashboard_release_metadata.json",
    )
    if not all(path.is_file() for path in paths):
        return pd.DataFrame(), pd.DataFrame(), {"status": "NOT READY"}
    registry = pd.read_csv(paths[0])
    matrix = pd.read_csv(paths[1])
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    mismatches = []
    for row in registry.itertuples(index=False):
        path = ROOT / row.evidence_path
        if not path.is_file() or not _evidence_hash_matches(path, row.sha256):
            mismatches.append(row.evidence_path)
    if (
        mismatches
        or not registry["readiness_status"].eq("READY").all()
        or registry["synthetic_fixture"].astype(bool).any()
        or not matrix["readiness_status"].eq("READY").all()
    ):
        metadata = metadata | {"status": "NOT READY", "hash_mismatches": mismatches}
    return registry, matrix, metadata


def load_research_design() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(REVIEWER_DOCS / "research_question_objective_map.csv"),
        pd.read_csv(REVIEWER_DOCS / "hypothesis_map.csv"),
    )


def load_reward_evidence() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    return (
        json.loads((REWARD_RESULTS / "reward_confirmation_decision.json").read_text(encoding="utf-8")),
        pd.read_csv(REWARD_RESULTS / "reward_confirmation_summary.csv"),
        pd.read_csv(REWARD_RESULTS / "reward_pareto.csv"),
    )


def load_simple_case_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(SIMPLE_RESULTS / "simple_case_results.csv"),
        pd.read_csv(SIMPLE_RESULTS / "raw_episode_summary.csv"),
    )


def load_confirmatory_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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
        raise ValueError("Confirmatory source-of-truth reconciliation failed")
    return (
        main,
        factorial,
        pd.read_csv(CONFIRMATORY / "friedman_results.csv"),
        pd.read_csv(CONFIRMATORY / "planned_contrasts.csv"),
        pd.read_csv(CONFIRMATORY / "factorial_inference.csv"),
        json.loads((CONFIRMATORY / "final_statistics_summary.json").read_text(encoding="utf-8")),
    )


def summarize_runs(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    summary = frame.groupby(group, sort=False)[
        [
            "time_in_target_pct",
            "total_irrigation_mm",
            "rmse_band",
            "violation_rate_pct",
            "action_smoothness",
        ]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def load_robustness_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(POMDP_RESULTS / "context_intervention_results.csv"),
        pd.read_csv(POMDP_RESULTS / "forecast_robustness.csv"),
        pd.read_csv(POMDP_RESULTS / "sequence_sensitivity.csv"),
    )


def filter_dates(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end_exclusive)].copy()


def export_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def export_json(frame: pd.DataFrame) -> bytes:
    return frame.to_json(orient="records", date_format="iso", indent=2).encode("utf-8")


def export_xlsx(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="simulation")
    return output.getvalue()


def export_xlsx_bundle(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=name[:31])
    return output.getvalue()


def export_json_bundle(frames: dict[str, pd.DataFrame], metadata: dict) -> bytes:
    payload = {
        "metadata": metadata,
        **{name: json.loads(frame.to_json(orient="records", date_format="iso")) for name, frame in frames.items()},
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def export_experiment_zip(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def png_chart_config(filename: str) -> dict:
    """Browser-side PNG export; avoids a server Chrome/Kaleido dependency."""
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png", "filename": filename, "height": 720, "width": 1280, "scale": 2,
        },
    }
