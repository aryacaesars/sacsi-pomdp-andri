"""Build Module 9B dissertation drafts directly from the frozen evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Dashboard.data import (  # noqa: E402
    load_confirmatory_evidence,
    load_dashboard_release,
    load_reward_evidence,
    load_robustness_evidence,
    load_simple_case_evidence,
    sha256_file,
    summarize_runs,
)


RESULTS = ROOT / "Results" / "Dissertation_Evidence"
DOCS = ROOT / "Docs" / "Dissertation"


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def build_dissertation_release() -> dict:
    registry, reviewer, dashboard = load_dashboard_release()
    if dashboard.get("status") != "READY":
        raise RuntimeError("Module 9A evidence release is NOT READY")

    source_hash = registry.set_index("evidence_path")["sha256"].to_dict()

    def require_source(path: str) -> str:
        digest = source_hash.get(path)
        if not digest:
            raise RuntimeError(f"Unregistered dissertation source: {path}")
        return digest

    scope_path = "Docs/Reviewer_Alignment/scope_and_data_classification.md"
    rq_path = "Docs/Reviewer_Alignment/research_question_objective_map.csv"
    reward_path = "Results/Reward_Validation/reward_confirmation_decision.json"
    simple_path = "Results/Simple_Case_Validation/simple_case_results.csv"
    raw_path = "Results/Simple_Case_Validation/raw_episode_summary.csv"
    fair_path = "Results/Fair_DRL/fair_drl_results_2025.csv"
    fair_audit_path = "Results/Fair_DRL/fairness_audit.json"
    pomdp_path = "Results/POMDP_Ablation/pomdp_ablation_manifest.json"
    context_path = "Results/POMDP_Ablation/context_intervention_results.csv"
    forecast_path = "Results/POMDP_Ablation/forecast_robustness.csv"
    sequence_path = "Results/POMDP_Ablation/sequence_sensitivity.csv"
    main_path = "Results/Confirmatory_10Seed/main_10seed_results_2025.csv"
    factorial_path = "Results/Confirmatory_10Seed/sac_family_10seed_factorial.csv"
    friedman_path = "Results/Confirmatory_10Seed/friedman_results.csv"
    contrasts_path = "Results/Confirmatory_10Seed/planned_contrasts.csv"
    effects_path = "Results/Confirmatory_10Seed/factorial_inference.csv"
    statistics_path = "Results/Confirmatory_10Seed/final_statistics_summary.json"

    reward, _, _ = load_reward_evidence()
    simple, raw = load_simple_case_evidence()
    main, factorial, friedman, planned, effects, statistics = load_confirmatory_evidence()
    context, forecast, sequence = load_robustness_evidence()
    fair = pd.read_csv(ROOT / fair_path)
    fair_audit = json.loads((ROOT / fair_audit_path).read_text(encoding="utf-8"))
    pomdp = json.loads((ROOT / pomdp_path).read_text(encoding="utf-8"))

    main_summary = summarize_runs(main, "model").set_index("model")
    factorial_summary = summarize_runs(factorial, "variant").set_index("variant")
    fair_summary = fair.groupby("algorithm_family", sort=False)[
        ["time_in_target_pct", "total_irrigation_mm"]
    ].agg(["mean", "std"])
    main_planned = planned.loc[planned["analysis_family"].eq("main_benchmark")].copy()

    if (
        len(main) != 40
        or len(factorial) != 40
        or len(main_planned) != 3
        or len(effects) != 3
        or not simple["passed"].astype(bool).all()
        or len(reviewer) != 12
    ):
        raise RuntimeError("Frozen evidence does not satisfy the Module 9B design")

    insertion_rows: list[dict] = []

    def insertion(
        insertion_id: str,
        section: str,
        item: str,
        exact: object,
        display: str,
        source: str,
        field: str,
        page: str,
    ) -> None:
        insertion_rows.append({
            "insertion_id": insertion_id,
            "chapter_section": section,
            "metric_or_statement": item,
            "source_value_exact": exact,
            "display_text": display,
            "source_file": source,
            "source_field": field,
            "source_sha256": require_source(source),
            "dashboard_page": page,
            "readiness_status": "READY",
        })

    insertion("R01", "5.1", "Data split", "2021-2023|2024|2025", "training 2021–2023; validation 2024; retrospective benchmark 2025", fair_audit_path, "common_fields.training_period|validation_period; benchmark_classification", "Research Design")
    insertion("R02", "5.2", "Target soil-moisture band", "0.22|0.32", "0.22–0.32 m³/m³", rq_path, "RM1.measurable_objective", "Research Design")
    insertion("R03", "5.2", "Irrigation action bounds", "0|5", "0–5 mm/hour", fair_audit_path, "common_fields.action_min_mm_h|action_max_mm_h", "Fair DRL Benchmark")
    insertion("R04", "5.2", "Simple cases passed", int(simple["passed"].astype(bool).sum()), f"{int(simple['passed'].astype(bool).sum())}/{len(simple)} passed", simple_path, "passed", "Simple-Case & Raw-Data Validation")
    raw_mass_balance = float(pd.to_numeric(raw["max_abs_mass_balance_error_mm"]).max())
    insertion("R05", "5.2", "Maximum raw-episode mass-balance residual", raw_mass_balance, f"{raw_mass_balance:.3e} mm", raw_path, "max_abs_mass_balance_error_mm", "Simple-Case & Raw-Data Validation")
    selected = reward["selected_reward"]
    reward_summary = selected["validation_summary"]
    insertion("R06", "5.3", "Locked reward", selected["reward_version"], selected["reward_version"], reward_path, "selected_reward.reward_version", "Reward Lab")
    insertion("R07", "5.3", "Reward-validation Time in Target", reward_summary["time_in_target_pct_mean"], f"{reward_summary['time_in_target_pct_mean']:.3f} ± {reward_summary['time_in_target_pct_std']:.3f}%", reward_path, "selected_reward.validation_summary.time_in_target_pct_mean|std", "Reward Lab")
    insertion("R08", "5.3", "Reward-validation irrigation", reward_summary["total_irrigation_mm_mean"], f"{reward_summary['total_irrigation_mm_mean']:.3f} ± {reward_summary['total_irrigation_mm_std']:.3f} mm", reward_path, "selected_reward.validation_summary.total_irrigation_mm_mean|std", "Reward Lab")
    insertion("R09", "5.4", "Raw-data validation episodes", "DRY|WET|MIXED", "DRY, WET, and MIXED", raw_path, "episode", "Simple-Case & Raw-Data Validation")
    insertion("R10", "5.5", "Fairness audit", fair_audit["status"], fair_audit["status"], fair_audit_path, "status", "Fair DRL Benchmark")

    for index, model in enumerate(("SAC", "TD3", "DDPG"), start=11):
        row = fair_summary.loc[model]
        mean = float(row[("time_in_target_pct", "mean")])
        std = float(row[("time_in_target_pct", "std")])
        insertion(f"R{index:02d}", "5.5", f"{model} exploratory Time in Target", mean, f"{mean:.3f} ± {std:.3f}%", fair_path, f"algorithm_family={model}; time_in_target_pct", "Fair DRL Benchmark")

    insertion("R14", "5.6", "POMDP exploratory seeds", len(pomdp["seeds"]), f"{len(pomdp['seeds'])} matched seeds", pomdp_path, "seeds", "POMDP Contribution")
    insertion("R15", "5.7", "Main confirmatory rows", len(main), f"{len(main)} model-seed rows", main_path, "row_count", "10-Seed Confirmatory Statistics")
    insertion("R16", "5.7", "Matched seeds", main["seed"].nunique(), f"{main['seed'].nunique()} matched seeds", main_path, "seed", "10-Seed Confirmatory Statistics")

    main_table_rows = []
    for index, model in enumerate(("SACSI-POMDP", "SAC", "TD3", "DDPG"), start=17):
        row = main_summary.loc[model]
        target_mean = float(row["time_in_target_pct_mean"])
        target_std = float(row["time_in_target_pct_std"])
        water_mean = float(row["total_irrigation_mm_mean"])
        water_std = float(row["total_irrigation_mm_std"])
        insertion(f"R{index:02d}", "5.7", f"{model} confirmatory summary", target_mean, f"Time in Target {target_mean:.3f} ± {target_std:.3f}%; irrigation {water_mean:.3f} ± {water_std:.3f} mm", main_path, f"model={model}; aggregate mean|std", "10-Seed Confirmatory Statistics")
        main_table_rows.append((model, f"{target_mean:.3f} ± {target_std:.3f}", f"{water_mean:.3f} ± {water_std:.3f}"))

    omnibus = friedman.iloc[0]
    insertion("R21", "5.7", "Friedman omnibus", float(omnibus["friedman_p"]), f"χ²(3)={float(omnibus['friedman_chi_square']):.3f}, p={float(omnibus['friedman_p']):.6f}", friedman_path, "friedman_chi_square|friedman_df|friedman_p", "10-Seed Confirmatory Statistics")

    contrast_table_rows = []
    for index, row in enumerate(main_planned.itertuples(index=False), start=22):
        insertion(f"R{index:02d}", "5.7", row.comparison, row.mean_difference_pp, f"{row.mean_difference_pp:.3f} pp; 95% CI [{row.bootstrap_ci95_low_pp:.3f}, {row.bootstrap_ci95_high_pp:.3f}]; dz={row.cohens_dz:.3f}; exact-Holm p={row.primary_p_holm:.6f}", contrasts_path, f"comparison={row.comparison}", "10-Seed Confirmatory Statistics")
        contrast_table_rows.append((row.comparison, f"{row.mean_difference_pp:.3f}", f"[{row.bootstrap_ci95_low_pp:.3f}, {row.bootstrap_ci95_high_pp:.3f}]", f"{row.cohens_dz:.3f}", f"{row.primary_p_holm:.6f}"))

    insertion("R25", "5.8", "Factorial confirmatory rows", len(factorial), f"{len(factorial)} variant-seed rows", factorial_path, "row_count", "POMDP Contribution")
    effect_table_rows = []
    for index, row in enumerate(effects.itertuples(index=False), start=26):
        decision = "supported" if bool(row.exact_significant_holm_0_05) else "not supported"
        insertion(f"R{index:02d}", "5.8", row.effect, row.mean_effect, f"{row.mean_effect:.3f} pp; 95% CI [{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}]; exact-Holm p={row.exact_sign_flip_p_holm:.6f}; {decision}", effects_path, f"effect={row.effect}", "10-Seed Confirmatory Statistics")
        effect_table_rows.append((row.effect, f"{row.mean_effect:.3f}", f"[{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}]", f"{row.exact_sign_flip_p_holm:.6f}", decision))

    insertion("R29", "5.8", "Forecast robustness levels", "SF10|SF20|SF30", ", ".join(pomdp["forecast_levels"]), forecast_path, "forecast_level", "Robustness & Context Diagnostics")
    insertion("R30", "5.8", "Sequence windows", "6|12|24|48", ", ".join(str(value) for value in pomdp["sequence_lengths"]) + " hours", sequence_path, "sequence_length", "Robustness & Context Diagnostics")
    insertion("R31", "5.8", "Context intervention conditions", len(pomdp["context_conditions"]), f"{len(pomdp['context_conditions'])} conditions", context_path, "condition", "Robustness & Context Diagnostics")
    insertion("R32", "5.7", "Failed validation rows retained", f"{statistics['failed_validation_seeds_retained']['main_rows']}|{statistics['failed_validation_seeds_retained']['factorial_rows']}", f"{statistics['failed_validation_seeds_retained']['main_rows']} main and {statistics['failed_validation_seeds_retained']['factorial_rows']} factorial rows", statistics_path, "failed_validation_seeds_retained", "10-Seed Confirmatory Statistics")
    insertion("R33", "5.9", "Reviewer evidence coverage", len(reviewer), f"{len(reviewer)}/{len(reviewer)} mapped", "Docs/Reviewer_Alignment/reviewer_alignment_matrix.csv", "reviewer_item_id", "Reviewer Evidence Matrix")

    hypothesis_rows = [
        {
            "hypothesis_id": "H1", "research_question_id": "RM1", "decision": "SUPPORTED_ENGINEERING_GATES",
            "evidence_summary": f"Reward {selected['reward_version']} is locked and Pareto non-dominated; {len(simple)}/{len(simple)} simple cases passed.",
            "evidence_files": f"{reward_path}|{simple_path}|{raw_path}",
            "released_claim": "The Virtual Garden and reward passed the locked numerical, physical, and validation-2024 selection gates.",
            "claim_guard": "Simulator validity is not field validation.", "readiness_status": "READY",
        },
        {
            "hypothesis_id": "H2", "research_question_id": "RM2", "decision": "INCONCLUSIVE_DIRECT_TEST_NOT_EXPORTED",
            "evidence_summary": "The fair three-algorithm benchmark is complete, but the frozen omnibus includes SACSI as a fourth condition and does not isolate the DDPG–TD3–SAC null.",
            "evidence_files": f"{fair_audit_path}|{main_path}|{friedman_path}",
            "released_claim": "DDPG, TD3, and SAC were compared fairly and their descriptive outcomes are reported.",
            "claim_guard": "Do not claim a direct three-algorithm inferential difference from the four-condition omnibus.", "readiness_status": "READY_WITH_GUARD",
        },
        {
            "hypothesis_id": "H3", "research_question_id": "RM3", "decision": "REJECT_H0_WITH_COMPONENT_GUARD",
            "evidence_summary": "The memory main effect is supported; forecast and forecast-by-memory effects are not supported.",
            "evidence_files": f"{factorial_path}|{effects_path}|{context_path}",
            "released_claim": "At least one pre-specified context component effect was non-zero for the locked pipelines: memory.",
            "claim_guard": "Do not claim standalone forecast benefit or forecast-memory synergy.", "readiness_status": "READY",
        },
        {
            "hypothesis_id": "H4", "research_question_id": "RM4", "decision": "REJECT_H0_LOCKED_PIPELINE_SCOPE",
            "evidence_summary": "All three positive SACSI planned contrasts passed exact sign-flip Holm correction.",
            "evidence_files": f"{main_path}|{contrasts_path}|{statistics_path}",
            "released_claim": "The locked SACSI warm-start pipeline achieved higher matched-seed Time in Target than SAC, TD3, and DDPG in the retrospective 2025 simulation benchmark.",
            "claim_guard": "Not an equal-total-budget architecture claim and not field-effectiveness evidence.", "readiness_status": "READY_WITH_GUARD",
        },
    ]
    hypotheses = pd.DataFrame(hypothesis_rows)

    claim_rows = [
        ("C01", "Framework validity", "Virtual Garden numerical consistency", f"{simple_path}|{raw_path}", "RELEASED", "The simulator passed the locked numerical and physical-response gates.", "Field validation"),
        ("C02", "Framework validity", "Reward selection validity", reward_path, "RELEASED", "reward_v4 was selected on validation 2024 by the pre-specified Pareto/stability rule.", "Selection based on 2025"),
        ("C03", "Framework validity", "Fair DDPG–TD3–SAC protocol", fair_audit_path, "RELEASED", "The three algorithms used the locked common environment, split, reward, interaction budget, seeds, and checkpoint rule.", "Mechanisms were made identical"),
        ("C04", "Context activation", "History and forecast branches were exercised", f"{context_path}|{pomdp_path}", "RELEASED_WITH_GUARD", "Context diagnostics show branch activation under the exploratory protocol.", "Activation proves benefit"),
        ("C05", "Performance benefit", "Standalone forecast benefit", effects_path, "NOT_RELEASED", "The forecast main effect was not supported.", "Forecast improves performance"),
        ("C06", "Performance benefit", "Memory benefit", effects_path, "RELEASED_WITH_GUARD", "The memory main effect was supported for the locked warm-start pipelines.", "Universal memory benefit"),
        ("C07", "Statistical superiority", "Forecast-memory interaction", effects_path, "NOT_RELEASED", "The forecast-by-memory interaction was not supported.", "Synergy is proven"),
        ("C08", "Statistical superiority", "SACSI versus SAC, TD3, and DDPG", f"{main_path}|{contrasts_path}", "RELEASED_WITH_GUARD", "The locked SACSI pipeline achieved higher matched-seed Time in Target in the retrospective 2025 simulation benchmark.", "Universal equal-budget architecture superiority"),
        ("C09", "Performance benefit", "Forecast robustness", forecast_path, "DESCRIPTIVE_ONLY", "SF10–SF30 results are exploratory robustness diagnostics.", "Statistical robustness superiority"),
        ("C10", "Framework validity", "Meteorology, soil, and forecast classification", scope_path, "RELEASED", "Meteorology is raw/real, soil state is simulated, and forecast input is a controlled synthetic proxy.", "All inputs are field measurements"),
        ("C11", "Framework validity", "IoT role", scope_path, "CONTEXT_ONLY", "IoT is an implementation and sensing context.", "IoT field deployment was validated"),
        ("C12", "Framework validity", "Energy outcome", scope_path, "NOT_REPORTED", "No energy metric is reported because no pump-power model is available.", "Energy savings"),
        ("C13", "Framework validity", "Field effectiveness", scope_path, "NOT_RELEASED", "Evidence is limited to the locked Virtual Garden simulation.", "Real-world field effectiveness"),
    ]
    claims = pd.DataFrame(claim_rows, columns=(
        "claim_id", "claim_level", "claim_topic", "evidence_files", "release_status",
        "released_wording", "prohibited_wording",
    ))
    claims["readiness_status"] = "READY"
    for sources in claims["evidence_files"]:
        for source in sources.split("|"):
            require_source(source)

    update_rows = [
        ("5.1", "Data Audit & Provenance", "INSERT", f"{scope_path}|{fair_audit_path}|{statistics_path}", "Separate raw meteorology, simulated soil state, and controlled synthetic forecast proxy."),
        ("5.2", "Virtual Garden Validation", "INSERT", f"{rq_path}|{simple_path}|{raw_path}", "Report numerical and physical-response gates; do not call this field validation."),
        ("5.3", "Reward Validation", "INSERT", reward_path, "Report validation-2024 selection and the locked reward trade-off."),
        ("5.4", "Simple-Case & Raw-Data Validation", "INSERT", f"{simple_path}|{raw_path}", "Distinguish raw meteorological forcing from simulated soil response."),
        ("5.5", "Fair DDPG–TD3–SAC Benchmark", "INSERT", f"{fair_audit_path}|{fair_path}", "Treat the three-seed results as descriptive development evidence."),
        ("5.6", "Incremental POMDP Contribution", "INSERT", f"{pomdp_path}|{context_path}", "Separate activation, benefit, and superiority."),
        ("5.7", "10-Seed Confirmatory Benchmark", "INSERT", f"{main_path}|{friedman_path}|{contrasts_path}|{statistics_path}", "Limit superiority wording to locked pipelines and retrospective simulation."),
        ("5.8", "Robustness & Diagnostics", "INSERT", f"{factorial_path}|{effects_path}|{forecast_path}|{sequence_path}", "Retain null forecast and interaction findings."),
        ("5.9", "Answers to RM1–RM4", "INSERT", "Results/Dissertation_Evidence/hypothesis_decision_table.csv", "Use the exact decision and claim guard for each hypothesis."),
        ("6.1", "Reward trade-off", "INSERT", reward_path, "Do not rank by cumulative reward alone."),
        ("6.2", "DDPG versus TD3 versus SAC", "INSERT", f"{fair_path}|{main_path}", "Discuss stability and target occupancy together with water use."),
        ("6.3", "Partial observability", "INSERT", f"{scope_path}|{context_path}", "POMDP rationale is not itself performance evidence."),
        ("6.4", "Forecast contribution", "INSERT", f"{effects_path}|{forecast_path}", "State that standalone forecast benefit was not supported."),
        ("6.5", "Memory contribution", "INSERT", effects_path, "Limit support to the locked pipeline and simulation scope."),
        ("6.6", "SACSI integration", "INSERT", f"{main_path}|{contrasts_path}|{statistics_path}", "Include the unequal effective interaction-budget limitation."),
        ("6.7", "Precision-farming implication", "INSERT", scope_path, "Frame as potential application, not validated deployment."),
        ("6.8", "Limitations", "INSERT", f"{scope_path}|{statistics_path}", "Explicitly list synthetic forecast, simulated soil, single setting, and budget limitation."),
        ("6.9", "Future work", "INSERT", scope_path, "Require field trials, archived as-issued forecasts, and equal-budget retraining before broader claims."),
    ]
    updates = pd.DataFrame(update_rows, columns=(
        "target_section", "section_title", "action", "source_files", "required_wording",
    ))
    updates["readiness_status"] = "READY"
    for sources in updates["source_files"]:
        for source in sources.split("|"):
            if not source.startswith("Results/Dissertation_Evidence/"):
                require_source(source)

    source = lambda path: f"[Source: `{path}`]"  # noqa: E731
    results_markdown = f"""# Draft Bab 5 — Hasil

Status evidence: **FROZEN / READY FOR EDITORIAL INTEGRATION**

Primary endpoint: **Time in Target (%)**

Naskah ini dihasilkan dari artefak beku Modul 8A–9A. Penyuntingan gaya bahasa diperbolehkan, tetapi angka dan batas klaim harus tetap mengikuti `result_insertion_matrix.csv` dan `claim_matrix.csv`.

## 5.1 Data Audit & Provenance

Eksperimen menggunakan forcing meteorologi observasional untuk simulator. Kelembapan tanah, runoff, drainage, deficit, dan surplus adalah keluaran Virtual Garden, bukan pengukuran tanah lapangan. Input forecast controller adalah **SF-20 h+1 controlled synthetic forecast proxy**, bukan archived as-issued operational forecast. {source(scope_path)}

Training menggunakan 2021–2023, pemilihan reward dan checkpoint hanya menggunakan 2024, sedangkan 2025 dibuka setelah registry checkpoint terkunci sebagai retrospective final benchmark. Tidak dilakukan retraining atau checkpoint reselection setelah pembukaan benchmark. {source(fair_audit_path)} {source(statistics_path)}

## 5.2 Virtual Garden Validation

Virtual Garden menggunakan target kelembapan 0.22–0.32 m³/m³ dan batas aksi irigasi 0–5 mm/hour. Seluruh {len(simple)} dari {len(simple)} simple cases memenuhi respons yang diharapkan, aksi terbatas, keluaran finite, dan pemeriksaan neraca massa. Residual neraca massa maksimum pada episode raw-data adalah {raw_mass_balance:.3e} mm. Hasil ini mendukung konsistensi numerik simulator, bukan validasi lapangan. {source(rq_path)} {source(fair_audit_path)} {source(simple_path)} {source(raw_path)}

## 5.3 Reward Validation

Reward final dikunci sebagai **{selected['reward_version']}** melalui validasi 2024 tanpa mengakses benchmark 2025. Kandidat terpilih bersifat Pareto non-dominated dan mencapai Time in Target {reward_summary['time_in_target_pct_mean']:.3f} ± {reward_summary['time_in_target_pct_std']:.3f}% dengan irigasi {reward_summary['total_irrigation_mm_mean']:.3f} ± {reward_summary['total_irrigation_mm_std']:.3f} mm pada {reward_summary['n_seeds']} seed. Keputusan ini adalah trade-off multi-objective, bukan pemaksimalan cumulative reward semata. {source(reward_path)}

## 5.4 Simple-Case & Raw-Data Validation

Enam simple cases mencakup pengeringan, pulse hujan, pulse irigasi, kondisi dekat batas atas, pemulihan dari kondisi di bawah target, dan respons anticipatory terhadap hujan. Episode DRY, WET, dan MIXED menggunakan forcing meteorologi 2024 yang sama untuk controller referensi tanpa retuning per episode. Respons soil moisture tetap merupakan simulasi. {source(simple_path)} {source(raw_path)}

## 5.5 Fair DDPG–TD3–SAC Benchmark

Audit fairness berstatus **{fair_audit['status']}**: DDPG, TD3, dan SAC memakai environment, observation, action bounds, reward, split, interaction budget, seed, metric engine, dan checkpoint rule yang sama, sementara mekanisme algoritmik masing-masing dipertahankan. {source(fair_audit_path)}

Pada benchmark pengembangan tiga-seed, Time in Target deskriptif adalah SAC {fair_summary.loc['SAC', ('time_in_target_pct', 'mean')]:.3f} ± {fair_summary.loc['SAC', ('time_in_target_pct', 'std')]:.3f}%, TD3 {fair_summary.loc['TD3', ('time_in_target_pct', 'mean')]:.3f} ± {fair_summary.loc['TD3', ('time_in_target_pct', 'std')]:.3f}%, dan DDPG {fair_summary.loc['DDPG', ('time_in_target_pct', 'mean')]:.3f} ± {fair_summary.loc['DDPG', ('time_in_target_pct', 'std')]:.3f}%. Angka ini merupakan evidence pengembangan deskriptif; keputusan final menggunakan desain sepuluh-seed. {source(fair_path)}

## 5.6 Incremental POMDP Contribution

Eksperimen eksploratori tiga-seed membentuk desain 2×2 SAC Basic, SAC + Forecast, SAC + LSTM, dan SACSI Full serta intervensi terhadap history dan forecast. Diagnostik menunjukkan bahwa branch context dapat diaktifkan, tetapi aktivasi tidak disamakan dengan manfaat performa atau superioritas statistik. {source(pomdp_path)} {source(context_path)}

## 5.7 10-Seed Confirmatory Benchmark

Tabel final berisi {len(main)} baris untuk empat pipeline dan {main['seed'].nunique()} matched seeds. Seluruh seed yang gagal validation gate tetap dipertahankan. {source(main_path)} {source(statistics_path)}

{_table(('Pipeline', 'Time in Target mean ± SD (%)', 'Irrigation mean ± SD (mm)'), main_table_rows)}

Friedman omnibus menunjukkan perbedaan kondisi, χ²({int(omnibus['friedman_df'])}) = {float(omnibus['friedman_chi_square']):.3f}, p = {float(omnibus['friedman_p']):.6f}. {source(friedman_path)}

{_table(('Planned contrast', 'Difference (pp)', '95% bootstrap CI', "Cohen's dz", 'Exact-Holm p'), contrast_table_rows)}

Ketiga planned contrast mendukung Time in Target yang lebih tinggi untuk **locked SACSI warm-start training pipeline** pada benchmark simulasi retrospektif 2025. Selisih terhadap SAC hanya {main_planned.iloc[0]['mean_difference_pp']:.3f} percentage points sehingga tidak ditafsirkan sebagai peningkatan operasional besar. {source(contrasts_path)}

## 5.8 Robustness & Diagnostics

Tabel factorial berisi {len(factorial)} baris untuk empat varian dan sepuluh matched seeds. {source(factorial_path)}

{_table(('Effect', 'Mean (pp)', '95% bootstrap CI', 'Exact-Holm p', 'Decision'), effect_table_rows)}

Hanya memory main effect yang didukung. Standalone forecast effect dan forecast × memory interaction tidak didukung. Eksperimen SF10, SF20, dan SF30, sequence window 6, 12, 24, dan 48 jam, serta {len(pomdp['context_conditions'])} kondisi intervensi diperlakukan sebagai diagnostik eksploratori, bukan uji superiority baru. {source(effects_path)} {source(forecast_path)} {source(sequence_path)} {source(context_path)}

## 5.9 Jawaban RM1–RM4

- **RM1/H1:** gate rekayasa Virtual Garden dan reward didukung, dengan batas bahwa evidence berasal dari simulasi.
- **RM2/H2:** fairness dan hasil deskriptif DDPG–TD3–SAC tersedia, tetapi hipotesis inferensial khusus tiga algoritma belum konklusif karena omnibus beku mencakup SACSI sebagai kondisi keempat.
- **RM3/H3:** null gabungan ditolak karena memory main effect didukung; forecast dan interaction tetap null.
- **RM4/H4:** null ditolak untuk ruang lingkup locked warm-start pipelines pada benchmark simulasi retrospektif; klaim equal-total-budget architecture dan field effectiveness tidak dibuka.

Keputusan lengkap dan evidence per hipotesis tersedia di `Results/Dissertation_Evidence/hypothesis_decision_table.csv`.
"""

    sac_contrast = main_planned.loc[main_planned["comparison"].eq("SACSI-POMDP - SAC")].iloc[0]
    memory = effects.loc[effects["effect"].eq("Memory main effect")].iloc[0]
    discussion_markdown = f"""# Draft Bab 6 — Pembahasan

## 6.1 Reward trade-off

Pemilihan {selected['reward_version']} memperlihatkan bahwa objective irigasi tidak cukup dinilai dari cumulative reward. Time in Target, penggunaan air, violation, deficit, kestabilan antar-seed, dan Pareto non-dominance harus dibaca bersama. Karena reward dipilih hanya pada validasi 2024, hasil 2025 tetap berfungsi sebagai evaluasi retrospektif, bukan sumber tuning. {source(reward_path)}

## 6.2 DDPG versus TD3 versus SAC

DDPG dan TD3 menggunakan air lebih sedikit pada sejumlah seed, tetapi pencapaian targetnya juga lebih rendah dan lebih bervariasi. Hal tersebut menunjukkan bahwa irigasi minimum tidak otomatis berarti efisien; controller dapat terlihat hemat karena under-irrigation. SAC lebih stabil pada primary endpoint dalam evidence ini. Perbandingan mekanisme algoritmik tetap dibatasi pada protokol Virtual Garden yang dikunci. {source(main_path)} {source(fair_audit_path)}

## 6.3 Partial observability

Formulasi POMDP relevan karena controller tidak mengamati seluruh state dan proses laten tanah serta tidak mengetahui forcing mendatang secara sempurna. History dan forecast menyediakan context tambahan, tetapi alasan formulasi ini hanya mendukung validitas kerangka; manfaat performanya tetap harus dibuktikan melalui ablation dan inferensi. {source(scope_path)} {source(context_path)}

## 6.4 Forecast contribution

Standalone forecast main effect sebesar {effects.loc[effects['effect'].eq('Forecast main effect'), 'mean_effect'].iloc[0]:.3f} percentage points tidak didukung setelah koreksi exact-Holm. Oleh sebab itu, penelitian ini tidak mengklaim bahwa forecast proxy secara mandiri meningkatkan performa. Hasil robustness SF10–SF30 tetap berguna sebagai diagnostik sensitivitas terhadap error proxy. {source(effects_path)} {source(forecast_path)}

## 6.5 Memory contribution

Memory main effect sebesar {memory['mean_effect']:.3f} percentage points dengan 95% bootstrap CI [{memory['bootstrap_ci95_low']:.3f}, {memory['bootstrap_ci95_high']:.3f}] dan exact-Holm p = {memory['exact_sign_flip_p_holm']:.6f} didukung pada locked warm-start pipelines. Temuan ini konsisten dengan peran history dalam merangkum dinamika laten, tetapi belum membuktikan manfaat universal di luar simulator dan protokol ini. {source(effects_path)}

## 6.6 SACSI integration

SACSI mengintegrasikan current observation, representasi history, dan controlled synthetic forecast proxy. Dibanding SAC, peningkatan Time in Target adalah {sac_contrast['mean_difference_pp']:.3f} percentage points dengan 95% bootstrap CI [{sac_contrast['bootstrap_ci95_low_pp']:.3f}, {sac_contrast['bootstrap_ci95_high_pp']:.3f}] dan exact-Holm p = {sac_contrast['primary_p_holm']:.6f}. Hasil ini mendukung locked end-to-end pipeline, tetapi tidak memisahkan efek arsitektur dari tambahan adaptation budget. {source(contrasts_path)} {source(statistics_path)}

## 6.7 Precision-farming implication

Kerangka ini menyediakan cara terstruktur untuk menggabungkan sensing saat ini, history, dan informasi prediktif dalam continuous irrigation control. Implikasinya adalah potensi desain controller yang lebih sadar context, bukan bukti kesiapan deployment. Integrasi sensor, aktuator, komunikasi, keselamatan, dan kalibrasi lokasi masih memerlukan validasi tersendiri. {source(scope_path)}

## 6.8 Limitations

Penelitian dibatasi oleh Virtual Garden untuk hortikultura generik, soil state hasil simulasi, satu setting meteorologi retrospektif, dan forecast terkontrol yang bukan archived as-issued operational forecast. Tidak ada field trial, pengukuran hasil panen, atau model daya pompa sehingga klaim efektivitas lapangan, yield, dan penghematan energi tidak dibuat. Selain itu, context variants menggunakan SAC anchor 6,720 interactions ditambah 6,720 adaptation interactions, sedangkan non-context algorithms menggunakan total 6,720 interactions. {source(scope_path)} {source(statistics_path)}

## 6.9 Future work

Pekerjaan berikutnya perlu menguji ulang controller dengan archived as-issued forecasts, multi-location meteorology and soil calibration, equal-total-budget training, hardware-in-the-loop testing, serta field trials dengan sensor dan aktuator nyata. Model daya pompa yang terkalibrasi diperlukan sebelum energy outcome dapat dilaporkan. {source(scope_path)} {source(statistics_path)}
"""

    prohibited = (
        "universally superior",
        "field effectiveness was demonstrated",
        "archived operational forecast was used",
        "equal-total-budget architecture superiority was proven",
    )
    combined = (results_markdown + discussion_markdown).lower()
    if any(phrase in combined for phrase in prohibited):
        raise RuntimeError("Unsupported wording detected in dissertation drafts")

    insertion_matrix = pd.DataFrame(insertion_rows)
    if (
        insertion_matrix["source_sha256"].isna().any()
        or hypotheses["evidence_files"].isna().any()
        or claims["readiness_status"].ne("READY").any()
    ):
        raise RuntimeError("Module 9B evidence reconciliation is incomplete")

    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "dissertation_update_map.csv": RESULTS / "dissertation_update_map.csv",
        "result_insertion_matrix.csv": RESULTS / "result_insertion_matrix.csv",
        "hypothesis_decision_table.csv": RESULTS / "hypothesis_decision_table.csv",
        "claim_matrix.csv": RESULTS / "claim_matrix.csv",
        "chapter_results_draft.md": DOCS / "chapter_results_draft.md",
        "chapter_discussion_draft.md": DOCS / "chapter_discussion_draft.md",
    }
    updates.to_csv(output_paths["dissertation_update_map.csv"], index=False)
    insertion_matrix.to_csv(output_paths["result_insertion_matrix.csv"], index=False)
    hypotheses.to_csv(output_paths["hypothesis_decision_table.csv"], index=False)
    claims.to_csv(output_paths["claim_matrix.csv"], index=False)
    output_paths["chapter_results_draft.md"].write_text(results_markdown, encoding="utf-8")
    output_paths["chapter_discussion_draft.md"].write_text(discussion_markdown, encoding="utf-8")

    output_records = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in output_paths.items()
    }
    metadata_path = RESULTS / "dissertation_release_metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    metadata = {
        "module": "9B",
        "status": "READY",
        "source_release": "Results/Dashboard/dashboard_release_metadata.json",
        "dashboard_status": dashboard["status"],
        "update_map_rows": len(updates),
        "result_insertions": len(insertion_matrix),
        "hypothesis_decisions": len(hypotheses),
        "claims": len(claims),
        "source_files_registered": int(insertion_matrix["source_file"].nunique()),
        "unsupported_superiority_claims": 0,
        "h2_direct_test_guard_retained": True,
        "outputs": output_records,
        "generated_utc": previous.get("generated_utc") if previous.get("outputs") == output_records else datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    release = build_dissertation_release()
    print(json.dumps(release, indent=2))
    if release["status"] != "READY":
        raise SystemExit("Dissertation evidence release is NOT READY")
