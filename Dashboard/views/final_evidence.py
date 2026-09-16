"""Reviewer-oriented renderers for the nine Module 9A dashboard pages."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Dashboard.data import (
    CONFIRMATORY,
    METHOD_SUPERIORITY_PAGE,
    PAGE_NAMES,
    load_confirmatory_evidence,
    load_research_design,
    load_reward_evidence,
    load_robustness_evidence,
    load_simple_case_evidence,
    localize_columns,
    summarize_runs,
    translate,
)
from Dashboard.views.virtual_garden import render_virtual_garden
from evaluation.decision_engine import classify_superiority
from evaluation.pareto import pareto_frontier


def say(language: str, english: str, indonesian: str) -> str:
    return indonesian if language == "Bahasa Indonesia" else english


def table(frame: pd.DataFrame, language: str) -> None:
    st.dataframe(localize_columns(frame, language), hide_index=True, width="stretch")


def research_design(language, **_):
    questions, hypotheses = load_research_design()
    st.header(translate(PAGE_NAMES[0], language))
    st.caption(say(
        language,
        "Locked one-to-one research questions, measurable objectives, hypotheses, and claim boundaries.",
        "Rumusan masalah, tujuan terukur, hipotesis, dan batas klaim yang terkunci satu-ke-satu.",
    ))
    cards = st.columns(3)
    cards[0].metric("RM ↔ T", len(questions))
    cards[1].metric(say(language, "Hypotheses", "Hipotesis"), len(hypotheses))
    cards[2].metric(say(language, "Measurement status", "Status pengukuran"), "LOCKED")
    st.subheader(say(language, "Research-question map", "Peta rumusan masalah"))
    table(questions, language)
    st.subheader(say(language, "Hypothesis decision rules", "Aturan keputusan hipotesis"))
    table(hypotheses, language)
    st.info(say(
        language,
        "Framework validity ≠ context activation ≠ performance benefit ≠ statistical superiority. Virtual-garden evidence is not field validation.",
        "Validitas framework ≠ aktivasi konteks ≠ manfaat performa ≠ superioritas statistik. Evidence Virtual Garden bukan validasi lapangan.",
    ))


def reward_lab(language, **_):
    decision, confirmation, pareto = load_reward_evidence()
    selected = decision["selected_reward"]
    st.header(translate(PAGE_NAMES[1], language))
    cards = st.columns(4)
    cards[0].metric(translate("Locked reward", language), selected["reward_version"])
    cards[1].metric("wI", selected["water_weight"])
    cards[2].metric("wV", selected["violation_weight"])
    cards[3].metric(translate("Matched seeds", language), selected["validation_summary"]["n_seeds"])
    st.success(say(
        language,
        "reward_v4 was selected on validation 2024 only; benchmark 2025 was not accessed.",
        "reward_v4 dipilih hanya pada validasi 2024; benchmark 2025 tidak diakses.",
    ))
    chart = px.scatter(
        pareto,
        x="total_irrigation_mm_mean",
        y="time_in_target_pct_mean",
        color="pareto_non_dominated",
        hover_name="candidate_id",
        symbol="experiment",
        labels={
            "total_irrigation_mm_mean": say(language, "Mean irrigation (mm)", "Rerata irigasi (mm)"),
            "time_in_target_pct_mean": say(language, "Mean Time in Target (%)", "Rerata waktu dalam target (%)"),
        },
    )
    st.plotly_chart(chart, width="stretch")
    st.subheader(say(language, "10-seed reward confirmation", "Konfirmasi reward 10-seed"))
    table(confirmation, language)
    with st.expander(say(language, "All Pareto candidates", "Seluruh kandidat Pareto")):
        table(pareto, language)


def simple_cases(language, **_):
    cases, episodes = load_simple_case_evidence()
    st.header(translate(PAGE_NAMES[2], language))
    cards = st.columns(3)
    cards[0].metric(say(language, "Cases passed", "Kasus lulus"), f"{int(cases['passed'].sum())}/{len(cases)}")
    cards[1].metric(say(language, "Raw episodes", "Episode raw"), episodes["episode"].nunique())
    cards[2].metric(say(language, "Controllers", "Controller"), episodes["controller"].nunique())
    st.subheader(say(language, "Simple-case response panels", "Panel respons simple-case"))
    st.caption(say(
        language,
        "Initial, minimum, maximum, and final evidence checkpoints; these panels are not hour-by-hour trajectories.",
        "Checkpoint evidence awal, minimum, maksimum, dan akhir; panel ini bukan trajektori per jam.",
    ))
    case_panels = (
        ("C1", "Dry-down response", "Respons pengeringan"),
        ("C2", "Rainfall pulse", "Pulse hujan"),
        ("C3", "Irrigation pulse", "Pulse irigasi"),
        ("C5", "Moisture recovery", "Pemulihan kelembapan"),
        ("C4", "Target-band protection", "Perlindungan target band"),
    )
    tabs = st.tabs([say(language, english, indonesian) for _, english, indonesian in case_panels])
    for tab, (case_id, english, indonesian) in zip(tabs, case_panels):
        row = cases.set_index("case_id").loc[case_id]
        checkpoints = pd.DataFrame({
            "checkpoint": ["Initial", "Minimum", "Maximum", "Final"],
            "theta": [row.initial_theta, row.min_theta, row.max_theta, row.final_theta],
        })
        figure = go.Figure(go.Bar(
            x=checkpoints["checkpoint"],
            y=checkpoints["theta"],
            marker_color=["#607D8B", "#E65100", "#1565C0", "#2E7D32"],
            text=checkpoints["theta"],
            texttemplate="%{text:.3f}",
            textposition="outside",
        ))
        figure.add_hrect(y0=0.22, y1=0.32, fillcolor="green", opacity=0.12, line_width=0)
        figure.update_layout(
            title=say(language, english, indonesian),
            yaxis={"title": "θ (m³/m³)", "range": [0.15, 0.38]},
            height=330,
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
        )
        with tab:
            st.plotly_chart(figure, width="stretch", key=f"simple_case_{case_id}")
            st.caption(say(language, "Expected response: ", "Respons harapan: ") + str(row.expected_response))
    st.subheader(say(language, "Deterministic physical/control cases", "Kasus fisik/kontrol deterministik"))
    table(cases, language)
    st.subheader(say(language, "Locked validation-2024 raw-weather episodes", "Episode raw-weather validasi 2024 terkunci"))
    table(episodes, language)
    st.info(say(
        language,
        "Meteorology is real/raw; soil-water states and trajectories are simulated; forecast input is a controlled synthetic proxy.",
        "Meteorologi bersifat real/raw; state air tanah dan trajektori adalah simulasi; input forecast adalah proxy sintetis terkontrol.",
    ))


def fair_drl(language, **_):
    main, _, _, planned, _, summary = load_confirmatory_evidence()
    aggregate = summarize_runs(main, "model").sort_values("time_in_target_pct_mean", ascending=False)
    main_pairs = planned.loc[planned["analysis_family"] == "main_benchmark"].copy()
    st.header(translate(PAGE_NAMES[3], language))
    st.caption(say(
        language,
        "Final reward-v4 retrospective benchmark over ten matched seeds.",
        "Benchmark retrospektif reward-v4 final pada sepuluh seed berpasangan.",
    ))
    chart = px.bar(
        aggregate.sort_values("time_in_target_pct_mean"),
        x="time_in_target_pct_mean",
        y="model",
        orientation="h",
        error_x="time_in_target_pct_std",
        color="model",
        labels={"time_in_target_pct_mean": say(language, "Mean Time in Target (%)", "Rerata waktu dalam target (%)")},
    )
    chart.update_layout(showlegend=False)
    st.plotly_chart(chart, width="stretch")
    st.subheader(translate("Main confirmatory results", language))
    table(aggregate, language)
    st.subheader(say(language, "Pre-specified main contrasts", "Kontras utama yang ditetapkan sebelumnya"))
    table(main_pairs, language)
    st.warning(translate(
        "Locked-pipeline claim only; unequal effective total interaction budget blocks an unqualified architecture claim.",
        language,
    ))
    st.caption(summary["protocol_limitation"])


def pomdp_contribution(language, **_):
    _, factorial, _, planned, effects, _ = load_confirmatory_evidence()
    aggregate = summarize_runs(factorial, "variant").sort_values("time_in_target_pct_mean", ascending=False)
    family_pairs = planned.loc[planned["analysis_family"] == "factorial_pairwise"]
    st.header(translate(PAGE_NAMES[4], language))
    st.caption("F0M0 → F1M0 / F0M1 → F1M1")
    chart = px.bar(
        aggregate.sort_values("time_in_target_pct_mean"),
        x="time_in_target_pct_mean",
        y="variant",
        orientation="h",
        error_x="time_in_target_pct_std",
        color="variant",
    )
    chart.update_layout(showlegend=False)
    st.plotly_chart(chart, width="stretch")
    st.subheader(say(language, "SAC-family 10-seed summary", "Ringkasan SAC-family 10-seed"))
    table(aggregate, language)
    st.subheader(say(language, "Factorial effects", "Efek faktorial"))
    table(effects, language)
    st.subheader(say(language, "Family pairwise contrasts", "Kontras pairwise family"))
    table(family_pairs, language)
    st.info(say(
        language,
        "Memory pipeline effect is supported. Forecast and Forecast×Memory interaction are not supported; SACSI is not significantly better than LSTM-only.",
        "Efek pipeline memory didukung. Forecast dan interaksi Forecast×Memory tidak didukung; SACSI tidak signifikan lebih baik dari LSTM-only.",
    ))


def method_superiority(language, **_):
    main, _, _, planned, _, summary = load_confirmatory_evidence()
    aggregate = summarize_runs(main, "model")
    main_pairs = planned.loc[planned["analysis_family"] == "main_benchmark"].copy()
    decisions = classify_superiority(main_pairs)
    tradeoff = pareto_frontier(
        aggregate,
        method_column="model",
        maximize="time_in_target_pct_mean",
        minimize="total_irrigation_mm_mean",
    )

    water_leader = tradeoff.loc[tradeoff["total_irrigation_mm_mean"].idxmin(), "model"]
    control_leader = tradeoff.loc[tradeoff["time_in_target_pct_mean"].idxmax(), "model"]
    tradeoff["pareto_role"] = ""
    tradeoff.loc[tradeoff["model"] == water_leader, "pareto_role"] = "LOWEST WATER USE"
    tradeoff.loc[tradeoff["model"] == control_leader, "pareto_role"] = "MOISTURE CONTROL LEADER"

    st.header(translate(METHOD_SUPERIORITY_PAGE, language))
    st.caption(say(
        language,
        "Automatic decisions consume the frozen 8H evidence; no new hypothesis test is run here.",
        "Keputusan otomatis membaca evidence 8H yang beku; tidak ada uji hipotesis baru di halaman ini.",
    ))
    cards = st.columns(4)
    cards[0].metric(translate("Primary endpoint", language), "Time in Target (%)")
    cards[1].metric(
        say(language, "Supported contrasts", "Kontras didukung"),
        f"{int(decisions['decision'].eq('STATISTICALLY SUPERIOR').sum())}/{len(decisions)}",
    )
    cards[2].metric(
        say(language, "Pareto non-dominated", "Pareto non-dominated"),
        int(tradeoff["pareto_non_dominated"].sum()),
    )
    cards[3].metric(say(language, "Claim scope", "Ruang lingkup klaim"), "LOCKED PIPELINE")

    st.subheader(say(language, "Evidence-based superiority decisions", "Keputusan superioritas berbasis evidence"))
    table(decisions, language)
    if decisions["decision"].eq("STATISTICALLY SUPERIOR").all():
        st.success(say(
            language,
            "SACSI-POMDP is statistically superior to SAC, TD3, and DDPG for Time in Target under the locked pipeline.",
            "SACSI-POMDP superior secara statistik terhadap SAC, TD3, dan DDPG untuk Time in Target pada pipeline terkunci.",
        ))

    st.subheader(say(language, "Water versus moisture-control Pareto analysis", "Analisis Pareto air versus kontrol kelembapan"))
    chart = px.scatter(
        tradeoff,
        x="total_irrigation_mm_mean",
        y="time_in_target_pct_mean",
        color="pareto_status",
        symbol="pareto_status",
        text="model",
        hover_data=["pareto_role", "dominated_by"],
        labels={
            "total_irrigation_mm_mean": say(language, "Mean irrigation (mm)", "Rerata irigasi (mm)"),
            "time_in_target_pct_mean": say(language, "Mean Time in Target (%)", "Rerata waktu dalam target (%)"),
        },
    )
    chart.update_traces(textposition="top center", marker={"size": 14})
    chart.update_layout(height=460)
    st.plotly_chart(chart, width="stretch", key="method_pareto")
    table(tradeoff[[
        "model", "total_irrigation_mm_mean", "time_in_target_pct_mean",
        "pareto_status", "pareto_role", "dominated_by",
    ]], language)

    objectives = (
        ("Time in Target", "time_in_target_pct_mean", "MAXIMIZE"),
        ("Total Irrigation", "total_irrigation_mm_mean", "MINIMIZE"),
        ("Violation Rate", "violation_rate_pct_mean", "MINIMIZE"),
        ("RMSE to Target Band", "rmse_band_mean", "MINIMIZE"),
        ("Action Smoothness", "action_smoothness_mean", "MINIMIZE"),
    )
    objective_rows = []
    for objective, metric, direction in objectives:
        index = aggregate[metric].idxmax() if direction == "MAXIMIZE" else aggregate[metric].idxmin()
        leader = aggregate.loc[index]
        evidence_level = (
            "STATISTICALLY SUPERIOR"
            if metric == "time_in_target_pct_mean" and decisions["decision"].eq("STATISTICALLY SUPERIOR").all()
            else "DESCRIPTIVELY BETTER"
        )
        objective_rows.append({
            "objective": objective,
            "direction": direction,
            "leader": leader["model"],
            "value": float(leader[metric]),
            "evidence_level": evidence_level,
            "claim_guard": "Primary inference" if metric == "time_in_target_pct_mean" else "No metric-specific inference",
        })
    st.subheader(say(language, "Objective-specific leaders", "Pemimpin berdasarkan objektif"))
    table(pd.DataFrame(objective_rows), language)
    st.warning(say(
        language,
        "Lowest water use is not proof of efficiency: DDPG/TD3 can save water through under-irrigation. Secondary-metric leaders are descriptive only.",
        "Penggunaan air terendah bukan bukti efisiensi: DDPG/TD3 dapat menghemat air akibat under-irrigation. Pemimpin metrik sekunder hanya bersifat deskriptif.",
    ))
    st.warning(translate(
        "Locked-pipeline claim only; unequal effective total interaction budget blocks an unqualified architecture claim.",
        language,
    ))
    st.caption(summary["protocol_limitation"])


def confirmatory_statistics(language, **_):
    _, _, friedman, planned, effects, summary = load_confirmatory_evidence()
    st.header(translate(PAGE_NAMES[5], language))
    cards = st.columns(4)
    cards[0].metric("Friedman p", f"{friedman.loc[0, 'friedman_p']:.6f}")
    cards[1].metric("RM-ANOVA p", f"{friedman.loc[0, 'rm_anova_p']:.6f}")
    cards[2].metric(translate("Matched seeds", language), 10)
    cards[3].metric(say(language, "Bootstrap resamples", "Bootstrap resample"), "20,000")
    st.subheader(say(language, "Omnibus tests", "Uji omnibus"))
    table(friedman, language)
    st.subheader(say(language, "Planned exact-Holm contrasts", "Kontras exact-Holm terencana"))
    table(planned, language)
    st.subheader(say(language, "2×2 factorial inference", "Inferensi faktorial 2×2"))
    table(effects, language)
    if summary["main_locked_pipeline_superiority_supported"]:
        st.success(say(
            language,
            "All three locked-pipeline main contrasts are supported for Time in Target.",
            "Ketiga kontras utama pipeline terkunci didukung untuk Time in Target.",
        ))
    st.warning(translate(
        "Locked-pipeline claim only; unequal effective total interaction budget blocks an unqualified architecture claim.",
        language,
    ))


def robustness_context(language, **_):
    context, forecast, sequence = load_robustness_evidence()
    st.header(translate(PAGE_NAMES[6], language))
    st.warning(translate(
        "Exploratory reward-v4 diagnostics from Module 8G; no significance claim is inferred here.",
        language,
    ))
    choice = st.selectbox(
        say(language, "Diagnostic", "Diagnostik"),
        ("Context intervention", "Forecast SF10/SF20/SF30", "Sequence k6/k12/k24/k48"),
    )
    if choice == "Context intervention":
        grouped = context.groupby("condition", as_index=False)[
            ["time_in_target_pct", "total_irrigation_mm", "mean_abs_action_delta_vs_full_mm"]
        ].mean()
        chart = px.bar(grouped, x="condition", y="time_in_target_pct", color="condition")
        data = context
    elif choice.startswith("Forecast"):
        grouped = forecast.groupby("forecast_level", as_index=False)[
            ["time_in_target_pct", "total_irrigation_mm"]
        ].mean()
        chart = px.line(grouped, x="forecast_level", y="time_in_target_pct", markers=True)
        data = forecast
    else:
        grouped = sequence.groupby("sequence_length", as_index=False)[
            ["time_in_target_pct", "total_irrigation_mm"]
        ].mean()
        chart = px.line(grouped, x="sequence_length", y="time_in_target_pct", markers=True)
        data = sequence
    st.plotly_chart(chart, width="stretch")
    table(grouped, language)
    with st.expander(say(language, "Per-seed evidence", "Evidence per-seed")):
        table(data, language)


def reviewer_matrix(language, matrix, **_):
    st.header(translate(PAGE_NAMES[7], language))
    ready = int(matrix["readiness_status"].eq("READY").sum())
    st.metric(translate("All reviewer items mapped", language), f"{ready}/{len(matrix)}")
    table(matrix, language)
    if ready == len(matrix):
        st.success("100% MAPPED · READY")
    else:
        st.error(translate("NOT READY", language))


def reproducibility(language, registry, metadata, **_):
    st.header(translate(PAGE_NAMES[8], language))
    cards = st.columns(4)
    cards[0].metric(say(language, "Release status", "Status release"), metadata["status"])
    cards[1].metric(say(language, "Evidence files", "File evidence"), len(registry))
    cards[2].metric(say(language, "Reviewer mapping", "Pemetaan reviewer"), f"{metadata['reviewer_mapping_pct']:.0f}%")
    cards[3].metric(say(language, "Synthetic fixtures", "Fixture sintetis"), metadata["synthetic_production_evidence_count"])
    st.info(translate(
        "No test, fixture, or smoke artifact is accepted as production evidence.", language
    ))
    table(registry, language)
    with st.expander(say(language, "Release metadata", "Metadata release")):
        st.code(json.dumps(metadata, indent=2), language="json")
    st.download_button(
        say(language, "Download result registry", "Unduh result registry"),
        registry.to_csv(index=False).encode("utf-8"),
        "result_registry.csv",
        "text/csv",
    )


RENDERERS = {
    "Virtual Garden": render_virtual_garden,
    PAGE_NAMES[0]: research_design,
    PAGE_NAMES[1]: reward_lab,
    PAGE_NAMES[2]: simple_cases,
    PAGE_NAMES[3]: fair_drl,
    PAGE_NAMES[4]: pomdp_contribution,
    METHOD_SUPERIORITY_PAGE: method_superiority,
    PAGE_NAMES[5]: confirmatory_statistics,
    PAGE_NAMES[6]: robustness_context,
    PAGE_NAMES[7]: reviewer_matrix,
    PAGE_NAMES[8]: reproducibility,
}


def render_page(page: str, language: str, registry, matrix, metadata) -> None:
    RENDERERS[page](
        language=language,
        registry=registry,
        matrix=matrix,
        metadata=metadata,
    )
