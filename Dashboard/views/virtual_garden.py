"""Interactive dashboard view over frozen 2025 Virtual Garden trajectories."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from evaluation import compute_metrics  # noqa: E402
from Dashboard.data import (  # noqa: E402
    export_csv,
    filter_dates,
    load_final_logs,
    load_virtual_garden_registry,
    localize_columns,
    method_slug,
    png_chart_config,
    VIRTUAL_GARDEN_METHODS,
)


TARGET_MIN = 0.22
TARGET_MAX = 0.32
DRY_COLOR = "#E65100"
TARGET_COLOR = "#2E7D32"
WET_COLOR = "#1565C0"


def say(language: str, english: str, indonesian: str) -> str:
    return indonesian if language == "Bahasa Indonesia" else english


@st.cache_data(show_spinner=False)
def cached_final_logs(methods: tuple[str, ...], seed: int) -> pd.DataFrame:
    return load_final_logs(list(methods), seed)


def _status(theta: float, language: str) -> tuple[str, str]:
    if theta < TARGET_MIN:
        return "🥀", say(language, "Below target", "Di bawah target")
    if theta > TARGET_MAX:
        return "🌿", say(language, "Above target", "Di atas target")
    return "🌱", say(language, "Inside target", "Di dalam target")


def _moisture_gauge(method: str, theta: float, language: str) -> go.Figure:
    icon, status = _status(theta, language)
    bar_color = DRY_COLOR if theta < TARGET_MIN else WET_COLOR if theta > TARGET_MAX else TARGET_COLOR
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=theta,
        number={"valueformat": ".3f"},
        title={"text": f"{icon} {method}<br><span style='font-size:0.75em'>{status}</span>"},
        gauge={
            "axis": {"range": [0.10, 0.45], "tickformat": ".2f"},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0.10, TARGET_MIN], "color": "#FDE2E2"},
                {"range": [TARGET_MIN, TARGET_MAX], "color": "#DDF3E4"},
                {"range": [TARGET_MAX, 0.45], "color": "#DDEBFA"},
            ],
            "threshold": {
                "line": {"color": "#424242", "width": 4},
                "thickness": 0.75,
                "value": (TARGET_MIN + TARGET_MAX) / 2,
            },
        },
    ))
    figure.update_layout(autosize=True, height=250, margin={"l": 25, "r": 25, "t": 75, "b": 15})
    return figure


def _ordered_snapshot(
    view: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    selected: list[str],
) -> pd.DataFrame:
    """Keep gauge rows aligned with the controller selection order."""
    snapshot = view.loc[view["timestamp"] == snapshot_time].set_index("controller", drop=False)
    return snapshot.loc[selected].reset_index(drop=True)


def render_virtual_garden(language: str, **_) -> None:
    runs = load_virtual_garden_registry()
    method_order = list(VIRTUAL_GARDEN_METHODS)
    rl_methods = set(runs.loc[runs["method_type"] == "rl", "method"])
    seeds = sorted(runs.loc[runs["method_type"] == "rl", "seed"].dropna().astype(int).unique())

    st.header(say(language, "Virtual Garden", "Kebun Virtual"))
    st.caption(say(
        language,
        "Interactive replay of real 2025 meteorology and simulated soil-water trajectories.",
        "Replay interaktif meteorologi riil 2025 dan trajektori air tanah hasil simulasi.",
    ))

    single_label = say(language, "Single method", "Satu metode")
    compare_label = say(language, "Compare 2–4 methods", "Bandingkan 2–4 metode")
    controls = st.columns((1.0, 2.0, 1.0))
    with controls[0]:
        mode = st.radio(
            say(language, "View mode", "Mode tampilan"),
            (single_label, compare_label),
            horizontal=True,
            key="garden_view_mode",
        )
    with controls[1]:
        if mode == single_label:
            default_index = method_order.index("SACSI Full")
            selected = [st.selectbox(
                say(language, "Controller", "Controller"),
                method_order,
                index=default_index,
                key="garden_single_method",
            )]
        else:
            selected = st.multiselect(
                say(language, "Controllers", "Controller"),
                method_order,
                default=["Threshold-Based", "SAC Basic", "SACSI Full"],
                max_selections=4,
                key="garden_compared_methods",
            )
            if len(selected) < 2:
                st.info(say(language, "Select at least two methods.", "Pilih minimal dua metode."))
                return
    with controls[2]:
        has_rl = bool(set(selected) & rl_methods)
        seed = st.selectbox(
            say(language, "Matched RL seed", "Seed RL berpasangan"),
            seeds,
            disabled=not has_rl,
            key="garden_seed",
        )

    if {"DDPG", "TD3"}.intersection(selected):
        st.caption(say(
            language,
            "DDPG/TD3 replay uses the locked reward-v4 confirmatory checkpoints evaluated on 2025.",
            "Replay DDPG/TD3 memakai checkpoint konfirmatori reward-v4 terkunci yang dievaluasi pada 2025.",
        ))

    logs = cached_final_logs(tuple(selected), int(seed))
    minimum_date = logs["timestamp"].min().date()
    maximum_date = logs["timestamp"].max().date()
    dates = st.date_input(
        say(language, "Displayed period", "Periode tampilan"),
        value=(minimum_date, min(minimum_date + pd.Timedelta(days=30), maximum_date)),
        min_value=minimum_date,
        max_value=maximum_date,
        key="garden_dates",
    )
    if len(dates) != 2:
        st.info(say(language, "Select both start and end dates.", "Pilih tanggal mulai dan selesai."))
        return

    view = filter_dates(logs, dates[0], dates[1])
    if view.empty:
        st.warning(say(language, "No data in the selected period.", "Tidak ada data pada periode terpilih."))
        return

    timestamps = view["timestamp"].drop_duplicates().sort_values().tolist()
    snapshot_index = st.slider(
        say(language, "Garden replay hour", "Jam replay kebun"),
        min_value=0,
        max_value=len(timestamps) - 1,
        value=0,
        format=say(language, "Hour %d", "Jam %d"),
        key="garden_hour",
    )
    snapshot_time = timestamps[snapshot_index]
    snapshot = _ordered_snapshot(view, snapshot_time, selected)
    weather = snapshot.iloc[0]

    cards = st.columns(4)
    cards[0].metric(say(language, "Simulation time", "Waktu simulasi"), snapshot_time.strftime("%d %b %Y %H:%M"))
    cards[1].metric(say(language, "Actual rain", "Hujan aktual"), f"{weather['actual_precipitation_mm']:.2f} mm/h")
    cards[2].metric(say(language, "SF20 h+1 proxy", "Proxy SF20 h+1"), f"{weather['forecast_precipitation_mm']:.2f} mm/h")
    cards[3].metric(say(language, "Target band", "Target band"), f"{TARGET_MIN:.2f}–{TARGET_MAX:.2f}")

    gauge_columns = st.columns(len(selected))
    for column, row in zip(gauge_columns, snapshot.itertuples(index=False)):
        with column:
            st.plotly_chart(
                _moisture_gauge(row.controller, float(row.theta), language),
                width="stretch",
                key=f"garden_gauge_{method_slug(row.controller)}",
                config={**png_chart_config(f"garden_{row.controller}"), "responsive": True},
            )
            st.metric(
                say(language, "Irrigation now", "Irigasi saat ini"),
                f"{row.irrigation_mm:.3f} mm/h",
            )

    st.markdown(say(language, "**Meter color legend**", "**Keterangan warna meter**"))
    legend = st.columns(4)
    legend[0].markdown(say(
        language,
        "🟧 **Below target**  \nθ < 0.22",
        "🟧 **Di bawah target**  \nθ < 0,22",
    ))
    legend[1].markdown(say(
        language,
        "🟩 **Inside target**  \n0.22 ≤ θ ≤ 0.32",
        "🟩 **Di dalam target**  \n0,22 ≤ θ ≤ 0,32",
    ))
    legend[2].markdown(say(
        language,
        "🟦 **Above target**  \nθ > 0.32",
        "🟦 **Di atas target**  \nθ > 0,32",
    ))
    legend[3].markdown(say(
        language,
        "⬛ **Dark marker**  \nTarget midpoint: 0.27",
        "⬛ **Penanda gelap**  \nTitik tengah target: 0,27",
    ))

    st.subheader(say(language, "Soil moisture and target band", "Kelembapan tanah dan target band"))
    soil = px.line(
        view,
        x="timestamp",
        y="theta",
        color="controller",
        labels={"controller": say(language, "Method", "Metode")},
    )
    soil.add_hrect(y0=TARGET_MIN, y1=TARGET_MAX, fillcolor="green", opacity=0.12, line_width=0)
    soil.add_hline(y=TARGET_MIN, line_dash="dash", line_color="green")
    soil.add_hline(y=TARGET_MAX, line_dash="dash", line_color="green")
    soil.add_vline(x=snapshot_time.timestamp() * 1000, line_dash="dot", line_color="#555")
    soil.update_layout(yaxis_title=say(
        language,
        "Volumetric soil moisture (m³/m³)",
        "Kelembapan tanah volumetrik (m³/m³)",
    ))
    st.plotly_chart(soil, width="stretch", config=png_chart_config("virtual_garden_soil_moisture"))

    left, right = st.columns(2)
    with left:
        irrigation = px.line(
            view,
            x="timestamp",
            y="irrigation_mm",
            color="controller",
            labels={
                "controller": say(language, "Method", "Metode"),
                "irrigation_mm": say(language, "Irrigation (mm/h)", "Irigasi (mm/jam)"),
            },
        )
        st.plotly_chart(irrigation, width="stretch", config=png_chart_config("virtual_garden_irrigation"))
    with right:
        cumulative = px.line(
            view,
            x="timestamp",
            y="cumulative_irrigation_mm",
            color="controller",
            labels={
                "controller": say(language, "Method", "Metode"),
                "cumulative_irrigation_mm": say(language, "Cumulative irrigation (mm)", "Irigasi kumulatif (mm)"),
            },
        )
        st.plotly_chart(cumulative, width="stretch", config=png_chart_config("virtual_garden_cumulative_water"))

    precipitation = view.drop_duplicates("timestamp")[[
        "timestamp", "actual_precipitation_mm", "forecast_precipitation_mm",
    ]]
    rain = go.Figure()
    rain.add_bar(
        x=precipitation["timestamp"],
        y=precipitation["actual_precipitation_mm"],
        name=say(language, "Actual rain", "Hujan aktual"),
    )
    rain.add_scatter(
        x=precipitation["timestamp"],
        y=precipitation["forecast_precipitation_mm"],
        name="SF20 h+1 proxy",
        mode="lines",
    )
    rain.update_layout(
        yaxis_title=say(language, "Precipitation (mm/h)", "Presipitasi (mm/jam)"),
        barmode="overlay",
    )
    st.plotly_chart(rain, width="stretch", config=png_chart_config("virtual_garden_rain"))

    metric_table = pd.DataFrame([
        {"method": method, **compute_metrics(group)}
        for method, group in view.groupby("controller", sort=False)
    ])
    shown = [
        "method", "total_irrigation_mm", "time_in_target_pct", "violation_rate_pct",
        "deficit_rate_pct", "surplus_rate_pct", "rmse_band", "action_smoothness",
        "runoff_total_mm", "drainage_total_mm", "max_abs_mass_balance_error_mm",
    ]
    st.subheader(say(language, "Metrics for displayed period", "Metrik periode tampilan"))
    st.dataframe(localize_columns(metric_table[shown], language), hide_index=True, width="stretch")

    st.download_button(
        say(language, "Download displayed trajectory", "Unduh trajektori tampilan"),
        export_csv(view),
        f"virtual_garden_{dates[0]}_{dates[1]}_seed{seed}.csv",
        "text/csv",
    )
    st.info(say(
        language,
        "This page replays frozen model outputs; it does not generate mock data or recalculate dissertation evidence.",
        "Halaman ini memutar ulang output model terkunci; tidak membuat mock data atau menghitung ulang evidence disertasi.",
    ))
