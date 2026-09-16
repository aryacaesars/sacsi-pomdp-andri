"""Module 9A reviewer-oriented dashboard over locked 8A–8H evidence."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dashboard.data import UI_PAGE_NAMES, load_dashboard_release, translate
from Dashboard.views import render_page


st.set_page_config(page_title="SACSI-POMDP Final Evidence", layout="wide")

with st.sidebar:
    language = st.selectbox(
        "Language / Bahasa", ("English", "Bahasa Indonesia"), key="ui_language"
    )
    display_pages = [translate(page, language) for page in UI_PAGE_NAMES]
    selected_display = st.radio(
        translate("Dashboard Page", language), display_pages, key=f"page_{language}"
    )
    page = UI_PAGE_NAMES[display_pages.index(selected_display)]

st.title(translate("SACSI-POMDP Final Evidence Dashboard", language))
st.caption(translate("Reviewer-oriented evidence from Modules 8A–8H", language))


@st.cache_data
def release_evidence():
    return load_dashboard_release()


registry, matrix, metadata = release_evidence()
if metadata.get("status") != "READY":
    st.error(translate("NOT READY", language))
    st.json(metadata)
    st.stop()

render_page(page, language, registry, matrix, metadata)

st.divider()
st.caption(
    "Source of truth: Results/Confirmatory_10Seed · reward_v4 · retrospective final benchmark 2025 · "
    "SF-20 h+1 controlled synthetic forecast proxy · no field-validation claim"
)
