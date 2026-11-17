"""Tahap 6 — Pilih Model."""
from __future__ import annotations

import streamlit as st

from core.config import MODEL_LABEL, MODELS
from ui import nav_footer, pick, stage_header


def render(P, status):
    stage_header(6)
    pick("Model", "model_key", MODELS, fmt=MODEL_LABEL.get)
    st.info(f"Model dipilih: **{st.session_state['model_key']}**")
    if st.session_state["model_key"] == "ELM":
        st.write("Bobot input & bias diacak (uniform), hanya bobot output β yang dihitung secara analitik.")
    else:
        st.write("Bobot input & bias dioptimasi Genetic Algorithm sebelum β dihitung; lebih lambat namun biasanya lebih akurat.")
    nav_footer(6, status)
