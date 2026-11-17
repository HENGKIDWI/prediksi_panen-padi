"""Tahap 3 — Preprocessing."""
from __future__ import annotations

import streamlit as st

from core.config import CLEAN_LABEL, CLEAN_METHODS, COLS_ASLI, DATA_MODE_LABEL, DATA_MODES, TARGET
from core.plots import plot_distribusi, plot_perbandingan_data
from state import dn
from ui import guard, nav_footer, pick, show, stage_header


def render(P, status):
    stage_header(3)
    c1, c2 = st.columns(2)
    with c1:
        pick("Gunakan data", "data_mode", DATA_MODES, fmt=DATA_MODE_LABEL.get)
    with c2:
        if st.session_state["data_mode"] == "cleaning":
            pick("Metode cleaning", "clean_method", CLEAN_METHODS, fmt=CLEAN_LABEL.get)
        else:
            st.selectbox("Metode cleaning", ["—"], disabled=True, help="Aktif bila memilih 'Data cleaning'.")
    if not guard(P, 3):
        return
    if st.session_state["data_mode"] == "cleaning":
        st.success(f"{P['label_data']} selesai. Data bersih: **{len(P['data'])}** baris (baris dihapus: {P['n_hilang']}).")
    else:
        if P["n_nan"]:
            st.warning(f"{P['n_nan']} baris berisi NaN/teks dibuang karena ELM tidak bisa memprosesnya.")
        st.success(f"Memakai data asli: **{len(P['data'])}** baris.")
    show(plot_perbandingan_data(P["df"], P["data"], P["label_data"], dn(P, TARGET)))
    with st.expander("📈 Distribusi data yang dipakai"):
        show(plot_distribusi(P["data"], COLS_ASLI, f"Distribusi Data yang Dipakai ({P['label_data']})",
                             labels=P["disp"]))
    nav_footer(3, status)
