"""Tahap 5 — Split Data."""
from __future__ import annotations

import streamlit as st

from core.config import SPLIT_METHOD_LABEL, SPLIT_METHODS, SPLIT_OPTIONS, TARGET
from core.plots import plot_split
from state import dn
from ui import guard, nav_footer, pick, show, stage_header


def render(P, status):
    stage_header(5)
    c1, c2 = st.columns(2)
    with c1:
        pick("Metode split", "split_method", SPLIT_METHODS, fmt=SPLIT_METHOD_LABEL.get,
             help="Per tahun: batas latih/uji jatuh di pergantian tahun, sehingga uji selalu mencakup musim penuh (Q1–Q4).")
    with c2:
        if st.session_state["split_method"] == "tahun":
            pick("Jumlah tahun uji (terakhir)", "split_n_tahun", [1, 2],
                 fmt=lambda n: f"{n} tahun ({n * 12} bulan)", help="1 tahun = sama dengan horizon peramalan.")
        else:
            pick("Rasio split data (terurut waktu)", "split_key", list(SPLIT_OPTIONS), fmt=lambda k: SPLIT_OPTIONS[k][1],
                 help="Rasio persen bisa memotong tahun di tengah sehingga uji tidak mencakup semua musim.")
    if not guard(P, 5):
        return
    tr, te = P["df_train"], P["df_test"]
    st.success(f"Split terurut — {P['split_label']}: latih = **{len(tr)}** bulan ({tr.index.min()} s.d. {tr.index.max()}), "
               f"uji = **{len(te)}** bulan ({te.index.min()} s.d. {te.index.max()})")
    if len(te) % 12 != 0:
        st.warning(f"Uji {len(te)} bulan bukan kelipatan 12 — evaluasi per Q tidak mewakili semua musim. "
                   "Pertimbangkan split per tahun.")
    show(plot_split(tr, te, dn(P, TARGET)))
    nav_footer(5, status)
