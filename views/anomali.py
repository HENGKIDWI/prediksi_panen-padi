"""Tahap 2 — Cek Anomali."""
from __future__ import annotations

import streamlit as st

from core.plots import plot_anomali
from ui import STRETCH, guard, nav_footer, show, stage_header


def render(P, status):
    stage_header(2)
    if not guard(P, 2):
        return
    n_cells, n_rows = int(P["issue_mask"].values.sum()), int(P["issue_mask"].any(axis=1).sum())
    a, b, c = st.columns(3)
    a.metric("Total sel bermasalah", f"{n_cells:,}")
    b.metric("Baris terdampak", f"{n_rows:,}", f"dari {len(P['df']):,}", delta_color="off")
    c.metric("Persentase baris", f"{n_rows / max(len(P['df']), 1) * 100:.1f}%")
    st.caption("Anomali = NaN, non-numerik, negatif, atau nol pada seluruh kolom.")
    rep_disp = P["report"].rename(index=P["disp"])
    st.dataframe(rep_disp, **STRETCH)
    show(plot_anomali(rep_disp, P["issue_mask"]))
    nav_footer(2, status)
