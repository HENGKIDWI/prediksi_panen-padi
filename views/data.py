"""Tahap 1 — Muat Data."""
from __future__ import annotations

import hashlib

import streamlit as st

from core.config import COLS_ASLI, DEFAULT_SHEET
from core.plots import plot_distribusi
from state import clear_src, load_demo
from ui import STRETCH, guard, nav_footer, pick, render_mapping, show, stage_header


def render(P, status):
    stage_header(1)
    c1, c2 = st.columns([3, 1.2])
    with c1:
        up = st.file_uploader("File Excel", type=["xlsx", "xlsm", "xls"], key="uploader",
                              help="Nama kolom bebas — dicocokkan lewat bagian 'Pemetaan kolom' di bawah.")
    with c2:
        st.write("")
        st.button("⚡ Muat data demo", on_click=load_demo, **STRETCH)
        if P.get("src"):
            st.button("🗑️ Hapus data", on_click=clear_src, **STRETCH)

    if up is not None:
        b = up.getvalue()
        h = hashlib.md5(b).hexdigest()[:12]
        if h != st.session_state["last_upload"]:                     # hanya proses unggahan baru
            st.session_state["last_upload"] = h
            st.session_state["src"] = {"name": up.name, "bytes": b, "hash": h, "origin": "upload"}
            st.session_state["confirmed"] = set()
            st.rerun()

    if not P.get("src"):
        st.info("Unggah file Excel atau klik **Muat data demo** untuk memulai.")
        return
    if P.get("error") and "sheets" not in P:
        st.error(P["error"])
        return

    pick("Sheet", "sheet", P["sheets"], help=f"Default: '{DEFAULT_SHEET}' bila tersedia.")
    if "raw" in P:
        render_mapping(P)
    if not guard(P, 1):
        return
    raw = P["raw"]
    st.success(f"File: **{P['src']['name']}** · Sheet: **{P['sheet']}** · {raw.shape[0]:,} baris, {raw.shape[1]} kolom")

    st.markdown("**Cuplikan data**")
    st.dataframe(raw.head(10), **STRETCH)
    st.markdown("**Statistik deskriptif**")
    st.dataframe(P["df"][COLS_ASLI].describe().T.rename(index=P["disp"]).round(3), **STRETCH)
    with st.expander("📈 Distribusi data awal", expanded=True):
        show(plot_distribusi(P["df"], COLS_ASLI, "Distribusi Data Awal", labels=P["disp"]))
    nav_footer(1, status)
