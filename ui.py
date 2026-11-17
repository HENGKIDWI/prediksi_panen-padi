"""Komponen UI bersama untuk semua tahap (satu-satunya modul selain `state` dan `views/` yang memakai Streamlit)."""
from __future__ import annotations

import inspect
from pathlib import Path

import streamlit as st

from core.config import DEFAULTS, NONE, ROLE_LABEL, ROLES, STAGES
from state import advance, go, reset_all, sane

LOGO = Path(__file__).parent / "assets" / "logo1.png"

# Kompatibilitas lintas versi Streamlit: `width="stretch"` (baru) vs `use_container_width=True` (lama)
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})


def show(fig):
    st.pyplot(fig)


def pick(label, key, options, fmt=None, help=None):
    """Dropdown standar: nilai disimpan di session_state[key], tanpa `index` ganda."""
    sane(key, options, DEFAULTS.get(key))
    return st.selectbox(label, options, key=key, format_func=fmt or str, help=help)


def fmt_g(v):
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def render_mapping(P):
    opts = [NONE] + P["columns"]
    with st.container(border=True):
        st.markdown("**🧭 Pemetaan kolom** — cocokkan peran data dengan kolom di file Anda "
                    "(ditebak otomatis; ubah bila keliru). Nama kolom Anda dipakai di seluruh tampilan.")
        grid = st.columns(3)
        for i, role in enumerate(ROLES):
            with grid[i % 3]:
                pick(ROLE_LABEL[role], f"map_{i}", opts)


def stage_header(n):
    title, icon, desc = STAGES[n]
    st.markdown(f"### {icon} Tahap {n} — {title}")
    st.caption(desc)


def nav_footer(n, status):
    last = len(STAGES)
    st.divider()
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        st.button("◀ Sebelumnya", key=f"prev_{n}", on_click=go, args=(n - 1,), **STRETCH)
    with c2:
        if n < last:
            st.button("Lanjut ▶", key=f"next_{n}", on_click=advance, args=(n,),
                      disabled=not status["ready"][n], type="primary", **STRETCH,
                      help="Konfirmasi tahap ini dan buka tahap berikutnya.")


def locked_notice(n, status, P):
    first_open = next((k for k in range(1, len(STAGES) + 1) if not status["done"][k]), 1)
    stage_header(n)
    if P.get("error"):
        st.error(P["error"])
    st.warning(f"🔒 Tahap {n} belum bisa dibuka. Selesaikan **Tahap {first_open} — {STAGES[first_open][0]}** "
               "lalu klik **Lanjut**.")
    st.button(f"Ke Tahap {first_open} ➔", on_click=go, args=(first_open,), type="primary")


def guard(P, level):
    """Hentikan render tahap bila hasil pipeline belum tersedia."""
    if P["level"] >= level:
        return True
    st.error(P.get("error", "Data belum siap."))
    return False


def csv_button(df, name, label="⬇️ Unduh CSV", key=None):
    st.download_button(label, df.to_csv().encode("utf-8-sig"), file_name=name, mime="text/csv", key=key, **STRETCH)


def stage_label(n, status):
    if n == 0:
        return "🏠 Beranda"
    mark = "✅" if status["done"][n] else ("🔓" if status["open"][n] else "🔒")
    return f"{mark} {n}. {STAGES[n][0]}"


def render_sidebar(P, status):
    with st.sidebar:
        if LOGO.exists():
            st.image(str(LOGO), width=120)
        else:
            st.markdown("# 🌾")

        st.markdown("## Peramalan Padi")
        st.caption("ELM / GA-ELM · Deret bulanan total")
        st.radio("Tahapan", list(range(0, len(STAGES) + 1)), key="stage", format_func=lambda n: stage_label(n, status))
        st.divider()
        st.markdown("**Ringkasan konfigurasi**")
        src = P.get("src")
        st.caption(f"📂 {src['name'] if src else '—'}")
        if P["level"] >= 3:
            st.caption(f"🧹 {P['label_data']} · ✂️ {P.get('split_label', '—')}")
        p = P.get("params")
        st.caption(f"🧠 {P['nama_model']}" + (f" · L={p['L']} · {p['act']}" if p else ""))
        st.button("♻️ Reset semua", on_click=reset_all, **STRETCH, help="Hapus data, pilihan, dan hasil pelatihan.")
