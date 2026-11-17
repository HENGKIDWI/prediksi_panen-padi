"""Tahap 4 — Fitur Lag."""
from __future__ import annotations

import streamlit as st

from core.config import LAG_OPTIONS, OPT
from core.plots import plot_deret
from state import dn, unit_factor
from ui import STRETCH, guard, nav_footer, pick, show, stage_header


def render(P, status):
    stage_header(4)
    c1, c2 = st.columns([2, 1])
    with c1:
        pick("Fitur lag yang dipakai", "lag_key", list(LAG_OPTIONS), fmt=lambda k: LAG_OPTIONS[k][1],
             help="Lag k = produksi k bulan sebelumnya. Nilainya dihitung otomatis dari kolom produksi (tanpa input manual).")
    with c2:
        pick("Satuan produksi pada data", "satuan", OPT["satuan"],
             help="Dipakai untuk konversi tampilan ke Ton/Kuintal (1 ton = 10 kuintal).")
    with st.expander("ℹ️ Apa itu deret bulanan dan fitur lag?"):
        st.markdown(
            "- **Deret bulanan:** semua kecamatan digabung jadi satu baris per bulan "
            "(luas, pupuk, produksi dijumlahkan; curah hujan & kelembapan dirata-ratakan).\n"
            "- **Lag k:** produksi *k* bulan sebelumnya sebagai fitur. **Lag 1** menangkap kesinambungan jangka pendek; "
            "**lag 12** menangkap pola musiman (produksi bulan yang sama tahun lalu).\n"
            "- Tiap lag membuang baris awal yang belum punya nilai lag, sehingga sampel berkurang.\n"
            "- **Lag 1** memaksa peramalan *rekursif* (lag bulan berikutnya = hasil ramalan sendiri, galat menumpuk). "
            "**Lag 12** untuk horizon ≤ 12 bulan memakai produksi aktual tahun lalu, sehingga tidak ada galat yang menumpuk.")
    if not guard(P, 4):
        return
    agg, d_all, feats = P["agg"], P["d_all"], P["feats"]
    fak = unit_factor()
    a, b, c, d = st.columns(4)
    a.metric("Periode", f"{agg.index.min()} → {agg.index.max()}")
    b.metric("Jumlah bulan", len(agg), f"{P['n_kosong']} bulan kosong diisi 0", delta_color="off")
    c.metric("Sampel model", len(d_all), f"-{P['n_lag_hilang']} baris awal (lag)", delta_color="off")
    d.metric("Jumlah fitur", len(feats))
    st.markdown(f"**Fitur ({len(feats)}):** " + ", ".join(f"`{dn(P, f)}`" for f in feats))
    st.caption("Latih: fitur = nilai aktual bulan itu + lag aktual · Uji & ramal: fitur = rata-rata historis per bulan; "
               "lag diambil dari riwayat aktual bila masih tersedia, selain itu dari hasil ramalan sebelumnya.")
    if 1 not in P["lags"]:
        st.info("Tanpa lag 1, peramalan 12 bulan memakai produksi aktual tahun sebelumnya — tidak ada galat yang menumpuk.")
    st.dataframe(d_all.head(5).rename(columns=P["disp"]).round(2), **STRETCH)
    show(plot_deret(agg, fak, "Ton"))
    nav_footer(4, status)
