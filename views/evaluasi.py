"""Tahap 9 — Evaluasi & Peramalan."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.config import TARGET
from core.forecast import hitung_metrik, tabel_per_q
from core.plots import plot_evaluasi, plot_q
from state import unit_factor
from ui import STRETCH, csv_button, nav_footer, show, stage_header
from views.peramalan import render as render_peramalan


def render(P, T, status):
    stage_header(9)
    fak, nm = unit_factor(), T["nama"]
    df_test = P["df_test"]
    aktual = df_test[TARGET].values * fak
    pred = T["pred_uji"] * fak
    st.caption(f"Prediksi = {1 - T['blend']:.0%} model + {T['blend']:.0%} rata-rata produksi bulan sama · rata-rata {T['n_s']} seed · "
               "fitur uji = rata-rata historis per bulan; lag memakai riwayat latih lalu hasil ramalan (tanpa produksi aktual periode uji).")
    met = hitung_metrik(aktual, pred)
    a, b, c, d = st.columns(4)
    a.metric("MAE", f"{met['MAE']:,.1f}")
    b.metric("RMSE", f"{met['RMSE']:,.1f}")
    c.metric("MAPE", f"{met['MAPE (%)']:.2f}%")
    d.metric("R²", f"{met['R2']:.4f}")

    tab_bln = pd.DataFrame({"Aktual (Ton)": aktual, "Prediksi (Ton)": pred}, index=df_test.index.astype(str))
    tab_bln["Selisih (Ton)"] = tab_bln["Prediksi (Ton)"] - tab_bln["Aktual (Ton)"]
    tab_bln["Error (%)"] = tab_bln["Selisih (Ton)"].abs() / tab_bln["Aktual (Ton)"].replace(0, np.nan) * 100

    tab_q = tabel_per_q(df_test.index, aktual, pred)
    met_q = hitung_metrik(tab_q["Aktual (Ton)"], tab_q["Prediksi (Ton)"])
    tab_q.loc["Total"] = [tab_q["Aktual (Ton)"].sum(), tab_q["Prediksi (Ton)"].sum(),
                          tab_q["Prediksi (Ton)"].sum() - tab_q["Aktual (Ton)"].sum(),
                          abs(tab_q["Prediksi (Ton)"].sum() - tab_q["Aktual (Ton)"].sum()) / tab_q["Aktual (Ton)"].sum() * 100]

    t1, t2, t3 = st.tabs(["📅 Evaluasi per bulan", "🗓️ Evaluasi per Q", f"🔮 Peramalan {T['tahun_ramal']}"])
    with t1:
        show(plot_evaluasi(df_test.index, aktual, pred, nm,
                           f"Aktual vs prediksi per bulan (uji {df_test.index.min()} s.d. {df_test.index.max()})", "Ton"))
        st.dataframe(tab_bln.round(1), **STRETCH)
        csv_button(tab_bln.round(2), "evaluasi_per_bulan.csv", key="dl_bln")
    with t2:
        show(plot_q(tab_q.iloc[:-1], nm, "Aktual vs prediksi per Q (periode uji)", "Ton"))
        st.dataframe(tab_q.round(1), **STRETCH)
        st.caption(f"MAPE per Q = **{met_q['MAPE (%)']:.2f}%** · MAE per Q = **{met_q['MAE']:,.0f} Ton**")
        csv_button(tab_q.round(2), "evaluasi_per_q.csv", key="dl_q")
    with t3:
        render_peramalan(P, T, fak)

    if T["robust"]:
        st.markdown("#### 🛡️ Uji ketahanan (robustness)")
        st.caption("Memeriksa apakah akurasi model stabil, bukan kebetulan bagus di satu periode uji. Untuk tiap tahun di bawah, "
                   "model dilatih hanya dengan data sebelum tahun itu, lalu meramal 12 bulan tahun tsb. dan dibandingkan dengan "
                   "data aktual. Error yang mirip dengan evaluasi utama = model konsisten; jauh lebih buruk = model rapuh.")
    for Y, (per_y, akt_y, pred_y) in T["robust"].items():
        with st.expander(f"Tahun {Y} (latih hanya dari tahun sebelumnya)", expanded=False):
            akt_y, pred_y = akt_y * fak, pred_y * fak
            m_y = hitung_metrik(akt_y, pred_y)
            t_y = tabel_per_q(per_y, akt_y, pred_y)
            mq_y = hitung_metrik(t_y["Aktual (Ton)"], t_y["Prediksi (Ton)"])
            st.dataframe(pd.DataFrame([m_y]).round(4), hide_index=True, **STRETCH)
            g1, g2 = st.columns([1.6, 1])
            with g1:
                show(plot_evaluasi(per_y, akt_y, pred_y, nm, f"Aktual vs prediksi per bulan ({Y})", "Ton"))
            with g2:
                show(plot_q(t_y, nm, f"Per Q ({Y})", "Ton"))
            st.dataframe(t_y.round(1), **STRETCH)
            st.caption(f"MAPE per Q = **{mq_y['MAPE (%)']:.2f}%**")
    nav_footer(9, status)
