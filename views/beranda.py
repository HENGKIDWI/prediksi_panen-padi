"""Tahap 0 — Beranda."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import STAGES, TARGET
from core.forecast import hitung_metrik
from state import go, load_demo, unit_factor
from ui import STRETCH


def render(P, T, status):
    st.markdown("""
<div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%); border-radius: 20px;
            padding: 34px 38px; color: white; margin-bottom: 22px; box-shadow: 0 12px 32px rgba(6,78,59,.22);">
  <div style="display:inline-block; background: rgba(52,211,153,.2); border: 1px solid rgba(167,243,208,.4);
              padding: 4px 14px; border-radius: 20px; font-size: .8rem; font-weight: 700; letter-spacing: 1px;
              text-transform: uppercase; color: #a7f3d0; margin-bottom: 12px;">
    MBKM Riset • Artificial Intelligence & Agriculture 4.0
  </div>
  <h1 style="font-size: 2.2rem; font-weight: 800; margin: 0 0 12px 0; color: #ffffff; letter-spacing: -.5px;">
    Sistem Cerdas Peramalan Produksi Padi
  </h1>
  <p style="font-size: 1.03rem; margin: 0; color: #d1fae5; line-height: 1.6; max-width: 900px;">
    Peramalan deret waktu produksi padi <strong>total bulanan</strong> dengan
    <strong>Extreme Learning Machine (ELM)</strong> dan <strong>GA-ELM</strong> (bobot dioptimasi Genetic Algorithm),
    dievaluasi secara rekursif tanpa produksi aktual dan dilaporkan per triwulan (Q1–Q4).
  </p>
</div>
""", unsafe_allow_html=True)

    c1, c2, _ = st.columns([1.6, 1.8, 2.6])
    with c1:
        st.button("🚀 Mulai Pipeline ➔", on_click=go, args=(1,), type="primary", **STRETCH)
    with c2:
        st.button("⚡ Muat Data Demo (5 Tahun)", on_click=load_demo, **STRETCH,
                  help="Memuat data simulasi (6 kecamatan × 60 bulan) agar semua tahap bisa dicoba.")

    k1, k2, k3, k4 = st.columns(4)
    raw = P.get("raw")
    with k1:
        if raw is not None:
            st.metric("📊 Dataset", f"{len(raw):,} Baris", f"{raw.shape[1]} Kolom", delta_color="off")
        else:
            st.metric("📊 Dataset", "Belum Ada", "Tahap 1", delta_color="off")
    with k2:
        if P["level"] >= 4:
            st.metric("📅 Deret Bulanan", f"{len(P['agg'])} Bulan", f"{len(P['d_all'])} sampel · {len(P['feats'])} fitur", delta_color="off")
        else:
            st.metric("📅 Deret Bulanan", "Belum Dibentuk", "Tahap 4", delta_color="off")
    with k3:
        p = P.get("params")
        if P["level"] >= 5 and p:
            st.metric("🧠 Model Aktif", P["nama_model"], f"L={p['L']} · {p['act']}", delta_color="off")
        else:
            st.metric("🧠 Model Aktif", P["nama_model"], "Tahap 6–7", delta_color="off")
    with k4:
        if T:
            fak = unit_factor()
            met = hitung_metrik(P["df_test"][TARGET].values * fak, T["pred_uji"] * fak)
            st.metric("📈 MAPE Uji", f"{met['MAPE (%)']:.2f}%", f"R² {met['R2']:.3f}", delta_color="off")
        else:
            st.metric("📈 MAPE Uji", "—", "Menunggu evaluasi", delta_color="off")

    st.markdown("#### 🗺️ Alur Kerja & Tahapan Pipeline")
    st.caption("Tahap dibuka bertahap agar urutan metodologi deret waktu tetap terjaga.")
    for start in range(1, len(STAGES) + 1, 3):
        cols = st.columns(3)
        for j, n in enumerate(range(start, min(start + 3, len(STAGES) + 1))):
            title, icon, desc = STAGES[n]
            done, op = status["done"][n], status["open"][n]
            badge = "✅ Selesai" if done else ("🔓 Terbuka" if op else "🔒 Terkunci")
            with cols[j]:
                with st.container(border=True):
                    st.markdown(f"**{icon} {n}. {title}**")
                    st.caption(f"{badge}")
                    st.write(desc)
                    st.button(f"Buka Tahap {n} ➔" if op else "Terkunci 🔒", key=f"open_{n}", on_click=go, args=(n,),
                              disabled=not op, **STRETCH)

    st.markdown("#### 🔬 Panduan Metodologi")
    t1, t2, t3 = st.tabs(["🧠 ELM vs GA-ELM", "🛡️ Protokol Anti-Data Leakage", "📐 Interpretasi Metrik"])
    with t1:
        a, b = st.columns(2)
        with a, st.container(border=True):
            st.markdown("**1. ELM (Extreme Learning Machine)**")
            st.write("Jaringan satu *hidden layer*. Bobot input **W** dan bias **b** diacak, hanya bobot output **β** "
                     "yang dihitung analitik (*ridge regression*): β = (HᵀH + λI)⁻¹ HᵀY.")
            st.caption("Parameter: jumlah neuron L, fungsi aktivasi, l2_reg (λ), rentang bobot awal.")
        with b, st.container(border=True):
            st.markdown("**2. GA-ELM (Genetic Algorithm + ELM)**")
            st.write("Kromosom = seluruh W dan b. GA (roulette, crossover aritmetik, mutasi, elitism) mencari bobot "
                     "yang meminimalkan error (MAPE/WAPE) pada data latih dan validasi internal.")
            st.caption("Parameter GA: populasi, generasi, crossover/mutation rate, bobot validasi, patience.")
    with t2:
        with st.container(border=True):
            st.markdown("**Prinsip yang diterapkan**")
            st.markdown(
                "- **Split terurut waktu** — tanpa *shuffle*; bagian awal untuk latih, akhir untuk uji.\n"
                "- **Normalisasi min-max dipelajari dari data latih saja**, lalu diterapkan ke data uji/ramal.\n"
                "- **Evaluasi rekursif tanpa produksi aktual** — fitur uji = rata-rata historis per bulan, "
                "lag = produksi bulan-bulan sebelumnya; yang jatuh di periode uji/ramal diisi hasil ramalan, bukan aktual.\n"
                "- **Seleksi parameter** memakai validasi di *ujung data latih* (terurut); data uji tidak disentuh.\n"
                "- **Prediksi akhir** = (1−w) × model + w × rata-rata produksi bulan yang sama (blend musiman), "
                "dirata-ratakan lintas seed.")
    with t3:
        met_tbl = pd.DataFrame({
            "Metrik": ["MAE", "RMSE", "MAPE (%)", "R²"],
            "Definisi": ["Rata-rata selisih absolut", "Akar rata-rata selisih kuadrat",
                         "Rata-rata persentase kesalahan absolut", "Variansi target yang dijelaskan model"],
            "Interpretasi": ["Semakin kecil semakin baik (satuan Ton).", "Sensitif terhadap lonjakan error besar.",
                             "< 10% sangat akurat · 10–20% akurat · 20–50% layak.",
                             "Mendekati 1 = sangat baik; negatif = lebih buruk dari rata-rata."]})
        st.dataframe(met_tbl, hide_index=True, **STRETCH)
