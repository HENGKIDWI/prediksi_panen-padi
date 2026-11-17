"""Peramalan Q1–Q4 tahun target (tab pada Tahap 9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.config import KUINTAL_PER_TON, SERI, TARGET
from core.plots import plot_q, plot_ramalan
from ui import STRETCH, csv_button, show


def render(P, T, fak):
    nm, n_s, thn = T["nama"], T["n_s"], T["tahun_ramal"]
    agg, periode = P["agg"], T["periode_ramal"]
    ps = T["pred_seed"] * fak
    rata, lo, hi = ps.mean(0), ps.min(0), ps.max(0)

    bln = pd.DataFrame({"Ton": rata, "Kuintal": rata * KUINTAL_PER_TON}, index=periode.astype(str))
    qlab = np.array([f"Q{(p.month - 1) // 3 + 1}" for p in periode])
    q_seed = np.array([[row[qlab == q].sum() for q in ["Q1", "Q2", "Q3", "Q4"]] for row in ps])
    ring = pd.DataFrame({"Ramalan (Ton)": q_seed.mean(0), "Min (Ton)": q_seed.min(0), "Maks (Ton)": q_seed.max(0)},
                        index=[f"Q{i}" for i in range(1, 5)])
    ring["Ramalan (Kuintal)"] = ring["Ramalan (Ton)"] * KUINTAL_PER_TON
    ak = agg[agg["tahun"] == thn - 1][TARGET] * fak
    ada_ak = len(ak) == 12
    if ada_ak:
        ring.insert(0, f"Aktual {thn - 1} (Ton)", [ak.values[i * 3:(i + 1) * 3].sum() for i in range(4)])
        ring[f"Selisih vs {thn - 1} (%)"] = (ring["Ramalan (Ton)"] / ring[f"Aktual {thn - 1} (Ton)"] - 1) * 100
    ring.loc["TOTAL"] = ring.iloc[:4].sum()
    if ada_ak:
        ring.loc["TOTAL", f"Selisih vs {thn - 1} (%)"] = (
            ring.loc["TOTAL", "Ramalan (Ton)"] / ring.loc["TOTAL", f"Aktual {thn - 1} (Ton)"] - 1) * 100

    st.caption(f"Peramalan {thn} (tanpa produksi aktual) · {nm} · rata-rata {n_s} seed · 1 ton = {KUINTAL_PER_TON} kuintal")
    cols = st.columns(5)
    for i, q in enumerate(["Q1", "Q2", "Q3", "Q4", "TOTAL"]):
        delta = f"{ring.loc[q, f'Selisih vs {thn - 1} (%)']:+.1f}% vs {thn - 1}" if ada_ak else None
        cols[i].metric(q, f"{ring.loc[q, 'Ramalan (Ton)']:,.0f} Ton", delta, delta_color="off")

    show(plot_ramalan(agg, periode, rata, lo, hi, nm, fak, "Ton"))
    show(plot_q(ring.iloc[:4], nm, f"Ramalan per Q {thn}", "Ton", kol_a=(f"Aktual {thn - 1} (Ton)" if ada_ak else None),
                kol_p="Ramalan (Ton)", lab_a=f"Aktual {thn - 1}", lab_p=f"Ramalan {nm}"))
    st.markdown("**Per Q**")
    st.dataframe(ring.round(1), **STRETCH)
    h = ring.copy()
    h.index.name = "Triwulan"
    csv_button(h.round(2), SERI["file_csv"], "⬇️ Unduh hasil triwulan (CSV)", key="dl_ring")
    st.markdown("**Per bulan**")
    st.dataframe(bln.round(1), **STRETCH)
    csv_button(bln.round(2), f"ramalan_bulanan_{thn}.csv", key="dl_bln_ramal")
