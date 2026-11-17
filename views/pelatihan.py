"""Tahap 8 — Pelatihan Model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.config import OPT, SERI
from core.forecast import build_model, evaluasi_tahun, latih_nilai, ramal_rekursif
from core.plots import plot_ga_history
from state import train_sig
from ui import STRETCH, nav_footer, pick, show, stage_header


def _months_left(agg, year):
    last = agg.index.max()
    return (year * 12 + 12) - (last.year * 12 + last.month)


def render(P, T, status):
    stage_header(8)
    params = P["params"]
    agg, d_all, df_test = P["agg"], P["d_all"], P["df_test"]
    last_year = int(agg.index.max().year)
    # Hanya 1 tahun ke depan: tahun berikutnya akan memakai lag/ramalan sebelumnya sebagai input (galat menumpuk)
    ramal_opts = [y for y in range(last_year, last_year + 4) if _months_left(agg, y) > 0][:1]
    uji_opts = [y for y in sorted(set(agg["tahun"]))
                if (agg["tahun"] == y).sum() == 12 and len(d_all[d_all["tahun"] < y]) >= 12
                and y not in set(df_test.index.year)]

    c = st.columns(4)
    with c[0]:
        pick("Tahun yang diramal", "tahun_ramal", ramal_opts, help="Hanya 1 tahun ke depan (sampai Desember). Tahun berikutnya akan memakai hasil ramalan sebagai input sehingga kurang andal.")
    with c[1]:
        pick("Jumlah seed (rata-rata)", "n_seeds", OPT["n_seeds"])
    with c[2]:
        pick("Blend musiman (w)", "blend", OPT["blend"], fmt=lambda v: f"{v:.0%}" + (" (model murni)" if v == 0 else ""),
             help="Prediksi = (1−w) × model + w × rata-rata produksi bulan yang sama.")
    with c[3]:
        st.session_state["tahun_uji"] = [y for y in st.session_state["tahun_uji"] if y in uji_opts]
        st.multiselect("Uji ketahanan tahun lain", uji_opts, key="tahun_uji",
                       help="Cek apakah akurasi model stabil di tahun lain: latih hanya dari tahun sebelumnya, "
                            "lalu uji 12 bulan tahun tsb.")

    n_s = st.session_state["n_seeds"]
    tahun_uji = list(st.session_state["tahun_uji"])
    st.caption(f"Model: **{P['nama_model']}** (L={params['L']}, aktivasi={params['act']}) · "
               f"≈ {2 * n_s + n_s * len(tahun_uji)} kali pelatihan" + (" — GA membuat proses lebih lama." if P["use_ga"] else "."))

    if T is None and st.session_state.get("train"):
        st.warning("Hasil pelatihan sebelumnya sudah kedaluwarsa karena pengaturan berubah. Latih ulang.")

    if st.button("🚀 Latih & Ramal", type="primary", **STRETCH):
        run_training(P, params, tahun_uji)
        st.rerun()

    if T:
        st.success(f"Pelatihan selesai — {T['n_s']} seed · peramalan {T['tahun_ramal']} ({len(T['periode_ramal'])} bulan). "
                   "Klik **Lanjut** untuk melihat hasil di Tahap 9.")
        if T["history"] is not None:
            show(plot_ga_history(T["history"]))
    nav_footer(8, status)


def run_training(P, params, tahun_uji):
    g = st.session_state.get
    n_s, w_m, tahun_ramal = g("n_seeds"), g("blend"), g("tahun_ramal")
    FEATS = P["feats"]
    use_ga, nama = P["use_ga"], P["nama_model"]
    df_train, df_test, agg, d_all, agg_latih = P["df_train"], P["df_test"], P["agg"], P["d_all"], P["agg_latih"]
    tahun_uji = [y for y in tahun_uji if y not in set(df_test.index.year)]
    periode_ramal = pd.period_range(agg.index.max() + 1, f"{tahun_ramal}-12", freq="M")
    if len(periode_ramal) == 0:
        st.error(f"Data sudah mencakup sampai {tahun_ramal}; pilih tahun setelah data.")
        return
    total = n_s + n_s + n_s * len(tahun_uji)
    done = {"n": 0}
    prog = st.progress(0.0, text="Menyiapkan ...")

    def step(text=None):
        done["n"] += 1
        prog.progress(min(done["n"] / total, 1.0), text=text or f"Model {done['n']}/{total}")

    with st.status(f"Melatih {nama} ...", expanded=True) as stt:
        stt.write(f"[a] Model evaluasi pada data latih ({len(df_train)} bulan) × {n_s} seed")
        R = latih_nilai(df_train, df_test, params, use_ga, FEATS, log=stt.write)   # seed utama: riwayat GA
        step("Model evaluasi 1")
        models_eval = [R["model"]]                                                  # identik dengan build_model(seed+0)
        for i in range(1, n_s):
            models_eval.append(build_model(df_train, dict(params, ga_seed=params["ga_seed"] + i), use_ga, FEATS))
            step(f"Model evaluasi {i + 1}")
        pred_uji = np.mean([ramal_rekursif(m, agg_latih, df_test.index, SERI, FEATS, w_m) for m in models_eval], axis=0)

        stt.write(f"[b] Model final pada seluruh data ({len(d_all)} bulan), ramal {len(periode_ramal)} bulan ke depan "
                  f"({1 - w_m:.0%} model + {w_m:.0%} rata-rata bulan sama)")
        models_final = []
        for i in range(n_s):
            models_final.append(build_model(d_all, dict(params, ga_seed=params["ga_seed"] + i), use_ga, FEATS))
            step(f"Model final {i + 1}")
        pred_seed = np.array([ramal_rekursif(m, agg, periode_ramal, SERI, FEATS, w_m) for m in models_final])

        robust = {}
        for Y in tahun_uji:
            stt.write(f"[c] Uji ketahanan tahun {Y} ...")
            r = evaluasi_tahun(agg, d_all, Y, params, use_ga, {**SERI, "blend_musiman": w_m}, n_s, FEATS,
                               step=lambda: step(f"Uji ketahanan {Y}"))
            if r is None:
                stt.write(f"(Uji ketahanan {Y} dilewati: data tidak cukup)")
            else:
                robust[Y] = r
        stt.update(label="Pelatihan & peramalan selesai", state="complete", expanded=False)
    prog.empty()

    st.session_state["train"] = {
        "sig": train_sig(P, params), "history": R["history"], "pred_uji": pred_uji, "pred_seed": pred_seed,
        "periode_ramal": periode_ramal, "robust": robust, "n_s": n_s, "blend": w_m, "tahun_ramal": tahun_ramal,
        "params": params, "nama": nama,
    }
