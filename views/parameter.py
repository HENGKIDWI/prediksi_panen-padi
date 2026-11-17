"""Tahap 7 — Parameter Model."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import OPT, PARAM_MODE_LABEL, PARAM_MODES, SEARCH
from core.plots import plot_grid_elm, plot_pencarian_neuron, plot_sensitivitas_ga
from core.search import jalankan_pencarian
from state import manual_params, sane, search_cfg, search_sig
from ui import STRETCH, fmt_g, nav_footer, pick, show, stage_header


def render(P, status):
    stage_header(7)
    use_ga, nama = P["use_ga"], P["nama_model"]
    modes = PARAM_MODES if use_ga else PARAM_MODES[:2]
    pick("Parameter model", "param_mode", modes, fmt=PARAM_MODE_LABEL.get)
    mode = st.session_state["param_mode"]

    title = "Parameter manual" if mode == "manual" else "Parameter dasar (nilai awal pencarian)"
    with st.container(border=True):
        st.markdown(f"**{title} — ELM**")
        c = st.columns(5)
        with c[0]:
            pick("Fungsi aktivasi", "p_act", OPT["act"])
        with c[1]:
            pick("Hidden neuron (L)", "p_L", OPT["L"], fmt=fmt_g)
        with c[2]:
            pick("Regularisasi l2_reg", "p_l2_reg", OPT["l2_reg"], fmt=fmt_g)
        with c[3]:
            pick("Rentang bobot awal", "p_w_range", OPT["w_range"], fmt=fmt_g)
        with c[4]:
            pick("Seed", "p_ga_seed", OPT["ga_seed"], fmt=fmt_g)
    if use_ga:
        with st.container(border=True):
            st.markdown(f"**{title} — GA**")
            r1 = st.columns(4)
            with r1[0]:
                pick("Ukuran populasi", "p_pop_size", OPT["pop_size"], fmt=fmt_g)
            with r1[1]:
                pick("Generasi maksimum", "p_generations", OPT["generations"], fmt=fmt_g)
            with r1[2]:
                pick("Crossover rate", "p_cr", OPT["cr"], fmt=fmt_g)
            with r1[3]:
                pick("Mutation rate", "p_mr", OPT["mr"], fmt=fmt_g)
            r2 = st.columns(4)
            with r2[0]:
                pick("Porsi gen dimutasi", "p_mut_gene_rate", OPT["mut_gene_rate"], fmt=fmt_g)
            with r2[1]:
                pick("Langkah mutasi", "p_mut_step", OPT["mut_step"], fmt=fmt_g)
            with r2[2]:
                pick("Bobot error validasi", "p_ga_val_weight", OPT["ga_val_weight"], fmt=fmt_g,
                     help="0 = hanya data latih, 1 = hanya validasi.")
            with r2[3]:
                pick("Metrik fitness", "p_ga_metric", OPT["ga_metric"])
            r3 = st.columns(4)
            with r3[0]:
                pick("Rasio validasi GA", "p_ga_val_ratio", OPT["ga_val_ratio"], fmt=fmt_g)
            with r3[1]:
                pick("Patience (berhenti dini)", "p_ga_patience", OPT["ga_patience"],
                     fmt=lambda v: "Nonaktif" if v == 0 else f"{v} generasi")

    if mode == "manual":
        st.success("Parameter manual dipakai apa adanya.")
    else:
        with st.expander("🔧 Pengaturan pencarian otomatis", expanded=False):
            c = st.columns(4)
            with c[0]:
                pick("L minimum", "s_L_min", OPT["L_min"])
            with c[1]:
                pick("L maksimum", "s_L_max", OPT["L_max"])
            with c[2]:
                pick("Langkah L", "s_L_step", OPT["L_step"])
            with c[3]:
                pick("Rasio validasi seleksi", "s_val_ratio", OPT["s_val_ratio"], fmt=fmt_g)
            c = st.columns(4)
            with c[0]:
                pick("Seed per kandidat (ELM)", "s_n_seeds", OPT["s_n_seeds"])
            with c[1]:
                pick("Seed per kandidat (GA)", "s_n_seeds_ga", OPT["s_n_seeds_ga"])
            with c[2]:
                pick("Patience kandidat L", "s_patience", OPT["s_patience"])
            with c[3]:
                pick("Generasi GA saat pencarian", "s_ga_gen", OPT["s_ga_gen"])
            c = st.columns(3)
            with c[0]:
                pick("Aturan 1-std (model sederhana)", "s_se_rule", OPT["s_se_rule"],
                     fmt=lambda v: "Ya" if v else "Tidak",
                     help="Pilih L terkecil yang RMSE-nya masih dalam 1 std dari terbaik.")
            with c[1]:
                sane("s_grid_act", OPT["act"])
                st.session_state["s_grid_act"] = [a for a in st.session_state["s_grid_act"] if a in OPT["act"]]
                st.multiselect("Kandidat aktivasi", OPT["act"], key="s_grid_act")
            with c[2]:
                st.session_state["s_grid_l2"] = [v for v in st.session_state["s_grid_l2"] if v in OPT["l2_reg"]]
                st.multiselect("Kandidat l2_reg", OPT["l2_reg"], key="s_grid_l2", format_func=fmt_g)

        sig = search_sig(P)
        res = st.session_state.get("search")
        valid = bool(res) and res["sig"] == sig
        if st.button("🔎 Jalankan pencarian parameter", type="primary", **STRETCH):
            S = search_cfg()
            base = manual_params()
            with st.status(f"Mencari parameter terbaik untuk {nama} ...", expanded=True) as stt:
                hasil = jalankan_pencarian(P["df_train"], base, S, use_ga, cari_ga=(mode == "full"),
                                           feats=P["feats"], log=stt.write)
                p = dict(base)
                p.update(hasil["cfg"])
                kunci = ["L", "act", "l2_reg", "w_range"] + (list(SEARCH["grid_ga"]) if (use_ga and mode == "full") else [])
                ringkas = pd.DataFrame({"Awal": {k: base[k] for k in kunci}, "Terpilih": {k: p[k] for k in kunci}}).astype(str)
                stt.update(label="Pencarian selesai", state="complete", expanded=False)
            st.session_state["search"] = {"sig": sig, "params": p, "ringkas": ringkas,
                                          "elm": hasil["elm"], "elm_pilih": hasil["elm_pilih"],
                                          "gaL": hasil.get("gaL"), "gaL_pilih": hasil.get("gaL_pilih"),
                                          "sens": hasil.get("sens")}
            st.rerun()
        if not valid:
            st.warning("Belum ada hasil pencarian untuk pengaturan saat ini. Klik tombol di atas "
                       "(proses bisa memakan waktu beberapa menit, terutama mode GA).")
        else:
            st.success("Pencarian selesai — parameter terpilih dipakai untuk pelatihan.")
            st.markdown("**Ringkasan parameter (awal → terpilih)**")
            st.dataframe(res["ringkas"], **STRETCH)
            tabs = st.tabs(["Grid ELM"] + (["Neuron GA-ELM"] if res["gaL"] is not None else [])
                           + (["Sensitivitas GA"] if res["sens"] is not None else []))
            with tabs[0]:
                show(plot_grid_elm(res["elm"], res["elm_pilih"]))
            i = 1
            if res["gaL"] is not None:
                with tabs[i]:
                    show(plot_pencarian_neuron(res["gaL"], res["gaL_pilih"], "Jumlah Neuron GA-ELM"))
                i += 1
            if res["sens"] is not None:
                with tabs[i]:
                    show(plot_sensitivitas_ga(res["sens"]))

    p = P.get("params")
    if p:
        st.caption(f"Parameter yang akan dipakai: L={p['L']} · l2_reg={p['l2_reg']:g} · aktivasi={p['act']} · w_range={p['w_range']:g}")
    nav_footer(7, status)
