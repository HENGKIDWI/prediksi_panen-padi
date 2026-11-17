"""Tahap 10-11: dekomposisi per kecamatan, backtest produktivitas, parameter & pelatihan luas panen."""
import streamlit as st

import ui
from core import data as D
from core import plots
from core.config import DEFAULT_PARAMS, FEATURES_DEKOMP, PRODUKTIVITAS_METODE, SPLIT_OPTIONS
from core.decomposition import siapkan_dekomp

ui.init_state()
ss = st.session_state


@st.cache_data(show_spinner="Menyiapkan dekomposisi ...", max_entries=4)
def _siapkan(data, cm: dict) -> dict:
    return siapkan_dekomp(data, cm, "auto")


st.title("🧩 Dekomposisi per Kecamatan")
st.caption("Tahap 10–11 notebook: produksi = luas panen (ELM/GA-ELM, musiman) × produktivitas (statistik).")
ui.require(ss.data is not None, "Siapkan data terlebih dahulu di halaman **Data & Pembersihan**.")
mp = ss.mapping

try:
    info = _siapkan(ss.data, mp["cm"])
except (ValueError, KeyError) as e:
    st.error(f"Dekomposisi tidak dapat disiapkan: {e}. Periksa pemetaan kolom di halaman Data.")
    st.stop()

metode_label = st.selectbox("Metode peramalan produktivitas", list(PRODUKTIVITAS_METODE), index=0,
                            help="Otomatis memilih metode dengan galat backtest terkecil.")
kode = PRODUKTIVITAS_METODE[metode_label]
if kode != "auto":
    info["metode_prod"] = kode
if ss.dk:
    ss.dk["info"]["metode_prod"] = info["metode_prod"]

tab_data, tab_model = st.tabs(["Data dekomposisi", "Parameter & pelatihan luas panen"])

with tab_data:
    a, b, c, d = st.columns(4)
    a.metric("Kecamatan", len(info["kecs"]))
    b.metric("Bulan teramati", f"{info['t_last'] - info['t_first'] + 1}")
    c.metric("Sel kecamatan-bulan", f"{len(info['grid_full']):,}")
    d.metric("Sel dipakai model", f"{len(info['data_model']):,}")
    st.caption("Bulan tanpa baris pada suatu kecamatan dianggap luas panen 0. "
               f"Fitur model: {', '.join(FEATURES_DEKOMP)} (hanya dari tahun-tahun sebelumnya).")
    st.plotly_chart(plots.fig_total_luas(info))
    st.plotly_chart(plots.fig_bt_produktivitas(info["bt_prod"], info["metode_prod"]))
    st.success(f"Metode produktivitas dipakai: **{info['metode_prod']}**")

with tab_model:
    split_label = st.selectbox("Rasio split data (terurut)", list(SPLIT_OPTIONS), index=1, key="dk_split")
    an = ss.analisis
    use_ga, nama_model, mode = ui.pilih_model_dan_mode("dk", ada_analisis=bool(an))

    with st.form("form_dekomp"):
        S = None
        base = dict(DEFAULT_PARAMS, ga_metric="WAPE (%)")      # WAPE: MAPE tidak stabil untuk luas panen bernilai 0/kecil
        if mode == ui.MODE_MANUAL:
            p_in = ui.param_inputs("dk", base, use_ga)
        elif mode.startswith("Sama seperti"):
            p_in = dict(an["params"], ga_metric="WAPE (%)")
            st.caption(f"Memakai parameter Analisis: L={p_in['L']}, l2={p_in['l2_reg']:g}, aktivasi={p_in['act']}.")
        else:
            st.caption("Parameter dicari otomatis pada data validasi (bagian akhir data latih).")
            S = ui.search_settings("dk")
            p_in = base
        kirim = st.form_submit_button("Latih luas panen", type="primary")

    if kirim:
        dm = info["data_model"]
        df_train, df_test = D.split_sequential(dm, SPLIT_OPTIONS[split_label])
        p, model, out = ui.latih_dengan_mode(df_train, FEATURES_DEKOMP, info["luas_col"], p_in, use_ga,
                                             ui.MODE_ELM if mode.startswith("Sama") else mode, S)
        m_tr, m_te, yp_tr, yp_te = ui.evaluasi_dua_bagian(model, df_train, df_test)
        ss.dk = {"info": info, "params": p, "base": p_in, "use_ga": use_ga, "nama": nama_model, "model": model,
                 "search": out, "m_tr": m_tr, "m_te": m_te, "y_te": df_test[info["luas_col"]].values, "yp_te": yp_te,
                 "n_tr": len(df_train), "n_te": len(df_test), "split": split_label}
        ss.roll = None
        ss.forecast = None

    dk = ss.dk
    if dk:
        st.subheader(f"Hasil luas panen: {dk['nama']}")
        ui.kartu_metrik(dk["m_te"], ("RMSE", "WAPE (%)", "R2"))
        ui.tampil_df(ui.tabel_metrik(dk["m_tr"], dk["m_te"], dk["n_tr"], dk["n_te"]), "{:,.4f}")
        st.caption("Luas panen banyak bernilai 0, jadi baca WAPE (bukan MAPE). "
                   "Evaluasi PRODUKSI yang adil ada di halaman *Evaluasi Bergulir*.")
        st.plotly_chart(plots.fig_actual_vs_pred(dk["y_te"], dk["yp_te"], "Luas panen: aktual vs prediksi (data uji)"))
        if dk["model"].history:
            st.plotly_chart(plots.fig_ga_history(dk["model"].history))
        if dk["search"]:
            ui.tampil_hasil_pencarian(dk["search"], dk["base"], dk["params"], dk["use_ga"])
        st.success("Parameter tersimpan. Lanjut ke **Evaluasi Bergulir** dan **Peramalan Q1–Q4**.", icon="✅")
