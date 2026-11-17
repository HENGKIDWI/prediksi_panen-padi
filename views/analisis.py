"""Tahap 4-9: analisis hubungan dengan fitur periode yang sama (bukan peramalan)."""
import pandas as pd
import streamlit as st

import ui
from core import data as D
from core import plots
from core.config import DEFAULT_PARAMS, SPLIT_OPTIONS

ui.init_state()
ss = st.session_state

st.title("📈 Analisis Hubungan")
st.caption("Tahap 4–9 notebook: split, model, parameter, pelatihan, pengujian, bobot & bias.")
ui.require(ss.data is not None, "Siapkan data terlebih dahulu di halaman **Data & Pembersihan**.")
st.info("Tahap ini memakai fitur pada periode yang sama (mis. luas panen, produktivitas), jadi hasilnya adalah "
        "**analisis hubungan, bukan peramalan**. Peramalan ada di halaman *Dekomposisi per Kecamatan*.", icon="ℹ️")

mp, data = ss.mapping, ss.data
split_label = st.selectbox("Rasio split data (terurut, tanpa acak)", list(SPLIT_OPTIONS), index=1)
use_ga, nama_model, mode = ui.pilih_model_dan_mode("an")

with st.form("form_analisis"):
    S = None
    if mode == ui.MODE_MANUAL:
        p_in = ui.param_inputs("an", DEFAULT_PARAMS, use_ga)
    else:
        st.caption("Parameter dicari otomatis pada data validasi (bagian akhir data latih); data uji tidak disentuh.")
        S = ui.search_settings("an")
        p_in = dict(DEFAULT_PARAMS)
    kirim = st.form_submit_button("Latih & evaluasi", type="primary")

if kirim:
    df_train, df_test = D.split_sequential(data, SPLIT_OPTIONS[split_label])
    p, model, out = ui.latih_dengan_mode(df_train, mp["features"], mp["target"], p_in, use_ga, mode, S)
    m_tr, m_te, yp_tr, yp_te = ui.evaluasi_dua_bagian(model, df_train, df_test)
    ss.analisis = {"nama": nama_model, "use_ga": use_ga, "split": split_label, "params": p, "base": p_in,
                   "search": out, "model": model, "m_tr": m_tr, "m_te": m_te,
                   "y_te": df_test[mp["target"]].values, "yp_te": yp_te, "n_tr": len(df_train), "n_te": len(df_test),
                   "n_nol": int((df_test[mp["target"]] == 0).sum() + (df_train[mp["target"]] == 0).sum())}

res = ss.analisis
if res is None:
    st.stop()

st.subheader(f"Hasil {res['nama']}  ({res['split']})")
ui.kartu_metrik(res["m_te"])
tabs = st.tabs(["Evaluasi", "Aktual vs prediksi", "Konvergensi GA", "Bobot & bias", "Pencarian parameter"])
with tabs[0]:
    ui.tampil_df(ui.tabel_metrik(res["m_tr"], res["m_te"], res["n_tr"], res["n_te"]), "{:,.4f}")
    if res["n_nol"]:
        st.caption(f"{res['n_nol']} data bernilai aktual 0 dikeluarkan dari MAPE (WAPE tidak terpengaruh).")
    st.caption(f"Parameter: L={res['params']['L']}, l2={res['params']['l2_reg']:g}, aktivasi={res['params']['act']}.")
with tabs[1]:
    st.plotly_chart(plots.fig_actual_vs_pred(res["y_te"], res["yp_te"], f"{res['nama']}: aktual vs prediksi (data uji)"))
with tabs[2]:
    if res["model"].history:
        st.plotly_chart(plots.fig_ga_history(res["model"].history))
    else:
        st.info("Hanya tersedia untuk GA-ELM.")
with tabs[3]:
    m = res["model"]
    st.plotly_chart(plots.fig_weights(m.W, m.b, m.norm.features))
    with st.expander("Tabel bobot input W"):
        st.dataframe(pd.DataFrame(m.W, index=m.norm.features, columns=[f"N{i + 1}" for i in range(m.W.shape[1])]).round(4))
    st.caption("ELM: W & b acak (tidak dilatih). GA-ELM: hasil optimasi GA. Bobot input bukan ukuran pentingnya fitur.")
with tabs[4]:
    if res["search"]:
        ui.tampil_hasil_pencarian(res["search"], res["base"], res["params"], res["use_ga"])
    else:
        st.info("Mode parameter manual: tidak ada pencarian.")
