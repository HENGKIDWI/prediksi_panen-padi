"""Sistem Cerdas Peramalan Produksi Padi — ELM / GA-ELM (titik masuk).

Jalankan:  streamlit run app.py

Alur 9 tahap (sama dengan notebook GA_ELM_PERAMALAN_TOTAL_Q): Muat Data -> Cek Anomali -> Preprocessing -> Fitur Lag
-> Split Data -> Pilih Model -> Parameter Model -> Pelatihan Model -> Evaluasi & Peramalan.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))   # agar `core`, `state`, `ui`, `views` selalu dapat diimpor

import streamlit as st  # noqa: E402

import state  # noqa: E402
import ui  # noqa: E402
from views import anomali, beranda, data, deret, evaluasi, model, parameter, pelatihan, pilih_data, split  # noqa: E402

st.set_page_config(
    page_title="Peramalan Panen Padi | GA-ELM",
    page_icon=str(ui.LOGO) if ui.LOGO.exists() else "🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    state.init_state()
    P = state.build_pipeline()
    P["params"] = state.effective_params(P) if P["level"] >= 5 else None
    T = st.session_state.get("train")
    if not (T and P["level"] >= 5 and P["params"] is not None and T["sig"] == state.train_sig(P, P["params"])):
        T = None
    status = state.compute_status(P, T)
    ui.render_sidebar(P, status)

    n = st.session_state["stage"]
    if n == 0:
        beranda.render(P, T, status)
    elif not status["open"][n]:
        ui.locked_notice(n, status, P)
    elif n == 1:
        data.render(P, status)
    elif n == 2:
        anomali.render(P, status)
    elif n == 3:
        pilih_data.render(P, status)
    elif n == 4:
        deret.render(P, status)
    elif n == 5:
        split.render(P, status)
    elif n == 6:
        model.render(P, status)
    elif n == 7:
        parameter.render(P, status)
    elif n == 8:
        pelatihan.render(P, T, status)
    elif n == 9:
        evaluasi.render(P, T, status)


main()
