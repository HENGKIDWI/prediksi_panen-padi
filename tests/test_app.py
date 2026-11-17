"""Uji antarmuka lewat `streamlit.testing` (AppTest): beranda, penguncian tahap, dan alur 9 tahap."""
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parents[1] / "app.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def _set(at, **kv):
    for k, v in kv.items():
        at.session_state[k] = v
    at.run()


def _klik(at, awalan):
    btn = next(b for b in at.button if b.label.startswith(awalan) or awalan in b.label)
    btn.click().run()
    assert not at.exception, at.exception


def _lanjut(at):
    b = [b for b in at.button if b.label.startswith("Lanjut")]
    assert b and not b[0].proto.disabled, f"Tombol Lanjut tidak aktif di tahap {at.session_state['stage']}"
    b[0].click().run()
    assert not at.exception, at.exception


def _demo_sampai_tahap(at, tahap):
    _klik(at, "Muat Data Demo")
    at.session_state["stage"] = 1
    at.run()
    for _ in range(1, tahap):
        _lanjut(at)
    assert at.session_state["stage"] == tahap


def _latih(at, seeds=1, tahun_uji=()):
    at.session_state["n_seeds"] = seeds
    at.session_state["tahun_uji"] = list(tahun_uji)
    at.run()
    _klik(at, "Latih & Ramal")


def test_beranda_tampil_tanpa_galat():
    at = _app()
    assert not at.exception
    assert "Sistem Cerdas Peramalan Produksi Padi" in at.markdown[0].value
    assert any("GA-ELM" in m.value for m in at.markdown)
    assert any(b.label.startswith("Muat Data Demo") or "Muat Data Demo" in b.label for b in at.button)


def test_tahap_terkunci_sebelum_data_dimuat():
    at = _app()
    at.session_state["stage"] = 4
    at.run()
    assert not at.exception
    assert any("belum bisa dibuka" in w.value for w in at.warning)


def test_alur_penuh_elm_manual_split_per_tahun():
    at = _app()
    _demo_sampai_tahap(at, 6)
    at.session_state["model_key"] = "ELM"
    at.run()
    _lanjut(at)                                   # tahap 6 -> 7 (parameter manual)
    _lanjut(at)                                   # tahap 7 -> 8
    _latih(at)
    _lanjut(at)                                   # tahap 8 -> 9
    nilai = {m.label: m.value for m in at.metric}
    assert {"MAE", "RMSE", "MAPE", "R²", "Q1", "Q2", "Q3", "Q4", "TOTAL"} <= set(nilai)
    assert nilai["MAPE"].endswith("%")


def test_alur_ga_elm_dengan_uji_ketahanan():
    at = _app()
    _demo_sampai_tahap(at, 6)
    _set(at, model_key="GA-ELM", p_generations=10, p_pop_size=10, p_L=8)
    _lanjut(at)
    _lanjut(at)
    _latih(at, seeds=1, tahun_uji=[2024])
    _lanjut(at)
    assert at.session_state["train"]["nama"] == "GA-ELM"
    assert 2024 in at.session_state["train"]["robust"]
    assert any("Uji ketahanan" in m.value for m in at.markdown)


def test_pencarian_otomatis_elm():
    at = _app()
    _demo_sampai_tahap(at, 6)
    at.session_state["model_key"] = "ELM"
    at.run()
    _lanjut(at)
    _set(at, param_mode="elm", s_L_min=4, s_L_max=32, s_L_step=8, s_n_seeds=3, s_grid_act=["tanh"],
         s_grid_l2=[1e-2, 1e-1])
    _klik(at, "Jalankan pencarian parameter")
    assert at.session_state["search"]["params"]["L"] in (4, 12, 20, 28)
    _lanjut(at)
    _latih(at)
    assert at.session_state["train"] is not None


@pytest.mark.parametrize("lag_key", ["12", "1+12"])
def test_opsi_lag_cleaning_median_dan_split_rasio(lag_key):
    at = _app()
    _klik(at, "Muat Data Demo")
    _set(at, stage=1, data_mode="cleaning", clean_method="median", lag_key=lag_key, split_method="rasio",
         split_key="2")
    for _ in range(4):                            # tahap 1 -> 5
        _lanjut(at)
    assert at.session_state["stage"] == 5
    assert any("Split terurut" in s.value for s in at.success)
    assert at.session_state["lag_key"] == lag_key


def test_reset_semua_menghapus_state():
    at = _app()
    _klik(at, "Muat Data Demo")
    assert at.session_state["src"] is not None
    _klik(at, "Reset semua")
    assert at.session_state["src"] is None
