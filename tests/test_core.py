"""Uji logika inti (tanpa Streamlit): ELM, GA, data, pencarian, peramalan rekursif, dan kebocoran data."""
import io

import numpy as np
import pandas as pd
import pytest

from core.config import COLS_ASLI, DEFAULT_SHEET, PARAMS, ROLES, SEARCH, SERI, TARGET
from core.data import (agregasi_bulanan, auto_map, buat_dataset, clean_data, convert_numeric, detect_issues,
                       kolom_fitur, make_demo_xlsx, split_per_tahun, split_sequential)
from core.elm import calc_metrics, init_wb, predict_elm, train_elm
from core.forecast import (build_model, evaluasi_tahun, hitung_metrik, latih_nilai, ramal_rekursif, tabel_per_q)
from core.ga import run_ga
from core.search import jalankan_pencarian


@pytest.fixture(scope="module")
def raw():
    return pd.read_excel(io.BytesIO(make_demo_xlsx()), sheet_name=DEFAULT_SHEET)


@pytest.fixture(scope="module")
def seri(raw):
    df, nn = convert_numeric(raw)
    _, issue = detect_issues(df, nn)
    data, _ = clean_data(df, issue, "drop")
    agg, n_kosong = agregasi_bulanan(data, SERI)
    return agg, buat_dataset(agg, (1,))


# ---------------- ELM & metrik ----------------
def test_elm_menyesuaikan_fungsi_linear():
    rng = np.random.RandomState(0)
    X = rng.rand(80, 3)
    Y = (X @ np.array([[0.5], [0.3], [0.2]]))
    W, b = init_wb(1, 3, 20, 1.0)
    beta = train_elm(X, Y, W, b, 1e-8, "tanh")
    assert np.mean((predict_elm(X, W, b, beta, "tanh") - Y) ** 2) < 1e-4


def test_calc_metrics_mape_mengabaikan_aktual_nol():
    y = np.array([[0.0], [10.0], [20.0]])
    m, _ = calc_metrics(y, y * 1.1, 0.0, 1.0)
    assert set(["MSE", "RMSE", "MAE", "MAPE (%)", "WAPE (%)", "R2"]) <= set(m)
    assert m["MAPE (%)"] == pytest.approx(10.0)          # nol tidak ikut dihitung


# ---------------- data ----------------
def test_auto_map_menebak_nama_mirip():
    cols = ["tahun panen", "BULAN", "Luas Tanam (Ha)", "luas panen (Ha)", "Curah Hujan (mm)", "Urea (kg)", "NPK (kg)",
            "Kelembapan_Tanah (%)", "Kelembapan Udara (%)", "Produktivitas (Ton/Ha)", "Produksi (Ton)"]
    g = auto_map(cols)
    assert g["Tahun panen"] == "tahun panen" and g["Bulan Panen"] == "BULAN" and g[TARGET] == "Produksi (Ton)"
    assert len(set(g.values())) == len(g)                 # satu kolom file hanya untuk satu peran


def test_convert_numeric_koma_desimal_dan_teks(raw):
    d = raw[ROLES].copy()
    d["Urea (kg)"] = d["Urea (kg)"].astype(object)
    d.loc[0, "Urea (kg)"] = "1,5"
    d.loc[1, "Urea (kg)"] = "abc"
    out, nn = convert_numeric(d)
    assert out.loc[0, "Urea (kg)"] == 1.5 and np.isnan(out.loc[1, "Urea (kg)"]) and bool(nn.loc[1, "Urea (kg)"])


def test_cleaning_drop_vs_median(raw):
    df, nn = convert_numeric(raw[ROLES])
    report, issue = detect_issues(df, nn)
    assert report["NaN"].sum() > 0                        # data demo sengaja memuat NaN
    drop, n_drop = clean_data(df, issue, "drop")
    med, n_med = clean_data(df, issue, "median")
    assert len(drop) == len(df) - n_drop and n_drop > 0
    assert len(med) == len(df) and not med[COLS_ASLI].isna().any().any()


def test_deret_bulanan_dan_lag(seri):
    agg, d = seri
    assert agg.index.is_monotonic_increasing and len(agg) == 60
    assert (d["lag1"].values == agg[TARGET].shift(1).dropna().values).all()
    assert kolom_fitur(SERI, (1,))[-2:] == ["lag1", "bulan"]


def test_split_per_tahun_terurut_tanpa_tumpang_tindih(seri):
    _, d = seri
    tr, te, uji = split_per_tahun(d, 1)
    assert tr.index.max() < te.index.min() and len(te) == 12 and uji == [2025]


def test_split_per_tahun_tahun_tidak_lengkap_ditolak(seri):
    _, d = seri
    with pytest.raises(ValueError):
        split_per_tahun(d.iloc[:-3], 1)


def test_split_rasio_urut_waktu(seri):
    _, d = seri
    tr, te = split_sequential(d, 0.8)
    assert len(tr) + len(te) == len(d) and tr.index.max() < te.index.min()


# ---------------- GA ----------------
def _xy(seri):
    _, d = seri
    feats = kolom_fitur(SERI, (1,))
    cols = feats + [TARGET]
    mn, sp = d[cols].min(), (d[cols].max() - d[cols].min()).replace(0, 1)
    X = ((d[feats] - mn[feats]) / sp[feats]).values
    Y = ((d[TARGET] - mn[TARGET]) / sp[TARGET]).values.reshape(-1, 1)
    return X, Y, mn[TARGET], mn[TARGET] + sp[TARGET]


def test_ga_deterministik_dan_fitness_tidak_turun(seri):
    X, Y, lo, hi = _xy(seri)
    p = dict(PARAMS, L=8, generations=8, pop_size=10, ga_val_weight=0.0, ga_seed=3)
    W1, b1, h1 = run_ga(X, Y, lo, hi, p)
    W2, b2, _ = run_ga(X, Y, lo, hi, p)
    assert np.allclose(W1, W2) and np.allclose(b1, b2)
    fit = [r["fitness"] for r in h1]
    assert all(b >= a for a, b in zip(fit, fit[1:])) and fit[-1] >= fit[0]


# ---------------- pencarian ----------------
def test_pencarian_elm_memilih_L_dalam_grid(seri):
    _, d = seri
    tr, _, _ = split_per_tahun(d, 1)
    S = dict(SEARCH, L_min=4, L_max=8, L_step=4, n_seeds=1, grid_act=["tanh"], grid_l2=[1e-2, 1e-1])
    out = jalankan_pencarian(tr, dict(PARAMS), S, use_ga=False, cari_ga=False, feats=kolom_fitur(SERI, (1,)))
    assert out["cfg"]["L"] in (4, 8) and out["cfg"]["act"] == "tanh" and {"elm", "elm_pilih"} <= set(out)


# ---------------- peramalan & kebocoran data ----------------
def test_latih_nilai_memuat_metrik_latih_dan_uji(seri):
    _, d = seri
    tr, te, _ = split_per_tahun(d, 1)
    r = latih_nilai(tr, te, dict(PARAMS, L=8), False, kolom_fitur(SERI, (1,)))
    assert {"RMSE", "MAPE (%)", "R2"} <= set(r["m_train"]) and np.isfinite(r["m_test"]["RMSE"])


def test_ramal_rekursif_tidak_memakai_produksi_aktual_periode_uji(seri):
    agg, d = seri
    feats = kolom_fitur(SERI, (1,))
    tr = d[d["tahun"] < 2025]
    m = build_model(tr, dict(PARAMS, L=8), False, feats)
    te = agg[agg["tahun"] == 2025]
    p1 = ramal_rekursif(m, agg[agg["tahun"] < 2025], te.index, SERI, feats, 0.5)
    agg2 = agg.copy()
    agg2.loc[agg2["tahun"] == 2025, TARGET] *= 10         # ubah drastis produksi aktual tahun uji
    p2 = ramal_rekursif(m, agg2[agg2["tahun"] < 2025], te.index, SERI, feats, 0.5)
    assert np.allclose(p1, p2) and (p1 >= 0).all() and len(p1) == 12


def test_evaluasi_tahun_tidak_bocor_dan_data_kurang_dilewati(seri):
    agg, d = seri
    feats = kolom_fitur(SERI, (1,))
    idx, akt, pred = evaluasi_tahun(agg, d, 2025, dict(PARAMS, L=8), False, SERI, 2, feats)
    agg2 = agg.copy()
    agg2.loc[agg2["tahun"] == 2025, TARGET] *= 10
    d2 = buat_dataset(agg2, (1,))
    _, _, pred2 = evaluasi_tahun(agg2, d2, 2025, dict(PARAMS, L=8), False, SERI, 2, feats)
    assert np.allclose(pred, pred2) and len(idx) == 12
    assert evaluasi_tahun(agg, d, 2021, dict(PARAMS, L=8), False, SERI, 1, feats) is None


def test_hitung_metrik_dan_tabel_per_q():
    per = pd.period_range("2025-01", periods=12, freq="M")
    a, p = np.arange(1, 13, dtype=float), np.arange(1, 13, dtype=float) * 1.1
    m = hitung_metrik(a, p)
    assert m["MAPE (%)"] == pytest.approx(10.0) and set(m) == {"MAE", "RMSE", "MAPE (%)", "R2"}
    t = tabel_per_q(per, a, p)
    assert list(t.index) == ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]
    assert t["Aktual (Ton)"].sum() == a.sum() and t["Error (%)"].round(6).eq(10.0).all()
