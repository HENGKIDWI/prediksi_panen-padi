"""Pelatihan, peramalan rekursif, metrik, tabel per Q, dan uji ketahanan (tanpa Streamlit)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import TARGET
from core.elm import calc_metrics, init_wb, predict_elm, train_elm
from core.ga import run_ga_final


def latih_nilai(df_train, df_test, p, use_ga, feats, log=None):
    cols = feats + [TARGET]
    mn = df_train[cols].min()
    sp = (df_train[cols].max() - mn).replace(0, 1)
    y_min = mn[TARGET]
    y_max = y_min + sp[TARGET]

    def norm(d):
        X = ((d[feats] - mn[feats]) / sp[feats]).values
        Y = ((d[TARGET] - mn[TARGET]) / sp[TARGET]).values.reshape(-1, 1)
        return X, Y

    X_tr, Y_tr = norm(df_train)
    X_te, Y_te = norm(df_test)
    history = None
    if use_ga:
        W, b, history = run_ga_final(X_tr, Y_tr, y_min, y_max, p, log)
    else:
        W, b = init_wb(p["ga_seed"], len(feats), p["L"], p["w_range"])
    beta = train_elm(X_tr, Y_tr, W, b, p["l2_reg"], p["act"])
    m_tr, _ = calc_metrics(Y_tr, predict_elm(X_tr, W, b, beta, p["act"]), y_min, y_max)
    m_te, _ = calc_metrics(Y_te, predict_elm(X_te, W, b, beta, p["act"]), y_min, y_max)
    return {"model": {"W": W, "b": b, "beta": beta, "mn": mn, "sp": sp, "y_min": y_min, "y_max": y_max, "act": p["act"]},
            "history": history, "m_train": m_tr, "m_test": m_te}


def build_model(df_fit, p, use_ga, feats, log=None):
    cols = feats + [TARGET]
    mn = df_fit[cols].min()
    sp = (df_fit[cols].max() - mn).replace(0, 1)
    y_min = mn[TARGET]
    y_max = y_min + sp[TARGET]
    X = ((df_fit[feats] - mn[feats]) / sp[feats]).values
    Y = ((df_fit[TARGET] - mn[TARGET]) / sp[TARGET]).values.reshape(-1, 1)
    if use_ga:
        W, b, _ = run_ga_final(X, Y, y_min, y_max, p, log)
    else:
        W, b = init_wb(p["ga_seed"], len(feats), p["L"], p["w_range"])
    beta = train_elm(X, Y, W, b, p["l2_reg"], p["act"])
    return {"W": W, "b": b, "beta": beta, "mn": mn, "sp": sp, "y_min": y_min, "y_max": y_max, "act": p["act"]}


def klimatologi(agg_hist, S):
    return agg_hist[S["kolom_jumlah"] + S["kolom_rata"] + [TARGET]].groupby(agg_hist.index.month).mean()


def ramal_rekursif(model, agg_hist, periode, S, feats, blend=0.0):
    """Ramal bulan demi bulan TANPA produksi aktual periode ramal/uji.
    Fitur = rata-rata historis per bulan; lag k = produksi k bulan sebelumnya (aktual bila masih di riwayat, selain itu ramalan)."""
    kl = klimatologi(agg_hist, S)
    kolom_x = S["kolom_jumlah"] + S["kolom_rata"]
    lags = [int(c[3:]) for c in feats if c.startswith("lag")]
    hist = [float(v) for v in agg_hist[TARGET].values]
    hasil = []
    for p in periode:
        x = {c: kl.loc[p.month, c] for c in kolom_x}
        for k in lags:
            x[f"lag{k}"] = hist[-k]
        x["bulan"] = p.month
        xn = ((pd.DataFrame([x])[feats] - model["mn"][feats]) / model["sp"][feats]).values
        yn = predict_elm(xn, model["W"], model["b"], model["beta"], model["act"]).ravel()[0]
        y = max(0.0, float(yn * (model["y_max"] - model["y_min"]) + model["y_min"]))
        if blend > 0:
            y = (1 - blend) * y + blend * float(kl.loc[p.month, TARGET])
        hasil.append(y)
        hist.append(y)
    return np.array(hasil)


def hitung_metrik(aktual, prediksi):
    a, p = np.asarray(aktual, float), np.asarray(prediksi, float)
    nz = a != 0
    sst = ((a - a.mean()) ** 2).sum()
    return {"MAE": float(np.mean(np.abs(a - p))), "RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
            "MAPE (%)": float(np.mean(np.abs(a[nz] - p[nz]) / a[nz]) * 100) if nz.any() else np.nan,
            "R2": float(1 - ((a - p) ** 2).sum() / sst) if sst > 0 else np.nan}


def tabel_per_q(periode, aktual, prediksi):
    q = pd.Series([f"{p.year}Q{(p.month - 1) // 3 + 1}" for p in periode])
    t = pd.DataFrame({"Q": q, "Aktual (Ton)": np.asarray(aktual, float), "Prediksi (Ton)": np.asarray(prediksi, float)})
    t = t.groupby("Q", sort=True).sum()
    t["Selisih (Ton)"] = t["Prediksi (Ton)"] - t["Aktual (Ton)"]
    t["Error (%)"] = t["Selisih (Ton)"].abs() / t["Aktual (Ton)"].replace(0, np.nan) * 100
    return t


def evaluasi_tahun(agg, d, Y, p, use_ga, S, n_seeds, feats, step=None):
    """Uji ketahanan: latih hanya dari tahun < Y, ramal 12 bulan tahun Y secara rekursif."""
    tr = d[d["tahun"] < Y]
    te = agg[agg["tahun"] == Y]
    if len(tr) < 12 or len(te) != 12:
        return None
    models = []
    for i in range(n_seeds):
        models.append(build_model(tr, dict(p, ga_seed=p["ga_seed"] + i), use_ga, feats))
        if step:
            step()
    pred = np.mean([ramal_rekursif(m, agg[agg["tahun"] < Y], te.index, S, feats, S["blend_musiman"]) for m in models], axis=0)
    return te.index, te[TARGET].values, pred
