"""Dekomposisi per kecamatan: Produksi = luas panen (ELM / GA-ELM, musiman) x produktivitas (statistik).

Seluruh fitur memakai data tahun-tahun sebelumnya, sehingga bisa dipakai meramal ke depan.
Tanpa Streamlit; kolom data dipetakan lewat dict `cm` (kec, tahun, bulan, luas, prod, produksi).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import FEATURES_DEKOMP, KUINTAL_PER_TON
from .elm import fit_elm

TINGKAT = ["Kecamatan-bulan", "Kecamatan-triwulan", "Kecamatan-tahun", "Total-triwulan"]
PEMBANDING = [("Naif: bulan sama tahun lalu", "a_naif"),
              ("Rata-rata bulan sama (tahun-tahun lalu)", "a_pm"),
              ("Statistik musiman (tahun lalu x pola bulan)", "a_musiman")]


# ---------------- produktivitas ----------------
def ramal_produktivitas(pr_tab: pd.DataFrame, kec, tahun: int, metode: str) -> float:
    """Ramal produktivitas kecamatan 'kec' pada 'tahun' dari tahun-tahun sebelumnya.
    metode: 'last' | 'mean2' | 'trend' (tren linear, min. 3 tahun)."""
    cols = [c for c in pr_tab.columns if c < tahun]
    h = pr_tab.loc[kec, cols].dropna() if (kec in pr_tab.index and cols) else pd.Series(dtype=float)
    if len(h) == 0:
        return float(np.nanmean(pr_tab[cols[-1]].values)) if cols else float(np.nanmean(pr_tab.values))
    if metode == "mean2":
        return float(h.iloc[-2:].mean())
    if metode == "trend" and len(h) >= 3:
        return float(max(np.polyval(np.polyfit(np.array(h.index, float), h.values, 1), tahun), 0.0))
    return float(h.iloc[-1])


def backtest_produktivitas(pr_tab: pd.DataFrame) -> pd.Series:
    """MAPE (%) tiap metode: ramal tiap tahun (mulai tahun ke-3) dari tahun-tahun sebelumnya."""
    hasil = {}
    for m in ["last", "mean2", "trend"]:
        e = []
        for Y in list(pr_tab.columns)[2:]:
            for kk in pr_tab.index:
                a = pr_tab.loc[kk, Y]
                if pd.notna(a) and a > 0:
                    e.append(abs(ramal_produktivitas(pr_tab, kk, Y, m) - a) / a * 100)
        hasil[m] = float(np.mean(e)) if e else np.nan
    return pd.Series(hasil, name="MAPE produktivitas (%)")


# ---------------- persiapan data ----------------
def siapkan_dekomp(data: pd.DataFrame, cm: dict, metode: str = "auto") -> dict:
    """Bentuk grid kecamatan x bulan panen (bulan tanpa baris = luas panen 0) beserta fitur musiman."""
    k, ty, bm = cm["kec"], cm["tahun"], cm["bulan"]
    lp, pc, ps = cm["luas"], cm["prod"], cm["produksi"]
    hilang = [c for c in [k, ty, bm, lp, pc, ps] if c not in data.columns]
    if hilang:
        raise ValueError(f"Kolom tidak ditemukan di data: {hilang}")
    d = data.copy()
    t0 = int(d[ty].min())
    d["t"] = (d[ty].astype(int) - t0) * 12 + d[bm].astype(int)       # 1 = Januari tahun pertama
    t_first, t_last = int(d["t"].min()), int(d["t"].max())
    kecs = sorted(d[k].unique())

    a = d.groupby([k, "t"])[[lp, ps]].sum().reset_index()
    grid = pd.MultiIndex.from_product([kecs, range(t_first, t_last + 1)], names=[k, "t"]).to_frame(index=False)
    grid = grid.merge(a, how="left", on=[k, "t"]).fillna({lp: 0.0, ps: 0.0})
    grid = grid.rename(columns={ps: "produksi_sel"})
    grid["tahun"] = t0 + (grid["t"] - 1) // 12
    grid["bulan"] = (grid["t"] - 1) % 12 + 1
    grid["bulan_sin"] = np.sin(2 * np.pi * grid["bulan"] / 12)
    grid["bulan_cos"] = np.cos(2 * np.pi * grid["bulan"] / 12)
    grid = grid.sort_values([k, "t"]).reset_index(drop=True)
    grid["lag12"] = grid.groupby(k)[lp].shift(12)
    rata = grid.groupby([k, "tahun"])[lp].mean().rename("lvl_prev").reset_index()
    rata["tahun"] += 1
    grid = grid.merge(rata, on=[k, "tahun"], how="left")
    grid["pm"] = grid.groupby([k, "bulan"])[lp].transform(lambda s: s.expanding().mean().shift(1))
    data_model = grid.dropna(subset=FEATURES_DEKOMP).sort_values(["t", k]).reset_index(drop=True)

    pr_tab = d[d[pc] > 0].groupby([k, ty])[pc].mean().unstack()
    pr_tab.columns = [int(c) for c in pr_tab.columns]
    bt = backtest_produktivitas(pr_tab)
    if metode == "auto":
        metode = bt.idxmin() if bt.notna().any() else "last"
    return {"grid_full": grid, "data_model": data_model, "pr_tab": pr_tab, "bt_prod": bt,
            "metode_prod": metode, "cm": dict(cm), "kec_col": k, "luas_col": lp, "t0": t0,
            "t_first": t_first, "t_last": t_last, "kecs": kecs}


def tahun_data_terakhir(info: dict) -> int:
    return info["t0"] + (info["t_last"] - 1) // 12


def horizon_sampai(info: dict, tahun_ramal: int) -> int:
    """Jumlah bulan dari bulan terakhir data sampai Desember 'tahun_ramal'."""
    h = (tahun_ramal - info["t0"]) * 12 + 12 - info["t_last"]
    if h <= 0:
        raise ValueError(f"Data sudah mencakup sampai {tahun_ramal}; pilih tahun setelah data.")
    return h


def ke_ton_kuintal(nilai, satuan: str = "ton"):
    """Ubah nilai produksi (satuan data: 'ton'/'kuintal') menjadi (ton, kuintal)."""
    ton = np.asarray(nilai, dtype=float) * (1.0 if satuan == "ton" else 1.0 / KUINTAL_PER_TON)
    return ton, ton * KUINTAL_PER_TON


# ---------------- pembanding statistik ----------------
def baseline_musiman(grid: pd.DataFrame, k: str, lp: str, d: pd.DataFrame) -> np.ndarray:
    """Luas panen = (rata-rata bulanan tahun lalu x 12) x porsi bulan itu
    (rata-rata porsi dari tahun-tahun LENGKAP sebelumnya, per kecamatan)."""
    g = grid.copy()
    tot = g.groupby([k, "tahun"])[lp].transform("sum")
    cnt = g.groupby([k, "tahun"])[lp].transform("size")
    g["share"] = np.where((cnt == 12) & (tot > 0), g[lp] / tot.replace(0, np.nan), np.nan)
    kunci = {key: s for key, s in g[cnt == 12].groupby([k, "bulan"])}
    out = []
    for kk, yy, bb, lvl in zip(d[k], d["tahun"], d["bulan"], d["lvl_prev"]):
        s = kunci.get((kk, bb))
        sh = s.loc[s["tahun"] < yy, "share"].mean() if s is not None else np.nan
        out.append(0.0 if (np.isnan(sh) or np.isnan(lvl)) else lvl * 12 * sh)
    return np.array(out)


# ---------------- peramalan ----------------
def ramal_dekomp(model, info: dict, horizon: int, metode_pertama: str | None = None,
                 metode_lanjut: str = "last") -> pd.DataFrame:
    """Ramal luas panen tiap kecamatan bulan demi bulan (rekursif untuk tahun kedua dst.), lalu kalikan
    produktivitas. Tahun pertama memakai 'metode_pertama'; tahun-tahun berikutnya 'metode_lanjut'
    (default 'last': tidak mengekstrapolasi tren terlalu jauh)."""
    k, t0 = info["kec_col"], info["t0"]
    g, kecs, t_last = info["grid_full"], info["kecs"], info["t_last"]
    metode_pertama = metode_pertama or info["metode_prod"]
    tahun_pertama = t0 + t_last // 12
    ser = {}
    for kk in kecs:
        s = np.full(t_last + horizon + 1, np.nan)
        sub = g[g[k] == kk]
        s[sub["t"].values] = sub[info["luas_col"]].values
        ser[kk] = s
    rows = []
    for t in range(t_last + 1, t_last + horizon + 1):
        bulan, tahun = (t - 1) % 12 + 1, t0 + (t - 1) // 12
        t_lo = (tahun - 1 - t0) * 12 + 1
        feat = []
        for kk in kecs:
            s = ser[kk]
            feat.append([np.sin(2 * np.pi * bulan / 12), np.cos(2 * np.pi * bulan / 12), s[t - 12],
                         np.nanmean(s[max(t_lo, 1): t_lo + 12]), np.nanmean(s[bulan:t:12])])
        Xf = pd.DataFrame(feat, columns=FEATURES_DEKOMP)
        A = np.maximum(model.predict(Xf), 0.0)
        tw = (bulan - 1) // 3 + 1
        metode = metode_pertama if tahun == tahun_pertama else metode_lanjut
        for kk, a_ in zip(kecs, A):
            ser[kk][t] = a_
            pr_ = ramal_produktivitas(info["pr_tab"], kk, tahun, metode)
            rows.append({k: kk, "t": t, "tahun": tahun, "bulan": bulan, "tw": tw, "Periode": f"{tahun}Q{tw}",
                         "luas_ramal": a_, "produktivitas": pr_, "Produksi": a_ * pr_})
    return pd.DataFrame(rows)


def ramal_ensemble(data_model: pd.DataFrame, info: dict, p: dict, use_ga: bool, tahun_ramal: int,
                   n_seeds: int, satuan: str = "ton", metode_prod: str | None = None, progress=None) -> dict:
    """Latih model pada SELURUH data dengan banyak seed, rata-ratakan luas panen hasil ramalan.
    Return dict: fc (rata-rata, kolom Ton & Kuintal), tot_seed (n_seeds x n_periode, satuan data), periode."""
    H = horizon_sampai(info, tahun_ramal)
    daftar = []
    for i in range(n_seeds):
        m = fit_elm(data_model, FEATURES_DEKOMP, info["luas_col"], dict(p, ga_seed=p["ga_seed"] + i), use_ga)
        daftar.append(ramal_dekomp(m, info, H, metode_pertama=metode_prod))
        if progress:
            progress((i + 1) / n_seeds, f"Seed {i + 1}/{n_seeds}")
    fc = daftar[0].copy()
    fc["luas_ramal"] = np.mean([f["luas_ramal"].values for f in daftar], axis=0)
    fc["Produksi"] = fc["luas_ramal"] * fc["produktivitas"]
    fc["Ton"], fc["Kuintal"] = ke_ton_kuintal(fc["Produksi"], satuan)
    periode = sorted(fc["Periode"].unique())
    tot_seed = np.array([(f["luas_ramal"] * f["produktivitas"]).groupby(f["Periode"]).sum().reindex(periode).values
                         for f in daftar])
    return {"fc": fc, "tot_seed": tot_seed, "periode": periode, "H": H}


def tabel_kecamatan(fc: pd.DataFrame, kc: str, kolom: str) -> pd.DataFrame:
    """Produksi per kecamatan per periode (triwulan) + total per kecamatan + baris TOTAL."""
    tab = fc.pivot_table(index=kc, columns="Periode", values=kolom, aggfunc="sum")
    tab["Total"] = tab.sum(axis=1)
    tab.loc["TOTAL"] = tab.sum()
    return tab


def tabel_total(res: dict, info: dict, satuan: str) -> pd.DataFrame:
    """Total semua kecamatan per triwulan: rata-rata & rentang antar-seed (ton & kuintal) + aktual tahun lalu."""
    periode, tot = res["periode"], res["tot_seed"]
    ton_m, ku_m = ke_ton_kuintal(tot.mean(0), satuan)
    ton_lo, ku_lo = ke_ton_kuintal(tot.min(0), satuan)
    ton_hi, ku_hi = ke_ton_kuintal(tot.max(0), satuan)
    df = pd.DataFrame({"Ramalan (Ton)": ton_m, "Min (Ton)": ton_lo, "Maks (Ton)": ton_hi,
                       "Ramalan (Kuintal)": ku_m, "Min (Kuintal)": ku_lo, "Maks (Kuintal)": ku_hi}, index=periode)
    g = info["grid_full"].copy()
    g["tw"] = (g["bulan"] - 1) // 3 + 1
    akt = g.groupby(["tahun", "tw"])["produksi_sel"].sum()
    a_ton = []
    for per in periode:
        y, q = int(per[:4]), int(per[-1])
        a_ton.append(ke_ton_kuintal(akt.get((y - 1, q), np.nan), satuan)[0])
    df.insert(0, "Aktual tahun lalu (Ton)", np.array(a_ton, dtype=float))
    df["Selisih vs tahun lalu (%)"] = (df["Ramalan (Ton)"] / df["Aktual tahun lalu (Ton)"] - 1) * 100
    return df


# ---------------- evaluasi bergulir ----------------
def basis_prediksi(te: pd.DataFrame, info: dict, metode: str) -> pd.DataFrame:
    """Sel uji + produktivitas ramalan (hanya dari tahun sebelumnya) + luas panen pembanding."""
    k, lp = info["kec_col"], info["luas_col"]
    d = te[[k, "tahun", "bulan", "produksi_sel", "lvl_prev", "lag12", "pm"]].copy()
    d["prod_hat"] = [ramal_produktivitas(info["pr_tab"], kk, yy, metode) for kk, yy in zip(d[k], d["tahun"])]
    d["a_naif"] = d["lag12"].values
    d["a_pm"] = d["pm"].values
    d["a_musiman"] = baseline_musiman(info["grid_full"], k, lp, d)
    return d


def hitung_tingkat(d: pd.DataFrame, a, k: str) -> dict:
    """Produksi = luas panen (a) x produktivitas; metrik pada beberapa tingkat agregasi."""
    z = d.copy()
    z["P"] = np.maximum(np.asarray(a, float), 0.0) * z["prod_hat"].values
    z["tw"] = (z["bulan"] - 1) // 3 + 1
    kunci = {"Kecamatan-bulan": [k, "tahun", "bulan"], "Kecamatan-triwulan": [k, "tahun", "tw"],
             "Kecamatan-tahun": [k, "tahun"], "Total-triwulan": ["tahun", "tw"]}
    out = {}
    for lev, ks in kunci.items():
        g = z.groupby(ks)[["produksi_sel", "P"]].sum()
        y, p_ = g["produksi_sel"].values, g["P"].values
        nz = y > 0
        out[lev] = {"n": len(g),
                    "WAPE": np.abs(y - p_).sum() / y.sum() * 100 if y.sum() > 0 else np.nan,
                    "RMSE": float(np.sqrt(np.mean((y - p_) ** 2))),
                    "MAPE": float(np.mean(np.abs(y[nz] - p_[nz]) / y[nz]) * 100) if nz.any() else np.nan,
                    "Bias": (p_.sum() / y.sum() - 1) * 100 if y.sum() > 0 else np.nan}
    return out


def evaluasi_rolling(data_model, info, p, use_ga, label, tahun_uji, n_seeds, metode_prod="auto",
                     progress=None) -> pd.DataFrame:
    """Evaluasi bergulir: tiap tahun uji dilatih HANYA dari tahun-tahun sebelumnya.
    Model diulang dengan banyak seed; pembanding naif/statistik dihitung pada sel uji yang sama."""
    k = info["kec_col"]
    baris, total, done = [], max(len(tahun_uji) * n_seeds, 1), 0
    for Y in tahun_uji:
        tr = data_model[data_model["tahun"] < Y]
        te = data_model[data_model["tahun"] == Y]
        if len(tr) < 100 or len(te) == 0:
            done += n_seeds
            continue
        if metode_prod == "auto":
            pr_sub = info["pr_tab"][[c for c in info["pr_tab"].columns if c < Y]]
            bt = backtest_produktivitas(pr_sub)
            metode = bt.idxmin() if bt.notna().any() else "last"
        else:
            metode = metode_prod
        d = basis_prediksi(te, info, metode)
        for nm, kol in PEMBANDING:
            for lev, m in hitung_tingkat(d, d[kol].values, k).items():
                baris.append({"tahun_uji": Y, "tingkat": lev, "metode": nm, "seed": -1, "waktu": 0.0,
                              "produktivitas": metode, **m})
        for i in range(n_seeds):
            s = p["ga_seed"] + i
            t0 = time.time()
            model = fit_elm(tr, FEATURES_DEKOMP, info["luas_col"], dict(p, ga_seed=s), use_ga)
            a_hat = model.predict(te)
            waktu = time.time() - t0
            for lev, m in hitung_tingkat(d, a_hat, k).items():
                baris.append({"tahun_uji": Y, "tingkat": lev, "metode": label, "seed": s, "waktu": waktu,
                              "produktivitas": metode, **m})
            done += 1
            if progress:
                progress(done / total, f"Tahun uji {Y}: seed {i + 1}/{n_seeds}")
    return pd.DataFrame(baris)


def ringkas_rolling(h: pd.DataFrame, label: str):
    """Ringkasan: (tabel rata-rata +/- std per tahun/tingkat/metode, tabel putusan terhadap pembanding terbaik)."""
    urut = [label] + [nm for nm, _ in PEMBANDING]
    rows, putusan = [], []
    for Y in sorted(h["tahun_uji"].unique()):
        for lev in TINGKAT:
            sub = h[(h["tahun_uji"] == Y) & (h["tingkat"] == lev)]
            rata = {}
            for nm in urut:
                s = sub[sub["metode"] == nm]
                if s.empty:
                    continue
                rata[nm] = (s["WAPE"].mean(), s["WAPE"].std(ddof=0))
                rows.append({"Tahun uji": int(Y), "Tingkat": lev, "n": int(s["n"].iloc[0]), "Metode": nm,
                             "Seed": len(s) if nm == label else 1,
                             "WAPE (%)": s["WAPE"].mean(), "WAPE std": s["WAPE"].std(ddof=0),
                             "RMSE": s["RMSE"].mean(), "RMSE std": s["RMSE"].std(ddof=0),
                             "MAPE (%)": s["MAPE"].mean(), "Bias (%)": s["Bias"].mean(),
                             "Waktu/seed (s)": s["waktu"].mean() if nm == label else np.nan})
            base = {k_: v for k_, v in rata.items() if k_ != label}
            if label in rata and base:
                terbaik = min(base, key=lambda k_: base[k_][0])
                sel = base[terbaik][0] - rata[label][0]
                putusan.append({"Tahun uji": int(Y), "Tingkat": lev, "Pembanding terbaik": terbaik,
                                "Selisih WAPE (poin)": sel, "Lebih baik": bool(sel > 0),
                                "Jelas (> 1 std seed)": bool(sel > rata[label][1])})
    return pd.DataFrame(rows), pd.DataFrame(putusan)
