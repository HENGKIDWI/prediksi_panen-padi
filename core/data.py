"""Data: pemetaan kolom, konversi numerik, deteksi anomali, cleaning, deret bulanan, fitur lag, split, data demo."""
from __future__ import annotations

import difflib
import io
import re

import numpy as np
import pandas as pd

from core.config import COLS_ASLI, DEFAULT_SHEET, ROLES, TARGET


def kolom_fitur(S, lags=(1,)):
    """Fitur deret bulanan: variabel bulan itu + lag produksi (sesuai pilihan) + nomor bulan."""
    return S["kolom_jumlah"] + S["kolom_rata"] + [f"lag{k}" for k in lags] + ["bulan"]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def auto_map(columns):
    """Tebak pemetaan peran -> kolom file, bertahap (satu kolom file hanya untuk satu peran):
    1) cocok persis setelah normalisasi (abaikan huruf besar/spasi/_/tanda baca), 2) mirip (>= 0,85),
    3) salah satu nama memuat yang lain (mis. 'BULAN' ~ 'Bulan Panen') bila kandidatnya tunggal,
    4) awalan sama >= 8 huruf (mis. 'Produksi (Ton)' ~ 'Produksi (Ha)') bila kandidatnya tunggal."""
    guess, used = {}, set()

    def ambil(role, c):
        guess[role] = c
        used.add(c)

    def sisa():
        return [c for c in columns if c not in used]

    for role in ROLES:
        c = next((c for c in sisa() if _norm(c) == _norm(role)), None)
        if c is not None:
            ambil(role, c)
    for role in ROLES:
        if role in guess:
            continue
        m = difflib.get_close_matches(_norm(role), [_norm(c) for c in sisa()], n=1, cutoff=0.85)
        if m:
            ambil(role, next(c for c in sisa() if _norm(c) == m[0]))

    def prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    aturan = [lambda r, c: min(len(r), len(c)) >= 4 and (c in r or r in c),
              lambda r, c: prefix(r, c) >= 8]
    for cocok in aturan:
        peran = [r for r in ROLES if r not in guess]
        kand = {r: [c for c in sisa() if cocok(_norm(r), _norm(c))] for r in peran}
        for role in peran:
            if len(kand[role]) != 1:
                continue
            c = kand[role][0]
            if sum(c in v for v in kand.values()) == 1:          # kolom itu hanya cocok untuk satu peran
                ambil(role, c)
    return guess


def convert_numeric(df):
    """Konversi ke numerik (koma desimal -> titik). Return df + mask teks tak valid."""
    out = df.copy()
    non_numeric = pd.DataFrame(False, index=df.index, columns=COLS_ASLI)
    for col in COLS_ASLI:
        orig = out[col]
        s = orig.astype(str).str.replace(",", ".") if orig.dtype == "object" else orig
        conv = pd.to_numeric(s, errors="coerce")
        non_numeric[col] = conv.isna() & orig.notna()
        out[col] = conv
    return out, non_numeric


def detect_issues(df, non_numeric):
    nan = df[COLS_ASLI].isna() & ~non_numeric
    negative = (df[COLS_ASLI] < 0).fillna(False)
    zero = (df[COLS_ASLI] == 0).fillna(False)
    report = pd.DataFrame({"NaN": nan.sum(), "Non-numerik": non_numeric.sum(),
                           "Negatif": negative.sum(), "Nol": zero.sum()})
    return report, (nan | non_numeric | negative | zero)


def clean_data(df, issue_mask, method):
    d = df.copy()
    d[COLS_ASLI] = d[COLS_ASLI].mask(issue_mask)
    n_awal = len(d)
    if method == "drop":
        d = d.dropna(subset=COLS_ASLI)
    else:
        d[COLS_ASLI] = d[COLS_ASLI].fillna(d[COLS_ASLI].median())
        d = d.dropna(subset=COLS_ASLI)
    return d.reset_index(drop=True), n_awal - len(d)


def split_sequential(df, ratio):
    n_train = min(max(int(round(len(df) * ratio)), 1), len(df) - 1)
    return df.iloc[:n_train], df.iloc[n_train:]


def agregasi_bulanan(data, S):
    """Jumlahkan semua kecamatan menjadi satu baris per bulan panen. Bulan tanpa data -> 0."""
    d = data.assign(tahun=pd.to_numeric(data[S["tahun_col"]]).astype(int),
                    bulan=pd.to_numeric(data[S["bulan_col"]]).astype(int))
    g = d.groupby(["tahun", "bulan"])
    agg = pd.concat([g[S["kolom_jumlah"] + [TARGET]].sum(), g[S["kolom_rata"]].mean()], axis=1).reset_index()
    agg.index = pd.PeriodIndex([f"{t}-{b:02d}" for t, b in zip(agg["tahun"], agg["bulan"])], freq="M")
    agg = agg.drop(columns=["tahun", "bulan"])
    agg = agg.reindex(pd.period_range(agg.index.min(), agg.index.max(), freq="M"))
    kosong = agg[TARGET].isna()
    agg.loc[kosong, S["kolom_jumlah"] + [TARGET]] = 0.0
    rata_bulan = agg[S["kolom_rata"]].groupby(agg.index.month).transform("mean")
    agg[S["kolom_rata"]] = agg[S["kolom_rata"]].fillna(rata_bulan).fillna(agg[S["kolom_rata"]].mean())
    agg["bulan"] = agg.index.month
    agg["tahun"] = agg.index.year
    return agg, int(kosong.sum())


def buat_dataset(agg, lags=(1,)):
    """Tambah fitur lag produksi (shift k bulan); baris awal yang tidak punya lag dibuang."""
    d = agg.copy()
    for k in lags:
        d[f"lag{k}"] = d[TARGET].shift(k)
    return d.dropna().copy()


def split_per_tahun(df, n_tahun):
    """Uji = n tahun kalender terakhir (harus lengkap 12 bulan), latih = semua tahun sebelumnya."""
    tahun = sorted(set(df.index.year))
    uji = tahun[-n_tahun:]
    for y in uji:
        n = int((df.index.year == y).sum())
        if n != 12:
            raise ValueError(f"Tahun {y} hanya punya {n} bulan sampel — tidak lengkap. Gunakan split rasio persen.")
    te = df[df.index.year.isin(uji)]
    tr = df[~df.index.year.isin(uji)]
    if len(tr) < 12:
        raise ValueError(f"Data latih hanya {len(tr)} bulan (minimal 12). Kurangi jumlah tahun uji atau pakai rasio persen.")
    return tr, te, uji


def make_demo_xlsx() -> bytes:
    """Data simulasi (6 kecamatan x 60 bulan) agar aplikasi bisa dicoba tanpa file."""
    rng = np.random.RandomState(7)
    musim = {1: .8, 2: 1.0, 3: 1.5, 4: 1.7, 5: 1.0, 6: .5, 7: .6, 8: 1.2, 9: 1.6, 10: 1.4, 11: .6, 12: .5}
    hujan = {1: 380, 2: 340, 3: 310, 4: 220, 5: 150, 6: 90, 7: 60, 8: 50, 9: 70, 10: 140, 11: 260, 12: 350}
    luas_dasar = [900, 700, 1100, 850, 600, 1000]
    rows = []
    for tahun in range(2021, 2026):
        for bulan in range(1, 13):
            for k, lb in enumerate(luas_dasar):
                lp = lb * musim[bulan] * rng.uniform(0.85, 1.15)
                lt = lp * rng.uniform(1.05, 1.2)
                prod = rng.uniform(5.0, 6.2) + 0.3 * np.sin(bulan / 12 * 2 * np.pi)
                rows.append({
                    "Kecamatan": f"Kecamatan {chr(65 + k)}", "Tahun panen": tahun, "Bulan Panen": bulan,
                    "Luas tanam (Ha)": round(lt, 1), "luas panen (Ha)": round(lp, 1),
                    "Curah Hujan (mm)": round(hujan[bulan] * rng.uniform(0.8, 1.2), 1),
                    "Urea (kg)": round(lt * rng.uniform(180, 230), 0), "NPK (kg)": round(lt * rng.uniform(120, 170), 0),
                    "Kelembapan_Tanah (%)": round(rng.uniform(62, 85), 1),
                    "Kelembapan Udara (%)": round(rng.uniform(70, 90), 1),
                    "Produktivitas (Ton/Ha)": round(prod, 2), "Produksi (Ha)": round(lp * prod, 1),
                })
    df = pd.DataFrame(rows)
    for col in ["Urea (kg)", "Curah Hujan (mm)"]:           # sedikit anomali agar tahap 2 terlihat hidup
        df.loc[rng.choice(len(df), 6, replace=False), col] = np.nan
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=DEFAULT_SHEET, index=False)
    return buf.getvalue()
