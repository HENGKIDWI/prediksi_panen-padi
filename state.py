"""State sesi, cache ringan, pipeline data (tahap 1-5), dan status tahap. Bagian dari lapisan UI (memakai Streamlit)."""
from __future__ import annotations

import copy
import hashlib
import io

import pandas as pd
import streamlit as st

from core.config import (
    COLS_ASLI,
    DEFAULT_SHEET,
    DEFAULTS,
    KUINTAL_PER_TON,
    LAG_OPTIONS,
    NONE,
    PARAMS,
    ROLE_LABEL,
    ROLES,
    SEARCH,
    SERI,
    SPLIT_OPTIONS,
    STAGES,
    WIDGET_KEYS,
)
from core.data import (
    agregasi_bulanan,
    auto_map,
    buat_dataset,
    clean_data,
    convert_numeric,
    detect_issues,
    kolom_fitur,
    make_demo_xlsx,
    split_per_tahun,
    split_sequential,
)

_MISSING = object()

def init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, copy.deepcopy(v))
    for k in ("src", "search", "train", "last_upload", "map_sig"):
        st.session_state.setdefault(k, None)
    st.session_state.setdefault("confirmed", set())      # tahap yang sudah dikonfirmasi pengguna (tombol Lanjut)
    # Cegah Streamlit membuang state widget yang sedang tidak tampil (tahap lain)
    for k in WIDGET_KEYS:
        st.session_state[k] = st.session_state[k]


def sane(key, options, default=None):
    """Pastikan nilai tersimpan valid terhadap opsi dropdown (mis. setelah ganti file)."""
    if st.session_state.get(key, _MISSING) not in options:
        st.session_state[key] = default if default in options else options[0]


def go(n):
    st.session_state["stage"] = n


def advance(n):
    """Konfirmasi tahap n selesai, lalu buka & pindah ke tahap berikutnya."""
    st.session_state["confirmed"].add(n)
    st.session_state["stage"] = n + 1


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.cache_data.clear()


def load_demo():
    b = demo_xlsx()
    st.session_state["src"] = {"name": "Data_Demo_Panen_Padi.xlsx", "bytes": b,
                               "hash": hashlib.md5(b).hexdigest()[:12], "origin": "demo"}
    st.session_state["sheet"] = DEFAULT_SHEET
    st.session_state["confirmed"] = set()


def clear_src():
    st.session_state["src"] = None
    st.session_state["confirmed"] = set()


@st.cache_data(show_spinner=False)
def list_sheets(file_bytes: bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


@st.cache_data(show_spinner="Membaca sheet ...")
def read_sheet(file_bytes: bytes, sheet: str):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)


@st.cache_data(show_spinner=False)
def analyze(raw):
    df, non_numeric = convert_numeric(raw)
    report, issue_mask = detect_issues(df, non_numeric)
    return df, report, issue_mask


@st.cache_data(show_spinner=False)
def prepare(df, issue_mask, mode, method):
    if mode == "cleaning":
        data, n_hilang = clean_data(df, issue_mask, method)
        label = "Drop" if method == "drop" else "Imputasi median"
        return data, n_hilang, label, 0
    n_nan = int(df[COLS_ASLI].isna().any(axis=1).sum())
    data = df.dropna(subset=COLS_ASLI).reset_index(drop=True) if n_nan else df
    return data, n_nan, "Data asli", n_nan


@st.cache_data(show_spinner=False)
def build_series(data, lags):
    d = data.dropna(subset=[SERI["tahun_col"], SERI["bulan_col"]])
    agg, n_kosong = agregasi_bulanan(d, SERI)
    return agg, n_kosong, buat_dataset(agg, lags)


def manual_params():
    p = dict(PARAMS)
    for k in PARAMS:
        p[k] = st.session_state.get(f"p_{k}", PARAMS[k])
    p["ga_patience"] = p["ga_patience"] or None
    return p


def search_cfg():
    S = copy.deepcopy(SEARCH)
    g = st.session_state.get
    S.update(L_min=g("s_L_min"), L_max=g("s_L_max"), L_step=g("s_L_step"), val_ratio=g("s_val_ratio"),
             n_seeds=g("s_n_seeds"), n_seeds_ga=g("s_n_seeds_ga"), n_seeds_sens=g("s_n_seeds_sens"),
             ga_search_generations=g("s_ga_gen"), patience=g("s_patience"), se_rule=g("s_se_rule"))
    S["grid_act"] = g("s_grid_act") or list(SEARCH["grid_act"])
    S["grid_l2"] = g("s_grid_l2") or list(SEARCH["grid_l2"])
    if S["L_min"] > S["L_max"]:
        S["L_min"] = S["L_max"]
    return S


def search_sig(P):
    return repr((P["dsig"], st.session_state["model_key"], st.session_state["param_mode"],
                 sorted(manual_params().items(), key=lambda kv: kv[0]), search_cfg()))


def train_sig(P, params):
    g = st.session_state.get
    return repr((P["dsig"], g("model_key"), sorted(params.items(), key=lambda kv: kv[0]),
                 g("tahun_ramal"), g("n_seeds"), g("blend"), tuple(g("tahun_uji") or [])))


def effective_params(P):
    mode = st.session_state["param_mode"]
    if mode == "manual":
        return manual_params()
    res = st.session_state.get("search")
    return res["params"] if res and res["sig"] == search_sig(P) else None


def build_pipeline():
    P = {"level": 0, "use_ga": st.session_state["model_key"] == "GA-ELM",
         "nama_model": st.session_state["model_key"]}
    src = st.session_state.get("src")
    if not src:
        return P
    P["src"] = src
    try:
        sheets = list_sheets(src["bytes"])
    except Exception as e:                                           # noqa: BLE001
        P["error"] = f"Gagal membuka file Excel: {e}"
        return P
    P["sheets"] = sheets
    sane("sheet", sheets, DEFAULT_SHEET)
    sheet = st.session_state["sheet"]
    try:
        raw = read_sheet(src["bytes"], sheet)
    except Exception as e:                                           # noqa: BLE001
        P["error"] = f"Gagal membaca sheet '{sheet}': {e}"
        return P
    P["raw"], P["sheet"] = raw, sheet
    cols = [str(c) for c in raw.columns]
    P["columns"] = cols
    sig = (src["hash"], sheet)
    if st.session_state.get("map_sig") != sig:                       # file/sheet baru -> tebak ulang pemetaan
        guess = auto_map(cols)
        for i, role in enumerate(ROLES):
            st.session_state[f"map_{i}"] = guess.get(role, NONE)
        st.session_state["map_sig"] = sig
    chosen = {}
    for i, role in enumerate(ROLES):
        sane(f"map_{i}", [NONE] + cols, NONE)
        chosen[role] = st.session_state[f"map_{i}"]
    kosong = [ROLE_LABEL[r] for r in ROLES if chosen[r] == NONE]
    if kosong:
        P["error"] = "Pemetaan kolom belum lengkap. Pilih kolom untuk: " + "; ".join(kosong) + "."
        return P
    dobel = sorted({c for c in chosen.values() if list(chosen.values()).count(c) > 1})
    if dobel:
        P["error"] = f"Kolom file dipakai lebih dari satu peran: {dobel}. Setiap peran harus memakai kolom berbeda."
        return P
    by_name = {str(c): c for c in raw.columns}
    raw_std = raw[[by_name[chosen[r]] for r in ROLES]].copy()
    raw_std.columns = ROLES                                          # nama baku internal; tampilan memakai P["disp"]
    P["disp"] = dict(chosen)
    P["level"] = 1

    P["df"], P["report"], P["issue_mask"] = analyze(raw_std)
    P["level"] = 2

    mode = st.session_state["data_mode"]
    method = st.session_state["clean_method"]
    data, n_hilang, label, n_nan = prepare(P["df"], P["issue_mask"], mode, method)
    P.update(data=data, n_hilang=n_hilang, label_data=label, n_nan=n_nan)
    if len(data) == 0:
        P["error"] = "Tidak ada baris tersisa setelah pembersihan. Coba metode lain (mis. imputasi median)."
        return P
    P["level"] = 3

    lag_key = st.session_state["lag_key"]
    lags = LAG_OPTIONS[lag_key][0]
    feats = kolom_fitur(SERI, lags)
    try:
        agg, n_kosong, d_all = build_series(data, lags)
    except Exception as e:                                           # noqa: BLE001
        P["error"] = f"Gagal membentuk deret bulanan: {e}"
        return P
    if len(d_all) < 15:
        P["error"] = f"Sampel bulanan hanya {len(d_all)} — terlalu sedikit untuk dilatih (minimal ±15 bulan)."
        return P
    P.update(agg=agg, n_kosong=n_kosong, d_all=d_all, lags=lags, feats=feats, n_lag_hilang=len(agg) - len(d_all))
    P["level"] = 4

    sm = st.session_state["split_method"]
    n_th = st.session_state["split_n_tahun"]
    try:
        if sm == "tahun":
            df_train, df_test, uji = split_per_tahun(d_all, n_th)
            split_label = f"Per tahun: uji {', '.join(map(str, uji))} ({len(df_test)} bulan)"
        else:
            ratio, split_label = SPLIT_OPTIONS[st.session_state["split_key"]]
            df_train, df_test = split_sequential(d_all, ratio)
    except ValueError as e:
        P["error"] = str(e)
        return P
    P.update(df_train=df_train, df_test=df_test, split_label=split_label,
             agg_latih=agg[agg.index <= df_train.index.max()])
    P["level"] = 5
    P["dsig"] = (src["hash"], sheet, tuple(chosen.values()), mode, method if mode == "cleaning" else None, lag_key, sm,
                 n_th if sm == "tahun" else st.session_state["split_key"])
    return P


def dn(P, c):
    """Nama tampilan (sesuai kolom di file pengguna) untuk nama baku internal."""
    return P.get("disp", {}).get(c, c)


def unit_factor():
    return 1.0 if st.session_state["satuan"] == "ton" else 1.0 / KUINTAL_PER_TON


def compute_status(P, T):
    """ready = syarat teknis tahap terpenuhi; done = ready + sudah dikonfirmasi (Lanjut) + tahap sebelumnya done."""
    lv, ok_params = P["level"], P.get("params") is not None
    conf = st.session_state["confirmed"]
    ready = {1: lv >= 1, 2: lv >= 2, 3: lv >= 3, 4: lv >= 4, 5: lv >= 5, 6: lv >= 5,
             7: lv >= 5 and ok_params, 8: T is not None, 9: T is not None}
    done, op = {}, {1: True}
    for n in range(1, len(STAGES) + 1):
        done[n] = ready[n] and (n in conf or n >= 8) and (n == 1 or done[n - 1])
        op[n + 1] = done[n]
    return {"ready": ready, "done": done, "open": op}


demo_xlsx = st.cache_data(show_spinner=False)(make_demo_xlsx)
