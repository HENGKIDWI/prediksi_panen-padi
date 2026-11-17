"""Konfigurasi & konstanta aplikasi Peramalan Produksi Padi GA-ELM (tanpa Streamlit)."""
from __future__ import annotations

DEFAULT_SHEET = "Data awal"

FEATURES_ASLI = [
    "Luas tanam (Ha)", "luas panen (Ha)", "Curah Hujan (mm)", "Urea (kg)", "NPK (kg)",
    "Kelembapan_Tanah (%)", "Kelembapan Udara (%)", "Produktivitas (Ton/Ha)",
]
TARGET = "Produksi (Ha)"                       # produksi (satuan data: ton)
COLS_ASLI = FEATURES_ASLI + [TARGET]

SERI = {
    "tahun_col": "Tahun panen",
    "bulan_col": "Bulan Panen",
    "kolom_jumlah": ["Luas tanam (Ha)", "luas panen (Ha)", "Urea (kg)", "NPK (kg)"],
    "kolom_rata": ["Curah Hujan (mm)", "Kelembapan_Tanah (%)", "Kelembapan Udara (%)"],
    "tahun_ramal": 2026,
    "n_seeds": 10,
    "blend_musiman": 0.5,
    "tahun_uji_tambahan": [2024],
    "file_csv": "hasil_peramalan_triwulan.csv",
}
KUINTAL_PER_TON = 10

# Peran kolom yang dipetakan ke kolom file pengguna (nama baku di dalam program = nama pada notebook)
NONE = "— pilih kolom —"
ROLES = COLS_ASLI + [SERI["tahun_col"], SERI["bulan_col"]]
ROLE_LABEL = {**{c: f"Fitur: {c}" for c in FEATURES_ASLI}, TARGET: f"Target (produksi): {TARGET}",
              SERI["tahun_col"]: "Tahun panen", SERI["bulan_col"]: "Bulan panen"}

PARAMS = {
    "L": 16, "l2_reg": 1e-1, "act": "tanh", "w_range": 0.5,
    "pop_size": 20, "generations": 50, "cr": 0.8, "mr": 0.1, "mut_gene_rate": 0.2, "mut_step": 0.1,
    "ga_val_weight": 1.0, "ga_metric": "MAPE (%)", "ga_val_ratio": 0.2, "ga_patience": 20, "ga_seed": 42,
}

SEARCH = {
    "L_min": 4, "L_max": 64, "L_step": 4, "val_ratio": 0.2,
    "n_seeds": 5, "n_seeds_ga": 2, "n_seeds_sens": 3, "ga_search_generations": 20,
    "patience": 4, "se_rule": True,
    "grid_act": ["sigmoid", "tanh", "relu", "sin"],
    "grid_l2": [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0],
    "grid_w": [0.5],
    "grid_ga": {
        "ga_val_weight": [0.0, 0.5, 1.0], "pop_size": [10, 20, 40], "cr": [0.6, 0.8, 0.95],
        "mr": [0.05, 0.1, 0.3], "mut_gene_rate": [0.1, 0.2, 0.4], "mut_step": [0.05, 0.1, 0.2],
    },
}

SPLIT_METHODS = ["tahun", "rasio"]
SPLIT_METHOD_LABEL = {"tahun": "Per tahun (disarankan) — uji = tahun kalender terakhir",
                      "rasio": "Rasio persen"}
SPLIT_OPTIONS = {
    "1": (0.9, "90% latih / 10% uji"),
    "2": (0.8, "80% latih / 20% uji"),
    "3": (0.75, "75% latih / 25% uji"),
    "4": (0.7, "70% latih / 30% uji"),
    "5": (0.6, "60% latih / 40% uji"),
}


# pilihan fitur lag: kode -> (daftar lag, label)
LAG_OPTIONS = {
    "1": ((1,), "Lag 1 — produksi bulan sebelumnya"),
    "12": ((12,), "Lag 12 — produksi bulan yang sama tahun lalu (musiman)"),
    "1+12": ((1, 12), "Lag 1 + Lag 12"),
}

# =====================================================================================
# 2. PILIHAN DROPDOWN
# =====================================================================================
OPT = {
    "act": ["sigmoid", "tanh", "relu", "sin"],
    "L": list(range(4, 65, 4)),
    "l2_reg": [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0],
    "w_range": [0.1, 0.25, 0.5, 1.0],
    "pop_size": [10, 20, 30, 40, 60],
    "generations": [10, 20, 30, 50, 100, 150, 200],
    "cr": [0.6, 0.7, 0.8, 0.9, 0.95],
    "mr": [0.05, 0.1, 0.2, 0.3],
    "mut_gene_rate": [0.1, 0.2, 0.3, 0.4],
    "mut_step": [0.05, 0.1, 0.2],
    "ga_val_weight": [0.0, 0.5, 1.0],
    "ga_metric": ["MAPE (%)", "WAPE (%)"],
    "ga_val_ratio": [0.1, 0.2, 0.3],
    "ga_patience": [0, 10, 20, 30, 50],          # 0 = nonaktif
    "ga_seed": [1, 7, 42, 123, 2024],
    "n_seeds": [1, 3, 5, 10],
    "blend": [0.0, 0.25, 0.5, 0.75, 1.0],
    "satuan": ["ton", "kuintal"],
    "L_min": [2, 4, 8], "L_max": [32, 48, 64, 96, 128], "L_step": [2, 4, 8],
    "s_val_ratio": [0.1, 0.2, 0.3], "s_n_seeds": [3, 5, 7], "s_n_seeds_ga": [1, 2, 3],
    "s_n_seeds_sens": [2, 3, 5], "s_ga_gen": [10, 20, 30, 50], "s_patience": [2, 3, 4, 6],
    "s_se_rule": [True, False],
}
DATA_MODES = ["asli", "cleaning"]
DATA_MODE_LABEL = {"asli": "Data asli", "cleaning": "Data cleaning"}
CLEAN_METHODS = ["drop", "median"]
CLEAN_LABEL = {"drop": "Drop (hapus baris bermasalah)",
               "median": "Imputasi median (isi sel bermasalah dengan median kolom)"}
MODELS = ["ELM", "GA-ELM"]
MODEL_LABEL = {"ELM": "ELM saja (bobot acak)", "GA-ELM": "GA-ELM (bobot dioptimasi GA)"}
PARAM_MODES = ["manual", "elm", "full"]
PARAM_MODE_LABEL = {
    "manual": "Manual (atur sendiri di bawah)",
    "elm": "Cari otomatis: jumlah neuron + parameter ELM (aktivasi, l2_reg)",
    "full": "Cari otomatis lengkap: + parameter GA (fitness validasi, populasi, crossover, mutasi)",
}

DEFAULTS = {
    "stage": 0,
    "sheet": DEFAULT_SHEET, "data_mode": "asli", "clean_method": "drop",
    **{f"map_{i}": NONE for i in range(len(ROLES))},
    "lag_key": "1", "split_method": "tahun", "split_n_tahun": 1,
    "split_key": "2", "model_key": "GA-ELM", "param_mode": "manual",
    "satuan": "ton", "tahun_ramal": SERI["tahun_ramal"], "n_seeds": SERI["n_seeds"],
    "blend": SERI["blend_musiman"], "tahun_uji": list(SERI["tahun_uji_tambahan"]),
    **{f"p_{k}": v for k, v in PARAMS.items()},
    "s_L_min": SEARCH["L_min"], "s_L_max": SEARCH["L_max"], "s_L_step": SEARCH["L_step"],
    "s_val_ratio": SEARCH["val_ratio"], "s_n_seeds": SEARCH["n_seeds"],
    "s_n_seeds_ga": SEARCH["n_seeds_ga"], "s_n_seeds_sens": SEARCH["n_seeds_sens"],
    "s_ga_gen": SEARCH["ga_search_generations"], "s_patience": SEARCH["patience"],
    "s_se_rule": SEARCH["se_rule"], "s_grid_act": list(SEARCH["grid_act"]),
    "s_grid_l2": list(SEARCH["grid_l2"]),
}
WIDGET_KEYS = [k for k in DEFAULTS if k != "stage"]

STAGES = {
    1: ("Muat Data", "📂", "Unggah file Excel, pilih sheet, lihat cuplikan & distribusi."),
    2: ("Cek Anomali", "🔍", "Deteksi NaN, non-numerik, negatif, dan nol pada seluruh kolom."),
    3: ("Preprocessing", "🧹", "Pilih data asli atau cleaning (drop / imputasi median) sebelum dimodelkan."),
    4: ("Fitur Lag", "📅", "Jumlahkan kecamatan menjadi deret bulanan, lalu pilih fitur lag (lag 1 / lag 12 / keduanya)."),
    5: ("Split Data", "✂️", "Pembagian terurut waktu (tanpa shuffle): per tahun (disarankan) atau rasio persen."),
    6: ("Pilih Model", "🧠", "ELM biasa atau GA-ELM (bobot dioptimasi algoritma genetika)."),
    7: ("Parameter Model", "⚙️", "Atur manual atau cari otomatis neuron, aktivasi, l2_reg, dan parameter GA."),
    8: ("Pelatihan Model", "🚀", "Latih model multi-seed (data latih untuk evaluasi, seluruh data untuk peramalan)."),
    9: ("Evaluasi & Peramalan", "📊", "Akurasi per bulan & per Q, uji ketahanan, dan peramalan Q1–Q4 tahun target."),
}
