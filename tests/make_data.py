"""Data sintetis berstruktur sama dengan dataset (per kecamatan-bulan panen) untuk pengujian."""
import numpy as np
import pandas as pd


def make_dataset(n_kec: int = 6, tahun=(2021, 2022, 2023, 2024, 2025), seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for k in range(n_kec):
        base = rng.uniform(80, 250)
        prod_th = {y: 4.5 + 0.1 * (y - tahun[0]) + rng.normal(0, 0.05) + 0.05 * k for y in tahun}
        for y in tahun:
            for b in range(1, 13):
                if y == tahun[0] and b < 4:
                    continue
                musim = max(0.0, np.sin((b - 1) * np.pi / 6) + 0.4)
                if rng.rand() < 0.3:
                    continue                                    # bulan tanpa panen
                luas = max(1.0, base * musim * (1 + 0.08 * (y - tahun[0])) * rng.uniform(0.6, 1.4))
                rows.append({
                    "Tahun": y, "kabupaten": "Bangkalan", "Kecamatan": f"Kec-{k + 1}",
                    "Luas tanam (Ha)": luas, "luas panen (Ha)": luas,
                    "Curah Hujan (mm)": rng.uniform(10, 300), "Urea (kg)": rng.uniform(1000, 5000),
                    "NPK (kg)": rng.uniform(500, 3000), "Kelembapan_Tanah (%)": rng.uniform(30, 90),
                    "Kelembapan Udara (%)": rng.uniform(50, 95), "Tahun panen": y, "Bulan Panen": b,
                    "Produktivitas (Ton/Ha)": prod_th[y], "Produksi (Ha)": luas * prod_th[y],
                })
    return pd.DataFrame(rows)
