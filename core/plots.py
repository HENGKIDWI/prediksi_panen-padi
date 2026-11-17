"""Visualisasi memakai matplotlib `Figure` langsung (thread-safe untuk Streamlit). Tanpa impor Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from core.config import TARGET


def plot_distribusi(df, cols, judul, labels=None):
    labels = labels or {}
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig = Figure(figsize=(14, 3.2 * nrow))
    axes = np.ravel(fig.subplots(nrow, ncol))
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=20, color="tab:blue", edgecolor="k", alpha=0.75)
        ax.set_title(labels.get(col, col), fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(judul, fontsize=13)
    fig.tight_layout()
    return fig


def plot_anomali(report, issue_mask):
    fig = Figure(figsize=(15, 5))
    a0, a1 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})
    n, x = len(report.columns), np.arange(len(report))
    w = 0.8 / n
    for i, c in enumerate(report.columns):
        a0.bar(x - 0.4 + w / 2 + i * w, report[c].values, w, label=c)
    a0.set_xticks(x)
    a0.set_xticklabels(report.index, rotation=45, ha="right", fontsize=8)
    a0.set_title("Jumlah Anomali per Kolom")
    a0.set_ylabel("Jumlah sel")
    a0.grid(axis="y", alpha=0.3)
    a0.legend()
    a1.imshow(issue_mask.values.T, aspect="auto", cmap="Reds", interpolation="nearest")
    a1.set_yticks(range(len(report)))
    a1.set_yticklabels(list(report.index), fontsize=8)
    a1.set_xlabel("Indeks baris")
    a1.set_title("Peta Sel Bermasalah (merah = anomali)")
    fig.tight_layout()
    return fig


def plot_perbandingan_data(df_awal, df_pakai, label, target_label=TARGET):
    fig = Figure(figsize=(14, 4.5))
    a0, a1 = fig.subplots(1, 2)
    a0.bar(["Data awal", label], [len(df_awal), len(df_pakai)], color=["tab:gray", "tab:green"], edgecolor="k")
    for i, v in enumerate([len(df_awal), len(df_pakai)]):
        a0.text(i, v, str(v), ha="center", va="bottom")
    a0.set_title("Jumlah Baris")
    a0.grid(axis="y", alpha=0.3)
    a1.plot(df_awal[TARGET].values, color="tab:gray", alpha=0.6, label="Data awal")
    a1.plot(df_pakai[TARGET].values, color="tab:green", label=label)
    a1.set_title(f"Target: {target_label}")
    a1.set_xlabel("Urutan baris")
    a1.grid(alpha=0.3)
    a1.legend()
    fig.tight_layout()
    return fig


def plot_deret(agg, fak, satuan_label):
    fig = Figure(figsize=(12, 4))
    ax = fig.subplots()
    ax.plot(agg.index.to_timestamp(), agg[TARGET].values * fak, marker="o", color="tab:blue", lw=1.8)
    ax.set_title("Produksi total per bulan (semua kecamatan)", fontweight="bold")
    ax.set_ylabel(f"Produksi ({satuan_label})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_split(df_train, df_test, target_label=TARGET):
    fig = Figure(figsize=(12, 4))
    ax = fig.subplots()
    xtr = np.arange(len(df_train))
    xte = np.arange(len(df_train), len(df_train) + len(df_test))
    ax.plot(xtr, df_train[TARGET].values, color="tab:blue", label=f"Latih ({len(df_train)})")
    ax.plot(xte, df_test[TARGET].values, color="tab:orange", label=f"Uji ({len(df_test)})")
    ax.axvline(len(df_train) - 0.5, color="r", ls="--", label="Batas split")
    ax.set_title(f"Split Data Terurut - {target_label}")
    ax.set_xlabel("Urutan bulan")
    ax.set_ylabel(target_label)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_grid_elm(hasil, pilih):
    fig = Figure(figsize=(15, 4.8))
    a0, a1 = fig.subplots(1, 2)
    for act, g in hasil.groupby("act"):
        best = g.groupby("l2_reg")["rmse_val"].min()
        a0.plot(best.index, best.values, marker="o", label=act)
    a0.set_xscale("log")
    a0.axvline(pilih["l2_reg"], color="r", ls=":", label="l2_reg terpilih")
    a0.set_title("RMSE validasi terbaik per aktivasi vs l2_reg")
    a0.set_xlabel("l2_reg")
    a0.set_ylabel("RMSE validasi")
    a0.grid(alpha=0.3)
    a0.legend(fontsize=8)
    g = hasil[(hasil["act"] == pilih["act"]) & (hasil["l2_reg"] == pilih["l2_reg"]) & (hasil["w_range"] == pilih["w_range"])]
    a1.errorbar(g["L"], g["rmse_val"], yerr=g["std"], marker="o", capsize=3, color="tab:blue")
    a1.axvline(pilih["L"], color="r", ls=":", label=f"L terpilih = {int(pilih['L'])}")
    a1.set_title(f"Kurva jumlah neuron ({pilih['act']}, l2={pilih['l2_reg']:g})")
    a1.set_xlabel("Jumlah neuron (L)")
    a1.set_ylabel("RMSE validasi (rata-rata ± std antar-seed)")
    a1.grid(alpha=0.3)
    a1.legend()
    fig.tight_layout()
    return fig


def plot_pencarian_neuron(hasil, pilih, judul):
    fig = Figure(figsize=(10, 4.5))
    ax = fig.subplots()
    ax.errorbar(hasil["L"], hasil["rmse_val"], yerr=hasil["std"], marker="o", capsize=3, color="tab:blue",
                label="RMSE validasi (rata-rata ± std antar-seed)")
    ax.axvline(pilih["L"], color="r", ls=":", lw=2, label=f"L terpilih = {int(pilih['L'])}")
    ax.set_title(judul)
    ax.set_xlabel("Jumlah neuron (L)")
    ax.set_ylabel("RMSE validasi")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sensitivitas_ga(sens):
    params = list(sens["param"].unique())
    ncol = 3
    nrow = int(np.ceil(len(params) / ncol))
    fig = Figure(figsize=(14, 3.6 * nrow))
    axes = np.ravel(fig.subplots(nrow, ncol))
    for ax, name in zip(axes, params):
        g = sens[sens["param"] == name]
        x = np.arange(len(g))
        ax.errorbar(x, g["rmse_val"], yerr=g["std"], marker="o", capsize=3, color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in g["nilai"]])
        for xi, dp in zip(x, g["dipilih"]):
            if dp:
                ax.axvline(xi, color="r", ls=":")
        ax.set_title(name)
        ax.set_ylabel("RMSE validasi")
        ax.grid(alpha=0.3)
    for ax in axes[len(params):]:
        ax.axis("off")
    fig.suptitle("Sensitivitas Parameter GA (garis merah = nilai terpilih)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_ga_history(history):
    h = pd.DataFrame(history)
    fig = Figure(figsize=(15, 4))
    axes = fig.subplots(1, 3)
    for ax, (col, ttl, c) in zip(axes, [("fitness", "Fitness terbaik", "tab:green"),
                                        ("mape", "Error latih (%) (metrik fitness)", "tab:red"),
                                        ("rmse", "RMSE latih", "tab:purple")]):
        ax.plot(h["gen"], h[col], color=c, lw=1.8)
        ax.set_title(ttl)
        ax.set_xlabel("Generasi")
        ax.grid(alpha=0.3)
    fig.suptitle("Konvergensi GA", fontsize=13)
    fig.tight_layout()
    return fig


def plot_evaluasi(periode, aktual, prediksi, nama_model, judul, satuan_label):
    fig = Figure(figsize=(12, 4))
    ax = fig.subplots()
    x = periode.to_timestamp()
    ax.plot(x, aktual, marker="o", color="tab:blue", lw=2, label="Aktual")
    ax.plot(x, prediksi, marker="s", color="tab:orange", lw=2, ls="--", label=f"Prediksi {nama_model}")
    ax.set_title(judul, fontweight="bold")
    ax.set_ylabel(f"Produksi ({satuan_label})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_q(tabel, nama_model, judul, satuan_label, kol_a="Aktual (Ton)", kol_p="Prediksi (Ton)",
           lab_a="Aktual", lab_p=None):
    """Grafik per Q — garis, gaya sama dengan grafik per bulan."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    x = np.arange(len(tabel))
    if kol_a:
        ax.plot(x, tabel[kol_a].values, marker="o", color="tab:blue", lw=2, label=lab_a)
    ax.plot(x, tabel[kol_p].values, marker="s", color="tab:orange", lw=2, ls="--", label=lab_p or f"Prediksi {nama_model}")
    ax.set_xticks(x)
    ax.set_xticklabels(tabel.index)
    ax.set_title(judul, fontweight="bold")
    ax.set_ylabel(f"Produksi ({satuan_label})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ramalan(agg, periode, rata, bawah, atas, nama_model, fak, satuan_label):
    fig = Figure(figsize=(12, 4.5))
    ax = fig.subplots()
    ax.plot(agg.index.to_timestamp(), agg[TARGET].values * fak, marker="o", color="tab:blue", lw=2, label="Aktual")
    xr = periode.to_timestamp()
    ax.plot(xr, rata, marker="s", color="tab:orange", lw=2, ls="--", label=f"Ramalan {nama_model}")
    ax.fill_between(xr, bawah, atas, color="tab:orange", alpha=0.15, label="Rentang min-maks antar seed")
    ax.set_title("Peramalan produksi total per bulan", fontweight="bold")
    ax.set_ylabel(f"Produksi ({satuan_label})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
