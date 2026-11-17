"""ELM: aktivasi, bobot output (ridge), metrik, normalisasi (tanpa Streamlit)."""
from __future__ import annotations

import numpy as np

from core.config import TARGET


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


ACTIVATIONS = {"sigmoid": sigmoid, "tanh": np.tanh, "relu": lambda x: np.maximum(0, x), "sin": np.sin}


def hidden_layer(X, W, b, act="sigmoid"):
    return ACTIVATIONS[act](np.dot(X, W) + b)


def train_elm(X, Y, W, b, l2_reg, act="sigmoid"):
    H = hidden_layer(X, W, b, act)
    HTH = np.dot(H.T, H) + l2_reg * np.eye(H.shape[1])
    return np.linalg.solve(HTH, np.dot(H.T, Y))


def predict_elm(X, W, b, beta, act="sigmoid"):
    return np.dot(hidden_layer(X, W, b, act), beta)


def chrom_to_wb(chrom, d, L):
    return chrom[:d * L].reshape(d, L), chrom[d * L:].reshape(1, L)


def init_wb(seed, d, L, w_range):
    chrom = np.random.RandomState(seed).uniform(-w_range, w_range, d * L + L)
    return chrom_to_wb(chrom, d, L)


def calc_metrics(Y_true_norm, Y_pred_norm, y_min, y_max):
    y_true = Y_true_norm * (y_max - y_min) + y_min
    y_pred = Y_pred_norm * (y_max - y_min) + y_min
    mse = np.mean((y_true - y_pred) ** 2)
    nonzero = y_true != 0
    mape = (np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
            if nonzero.any() else np.nan)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sum_abs = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / sum_abs * 100 if sum_abs > 0 else np.nan
    return {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(y_true - y_pred)),
            "MAPE (%)": mape, "WAPE (%)": wape, "R2": r2}, y_pred


def make_normalizer(df_fit, feats):
    cols = feats + [TARGET]
    mn = df_fit[cols].min()
    sp = (df_fit[cols].max() - mn).replace(0, 1)
    y_min = mn[TARGET]
    y_max = y_min + sp[TARGET]

    def norm(d):
        X = ((d[feats] - mn[feats]) / sp[feats]).values
        Y = ((d[TARGET] - mn[TARGET]) / sp[TARGET]).values.reshape(-1, 1)
        return X, Y
    return norm, y_min, y_max
