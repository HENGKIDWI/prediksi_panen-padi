"""Pencarian parameter otomatis: jumlah neuron, aktivasi, l2_reg, dan sensitivitas parameter GA (tanpa Streamlit)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from core.elm import calc_metrics, init_wb, make_normalizer, predict_elm, train_elm
from core.ga import _say, run_ga_final


def make_ctx(df_train, S, feats):
    n_sel = min(max(int(round(len(df_train) * S["val_ratio"])), 1), len(df_train) - 1)
    df_rest, df_sel = df_train.iloc[:-n_sel], df_train.iloc[-n_sel:]
    norm, y_min, y_max = make_normalizer(df_rest, feats)
    X_rest, Y_rest = norm(df_rest)
    X_sel, Y_sel = norm(df_sel)
    return {"X_rest": X_rest, "Y_rest": Y_rest, "X_sel": X_sel, "Y_sel": Y_sel, "y_min": y_min, "y_max": y_max}


def eval_config(cfg, ctx, use_ga, seeds):
    d = ctx["X_rest"].shape[1]
    scores = []
    for s in seeds:
        if use_ga:
            W, b, _ = run_ga_final(ctx["X_rest"], ctx["Y_rest"], ctx["y_min"], ctx["y_max"], dict(cfg, ga_seed=s))
        else:
            W, b = init_wb(s, d, cfg["L"], cfg["w_range"])
        beta = train_elm(ctx["X_rest"], ctx["Y_rest"], W, b, cfg["l2_reg"], cfg["act"])
        m, _ = calc_metrics(ctx["Y_sel"], predict_elm(ctx["X_sel"], W, b, beta, cfg["act"]), ctx["y_min"], ctx["y_max"])
        scores.append(m["RMSE"])
    return np.array(scores)


def search_L(cfg, ctx, S, use_ga, log=None):
    n_seeds = S["n_seeds_ga"] if use_ga else S["n_seeds"]
    seeds = [cfg["ga_seed"] + i for i in range(n_seeds)]
    rows, best_mean, stagnan = [], np.inf, 0
    for L in range(S["L_min"], S["L_max"] + 1, S["L_step"]):
        sc = eval_config(dict(cfg, L=L), ctx, use_ga, seeds)
        mean, std = float(sc.mean()), float(sc.std())
        rows.append({"act": cfg["act"], "l2_reg": cfg["l2_reg"], "w_range": cfg["w_range"],
                     "L": L, "rmse_val": mean, "std": std})
        if mean < best_mean:
            best_mean, stagnan, tanda = mean, 0, "<- terbaik sejauh ini"
        else:
            stagnan += 1
            tanda = f"(tanpa perbaikan {stagnan}/{S['patience']})"
        _say(log, f"L={L:>4} | RMSE validasi {mean:,.4f} ± {std:,.4f} {tanda}")
        if stagnan >= S["patience"]:
            _say(log, f"Berhenti lebih awal: {S['patience']} kandidat berturut-turut tidak membaik.")
            break
    return rows


def pilih_baris(hasil, se_rule):
    best = hasil.loc[hasil["rmse_val"].idxmin()]
    if not se_rule:
        return best
    layak = hasil[hasil["rmse_val"] <= best["rmse_val"] + best["std"]]
    return layak.sort_values(["L", "rmse_val"]).iloc[0]


def cari_elm(base, ctx, S, log=None):
    combos = list(itertools.product(S["grid_act"], S["grid_l2"], S["grid_w"]))
    rows = []
    for i, (act, l2, w) in enumerate(combos, 1):
        r = search_L(dict(base, act=act, l2_reg=l2, w_range=w), ctx, S, use_ga=False)
        rows += r
        rb = min(r, key=lambda x: x["rmse_val"])
        _say(log, f"[{i:>2}/{len(combos)}] aktivasi={act:<7} l2={l2:<7g} w={w:<4g} | {len(r):>2} kandidat L | "
                  f"terbaik L={rb['L']:>3} RMSE val {rb['rmse_val']:,.4f} ± {rb['std']:,.4f}")
    hasil = pd.DataFrame(rows)
    return hasil, pilih_baris(hasil, S["se_rule"])


def sensitivitas_ga(cfg, ctx, S, log=None):
    seeds = [cfg["ga_seed"] + i for i in range(S["n_seeds_sens"])]
    cfg, rows = dict(cfg), []
    for name, values in S["grid_ga"].items():
        nilai = sorted(set(list(values) + [cfg[name]]))
        res = []
        for v in nilai:
            sc = eval_config(dict(cfg, **{name: v}), ctx, True, seeds)
            res.append((v, float(sc.mean()), float(sc.std())))
        best = min(res, key=lambda r: r[1])
        cur = next(r for r in res if r[0] == cfg[name])
        pilih = cur if (not S["se_rule"] or cur[1] <= best[1] + best[2]) else best
        for v, mean, std in res:
            rows.append({"param": name, "nilai": v, "rmse_val": mean, "std": std, "dipilih": v == pilih[0]})
        ringkas = " | ".join(f"{v:g}: {mean:,.3f}" for v, mean, _ in res)
        _say(log, f"{name:<14} {ringkas}  => {pilih[0]:g}")
        cfg[name] = pilih[0]
    return cfg, pd.DataFrame(rows)


def jalankan_pencarian(df_train, base, S, use_ga, cari_ga, feats, log=None):
    ctx = make_ctx(df_train, S, feats)
    _say(log, f"Data latih dibagi (urutan waktu dijaga): latih={len(ctx['X_rest'])} | validasi seleksi={len(ctx['X_sel'])}")
    out = {}

    _say(log, "**TAHAP 1 — ELM biasa:** aktivasi × l2_reg × jumlah neuron")
    out["elm"], p1 = cari_elm(base, ctx, S, log)
    out["elm_pilih"] = p1
    cfg = dict(base, act=p1["act"], l2_reg=float(p1["l2_reg"]), w_range=float(p1["w_range"]), L=int(p1["L"]))
    _say(log, f"✅ Terpilih (ELM): aktivasi={cfg['act']}, l2_reg={cfg['l2_reg']:g}, w_range={cfg['w_range']:g}, L={cfg['L']}")

    if use_ga:
        _say(log, "**TAHAP 2 — GA-ELM:** jumlah neuron (aktivasi & l2_reg dari tahap 1)")
        cfg_s = dict(cfg, generations=S["ga_search_generations"])
        out["gaL"] = pd.DataFrame(search_L(cfg_s, ctx, S, use_ga=True, log=log))
        out["gaL_pilih"] = pilih_baris(out["gaL"], S["se_rule"])
        cfg["L"] = int(out["gaL_pilih"]["L"])
        _say(log, f"✅ Terpilih (GA-ELM): L={cfg['L']}")

        if cari_ga:
            _say(log, "**TAHAP 3 — sensitivitas parameter GA** (RMSE validasi, lebih kecil lebih baik)")
            cfg_s = dict(cfg, generations=S["ga_search_generations"])
            cfg_s, out["sens"] = sensitivitas_ga(cfg_s, ctx, S, log)
            for k in S["grid_ga"]:
                cfg[k] = cfg_s[k]
    out["cfg"] = cfg
    return out
