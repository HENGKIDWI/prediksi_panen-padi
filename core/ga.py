"""Genetic Algorithm untuk mengoptimasi bobot input (W) & bias (b) ELM -> GA-ELM (tanpa Streamlit)."""
from __future__ import annotations

import numpy as np

from core.elm import calc_metrics, chrom_to_wb, predict_elm, train_elm


def _say(log, msg):
    if log is not None:
        log(msg)


def roulette(pop, fits, rng):
    total = fits.sum()
    probs = fits / total if total > 0 else np.full(len(fits), 1 / len(fits))
    idx = np.minimum(np.searchsorted(np.cumsum(probs), rng.rand(len(pop))), len(pop) - 1)
    return pop[idx].copy()


def crossover(pop, cr, rng):
    for i in range(0, len(pop) - 1, 2):
        if rng.rand() < cr:
            a = rng.rand(pop.shape[1])
            p1, p2 = pop[i].copy(), pop[i + 1].copy()
            pop[i] = a * p1 + (1 - a) * p2
            pop[i + 1] = (1 - a) * p1 + a * p2
    return pop


def mutate(pop, mr, gene_rate, rng, step=0.1):
    for i in range(len(pop)):
        if rng.rand() < mr:
            mask = rng.rand(pop.shape[1]) < gene_rate
            pop[i] += mask * rng.uniform(-step, step, pop.shape[1])
    return pop


def run_ga(X, Y, y_min, y_max, p, log=None, X_val=None, Y_val=None):
    """GA dengan elitism. Fitness = 1/(1+error_gabungan); berhenti dini bila fitness tak naik `ga_patience` generasi."""
    L, d = p["L"], X.shape[1]
    act, w_range = p["act"], p["w_range"]
    rng = np.random.RandomState(p["ga_seed"])
    use_val = X_val is not None and p["ga_val_weight"] > 0
    wv = p["ga_val_weight"] if use_val else 0.0
    metrik = p.get("ga_metric", "MAPE (%)")
    nm = metrik.replace(" (%)", "")

    def evaluate(P):
        fits, ms = [], []
        for chrom in P:
            W, b = chrom_to_wb(chrom, d, L)
            try:
                beta = train_elm(X, Y, W, b, p["l2_reg"], act)
                m, _ = calc_metrics(Y, predict_elm(X, W, b, beta, act), y_min, y_max)
                mape = m[metrik]
                if use_val:
                    mv, _ = calc_metrics(Y_val, predict_elm(X_val, W, b, beta, act), y_min, y_max)
                    m["MAPE_val"] = mv[metrik]
                    mape = (1 - wv) * mape + wv * mv[metrik]
                f = 1 / (1 + mape)
                if not np.isfinite(f):
                    f = 0.0
            except np.linalg.LinAlgError:
                m, f = {"RMSE": np.inf, "MAPE (%)": np.inf, "WAPE (%)": np.inf}, 0.0
            fits.append(f)
            ms.append(m)
        return np.array(fits), ms

    def rec(gen, fit, m):
        return {"gen": gen, "fitness": fit, "rmse": m["RMSE"], "mape": m[metrik],
                "mape_val": m.get("MAPE_val", np.nan)}

    pop = rng.uniform(-w_range, w_range, (p["pop_size"], d * L + L))
    fits, ms = evaluate(pop)
    j = int(np.argmax(fits))
    best_fit, best_chrom, best_m = fits[j], pop[j].copy(), ms[j]
    history = [rec(0, best_fit, best_m)]
    stale = 0

    for gen in range(p["generations"]):
        parents = roulette(pop, fits, rng)
        children = mutate(crossover(parents, p["cr"], rng), p["mr"], p["mut_gene_rate"], rng, p.get("mut_step", 0.1))
        c_fits, c_ms = evaluate(children)
        all_pop = np.vstack([pop, children])
        all_fit = np.concatenate([fits, c_fits])
        all_ms = ms + c_ms
        order = np.argsort(-all_fit)[:p["pop_size"]]
        pop, fits, ms = all_pop[order], all_fit[order], [all_ms[i] for i in order]
        j = int(np.argmax(fits))
        if fits[j] > best_fit:
            best_fit, best_chrom, best_m = fits[j], pop[j].copy(), ms[j]
            stale = 0
        else:
            stale += 1
        history.append(rec(gen + 1, best_fit, best_m))
        if log and ((gen + 1) % 10 == 0 or gen == 0):
            extra = f" | {nm} valid {best_m['MAPE_val']:.2f}%" if "MAPE_val" in best_m else ""
            _say(log, f"Generasi {gen + 1}/{p['generations']} | Fitness {best_fit:.6f} | "
                      f"RMSE latih {best_m['RMSE']:.4f} | {nm} latih {best_m[metrik]:.2f}%{extra}")
        if p.get("ga_patience") and stale >= p["ga_patience"]:
            _say(log, f"Berhenti dini di generasi {gen + 1}: fitness tidak naik {p['ga_patience']} generasi.")
            break

    W, b = chrom_to_wb(best_chrom, d, L)
    return W, b, history


def run_ga_final(X, Y, y_min, y_max, p, log=None):
    if p["ga_val_weight"] > 0 and len(X) > 10:
        n_iv = min(max(int(round(len(X) * p["ga_val_ratio"])), 1), len(X) - 1)
        return run_ga(X[:-n_iv], Y[:-n_iv], y_min, y_max, p, log, X_val=X[-n_iv:], Y_val=Y[-n_iv:])
    return run_ga(X, Y, y_min, y_max, p, log)
