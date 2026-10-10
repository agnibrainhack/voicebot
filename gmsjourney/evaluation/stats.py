"""Statistics for failure attribution (book Ch. 10), dependency-light (numpy/scipy).

* ``logit_fit``        binomial GLM with logit link by IRLS (a tiny ridge keeps
                       separated designs finite; it is reported, not hidden).
* ``deviance_test``    per-factor likelihood-ratio test: G^2 = D_null - D_full on
                       L-1 df, McFadden pseudo-R^2.
* ``bh_adjust``        Benjamini-Hochberg false-discovery-rate correction.
* ``wilson``           Wilson score interval for a level's failure rate.
* ``odds_ratios``      per-level odds ratios vs the reference level with Wald CI
                       (Haldane-Anscombe +0.5 when a cell is empty).
* ``joint_model``      all factors at once: pseudo-R^2, AIC, BIC, ROC AUC and the
                       Hosmer-Lemeshow calibration p-value.
* ``interaction_test`` additive vs product-term model for a pair of factors.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats


def _design(rows: Sequence[dict], factors: list[str], interactions: list[tuple[str, str]] = ()) -> tuple[np.ndarray, list[str], dict]:
    cols = [np.ones(len(rows))]
    names = ["(intercept)"]
    levels = {}
    for f in factors:
        lv = sorted({str(r[f]) for r in rows})
        levels[f] = lv
        for l in lv[1:]:
            cols.append(np.array([1.0 if str(r[f]) == l else 0.0 for r in rows]))
            names.append(f"{f}={l}")
    for a, b in interactions:
        for la in levels[a][1:]:
            for lb in levels[b][1:]:
                cols.append(np.array([1.0 if (str(r[a]) == la and str(r[b]) == lb) else 0.0 for r in rows]))
                names.append(f"{a}={la}:{b}={lb}")
    return np.column_stack(cols), names, levels


def logit_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4, iters: int = 100) -> dict[str, Any]:
    n, k = X.shape
    beta = np.zeros(k)
    pen = ridge * np.eye(k)
    pen[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - pen @ beta
        H = X.T @ (X * W[:, None]) + pen
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ beta, -30, 30)
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    try:
        cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + pen)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    return {"beta": beta, "ll": ll, "p": p, "cov": cov, "k": k}


def bh_adjust(pvals: Sequence[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def deviance_test(rows: Sequence[dict], factor: str, metric: str) -> dict[str, Any]:
    y = np.array([float(r[metric]) for r in rows])
    X0 = np.ones((len(rows), 1))
    null = logit_fit(X0, y)
    X, names, levels = _design(rows, [factor])
    full = logit_fit(X, y)
    df = X.shape[1] - 1
    G2 = max(0.0, 2 * (full["ll"] - null["ll"]))
    p = float(stats.chi2.sf(G2, df)) if df > 0 else 1.0
    r2 = 1 - full["ll"] / null["ll"] if null["ll"] < 0 else 0.0
    return {"factor": factor, "deviance": G2, "df": df, "p_value": p, "pseudo_r2": r2, "levels": levels[factor]}


def level_table(rows: Sequence[dict], factor: str, metric: str) -> list[dict[str, Any]]:
    """Failure rate per level with Wilson CI, and odds ratio of success vs the reference level."""
    lv = sorted({str(r[factor]) for r in rows})
    out = []
    ref = [r for r in rows if str(r[factor]) == lv[0]]
    a_ref = sum(int(r[metric]) for r in ref)
    b_ref = len(ref) - a_ref
    for l in lv:
        sub = [r for r in rows if str(r[factor]) == l]
        succ = sum(int(r[metric]) for r in sub)
        fail = len(sub) - succ
        lo, hi = wilson(fail, len(sub))
        a, b, c, d = succ, fail, a_ref, b_ref
        if 0 in (a, b, c, d):
            a, b, c, d = a + .5, b + .5, c + .5, d + .5
        orr = (a / b) / (c / d)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        out.append({"level": l, "n": len(sub), "failures": fail, "failure_rate": fail / max(1, len(sub)),
                    "wilson_lo": lo, "wilson_hi": hi, "odds_ratio_success": orr,
                    "or_ci": (math.exp(math.log(orr) - 1.96 * se), math.exp(math.log(orr) + 1.96 * se)),
                    "reference": l == lv[0]})
    return out


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> float:
    order = np.argsort(p)
    bins = np.array_split(order, groups)
    hl = 0.0
    g_used = 0
    for b in bins:
        if len(b) == 0:
            continue
        o, e, n = y[b].sum(), p[b].sum(), len(b)
        if 0 < e < n:
            hl += (o - e) ** 2 / (e * (1 - e / n))
            g_used += 1
    return float(stats.chi2.sf(hl, max(1, g_used - 2)))


def joint_model(rows: Sequence[dict], factors: list[str], metric: str) -> dict[str, Any]:
    y = np.array([float(r[metric]) for r in rows])
    null = logit_fit(np.ones((len(rows), 1)), y)
    X, names, _ = _design(rows, factors)
    full = logit_fit(X, y)
    n, k = X.shape
    return {"pseudo_r2": 1 - full["ll"] / null["ll"] if null["ll"] < 0 else 0.0,
            "aic": 2 * k - 2 * full["ll"], "bic": k * math.log(n) - 2 * full["ll"],
            "roc_auc": auc(y, full["p"]), "hl_p_value": hosmer_lemeshow(y, full["p"]),
            "n": n, "parameters": k}


def interaction_test(rows: Sequence[dict], a: str, b: str, metric: str) -> dict[str, Any]:
    y = np.array([float(r[metric]) for r in rows])
    Xa, _, _ = _design(rows, [a, b])
    Xi, _, _ = _design(rows, [a, b], [(a, b)])
    fa, fi = logit_fit(Xa, y), logit_fit(Xi, y)
    df = Xi.shape[1] - Xa.shape[1]
    G2 = max(0.0, 2 * (fi["ll"] - fa["ll"]))
    return {"pair": (a, b), "deviance": G2, "df": df, "p_value": float(stats.chi2.sf(G2, df)) if df else 1.0}
