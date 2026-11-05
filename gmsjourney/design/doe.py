"""Space-filling designs and scenario composition (book Ch. 7, Appendix C).

phi_p optimisation follows Appendix C exactly:
  * relax the design to X in [0,1]^{n x m} and optimise an unconstrained Z with
    X = sigmoid(Z) (soft repulsion from the cube's walls);
  * evaluate log phi_p in log space with a logsumexp, so large p never overflows;
  * closed-form gradient dL/dX_ij = -sum_l G_il w_j^2 (X_ij - X_lj),
    G_il = d_il^{-(p+2)} / S, chained through the sigmoid;
  * anneal p geometrically from p_start to p_final, L-BFGS per stage;
  * initialise from a scrambled Sobol sequence clamped away from the tails;
  * cardinality-weighted distance w_j = |L_j| / mean|L| so many-level factors
    are not starved by binary ones.
The continuous design is discretised to factor levels by equal-width bins.

Composition mirrors the book's two arrangements: `cross` (every base x every
design row) and `embedded` (base is a balanced blocking factor). In embedded
mode rows are dealt to bases along a farthest-point ordering ("dispersed"), so
each base receives a spread of the factor space rather than a random clump.
"""
from __future__ import annotations

import itertools
import random
import warnings
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import chisquare, qmc

from ..core import QAItem, Scenario

LevelSpec = list[tuple[str, list[str]]]


# --------------------------------------------------------------------------
# phi_p machinery
# --------------------------------------------------------------------------
def _weights(level_specs: LevelSpec) -> np.ndarray:
    card = np.array([len(l) for _, l in level_specs], dtype=float)
    return card / card.mean()


def log_phi_p(X: np.ndarray, p: float, w: np.ndarray) -> float:
    D = X[:, None, :] - X[None, :, :]
    d = np.sqrt(np.sum((D * w) ** 2, axis=-1))
    iu = np.triu_indices(len(X), 1)
    dv = np.maximum(d[iu], 1e-12)
    return float(logsumexp(-p * np.log(dv)) / p)


def _loss_grad(zflat: np.ndarray, n: int, m: int, p: float, w: np.ndarray) -> tuple[float, np.ndarray]:
    Z = zflat.reshape(n, m)
    X = expit(Z)
    D = X[:, None, :] - X[None, :, :]
    d2 = np.sum((D * w) ** 2, axis=-1)
    np.fill_diagonal(d2, 1.0)
    logd = 0.5 * np.log(np.maximum(d2, 1e-24))
    iu = np.triu_indices(n, 1)
    logS = logsumexp(-p * logd[iu])
    L = logS / p
    logG = -(p + 2) * logd - logS
    np.fill_diagonal(logG, -np.inf)
    G = np.exp(logG)
    gX = -(G[:, :, None] * D).sum(axis=1) * (w ** 2)
    gZ = gX * X * (1 - X)
    return float(L), gZ.ravel()


def phi_p_design(level_specs: LevelSpec, n_runs: int, seed: int = 42, p_start: float = 2.0,
                 p_final: float = 50.0, stages: int = 5, iters: int = 60,
                 method: str = "sobol+refine") -> list[dict[str, str]]:
    if not level_specs:
        return [{} for _ in range(n_runs)]
    m = len(level_specs)
    w = _weights(level_specs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X0 = qmc.Sobol(d=m, scramble=True, seed=seed).random(n_runs)
    if method == "random":
        X = np.random.default_rng(seed).random((n_runs, m))
    elif method == "sobol":
        X = X0
    else:
        Z = logit(np.clip(X0, 0.02, 0.98))
        for p in np.geomspace(p_start, p_final, stages):
            res = minimize(_loss_grad, Z.ravel(), args=(n_runs, m, p, w), jac=True,
                           method="L-BFGS-B", options={"maxiter": iters})
            Z = res.x.reshape(n_runs, m)
        X = expit(Z)
        # Rank transform: keep the optimiser's relative arrangement but give every
        # column uniform (Latin-hypercube) marginals, so binning cannot starve the
        # middle levels; then improve phi_p by within-column level swaps, which
        # preserve the marginal counts exactly.
        X = (np.argsort(np.argsort(X, axis=0), axis=0) + 0.5) / n_runs
        idx = np.array([[min(int(X[i, j] * len(l)), len(l) - 1) for j, (_, l) in enumerate(level_specs)]
                        for i in range(n_runs)])
        idx = _swap_refine(idx, np.array([len(l) for _, l in level_specs]), w, seed=seed)
        return [{name: levels[idx[i, j]] for j, (name, levels) in enumerate(level_specs)} for i in range(n_runs)]
    return discretize(X, level_specs)


def _swap_refine(idx: np.ndarray, card: np.ndarray, w: np.ndarray, p: float = 15.0,
                 proposals: int = 4000, seed: int = 42) -> np.ndarray:
    """Greedy within-column swaps that lower sum d^-p on the level-centred encoding."""
    rng = np.random.default_rng(seed)
    n, m = idx.shape
    X = (idx + 0.5) / card
    D2 = np.sum(((X[:, None, :] - X[None, :, :]) * w) ** 2, axis=-1)
    np.fill_diagonal(D2, np.inf)
    for _ in range(proposals):
        j = int(rng.integers(m))
        a, b = rng.choice(n, 2, replace=False)
        if idx[a, j] == idx[b, j]:
            continue
        Xa, Xb = X[a].copy(), X[b].copy()
        Xa[j], Xb[j] = X[b, j], X[a, j]
        others = np.ones(n, bool)
        others[[a, b]] = False
        new_a = np.sum(((X[others] - Xa) * w) ** 2, axis=1)
        new_b = np.sum(((X[others] - Xb) * w) ** 2, axis=1)
        old = np.sum(D2[a, others] ** (-p / 2)) + np.sum(D2[b, others] ** (-p / 2))
        new = np.sum(np.maximum(new_a, 1e-12) ** (-p / 2)) + np.sum(np.maximum(new_b, 1e-12) ** (-p / 2))
        if new < old:
            X[a], X[b] = Xa, Xb
            idx[a, j], idx[b, j] = idx[b, j], idx[a, j]
            D2[a, others], D2[others, a] = new_a, new_a
            D2[b, others], D2[others, b] = new_b, new_b
    return idx


def discretize(X: np.ndarray, level_specs: LevelSpec) -> list[dict[str, str]]:
    rows = []
    for i in range(len(X)):
        row = {}
        for j, (name, levels) in enumerate(level_specs):
            k = min(int(X[i, j] * len(levels)), len(levels) - 1)
            row[name] = levels[k]
        rows.append(row)
    return rows


def simple_design(level_specs: LevelSpec, n_runs: int, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return [{n: rng.choice(l) for n, l in level_specs} for _ in range(n_runs)]


def balanced_design(factors: dict[str, list[Any]], num_cases: int, seed: int = 0) -> list[dict[str, Any]]:
    """proofloop.evaluation.doe.balanced_design (Apache-2.0, Knowledge/beyond-ship-and-pray).

    Each factor's levels appear num_cases / len(levels) times (+-1), independently
    shuffled per factor: perfect marginal balance, no control of joint (pairwise)
    coverage or space filling. The book's open-core baseline that the GMS
    Sobol+refine generator replaces; kept here as a comparison point and for the
    Appendix C threshold design."""
    if num_cases <= 0:
        return []
    if not factors:
        return [{} for _ in range(num_cases)]
    rng = random.Random(seed)
    out: list[dict[str, Any]] = [{} for _ in range(num_cases)]
    for fname, levels in factors.items():
        if not levels:
            continue
        assignments = [levels[i % len(levels)] for i in range(num_cases)]
        rng.shuffle(assignments)
        for i in range(num_cases):
            out[i][fname] = assignments[i]
    return out


def coverage_report(design: list[dict[str, Any]], factors: dict[str, list[Any]]) -> dict[str, dict[Any, int]]:
    """proofloop.evaluation.doe.coverage_report: level counts per factor."""
    out: dict[str, dict[Any, int]] = {}
    for fname, levels in factors.items():
        counts: dict[Any, int] = {lv: 0 for lv in levels}
        for row in design:
            v = row.get(fname)
            if v in counts:
                counts[v] += 1
        out[fname] = counts
    return out


def encode(rows: list[dict[str, str]], level_specs: LevelSpec) -> np.ndarray:
    return np.array([[(levels.index(r[n]) + 0.5) / len(levels) for n, levels in level_specs] for r in rows])


# --------------------------------------------------------------------------
# Design quality
# --------------------------------------------------------------------------
def design_quality(rows: list[dict[str, str]], level_specs: LevelSpec, p: float = 50.0) -> dict[str, Any]:
    X = encode(rows, level_specs)
    w = _weights(level_specs)
    D = X[:, None, :] - X[None, :, :]
    d = np.sqrt(np.sum((D * w) ** 2, axis=-1))
    iu = np.triu_indices(len(X), 1)
    marg = {}
    worst_p = 1.0
    for n, levels in level_specs:
        counts = [sum(1 for r in rows if r[n] == l) for l in levels]
        pv = float(chisquare(counts).pvalue) if len(levels) > 1 else 1.0
        worst_p = min(worst_p, pv)
        marg[n] = counts
    have, total = 0, 0
    for (a, la), (b, lb) in itertools.combinations(level_specs, 2):
        seen = {(r[a], r[b]) for r in rows}
        have += len(seen)
        total += len(la) * len(lb)
    dup = len(rows) - len({tuple(sorted(r.items())) for r in rows})
    return {"log_phi_p": log_phi_p(X, p, w), "min_distance": float(d[iu].min()) if len(iu[0]) else 0.0,
            "pairwise_level_coverage": have / max(1, total), "marginal_counts": marg,
            "worst_marginal_chi2_p": worst_p, "duplicate_rows": dup}


def compare_designs(level_specs: LevelSpec, n_runs: int, seed: int = 42) -> dict[str, dict[str, Any]]:
    return {
        "random": design_quality(simple_design(level_specs, n_runs, seed), level_specs),
        "balanced (proofloop)": design_quality(balanced_design(dict(level_specs), n_runs, seed), level_specs),
        "sobol": design_quality(phi_p_design(level_specs, n_runs, seed, method="sobol"), level_specs),
        "sobol+refine": design_quality(phi_p_design(level_specs, n_runs, seed), level_specs),
    }


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
def _farthest_point_order(X: np.ndarray, seed: int) -> list[int]:
    n = len(X)
    rng = np.random.default_rng(seed)
    order = [int(rng.integers(n))]
    dmin = np.linalg.norm(X - X[order[0]], axis=1)
    for _ in range(n - 1):
        dmin[order] = -1
        nxt = int(np.argmax(dmin))
        order.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1))
    return order


def compose(level_specs: LevelSpec, base_items: list[QAItem], n_runs: int, seed: int = 42,
            mode: str = "embedded", design_fn: Callable | None = None,
            assign: str = "dispersed") -> list[Scenario]:
    design_fn = design_fn or phi_p_design
    if not base_items:
        return []
    if mode == "cross":
        rows = design_fn(level_specs, n_runs, seed)
        return [Scenario(f"{it.qid}~{r:04d}", it, dict(row), "cross")
                for it in base_items for r, row in enumerate(rows)]
    rows = design_fn(level_specs, n_runs, seed)
    k = len(base_items)
    qids = [it.qid for it in base_items]
    if assign == "dispersed" and level_specs:
        assignment = _dispersed_assignment(encode(rows, level_specs), qids, seed)
    else:
        q = (qids * (len(rows) // k + 1))[:len(rows)]
        random.Random(seed).shuffle(q)
        assignment = dict(enumerate(q))
    by_qid = {it.qid: it for it in base_items}
    return [Scenario(f"{assignment[r]}~{r:04d}", by_qid[assignment[r]], dict(row), "embedded")
            for r, row in enumerate(rows)]


def _dispersed_assignment(X: np.ndarray, qids: list[str], seed: int) -> dict[int, str]:
    """Balanced blocking that also spreads each base's rows across the factor space.

    Rows are visited in farthest-point order; each goes to the base (with capacity
    left) whose already-assigned rows are farthest from it (maximin), so every base
    is exercised under dissimilar presentation conditions, not a random clump."""
    n, k = len(X), len(qids)
    cap = {q: n // k + (1 if i < n % k else 0) for i, q in enumerate(qids)}
    rng = random.Random(seed)
    order = _farthest_point_order(X, seed)
    members: dict[str, list[int]] = {q: [] for q in qids}
    out: dict[int, str] = {}
    for r in order:
        best, best_key = None, None
        for q in qids:
            if len(members[q]) >= cap[q]:
                continue
            dmin = min((float(np.linalg.norm(X[r] - X[m])) for m in members[q]), default=float("inf"))
            key = (dmin, -len(members[q]), rng.random())
            if best_key is None or key > best_key:
                best, best_key = q, key
        members[best].append(r)
        out[r] = best
    return out


def per_base_dispersion(scenarios: list[Scenario], level_specs: LevelSpec) -> float:
    """Mean over bases of the minimum distance between that base's rows (maximin; higher = more spread)."""
    by: dict[str, list[dict]] = {}
    for s in scenarios:
        by.setdefault(s.base.qid, []).append(s.factor_levels)
    vals = []
    for rows in by.values():
        if len(rows) < 2:
            continue
        X = encode(rows, level_specs)
        iu = np.triu_indices(len(X), 1)
        vals.append(float(np.linalg.norm(X[:, None] - X[None], axis=-1)[iu].min()))
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------
# Power (Appendix C.6): choose the run budget
# --------------------------------------------------------------------------
def power_curve(level_specs: LevelSpec, n_list: list[int], odds_ratio: float = 3.0,
                base_failure: float = 0.15, alpha: float = 0.05, reps: int = 150,
                seed: int = 7) -> list[dict[str, Any]]:
    """Monte Carlo power to detect one level of one 3-level factor that raises the odds
    of failure by `odds_ratio`, with a Bonferroni-adjusted LR test over all factors."""
    from ..evaluation.stats import deviance_test
    rng = np.random.default_rng(seed)
    target = max(level_specs, key=lambda x: len(x[1]) == 3)[0]
    m = len(level_specs)
    out = []
    for n in n_list:
        rows = phi_p_design(level_specs, n, seed)
        lv = dict(level_specs)[target]
        hits = 0
        for _ in range(reps):
            data = []
            for r in rows:
                odds = base_failure / (1 - base_failure) * (odds_ratio if r[target] == lv[-1] else 1.0)
                fail = rng.random() < odds / (1 + odds)
                data.append({target: r[target], "correct": 0 if fail else 1})
            if deviance_test(data, target, "correct")["p_value"] < alpha / m:
                hits += 1
        out.append({"n_runs": n, "power": hits / reps})
    return out
