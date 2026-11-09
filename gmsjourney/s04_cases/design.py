"""Which cases to run: covering designs for journeys, space-filling designs for wordings.

There are millions of possible journeys and wordings; a campaign runs a few hundred. Two designs pick them so
that little is missed, like seating a dinner party so every pair of guests meets at least once.

Parts of this file:
  1. Space-filling designs for the enrichment factors (Appendix C): phi_p (maximin) optimisation, Latin-hypercube
     marginals, level swaps, balanced blocking over bases, design-quality metrics and a Monte Carlo power curve.
  2. ``JourneyBaseGenerator``: journeys as a covering array (every feasible pair of factor levels in some case,
     counting only levels relevant to that case), then a closure that adds cases until every rule, transition,
     handoff trigger and boundary is exercised, and turn-limit cases padded to their exact target.
"""
from __future__ import annotations

import copy
import itertools
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import chisquare, qmc


# =====================================================================================================================
# PART 1 - Space-filling designs and scenario composition (book Ch. 7, Appendix C).
#
# phi_p optimisation follows Appendix C exactly:
#   * relax the design to X in [0,1]^{n x m} and optimise an unconstrained Z with
#     X = sigmoid(Z) (soft repulsion from the cube's walls);
#   * evaluate log phi_p in log space with a logsumexp, so large p never overflows;
#   * closed-form gradient dL/dX_ij = -sum_l G_il w_j^2 (X_ij - X_lj),
#     G_il = d_il^{-(p+2)} / S, chained through the sigmoid;
#   * anneal p geometrically from p_start to p_final, L-BFGS per stage;
#   * initialise from a scrambled Sobol sequence clamped away from the tails;
#   * cardinality-weighted distance w_j = |L_j| / mean|L| so many-level factors
#     are not starved by binary ones.
# The continuous design is discretised to factor levels by equal-width bins.
#
# Composition mirrors the book's two arrangements: `cross` (every base x every
# design row) and `embedded` (base is a balanced blocking factor). In embedded
# mode rows are dealt to bases along a farthest-point ordering ("dispersed"), so
# each base receives a spread of the factor space rather than a random clump.
# =====================================================================================================================

from ..s05_rewording.catalog import QAItem, Scenario

LevelSpec = list[tuple[str, list[str]]]


# --------------------------------------------------------------------------
# phi_p machinery
# --------------------------------------------------------------------------
def _weights(level_specs: LevelSpec) -> np.ndarray:
    card = np.array([len(l) for _, l in level_specs], dtype=float)
    return card / card.mean()


def log_phi_p(X: np.ndarray, p: float, w: np.ndarray) -> float:
    """log phi_p of a design X (rows = runs, columns in [0,1]) with column weights w.

    phi_p = (sum over pairs of d_ij^-p)^(1/p): small when no two runs are close (maximin as p grows). Computed in
    log space with logsumexp so a large p never overflows.
    """
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
    """A space-filling design of ``n_runs`` rows over the factor levels (Appendix C): Sobol start, phi_p
    minimised by L-BFGS with p annealed from p_start to p_final, then rank-transformed (Latin-hypercube marginals) and
    refined by level swaps that keep every level's count; ``method`` picks a simpler variant for comparison.
    """
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
    """Map a continuous design in [0,1] to factor levels by equal-width bins."""
    rows = []
    for i in range(len(X)):
        row = {}
        for j, (name, levels) in enumerate(level_specs):
            k = min(int(X[i, j] * len(levels)), len(levels) - 1)
            row[name] = levels[k]
        rows.append(row)
    return rows


def simple_design(level_specs: LevelSpec, n_runs: int, seed: int = 42) -> list[dict[str, str]]:
    """A random design (each level drawn uniformly): the baseline every other design is compared with."""
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
    """Map level rows back into [0,1] (the centre of each level's bin), to measure distances."""
    return np.array([[(levels.index(r[n]) + 0.5) / len(levels) for n, levels in level_specs] for r in rows])


# --------------------------------------------------------------------------
# Design quality
# --------------------------------------------------------------------------
def design_quality(rows: list[dict[str, str]], level_specs: LevelSpec, p: float = 50.0) -> dict[str, Any]:
    """How good a design is: log phi_p, minimum distance, marginal balance (worst chi-square p) and the
    share of level pairs covered.
    """
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
    """The quality of random, balanced (proofloop), Sobol and Sobol+refine designs on the same factors."""
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
    """Scenarios = bases x design rows. ``cross``: every base with every row; ``embedded``: rows dealt to bases
    as a balanced blocking factor, in a farthest-point order so each base sees dissimilar conditions.
    """
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
    from ..s07_checks.diagnosis import deviance_test
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


# =====================================================================================================================
# PART 2 - Optimal journey base cases: constrained covering array + coverage closure.
#
# The content dimensions (journey.DIMENSIONS) are what a base case *is*. Testing
# every combination is infeasible (millions), and random sampling wastes cases on
# irrelevant ones. Instead:
#
# 1. Relevance. ``relevant_dims`` collapses dimensions that cannot influence the
#    ground truth of a case to N/A (after a failed authentication, the number of
#    transactions does not matter). A level pair only counts as covered when both
#    levels are relevant in the same case, so coverage is never inflated.
# 2. Feasibility. A pair is a coverage target only if some valid, relevant case
#    contains it (estimated by constrained random completion); infeasible pairs
#    are reported, not silently dropped.
# 3. Covering array (AETG-style greedy). Each new case is the best of many
#    candidates, each seeded with an uncovered pair and hill-climbed to cover as
#    many further uncovered pairs as possible. Turn-pressure cases are made exact
#    by padding with policy questions until the oracle reports exactly 20 turns
#    (completes) or 21 (must transfer at the limit).
# 4. Closure. The oracle reports which rules fired, which workflow edges were
#    traversed and why it transferred. Cases are added until every journey-
#    reachable rule, edge, trigger and boundary is exercised; anything reachable
#    only through tool faults is left to the resilience probe and listed.
#
# The result is close to the smallest case set that still exercises every
# interaction of two content dimensions and every policy element.
# =====================================================================================================================

from ..s02_graph.flow import evaluate
from .journey import DIMENSIONS, JourneySpec, build_fixture, is_valid, relevant_dims
from ..s01_packs.project import NA
from .journey import primary_category as _category
from .meaning import SemanticFrame as F
from ..s01_packs.project import CaseView, project
from ..s01_packs.names import N

def _activate(proj=None) -> None:
    """Read the generator's settings from a project (default: the active pack's); run again on a pack switch."""
    global _P, PADDING_POOL, FAULT_ONLY, BOUNDARIES, _QUESTIONS, _TURNS
    _P = proj or project()
    PADDING_POOL = _P.padding_questions()                # questions that pad a call to a turn target
    FAULT_ONLY = set(_P.generator.get("fault_only_rules") or [])   # rules only tool faults reach
    BOUNDARIES = _P.boundaries()                         # levels marked boundary: true
    _QUESTIONS = _P.factor_with_role("questions")        # the factor whose level asks a policy question
    _TURNS = _P.factor_with_role("turn_target")          # the factor whose level pads to a turn target


_activate()


@dataclass
class CoverageReport:
    """What a journey suite covers: level pairs (required, covered, infeasible), rules, workflow edges,
    handoff triggers and boundaries, covered and missing.
    """
    pairs_required: int = 0
    pairs_covered: int = 0
    pairs_infeasible: int = 0
    triples_required: int = 0
    triples_covered: int = 0
    rules_covered: list[str] = field(default_factory=list)
    rules_missing: list[str] = field(default_factory=list)
    edges_covered: list[str] = field(default_factory=list)
    edges_missing: list[str] = field(default_factory=list)
    triggers_covered: list[str] = field(default_factory=list)
    triggers_missing: list[str] = field(default_factory=list)
    boundaries_covered: list[str] = field(default_factory=list)
    n_covering: int = 0
    n_closure: int = 0


def _pairs(rel: dict[str, Any]) -> set[tuple]:
    items = [(d, v) for d, v in rel.items() if v != NA]
    return {tuple(sorted(p)) for p in itertools.combinations(items, 2)}


class JourneyBaseGenerator:
    """Builds the journey bases: a covering array over the factors (AETG-style greedy: each new case is the
    candidate that covers most still-uncovered feasible level pairs), then a closure that adds cases until every
    rule, edge, trigger and boundary is exercised; ``finalize`` adds policy questions and turn padding.
    """
    def __init__(self, oracle, kb_pool: list[F] | None = None, seed: int = 42,
                 max_turn_limit_cases: int | None = None):
        """max_turn_limit_cases: None = turn_pressure is an ordinary covering dimension (every level paired
        with every other dimension: many 20+ turn conversations). An int keeps the covering array at normal
        length and lets the closure add turn-limit conversations only to reach the at_limit / over_limit
        boundaries (and any rule only they fire) - at most this many in total."""
        self.max_turn_limit_cases = max_turn_limit_cases
        self.oracle = oracle
        self.rng = random.Random(seed)
        self.kb_pool = kb_pool or []
        self._kb_i = 0
        self._long_oracle = copy.copy(oracle)
        self._long_oracle.limits = copy.copy(oracle.limits)
        self._long_oracle.limits.max_turns = 10_000

    # -------------------------------------------------------------- specs
    def random_spec(self, fixed: dict[str, Any] | None = None) -> JourneySpec:
        for _ in range(200):
            d = {k: self.rng.choice(v) for k, v in DIMENSIONS.items()}
            d.update(fixed or {})
            spec = JourneySpec(**d)
            if is_valid(spec):
                return spec
        return JourneySpec(**{**JourneySpec().dims(), **(fixed or {})})

    def finalize(self, spec: JourneySpec) -> JourneySpec | None:
        """Attach KB questions and turn padding; return None if the turn target is infeasible."""
        rel = relevant_dims(spec)
        default = JourneySpec()
        spec = spec.with_(**{d: getattr(default, d) for d in DIMENSIONS if rel[d] == NA})   # canonical form
        kb: list[F] = []
        if _QUESTIONS is not None:
            lv = _QUESTIONS.levels.get(rel.get(_QUESTIONS.name))
            for _ in range(lv.questions if lv is not None and self.kb_pool else 0):
                kb.append(self.kb_pool[self._kb_i % len(self.kb_pool)])
                self._kb_i += 1
        spec = spec.with_(kb_frames=tuple(kb), extra_kb=0)
        if _TURNS is None:
            return spec
        tp = rel.get(_TURNS.name, NA)
        level = _TURNS.levels.get(tp)
        if level is None or not level.turn_target:
            if tp == NA and getattr(spec, _TURNS.name) != _TURNS.default:
                spec = spec.with_(**{_TURNS.name: _TURNS.default})
            return spec
        natural = self._long_oracle.run(spec).turns
        target = int(evaluate(level.turn_target, CaseView(_P, spec.dims(), limits=self.oracle.limits)))
        extra = target - natural
        if extra < 0:
            return None
        # a level that must end in a transfer takes the smallest padding at or past the target that triggers it: a
        # limit counted "before the journey is complete" lets a last wrap-up turn through
        for more in range(4 if level.transfer else 1):
            pad = tuple(PADDING_POOL[i % len(PADDING_POOL)] for i in range(extra + more))
            cand = spec.with_(kb_frames=tuple(kb) + pad, extra_kb=extra + more)
            tr = self.oracle.run(cand)
            if not level.transfer or tr.transfer_reason == level.transfer:
                break
        spec = cand
        if level.exact and (tr.turns != target or (level.avoid_transfer and tr.transfer_reason == level.avoid_transfer)):
            return None
        if level.transfer and tr.transfer_reason != level.transfer:
            return None
        return spec

    # -------------------------------------------------------------- feasibility
    def feasible_pairs(self, tries: int = 150) -> tuple[set[tuple], set[tuple]]:
        allp = set()
        dims = list(DIMENSIONS)
        for a, b in itertools.combinations(dims, 2):
            for va in DIMENSIONS[a]:
                for vb in DIMENSIONS[b]:
                    allp.add(tuple(sorted([(a, va), (b, vb)])))
        feasible = set()
        for p in sorted(allp, key=str):
            fixed = dict(p)
            for _ in range(tries):
                spec = self.random_spec(fixed)
                if all(getattr(spec, d) == v for d, v in p) and is_valid(spec):
                    if p in _pairs(relevant_dims(spec)):
                        feasible.add(p)
                        break
        return feasible, allp - feasible

    # -------------------------------------------------------------- covering
    def covering(self, candidates: int = 40, max_rows: int = 200) -> tuple[list[JourneySpec], CoverageReport]:
        if self.max_turn_limit_cases is not None and _TURNS is not None:   # pairs over normal-length calls only
            saved = DIMENSIONS[_TURNS.name]
            DIMENSIONS[_TURNS.name] = [_TURNS.default]
            try:
                return self._covering(candidates, max_rows)
            finally:
                DIMENSIONS[_TURNS.name] = saved
        return self._covering(candidates, max_rows)

    def _covering(self, candidates: int = 40, max_rows: int = 200) -> tuple[list[JourneySpec], CoverageReport]:
        feasible, infeasible = self.feasible_pairs()
        uncovered = set(feasible)
        rows: list[JourneySpec] = []
        stall = 0
        while uncovered and len(rows) < max_rows and stall < 25:
            best, best_gain = None, 0
            for _ in range(candidates):
                seed_pair = self.rng.choice(sorted(uncovered, key=str))
                spec = self.random_spec(dict(seed_pair))
                gain = len(_pairs(relevant_dims(spec)) & uncovered)
                dims = list(DIMENSIONS)
                self.rng.shuffle(dims)
                for d in dims:                     # hill-climb one dimension at a time
                    if d in dict(seed_pair):
                        continue
                    for v in DIMENSIONS[d]:
                        cand = spec.with_(**{d: v})
                        if not is_valid(cand):
                            continue
                        g = len(_pairs(relevant_dims(cand)) & uncovered)
                        if g > gain:
                            spec, gain = cand, g
                if gain > best_gain:
                    best, best_gain = spec, gain
            if best is None:
                stall += 1
                continue
            fin = self.finalize(best)
            if fin is None:                        # turn target infeasible: fall back to normal
                fin = self.finalize(best.with_(**{_TURNS.name: _TURNS.default})) if _TURNS else None
            if fin is None:
                stall += 1
                continue
            got = _pairs(relevant_dims(fin)) & uncovered
            if not got:
                stall += 1
                continue
            rows.append(fin)
            uncovered -= got
            stall = 0
        rep = CoverageReport(pairs_required=len(feasible), pairs_covered=len(feasible) - len(uncovered),
                             pairs_infeasible=len(infeasible), n_covering=len(rows))
        return rows, rep

    # -------------------------------------------------------------- closure
    def closure(self, rows: list[JourneySpec], store, rep: CoverageReport,
                extra_traces: list = (), budget: int = 4000) -> list[JourneySpec]:
        all_rules = set(store.rules) - FAULT_ONLY
        all_edges = {w.wid for w in store.workflow}
        all_triggers = {t for t, ty in store.entity_types.items() if ty == "escalation_trigger"}

        def goals(spec_list, traces=()):
            r, e, t, b = set(), set(), set(), set()
            for sp in spec_list:
                tr = self.oracle.run(sp)
                r |= tr.rules_fired
                e |= tr.edges(store.workflow)[0]
                if tr.transfer_reason:
                    t.add(tr.transfer_reason)
                rel = relevant_dims(sp)
                b |= {f"{k}={v}" for k, v in rel.items() if f"{k}={v}" in BOUNDARIES}
            for tr in traces:
                r |= tr.rules_fired
                e |= tr.edges(store.workflow)[0]
            return r, e, t, b

        r, e, t, b = goals(rows, extra_traces)
        added: list[JourneySpec] = []
        tries = 0
        cap = self.max_turn_limit_cases
        padded = lambda sp: _TURNS is not None and bool(getattr(_TURNS.levels.get(getattr(sp, _TURNS.name)),  # noqa: E731
                                                                 "turn_target", None))
        n_padded = sum(padded(s) for s in rows)
        while ((all_rules - r) or (all_edges - e) or (all_triggers - t) or (BOUNDARIES - b)) and tries < budget:
            tries += 1
            fixed = {_TURNS.name: _TURNS.default} if cap is not None and n_padded >= cap and _TURNS else None
            spec = self.finalize(self.random_spec(fixed))
            if spec is None:
                continue
            r2, e2, t2, b2 = goals([spec])
            if cap is not None and padded(spec) and not ({x for x in b2 - b if x.startswith(f"{_TURNS.name}=")}
                                                        or ((all_rules - r) & r2)):
                continue                       # a long conversation only for a boundary or rule nothing else reaches
            n_padded += int(padded(spec))
            if (r2 - r) or (e2 - e) or (t2 - t) or (b2 - b):
                added.append(spec)
                r, e, t, b = r | r2, e | e2, t | t2, b | b2
        rep.rules_covered, rep.rules_missing = sorted(r & all_rules), sorted(all_rules - r)
        rep.edges_covered, rep.edges_missing = sorted(e & all_edges), sorted(all_edges - e)
        rep.triggers_covered, rep.triggers_missing = sorted(t & all_triggers), sorted(all_triggers - t)
        rep.boundaries_covered = sorted(b)
        rep.n_closure = len(added)
        return added

# --------------------------------------------------------------------------
# Base items
# --------------------------------------------------------------------------
def primary_category(spec: JourneySpec) -> str:
    """Map a journey to the base category its distinguishing content exercises (project ``categories``)."""
    return _category(spec)


def to_item(spec: JourneySpec, oracle, qid: str, category: str | None = None, kb_item=None) -> QAItem:
    """A journey spec as a base case: its fixture, the oracle's expected answer and components, and the first
    customer turn's canonical wording.
    """
    fx = build_fixture(spec)
    tr = oracle.run(spec, fx)
    comps = {f"expected_{k}": v for k, v in tr.components().items()}
    from ..s05_rewording.wording import Realizer
    R = Realizer()
    first = tr.user_frames[0] if tr.user_frames else F.make(N.act.UNCLEAR)
    return QAItem(
        qid=qid, query=R.realize(first, {}, f"{qid}|0|base"), answer=tr.signature(),
        answer_type="decision", base=category or primary_category(spec), components=comps,
        metadata={"spec": spec, "fixture": fx, "relevant": relevant_dims(spec),
                  "oracle_turns": tr.turns, "oracle_tool_calls": len(tr.tool_log),
                  "oracle_tools": sorted({c["tool"] for c in tr.tool_log}),
                  "rules_fired": sorted(tr.rules_fired),
                  "edges": sorted(tr.edges(oracle.store.workflow)[0]),
                  "transfer_reason": tr.transfer_reason,
                  "kb": (kb_item.category if kb_item else None), "kb_qid": (kb_item.qid if kb_item else None)},
    )


from ..s01_packs.loading import on_switch as _on_switch  # noqa: E402

_on_switch(_activate)
