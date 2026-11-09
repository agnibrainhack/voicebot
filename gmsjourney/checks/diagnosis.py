"""Diagnosis: why the bot fails, and whether the test suite can be trusted.

Parts of this file:
  1. Statistics (numpy / scipy only): logistic fits, likelihood-ratio deviance tests, Benjamini-Hochberg,
     Wilson intervals, AUC, Hosmer-Lemeshow.
  2. Attribution: which factor levels raise the failure rate (per factor, joint model, interactions, per
     component), and the weak link.
  3. Mutation analysis: plant one defect at a time in the reference bot; a suite that cannot catch them cannot be
     trusted.
  4. Resilience: inject tool faults; does the bot fail loud (escalate) where the oracle would?
  5. Bot validation (``bot_validation.html``): did the right thing / said it right, split by enrichment type,
     category and factor, with response cards.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats


# =====================================================================================================================
# PART 1 - Statistics for failure attribution (book Ch. 10), dependency-light (numpy/scipy).
#
# * ``logit_fit``        binomial GLM with logit link by IRLS (a tiny ridge keeps
#                        separated designs finite; it is reported, not hidden).
# * ``deviance_test``    per-factor likelihood-ratio test: G^2 = D_null - D_full on
#                        L-1 df, McFadden pseudo-R^2.
# * ``bh_adjust``        Benjamini-Hochberg false-discovery-rate correction.
# * ``wilson``           Wilson score interval for a level's failure rate.
# * ``odds_ratios``      per-level odds ratios vs the reference level with Wald CI
#                        (Haldane-Anscombe +0.5 when a cell is empty).
# * ``joint_model``      all factors at once: pseudo-R^2, AIC, BIC, ROC AUC and the
#                        Hosmer-Lemeshow calibration p-value.
# * ``interaction_test`` additive vs product-term model for a pair of factors.
# =====================================================================================================================

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
    """Logistic regression by Newton-Raphson (IRLS) with a small ridge on the slopes (not the intercept).

    Returns the coefficients, fitted probabilities ``p`` and the log-likelihood ``ll``.
    """
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
    """Benjamini-Hochberg adjusted p-values (controls the false discovery rate over many factor tests)."""
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
    """Wilson score interval for k successes out of n (better than the plain +-1.96*se near 0 and 1)."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def deviance_test(rows: Sequence[dict], factor: str, metric: str) -> dict[str, Any]:
    """Does one factor change the metric? Likelihood-ratio test of a logit with the factor against the null.

    G2 = 2 (ll_full - ll_null) on (levels - 1) degrees of freedom; also McFadden's pseudo-R2.
    """
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
    """Area under the ROC curve from ranks (Mann-Whitney U / (n_pos * n_neg))."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> float:
    """Hosmer-Lemeshow goodness-of-fit p-value: observed vs predicted counts in deciles of predicted risk."""
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
    """One logit with every factor at once: pseudo-R2, AIC, BIC, ROC AUC and the Hosmer-Lemeshow p-value."""
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
    """Do two factors interact? Likelihood-ratio test of the model with their product terms against without."""
    y = np.array([float(r[metric]) for r in rows])
    Xa, _, _ = _design(rows, [a, b])
    Xi, _, _ = _design(rows, [a, b], [(a, b)])
    fa, fi = logit_fit(Xa, y), logit_fit(Xi, y)
    df = Xi.shape[1] - Xa.shape[1]
    G2 = max(0.0, 2 * (fi["ll"] - fa["ll"]))
    return {"pair": (a, b), "deviance": G2, "df": df, "p_value": float(stats.chi2.sf(G2, df)) if df else 1.0}


# =====================================================================================================================
# PART 2 - Which inputs break which part of the bot (book Ch. 10, 12.6).
#
# ``attribute``   per-factor analysis of deviance, Benjamini-Hochberg corrected;
# ``diagnose``    the full diagnosis: factor attribution on the outcome, per-level
#                 failure rates with Wilson intervals and odds ratios, the joint
#                 model (pseudo-R^2, AIC, BIC, AUC, Hosmer-Lemeshow), interactions
#                 among significant factors, per-category accuracy, per-component
#                 attribution (each component's own correctness as the metric) and
#                 the weak-link decomposition.
# When failures are few, logistic fits are unstable; the empirical per-level rates
# are then the robust report and the result says so (Ch. 12.6).
# =====================================================================================================================

from .scoring import weak_link

MIN_FAILURES_FOR_LOGIT = 10


def _bare(rows: list[dict], factors: list[str]) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        for f in factors:
            d[f] = r.get(f"f_{f}")
        out.append(d)
    return out


def attribute(rows: list[dict], factors: list[str], metric: str = "correct", alpha: float = 0.05) -> list[dict]:
    """Rank the factors by how much they change ``metric``: a deviance test per factor, BH-adjusted, with a
    per-level table (failure rate, Wilson interval, odds ratio against the reference level).
    """
    sub = [r for r in _bare(rows, factors) if metric in r and r[metric] is not None]
    if not sub:
        return []
    tests = [deviance_test(sub, f, metric) for f in factors if len({str(r[f]) for r in sub}) > 1]
    adj = bh_adjust([t["p_value"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_value_adj"] = a
        t["significant_adj"] = a < alpha
        t["levels_table"] = level_table(sub, t["factor"], metric)
    return sorted(tests, key=lambda t: t["p_value"])


def diagnose(rows: list[dict], factors: list[str], alpha: float = 0.05) -> dict[str, Any]:
    """The full failure attribution of a run set: per-factor tests, the joint model, interactions among the
    significant factors, the worst single levels, and per-category / per-component breakdowns.
    """
    n_fail = sum(1 - r["correct"] for r in rows)
    out: dict[str, Any] = {"n": len(rows), "failures": n_fail,
                           "logit_reliable": n_fail >= MIN_FAILURES_FOR_LOGIT}
    table = attribute(rows, factors, "correct", alpha)
    out["outcome_attribution"] = table
    sig = [t["factor"] for t in table if t["significant_adj"]]
    bare = _bare(rows, factors)
    out["joint_model"] = joint_model(bare, factors, "correct") if n_fail else None
    out["interactions"] = [interaction_test(bare, a, b, "correct") for a, b in combinations(sig, 2)]
    # worst levels: the single most damaging conditions, named outright
    worst = []
    for t in table:
        for lv in t["levels_table"]:
            worst.append({"factor": t["factor"], "level": lv["level"], "failure_rate": lv["failure_rate"],
                          "wilson": (lv["wilson_lo"], lv["wilson_hi"]), "n": lv["n"],
                          "significant_factor": t["significant_adj"]})
    out["worst_levels"] = sorted(worst, key=lambda w: -w["failure_rate"])[:10]
    # per category (tags) accuracy
    cats: dict[str, list[int]] = {}
    for r in rows:
        for c in r.get("tags", [r["base"]]):
            cats.setdefault(c, []).append(r["correct"])
    out["per_category"] = {c: {"n": len(v), "accuracy": sum(v) / len(v),
                               "wilson_failure": wilson(len(v) - sum(v), len(v))}
                           for c, v in sorted(cats.items())}
    # per component: attribute each component's own correctness (Ch. 12.6)
    comps = sorted({k[2:] for r in rows for k in r if k.startswith("c_")})
    per_comp = {}
    for c in comps:
        sub = [r for r in rows if f"c_{c}" in r]
        fails = sum(1 - r[f"c_{c}"] for r in sub)
        entry = {"graded": len(sub), "failures": fails, "accuracy": 1 - fails / max(1, len(sub))}
        if fails:
            att = attribute(sub, factors, f"c_{c}", alpha)
            entry["significant"] = [(t["factor"], round(t["p_value_adj"], 4)) for t in att if t["significant_adj"]]
            entry["top_factor"] = (att[0]["factor"], round(att[0]["p_value_adj"], 4)) if att else None
            entry["logit_reliable"] = fails >= MIN_FAILURES_FOR_LOGIT
        per_comp[c] = entry
    out["per_component"] = per_comp
    out["weak_link"] = weak_link(rows)
    adh = [r for r in rows if "workflow_adherent" in r]
    if adh and any(r["workflow_adherent"] == 0 for r in adh):
        out["trajectory_attribution"] = [
            {k: v for k, v in t.items() if k != "levels_table"}
            for t in attribute(adh, factors, "workflow_adherent", alpha)]
    return out


# =====================================================================================================================
# PART 3 - Mutation testing: does the suite catch real bugs?
#
# A suite "truly tests the bot in all aspects" only if seeded defects make it
# fail. Each mutant in ``reference_bot.MUTATIONS`` is a realistic bug. A mutant
# is killed by a scenario when the reference bot passes that scenario (outcome,
# trajectory and claims all correct) and the mutant does not, or, for fault
# probes, when the mutant's outcome under an injected fault diverges from the
# reference bot's.
#
# Comparing suites at the same size isolates what the design buys: the covering-
# array + closure suite against a random suite of equal size, with and without
# enrichment, with and without fault probes.
# =====================================================================================================================

from ..bots.harness import JourneyRunner, make_reference_factory
from ..bots.reference import MUTATIONS
from ..bots.tools import FaultProfile
from .scoring import run


def _ok(r: dict) -> bool:
    return bool(r["correct"] and r["workflow_adherent"] and r.get("c_summary", 1) and r.get("c_claims", 1)
                and r.get("c_said", 1))


def default_fault_plan() -> tuple:
    """The tool faults the mutants are also run under: for a pack with per-item records, its writes erroring and
    returning plausible wrong results, its reads erroring; otherwise every tool but the handoff, both ways."""
    from ..packs import names as roles
    if roles.RECORD and roles.ACTION:
        return ((roles.RECORD, "error"), (roles.RECORD, "plausible_wrong"), (roles.OPTIONS, "error"),
                (roles.ACTION, "error"), (roles.ACTION, "plausible_wrong"), (roles.ITEMS, "error"))
    from ..packs.names import graph_names
    D = graph_names()
    return tuple((t, k) for t in D.tools if t != D.transfer_tool for k in ("error", "plausible_wrong"))


def mutation_analysis(store, runner_kwargs: dict, suites: dict[str, list], limits, workflow, validator,
                      fault_scenarios: list | None = None,
                      fault_plan: tuple | None = None,
                      mutants: dict[str, str] = MUTATIONS, governance=None,
                      runner_cls=JourneyRunner, factory=make_reference_factory) -> dict[str, Any]:
    """governance: a GovernanceConfig; when given, every run is also watched by the runtime
    monitor and a mutant counts as *flagged at runtime* when the monitor raises an alarm
    (deny/escalate) on a scenario where it raised none of that kind for the reference bot -
    detection without an oracle, as in production."""
    if fault_plan is None:
        fault_plan = default_fault_plan()
    runner_kwargs = dict(runner_kwargs)
    if governance is not None:
        runner_kwargs["governance"] = governance
    ref = runner_cls(bot_factory=factory(store), **runner_kwargs)
    ref_rows = {name: {r["sid"]: r for r in run(scns, ref, limits, workflow, validator)} for name, scns in suites.items()}
    ref_fault = {}
    for tool, kind in (fault_plan if fault_scenarios else ()):
        prof = [FaultProfile(tool, kind)]
        for scn in fault_scenarios:
            ref_fault[(tool, kind, scn.sid)] = ref.run(scn, faults=prof)

    results: dict[str, Any] = {}
    for m, desc in mutants.items():
        runner = runner_cls(bot_factory=factory(store, frozenset({m})), **runner_kwargs)
        entry: dict[str, Any] = {"description": desc, "killed_by": {}}
        for name, scns in suites.items():
            rows = {r["sid"]: r for r in run(scns, runner, limits, workflow, validator)}
            killers = [sid for sid, r in rows.items() if sid in ref_rows[name] and _ok(ref_rows[name][sid]) and not _ok(r)]
            if governance is not None:
                new = sorted({g for sid, r in rows.items() if sid in ref_rows[name]
                              for g in set(r.get("gate_rules", [])) - set(ref_rows[name][sid].get("gate_rules", []))})
                fl = entry.setdefault("runtime_monitor", {})
                fl[name] = new
                entry["flagged_at_runtime"] = entry.get("flagged_at_runtime", False) or bool(new)
            ex = None
            if killers:
                r = rows[killers[0]]
                ex = {"sid": killers[0], "category": r["base"], "violations": r["violations"],
                      "expected_transfer": r["expected_transfer_reason"], "got_transfer": r["transfer_reason"]}
            entry["killed_by"][name] = {"killed": bool(killers), "n_killing_scenarios": len(killers),
                                        "example": ex}
        if fault_scenarios:
            fk = []
            for tool, kind in fault_plan:
                prof = [FaultProfile(tool, kind)]
                for scn in fault_scenarios:
                    base = ref_fault[(tool, kind, scn.sid)]
                    if not base.extras["injected"]:
                        continue
                    res = runner.run(scn, faults=prof)
                    bt, mt = base.extras["trace"], res.extras["trace"]
                    if res.answer != base.answer or (bt.summary_consistent() and not mt.summary_consistent()):
                        fk.append({"tool": tool, "kind": kind, "sid": scn.sid})
                    if governance is not None:
                        g0 = set((base.extras.get("governance") or {}).get("by_gate_rule", {}))
                        g1 = set((res.extras.get("governance") or {}).get("by_gate_rule", {}))
                        if g1 - g0:
                            fl = entry.setdefault("runtime_monitor", {})
                            fl.setdefault("fault_probes", [])
                            fl["fault_probes"] = sorted(set(fl["fault_probes"]) | (g1 - g0))
                            entry["flagged_at_runtime"] = True
            entry["killed_by"]["fault_probes"] = {"killed": bool(fk), "n_killing_scenarios": len(fk),
                                                  "example": fk[0] if fk else None}
        entry["killed"] = any(v["killed"] for v in entry["killed_by"].values())
        results[m] = entry
    names = list(suites) + (["fault_probes"] if fault_scenarios else [])
    scores = {n: sum(results[m]["killed_by"].get(n, {}).get("killed", False) for m in results) / len(results)
              for n in names}
    scores["all_suites_combined"] = sum(results[m]["killed"] for m in results) / len(results)
    out = {"mutants": results, "mutation_score": scores, "n_mutants": len(results)}
    if governance is not None:
        out["runtime_monitor_catch_rate"] = sum(bool(results[m].get("flagged_at_runtime")) for m in results) / len(results)
    return out


# =====================================================================================================================
# PART 4 - Resilience under tool faults: does the bot fail loud? (book Ch. 11, 12.8)
#
# For each tool and fault kind, a ToolGateway injects the fault on every call to
# that tool while the other tools behave normally. Over the runs that actually
# reached the faulted tool:
#
#   expected_loud  runs where the oracle, under the same fault, escalates;
#   detected       of those, runs where the bot also escalated and made no claim that
#                  rests on the faulted result;
#   silent         expected_loud - detected: the bot carried on as if nothing happened;
#   undetectable   reached - expected_loud: faults no bot can see at its boundary (a
#                  forced AUTHENTICATED carries no echo to verify); the oracle proceeds too;
#   agrees         the bot's outcome equals the oracle's outcome under the same fault.
#
# Outright errors and timeouts are the easy case. Stale and plausible-but-wrong
# results are well formed, so only an echo/verifier check catches them (R33).
# =====================================================================================================================

from ..bots.tools import ToolGateway
from ..packs import names as roles

def fault_tools() -> list[str]:
    """Every tool of the pack but the handoff, in graph order (the auth tool and the account tools)."""
    from ..packs.names import graph_names
    D = graph_names()
    return [t for t in D.tools if t != D.transfer_tool]
KINDS = ["error", "timeout", "stale", "plausible_wrong"]


def probe(runner, oracle, scenarios, tools=None, kinds=KINDS, show: bool = True) -> list[dict[str, Any]]:
    """Resilience probe: inject each fault kind into each tool and check that the bot fails loud (escalates or
    transfers) where the oracle, under the same fault, says it should.
    """
    tools = list(tools) if tools is not None else fault_tools()
    from .scoring import progress_bar
    out = []
    bar = progress_bar(len(tools) * len(kinds) * len(scenarios), "SUT · resilience (tool faults)", show)
    for tool in tools:
        for kind in kinds:
            prof = [FaultProfile(tool_pattern=tool, kind=kind, error_message=f"{tool} unavailable (injected)")]
            reached = detected = agrees = expected_loud = 0
            examples = []
            for scn in scenarios:
                if bar is not None:
                    bar.update(1)
                res = runner.run(scn, faults=prof)
                if not any(i["tool"] == tool for i in res.extras["injected"]):
                    continue
                reached += 1
                tr = res.extras["trace"]
                exp = oracle.run(scn.base.metadata["spec"], scn.base.metadata["fixture"],
                                 gateway_factory=lambda env: ToolGateway(env, list(prof)))
                ok = res.answer == exp.signature()
                agrees += int(ok)
                if not exp.transferred:
                    continue
                expected_loud += 1
                loud = tr.transferred and tr.summary_consistent()
                detected += int(loud)
                if not loud and len(examples) < 2:
                    examples.append({"sid": scn.sid, "status": res.status, "claims": tr.claims_recorded,
                                     "expected": exp.signature(), "got": res.answer})
            if reached:
                out.append({"tool": tool, "kind": kind, "reached": reached, "expected_loud": expected_loud,
                            "detected": detected, "silent": expected_loud - detected,
                            "undetectable": reached - expected_loud,
                            "detection_rate": detected / expected_loud if expected_loud else None,
                            "oracle_agreement": agrees / reached, "silent_examples": examples})
    if bar is not None:
        bar.close()
    return out


# =====================================================================================================================
# PART 5 - Bot validation: what the bot did and what it said, case by case and by what each test case contained.
#
# Every run is validated twice:
#
#   did the right thing   the semantic check: the outcome (end state, transfer reason, what was truly recorded,
#                         which card action truly completed; for a policy question, the answer), graded against
#                         the oracle's answer key (evaluation/runner.score -> ``outcome_correct``)
#   said it right         the response check: every claim the bot's words make, checked where its truth lives
#                         (gmsjourney/speech -> ``c_speech``): the tools' results, the writes that truly succeeded,
#                         the customer's own answers, and for policy figures and stances the GMS
#
# A run is correct when both hold (the default ``--speech-gate fail``).
#
# This module turns the campaign's graded rows into one page, ``bot_validation.html``:
#
#   1. the two checks, per type of enrichment (base case / template / LLM)
#   2. performance by what the test case contained: type of enrichment, base-case category, the DOE factors
#      (from the official catalogs) and the enrichment factors this bot adds (custom), each level with the
#      three rates side by side; factors the deviance test finds significant first
#   3. the semantic check in detail: which stage of the journey went wrong
#   4. the response check in detail: where each claim's truth came from, and the policy figures checked
#      against the GMS (exact register verdict, GMS distance vs the relation's cap, knowlytix's own reading)
#   5. bot response cards: the responses that failed, grouped by problem, each with the case's factor levels,
#      what the customer said, what the bot said, and every claim with its truth
#   6. checking the checks: the word-only negative controls and the word-only mutants
#
# Input   the campaign's report dict ``R`` (report.json) and its graded rows (dataset/case_results.jsonl; each
#         row carries ``speech_turns``, the response cards kept by evaluation/runner.response_cards)
# Output  ``bot_validation.html`` in the campaign folder, and a summary dict (no cards) for report.json
#
#     from gmsjourney.checks.diagnosis import write_bot_validation, write_bot_validation_from_dir
#     summary = write_bot_validation(out_dir, R, rows)        # inside the campaign
#     write_bot_validation_from_dir("output/campaign_out")     # again later, from the files
#
# or ``python -m gmsjourney.checks.diagnosis output/campaign_out``.
# =====================================================================================================================

from ..reports.report import CSS, esc, pct, table

GMS_BASES = ("GMS exact register", "GMS stance check")
SESSION_BASES = ("tool result", "successful write", "what the customer said")
BASIS_ORDER = SESSION_BASES + GMS_BASES + ("policy text", "nothing to check", "other")
ARM_LABEL = {"base": "none (base case)", "template": "template", "llm": "LLM"}
PROBLEM_LABEL = {
    "contradicted": "said something false",
    "unsupported": "claimed with nothing behind it",
    "text_contradicts_label": "words contradict the act label",
    "label_not_said": "act labelled but not said",
    "missing": "left out required information",
    "not_allowed_here": "out of place in the dialog",
}
FIGURE_LABEL = {"fee": "Fees", "delivery_days": "Delivery times", "limit": "Limits"}
WORD_ONLY_MUTANTS = ("says_wrong_fee", "misstates_recorded_answer", "names_wrong_amount", "summary_omits")


# ------------------------------------------------------------------ the three rates
def did_right(r: dict[str, Any]) -> bool:
    """The semantic check: the outcome (before any speech gate)."""
    return bool(r.get("outcome_correct", r.get("correct")))


def said_right(r: dict[str, Any]) -> bool:
    """The response check: every claim grounded, faithful to the labels and complete (``c_speech``)."""
    return bool(r.get("c_speech", 1))


def rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """n, and the share of runs that did the right thing, said it right, and both (correct)."""
    n = len(rows)
    if not n:
        return {"n": 0, "did_right": None, "said_right": None, "correct": None}
    return {"n": n, "did_right": sum(did_right(r) for r in rows) / n, "said_right": sum(said_right(r) for r in rows) / n,
            "correct": sum(did_right(r) and said_right(r) for r in rows) / n}


def two_by_two(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many runs did / did not do the right thing, against said / did not say it right."""
    c = Counter((did_right(r), said_right(r)) for r in rows)
    return {"both": c[(True, True)], "did_not_said": c[(True, False)], "said_not_did": c[(False, True)],
            "neither": c[(False, False)]}


# ------------------------------------------------------------------ performance by what the case contained
def factor_groups(R: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(DOE factors from the official catalogs, enrichment factors this bot adds), in design order."""
    prov = {p["factor"]: p.get("origin") for p in (R.get("design") or {}).get("level_provenance", [])}
    factors = (R.get("design") or {}).get("factors") or list(prov)
    return [f for f in factors if prov.get(f) != "custom"], [f for f in factors if prov.get(f) == "custom"]


def by_level(rows: list[dict[str, Any]], key, levels: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """The three rates for each level of one dimension. ``key`` is a row field name or a function of a row."""
    get = key if callable(key) else (lambda r: r.get(key))
    groups: dict[Any, list] = defaultdict(list)
    for r in rows:
        v = get(r)
        if v not in (None, ""):
            groups[v].append(r)
    order = [lv for lv in (levels or []) if lv in groups] + sorted(v for v in groups if v not in set(levels or []))
    return [{"level": lv, **rates(groups[lv])} for lv in order]


def factor_performance(R: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Performance by type of enrichment, base category, DOE factors and enrichment factors.

    Factor levels exist only on enriched runs, so the factor tables are over the enriched runs (every arm);
    the type of enrichment and the base category are over every run. Each factor carries its deviance test
    (p adjusted for the number of factors, from the campaign's attribution) and its spread (best minus worst
    level's correct rate)."""
    levels = (R.get("design") or {}).get("levels") or {}
    attr = {a["factor"]: a for a in ((R.get("diagnosis") or {}).get("outcome_attribution") or [])}
    enriched = [r for r in rows if (r.get("arm") or "base") != "base"]
    doe, custom = factor_groups(R)

    def one(f: str) -> dict[str, Any]:
        lv = by_level(enriched, f"f_{f}", levels.get(f))
        cs = [x["correct"] for x in lv if x["correct"] is not None and x["n"] >= 5]
        a = attr.get(f) or {}
        return {"factor": f, "baseline": (levels.get(f) or [None])[0], "levels": lv,
                "spread": (max(cs) - min(cs)) if len(cs) > 1 else 0.0,
                "p_adj": a.get("p_value_adj"), "significant": bool(a.get("significant_adj"))}

    def ordered(fs: list[str]) -> list[dict[str, Any]]:
        out = [one(f) for f in fs]
        return sorted(out, key=lambda x: (not x["significant"], x["p_adj"] if x["p_adj"] is not None else 1.0,
                                          -x["spread"]))
    return {"arm": by_level(rows, lambda r: r.get("arm") or "base", ["base", "template", "llm"]),
            "category": sorted(by_level(rows, "base"), key=lambda x: (x["correct"] or 0, -x["n"])),
            "doe": ordered(doe), "enrichment": ordered(custom)}


# ------------------------------------------------------------------ the semantic check in detail
def stage_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each journey stage, the share of runs (per type of enrichment) where the bot's decision there was
    wrong, plus 'every step allowed' (workflow admissibility)."""
    from ..bots.tools import WORKFLOW_STAGES
    present = {k for r in rows for k in r if k.startswith("c_") and k not in ("c_speech", "c_flow")}
    stages = [f"c_{s}" for s in WORKFLOW_STAGES if f"c_{s}" in present] + sorted(
        present - {f"c_{s}" for s in WORKFLOW_STAGES})
    arms = [a for a in ("base", "template", "llm") if any((r.get("arm") or "base") == a for r in rows)]
    out = []
    for st in stages:
        cell = {}
        for a in arms:
            vals = [r[st] for r in rows if (r.get("arm") or "base") == a and r.get(st) is not None]
            cell[a] = (sum(1 for v in vals if not v) / len(vals)) if vals else None
        out.append({"stage": st[2:], **cell})
    adm = {a: (lambda xs: sum(1 for r in xs if not r.get("workflow_adherent", 1)) / len(xs) if xs else None)(
        [r for r in rows if (r.get("arm") or "base") == a]) for a in arms}
    out.append({"stage": "every step allowed (workflow)", **adm})
    return out


# ------------------------------------------------------------------ the response check in detail
def claims_of(rows: list[dict[str, Any]]) -> Iterable[tuple[dict, dict, dict]]:
    """(row, response card, claim) for every claim in every run."""
    for r in rows:
        for card in r.get("speech_turns") or []:
            for c in card.get("claims") or []:
                yield r, card, c


def basis_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Claims by where their truth was read, with their verdicts."""
    t: dict[str, Counter] = defaultdict(Counter)
    for _, _, c in claims_of(rows):
        t[c.get("basis") or "other"][c["status"]] += 1
    return [{"basis": b, "claims": sum(t[b].values()), **{s: t[b][s] for s in
             ("supported", "contradicted", "unsupported", "unverifiable")}} for b in BASIS_ORDER if t.get(b)]


def gms_figures(rows: list[dict[str, Any]], examples: int = 8) -> dict[str, Any]:
    """The policy figures and stances checked against the GMS.

    Per kind (fees, delivery times, limits, each rule stance): how many claims, the exact register's verdict,
    the GMS geometry (claimed value inside / outside the relation's cap) and whether it agrees with the
    register, and knowlytix's AnswerVerifier (confirmed / not found / not asked: a sentence it cannot read on
    its own). Examples: the figures the register found wrong, with their GMS reading."""
    kinds: dict[str, Counter] = defaultdict(Counter)
    bad: list[dict[str, Any]] = []
    seen = set()
    for r, card, c in claims_of(rows):
        if c.get("basis") not in GMS_BASES:
            continue
        k = FIGURE_LABEL.get(c["field"]) or f"Rule stance ({c.get('about')})"
        g = c.get("geometry") or {}
        st = c["status"]
        kinds[k]["claims"] += 1
        kinds[k][st] += 1
        if "inside_cap" in g:
            kinds[k]["geo"] += 1
            kinds[k]["inside" if g["inside_cap"] else "outside"] += 1
            kinds[k]["agree"] += int((st == "supported") == bool(g["inside_cap"]))
        kx = g.get("knowlytix")
        kinds[k]["kx_" + ("confirmed" if kx == "confirmed" else "not_found" if kx == "not found" else "not_asked")] += 1
        if st != "supported" and len(bad) < examples and (c["words"], c.get("about")) not in seen:
            seen.add((c["words"], c.get("about")))
            bad.append({"sid": r["sid"], "arm": r.get("arm") or "base", "text": card["text"], "words": c["words"],
                        "said": c["value"], "truth": c.get("truth"), "status": st, "why": c.get("why"),
                        "geometry": g})
    order = [v for v in FIGURE_LABEL.values() if v in kinds] + sorted(k for k in kinds if k not in FIGURE_LABEL.values())
    return {"kinds": [{"kind": k, **dict(kinds[k])} for k in order], "examples": bad}


# ------------------------------------------------------------------ bot response cards
def problems_in(card: dict[str, Any]) -> list[tuple[str, str]]:
    """(problem kind, field or act) for every failed claim and finding of one response."""
    out = [(c["status"], c["field"]) for c in card.get("claims") or [] if c["status"] in ("contradicted", "unsupported")]
    out += [(f["kind"], f.get("act") or "") for f in card.get("findings") or []]
    return out


def failing_cards(rows: list[dict[str, Any]], per_group: int = 2) -> list[dict[str, Any]]:
    """The failing responses grouped by problem (graded problems first, then the most frequent): per group the
    number of responses and of cases, and up to ``per_group`` example cards from different cases (different arms
    where possible)."""
    groups: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for r in rows:
        for card in r.get("speech_turns") or []:
            for p in set(problems_in(card)):
                groups[p].append((r, card))
    out = []
    for (kind, what), hits in sorted(groups.items(), key=lambda kv: (kv[0][0] == "not_allowed_here", -len(kv[1]))):
        queues: dict[str, list] = defaultdict(list)          # one queue per arm, so the examples span the arms
        for r, card in sorted(hits, key=lambda h: h[0]["sid"]):
            queues[r.get("arm") or "base"].append((r, card))
        picked, cases = [], set()
        while len(picked) < per_group and any(queues.values()):
            for q in queues.values():
                while q and q[0][0]["sid"] in cases:
                    q.pop(0)
                if q and len(picked) < per_group:
                    r, card = q.pop(0)
                    picked.append((r, card))
                    cases.add(r["sid"])
        out.append({"kind": kind, "what": what, "responses": len(hits), "cases": len({h[0]["sid"] for h in hits}),
                    "graded": kind != "not_allowed_here", "cards": picked})
    return out


def example_card(rows: list[dict[str, Any]]) -> tuple[dict, dict] | None:
    """A fully grounded response with a GMS-checked figure, to show how a card reads."""
    for r in rows:
        if not (did_right(r) and said_right(r)):
            continue
        for card in r.get("speech_turns") or []:
            cs = card.get("claims") or []
            if card.get("ok") and any(c.get("basis") == "GMS exact register" and (c.get("geometry") or {}).get(
                    "knowlytix") == "confirmed" for c in cs) and len(cs) >= 2:
                return r, card
    return None


# ------------------------------------------------------------------ the page
_CSS_MORE = """
.rate{display:flex;align-items:center;gap:8px;min-width:120px}.rate .b{flex:1;height:6px;border-radius:3px;
background:var(--grid);min-width:50px}.rate .b i{display:block;height:6px;border-radius:3px;background:var(--s1)}
.rate .t{font-variant-numeric:tabular-nums;min-width:46px;text-align:right}
.chip{display:inline-block;padding:1px 8px;margin:2px 4px 2px 0;border-radius:999px;border:1px solid var(--border);
font-size:12px;color:var(--ink2);background:var(--page)}.chip.k{color:var(--ink)}
.rc{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin:10px 0}
.rc .h{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline;margin-bottom:6px}
.say{margin:6px 0;padding:4px 10px;border-left:3px solid var(--grid)}.say.b{border-color:var(--s1)}
.say .who{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.st-supported{color:var(--goodtext);font-weight:600}.st-contradicted,.st-unsupported{color:var(--critical);font-weight:600}
.st-unverifiable{color:var(--muted)}.badge{font-size:12px;padding:1px 8px;border-radius:999px;border:1px solid var(--border)}
.badge.sig{color:var(--s1);border-color:var(--s1);font-weight:600}.chip.sig{border-color:var(--s1);color:var(--ink);
font-weight:600}.rates3{display:flex;flex-wrap:wrap;gap:4px 18px;margin:2px 0 10px}.rates3 .v{font-size:22px;
font-weight:600;font-variant-numeric:tabular-nums}.rates3 .l{font-size:12px;color:var(--ink2)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.mx td,.mx th{text-align:center}.mx td:first-child,.mx th:first-child{text-align:left}
"""
_MARK = {"supported": "✓ supported", "contradicted": "✗ contradicted", "unsupported": "✗ unsupported",
         "unverifiable": "– unverifiable"}


def _rate(x: float | None) -> str:
    if x is None:
        return "<span class='muted'>–</span>"
    return (f"<div class='rate'><div class='b'><i style='width:{100 * x:.1f}%'></i></div>"
            f"<span class='t'>{pct(x)}</span></div>")


def _rates_table(items: list[dict[str, Any]], label: str, fmt=lambda v: esc(v), baseline: str | None = None) -> str:
    head = [label, "runs", "did the right thing", "said it right", "correct (both)"]
    body = "".join(
        f"<tr><td>{fmt(x['level'])}" + (" <span class='muted'>(baseline)</span>" if x["level"] == baseline else "")
        + f"</td><td class='num'>{x['n']}</td><td>{_rate(x['did_right'])}</td><td>{_rate(x['said_right'])}</td>"
        f"<td>{_rate(x['correct'])}</td></tr>" for x in items)
    return ("<table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>" + body
            + "</tbody></table>")


def _human(s: Any) -> str:
    return esc(str(s).replace("_", " "))


def _value(v: Any) -> str:
    if isinstance(v, list):
        if len(v) == 2:
            return esc(str(v[0]) if v[0] == v[1] else f"{v[0]} to {v[1]}")
        return esc(", ".join(map(str, v)))
    return esc("–" if v is None else v)


def _gms_reading(g: dict[str, Any]) -> str:
    if not g:
        return "<span class='muted'>–</span>"
    parts = []
    if "distance" in g:
        parts.append(f"distance {g['distance']:.3f} vs cap {g['cap']:.3f} "
                     + ("<b>inside</b>" if g.get("inside_cap") else "<b>outside</b>"))
    if g.get("tension") is not None:
        parts.append(f"stance tension {g['tension']:.2f}")
    if g.get("knowlytix"):
        parts.append(f"knowlytix: {esc(g['knowlytix'])}")
    return "<br>".join(parts) or "<span class='muted'>–</span>"


def _case_chips(r: dict[str, Any], levels: dict[str, list], significant: frozenset = frozenset()) -> str:
    """The case's type of enrichment, base category, and every factor level that is not the baseline; the levels of
    factors with a significant effect on the bot are highlighted (they are the likely contributors)."""
    arm = r.get("arm") or "base"
    chips = [f"<span class='chip k'>enrichment: {esc(ARM_LABEL.get(arm, arm))}</span>",
             f"<span class='chip'>category: {_human(r.get('base'))}</span>"]
    on = [(k[2:], v) for k, v in r.items() if k.startswith("f_") and v not in (None, "")
          and v != (levels.get(k[2:]) or [None])[0]]
    on.sort(key=lambda fv: fv[0] not in significant)
    chips += [f"<span class='chip{' sig' if f in significant else ''}'>{_human(f)} = {_human(v)}</span>" for f, v in on]
    if arm != "base" and not on:
        chips.append("<span class='chip'>every factor at its baseline</span>")
    return "".join(chips)


GMS_NOTE = ("The exact register decides a figure. Two readings are reported next to it: the GMS geometry (is the "
            "claimed value's distance from the fact inside the relation's learned cap?) and knowlytix's own "
            "AnswerVerifier (does it find the fact, with that value, in the sentence? It reads one sentence without the "
            "dialog, so a sentence that does not name its subject is not given to it).")


def basis_table_html(bb: list[dict[str, Any]]) -> str:
    """Claims by where their truth was read (``basis_breakdown``), as a table."""
    return table(["truth read from", "claims", "supported", "contradicted", "unsupported", "unverifiable"],
                 [[("<b>GMS</b> · " + esc(b["basis"][4:]) if b["basis"] in GMS_BASES else
                    "session · " + esc(b["basis"]) if b["basis"] in SESSION_BASES else esc(b["basis"])),
                   b["claims"], b["supported"], b["contradicted"], b["unsupported"],
                   b["unverifiable"]] for b in bb], {1, 2, 3, 4, 5})


def gms_table_html(kinds: list[dict[str, Any]]) -> str:
    """The GMS-checked figures per kind (``gms_figures(...)["kinds"]``), as a table."""
    return table(["kind", "claims", "register: right", "register: wrong", "GMS: inside cap", "GMS: outside cap",
                  "GMS agrees with register", "knowlytix confirmed", "knowlytix not found", "not given to knowlytix"],
                 [[esc(k["kind"]), k.get("claims", 0), k.get("supported", 0),
                   k.get("contradicted", 0) + k.get("unsupported", 0), k.get("inside", 0), k.get("outside", 0),
                   f"{k.get('agree', 0)}/{k.get('geo', 0)}" if k.get("geo") else "–",
                   k.get("kx_confirmed", 0), k.get("kx_not_found", 0), k.get("kx_not_asked", 0)] for k in kinds],
                 set(range(1, 10)))


def render_card(r: dict[str, Any], card: dict[str, Any], levels: dict[str, list],
                significant: frozenset = frozenset()) -> str:
    """One bot response card: the case and its factors (significant ones first and highlighted), what the customer
    said, what the bot said (and its act labels), every checked claim with its truth and where it was read, and the
    turn's findings. Mentions with nothing to check against are counted, not listed."""
    verdict = ("outcome right" if did_right(r) else "outcome wrong") + " · " + \
        ("words grounded" if said_right(r) else "words not grounded")
    head = (f"<div class='h'><b class='mono'>{esc(r['sid'])}</b><span class='muted'>{esc(card['where'])}</span>"
            f"<span class='badge {'ok' if did_right(r) and said_right(r) else 'bad'}'>{esc(verdict)}</span></div>"
            f"<div>{_case_chips(r, levels, significant)}</div>")
    cust = (f"<div class='say'><span class='who'>Customer</span>“{esc(card['customer'])}”</div>"
            if card.get("customer") else "")
    acts = " ".join(f"<span class='chip mono'>{esc(a)}</span>" for a in card.get("acts") or [])
    bot = (f"<div class='say b'><span class='who'>Bot</span>“{esc(card['text'])}”"
           + (f"<div style='margin-top:4px'>{acts}</div>" if acts else "") + "</div>")
    rows = [[f"“{esc(c['words'])}”", f"{_human(c['field'])} = {_value(c['value'])}", _value(c.get("truth")),
             esc(c.get("basis") or ""), f"<span class='st-{c['status']}'>{_MARK.get(c['status'], c['status'])}</span>"
             + (f"<div class='muted'>{esc(c['why'])}</div>" if c.get("why") and c["status"] != "supported" else ""),
             _gms_reading(c.get("geometry") or {})] for c in card.get("claims") or [] if c["status"] != "unverifiable"]
    mentions = [c["words"] for c in card.get("claims") or [] if c["status"] == "unverifiable"]
    claims = ("<table><thead><tr><th>words</th><th>claim</th><th>truth</th><th>checked against</th><th>verdict</th>"
              "<th>GMS reading</th></tr></thead><tbody>"
              + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table>"
              if rows else "")
    finds = "".join(f"<li><b>{esc(PROBLEM_LABEL.get(f['kind'], f['kind']))}</b>"
                    + (" <span class='muted'>(reported, not graded)</span>" if f["kind"] == "not_allowed_here" else "")
                    + f": {esc(f['detail'])}" + (f" <span class='muted'>({esc(f['rule'])})</span>" if f.get("rule") else "")
                    + "</li>" for f in card.get("findings") or [])
    note = ("<p class='muted' style='margin:6px 0 0'>Also mentioned, with nothing to check against: "
            + esc(", ".join(f"“{m}”" for m in mentions)) + "</p>") if mentions else ""
    return (f"<div class='rc'>{head}{cust}{bot}<div style='overflow-x:auto'>{claims}</div>{note}"
            + (f"<ul style='margin:6px 0 0'>{finds}</ul>" if finds else "") + "</div>")


def render_bot_validation_html(R: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """The whole page (self-contained HTML; same look as report.html, light and dark)."""
    levels = (R.get("design") or {}).get("levels") or {}
    res = R.get("results") or {}
    bot = res.get("bot", "the bot under test")
    arms = [a for a in ("base", "template", "llm") if any((r.get("arm") or "base") == a for r in rows)]
    by_arm = {a: [r for r in rows if (r.get("arm") or "base") == a] for a in arms}
    perf = factor_performance(R, rows)
    significant = frozenset(f["factor"] for f in perf["doe"] + perf["enrichment"] if f["significant"])
    out: list[str] = []
    w = out.append

    # 1 -------------------------------------------------------------- the two checks
    w(f"<h1>Bot validation</h1><p class='lede'>{esc(bot)} · {len(rows)} graded runs · every run checked twice: did the "
      "bot <b>do the right thing</b> (the semantic check: the outcome against the oracle's answer key) and did it "
      "<b>say it right</b> (the response check: every claim in its words against where its truth lives, the GMS for "
      "policy figures). A run is correct when both hold.</p>")
    mx = []
    for a in arms:
        m, rt = two_by_two(by_arm[a]), rates(by_arm[a])
        three = "".join(f"<div><div class='v'>{pct(rt[k])}</div><div class='l'>{lab}</div></div>" for k, lab in (
            ("did_right", "did the right thing"), ("said_right", "said it right"), ("correct", "correct (both)")))
        mx.append(f"<div class='card'><h3 style='margin-top:0'>Enrichment: {esc(ARM_LABEL[a])} · {len(by_arm[a])} runs</h3>"
                  f"<div class='rates3'>{three}</div>"
                  "<table class='mx'><thead><tr><th></th><th>said it right</th><th>did not</th></tr></thead><tbody>"
                  f"<tr><th>did the right thing</th><td class='ok'>{m['both']}</td><td class='bad'>{m['did_not_said']}</td></tr>"
                  f"<tr><th>did not</th><td class='bad'>{m['said_not_did']}</td><td class='bad'>{m['neither']}</td></tr>"
                  "</tbody></table><p class='muted' style='margin:6px 0 0'>Top right: the outcome was right but the "
                  "customer was told something false or incomplete; only the response check sees these.</p></div>")
    w("<div class='grid2'>" + "".join(mx) + "</div>")

    # 2 -------------------------------------------------------------- by what the case contained
    w("<h2>Performance by what the test case contained</h2><p class='lede'>Each test case is a base case (what it "
      "tests) with or without enrichment (how the customer words it). The rates below split every run by one of "
      "these at a time.</p>")
    w("<h3>Type of enrichment</h3><div class='card'>"
      + _rates_table(perf["arm"], "enrichment", fmt=lambda v: esc(ARM_LABEL.get(v, v))) + "</div>")
    w("<h3>Base-case category</h3><p class='muted'>What the case tests. Weakest first.</p><div class='card'>"
      + _rates_table(perf["category"], "category", fmt=_human) + "</div>")
    for title, key, note in (
            ("DOE factors (official catalogs)", "doe", "The design-of-experiments factors from proofloop's official "
             "catalogs. Factors whose effect is significant (deviance test, adjusted for the number of factors) come "
             "first and are open; the rest are folded below."),
            ("Enrichment factors (added for this bot)", "enrichment", "The factors this voice bot's application layer "
             "adds to the catalogs.")):
        fs = perf[key]
        if not fs:
            continue
        w(f"<h3>{esc(title)}</h3><p class='muted'>{esc(note)} Enriched runs only (a base case has no factor levels).</p>")
        sig, rest = [f for f in fs if f["significant"]], [f for f in fs if not f["significant"]]
        for f in sig + (rest if key == "enrichment" else []):
            tag = (f"<span class='badge sig'>significant · p adj {f['p_adj']:.1e}</span>" if f["significant"] else
                   "<span class='badge'>not significant</span>" if f["p_adj"] is not None else "")
            w(f"<div class='card'><div class='h' style='display:flex;gap:10px;align-items:baseline;margin-bottom:6px'>"
              f"<b>{_human(f['factor'])}</b>{tag}<span class='muted'>spread {pct(f['spread'])}</span></div>"
              + _rates_table(f["levels"], "level", fmt=_human, baseline=f["baseline"]) + "</div>")
        if key == "doe" and rest:
            w(f"<details><summary>{len(rest)} DOE factors with no significant effect: "
              + esc(", ".join(x["factor"].replace("_", " ") for x in rest)) + "</summary>"
              + "".join(f"<div class='card'><b>{_human(f['factor'])}</b> <span class='muted'>spread {pct(f['spread'])}"
                        + (f" · p adj {f['p_adj']:.2f}" if f["p_adj"] is not None else "") + "</span>"
                        + _rates_table(f["levels"], "level", fmt=_human, baseline=f["baseline"]) + "</div>" for f in rest)
              + "</details>")

    # 3 -------------------------------------------------------------- semantic detail
    st = stage_failures(rows)
    w("<h2>The semantic check in detail</h2><p class='lede'>Where the bot's decisions went wrong: the share of runs "
      "in which each stage of the journey differed from the oracle's (tools called, arguments, what was recorded, "
      "which action completed, the transfer and its reason).</p><div class='card'>"
      + table(["stage"] + [f"wrong · {ARM_LABEL[a]}" for a in arms],
              [[_human(x["stage"])] + [pct(x.get(a)) for a in arms] for x in st], set(range(1, len(arms) + 1)))
      + "</div>")

    # 4 -------------------------------------------------------------- response detail
    w("<h2>The response check in detail</h2><p class='lede'>Every claim the bot's words made, by where its truth was "
      "read. The session (what the tools returned, the writes that truly succeeded, what the customer said) holds "
      "the facts of this call; the GMS holds the policy's.</p><div class='card'>" + basis_table_html(basis_breakdown(rows))
      + "</div>")
    gf = gms_figures(rows)
    if gf["kinds"]:
        w("<h3>Checked against the GMS</h3><p class='muted'>" + GMS_NOTE + "</p><div class='card'>"
          + gms_table_html(gf["kinds"]) + "</div>")
        if gf["examples"]:
            w("<h3>Figures the GMS found wrong</h3><div class='card'>" + table(
                ["case", "the bot said", "claim", "truth", "GMS reading"],
                [[f"<span class='mono'>{esc(e['sid'])}</span>", f"“{esc(e['text'])}”", f"“{esc(e['words'])}”",
                  _value(e["truth"]), _gms_reading(e["geometry"])] for e in gf["examples"]]) + "</div>")

    # 5 -------------------------------------------------------------- response cards
    w("<h2>Bot response cards</h2><p class='lede'>One card per bot response: the case and the factor levels it "
      "contained, what the customer said, what the bot said and the acts it labelled, and every claim in its words "
      "with the truth and where it was read.</p>")
    ex = example_card(rows)
    if ex:
        w("<h3>How to read a card (a response that holds up)</h3>" + render_card(ex[0], ex[1], levels, significant))
    groups = failing_cards(rows)
    if groups:
        w("<h3>The responses that failed, by problem</h3><div class='card'>" + table(
            ["problem", "about", "responses", "cases", "graded"],
            [[esc(PROBLEM_LABEL.get(g["kind"], g["kind"])), _human(g["what"]), g["responses"], g["cases"],
              "yes" if g["graded"] else "reported only"] for g in groups], {2, 3}) + "</div>")
        for g in groups:
            w(f"<details{' open' if g['graded'] else ''}><summary><b>{esc(PROBLEM_LABEL.get(g['kind'], g['kind']))}</b>: "
              f"{_human(g['what'])} · {g['responses']} responses in {g['cases']} cases</summary>"
              + "".join(render_card(r, c, levels, significant) for r, c in g["cards"]) + "</details>")
    else:
        w("<p>No response failed.</p>")

    # 6 -------------------------------------------------------------- checking the checks
    ctl = ((R.get("speech") or {}).get("controls") or {}).get("controls") or {}
    mut = (R.get("mutation") or {}).get("mutants") or {}
    if ctl or mut:
        w("<h2>Checking the checks</h2><p class='lede'>A check that never fires proves nothing. The controls edit only "
          "the words of one act in a real conversation (labels, tools and outcome unchanged); the word-only mutants "
          "are bots whose actions are right and whose words are not.</p><div class='grid2'>")
        if ctl:
            w("<div class='card'><h3 style='margin-top:0'>Negative controls</h3>" + table(
                ["control", "edits", "flagged"],
                [[_human(k), v["applied"], pct(v["detection_rate"])] for k, v in ctl.items()], {1, 2}) + "</div>")
        wm = [m for m in WORD_ONLY_MUTANTS if m in mut]
        if wm:
            w("<div class='card'><h3 style='margin-top:0'>Word-only mutants</h3>" + table(
                ["mutant", "caught", "flagged live"],
                [[_human(m), "✓" if mut[m].get("killed") else "✗", "✓" if mut[m].get("flagged_at_runtime") else "✗"]
                 for m in wm]) + "</div>")
        w("</div>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            "content='width=device-width,initial-scale=1'><title>Bot validation</title><style>" + CSS + _CSS_MORE
            + "</style></head><body><main>" + "\n".join(out) + "</main></body></html>")


# ------------------------------------------------------------------ in a notebook
def cards_page(r: dict[str, Any], levels: dict[str, list] | None = None) -> str:
    """One case's response cards as a small self-contained page (every response with a claim or a finding)."""
    body = "".join(render_card(r, c, levels or {}) for c in r.get("speech_turns") or []) or \
        "<p>No checkable response in this run.</p>"
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + CSS + _CSS_MORE + "</style></head><body>"
            f"<main style='padding:8px 4px'>{body}</main></body></html>")


def as_iframe(doc: str, height: int = 720) -> str:
    """A page wrapped in an ``<iframe srcdoc>`` so its styles stay inside the frame (for IPython's ``HTML``)."""
    import html as _html
    return f'<div><iframe srcdoc="{_html.escape(doc)}" style="width:100%;height:{height}px;border:0"></iframe></div>'


# ------------------------------------------------------------------ writing it
def validation_summary(R: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The page's numbers without the cards (for report.json and report.html)."""
    arms = sorted({r.get("arm") or "base" for r in rows})
    perf = factor_performance(R, rows)
    groups = failing_cards(rows, per_group=0)
    return {"by_arm": {a: {**rates([r for r in rows if (r.get("arm") or "base") == a]),
                           **two_by_two([r for r in rows if (r.get("arm") or "base") == a])} for a in arms},
            "by_category": perf["category"], "significant_factors": [
                {"factor": f["factor"], "p_adj": f["p_adj"], "levels": f["levels"]}
                for f in perf["doe"] + perf["enrichment"] if f["significant"]],
            "truth_basis": basis_breakdown(rows), "gms": gms_figures(rows),
            "problems": [{k: g[k] for k in ("kind", "what", "responses", "cases", "graded")} for g in groups]}


def write_bot_validation(out_dir: str | Path, R: dict[str, Any], rows: list[dict[str, Any]],
                         name: str = "bot_validation.html") -> dict[str, Any]:
    """Write the page into the campaign folder and return its summary."""
    Path(out_dir, name).write_text(render_bot_validation_html(R, rows), encoding="utf-8")
    return validation_summary(R, rows)


def write_bot_validation_from_dir(out_dir: str | Path) -> Path:
    """Rebuild the page from a finished campaign's files (report.json, dataset/case_results.jsonl)."""
    out = Path(out_dir)
    R = json.loads((out / "report.json").read_text())
    rows = [json.loads(line) for line in (out / "dataset" / "case_results.jsonl").read_text().splitlines() if line.strip()]
    write_bot_validation(out, R, rows)
    return out / "bot_validation.html"


if __name__ == "__main__":
    print(write_bot_validation_from_dir(sys.argv[1] if len(sys.argv) > 1 else "output/campaign_out"))
