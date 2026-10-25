"""Calibrating the runtime gates (book Appendix C; beyond-ship-and-pray appendix_C notebook).

* ``workflow_cohort``  labelled transitions: every workflow edge is ``admit``, every
  other (source, target) pair over the workflow's nodes is ``block``. The cohort
  comes from the policy graph, not from the agent under test (the appendix's first
  anti-pattern is calibrating on the agent's own traces).
* ``sweep``            false admits / false rejects / accuracy per candidate theta,
  on the book's grid and on every midpoint between observed scores.
* ``operating_points`` the regulated point - zero false admits, fewest false
  rejects (the appendix: in a regulated setting a false admit dominates) - and the
  maximum-accuracy point the appendix's code selects.
* ``plan_cohort`` / ``calibrate_plan`` the same for the whole-plan drift budget, with
  admissible plans taken from the oracle's journeys and inadmissible ones made by
  skipping, swapping or prepending a step.
* ``multi_threshold_design`` a balanced design (proofloop's ``balanced_design``) over
  the four thresholds that matter, each row scored on its own labelled cohort.
* ``recalibration_triggers`` store retrained, score drift (PSI), false-reject creep,
  policy changed.

Comparability with gmsh (GMSJudge / plausibility_calibration):

* ``logistic_p50``     gmsh's operating point: a one-feature L2 logistic fit of P(admit | score)
  and the score where P = 0.5, reported next to the regulated point;
* ``holdout_eval``     stratified k-fold: every operating-point rule is fitted on the other
  folds and scored on the held-out one (gmsh reports training-set accuracy only);
* ``bootstrap_thetas`` 200 stratified resamples of the cohort -> mean, sd and 95% interval of
  each rule's threshold (gmsh plausibility_calibration's bootstrap, CI kept for the audit);
* ``tau_bootstrap``    the same for every relation's cut on the store's own facts vs
  counterfactual tails, plus the realised true/false accept rates at the deployed cut.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

BOOK_GRID = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]


@dataclass
class LabeledEdge:
    context: str
    relation: str
    tail: str
    label: str          # admit | block


def workflow_cohort(store, relation: str = "enables") -> list[LabeledEdge]:
    edges = {(w.src, w.dst) for w in store.workflow}
    nodes = sorted({w.src for w in store.workflow} | {w.dst for w in store.workflow})
    heads = sorted({w.src for w in store.workflow})
    out = [LabeledEdge(a, relation, b, "admit") for a, b in sorted(edges)]
    out += [LabeledEdge(a, relation, b, "block") for a in heads for b in nodes
            if b != "start" and (a, b) not in edges]
    return out


def sweep(scores: list[tuple[float, str]], thetas: list[float]) -> list[dict[str, Any]]:
    rows = []
    for th in thetas:
        fa = sum(1 for s, l in scores if s <= th and l == "block")
        fr = sum(1 for s, l in scores if s > th and l == "admit")
        tp = sum(1 for s, l in scores if s <= th and l == "admit")
        tn = sum(1 for s, l in scores if s > th and l == "block")
        rows.append({"theta": round(th, 4), "false_admits": fa, "false_rejects": fr,
                     "accuracy": (tp + tn) / max(1, len(scores))})
    return rows


def operating_points(scores: list[tuple[float, str]]) -> dict[str, Any]:
    vals = sorted({s for s, _ in scores})
    cands = [vals[0] - 1e-6] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [vals[-1] + 1e-6]
    rows = sweep(scores, cands)
    zero_fa = [r for r in rows if r["false_admits"] == 0]
    regulated = min(zero_fa, key=lambda r: (r["false_rejects"], -r["theta"]))
    best_acc = max(rows, key=lambda r: (r["accuracy"], -r["false_admits"], -r["false_rejects"]))
    pos = [s for s, l in scores if l == "admit"]
    neg = [s for s, l in scores if l == "block"]
    auc = float(np.mean([1.0 if p < n else 0.5 if p == n else 0.0 for p in pos for n in neg])) if pos and neg else float("nan")
    return {"regulated": regulated, "max_accuracy": best_acc, "auc": auc,
            "separable": max(pos) < min(neg) if pos and neg else None,
            "max_admit_score": max(pos) if pos else None, "min_block_score": min(neg) if neg else None}


# ----------------------------------------------------------------------------- gmsh comparability
def logistic_p50(scores: list[tuple[float, str]], l2: float = 1.0, iters: int = 100) -> dict[str, Any]:
    """One-feature logistic P(admit | score) (L2 on the slope, like sklearn's default C=1) and the
    score where P = 0.5 - gmsh GMSJudge's operating-point rule - with its errors on the cohort."""
    x = np.array([s for s, _ in scores], dtype=float)
    y = np.array([1.0 if l == "admit" else 0.0 for _, l in scores])
    if len(set(y)) < 2:
        return {"theta": float("nan"), "note": "one class only"}
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    X = np.c_[np.ones_like(x), (x - mu) / sd]
    w = np.zeros(2)
    R = np.diag([0.0, l2])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (y - p) - R @ w
        H = X.T @ (X * (p * (1 - p) + 1e-9)[:, None]) + R
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    theta = float(mu + sd * (-w[0] / w[1])) if abs(w[1]) > 1e-12 else float("nan")
    at = sweep(scores, [theta])[0] if theta == theta else {}
    return {"theta": theta, "coef": [float(w[0]), float(w[1])], **{k: v for k, v in at.items() if k != "theta"},
            "rule": "P(admit|score) = 0.5 (gmsh GMSJudge)"}


def _rules(scores: list[tuple[float, str]]) -> dict[str, float]:
    ops = operating_points(scores)
    return {"regulated": ops["regulated"]["theta"], "max_accuracy": ops["max_accuracy"]["theta"],
            "p50": logistic_p50(scores)["theta"]}


def holdout_eval(scores: list[tuple[float, str]], k: int = 5, seed: int = 0) -> dict[str, Any]:
    """Stratified k-fold: fit each rule on k-1 folds, count its errors on the held-out fold."""
    rng = np.random.default_rng(seed)
    pos = [i for i, (_, l) in enumerate(scores) if l == "admit"]
    neg = [i for i, (_, l) in enumerate(scores) if l == "block"]
    k = max(2, min(k, len(pos), len(neg)))
    rng.shuffle(pos)
    rng.shuffle(neg)
    fold = {i: j % k for j, i in enumerate(pos)} | {i: j % k for j, i in enumerate(neg)}
    tot = {r: {"false_admits": 0, "false_rejects": 0, "n": 0} for r in ("regulated", "max_accuracy", "p50")}
    for f in range(k):
        train = [scores[i] for i in range(len(scores)) if fold[i] != f]
        test = [scores[i] for i in range(len(scores)) if fold[i] == f]
        for r, th in _rules(train).items():
            if th != th:
                continue
            sw = sweep(test, [th])[0]
            tot[r]["false_admits"] += sw["false_admits"]
            tot[r]["false_rejects"] += sw["false_rejects"]
            tot[r]["n"] += len(test)
    return {"k": k, **{r: {**v, "heldout_accuracy": 1 - (v["false_admits"] + v["false_rejects"]) / max(1, v["n"])}
                       for r, v in tot.items()}}


def bootstrap_thetas(scores: list[tuple[float, str]], B: int = 200, seed: int = 0) -> dict[str, Any]:
    """Stratified bootstrap of every rule's threshold: mean, sd and 95% interval."""
    rng = np.random.default_rng(seed)
    pos = [s for s in scores if s[1] == "admit"]
    neg = [s for s in scores if s[1] == "block"]
    draws: dict[str, list[float]] = {"regulated": [], "max_accuracy": [], "p50": []}
    for _ in range(B):
        sample = [pos[i] for i in rng.integers(0, len(pos), len(pos))] + [neg[i] for i in rng.integers(0, len(neg), len(neg))]
        for r, th in _rules(sample).items():
            if th == th:
                draws[r].append(th)
    out = {}
    for r, v in draws.items():
        if v:
            a = np.array(v)
            out[r] = {"mean": float(a.mean()), "sd": float(a.std()), "ci95": [float(np.percentile(a, 2.5)),
                                                                             float(np.percentile(a, 97.5))], "n": len(v)}
    return {"B": B, **out}


def tau_bootstrap(store, B: int = 200, seed: int = 7, max_relations: int = 60) -> dict[str, Any]:
    """Per relation: the cut re-fitted on bootstrap resamples of (true facts, counterfactual tails)
    with the store's own operating-point rule, and the realised TAR/FAR at the deployed cut."""
    import random as _random
    from ..knowledge.store import GMSStore
    rng_py = _random.Random(seed)
    rng = np.random.default_rng(seed)
    out = {}
    for rel, trs in sorted(store.by_r.items())[:max_relations]:
        try:
            tau = float(store.tau(rel))
        except (KeyError, ValueError):
            continue
        trs = sorted(trs)
        pos = [float(store.score_triple(h, r, t)) for h, r, t in trs]
        neg = [float(store.score_triple(h, r, c)) for h, r, t in trs for c in store.negatives_for(h, r, t, rng_py)]
        if not pos or not neg:
            continue
        taus, tars, fars = [], [], []
        for _ in range(B):
            P = [pos[i] for i in rng.integers(0, len(pos), len(pos))]
            N = [neg[i] for i in rng.integers(0, len(neg), len(neg))]
            op = GMSStore._operating_point(P, N)
            if op["tau"] == op["tau"]:
                taus.append(op["tau"])
            tars.append(float(np.mean([x <= tau for x in P])))
            fars.append(float(np.mean([x <= tau for x in N])))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
        out[rel] = {"tau_deployed": tau, "n_pos": len(pos), "n_neg": len(neg),
                    "refit_tau_ci95": ci(taus), "tar_ci95": ci(tars), "far_ci95": ci(fars)}
    return {"B": B, "relations": out,
            "note": "knowlytix cuts are learned cap radii (not re-fitted at run time); the refit interval shows how "
                    "stable a data-driven cut would be on this cohort, the TAR/FAR intervals how stable the deployed one is"}


def calibrate_plausibility(store, relation: str = "enables") -> dict[str, Any]:
    cohort = workflow_cohort(store, relation)
    scored = [(store.score_triple(e.context, e.relation, e.tail), e.label) for e in cohort]
    ops = operating_points(scored)
    false_rejects = [{"edge": f"{e.context}->{e.tail}", "score": round(s, 4)}
                     for e, (s, _) in zip(cohort, scored) if e.label == "admit" and s > ops["regulated"]["theta"]]
    return {"relation": relation, "n_admit": sum(e.label == "admit" for e in cohort),
            "n_block": sum(e.label == "block" for e in cohort),
            "book_grid": sweep(scored, BOOK_GRID), **ops, "theta": ops["regulated"]["theta"],
            "p50": logistic_p50(scored), "holdout": holdout_eval(scored), "bootstrap": bootstrap_thetas(scored),
            "false_rejects_at_theta": false_rejects,
            "cohort_scores": [{"edge": f"{e.context}->{e.tail}", "label": e.label, "score": round(s, 4)}
                              for e, (s, _) in zip(cohort, scored)]}


# ----------------------------------------------------------------------------- plans
def plan_cohort(store, journeys: list[list[str]], seed: int = 0) -> tuple[list[list[str]], list[list[str]]]:
    """Admissible tool plans (oracle journeys) and inadmissible perturbations of them."""
    edges = {(w.src, w.dst) for w in store.workflow}

    def admissible(seq):
        return all((a, b) in edges for a, b in zip(["start"] + seq, seq))

    pos = sorted({tuple(s) for s in journeys if s and admissible(s)})
    neg = set()
    for s in pos:
        s = list(s)
        for i in range(len(s)):                                  # skip a step
            t = s[:i] + s[i + 1:]
            if t and not admissible(t):
                neg.add(tuple(t))
        for i in range(len(s) - 1):                              # swap adjacent steps
            if s[i] != s[i + 1]:
                t = s[:i] + [s[i + 1], s[i]] + s[i + 2:]
                if not admissible(t):
                    neg.add(tuple(t))
        for tool in ("get_review_transactions", "record_charge_response", "submit_card_action"):
            t = [tool] + s                                        # act before authenticating
            if not admissible(t):
                neg.add(tuple(t))
    neg = sorted(neg - set(pos))
    return [list(p) for p in pos], [list(n) for n in neg]


def calibrate_plan(store, journeys: list[list[str]], relation: str = "enables") -> dict[str, Any]:
    from .gates import GMSPlanGate
    pos, neg = plan_cohort(store, journeys)
    g = GMSPlanGate(store, 0.0, relation)
    mean_scores = [(g.validate(p).drift, "admit") for p in pos] + [(g.validate(n).drift, "block") for n in neg]
    max_scores = [(g.validate(p).max_step, "admit") for p in pos] + [(g.validate(n).max_step, "block") for n in neg]
    mean_ops, max_ops = operating_points(mean_scores), operating_points(max_scores)
    # deploy the criterion with fewer false rejects at zero false admits (ties: the book's mean)
    use_max = max_ops["regulated"]["false_rejects"] < mean_ops["regulated"]["false_rejects"]
    chosen = max_ops if use_max else mean_ops
    chosen_scores = max_scores if use_max else mean_scores
    return {"n_admissible_plans": len(pos), "n_inadmissible_plans": len(neg),
            "criterion": "max" if use_max else "mean", "drift_budget": chosen["regulated"]["theta"],
            "p50": logistic_p50(chosen_scores), "holdout": holdout_eval(chosen_scores),
            "bootstrap": bootstrap_thetas(chosen_scores),
            "mean_drift": mean_ops, "max_step": max_ops,     # book criterion / per-step criterion
            "examples": {"admissible": pos[:3], "inadmissible": neg[:5]}}


# ----------------------------------------------------------------------------- multi-threshold DoE
def multi_threshold_design(store, theta_cal: float, n_runs: int = 12, seed: int = 42) -> dict[str, Any]:
    """The appendix's balanced design over the thresholds that matter, each row scored."""
    from ..design.doe import balanced_design, coverage_report
    from ..knowledge.ingest import Rule
    tens = store.calibration["tension"]
    hol = store.calibration["holonomy"]
    factors = {
        "theta_plausibility": [round(theta_cal * 0.8, 3), round(theta_cal, 3), round(theta_cal * 1.2, 3), 1.5],
        "tau_contra": [round(tens["tau_contra"], 3), 1.5, 1.7, 1.9],
        "numeric_tolerance": [0.005, 0.01, 0.02],
        "holonomy_threshold": [round(hol["tau_path"], 3), 0.3, 0.5],
    }
    design = balanced_design(factors, n_runs, seed)
    cohort = workflow_cohort(store)
    scored = [(store.score_triple(e.context, e.relation, e.tail), e.label) for e in cohort]
    planted = dict(store.rules)
    r13 = store.rules["R13"]
    planted["R13x"] = Rule("R13x", "must", r13.subject, r13.predicate, r13.obj, "reversed copy", r13.source)
    enm_vals = [e.as_number() for e in store.enm.values()]
    hp, hn = hol["positives"], hol["negatives"]
    rows = []
    saved = dict(tens)
    try:
        for row in design:
            th = row["theta_plausibility"]
            pl = sweep(scored, [th])[0]
            store.calibration["tension"] = {**saved, "tau_contra": row["tau_contra"]}
            c_err = int(store.find_contradictions() != []) + int(len(store.find_contradictions(planted)) == 0)
            tol = row["numeric_tolerance"]
            n_err = 0
            for v in enm_vals:          # a stated figure is accepted iff it is within tolerance of the register
                for claim, should in ((v, True), (float(f"{v:.1f}"), True), (v * 1.01 + (0.01 if v == 0 else 0), False),
                                      (v * 1.05 + (1 if v == 0 else 0), False), (v + 1, False)):
                    ok = abs(claim - v) <= tol * max(abs(v), 1.0)
                    n_err += ok != should
            h_err = sum(p > row["holonomy_threshold"] for p in hp) + sum(n <= row["holonomy_threshold"] for n in hn)
            rows.append({**row, "plausibility_errors": pl["false_admits"] + pl["false_rejects"],
                         "contradiction_errors": c_err, "numeric_errors": n_err, "holonomy_errors": h_err,
                         "total_errors": pl["false_admits"] + pl["false_rejects"] + c_err + n_err + h_err})
    finally:
        store.calibration["tension"] = saved
    cov = coverage_report(design, factors)
    balanced = all(max(c.values()) - min(c.values()) <= 1 for c in cov.values())
    return {"factors": factors, "rows": rows, "coverage": {k: {str(a): b for a, b in v.items()} for k, v in cov.items()},
            "balanced": balanced, "best": min(rows, key=lambda r: r["total_errors"])}


# ----------------------------------------------------------------------------- recalibration
def psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Population stability index between two score samples (quantile bins of `expected`)."""
    if not expected or not actual:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(min(expected + actual), max(expected + actual) + 1e-9, bins + 1)
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def store_fingerprint(store) -> str:
    facts = json.dumps(sorted(list(t) for t in store.triples)).encode()
    trn = json.dumps(store.calibration.get("training", {}), sort_keys=True, default=str).encode()
    return hashlib.sha256(facts + trn).hexdigest()[:16]


def policy_fingerprint(store) -> str:
    rules = json.dumps(sorted((r.rule_id, r.stance, r.subject, r.predicate, r.obj) for r in store.rules.values()))
    return hashlib.sha256(rules.encode()).hexdigest()[:16]


def recalibration_triggers(record: dict[str, Any], store, runtime_scores: list[float],
                           runtime_false_reject_rate: float, psi_limit: float = 0.2,
                           fr_margin: float = 0.05) -> dict[str, Any]:
    """record = what was stored at calibration time (fingerprints, reference scores, FR)."""
    p = psi(record["reference_scores"], runtime_scores)
    trig = {
        "store_retrained": store_fingerprint(store) != record["store_fingerprint"],
        "score_drift_psi": (not math.isnan(p)) and p > psi_limit,
        "false_reject_creep": runtime_false_reject_rate > record["false_reject_rate"] + fr_margin,
        "policy_changed": policy_fingerprint(store) != record["policy_fingerprint"],
    }
    return {"psi": p, "psi_limit": psi_limit, "runtime_false_reject_rate": runtime_false_reject_rate,
            "calibrated_false_reject_rate": record["false_reject_rate"], "triggers": trig,
            "recalibrate": any(trig.values())}


# ----------------------------------------------------------------------------- probe sweep
def gate_probe_sweep(store, theta: float, relation: str = "enables") -> dict[str, Any]:
    """Book 12.5.1: every (previous step, proposed tool) pair through the geometric gate,
    compared with the policy graph. Escalation actions are safe-harboured (never denied)."""
    edges = {(w.src, w.dst) for w in store.workflow}
    from ..domain import current
    wf = current().get("flow.workflow")
    tools = sorted({w.dst for w in store.workflow} - {wf["summary"], wf["end"]})
    prevs = [wf["start"]] + tools
    rows, conf = [], {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for p in prevs:
        for t in tools:
            s = store.score_triple(p, relation, t)
            admit = s <= theta
            decision = "allow" if admit else ("escalate" if t == "transfer_to_human" else "deny")
            truth = (p, t) in edges
            conf["tp" if admit and truth else "fp" if admit else "fn" if truth else "tn"] += 1
            rows.append({"prev": p, "proposed": t, "score": round(s, 4), "decision": decision, "in_policy_graph": truth})
    return {"theta": theta, "confusion": conf, "rows": rows,
            "false_denies": [r for r in rows if r["in_policy_graph"] and r["decision"] == "deny"],
            "false_allows": [r for r in rows if not r["in_policy_graph"] and r["decision"] == "allow"]}
