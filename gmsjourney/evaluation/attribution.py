"""Which inputs break which part of the bot (book Ch. 10, 12.6).

``attribute``   per-factor analysis of deviance, Benjamini-Hochberg corrected;
``diagnose``    the full diagnosis: factor attribution on the outcome, per-level
                failure rates with Wilson intervals and odds ratios, the joint
                model (pseudo-R^2, AIC, BIC, AUC, Hosmer-Lemeshow), interactions
                among significant factors, per-category accuracy, per-component
                attribution (each component's own correctness as the metric) and
                the weak-link decomposition.
When failures are few, logistic fits are unstable; the empirical per-level rates
are then the robust report and the result says so (Ch. 12.6).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .runner import weak_link
from .stats import bh_adjust, deviance_test, interaction_test, joint_model, level_table, wilson

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
