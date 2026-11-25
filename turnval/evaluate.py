"""Block 9 - how good is the validator: overall, per category, calibrated, with operating thresholds.

Conventions: the model outputs p = P(admissible). "Flag" = the validator says the turn is wrong (p < threshold); the
error class (label 0) is the one we want to catch.

Metrics
  overall      accuracy, balanced accuracy, MCC; precision / recall / F1 of flagging errors and of passing correct
               turns; ROC-AUC; PR-AUC for catching errors; Brier score; expected calibration error (10 bins)
  pairwise     for each context with a correct and a wrong response: how often the correct one scores higher
  thresholds   0.5; best F1 on errors (chosen on val); the lowest threshold whose flags are right >= target (e.g.
               95%) and the highest whose passes are right >= target, each with coverage and a Wilson interval
  categories   per error family, operator, AOP, state, phase (pre / error / post), variant, turn position,
               conversation length and every journey factor level: n, accuracy, error recall, correct-pass rate, AUC

  python -m turnval.evaluate --pred runs/x/test_predictions.csv --out runs/x/test_report   (rows + p column)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binary(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, Any]:
    pred_ok = p >= thr
    tp_err = int(((~pred_ok) & (y == 0)).sum())
    fp_err = int(((~pred_ok) & (y == 1)).sum())
    fn_err = int((pred_ok & (y == 0)).sum())
    tp_ok = int((pred_ok & (y == 1)).sum())
    def prf(tp, fp, fn):
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return round(pr, 4), round(rc, 4), round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0
    pe, re_, fe = prf(tp_err, fp_err, fn_err)
    po, ro, fo = prf(tp_ok, fn_err, fp_err)
    both = len(set(y.tolist())) == 2
    return {"n": int(len(y)), "threshold": round(float(thr), 4), "accuracy": round(float((pred_ok == (y == 1)).mean()), 4),
            "balanced_accuracy": round((re_ + ro) / 2, 4),
            "mcc": round(float(matthews_corrcoef(y, pred_ok.astype(int))), 4) if both else None,
            "error_precision": pe, "error_recall": re_, "error_f1": fe,
            "pass_precision": po, "pass_recall": ro, "pass_f1": fo,
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if both else None,
            "pr_auc_error": round(float(average_precision_score(1 - y, 1 - p)), 4) if both else None,
            "brier": round(float(np.mean((p - y) ** 2)), 4), "ece": round(ece(y, p), 4),
            "confusion": {"flag_wrong_turn": tp_err, "flag_correct_turn": fp_err, "pass_wrong_turn": fn_err,
                          "pass_correct_turn": tp_ok}}


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    cands = np.unique(np.round(p, 4))
    best, bt = -1.0, 0.5
    for t in cands:
        f = binary(y, p, float(t))["error_f1"]
        if f > best:
            best, bt = f, float(t)
    return bt


def target_thresholds(y: np.ndarray, p: np.ndarray, target: float = 0.95) -> dict[str, Any]:
    """Thresholds where the validator's decisions are right at least ``target`` of the time."""
    order = np.argsort(p)
    ys, ps = y[order], p[order]
    out: dict[str, Any] = {"target": target}
    # flag the lowest scores: largest prefix whose share of real errors >= target
    best = None
    errs = np.cumsum(ys == 0)
    for i in range(len(ps)):
        n = i + 1
        if errs[i] / n >= target:
            best = i
    if best is not None:
        n, k = best + 1, int(errs[best])
        out["flag"] = {"threshold": round(float(ps[best]) + 1e-6, 4), "flagged": n, "precision": round(k / n, 4),
                       "wilson95": wilson(k, n), "errors_caught": round(k / max(1, int((y == 0).sum())), 4)}
    best = None
    oks = np.cumsum((ys == 1)[::-1])
    for i in range(len(ps)):
        n = i + 1
        if oks[i] / n >= target:
            best = i
    if best is not None:
        n, k = best + 1, int(oks[best])
        out["pass"] = {"threshold": round(float(ps[::-1][best]), 4), "passed": n, "precision": round(k / n, 4),
                       "wilson95": wilson(k, n), "correct_passed": round(k / max(1, int((y == 1).sum())), 4)}
    return out


def pairwise_accuracy(df: pd.DataFrame) -> dict[str, Any]:
    wins = tot = 0
    by_family: dict[str, list[int]] = {}
    for _, g in df.groupby("group_id"):
        pos, neg = g[g.label == 1], g[g.label == 0]
        for _, a in pos.iterrows():
            for _, b in neg.iterrows():
                w = int(a.p > b.p)
                wins += w
                tot += 1
                by_family.setdefault(b.error_family, [0, 0])
                by_family[b.error_family][0] += w
                by_family[b.error_family][1] += 1
    return {"pairs": tot, "accuracy": round(wins / tot, 4) if tot else None,
            "by_error_family": {k: {"pairs": v[1], "accuracy": round(v[0] / v[1], 4)} for k, v in sorted(by_family.items())}}


CATEGORY_COLS = ["error_family", "error_operator", "aop", "gold_state", "phase", "variant", "position", "length_band"]


def by_category(df: pd.DataFrame, thr: float, min_n: int = 5) -> dict[str, list[dict[str, Any]]]:
    cols = [c for c in CATEGORY_COLS if c in df.columns] + [c for c in df.columns if c.startswith("f_")]
    out = {}
    for c in cols:
        rows = []
        for lv, g in df.groupby(c):
            if len(g) < min_n or str(lv) == "n/a":
                continue
            y, p = g.label.to_numpy(), g.p.to_numpy()
            pred_ok = p >= thr
            r = {"level": str(lv), "n": int(len(g)), "n_wrong": int((y == 0).sum()),
                 "accuracy": round(float((pred_ok == (y == 1)).mean()), 4),
                 "error_recall": round(float((~pred_ok[y == 0]).mean()), 4) if (y == 0).any() else None,
                 "pass_rate_correct": round(float(pred_ok[y == 1].mean()), 4) if (y == 1).any() else None,
                 "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(set(y.tolist())) == 2 else None}
            rows.append(r)
        out[("factor:" + c[2:]) if c.startswith("f_") else c] = sorted(rows, key=lambda r: (r["accuracy"], -r["n"]))
    return out


def full_report(df: pd.DataFrame, thr_val: dict[str, float] | None = None, target: float = 0.95) -> dict[str, Any]:
    """df: rows with label and p (P(admissible))."""
    y, p = df.label.to_numpy().astype(int), df.p.to_numpy().astype(float)
    thr_val = thr_val or {}
    t_f1 = thr_val.get("best_f1", best_f1_threshold(y, p))
    return {"overall@0.5": binary(y, p, 0.5), "overall@best_f1": binary(y, p, t_f1),
            "target_thresholds_here": target_thresholds(y, p, target),
            "at_val_thresholds": {k: binary(y, p, v) for k, v in thr_val.items()},
            "pairwise": pairwise_accuracy(df) if "group_id" in df.columns else None,
            "categories@best_f1": by_category(df, t_f1)}


def markdown(rep: dict[str, Any], title: str) -> str:
    o = rep["overall@best_f1"]
    lines = [f"# {title}", "", f"Rows: {o['n']}. Threshold (best F1 on val): {o['threshold']}.", "",
             "| metric | value |", "|---|---|"]
    for k in ("accuracy", "balanced_accuracy", "mcc", "error_precision", "error_recall", "error_f1", "pass_precision",
              "pass_recall", "roc_auc", "pr_auc_error", "brier", "ece"):
        lines.append(f"| {k} | {o[k]} |")
    tt = rep["target_thresholds_here"]
    for side in ("flag", "pass"):
        if side in tt:
            lines.append(f"| {side} @ precision>={tt['target']} | {json.dumps(tt[side])} |")
    if rep.get("pairwise"):
        lines.append(f"| pairwise accuracy | {rep['pairwise']['accuracy']} ({rep['pairwise']['pairs']} pairs) |")
    for cat in ("error_family", "aop", "phase", "position", "length_band", "variant"):
        rows = rep["categories@best_f1"].get(cat, [])
        if not rows:
            continue
        lines += ["", f"## By {cat}", "", "| level | n | wrong | accuracy | error recall | correct passed | AUC |",
                  "|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -r["n"]):
            lines.append(f"| {r['level']} | {r['n']} | {r['n_wrong']} | {r['accuracy']} | {r['error_recall']} | "
                         f"{r['pass_rate_correct']} | {r['roc_auc']} |")
    worst = sorted(((c.removeprefix("factor:"), r) for c, rs in rep["categories@best_f1"].items()
                    if c.startswith("factor:") for r in rs if r["n"] >= 20), key=lambda x: x[1]["accuracy"])[:15]
    if worst:
        lines += ["", "## Weakest journey-factor levels (n >= 20)", "", "| factor | level | n | accuracy | error recall |",
                  "|---|---|---|---|---|"]
        for c, r in worst:
            lines.append(f"| {c} | {r['level']} | {r['n']} | {r['accuracy']} | {r['error_recall']} |")
    return "\n".join(lines) + "\n"


def write(rep: dict[str, Any], out_prefix: str | Path, title: str) -> None:
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.json").write_text(json.dumps(rep, indent=1, default=str))
    Path(f"{out}.md").write_text(markdown(rep, title))
    flat = [{"category": c, **r} for c, rs in rep["categories@best_f1"].items() for r in rs]
    pd.DataFrame(flat).to_csv(f"{out}_categories.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turnval.evaluate")
    ap.add_argument("--pred", required=True, help="csv with the dataset columns plus p = P(admissible)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", help="json of val thresholds (from training)")
    ap.add_argument("--target", type=float, default=0.95)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.pred, keep_default_na=False)
    thr = json.loads(Path(a.thresholds).read_text()) if a.thresholds else None
    write(full_report(df, thr, a.target), a.out, Path(a.pred).stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
