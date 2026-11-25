"""Block 7 - train / val / test, stratified on every journey factor.

Units are groups of conversations (``group_by``: design_id by default, so the replicates of one design row - same
path, different words - never sit on both sides of a split; or conv_id). Each unit carries a set of labels:
every factor=level it has (factor columns f_*), its error family, its AOP, and its share of label-0 rows. Units are
dealt with iterative stratification (Sechidis et al., 2011): the rarest label first, each unit to the split that
most lacks that label, so every factor level appears in each split close to the target ratio.

  python -m turnval.split --rows turnval/out/rows.csv --out turnval/out

Writes train.csv, val.csv, test.csv and split_report.json (per factor level: share in each split, the largest
deviation from the target, rows and label balance per split).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def _unit_labels(df: pd.DataFrame, key: str) -> dict[str, set[str]]:
    fcols = [c for c in df.columns if c.startswith("f_")]
    out: dict[str, set[str]] = defaultdict(set)
    for unit, g in df.groupby(key):
        first = g.iloc[0]
        for c in fcols:
            v = str(first[c])
            if v != "n/a":
                out[unit].add(f"{c[2:]}={v}")
        for fam in g.loc[g["label"] == 0, "error_family"].astype(str).unique():
            out[unit].add(f"error={fam}")
        out[unit].add(f"aop={first['aop']}")
        out[unit].add(f"length={first['length_band']}")
    return out


def iterative_stratify(units: dict[str, set[str]], ratios: dict[str, float], seed: int = 13) -> dict[str, str]:
    rng = random.Random(seed)
    names = list(ratios)
    total = len(units)
    want_total = {s: ratios[s] * total for s in names}
    counts = Counter(l for ls in units.values() for l in ls)
    want = {s: {l: ratios[s] * c for l, c in counts.items()} for s in names}
    left = dict(units)
    assign: dict[str, str] = {}
    while left:
        remaining = Counter(l for ls in left.values() for l in ls)
        if not remaining:
            for u in list(left):
                s = max(names, key=lambda x: want_total[x])
                assign[u] = s
                want_total[s] -= 1
                del left[u]
            break
        label = min(remaining, key=lambda l: (remaining[l], l))
        us = [u for u, ls in left.items() if label in ls]
        rng.shuffle(us)
        for u in us:
            s = max(names, key=lambda x: (want[x][label], want_total[x], rng.random()))
            assign[u] = s
            want_total[s] -= 1
            for l in left[u]:
                want[s][l] -= 1
            del left[u]
    return assign


def refine(units: dict[str, set[str]], assign: dict[str, str], ratios: dict[str, float], seed: int = 13,
           iters: int = 40000) -> dict[str, str]:
    """Swap units between splits while it lowers sum_l sum_s (n_s(l) - r_s n(l))^2 / n(l): every label's share
    moves toward its target; split sizes stay as they are."""
    rng = random.Random(seed)
    names = list(ratios)
    n_l = Counter(l for ls in units.values() for l in ls)
    cnt = {s: Counter() for s in names}
    for u, s in assign.items():
        cnt[s].update(units[u])
    by = {s: [u for u, x in assign.items() if x == s] for s in names}

    def cost(s, l, c):
        return (c - ratios[s] * n_l[l]) ** 2 / n_l[l]

    for _ in range(iters):
        a, b = rng.sample(names, 2)
        if not by[a] or not by[b]:
            continue
        i, j = rng.randrange(len(by[a])), rng.randrange(len(by[b]))
        u, v = by[a][i], by[b][j]
        lu, lv = units[u], units[v]
        delta = 0.0
        for l in lu ^ lv:
            du = (1 if l in lv else 0) - (1 if l in lu else 0)     # change in split a's count of l
            ca, cb = cnt[a][l], cnt[b][l]
            delta += cost(a, l, ca + du) - cost(a, l, ca) + cost(b, l, cb - du) - cost(b, l, cb)
        if delta < -1e-9:
            by[a][i], by[b][j] = v, u
            assign[u], assign[v] = b, a
            cnt[a].subtract(lu); cnt[a].update(lv); cnt[b].subtract(lv); cnt[b].update(lu)
    return assign


def report(df: pd.DataFrame, ratios: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": df.groupby("split").size().to_dict(),
                           "conversations": df.groupby("split")["conv_id"].nunique().to_dict(),
                           "positive_rate": df.groupby("split")["label"].mean().round(4).to_dict(),
                           "factors": {}, "max_deviation": 0.0}
    conv = df.drop_duplicates("conv_id")
    fcols = [c for c in df.columns if c.startswith("f_")] + ["error_family"]
    for c in fcols:
        src = conv if c.startswith("f_") else df[df["label"] == 0]
        levels = {}
        for lv, g in src.groupby(c):
            if str(lv) == "n/a":
                continue
            share = (g["split"].value_counts() / len(g)).to_dict()
            dev = max(abs(share.get(s, 0.0) - r) for s, r in ratios.items())
            levels[str(lv)] = {"n": int(len(g)), **{s: round(share.get(s, 0.0), 3) for s in ratios}, "max_dev": round(dev, 3)}
            out["max_deviation"] = max(out["max_deviation"], dev if len(g) >= 10 else 0.0)
        out["factors"][c.removeprefix("f_")] = levels
    out["max_deviation"] = round(out["max_deviation"], 3)
    return out


def split_files(rows_csv: str | Path, out_dir: str | Path, cfg: dict[str, Any], factor_names: list[str] | None = None) -> dict:
    df = pd.read_csv(rows_csv, keep_default_na=False, dtype=str)
    df["label"] = df["label"].astype(int)
    key = cfg.get("group_by", "design_id")
    ratios = cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    units = _unit_labels(df, key)
    assign = iterative_stratify(units, ratios, int(cfg.get("seed", 13)))
    assign = refine(units, assign, ratios, int(cfg.get("seed", 13)), int(cfg.get("refine_iters", 40000)))
    df["split"] = df[key].map(assign)
    out = Path(out_dir)
    for s in ratios:
        df[df["split"] == s].drop(columns=["split"]).to_csv(out / f"{s}.csv", index=False)
    rep = report(df, ratios)
    rep["group_by"] = key
    (out / "split_report.json").write_text(json.dumps(rep, indent=1))
    print(f"[split] rows {rep['rows']} | positive rate {rep['positive_rate']} | "
          f"largest deviation of a factor level (n>=10) from its target share: {rep['max_deviation']}")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turnval.split")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-by", default="design_id")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)
    split_files(args.rows, args.out, {"group_by": args.group_by, "seed": args.seed,
                                      "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
