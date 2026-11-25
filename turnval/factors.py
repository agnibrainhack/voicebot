"""Block 2 - journey factors and the covering array.

``Factors.load()`` reads config/factors.yaml. A row assigns every factor a level, or "n/a" when the factor's ``when``
does not hold; rows must satisfy every constraint.

``covering_array()`` builds conversations so that every feasible pair of levels of relevant factors appears in at least
one row (and every feasible triple for the factor groups listed under ``triples``). Feasibility is decided by trying to
build a valid row that contains the combination; the greedy builder then adds the row that covers the most uncovered
combinations until none is left (AETG-style).

CLI:
  python -m turnval.factors design --out design.csv [--replicates 3] [--seed 7]
  python -m turnval.factors audit        # every outcome the flow branches on is driven by some factor level
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path(__file__).parent / "config" / "factors.yaml"
NA = "n/a"


@dataclass
class Factor:
    name: str
    levels: list[str]
    when: str | None
    source: str
    doc: str
    covers: dict[str, list[str]]


class Factors:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.factors: list[Factor] = [
            Factor(k, [str(x) for x in v["levels"]], v.get("when"), v.get("source", "suggested"), v.get("doc", ""),
                   v.get("covers") or {})
            for k, v in cfg["factors"].items()]
        self.by_name = {f.name: f for f in self.factors}
        self.constraints: list[str] = cfg.get("constraints", [])
        self.strength = int(cfg.get("strength", 2))
        self.triples: list[list[str]] = cfg.get("triples", [])
        self._compiled = {f.name: compile(f.when, f.name, "eval") for f in self.factors if f.when}
        self._cons = [compile(c, "constraint", "eval") for c in self.constraints]

    @classmethod
    def load(cls, path: str | Path = CONFIG) -> "Factors":
        return cls(yaml.safe_load(Path(path).read_text()))

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.factors]

    # ------------------------------------------------------------------ rows
    def relevant(self, f: Factor, row: dict[str, str]) -> bool:
        if not f.when:
            return True
        return bool(eval(self._compiled[f.name], {"__builtins__": {}}, dict(row)))

    def valid(self, row: dict[str, str]) -> bool:
        for f in self.factors:
            rel = self.relevant(f, row)
            if rel != (row.get(f.name, NA) != NA):
                return False
        env = {n: row.get(n, NA) for n in self.names}
        return all(eval(c, {"__builtins__": {}}, env) for c in self._cons)

    def build(self, fixed: dict[str, str], rng: random.Random, chooser=None, tries: int = 200) -> dict[str, str] | None:
        """A valid row containing ``fixed`` (None when none was found). ``chooser(name, levels, row)`` picks a level."""
        for _ in range(tries):
            row: dict[str, str] = {}
            ok = True
            for f in self.factors:
                rel = self.relevant(f, row)
                if f.name in fixed:
                    if not rel:
                        ok = False
                        break
                    row[f.name] = fixed[f.name]
                elif not rel:
                    row[f.name] = NA
                else:
                    row[f.name] = chooser(f.name, f.levels, row) if chooser else rng.choice(f.levels)
            if ok and self.valid(row):
                return row
            chooser = None if rng.random() < 0.5 else chooser          # randomise on retries
        return None

    # ------------------------------------------------------------------ coverage targets
    def _items(self, row: dict[str, str]) -> set[tuple]:
        """The combinations a row covers: pairs of relevant factors, and triples of the configured groups."""
        rel = [(n, row[n]) for n in self.names if row.get(n, NA) != NA]
        out = set(itertools.combinations(rel, 2)) if self.strength >= 2 else set()
        for grp in self.triples:
            vals = [(n, row.get(n, NA)) for n in grp]
            if all(v != NA for _, v in vals):
                out.add(tuple(vals))
        return out

    def targets(self, rng: random.Random) -> tuple[set[tuple], int]:
        """Every feasible pair (and configured triple), decided by building a row that contains it."""
        feasible: set[tuple] = set()
        infeasible = 0
        cand: list[tuple] = []
        for a, b in itertools.combinations(self.factors, 2):
            for va in a.levels:
                for vb in b.levels:
                    cand.append(((a.name, va), (b.name, vb)))
        for grp in self.triples:
            fs = [self.by_name[n] for n in grp]
            for vals in itertools.product(*[f.levels for f in fs]):
                cand.append(tuple(zip(grp, vals)))
        for item in cand:
            if item in feasible:
                continue
            row = self.build(dict(item), rng, tries=60)
            if row is None:
                infeasible += 1
                continue
            feasible |= self._items(row)
        return feasible, infeasible

    def covering_array(self, seed: int = 7, candidates: int = 12, max_rows: int = 5000,
                       log=print) -> tuple[list[dict[str, str]], dict[str, Any]]:
        rng = random.Random(seed)
        targets, infeasible = self.targets(rng)
        uncovered = set(targets)
        log(f"[factors] {len(self.factors)} factors, {len(targets)} feasible combinations to cover "
            f"({infeasible} infeasible skipped)")
        rows: list[dict[str, str]] = []
        while uncovered and len(rows) < max_rows:
            best, best_gain = None, -1
            seeds = rng.sample(sorted(uncovered), min(candidates, len(uncovered)))
            for item in seeds:
                def chooser(name, levels, row, _u=uncovered):
                    def gain(v):
                        g = 0
                        for n2, v2 in row.items():
                            if v2 == NA:
                                continue
                            if ((n2, v2), (name, v)) in _u or ((name, v), (n2, v2)) in _u:
                                g += 1
                        return g + rng.random() * 0.5
                    return max(levels, key=gain)
                row = self.build(dict(item), rng, chooser)
                if row is None:
                    continue
                g = len(self._items(row) & uncovered)
                if g > best_gain:
                    best, best_gain = row, g
            if best is None:
                break
            rows.append(best)
            uncovered -= self._items(best)
        report = {"factors": len(self.factors), "targets": len(targets), "infeasible": infeasible,
                  "rows": len(rows), "uncovered": len(uncovered), "seed": seed}
        log(f"[factors] covering array: {len(rows)} rows, {len(uncovered)} combinations left uncovered")
        return rows, report

    def design(self, seed: int = 7, replicates: int = 1, extra_random: int = 0,
               log=print) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """The covering array, repeated ``replicates`` times (each copy becomes a different conversation: other
        wording, data and error choices), plus ``extra_random`` valid random rows."""
        rows, rep = self.covering_array(seed, log=log)
        rng = random.Random(seed + 1)
        out = []
        for r in range(replicates):
            for i, row in enumerate(rows):
                out.append({"design_id": f"ca{i:05d}", "replicate": str(r), **row})
        for j in range(extra_random):
            row = self.build({}, rng)
            if row:
                out.append({"design_id": f"rnd{j:05d}", "replicate": "0", **row})
        rep.update({"replicates": replicates, "extra_random": extra_random, "conversations": len(out)})
        return out, rep

    # ------------------------------------------------------------------ reports
    def coverage(self, rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for f in self.factors:
            c = {lv: 0 for lv in f.levels}
            c[NA] = 0
            for r in rows:
                c[r.get(f.name, NA)] = c.get(r.get(f.name, NA), 0) + 1
            out[f.name] = c
        return out

    def audit(self, flow=None) -> dict[str, Any]:
        """Which outcomes the flow branches on are not driven by any factor level (`covers`)."""
        from .graph import Flow
        flow = flow or Flow()
        driven: dict[str, set[str]] = {}
        for f in self.factors:
            for k, vals in f.covers.items():
                driven.setdefault(k, set()).update(str(v) for v in vals)
        missing = {}
        for (tool, fld), vals in sorted(flow.decisions().items()):
            key = f"{tool}.{fld}" if fld else tool
            left = sorted(v for v in vals if v not in driven.get(key, set()) and v != "NO_USER_RESPONSE")
            if left:
                missing[key] = left
        return {"decisions": {f"{t}.{f}" if f else t: sorted(v) for (t, f), v in flow.decisions().items()},
                "not_driven_by_a_factor": missing}


def write_csv(rows: list[dict[str, str]], path: str | Path) -> None:
    cols = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turnval.factors")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design")
    d.add_argument("--config", default=str(CONFIG))
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--replicates", type=int, default=1)
    d.add_argument("--extra-random", type=int, default=0)
    a = sub.add_parser("audit")
    a.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args(argv)
    fx = Factors.load(args.config)
    if args.cmd == "audit":
        print(json.dumps(fx.audit()["not_driven_by_a_factor"], indent=1))
        return 0
    rows, rep = fx.design(args.seed, args.replicates, args.extra_random)
    write_csv(rows, args.out)
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(
        {**rep, "coverage": fx.coverage(rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
