"""Coverage of a mined suite, measured against the oracle's own exploration of the policy.

A *decision point* is a place where a conversation can branch: the n-th result of a tool
(``tool:record_charge_response#1``) or the customer's answer at the n-th visit of a prompt
(``ask:PRESENT_TXN#0``). Its *options* are what the miner enumerates there (graph/journeys.py).
Every explored conversation is a set of (decision point, option) pairs.

t-wise coverage (the covering-array criterion, Ch. 5 / 12.3): of all combinations of t options at
t different decision points that occur together in *some* conversation the policy allows (the
feasible universe, explored by the oracle at a deviation budget of at least t), the share that the
suite exercises. A combination counts as exercised when a conversation containing it behaves
exactly like a suite case (the same bot acts, tool results and outcome): an equivalent behaviour is
the same test.

``select`` builds a covering suite: greedy set cover over the t-wise combinations plus every state,
workflow edge, rule and transfer reason, then the ``must_cover`` additions, minus ``exclude``d
paths, within ``max_bases``.

Controls (domain.yaml -> coverage, all optional):

  strength      t of the t-wise criterion (default 2)
  explore       deviation budget of the exploration the suite is drawn from (default: strength)
  select        all (every explored behaviour, the default) | minimal (greedy covering suite)
  must_cover    lists of option / state patterns that must appear together on some case
  exclude       [{match: [patterns], reason: ...}]: paths deliberately left out, with the reason
  max_bases     a cap on the selected suite (the ledger shows what the cap leaves uncovered)
  ledger_strengths  the strengths reported (default [1, 2, 3])
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any


def combos(decisions, t: int) -> set[tuple]:
    pts = sorted(set(decisions))
    return {c for c in combinations(pts, t) if len({dp for dp, _ in c}) == t}


def _matches(patterns: list[str], spec, tr) -> bool:
    text = [lab for _, lab in getattr(tr, "decisions", [])] + list(spec.label) + list(tr.visited) + \
           list(tr.rules_fired) + [str(tr.transfer_reason)]
    return all(any(p in x for x in text) for p in patterns)


def structural_items(tr, workflow) -> set[tuple]:
    hit, _ = tr.edges(workflow)
    return ({("edge", e) for e in hit} | {("state", s) for s in tr.visited} | {("rule", r) for r in tr.rules_fired} |
            ({("reason", tr.transfer_reason)} if tr.transfer_reason else set()))


def explore(oracle, budget: int) -> tuple[list, list, dict]:
    """(behaviours, ledger, stats) of the oracle's exploration at a deviation budget."""
    from .journeys import mine_paths
    ledger: list = []
    stats: dict = {}
    paths = mine_paths(oracle, max_deviations=budget, ledger=ledger, stats=stats)
    return paths, ledger, stats


def measure(suite_keys: set[str], ledger: list, t: int, examples: int = 25) -> dict[str, Any]:
    """t-wise coverage of a suite (its behaviour keys) over the feasible combinations in a ledger."""
    universe: dict[tuple, bool] = {}
    for decided, key in ledger:
        hit = key in suite_keys
        for c in combos(decided, t):
            universe[c] = universe.get(c, False) or hit
    unc = [c for c, ok in universe.items() if not ok]
    kinds = Counter(" x ".join(sorted({dp.split("#")[0] for dp, _ in c})) for c in unc)
    return {"t": t, "feasible": len(universe), "covered": len(universe) - len(unc),
            "rate": (len(universe) - len(unc)) / max(1, len(universe)),
            "uncovered_by_kind": dict(kinds.most_common(15)),
            "uncovered_examples": [[f"{dp} = {opt}" for dp, opt in c] for c in unc[:examples]]}


def select(paths: list, ledger: list, workflow, cfg: dict[str, Any]) -> tuple[list, dict[str, Any]]:
    """The suite drawn from explored behaviours under the coverage controls."""
    from .journeys import behaviour_key
    t = int(cfg.get("strength", 2))
    keyed = [(behaviour_key(tr), sp, tr) for sp, tr in paths]
    excluded = []
    for ex in cfg.get("exclude") or []:
        for k, sp, tr in keyed:
            if _matches(ex["match"], sp, tr):
                excluded.append({"case": list(sp.label) or ["default"], "reason": ex.get("reason", "")})
    drop = {k for k, sp, tr in keyed for ex in (cfg.get("exclude") or []) if _matches(ex["match"], sp, tr)}
    keyed = [x for x in keyed if x[0] not in drop]
    if cfg.get("select", "all") == "all":
        chosen = [k for k, _, _ in keyed]
    else:
        by_key: dict[str, set] = defaultdict(set)
        for decided, k in ledger:
            by_key[k] |= combos(decided, t)
        items = {k: by_key.get(k, set()) | structural_items(tr, workflow) for k, sp, tr in keyed}
        need = set().union(*items.values()) if items else set()
        chosen, have = [], set()
        default = next((k for k, sp, tr in keyed if not sp.label), None)
        if default:
            chosen.append(default)
            have |= items[default]
        while need - have:
            best = max(items, key=lambda k: (len(items[k] - have), -len(chosen)))
            gain = items[best] - have
            if not gain:
                break
            chosen.append(best)
            have |= gain
    must = []
    for pats in cfg.get("must_cover") or []:
        hit = next((k for k, sp, tr in keyed if _matches(pats, sp, tr)), None)
        must.append({"patterns": pats, "case": hit is not None})
        if hit and hit not in chosen:
            chosen.append(hit)
    cap = cfg.get("max_bases")
    capped = bool(cap and len(chosen) > cap)
    if capped:
        chosen = chosen[:cap]
    ck = set(chosen)
    suite = [(sp, tr) for k, sp, tr in keyed if k in ck]
    return suite, {"strength": t, "select": cfg.get("select", "all"), "explored_behaviours": len(paths),
                   "selected": len(suite), "excluded": excluded, "must_cover": must, "capped_at": cap if capped else None}


def ledger_report(oracle, suite: list, cfg: dict[str, Any], workflow) -> dict[str, Any]:
    """Coverage of the selected suite at every reported strength, against an exploration one deviation
    deeper than the strength (so combinations the suite's own budget never reached are counted too),
    and the size of the whole path space when it is small enough to enumerate."""
    from .journeys import behaviour_key
    keys = {behaviour_key(tr) for _, tr in suite}
    out: dict[str, Any] = {"strengths": {}}
    for t in cfg.get("ledger_strengths", [1, 2, 3]):
        _, led, st = explore(oracle, t)
        out["strengths"][str(t)] = {**measure(keys, led, t), "explored_runs": st["runs"], "explored_behaviours": st["behaviours"]}
    _, _, full = explore_limited(oracle, cfg.get("enumerate_limit", 100_000))
    out["path_space"] = full
    return out


def explore_limited(oracle, limit: int) -> tuple[list, list, dict]:
    """The whole path space (no deviation budget) when it fits in ``limit`` exploration runs."""
    from .journeys import mine_paths
    stats: dict = {}
    paths = mine_paths(oracle, max_deviations=10 ** 6, max_paths=limit, stats=stats)
    return paths, [], {"enumerated": stats.get("complete", False), "runs": stats.get("runs"),
                       "behaviours": len(paths) if stats.get("complete") else None,
                       "note": "every path the policy allows" if stats.get("complete") else
                               f"more than {limit} exploration runs: too large to enumerate; t-wise coverage is the guarantee"}
