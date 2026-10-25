"""Optimal journey base cases: constrained covering array + coverage closure.

The content dimensions (journey.DIMENSIONS) are what a base case *is*. Testing
every combination is infeasible (millions), and random sampling wastes cases on
irrelevant ones. Instead:

1. Relevance. ``relevant_dims`` collapses dimensions that cannot influence the
   ground truth of a case to N/A (after a failed authentication, the number of
   transactions does not matter). A level pair only counts as covered when both
   levels are relevant in the same case, so coverage is never inflated.
2. Feasibility. A pair is a coverage target only if some valid, relevant case
   contains it (estimated by constrained random completion); infeasible pairs
   are reported, not silently dropped.
3. Covering array (AETG-style greedy). Each new case is the best of many
   candidates, each seeded with an uncovered pair and hill-climbed to cover as
   many further uncovered pairs as possible. Turn-pressure cases are made exact
   by padding with policy questions until the oracle reports exactly 20 turns
   (completes) or 21 (must transfer at the limit).
4. Closure. The oracle reports which rules fired, which workflow edges were
   traversed and why it transferred. Cases are added until every journey-
   reachable rule, edge, trigger and boundary is exercised; anything reachable
   only through tool faults is left to the resilience probe and listed.

The result is close to the smallest case set that still exercises every
interaction of two content dimensions and every policy element.
"""
from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from ..core import QAItem
from .. import journey as JM
from ..journey import NA, JourneySpec, build_fixture, holds, is_valid, relevant_dims
from ..ontology import SemanticFrame as F

# the domain's journey model (domain.yaml -> journey)
def _dims() -> dict[str, list[Any]]:
    return JM.model().dimensions


def _turn() -> dict[str, str]:
    """The turn-pressure dimension: its name and its normal / at_limit / over_limit levels."""
    return JM.model().cfg["turn_dim"]


def __getattr__(name: str):                       # backward-compatible module constants
    if name.startswith("__"):
        raise AttributeError(name)
    cfg = JM.model().cfg
    if name == "PADDING_POOL":
        return [JM.model().frame(f) for f in cfg.get("padding", [])]
    if name == "FAULT_ONLY":
        return set(cfg.get("fault_only_rules", []))
    if name == "BOUNDARIES":
        return set(cfg.get("boundaries", []))
    raise AttributeError(name)


@dataclass
class CoverageReport:
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
        setattr(self._long_oracle.limits, _turn()["limit"], 10_000)

    def _turn_limit(self) -> int:
        return getattr(self.oracle.limits, _turn()["limit"])

    @staticmethod
    def _turn_limit_reason() -> str:
        return _turn()["reason"]

    # -------------------------------------------------------------- specs
    def random_spec(self, fixed: dict[str, Any] | None = None) -> JourneySpec:
        for _ in range(200):
            d = {k: self.rng.choice(v) for k, v in _dims().items()}
            d.update(fixed or {})
            spec = JourneySpec(**d)
            if is_valid(spec):
                return spec
        return JourneySpec(**{**JourneySpec().dims(), **(fixed or {})})

    def finalize(self, spec: JourneySpec) -> JourneySpec | None:
        """Attach KB questions and turn padding; return None if the turn target is infeasible."""
        rel = relevant_dims(spec)
        default = JourneySpec()
        spec = spec.with_(**{d: getattr(default, d) for d in _dims() if rel[d] == NA})   # canonical form
        kb: list[F] = []
        q = JM.model().cfg["question_dim"]
        if rel.get(q["name"]) == q["ask"] and self.kb_pool:
            kb.append(self.kb_pool[self._kb_i % len(self.kb_pool)])
            self._kb_i += 1
        spec = spec.with_(kb_frames=tuple(kb), extra_kb=0)
        T = _turn()
        tp = rel.get(T["name"], NA)
        if tp in (NA, T["normal"]):
            if tp == NA and getattr(spec, T["name"]) != T["normal"]:
                spec = spec.with_(**{T["name"]: T["normal"]})
            return spec
        natural = self._long_oracle.run(spec).turns
        target = self._turn_limit() + (0 if tp == T["at_limit"] else 1)
        extra = target - natural
        if extra < 0:
            return None
        pool = [JM.model().frame(f) for f in JM.model().cfg.get("padding", [])]
        pad = tuple(pool[i % len(pool)] for i in range(extra))
        spec = spec.with_(kb_frames=tuple(kb) + pad, extra_kb=extra)
        tr = self.oracle.run(spec)
        reason = self._turn_limit_reason()
        if tp == T["at_limit"] and (tr.turns != target or tr.transfer_reason == reason):
            return None
        if tp == T["over_limit"] and tr.transfer_reason != reason:
            return None
        return spec

    # -------------------------------------------------------------- feasibility
    def feasible_pairs(self, tries: int = 150) -> tuple[set[tuple], set[tuple]]:
        allp = set()
        D = _dims()
        dims = list(D)
        for a, b in itertools.combinations(dims, 2):
            for va in D[a]:
                for vb in D[b]:
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
        if self.max_turn_limit_cases is not None:          # pairs over normal-length conversations only
            T, D = _turn(), _dims()
            saved = D[T["name"]]
            D[T["name"]] = [T["normal"]]
            try:
                return self._covering(candidates, max_rows)
            finally:
                D[T["name"]] = saved
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
                dims = list(_dims())
                self.rng.shuffle(dims)
                for d in dims:                     # hill-climb one dimension at a time
                    if d in dict(seed_pair):
                        continue
                    for v in _dims()[d]:
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
                fin = self.finalize(best.with_(**{_turn()["name"]: _turn()["normal"]}))
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
        cfg = JM.model().cfg
        boundaries = set(cfg.get("boundaries", []))
        all_rules = set(store.rules) - set(cfg.get("fault_only_rules", []))
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
                b |= {f"{k}={v}" for k, v in rel.items() if f"{k}={v}" in boundaries}
            for tr in traces:
                r |= tr.rules_fired
                e |= tr.edges(store.workflow)[0]
            return r, e, t, b

        r, e, t, b = goals(rows, extra_traces)
        added: list[JourneySpec] = []
        tries = 0
        cap = self.max_turn_limit_cases
        T = _turn()
        padded = lambda sp: getattr(sp, T["name"]) in (T["at_limit"], T["over_limit"])
        n_padded = sum(padded(s) for s in rows)
        while ((all_rules - r) or (all_edges - e) or (all_triggers - t) or (boundaries - b)) and tries < budget:
            tries += 1
            fixed = {T["name"]: T["normal"]} if cap is not None and n_padded >= cap else None
            spec = self.finalize(self.random_spec(fixed))
            if spec is None:
                continue
            r2, e2, t2, b2 = goals([spec])
            if cap is not None and padded(spec) and not ({x for x in b2 - b if x.startswith(T["name"])}
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
    """The base category a journey's distinguishing content exercises (domain.yaml ->
    journey.categories, first match, read on the relevant dimensions)."""
    rel = relevant_dims(spec)
    for c in JM.model().cfg["categories"]:
        if holds(c.get("when"), rel):
            return c["category"]
    raise ValueError("journey.categories: no category matched (add a final one without 'when')")


def tags_for(spec: JourneySpec) -> list[str]:
    """Every base category a journey exercises (domain.yaml -> journey.tags), plus its primary one."""
    rel = relevant_dims(spec)
    tags = {primary_category(spec)} | {t["tag"] for t in JM.model().cfg.get("tags", []) if holds(t.get("when"), rel)}
    return sorted(tags)


def to_item(spec: JourneySpec, oracle, qid: str, category: str | None = None, kb_item=None) -> QAItem:
    fx = build_fixture(spec)
    tr = oracle.run(spec, fx)
    comps = {f"expected_{k}": v for k, v in tr.components().items()}
    from ..enrich.realizers import Realizer
    R = Realizer()
    first = tr.user_frames[0] if tr.user_frames else F.make("UNCLEAR")
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
