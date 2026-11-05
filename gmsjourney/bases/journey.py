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
from ..flow import evaluate
from ..journey import DIMENSIONS, NA, JourneySpec, build_fixture, is_valid, relevant_dims
from ..journey import primary_category as _category
from ..ontology import SemanticFrame as F
from ..project import CaseView, project
from ..names import N

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
    fx = build_fixture(spec)
    tr = oracle.run(spec, fx)
    comps = {f"expected_{k}": v for k, v in tr.components().items()}
    from ..enrich.realizers import Realizer
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


from ..domain import on_switch as _on_switch  # noqa: E402

_on_switch(_activate)
