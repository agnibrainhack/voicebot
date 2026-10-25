"""Base-case factors for a compact policy graph: you choose the dimensions, the graph decides the truth.

``base_factors.yaml`` (in the pack) declares the content factors of a base case and their levels.
A level is a small script in the graph's own terms:

  tools:   {tool: [result, ...]}        what the tool returns on its 1st, 2nd, ... call (a status, or a
                                        result object such as {status: NOT_ELIGIBLE, reason: ...});
                                        the last result repeats for later calls
  says:    {PROMPT_STATE: [frame, ...]} what the customer means on the 1st, 2nd, ... visit of that
                                        prompt, written ACT(slot=VALUE, ...) (ASK_KB = a policy question
                                        drawn from the mined pool)
  before:  true                         the frames go before the prompt's answer (an interruption)
                                        instead of being the answer
  answer:  {PROMPT_STATE: {slot: value}} a slot of the prompt's (default) answer; the answer's other
                                        slots come from other factors or the default. ``@other``
                                        = another value than the answer's own (a correction)
  pad:     {state: S, to: limit | over} policy questions before the answer at S until the conversation
                                        reaches the turn limit exactly, or one turn over

Each factor names its ``anchor`` (a tool or a prompt state): the decision point it is about. The
default level (the first) is the default path there.

A case is one level per factor; levels are merged into one script (two factors scripting the same
tool or the same prompt answer cannot combine). The oracle plays it, and a level is *relevant* in
that case only if its script was consumed (the anchor was reached): an unreached level is N/A, as in
the v1 ``relevant_dims``. Two levels that are never relevant together are an infeasible pair.

``generate`` builds a covering array of strength t (default 2) over the factors, counting a
combination as covered only where all its levels are relevant, then closes it on the graph: cases
from the oracle's exploration are added for any state, workflow edge, rule or transfer reason the
array misses. The ledger reports feasible / covered / infeasible combinations and the closure.

``python -m gmsjourney.graph.factors draft <pack>`` writes a starting base_factors.yaml mined from the
graph's decision points, to rename, merge, prune or extend.
"""
from __future__ import annotations

import random
import re
from itertools import combinations
from typing import Any

import yaml

from ..ontology import SemanticFrame as F
from .journeys import GraphSpec, _tool_options, default_frame

FILE = "base_factors.yaml"
_FRAME = re.compile(r"^\s*([A-Z_]+)\s*(?:\((.*)\))?\s*$")


def _graph():
    from ..domain import current
    return current().graph


# ============================================================================ frames
def parse_frame(s: str | F) -> F:
    if isinstance(s, F):
        return s
    m = _FRAME.match(str(s))
    if not m:
        raise ValueError(f"not a frame: {s!r} (write ACT or ACT(slot=VALUE, ...))")
    act, body = m.group(1), m.group(2) or ""
    kw = {}
    for part in [p.strip() for p in body.split(",") if p.strip()]:
        k, _, v = part.partition("=")
        kw[k.strip()] = v.strip().strip("'\"")
    return F.make(act, **kw)


def frame_text(f: F) -> str:
    return f.act + ("(" + ", ".join(f"{k}={v}" for k, v in f.slots) + ")" if f.slots else "")


# ============================================================================ loading
def load(dom=None) -> dict[str, Any] | None:
    from ..domain import current
    dom = dom or current()
    p = dom.dir / FILE
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text()) or {}


def check(cfg: dict[str, Any]) -> list[str]:
    """Every level is a script in the graph's terms."""
    g = _graph()
    errs = []
    for name, fac in (cfg.get("factors") or {}).items():
        a = fac.get("anchor") or {}
        if a.get("tool") and a["tool"] not in g.tools:
            errs.append(f"base_factors.{name}.anchor: {a['tool']!r} is not a tool")
        if a.get("state") and a["state"] not in g.states:
            errs.append(f"base_factors.{name}.anchor: {a['state']!r} is not a state")
        if not fac.get("levels"):
            errs.append(f"base_factors.{name}: no levels")
        for lv, spec in (fac.get("levels") or {}).items():
            spec = spec or {}
            for t, rs in (spec.get("tools") or {}).items():
                if t not in g.tools:
                    errs.append(f"base_factors.{name}.{lv}: {t!r} is not a tool")
                    continue
                ok = {g.tools[t]["ok"], *g.tools[t].get("fail", []), "UNAVAILABLE"}
                for r in rs:
                    st = r if isinstance(r, str) else r.get("status")
                    if st not in ok:
                        errs.append(f"base_factors.{name}.{lv}: {st!r} is not a status of {t}")
            for st, frames in (spec.get("says") or {}).items():
                if st not in g.prompt_states:
                    errs.append(f"base_factors.{name}.{lv}: {st!r} is not a prompt state")
                    continue
                for fr in frames:
                    try:
                        f = parse_frame(fr)
                    except ValueError as e:
                        errs.append(f"base_factors.{name}.{lv}: {e}")
                        continue
                    if f.act not in g.acts and f.act != "ASK_KB":
                        errs.append(f"base_factors.{name}.{lv}: {f.act!r} is not a customer act")
            if spec.get("pad") and spec["pad"].get("state") not in g.prompt_states:
                errs.append(f"base_factors.{name}.{lv}.pad: {spec['pad'].get('state')!r} is not a prompt state")
    return errs


# ============================================================================ cases
class Case:
    """One assignment of a level to every factor, compiled into a GraphSpec."""

    def __init__(self, cfg: dict[str, Any], levels: dict[str, str], kb_pool: list[F]):
        self.cfg, self.levels = cfg, dict(levels)
        self.kb_pool = kb_pool
        self.ok, self.why = True, ""
        self.tools: dict[str, list] = {}
        self.says: dict[str, list[F]] = {}
        self.marks: dict[str, list[tuple[str, int]]] = {}   # factor -> [(tool:<t> | state:<S>, position)]
        before: dict[str, list[tuple[str, list[F]]]] = {}
        answer: dict[str, dict[str, Any]] = {}
        answer_facs: dict[str, list[str]] = {}
        pads = []
        kb_i = 0
        for fac, lv in self.levels.items():
            spec = (cfg["factors"][fac]["levels"].get(lv) or {})
            for st, slots in (spec.get("answer") or {}).items():
                cur = answer.setdefault(st, {})
                for k, v in slots.items():
                    if k in cur and cur[k] != v:
                        self.ok, self.why = False, f"{fac} sets {k}={v} at {st}, another factor {cur[k]}"
                    cur[k] = v
                answer_facs.setdefault(st, []).append(fac)
            for t, rs in (spec.get("tools") or {}).items():
                if t in self.tools:
                    self.ok, self.why = False, f"{fac} and another factor both script {t}"
                self.tools[t] = [({"status": r} if isinstance(r, str) else dict(r)) for r in rs]
                self.marks.setdefault(fac, []).append((f"tool:{t}", len(rs)))
            for st, frames in (spec.get("says") or {}).items():
                fr = []
                for x in frames:
                    f = parse_frame(x)
                    if f.act == "ASK_KB" and not f.slots and kb_pool:
                        f = kb_pool[kb_i % len(kb_pool)]
                        kb_i += 1
                    fr.append(f)
                if spec.get("before"):
                    before.setdefault(st, []).append((fac, fr))
                else:
                    if st in self.says:
                        self.ok, self.why = False, f"{fac} and another factor both script the answer at {st}"
                    self.says[st] = fr
                    self.marks.setdefault(fac, []).append((f"answer:{st}", len(fr)))
            if spec.get("pad"):
                pads.append((fac, spec["pad"]))
        g = _graph()
        for st, slots in answer.items():           # the default act's answer, assembled from slot factors
            if st in self.says:
                continue                           # another factor chose a different answer: these do not apply
            f0 = default_frame(st)
            kw = {**dict(f0.slots), **slots}
            for k, v in list(kw.items()):
                if v == "@other":
                    sp = g.slot_specs(f0.act).get(k) or {}
                    ref = kw.get(sp.get("correction_of"))
                    kw[k] = next((m for m in g.members(sp.get("set")) if m != ref), None)
            for k, sp in g.slot_specs(f0.act).items():
                for need, vals in (sp.get("with") or {}).items():
                    if kw.get(k) is not None and kw.get(need) not in vals:
                        self.ok, self.why = False, f"{k} at {st} needs {need} in {vals}"
            self.says[st] = [F.make(f0.act, **kw)]
            for fac in answer_facs[st]:
                self.marks.setdefault(fac, []).append((f"answer:{st}", 1))
        for st, items in before.items():           # interruptions first, in factor order, then the answer
            pre = []
            for fac, fr in items:
                self.marks.setdefault(fac, []).append((f"before:{st}:{len(pre)}", len(fr)))
                pre += fr
            self.says[st] = pre + self.says.get(st, [default_frame(st)])
        self.pads = pads

    def spec(self, extra_kb: int = 0) -> GraphSpec:
        tools = {**self.defaults, **self.tools}
        return GraphSpec(self.says, tools, label=tuple(f"{k}={v}" for k, v in self.levels.items()), extra_kb=extra_kb)

    def build(self, oracle) -> tuple[GraphSpec, Any]:
        self.defaults = tool_defaults(oracle)
        spec = self.spec()
        tr = oracle.run(spec)
        for fac, pad in self.pads:
            lim = oracle.L((self.cfg.get("turn_limit") or {}).get("limit") or
                           (_graph().glob.get("turn_limit") or {}).get("limit"))
            n = lim - tr.turns + (1 if pad.get("to") == "over" else 0)
            if n > 0:
                pre = [self.kb_pool[i % len(self.kb_pool)] for i in range(n)]
                st = pad["state"]
                self.says[st] = pre + self.says.get(st, [default_frame(st)])
                self.marks.setdefault(fac, []).append((f"before:{st}:0", n))
                spec = self.spec(extra_kb=n)
                tr = oracle.run(spec)
        self.trace, self.built = tr, spec
        return spec, tr

    def category(self) -> str:
        """The base category: the first interruption that applies, else the first non-default factor."""
        rel = getattr(self, "rel", {})
        facs = self.cfg["factors"]
        firsts = {f: next(iter(facs[f]["levels"])) for f in facs}
        moved = [f for f in self.levels if rel.get(f) and self.levels[f] != firsts[f]]
        for f in moved:
            if not (facs[f].get("anchor") or {}):
                return f
        return moved[0] if moved else "default_path"

    def relevant(self, oracle) -> dict[str, bool]:
        """Was each factor's level consumed (its anchor reached) in this case?"""
        tr, sim = self.trace, oracle._sim
        calls: dict[str, int] = {}
        for c in tr.tool_log:
            calls[c["tool"]] = calls.get(c["tool"], 0) + 1
        out = {}
        for fac in self.levels:
            marks = self.marks.get(fac)
            if not marks:                                   # the default level: relevant if the anchor is reached
                a = self.cfg["factors"][fac].get("anchor") or {}
                if a.get("tool"):
                    out[fac] = calls.get(a["tool"], 0) > 0
                elif a.get("state"):
                    out[fac] = sim.consumed.get(a["state"], 0) > 0
                else:
                    out[fac] = True
                continue
            ok = True
            for key, n in marks:
                kind, _, rest = key.partition(":")
                if kind == "tool":
                    ok &= calls.get(rest, 0) >= 1
                elif kind == "answer":
                    st = rest
                    pre = len(self.says.get(st, [])) - n
                    ok &= sim.consumed.get(st, 0) > pre
                else:
                    st, _, pos = rest.rpartition(":")
                    ok &= sim.consumed.get(st, 0) > int(pos)
            out[fac] = ok
        return out


_DEFAULTS: dict[int, dict] = {}


def tool_defaults(oracle) -> dict[str, list[dict]]:
    """What every tool returns on the default path (the miner's first option: success, one item, all options)."""
    if id(oracle) not in _DEFAULTS:
        g = _graph()
        _DEFAULTS[id(oracle)] = {t: [_tool_options(t, {}, oracle.L)[0][0]] for t in g.tools}
    return _DEFAULTS[id(oracle)]


# ============================================================================ generation
def generate(oracle, cfg: dict[str, Any], kb_pool: list[F], seed: int = 42) -> tuple[list[Case], dict[str, Any]]:
    """A covering array of strength t over the factors (relevance-aware), then compacted."""
    rng = random.Random(seed)
    facs = list(cfg["factors"])
    levels = {f: list(cfg["factors"][f]["levels"]) for f in facs}
    t = int(cfg.get("strength", 2))
    default = {f: levels[f][0] for f in facs}
    memo: dict[tuple, Case | None] = {}

    def play(assign: dict[str, str]) -> Case | None:
        key = tuple(sorted(assign.items()))
        if key not in memo:
            c = Case(cfg, assign, kb_pool)
            if c.ok:
                c.build(oracle)
                c.rel = c.relevant(oracle)
                memo[key] = c
            else:
                memo[key] = None
        return memo[key]

    def tuples_of(c: Case) -> set[tuple]:
        rel = [(f, c.levels[f]) for f in facs if c.rel.get(f)]
        return set(combinations(rel, t))

    # enablers: the levels of other factors (one, else two) under which a level becomes relevant
    base = play(default)
    enablers: dict[tuple, list[tuple] | None] = {}
    reachable = []                                         # non-default levels relevant on the default path
    for f in facs:
        for lv in levels[f][1:]:
            c = play({**default, f: lv})
            if c is not None and c.rel.get(f):
                reachable.append((f, lv))
    for f in facs:
        for lv in levels[f]:
            c = play({**default, f: lv})
            if c is not None and c.rel.get(f):
                enablers[(f, lv)] = [()]
                continue
            en = []
            for g2 in facs:
                if g2 == f:
                    continue
                for l2 in levels[g2][1:]:
                    c2 = play({**default, f: lv, g2: l2})
                    if c2 is not None and c2.rel.get(f) and c2.rel.get(g2):
                        en.append(((g2, l2),))
            if not en:                                     # two enablers, from levels reachable on the default path
                for (g2, l2), (g3, l3) in combinations(reachable, 2):
                    if len({f, g2, g3}) < 3:
                        continue
                    c3 = play({**default, f: lv, g2: l2, g3: l3})
                    if c3 is not None and c3.rel.get(f) and c3.rel.get(g2) and c3.rel.get(g3):
                        en.append(((g2, l2), (g3, l3)))
                        if len(en) >= 6:
                            break
            enablers[(f, lv)] = en or None                 # None: never relevant

    # the feasible universe, each tuple with a witness assignment that makes all its levels relevant
    universe: dict[tuple, dict[str, str]] = {}
    infeasible = []
    for fs in combinations(facs, t):
        for lvs in _product([levels[f] for f in fs]):
            tup = tuple(zip(fs, lvs))
            witness = None
            options = []
            for m in tup:
                en = enablers.get(m)
                if en is None:
                    options = None
                    break
                options.append(([()] if () in en else []) + [e for e in en if e][:8])
            if options is not None:
                for combo in _product(options):
                    extra = dict(x for part in combo for x in part)
                    if any(f in extra for f in fs) or len(extra) != sum(len(p) for p in combo):
                        continue
                    a = {**default, **extra, **dict(tup)}
                    c = play(a)
                    if c is not None and all(c.rel.get(f) for f in fs):
                        witness = a
                        break
            if witness is None:
                infeasible.append(tup)
            else:
                universe[tup] = witness
    U = set(universe)
    cases: list[Case] = [base]
    covered = tuples_of(base) & U
    tries = 0
    while U - covered and tries < 20 * len(U):
        tries += 1
        seed_t = sorted(U - covered)[0]
        assign = dict(universe[seed_t])
        fixed = set(dict(seed_t)) | {f for f in facs if assign[f] != default[f]}
        order = [f for f in facs if f not in fixed]
        rng.shuffle(order)
        cur = play(assign)
        cur_gain = len((tuples_of(cur) & U) - covered) if cur is not None else -1
        for f in order:                                   # AETG fill, scored by playing the case with the oracle
            for lv in levels[f]:
                if lv == assign[f]:
                    continue
                trial = {**assign, f: lv}
                c = play(trial)
                if c is None or not all(c.rel.get(ff) for ff, _ in seed_t):
                    continue
                gn = len((tuples_of(c) & U) - covered)
                if gn > cur_gain:
                    assign, cur_gain = trial, gn
        c = play(assign)
        new = (tuples_of(c) & U) - covered if c is not None else set()
        if seed_t not in new:                             # the fill blocked the seed: its witness alone
            c = play(universe[seed_t])
            new = (tuples_of(c) & U) - covered if c is not None else set()
        if new:
            cases.append(c)
            covered |= new
    # compaction: drop a case whose tuples the others cover too (the default case stays)
    own = [tuples_of(c) & U for c in cases]
    keep = list(range(len(cases)))
    for i in sorted(range(1, len(cases)), key=lambda i: len(own[i])):
        rest = set().union(*(own[j] for j in keep if j != i))
        if own[i] <= rest:
            keep.remove(i)
    cases = [cases[i] for i in keep]
    led = {"strength": t, "factors": {f: len(levels[f]) for f in facs},
           "feasible": len(U), "covered": len(covered & U), "infeasible": len(infeasible),
           "uncovered": [[f"{f}={l}" for f, l in tup] for tup in sorted(U - covered)][:50],
           "infeasible_examples": [[f"{f}={l}" for f, l in tup] for tup in infeasible[:25]],
           "never_relevant_levels": [f"{f}={l}" for (f, l), en in enablers.items() if en is None],
           "array_cases": len(cases), "oracle_runs": len(memo)}
    return cases, led


def _product(lists: list[list]) -> list[tuple]:
    out = [()]
    for xs in lists:
        out = [o + (x,) for o in out for x in xs]
    return out


def closure(oracle, cases: list[Case], budget: int = 3) -> tuple[list[tuple[GraphSpec, Any]], dict[str, Any]]:
    """Cases from the oracle's exploration for every state, edge, rule or transfer reason the array misses."""
    from .coverage import structural_items
    from .journeys import mine_paths
    wf = oracle.store.workflow
    have = set().union(*(structural_items(c.trace, wf) for c in cases))
    every = mine_paths(oracle, max_deviations=budget)
    reach = set().union(*(structural_items(tr, wf) for _, tr in every))
    out = []
    for sp, tr in every:
        new = structural_items(tr, wf) - have
        if new:
            out.append((sp, tr))
            have |= new
    missing = reach - have
    return out, {"added": len(out), "reachable_items": len(reach), "covered_items": len(reach - missing),
                 "unreached": sorted(f"{k}:{v}" for k, v in missing)}


# ============================================================================ the draft
def draft(oracle) -> dict[str, Any]:
    """A starting base_factors.yaml from the graph's decision points."""
    g = _graph()
    facs: dict[str, Any] = {}
    # tool outcomes (the first call; a retried tool gets its retry sequences)
    for name, s in g.states.items():
        if g.kind(name) != "call":
            continue
        t = s["call"]
        if t in facs or g.tools[t]["role"] == "handoff":
            continue
        lv: dict[str, Any] = {"ok": {}}
        for res, w, lab in _tool_options(t, {}, oracle.L)[1:]:
            key = _slug(lab.split("=", 1)[-1] if "=" in lab else lab.split(":", 1)[-1])
            lv[key] = {"tools": {t: [res if len(res) > 1 else res["status"]]}}
        retry = next((v for v in s["on"].values() if isinstance(v, dict) and "below" in v), None)
        if retry:                                       # a retried tool: its failure only as a sequence
            for st_ in g.tools[t].get("fail", []):
                lv.pop(_slug(st_), None)
            L = oracle.L(retry["below"])
            fail = (g.tools[t].get("fail") or ["FAILED"])[0]
            lv[f"retry_{_slug(g.tools[t]['ok'])}"] = {"tools": {t: [fail] * (L - 1) + [g.tools[t]["ok"]]}}
            lv[f"{_slug(fail)}_{L}x"] = {"tools": {t: [fail] * L}}
        facs[t] = {"anchor": {"tool": t}, "description": g.tools[t].get("text", ""), "levels": lv}
    # what the customer says at each prompt (the first visit): which act, then one factor per slot of
    # the default act (a value per required slot, none or a value per optional slot, a correction)
    for st in g.prompt_states:
        on = list(g.states[st]["on"])
        f0 = default_frame(st)
        s = st.lower()
        if len(on) > 1:
            lv = {_slug(f0.act): {}}
            for act in on[1:]:
                specs = g.slot_specs(act)
                req = [(k, sp) for k, sp in specs.items() if not sp.get("optional")]
                if not req:
                    lv[_slug(act)] = {"says": {st: [act]}}
                    continue
                k, sp = req[0]
                for v in g.members(sp["set"]):
                    lv[_slug(f"{act}_{v}")] = {"says": {st: [frame_text(F.make(act, **{k: v}))]}}
            facs[s] = {"anchor": {"state": st}, "description": f"what the customer does at {st}", "levels": lv}
        for k, sp in g.slot_specs(f0.act).items():
            vs = sp["set"]
            if vs == "credential_code":
                continue
            name = f"{s}_{k}"
            if not sp.get("optional"):
                vals = g.members(vs)
                if len(vals) < 2:
                    continue
                lv = {_slug(vals[0]): {}}
                lv.update({_slug(v): {"answer": {st: {k: v}}} for v in vals[1:]})
                facs[name] = {"anchor": {"state": st}, "description": f"{k} the customer gives at {st}", "levels": lv}
            elif sp.get("correction_of"):
                facs[f"{s}_correction"] = {"anchor": {"state": st}, "description": f"the customer corrects {sp['correction_of']} "
                                           f"within the answer at {st}",
                                           "levels": {"none": {}, "corrected": {"answer": {st: {k: "@other"}}}}}
            else:
                lv = {"none": {}}
                for v in g.members(vs):
                    extra = {need: vals[0] for need, vals in (sp.get("with") or {}).items()}
                    lv[_slug(v)] = {"answer": {st: {**extra, k: v}}}
                facs[name] = {"anchor": {"state": st}, "description": f"{k} the customer adds at {st}", "levels": lv}
    # what the customer does at a prompt besides answering: one factor per behaviour, so a case can combine them
    G = g.glob
    if G.get("human"):
        facs["human_request"] = {"anchor": {}, "description": "the customer asks for a person at a prompt",
                                 "levels": {"none": {}, **{f"at_{st.lower()}": {"says": {st: [G["human"]["act"]]}, "before": True}
                                                           for st in g.prompt_states}}}
    if G.get("unclear"):
        L = oracle.L(G["unclear"]["limit"])
        lv = {"none": {}}
        for st in g.prompt_states:
            lv[f"once_at_{st.lower()}"] = {"says": {st: [G["unclear"]["act"]] * L}, "before": True}
            lv[f"persists_at_{st.lower()}"] = {"says": {st: [G["unclear"]["act"]] * (L + 1)}, "before": True}
        facs["unclear_answer"] = {"anchor": {}, "description": "an answer that carries no decision, once (recovered) "
                                                               "or past the clarification limit", "levels": lv}
    if G.get("question"):
        facs["policy_question"] = {"anchor": {}, "description": "a policy question asked before answering a prompt",
                                   "levels": {"none": {}, **{f"at_{st.lower()}": {"says": {st: ["ASK_KB"]}, "before": True}
                                                             for st in g.prompt_states}}}
    if G.get("turn_limit"):
        st = g.prompt_states[min(1, len(g.prompt_states) - 1)]
        facs["turn_pressure"] = {"anchor": {}, "description": "the conversation runs into the turn limit",
                                 "levels": {"normal": {}, "at_limit": {"pad": {"state": st, "to": "limit"}},
                                            "over_limit": {"pad": {"state": st, "to": "over"}}}}
    return {"strength": 2, "closure_budget": 3, "factors": facs}


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()
    return s or "x"


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Draft or check base_factors.yaml for a compact policy graph pack.")
    ap.add_argument("command", choices=["draft", "check"])
    ap.add_argument("domain")
    ap.add_argument("--force", action="store_true", help="overwrite an existing base_factors.yaml")
    a = ap.parse_args(argv)
    from .. import domain
    dom = domain.set_current(a.domain)
    if dom.graph is None:
        raise SystemExit(f"{dom.name} is not a compact policy graph pack")
    from .compile import compile_graph
    from .machine import GraphOracle
    oracle = GraphOracle(compile_graph(dom.graph_path).store)
    p = dom.dir / FILE
    if a.command == "draft":
        if p.exists() and not a.force:
            raise SystemExit(f"{p} exists (use --force to overwrite)")
        d = draft(oracle)
        head = (f"# Base-case factors for {dom.name}, drafted from {dom.graph_path.name}.\n"
                "# Rename, merge, drop or add levels; the first level of each factor is its default.\n"
                "# Level scripts: tools {tool: [results]}, says {PROMPT: [frames]}, before: true, pad.\n"
                "# Run `python -m gmsjourney.graph.factors check <pack>` after editing.\n\n")
        p.write_text(head + yaml.safe_dump(d, sort_keys=False, width=140, allow_unicode=True, default_flow_style=None))
        print(f"wrote {p} ({len(d['factors'])} factors, {sum(len(f['levels']) for f in d['factors'].values())} levels)")
        return 0
    errs = check(load(dom) or {})
    print(f"{p.name}: " + ("clean" if not errs else f"{len(errs)} problem(s)"))
    for e in errs:
        print("  -", e)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
