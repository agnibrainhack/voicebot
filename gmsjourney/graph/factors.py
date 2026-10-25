"""Base-case factors for a compact policy graph: you choose the dimensions, the graph decides the truth.

``base_factors.yaml`` (in the pack) declares the content factors of a base case and their levels.
A level is a small script in the graph's own terms:

  tools:   {tool: [result, ...]}        what the tool returns on its 1st, 2nd, ... call (a status, or a
                                        result object such as {status: NOT_ELIGIBLE, reason: ...});
                                        the last result repeats for later calls; "@other" as the echoed
                                        field = a value other than the one the bot sent
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

A level may also carry ``expect``: the outcome the policy owner expects when that level happens with
every other factor at its default (end, transfer, writes, no_writes, tells, never_tells). ``review``
plays each level and reports where the graph disagrees, next to the coverage checks; ``apply_edits``
changes the file from a list of edits (drop, rename, add, expect, default, strength).

``python -m gmsjourney.graph.factors draft|check|review <pack>``: write a starting base_factors.yaml
mined from the graph's decision points, check it, or run every base-journey check on it.
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
        echo = g.tools[t].get("echo")
        for res, w, lab in _tool_options(t, {}, oracle.L)[1:]:
            key = _slug(lab.split("=", 1)[-1] if "=" in lab else lab.split(":", 1)[-1])
            if echo and lab.startswith(f"{t}=ECHO_MISMATCH"):     # whatever the bot sent, a different value comes back
                key, res = "echo_mismatch", {**res, echo: "@other"}
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


# ============================================================================ review and edits
END_VALUES = ("ENDED", "TRANSFERRED", "INCOMPLETE")      # how a conversation ends
EXPECT_KEYS = ("end", "transfer", "writes", "no_writes", "tells", "never_tells")


def outcome(tr) -> dict[str, Any]:
    """What a played case comes to, in the terms an ``expect`` block uses."""
    return {"end": tr.end_state, "transfer": tr.transfer_reason or None,
            "writes": [t for t, _ in tr.writes()], "tells": sorted({a["act"] for a in tr.told})}


def expect_problems(exp: dict[str, Any], out: dict[str, Any]) -> list[str]:
    """Where a played case departs from a level's ``expect`` block (empty = as expected)."""
    bad = []
    if "end" in exp and out["end"] != exp["end"]:
        bad.append(f"ends at {out['end']}, expected {exp['end']}")
    if "transfer" in exp and out["transfer"] != exp["transfer"]:
        bad.append(f"transfer {out['transfer']!r}, expected {exp['transfer']!r}")
    for t in exp.get("writes") or []:
        if t not in out["writes"]:
            bad.append(f"{t} never succeeds")
    if exp.get("no_writes") and out["writes"]:
        bad.append(f"writes {out['writes']}, expected none")
    for a in exp.get("tells") or []:
        if a not in out["tells"]:
            bad.append(f"never says {a}")
    for a in exp.get("never_tells") or []:
        if a in out["tells"]:
            bad.append(f"says {a}")
    return bad


def _check_expect(cfg: dict[str, Any]) -> list[str]:
    g = _graph()
    errs = []
    for name, fac in (cfg.get("factors") or {}).items():
        for lv, spec in (fac.get("levels") or {}).items():
            exp = (spec or {}).get("expect")
            if exp is None:
                continue
            where = f"base_factors.{name}.{lv}.expect"
            for k in exp:
                if k not in EXPECT_KEYS:
                    errs.append(f"{where}: unknown key {k!r} (use {', '.join(EXPECT_KEYS)})")
            if exp.get("end") and exp["end"] not in END_VALUES:
                errs.append(f"{where}.end: {exp['end']!r} is not one of {', '.join(END_VALUES)}")
            if exp.get("transfer") and exp["transfer"] not in g.transfers["reasons"]:
                errs.append(f"{where}.transfer: {exp['transfer']!r} is not a transfer reason")
            for t in exp.get("writes") or []:
                if t not in g.tools:
                    errs.append(f"{where}.writes: {t!r} is not a tool")
    return errs


def representative(cfg: dict[str, Any], cases: list["Case"], fac: str, lv: str, oracle, kb_pool) -> "Case | None":
    """The case that shows what one level does: that level with every other factor at its default, or,
    when the level is not reached that way, the array case reaching it with the fewest other changes."""
    facs = cfg["factors"]
    default = {f: next(iter(facs[f]["levels"])) for f in facs}

    def play(extra: dict[str, str]):
        c = Case(cfg, {**default, **extra, fac: lv}, kb_pool)
        if not c.ok:
            return None
        c.build(oracle)
        c.rel = c.relevant(oracle)
        return c if c.rel.get(fac) else None

    c = play({})
    if c:
        return c
    hits = [x for x in cases if x.levels.get(fac) == lv and x.rel.get(fac)]
    best = min(hits, key=lambda x: sum(x.levels[f] != default[f] for f in facs), default=None)
    if best is None:
        return None
    moved = [(f, best.levels[f]) for f in facs if f != fac and best.levels[f] != default[f]]
    for k in (1, 2, 3):                      # the fewest of that case's other changes that still reach the level
        for sub in combinations(moved, k):
            c = play(dict(sub))
            if c:
                return c
    return best


def review(oracle, cfg: dict[str, Any], kb_pool: list[F]) -> dict[str, Any]:
    """Every check the base journeys must pass, and what each level leads to, for a person to read."""
    lim = oracle.limits.as_dict()
    errs = check(cfg) + _check_expect(cfg)
    if errs:
        return {"ok": False, "checks": [{"check": "factor file is valid", "ok": False, "detail": errs}],
                "levels": [], "cases": [], "ledger": {}, "closure": {}}
    cases, led = generate(oracle, cfg, kb_pool)
    extra, cl = closure(oracle, cases, cfg.get("closure_budget", 3))
    wf = oracle.store.workflow if hasattr(oracle, "store") else None
    facs = cfg["factors"]
    default = {f: next(iter(facs[f]["levels"])) for f in facs}
    rows, unmet = [], []
    for f in facs:
        for lv, spec in facs[f]["levels"].items():
            rep = representative(cfg, cases, f, lv, oracle, kb_pool)
            out = outcome(rep.trace) if rep else {}
            exp = (spec or {}).get("expect")
            bad = expect_problems(exp, out) if (exp is not None and rep) else []
            if exp is not None and rep is None:
                bad = ["the level is never reached, so its expectation cannot be checked"]
            if bad:
                unmet.append(f"{f}={lv}: " + "; ".join(bad))
            rows.append({"factor": f, "level": lv, "default": lv == default[f],
                         "needs": ", ".join(f"{k}={v}" for k, v in (rep.levels.items() if rep else [])
                                            if k != f and v != default[k]),
                         **{k: (", ".join(v) if isinstance(v, list) else v) for k, v in out.items()},
                         "expected": "" if exp is None else ("as expected" if not bad else "; ".join(bad))})
    inadm = [(c, c.trace.admissibility(wf, lim)) for c in cases] if wf is not None else []
    inadm = [(c, a) for c, a in inadm if a]
    n_exp = sum(1 for f in facs for s in facs[f]["levels"].values() if (s or {}).get("expect") is not None)
    checks = [
        {"check": "factor file is valid", "ok": True, "detail": []},
        {"check": "every level is reached in some case", "ok": not led["never_relevant_levels"],
         "detail": led["never_relevant_levels"]},
        {"check": f"every feasible {led['strength']}-way combination is covered", "ok": led["covered"] == led["feasible"],
         "detail": led["uncovered"]},
        {"check": "every state, edge, rule and transfer reason is reached", "ok": cl["covered_items"] == cl["reachable_items"],
         "detail": cl.get("unreached", [])},
        {"check": "the oracle obeys every rule in every case", "ok": not inadm,
         "detail": [f"{dict(c.levels)}: {a}" for c, a in inadm[:10]]},
        {"check": f"every level's expectation holds ({n_exp} levels declare one)", "ok": not unmet, "detail": unmet},
    ]
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "levels": rows, "cases": cases,
            "ledger": led, "closure": cl}


def apply_edits(dom, edits: list[dict[str, Any]], dry_run: bool = False) -> tuple[dict[str, Any], list[str]]:
    """Apply edits to a pack's base_factors.yaml (a copy is kept as base_factors.yaml.bak).

      {"drop_level": "factor.level"}            {"drop_factor": "factor"}
      {"rename_level": "factor.old", "to": "new"}
      {"set_level": "factor.level", "script": {...}}      add or replace a level (tools / says / answer / ...)
      {"expect": "factor.level", "outcome": {...}}       the outcome the policy owner expects (None removes it)
      {"default": "factor.level"}                        make a level the factor's default (first)
      {"set": "strength" | "closure_budget", "value": n}

    Returns the new config and its problems; nothing is written when there are problems or on dry_run."""
    import copy
    p = dom.dir / FILE
    cfg = copy.deepcopy(load(dom) or {"strength": 2, "closure_budget": 3, "factors": {}})
    facs = cfg.setdefault("factors", {})
    errs: list[str] = []

    def split(ref):
        f, _, lv = str(ref).partition(".")
        if f not in facs:
            errs.append(f"{ref}: no factor {f!r}")
            return None, None
        return f, lv

    for e in edits:
        if "drop_factor" in e:
            facs.pop(e["drop_factor"], None) if e["drop_factor"] in facs else errs.append(f"no factor {e['drop_factor']!r}")
        elif "drop_level" in e:
            f, lv = split(e["drop_level"])
            if f and lv in facs[f]["levels"]:
                if len(facs[f]["levels"]) == 1:
                    errs.append(f"{e['drop_level']}: the only level (drop the factor instead)")
                else:
                    del facs[f]["levels"][lv]
            elif f:
                errs.append(f"{e['drop_level']}: no such level")
        elif "rename_level" in e:
            f, lv = split(e["rename_level"])
            if f and lv in facs[f]["levels"]:
                facs[f]["levels"] = {(e["to"] if k == lv else k): v for k, v in facs[f]["levels"].items()}
            elif f:
                errs.append(f"{e['rename_level']}: no such level")
        elif "set_level" in e:
            f, lv = split(e["set_level"])
            if f:
                facs[f]["levels"][lv] = e.get("script") or {}
        elif "expect" in e:
            f, lv = split(e["expect"])
            if f and lv in facs[f]["levels"]:
                spec = facs[f]["levels"][lv] = dict(facs[f]["levels"][lv] or {})
                if e.get("outcome") is None:
                    spec.pop("expect", None)
                else:
                    spec["expect"] = e["outcome"]
            elif f:
                errs.append(f"{e['expect']}: no such level")
        elif "default" in e:
            f, lv = split(e["default"])
            if f and lv in facs[f]["levels"]:
                lvs = facs[f]["levels"]
                facs[f]["levels"] = {lv: lvs[lv], **{k: v for k, v in lvs.items() if k != lv}}
            elif f:
                errs.append(f"{e['default']}: no such level")
        elif e.get("set") in ("strength", "closure_budget"):
            cfg[e["set"]] = int(e["value"])
        else:
            errs.append(f"unknown edit {e}")
    errs += check(cfg) + _check_expect(cfg)
    if not errs and not dry_run:
        if p.exists():
            (dom.dir / (FILE + ".bak")).write_text(p.read_text())
        head = "".join(l for l in (p.read_text().splitlines(True) if p.exists() else []) if l.startswith("#"))
        p.write_text((head + "\n" if head else "") +
                     yaml.safe_dump(cfg, sort_keys=False, width=140, allow_unicode=True, default_flow_style=None))
    return cfg, errs


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Draft or check base_factors.yaml for a compact policy graph pack.")
    ap.add_argument("command", choices=["draft", "check", "review"])
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
    if a.command == "review":
        from .kb import kb_frames
        r = review(oracle, load(dom) or {}, kb_frames(dom.graph))
        for c in r["checks"]:
            print(("PASS " if c["ok"] else "FAIL ") + c["check"])
            for d in (c["detail"] or [])[:10] if not c["ok"] else []:
                print("     -", d)
        return 0 if r["ok"] else 1
    errs = check(load(dom) or {}) + _check_expect(load(dom) or {})
    print(f"{p.name}: " + ("clean" if not errs else f"{len(errs)} problem(s)"))
    for e in errs:
        print("  -", e)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
