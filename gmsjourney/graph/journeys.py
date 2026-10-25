"""Journeys mined from a compact policy graph: base cases, fixtures, the simulated customer.

A base case is a *script*: what each tool returns on its n-th call and what the customer means
on the n-th visit of each prompt state. Nothing is hand-written; the scripts are mined by running
the oracle in explore mode, where every decision the script does not make yet stops the run and
branches:

  tool call      the tool's success, each failure status, UNAVAILABLE, and for a tool that
                 echoes a value a success echoing another value (ECHO_MISMATCH); a failure
                 that returns a value-set field (an ineligibility reason) once per value
  prompt state   each act the state accepts, with each value of its slot

Every option but the first (the default: success, the first act, the first value) is a
deviation. Paths deviate at most ``max_deviations`` times; a deviation that only changes a value
(the same branch as the default) is taken alone, so every value is covered without multiplying
paths. Paths with the same observable behaviour are kept once.

Failure perturbations are then laid over every prompt state, on the first mined path that reaches
it: a request for a person, an unclear answer (recovered once, then persisting), a policy question
mid-journey, and the turn limit (exactly at it, and one turn over).

``GraphSimulator`` answers each bot act from the script, keyed by the prompt state the act names,
so a bot that phrases or orders its prompts differently still gets the answer it asks for.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from ..ontology import SemanticFrame as F
from .model import ECHO_MISMATCH, UNAVAILABLE


def _graph():
    from ..domain import current
    return current().graph


# ============================================================================ spec
class GraphSpec:
    """One mined base case. Immutable; compared and hashed by content."""
    __slots__ = ("customer", "tools", "label", "kb_frames", "extra_kb")

    def __init__(self, customer: dict[str, list[F]] | tuple = (), tools: dict[str, list[dict]] | tuple = (),
                 label: tuple = (), kb_frames: tuple = (), extra_kb: int = 0):
        cust = customer.items() if isinstance(customer, dict) else customer
        tl = tools.items() if isinstance(tools, dict) else tools
        object.__setattr__(self, "customer", tuple(sorted((s, tuple(fs)) for s, fs in cust)))
        object.__setattr__(self, "tools", tuple(sorted((t, tuple(json.dumps(r, sort_keys=True) if not isinstance(r, str) else r
                                                                   for r in rs)) for t, rs in tl)))
        object.__setattr__(self, "label", tuple(label))
        object.__setattr__(self, "kb_frames", tuple(kb_frames))
        object.__setattr__(self, "extra_kb", extra_kb)

    def __setattr__(self, k, v):
        raise AttributeError("GraphSpec is immutable; use with_()")

    def customer_dict(self) -> dict[str, list[F]]:
        return {s: list(fs) for s, fs in self.customer}

    def tools_dict(self) -> dict[str, list[dict]]:
        return {t: [json.loads(r) for r in rs] for t, rs in self.tools}

    def key(self) -> str:
        body = {"customer": [[s, [f.canonical() for f in fs]] for s, fs in self.customer],
                "tools": [list(x) for x in self.tools], "kb": [f.canonical() for f in self.kb_frames]}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]

    def dims(self) -> dict[str, Any]:
        """The decisions this case makes (for reports): one entry per deviation / perturbation."""
        return {f"d{i}": x for i, x in enumerate(self.label)} or {"path": "default"}

    def with_(self, **kw) -> "GraphSpec":
        d = {"customer": self.customer, "tools": self.tools, "label": self.label, "kb_frames": self.kb_frames,
             "extra_kb": self.extra_kb, **kw}
        return GraphSpec(**d)

    def __eq__(self, other) -> bool:
        return isinstance(other, GraphSpec) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return f"GraphSpec({', '.join(self.label) or 'default'})"


def build_fixture(spec: GraphSpec) -> dict[str, Any]:
    from ..domain import current
    fx: dict[str, Any] = {"credential": str(current().get("fixtures.credential", "7316"))}
    for t, rs in spec.tools_dict().items():
        fx[f"tool:{t}"] = rs
    return fx


# ============================================================================ simulator
class GraphSimulator:
    def __init__(self, spec: GraphSpec, fixture: dict[str, Any] | None = None, explore: bool = False):
        self.g = _graph()
        self.queues = spec.customer_dict()
        self.explore = explore
        self.last_state: str | None = None
        self.last: dict[str, F] = {}
        self.consumed: Counter = Counter()
        self.emitted: list[F] = []

    def state_of(self, bot_act: dict[str, Any]) -> str | None:
        g = self.g
        act = bot_act.get("act")
        G = g.glob
        if G.get("unclear") and act == G["unclear"]["clarify"]:
            return bot_act.get("state") or self.last_state
        if G.get("question") and act == G["question"]["answer"]:
            return self.last_state
        if act in g.prompt_states:
            return act
        return None

    def respond(self, bot_act: dict[str, Any]) -> F | None:
        st = self.state_of(bot_act)
        if st is None:
            return None                                   # a terminal or unknown act: the customer stays silent
        self.last_state = st
        q = self.queues.get(st) or []
        if q:
            f = q.pop(0)
        elif self.explore:
            from .machine import NeedChoice
            raise NeedChoice("ask", st, self.consumed[st])
        else:
            f = self.last.get(st) or default_frame(st)
        self.consumed[st] += 1
        if f.act not in ("ASK_KB", "UNCLEAR"):
            self.last[st] = f
        self.emitted.append(f)
        return f


def default_frame(state: str) -> F:
    g = _graph()
    act = next(iter(g.states[state]["on"]))
    slots = {sl: sp["set"] for sl, sp in g.slot_specs(act).items() if not sp.get("optional")}
    kw = {}
    for sl, vs in slots.items():
        if vs == "credential_code":
            from ..domain import current
            kw[sl] = str(current().get("fixtures.credential", "7316"))
        elif g.members(vs):
            kw[sl] = g.members(vs)[0]
    return F.make(act, **kw)


# ============================================================================ mining
def _list_options(tool: str, field: str, spec: dict, L) -> list[tuple[list, int, str]]:
    """(list, weight, label) for a list a tool returns: items for a loop (counts around the loop's
    cap, the edges of its window), or a subset of a value set."""
    from ..domain import current
    g = _graph()
    dom = current()
    if "subset_of" in spec:
        vs = spec["subset_of"]
        full = list(dom.get(f"fixtures.options.{vs}") or g.members(vs))
        out = [(full, 0, f"{field}=all")]
        out += [([m], 1, f"{field}=[{m}]") for m in full if len(full) > 1]
        return out
    fx = dom.get(f"fixtures.items.{spec['items']}") or {}
    recs, ages, prefix = fx.get("records") or [{}], fx.get("ages") or [1], fx.get("id_prefix", "I")
    loop = next((s for s in g.states.values() if s.get("each") == f"${tool}.{field}"), {})
    age_f, id_f = spec.get("age"), spec["id"]

    def items(n: int, last_age=None, extra_age=None):
        out = []
        for i in range(n):
            r = {id_f: f"{prefix}{i + 1}", **recs[i % len(recs)]}
            if age_f:
                r[age_f] = ages[i % len(ages)] if (last_age is None or i < n - 1) else last_age
            out.append(r)
        if extra_age is not None:
            out.append({id_f: f"{prefix}{n + 1}", **recs[n % len(recs)], age_f: extra_age})
        return out

    out = [(items(1), 0, f"{field}=1")]
    cap = L(loop["cap"]) if loop.get("cap") else None
    counts = sorted({2} | ({cap, cap + 1} if cap else set()))
    out += [(items(n), 1, f"{field}={n}") for n in counts]
    for fld, lim in (loop.get("where") or {}).items():
        if fld == age_f and "max" in lim:
            w = L(lim["max"])
            out.append((items(1, last_age=w), 1, f"{field}:{fld}={w}"))
            out.append((items(1, extra_age=w + 1), 1, f"{field}:+{fld}={w + 1}"))
    return out


def _tool_options(tool: str, slots: dict[str, Any], L=None) -> list[tuple[dict, int, str]]:
    """(result, weight, label) for the next call of a tool; weight 0 = the default."""
    g = _graph()
    sp = g.tools[tool]
    out: list[tuple[dict, int, str]] = []
    ok = {"status": sp["ok"]}
    ret = sp.get("returns") or {}
    lists = {k: v for k, v in ret.items() if isinstance(v, dict)}
    if lists and L is not None:
        (fld, lspec), = list(lists.items())[:1]
        empty_status = (sp.get("fail") or [None])[0]
        for lst, w, lab in _list_options(tool, fld, lspec, L):
            out.append(({"status": sp["ok"], fld: lst}, w, f"{tool}:{lab}"))
        if empty_status:
            out.append(({"status": empty_status, fld: []}, 1, f"{tool}={empty_status}"))
        out.append(({"status": UNAVAILABLE}, 1, f"{tool}={UNAVAILABLE}"))
        return out
    if "reference" in ret:
        ok["reference"] = "SUP-10427"
    out.append((ok, 0, f"{tool}={sp['ok']}"))
    if sp["role"] == "handoff":
        return out                                    # the policy says nothing about a failed handoff
    for st in sp.get("fail", []):
        vs_fields = {k: vs for k, vs in ret.items() if vs in g.values and _said_after(tool, st, k)}
        if vs_fields and st != sp["ok"]:
            (fld, vs), = list(vs_fields.items())[:1]
            for m in g.members(vs):
                out.append(({"status": st, fld: m}, 1, f"{tool}={st}({m})"))
        else:
            out.append(({"status": st}, 1, f"{tool}={st}"))
    out.append(({"status": UNAVAILABLE}, 1, f"{tool}={UNAVAILABLE}"))
    echo = sp.get("echo")
    if echo:
        vs = (sp.get("args") or {}).get(echo)
        cur = slots.get(echo)
        other = next((m for m in g.members(vs) if m != cur), None) if vs else None
        if other:
            out.append(({**ok, echo: other}, 1, f"{tool}={ECHO_MISMATCH}({other})"))
    return out


def _said_after(tool: str, status: str, field: str) -> bool:
    """Does the branch a tool status takes say the returned field back (``$tool.field``)?"""
    g = _graph()
    ref = f"${tool}.{field}"
    for name, s in g.states.items():
        if g.kind(name) != "call" or s["call"] != tool or status not in s["on"]:
            continue
        for t, _ in g.alternatives(s["on"][status]):
            seen, todo = set(), [t]
            while todo:
                x = todo.pop()
                if x.startswith("!") or x in seen or x not in g.states:
                    continue
                seen.add(x)
                if ref in json.dumps(g.states[x].get("say") or {}):
                    return True
                if g.kind(x) == "say":
                    todo.append(g.states[x]["go"])
    return False


def _branch(state: str, act: str, slots: dict[str, Any]) -> str:
    """Which alternative an act takes in a state, given the captured slots."""
    from .machine import GraphOracle
    g = _graph()
    target = g.states[state]["on"][act]
    probe = GraphOracle.__new__(GraphOracle)
    probe.g, probe._slots = g, dict(slots)
    probe.store = None
    probe.attribute = lambda v, a: g.attr(v, a) if v is not None else None
    try:
        return json.dumps(probe._target(target, state) if not (isinstance(target, dict) and "below" in target) else "count")
    except Exception:                                  # noqa: BLE001 - a target that needs a count or a store
        return "?"


def _ask_options(state: str, slots: dict[str, Any]) -> list[tuple[F, int, str, bool]]:
    """(frame, weight, label, value_only) for the next customer turn in a prompt state."""
    g = _graph()
    from ..domain import current
    out: list[tuple[F, int, str, bool]] = []
    default = default_frame(state)
    default_branch = _branch(state, default.act, {**slots, **dict(default.slots)})
    cred = str(current().get("fixtures.credential", "7316"))
    vals = lambda vs: [cred] if vs == "credential_code" else g.members(vs)
    for act in g.states[state]["on"]:
        specs = g.slot_specs(act)
        if not specs:
            f = F.make(act)
            out.append((f, 0 if f == default else 1, f"{state}:{act}", False))
            continue
        req = [(s, sp) for s, sp in specs.items() if not sp.get("optional")]
        opt = [(s, sp) for s, sp in specs.items() if sp.get("optional")]
        bases: list[dict] = [{}]
        for s, sp in req:
            bases = [{**b, s: v} for b in bases for v in vals(sp["set"])]
        for b in bases:
            variants = [(b, False)]
            for s, sp in opt:
                if sp.get("correction_of"):
                    other = next((v for v in vals(sp["set"]) if v != b.get(sp["correction_of"])), None)
                    if other:
                        variants.append(({**b, s: other}, True))
                elif sp.get("with"):
                    for k, ws in sp["with"].items():
                        for w in ws:
                            variants += [({**b, k: w, s: v}, False) for v in vals(sp["set"])]
                else:
                    variants += [({**b, s: v}, False) for v in vals(sp["set"])]
            for kw, value_only in variants:
                if g.updates(act) and act != default.act and all(slots.get(k) == v for k, v in kw.items()):
                    continue                          # an update to what is already chosen changes nothing
                f = F.make(act, **kw)
                if f == default:
                    out.append((f, 0, f"{state}:{f}", False))
                    continue
                same = value_only or (act == default.act and set(kw) == set(dict(default.slots)) and
                                      _branch(state, act, {**slots, **kw}) == default_branch)
                out.append((f, 1, f"{state}:{f}", same))
    return out


def mine_paths(oracle, max_deviations: int = 2, max_paths: int = 200_000,
               ledger: list | None = None, stats: dict | None = None) -> list[tuple[GraphSpec, Any]]:
    """Every path through the machine with at most ``max_deviations`` deviations from the default,
    one per distinct behaviour. Returns (spec, oracle trace) in discovery order; each trace carries
    its ``decisions`` (decision point -> option, defaults included). With ``ledger``, every explored
    conversation is appended as (decisions, behaviour key), duplicates included."""
    from .machine import NeedChoice
    seen: dict[str, int] = {}
    out: list[tuple[GraphSpec, Any]] = []
    stack: list[tuple[dict, dict, int, bool, tuple, tuple]] = [({}, {}, 0, False, (), ())]
    runs = 0
    while stack and runs < max_paths:
        cust, tools, devs, value_dev, label, decided = stack.pop(0)
        spec = GraphSpec(cust, tools, label)
        runs += 1
        try:
            tr = oracle.run(spec, explore=True)
        except NeedChoice as nc:
            slots = getattr(nc, "slots", {})
            if nc.kind == "tool":
                opts = [(r, w, lab, False) for r, w, lab in _tool_options(nc.key, slots, oracle.L)]
            else:
                opts = _ask_options(nc.key, slots)
            for choice, w, lab, value_only in opts:
                nd = devs + w
                if nd > max_deviations:
                    continue
                if value_only and devs > 0:
                    continue                          # value coverage alone, never combined
                if value_dev and w:
                    continue
                c2 = {k: list(v) for k, v in cust.items()}
                t2 = {k: list(v) for k, v in tools.items()}
                if nc.kind == "tool":
                    t2.setdefault(nc.key, []).append(choice)
                else:
                    c2.setdefault(nc.key, []).append(choice)
                dp = f"{nc.kind}:{nc.key}#{nc.index}"
                stack.append((c2, t2, nd, value_dev or value_only, label + ((lab,) if w else ()),
                              decided + ((dp, lab),)))
            continue
        tr.decisions = list(decided)
        key = behaviour_key(tr)
        if ledger is not None:
            ledger.append((decided, key))
        if key in seen:
            continue
        seen[key] = len(out)
        out.append((spec, tr))
    if stats is not None:
        stats.update(runs=runs, complete=not stack, behaviours=len(out))
    return out


def coverage_closure(oracle, paths: list[tuple[GraphSpec, Any]], extra: int = 1,
                     max_paths: int = 200_000) -> list[tuple[GraphSpec, Any]]:
    """Paths with up to ``extra`` more deviations, kept only when they reach a workflow edge, a
    state, a rule or a transfer reason no mined path reaches yet."""
    def cov(tr):
        hit, _ = tr.edges(oracle.store.workflow)
        return {("edge", e) for e in hit} | {("state", s) for s in tr.visited} | \
               {("rule", r) for r in tr.rules_fired} | {("reason", tr.transfer_reason)}
    have = set().union(*(cov(tr) for _, tr in paths)) if paths else set()
    deeper = mine_paths(oracle, max_deviations=max(len(sp.label) for sp, _ in paths) + extra, max_paths=max_paths)
    seen = {behaviour_key(tr) for _, tr in paths}
    out = []
    for sp, tr in deeper:
        new = cov(tr) - have
        if new and behaviour_key(tr) not in seen:
            out.append((sp, tr))
            have |= new
    return out


def behaviour_key(tr) -> str:
    acts = [a for e in tr.events if e["role"] == "bot" for a in e["acts"]]
    return hashlib.sha256(json.dumps([tr.signature(), [json.dumps(a, sort_keys=True, default=str) for a in acts],
                                      [sorted(k for k, _ in f.slots) for f in tr.user_frames],
                                      [(c["tool"], json.dumps(c.get("result"), sort_keys=True, default=str))
                                       for c in tr.tool_log]], sort_keys=True).encode()).hexdigest()


def perturbations(oracle, paths: list[tuple[GraphSpec, Any]], kb_pool: list[F]) -> list[tuple[GraphSpec, str]]:
    """Failure situations over every prompt state, each on the first mined path reaching it."""
    g = _graph()
    G = g.glob
    out: list[tuple[GraphSpec, str]] = []
    done: set[str] = set()
    kb_i = 0
    paths = sorted(paths, key=lambda p: len(p[0].label))      # the default path first, then the fewest deviations
    for spec, tr in paths:
        cust = spec.customer_dict()
        visited = [e["state"] for e in tr.events if e["role"] == "user"]
        for st in dict.fromkeys(visited):
            if st in done:
                continue
            done.add(st)
            base = cust.get(st, [])

            def var(frames: list[F], kind: str):
                c2 = {**cust, st: frames + base}
                out.append((spec.with_(customer=c2, label=spec.label + (f"{kind}@{st}",)), kind))

            if G.get("human"):
                var([F.make(G["human"]["act"])], "human_request")
            if G.get("unclear"):
                lim = oracle.L(G["unclear"]["limit"])
                var([F.make(G["unclear"]["act"])] * lim, "ambiguity_recovered")
                var([F.make(G["unclear"]["act"])] * (lim + 1), "ambiguity_persists")
            if G.get("question") and kb_pool:
                var([kb_pool[kb_i % len(kb_pool)]], "policy_question")
                kb_i += 1
    # the turn limit: pad the longest default path with policy questions, exactly at the limit and one over
    tl = G.get("turn_limit")
    if tl and kb_pool and paths:
        spec, tr = paths[0]
        lim = oracle.L(tl["limit"])
        ask_states = [e["state"] for e in tr.events if e["role"] == "user"]
        if ask_states:
            st = ask_states[min(1, len(ask_states) - 1)]
            cust = spec.customer_dict()
            for kind, extra in (("turn_limit_at", 0), ("turn_limit_over", 1)):
                n = lim - tr.turns + extra
                if n <= 0:
                    continue
                pad = [kb_pool[(kb_i + i) % len(kb_pool)] for i in range(n)]
                c2 = {**cust, st: pad + cust.get(st, [])}
                out.append((spec.with_(customer=c2, label=spec.label + (f"{kind}@{st}+{n}",), extra_kb=n), kind))
    return out


def kb_base(paths: list[tuple[GraphSpec, Any]], frame: F) -> GraphSpec:
    """The default path with one policy question asked at the first prompt after the first tool."""
    spec, tr = min(paths, key=lambda p: len(p[0].label))
    states = [e["state"] for e in tr.events if e["role"] == "user"]
    st = states[min(1, len(states) - 1)]
    cust = spec.customer_dict()
    return spec.with_(customer={**cust, st: [frame] + cust.get(st, [])}, kb_frames=(frame,),
                      label=(f"policy_question@{st}",))


# ============================================================================ items
def deviation_kind(lab: str) -> str:
    """What one deviation of a mined path is: a tool failure, a tool's decline (a failure status),
    returned data at a boundary (item counts, windows, option lists), or a customer choice."""
    g = _graph()
    if f"={UNAVAILABLE}" in lab or ECHO_MISMATCH in lab:
        return "tool_failure"
    head = lab.split("=")[0].split(":")[0]
    if head in g.tools:
        if ":" in lab.split("=")[0]:
            return "data_boundary"
        st = lab.split("=", 1)[1].split("(")[0]
        return "tool_decline" if st in g.tools[head].get("fail", []) else "data_boundary"
    return "customer_choice"


_RANK = ["tool_failure", "tool_decline", "customer_choice", "data_boundary"]


def category_of(spec: GraphSpec, tr) -> str:
    for lab in spec.label:
        kind = lab.split("@")[0]
        if kind in ("human_request", "ambiguity_recovered", "ambiguity_persists", "policy_question",
                    "turn_limit_at", "turn_limit_over"):
            return kind
    if not spec.label:
        return "happy_path"
    kinds = {deviation_kind(lab) for lab in spec.label}
    return next(k for k in _RANK if k in kinds)


def tags_of(spec: GraphSpec, tr) -> list[str]:
    tags = [category_of(spec, tr)]
    st = GraphSimulatorStatus(tr)
    tags += [f"tool:{t}={s}" for t, s in st if s not in ("ok",)]
    tags += [f"said:{a['act']}" for a in tr.told]
    if tr.transferred:
        tags.append(f"transfer:{tr.transfer_reason}")
    if tr.writes():
        tags += [f"write:{w[0]}" for w in tr.writes()]
    return list(dict.fromkeys(tags))


def GraphSimulatorStatus(tr) -> list[tuple[str, str]]:
    g = _graph()
    out = []
    for c in tr.tool_log:
        s = tr.status_of(c)
        out.append((c["tool"], "ok" if s == g.tools.get(c["tool"], {}).get("ok") else s))
    return out


def to_item(spec: GraphSpec, oracle, qid: str, category: str | None = None, kb_item=None):
    from ..core import QAItem
    from ..enrich.realizers import Realizer
    fx = build_fixture(spec)
    tr = oracle.run(spec, fx)
    comps = {f"expected_{k}": v for k, v in tr.components().items()}
    first = tr.user_frames[0] if tr.user_frames else F.make("UNCLEAR")
    cat = category or category_of(spec, tr)
    return QAItem(
        qid=qid, query=Realizer().realize(first, {}, f"{qid}|0|base"), answer=tr.signature(),
        answer_type="decision", base=cat, components=comps,
        metadata={"spec": spec, "fixture": fx, "relevant": spec.dims(), "oracle_turns": tr.turns,
                  "oracle_tool_calls": len(tr.tool_log), "oracle_tools": sorted({c["tool"] for c in tr.tool_log}),
                  "rules_fired": sorted(tr.rules_fired), "edges": sorted(tr.edges(oracle.store.workflow)[0]),
                  "transfer_reason": tr.transfer_reason, "states": list(tr.visited),
                  "kb": (kb_item.category if kb_item else None), "kb_qid": (kb_item.qid if kb_item else None),
                  "tags": tags_of(spec, tr) if not kb_item else [kb_item.category]})
