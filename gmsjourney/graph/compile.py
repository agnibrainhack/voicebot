"""Compiling a compact policy graph (format 2) into everything the engine reads.

A compact graph is a short state machine (values, acts, tools, rules, states). The engine reads the full
triples shape. This file is the translator, like a compiler turning a short program into the machine code the
runtime executes: same meaning, more explicit.

Parts of this file:
  1. ``compile_pack``: compact graph -> full policy graph, with the executable flow (every transition with its
     guard expression, actions and rules), what each act carries, what each tool is called with, outcome and
     claim acts, and the speech layer. Its walk matches the compact interpreter's on every mined journey.
  2. ``compile_graph``: compact graph -> the symbolic store (exact figures, rules, workflow, value sets) and the
     tables the knowlytix ingest trains on (``render_gms_view``).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any


# =====================================================================================================================
# PART 1 - Compile a compact policy graph (``gmsjourney.policy_graph/2``) into the full policy graph every component reads.
#
# A compact pack states its dialog as a state machine (``graph/compact.py``): ask, say, call and end states, guarded
# targets, transfers with reasons and the always-available behaviours (a person on request, a policy question, an
# unclear answer, the turn limit). The rest of the harness reads one format, the head / relation / tail graph of
# ``graph/triples.py``, whose dialog is executable flow (``flow.py``). This module writes that graph:
#
# * every fact the compact graph states (``graph.model.to_triples``: limits, value sets with aliases and attributes,
#   acts and slots, tools with roles and statuses, rules and the facts they state, transfer reasons, workflow,
#   knowledge sentences, the opening), with the descriptive ``leads_to`` edges replaced by
# * the **flow**: ``leads_to`` triples whose qualifiers are expressions of the flow language (``on``, ``pre``,
#   ``guard``, ``do``, ``say``, ``rules``, ``quiet``, ``flush``, ``retract``). Call states become quiet tool steps
#   (``is_a tool_step``) left on the tool's status; the always-available behaviours become transitions out of every
#   prompt state, in the machine's order of precedence (a person, a policy question, the state's own acts, an
#   unclear answer); the turn limit is checked before a prompt is said (``any`` + ``retract``);
# * ``carries`` (what each act carries, from the states' ``say`` fields), ``has_arg`` (what each tool is called
#   with, from its call state), ``returns_field``;
# * the **speech layer**: ``says`` (each field an act must or may say, bound to where its truth comes from),
#   ``is_outcome`` and ``claims_tool`` for the acts the scoring reads.
#
# The compiled walk is checked against the compact graph's own interpreter (``graph/interpreter.py``) case for case
# (``tests/test_pack_compile.py``).
# =====================================================================================================================

from .compact import ECHO_MISMATCH, NEXT, UNAVAILABLE, PolicyGraph, attr_relation, to_triples


class PackCompileError(ValueError):
    """A compact graph cannot be compiled into the full graph as written."""
    pass


CALL_STEP = "tool_step"            # is_a class of a compiled call state (not a bot act)


def _q(v: Any) -> str:
    """A Python literal for an expression."""
    return repr(v)


def _counter_name(target: Any) -> str | None:
    """The name of the counter a compact target counts with ({"count": name, "upto": ...}), wherever it sits."""
    if isinstance(target, dict):
        if "count" in target:
            return target["count"]
        return next((n for v in target.values() for n in [_counter_name(v)] if n), None)
    if isinstance(target, list):
        return next((n for v in target for n in [_counter_name(v)] if n), None)
    return None


class _Compiler:
    def __init__(self, g: PolicyGraph):
        self.g = g
        self.num = g.numeric_attributes()
        self.rows: list[dict[str, Any]] = []
        self.T = g.transfers
        self.ts = g.transfer_state
        self.handoff = self.T.get("tool")
        self.handoff_arg = self.T.get("arg", "reason")
        self.call_args: dict[str, dict[str, Any]] = {}
        self.loop_vars = {st.get("as", "item") for n, st in g.states.items() if g.kind(n) == "each"}
        self.steps: list[str] = []                    # quiet steps the compiler adds (handoff, loop steps)

    # ------------------------------------------------------------------ expressions
    def attr_expr(self, value_expr: str, attr: str) -> str:
        if attr in self.num:                          # numbers come from exact memory
            return f"enm({_q(attr)}, {value_expr})"
        return f"tail({value_expr}, {_q(attr_relation(attr))})"

    def ref(self, ref: Any) -> str:
        """A field value of the compact graph ($opening, $<limit>, $<tool>.<field>, $<slot>[.<attr>]) as an
        expression."""
        if isinstance(ref, dict) and "$writes" in ref:           # what a write tool truly did: key -> value
            return f"writes({_q(ref['$writes'])}, {_q(ref['key'])}, {_q(ref['value'])})"
        if isinstance(ref, dict):
            raise PackCompileError(f"field value {ref!r} is not supported by the compiler yet")
        if not isinstance(ref, str) or not ref.startswith("$"):
            return _q(ref)
        body = ref[1:]
        if body == "opening":
            return "text('opening_statement')"
        if body in self.g.limits:
            return f"limit({_q(body)})"
        head, _, tail = body.partition(".")
        if head in self.g.tools:
            return f"field({_q(head)}, {_q(tail)})"
        if not tail:
            return f"get({_q(head)})"
        if head in self.loop_vars:                                # a field of the current loop item
            return f"at(get({_q(head)}), {_q(tail)})"
        return self.attr_expr(f"get({_q(head)})", tail)

    def binding(self, ref: Any) -> str:
        """Where a said field's truth comes from, in the speech check's binding language."""
        if not isinstance(ref, str) or not ref.startswith("$"):
            return "literal"
        body = ref[1:]
        if body == "opening":
            return "text.opening"
        if body in self.g.limits:
            return f"policy.limit:{body}"
        head, _, tail = body.partition(".")
        if head in self.g.tools:
            return f"result.{head}.{tail}"
        if not tail:
            return f"caller.{head}"
        if head in self.loop_vars:
            return f"item.{tail}"
        if tail in self.num:
            return f"policy.{tail}:caller.{head}"
        return f"attribute.{tail}:caller.{head}"

    def cond(self, cond: dict[str, Any], state: str) -> list[str]:
        """A compact condition ({"slot": [values]}, {"slot.attr": [...]}, {"#tool": [n]}, below / at_least ...)
        as guard expressions (all must hold); named counters are handled by the caller."""
        out = []
        for k, want in cond.items():
            k = k.split("#")[0] if not k.startswith("#") else k            # repeated keys come back as key#i
            if k in ("below", "at_least"):
                tool = self.g.states[state]["call"]
                op = "<" if k == "below" else ">="
                out.append(f"count({_q(tool)}) {op} limit({_q(want)})")
                continue
            if k in ("count_upto", "count_over"):
                continue
            if k.startswith("#"):
                out.append(f"count({_q(k[1:])}) in {_q(tuple(want))}")
                continue
            slot, _, attr = k.partition(".")
            v = f"get({_q(slot)})"
            if attr:
                v = self.attr_expr(v, attr)
            if isinstance(want, dict):
                if "not" in want:
                    out.append(f"{v} not in {_q(tuple(want['not']))}")
                if "in" in want:
                    tool, _, fld = str(want["in"])[1:].partition(".")
                    out.append(f"{v} in options({_q(tool)}, {_q(fld)})")
            else:
                out.append(f"{v} in {_q(tuple(want))}")
        return out

    # ------------------------------------------------------------------ rows
    def row(self, src: str, dst: str, on: str, *, when: str = "", guard: list[str] | None = None,
            pre: list[str] | None = None, do: list[str] | None = None, say: list[str] | None = None,
            rules: list[str] | None = None, quiet: bool = False, flush: bool = False, retract: bool = False) -> None:
        q: dict[str, Any] = {"on": on, "when": when}
        if pre:
            q["pre"] = list(pre)
        if guard:
            q["guard"] = " and ".join(f"({g})" for g in guard) if len(guard) > 1 else guard[0]
        if do:
            q["do"] = list(do)
        if say:
            q["say"] = list(say)
        if rules:
            q["rules"] = list(dict.fromkeys(rules))
        for k, v in (("quiet", quiet), ("flush", flush), ("retract", retract)):
            if v:
                q[k] = True
        self.rows.append({"src": src, "dst": dst, "q": q})

    def into(self, state: str) -> list[str]:
        """The rules a state exercises when it is entered."""
        return list((self.g.states.get(state) or {}).get("rules", []))

    def goto(self, src: str, on: str, target: Any, *, guard: list[str] | None = None, pre: list[str] | None = None,
             do: list[str] | None = None, say: list[str] | None = None, rules: list[str] | None = None,
             when: str = "") -> None:
        """Rows for every alternative of a compact target, in order (the first that holds is taken)."""
        alts = list(self.g.alternatives(target))
        counter = _counter_name(target)               # a counted branch, at the top or nested in a condition
        for t, c in alts:
            g2 = list(guard or []) + self.cond(c, src)
            d2 = list(do or [])
            if counter and ("count_upto" in c or "count_over" in c):
                ok = "count_upto" in c
                lim = c.get("count_upto") or c.get("count_over")
                g2.append(f"n({_q('count:' + counter)}) + 1 {'<=' if ok else '>'} limit({_q(lim)})")
                d2.append(f"inc({_q('count:' + counter)})")
            self._edge(src, on, t, g2, pre, d2, say, rules, when or json.dumps(c, sort_keys=True) if c else when)

    def _edge(self, src, on, t, guard, pre, do, say, rules, when) -> None:
        if t == NEXT:                                             # the next item: back to the loop
            t = self.g.loop_of.get(src)
            if t is None:
                raise PackCompileError(f"{src}: @next outside a loop")
        if isinstance(t, str) and t.startswith("!"):
            reason = t[1:]
            rid = (self.T.get("reasons") or {}).get(reason)
            if rid is None:
                raise PackCompileError(f"{src}: transfer reason {reason!r} is not declared under transfers.reasons")
            self.row(src, self.ts, on, when=when or f"transfer: {reason}", guard=guard, pre=pre,
                     do=[*(do or []), f"set('reason', {_q(reason)})"], say=say, rules=[*(rules or []), rid])
            return
        kind = self.g.kind(t)
        d = list(do or [])
        if kind == "end":
            d.append("end()")
        self.row(src, t, on, when=when, guard=guard, pre=pre, do=d, say=say,
                 rules=[*(rules or []), *self.into(t)], quiet=(kind in ("call", "each")))

    # ------------------------------------------------------------------ the machine
    def compile(self) -> None:
        g, G = self.g, self.g.glob
        human, question, unclear, tl = G.get("human"), G.get("question"), G.get("unclear"), G.get("turn_limit")
        global_acts = {x["act"] for x in (human, question) if x}
        for name, s in g.states.items():
            kind = g.kind(name)
            if kind == "each":
                self._each(name, s)
            elif kind == "say":
                rules = self.into(name) if name == g.start else []         # the first state is entered, not reached
                self.goto(name, "auto", s["go"], rules=rules)
            elif kind == "call":
                tool = s["call"]
                args = s.get("args") or {}
                if tool in self.call_args and self.call_args[tool] != args:
                    raise PackCompileError(f"{tool} is called with different arguments from different states")
                self.call_args[tool] = args
                sp = g.tools[tool]
                echo = sp.get("echo")
                for status, target in s["on"].items():
                    if status == ECHO_MISMATCH:
                        guard = [f"call({_q(tool)}) == {_q(sp['ok'])}", f"not echoed({_q(tool)}, {_q(echo)})"]
                    elif status == sp["ok"] and echo:
                        guard = [f"call({_q(tool)}) == {_q(sp['ok'])}", f"echoed({_q(tool)}, {_q(echo)})"]
                    else:
                        guard = [f"call({_q(tool)}) == {_q(status)}"]
                    self.goto(name, "auto", target, guard=guard, when=f"{tool} returns {status}")
            elif kind == "ask":
                self._ask(name, s, human, question, unclear, tl, global_acts)
        # the transfer: say TRANSFER (reason), end the bot's turn, call the handoff tool, stop
        if self.T:
            step = f"{self.ts}__HANDOFF"
            self.row(self.ts, step, "auto", when=f"call {self.handoff}", quiet=True, flush=True)
            self.row(step, self.T.get("then", g.end_state), "auto", when=f"{self.handoff} called",
                     guard=[f"call({_q(self.handoff)}) or True"], quiet=True)
            self.call_args[self.handoff] = {self.handoff_arg: "$reason"}
            self.steps.append(step)
            sm = self.T.get("summary")
            if sm:                               # before a transfer: what was truly done, when anything was
                val = self.ref(sm["value"])
                for r in self.rows:
                    if r["dst"] == self.ts and r["src"] != self.ts:
                        q = r["q"]
                        q["do"] = [*q.get("do", []), f"fire({_q(sm['rule'])}) if {val} else None"]
                        q["say"] = [f"{_q(sm['act'])} if {val} else None", *q.get("say", [])]

    def _each(self, name: str, s: dict[str, Any]) -> None:
        """A loop over a returned list: a quiet step advances it (loads and filters the items the first time, then
        moves to the next item, binding it and resetting the per-item counters) and branches on the outcome."""
        items = self.ref(s["each"])
        for fld, lim in (s.get("where") or {}).items():
            low = f"limit({_q(lim['min'])})" if "min" in lim else "None"
            high = f"limit({_q(lim['max'])})" if "max" in lim else "None"
            items = f"where({items}, {_q(fld)}, {low}, {high})"
        cap = f"limit({_q(s['cap'])})" if s.get("cap") else "None"
        var, step, key = s.get("as", "item"), f"{name}__STEP", f"loop:{name}"
        self.row(name, step, "auto", when="next item", quiet=True,
                 do=[f"set({_q(key)}, loop_step({_q(name)}, {items}, {cap}, {_q(var)}))"])
        self.steps.append(step)
        outcomes = [("item", s["do"]), ("empty", s.get("empty") or s.get("done")),
                    ("over", s.get("over_cap") or s["done"]), ("done", s["done"])]
        for outcome, target in outcomes:
            self.goto(step, "auto", target, guard=[f"get({_q(key)}) == {_q(outcome)}"], when=f"loop {name}: {outcome}")

    def _ask(self, name, s, human, question, unclear, tl, global_acts) -> None:
        g = self.g
        if tl:                                   # checked before the prompt is said
            wrap = tl.get("wrap_up")
            plain = [t for t in (s.get("on") or {}).values() if isinstance(t, str)]
            guard = [f"turns() >= limit({_q(tl['limit'])})"]
            if wrap and wrap in plain:
                self.row(name, wrap, "any", when=f"{tl['limit']} reached: wrap up", guard=guard, retract=True,
                         do=["end()"], rules=[tl["rule"], *self.into(wrap)])
            else:
                reason = tl["else"][1:]
                self.row(name, self.ts, "any", when=f"{tl['limit']} reached", guard=guard, retract=True,
                         do=[f"set('reason', {_q(reason)})"], rules=[tl["rule"], self.T["reasons"][reason]])
        if human:
            reason = human["then"][1:]
            self.row(name, self.ts, human["act"], when="customer asks for a human",
                     do=[f"set('reason', {_q(reason)})"], rules=[self.T["reasons"][reason]])
        if question:
            self.row(name, name, question["act"], when="policy question, then re-ask",
                     do=["kb_log()"], say=[_q(question["answer"])], rules=[question["rule"], *self.into(name)])
        for act, target in s["on"].items():
            if act in global_acts:
                continue
            update = g.updates(act)
            pre = []
            for sl in g.slot_specs(act):
                if update:
                    pre.append(f"set({_q(sl)}, slot({_q(sl)}) if slot({_q(sl)}) is not None else get({_q(sl)}))")
                else:
                    pre.append(f"set({_q(sl)}, slot({_q(sl)}))")
            self.goto(name, act, target, pre=pre, do=[f"set({_q('unclear:' + name)}, 0)"])
        if unclear:
            key = f"unclear:{name}"
            reason = unclear["else"][1:]
            self.row(name, name, "else", when=f"unclear, below {unclear['limit']}", pre=[f"inc({_q(key)})"],
                     guard=[f"n({_q(key)}) <= limit({_q(unclear['limit'])})"],
                     do=[f"set('clarify_state', {_q(name)})"], say=[_q(unclear["clarify"])],
                     rules=[unclear["rule"], *self.into(name)])
            self.row(name, self.ts, "else", when=f"still unclear after {unclear['limit']}",
                     do=[f"set('reason', {_q(reason)})"], rules=[unclear["rule"], self.T["reasons"][reason]])
        else:
            self.row(name, name, "else", when="not understood: ask again", rules=self.into(name))


def compile_pack(g: PolicyGraph):
    """The full policy graph (a PolicyKG) of a compact policy graph."""
    from .compact import graph_kg
    c = _Compiler(g)
    c.compile()
    triples = [t for t in to_triples(g) if t["relation"] != "leads_to"]
    G = g.glob
    call_states = [s for s in g.states if g.kind(s) in ("call", "each")]
    said = {}                                    # acts said that are not states of the compact machine
    if G.get("question"):
        said[G["question"]["answer"]] = "ANSWER_KB"
    if G.get("unclear"):
        said[G["unclear"]["clarify"]] = "ASK_CLARIFY"

    def add(h, r, t, kind, source, **kw):
        tr = {"id": f"P{len(triples) + 1:04d}", "head": h, "relation": r, "tail": t,
              "tail_type": "literal" if not isinstance(t, str) or t in ("yes", "no") else "entity",
              "kind": kind, "source": source, "evidence": kw.pop("evidence", ""), "gms": kw.pop("gms", False)}
        tr.update(kw)
        triples.append(tr)

    # call states are steps of the flow, not things the bot says
    for t in triples:
        if t["relation"] == "is_a" and t["tail"] == "dialog_state" and t["head"] in call_states:
            t["tail"] = CALL_STEP
            t["gms"] = False                     # the trained store keeps the compact graph's view of it
    for st in c.steps:
        add(st, "is_a", CALL_STEP, "dialog", "transfers" if st.endswith("__HANDOFF") else f"states.{st.split('__')[0]}")
    sm = (g.transfers or {}).get("summary")
    if sm and sm["act"] not in g.states:          # the summary said before a transfer, when it is not a state
        add(sm["act"], "is_a", "dialog_state", "dialog", "transfers.summary")
        add(sm["act"], "prompts_customer", "no", "dialog", "transfers.summary", state=sm["act"])
        add(sm["act"], "carries", sm["field"], "flow", "transfers.summary", qualifiers={"from": c.ref(sm["value"])})
    for act, stage_like in said.items():
        add(act, "is_a", "dialog_state", "dialog", "global")
        add(act, "prompts_customer", "no", "dialog", "global", state=act)
        st = g.stage(g.prompt_states[0]) if g.prompt_states else None
        if st:
            add(act, "in_stage", st, "dialog", "global", state=act)
    add(g.start, "starts_dialog", "yes", "dialog", f"states.{g.start}")
    # the flow
    for r in c.rows:
        add(r["src"], "leads_to", r["dst"], "trajectory", f"states.{r['src']}" if r["src"] in g.states else "flow",
            qualifiers=r["q"], gms=False, state=r["src"])
    # what each act carries, and what each said field is bound to
    for name, s in g.states.items():
        if g.kind(name) in ("call", "each"):
            continue
        for i, (fld, ref) in enumerate((s.get("say") or {}).items()):
            add(name, "carries", fld, "flow", f"states.{name}", qualifiers={"from": c.ref(ref)})
            add(name, "says", fld, "speech", f"states.{name}",
                qualifiers={"from": c.binding(ref), "required": True, "order": i, "basis": "stated"})
        if s.get("outcome"):
            add(name, "is_outcome", "yes", "dialog", f"states.{name}")
        if s.get("claims"):
            add(name, "claims_tool", s["claims"], "dialog", f"states.{name}")
    if c.ts:
        add(c.ts, "carries", "reason", "flow", "transfers", qualifiers={"from": "get('reason')"})
        add(c.ts, "says", "reason", "speech", "transfers",
            qualifiers={"from": "caller.reason", "required": False, "order": 0, "basis": "design"})
    if G.get("question"):
        ans = G["question"]["answer"]
        add(ans, "carries", "intent", "flow", "global.question", qualifiers={"from": "slot('intent')"})
        add(ans, "carries", "answer", "flow", "global.question", qualifiers={"from": "kb_answer()"})
    if G.get("unclear"):
        add(G["unclear"]["clarify"], "carries", "state", "flow", "global.unclear", qualifiers={"from": "get('clarify_state')"})
    # what each tool is called with, and what it returns
    for tool, args in c.call_args.items():
        for a, ref in args.items():
            expr = "get('reason')" if ref == "$reason" else c.ref(ref)
            add(tool, "has_arg", a, "tool", f"tools.{tool}", qualifiers={"from": expr})
    for tool, sp in g.tools.items():
        for fld, typ in (sp.get("returns") or {}).items():
            if isinstance(typ, dict) and "items" in typ:           # a list of items: the list and each item's fields
                add(tool, "returns_field", fld, "tool", f"tools.{tool}", qualifiers={"type": "list", "items": typ["items"]})
                for role, ftype in (("id", "id"), ("age", "days")):
                    if typ.get(role):
                        add(tool, "returns_field", typ[role], "tool", f"tools.{tool}", qualifiers={"type": ftype, "of": fld})
            elif isinstance(typ, dict) and "subset_of" in typ:     # some members of a value set
                add(tool, "returns_field", fld, "tool", f"tools.{tool}",
                    qualifiers={"type": "value", "values_from": typ["subset_of"], "subset": True})
            else:
                add(tool, "returns_field", fld, "tool", f"tools.{tool}",
                    qualifiers={"type": "value" if typ in g.values else str(typ), "values_from": typ if typ in g.values else None})
        if sp.get("echo"):
            add(tool, "echoes", sp["echo"], "tool", f"tools.{tool}")
    kg = graph_kg(g)
    kg.data["triples"] = triples
    kg.data["compiled_from"] = {"schema": g.data.get("schema"), "sha256": g.sha256}
    kg.__init__(kg.data, kg.path)
    kg.graph = g
    return kg


__all__ = ["compile_pack", "PackCompileError", "CALL_STEP", "UNAVAILABLE"]


# =====================================================================================================================
# PART 2 - Compile a compact policy graph into the symbolic store and the GMS view knowlytix ingests.
#
# The graph is already structured, so there is no table parsing: every fact is a triple with its
# source (``values.card_product.PREMIUM_SUPP``, ``rules.R13``, ...), every number goes to the exact
# numeric register (ENM, integrity hashed), and the dialog-state layer comes with it.
#
# The one noisy source is the knowledge-base prose (``knowledge``). As in the v1 compiler, its
# figures are proposed as candidate facts and reconciled against the register before anything is
# accepted: a figure that matches is corroborated, one that conflicts is dropped (the critic
# ``inject_misread`` demonstrates on a planted misread), one with nothing to check against is
# flagged for the GEODE loop.
#
# ``render_gms_view`` writes the same compiled facts in the table shape knowlytix's GEODE ingest
# reads (one table per value set with numbers, so the register keys are ``<member>/<attribute>``;
# one table per relation otherwise), the way knowledge/store.py does for the v1 graph.
# =====================================================================================================================

from ..knowledge.store import CandidateTriple, Rule, Source, WorkflowEdge, canonical_number
from ..knowledge.store import ENMEntry, GMSStore
from .compact import graph_kg, rule_reification

# relations kept out of the store (wording, not facts the geometry should learn)
_NOT_IN_STORE = {"has_text", "has_unit", "states"}


def _tail(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return canonical_number(str(v))
    return str(v)


def compile_graph(path: str | Path, *, inject_misread: bool = False):
    """The symbolic store for a v2 graph, plus the self-correction log (geode.BuildResult)."""
    from ..knowledge.build import BuildResult, Correction
    g = PolicyGraph.load(path)
    kg = compile_pack(g)                      # the full graph (executable flow, speech layer) every component reads
    triples = to_triples(g)
    accepted: dict[tuple[str, str, str], list[dict]] = {}
    for t in triples:
        if t["relation"] in _NOT_IN_STORE:
            continue
        tr = (t["head"], t["relation"], _tail(t["tail"]))
        accepted.setdefault(tr, []).append({"section": t["source"], "ref": t["id"], "span": t.get("evidence", ""),
                                            "path": "policy_graph"})
    # exact numeric register
    enm: dict[tuple[str, str], ENMEntry] = {}

    def reg(cat: str, key: str, value: Any, unit: str, src: str):
        v = _tail(value)
        e = enm.get((cat, key)) or ENMEntry(cat, key, v, unit, ENMEntry.digest(cat, key, v, unit))
        if e.value != v:
            raise ValueError(f"conflicting register values for {cat}/{key}: {e.value} vs {v}")
        e.sources.append({"section": src, "ref": key, "span": "", "path": "policy_graph"})
        enm[(cat, key)] = e

    for k, lv in g.limits.items():
        reg("limit", k, lv["value"], lv.get("unit", ""), f"limits.{k}")
    from .compact import _unit
    for attr, spec in g.numeric_attributes().items():
        for m, v in spec["members"].items():
            reg(attr, m, v, _unit(attr), f"values.{spec['set']}.{m}")
    # knowledge-base prose: candidate figures reconciled against the register
    corrections: list[Correction] = []
    cands = prose_candidates(g)
    if inject_misread:
        attr, spec = next(iter(g.numeric_attributes().items()))
        m, v = max(spec["members"].items(), key=lambda kv: kv[1])
        cands.append(CandidateTriple(m, attr_relation(attr), _tail(v * 10 + 50 if v else 150),
                                     Source("knowledge base", "KB?", f"{m} ... {v * 10 + 50 if v else 150} (OCR misread)",
                                            "prose")))
    for c in cands:
        src = asdict(c.source)
        backbone = _register_value(g, enm, c)
        if backbone is None:
            corrections.append(Correction(1, "flag", c.key, "no exact register value to check against; left to the "
                                          "GEODE loop", src))
            accepted.setdefault(c.key, []).append(src)
        elif backbone == c.tail:
            corrections.append(Correction(1, "corroborate", c.key, "prose figure matches the exact register", src))
            accepted.setdefault(c.key, []).append(src)
        else:
            corrections.append(Correction(1, "drop_conflict", c.key,
                                          f"prose figure {c.tail} conflicts with register value {backbone}", src))
    rules = {}
    for rid, r in g.rules.items():
        subj, pred, obj = rule_reification(g, rid)
        rules[rid] = Rule(rid, r["stance"], subj, pred, obj, r["text"], Source("rules", rid, r["text"], "rule"))
    workflow = [WorkflowEdge(w["wid"], w["src"], w["dst"], w["condition"],
                             Source("workflow", w["wid"], f"{w['src']} -> {w['dst']}: {w['condition']}", "workflow"))
                for w in g.workflow()]
    value_sets = {vs: list(members) for vs, members in g.values.items()}
    aliases = {m: list(spec["aliases"]) for members in g.values.values() for m, spec in members.items()}
    store = GMSStore(title=g.title, triples=list(accepted), provenance=accepted, enm=enm, rules=rules,
                     workflow=workflow, value_sets=value_sets, aliases=aliases, entity_types=kg.entity_types(),
                     kb_sentences=[(f"KB{i}", s) for i, s in enumerate(g.knowledge, start=1)],
                     required_language={"opening": g.opening} if g.opening else {})
    store.kg = kg
    return BuildResult(store=store, corrections=corrections, iterations=1, n_candidates=len(triples) + len(cands),
                       converged=True, max_iterations=1, doc=Path(path).name)


# ---------------------------------------------------------------------------- prose figures
_MONEY = re.compile(r"\$(\d+(?:\.\d+)?)")
_COUNT = re.compile(r"(?<![\d$.])(\d+)\s+([a-z]+(?: [a-z]+)?)")


def prose_candidates(g: PolicyGraph) -> list[CandidateTriple]:
    """Figures the knowledge-base sentences state, bound to the member they name (naive binding:
    the member mentioned last before the figure), as candidate facts for the critics."""
    out = []
    num = g.numeric_attributes()
    names: dict[str, list[str]] = {}
    for spec in num.values():
        for m in spec["members"]:
            names[m] = [m.lower().replace("_", " "), m.split("_")[0].lower()] + [a.lower() for a in g.aliases(m)]
    for i, sent in enumerate(g.knowledge, start=1):
        low = sent.lower()
        src = Source("knowledge base", f"KB{i}", sent, "prose")

        def member_before(pos: int) -> str | None:
            best, best_pos = None, -1
            for m, ns in names.items():
                for n in ns:
                    for mm in re.finditer(r"\b" + re.escape(n) + r"\b", low):
                        if best_pos < mm.start() < pos:
                            best, best_pos = m, mm.start()
            return best

        for mm in _MONEY.finditer(sent):
            m = member_before(mm.start())
            attr = next((a for a, sp in num.items() if a.endswith("_usd") and (m in sp["members"] if m else True)), None)
            if attr:
                out.append(CandidateTriple(m or "unbound", attr_relation(attr), canonical_number(mm.group(1)), src))
        for mm in _COUNT.finditer(low):
            n, unit = mm.group(1), mm.group(2)
            if "day" in unit:
                m = member_before(mm.start())
                attr = next((a for a, sp in num.items() if "days" in a and (m in sp["members"] if m else True)), None)
                if attr:
                    out.append(CandidateTriple(m or "unbound", attr_relation(attr), canonical_number(n), src))
                continue
            for k, lv in g.limits.items():
                u = str(lv.get("unit", ""))
                if u and (unit.startswith(u.rstrip("s")) or unit.split()[-1].startswith(u.rstrip("s"))):
                    out.append(CandidateTriple(g.root, f"has_{k}", canonical_number(n), src))
                    break
                if k.endswith(unit.split()[-1].rstrip("s") + "s") or unit.split()[-1].rstrip("s") in k:
                    out.append(CandidateTriple(g.root, f"has_{k}", canonical_number(n), src))
                    break
    return out


def _register_value(g: PolicyGraph, enm: dict, c: CandidateTriple) -> str | None:
    if c.head == g.root and c.relation.startswith("has_"):
        e = enm.get(("limit", c.relation[4:]))
        return e.value if e else None
    for attr in g.numeric_attributes():
        if c.relation == attr_relation(attr):
            e = enm.get((attr, c.head))
            return e.value if e else None
    return None


# ---------------------------------------------------------------------------- views
def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(title: str, header: list[str], rows: list[list]) -> list[str]:
    if not rows:
        return []
    return [f"## {title}", "", _row(*header), _row(*["---"] * len(header)), *[_row(*r) for r in rows], ""]


def yes_no_words(attr: str) -> tuple[str, str, str]:
    """(column, word for yes, word for no): the GEODE ingest reads yes/no columns as pass/fail
    flags, so a yes/no attribute is written as words."""
    if attr.startswith("requires_"):
        return attr[len("requires_"):], "required", "not_required"
    if attr.startswith("is_"):
        return attr[3:], attr[3:], f"not_{attr[3:]}"
    return attr, attr, f"not_{attr}"


def render_gms_view(store, g: PolicyGraph) -> str:
    """The compiled policy (a GMSStore from compile_graph) as GEODE input."""
    lines = [f"# {store.title} (GMS view)", "",
             "Generated from the policy graph; every row is a compiled fact with source provenance.", ""]
    lim = sorted((e.key, e.value, e.unit) for e in store.enm.values() if e.category == "limit")
    lines += _table("Limits", ["limit", "value"], [[k, v] for k, v, _ in lim])
    lines += _table("Limit units", ["limit", "unit"], [[k, u] for k, _, u in lim if u])
    rendered = set()
    num = g.numeric_attributes()
    for vs in g.values:
        attrs = [a for a, sp in num.items() if sp["set"] == vs]
        if attrs:
            members = sorted({m for a in attrs for m in num[a]["members"]})
            lines += _table(vs.replace("_", " ").capitalize(), [vs, *attrs],
                            [[m, *[store.enm[(a, m)].value for a in attrs]] for m in members])
            rendered |= {attr_relation(a) for a in attrs}
        for a in g.attributes(vs):
            if a in attrs:
                continue
            vals = {m: spec.get(a) for m, spec in g.values[vs].items() if a in spec}
            if vals and all(v in ("yes", "no") for v in vals.values()):
                col, yes, no = yes_no_words(a)
                lines += _table(col.replace("_", " ").capitalize(), [vs, col],
                                [[m, yes if v == "yes" else no] for m, v in vals.items()])
                rendered.add(attr_relation(a))
    lines += _table("Workflow", ["step", "enables"], [[w.src, w.dst] for w in store.workflow])
    lines += _table("Value sets", ["value", "member_of"],
                    [[v, vs] for vs, vals in sorted(store.value_sets.items()) for v in vals])
    lines += _table("Aliases", ["entity", "alias"], [[c, a] for c, als in sorted(store.aliases.items()) for a in als])
    opp = [[a, b] for vs, vals in sorted(store.value_sets.items()) for i, a in enumerate(vals) for b in vals[i + 1:]]
    opp.append(["must", "must_not"])
    lines += _table("Opposites", ["entity", "opposite_of"], opp)
    skip = rendered | {"member_of", "has_alias", "enables", "has_stance", "constrains"}
    limit_rels = {f"has_{k}" for k in g.limits}
    by_rel: dict[str, list[list[str]]] = defaultdict(list)
    for h, r, t in sorted(store.triples):
        if r in skip or h in store.rules:
            continue
        if h == g.root and r in limit_rels:
            continue
        if h == "bot" and any(rr.subject == h and rr.predicate == r for rr in store.rules.values()):
            continue                                   # rule reifications: in the Rules table
        by_rel[r].append([h, t])
    for r, rows in sorted(by_rel.items()):
        col = r[4:] if r.startswith("has_") else r
        lines += _table(f"Relation {col}", ["entity", col], rows)
    lines += _table("Rules", ["rule", "stance", "subject", "predicate", "object"],
                    [[r.rule_id, r.stance, r.subject, r.predicate, r.obj] for r in store.rules.values()])
    return "\n".join(lines) + "\n"


def render_policy_markdown(g: PolicyGraph) -> str:
    """A readable rendering of the graph (the policy as a document)."""
    L = [f"# {g.title}", "", f"Rendered from {g.path.name if g.path else g.id} (sha256 {g.sha256[:16]}).", ""]
    if g.opening:
        L += ["## Opening", "", f'Required opening: "{g.opening}"', ""]
    L += ["## Limits", "", _row("limit", "value", "unit", "text"), _row(*["---"] * 4)]
    L += [_row(k, lv["value"], lv.get("unit", ""), lv.get("text", "")) for k, lv in g.limits.items()] + [""]
    for vs, members in g.values.items():
        attrs = g.attributes(vs)
        L += [f"## {vs}", "", _row(vs, *attrs, "aliases"), _row(*["---"] * (len(attrs) + 2))]
        L += [_row(m, *[spec.get(a, "") for a in attrs], "; ".join(spec["aliases"])) for m, spec in members.items()]
        L.append("")
    L += ["## Tools", "", _row("tool", "role", "stage", "ok", "fail", "text"), _row(*["---"] * 6)]
    L += [_row(t, sp["role"], sp.get("stage", ""), sp["ok"], ";".join(sp.get("fail", [])), sp.get("text", ""))
          for t, sp in g.tools.items()] + [""]
    L += ["## Rules", "", _row("rule", "stance", "text"), _row(*["---"] * 3)]
    L += [_row(rid, r["stance"], r["text"]) for rid, r in g.rules.items()] + [""]
    L += ["## Dialog", ""]
    for e in g.edges():
        L.append(f"- {e['src']} -> {e['dst']} on {e['on']}" + (f" when {e['when']}" if e["when"] != "always" else "")
                 + (f" (transfer: {e['reason']})" if e.get("reason") else ""))
    L += ["", "## Knowledge base", "", *g.knowledge, ""]
    return "\n".join(L)


def attribute_phrasings(g: PolicyGraph) -> dict[str, list[str]]:
    """Per-relation policy-question phrasings for the u-channel contradiction SFT, rendered by
    the pack's realizer from the graph's figures (labelled by construction)."""
    from ..rewording.wording import Realizer
    from ..knowledge.questions import kb_frames
    R = Realizer()
    by_rel: dict[str, list] = defaultdict(list)
    for f in kb_frames(g):
        rel = "has_value" if f.get("intent") == "limit" else attr_relation(f.get("attr"))
        by_rel[rel].append(f)
    out = {}
    for rel, frs in by_rel.items():
        ph = set()
        for fr in frs:
            for pd in ("none", "light", "deep"):
                for qf in ("interrogative", "declarative"):
                    ph.add(R.realize(fr, {"paraphrase_depth": pd, "question_format": qf}, "u"))
        out[rel] = sorted(ph)
    return out
