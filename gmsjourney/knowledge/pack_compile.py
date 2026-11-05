"""Compile a compact policy graph (``gmsjourney.policy_graph/2``) into the full policy graph every component reads.

A compact pack states its dialog as a state machine (``graph/model.py``): ask, say, call and end states, guarded
targets, transfers with reasons and the always-available behaviours (a person on request, a policy question, an
unclear answer, the turn limit). The rest of the harness reads one format, the head / relation / tail graph of
``knowledge/policy_kg.py``, whose dialog is executable flow (``flow.py``). This module writes that graph:

* every fact the compact graph states (``graph.model.to_triples``: limits, value sets with aliases and attributes,
  acts and slots, tools with roles and statuses, rules and the facts they state, transfer reasons, workflow,
  knowledge sentences, the opening), with the descriptive ``leads_to`` edges replaced by
* the **flow**: ``leads_to`` triples whose qualifiers are expressions of the flow language (``on``, ``pre``,
  ``guard``, ``do``, ``say``, ``rules``, ``quiet``, ``flush``, ``retract``). Call states become quiet tool steps
  (``is_a tool_step``) left on the tool's status; the always-available behaviours become transitions out of every
  prompt state, in the machine's order of precedence (a person, a policy question, the state's own acts, an
  unclear answer); the turn limit is checked before a prompt is said (``any`` + ``retract``);
* ``carries`` (what each act carries, from the states' ``say`` fields), ``has_arg`` (what each tool is called
  with, from its call state), ``returns_field``;
* the **speech layer**: ``says`` (each field an act must or may say, bound to where its truth comes from),
  ``is_outcome`` and ``claims_tool`` for the acts the scoring reads.

The compiled walk is checked against the compact graph's own interpreter (``graph/machine.py``) case for case
(``tests/test_pack_compile.py``).
"""
from __future__ import annotations

import json
from typing import Any

from ..graph.model import ECHO_MISMATCH, NEXT, UNAVAILABLE, PolicyGraph, attr_relation, to_triples


class PackCompileError(ValueError):
    pass


CALL_STEP = "tool_step"            # is_a class of a compiled call state (not a bot act)


def _q(v: Any) -> str:
    """A Python literal for an expression."""
    return repr(v)


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

    # ------------------------------------------------------------------ expressions
    def attr_expr(self, value_expr: str, attr: str) -> str:
        if attr in self.num:                          # numbers come from exact memory
            return f"enm({_q(attr)}, {value_expr})"
        return f"tail({value_expr}, {_q(attr_relation(attr))})"

    def ref(self, ref: Any) -> str:
        """A field value of the compact graph ($opening, $<limit>, $<tool>.<field>, $<slot>[.<attr>]) as an
        expression."""
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
        counter = target.get("count") if isinstance(target, dict) and "count" in target else None
        for t, c in alts:
            g2 = list(guard or []) + self.cond(c, src)
            d2 = list(do or [])
            if counter:
                ok = "count_upto" in c
                lim = c.get("count_upto") or c.get("count_over")
                g2.append(f"n({_q('count:' + counter)}) + 1 {'<=' if ok else '>'} limit({_q(lim)})")
                d2.append(f"inc({_q('count:' + counter)})")
            self._edge(src, on, t, g2, pre, d2, say, rules, when or json.dumps(c, sort_keys=True) if c else when)

    def _edge(self, src, on, t, guard, pre, do, say, rules, when) -> None:
        if t == NEXT:
            raise PackCompileError(f"{src}: loops (each-states, @next) are not supported by the compiler yet")
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
                 rules=[*(rules or []), *self.into(t)], quiet=(kind == "call"))

    # ------------------------------------------------------------------ the machine
    def compile(self) -> None:
        g, G = self.g, self.g.glob
        human, question, unclear, tl = G.get("human"), G.get("question"), G.get("unclear"), G.get("turn_limit")
        global_acts = {x["act"] for x in (human, question) if x}
        for name, s in g.states.items():
            kind = g.kind(name)
            if kind == "each":
                raise PackCompileError(f"{name}: loops (each-states) are not supported by the compiler yet")
            if kind == "say":
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
            if self.T.get("summary"):
                raise PackCompileError("transfers.summary is not supported by the compiler yet")
            step = f"{self.ts}__HANDOFF"
            self.row(self.ts, step, "auto", when=f"call {self.handoff}", quiet=True, flush=True)
            self.row(step, self.T.get("then", g.end_state), "auto", when=f"{self.handoff} called",
                     guard=[f"call({_q(self.handoff)}) or True"], quiet=True)
            self.call_args[self.handoff] = {self.handoff_arg: "$reason"}
            self.steps = [step]
        else:
            self.steps = []

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
    from ..graph.model import graph_kg
    c = _Compiler(g)
    c.compile()
    triples = [t for t in to_triples(g) if t["relation"] != "leads_to"]
    G = g.glob
    call_states = [s for s in g.states if g.kind(s) == "call"]
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
    for st in getattr(c, "steps", []):
        add(st, "is_a", CALL_STEP, "dialog", "transfers")
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
        if g.kind(name) == "call":
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
