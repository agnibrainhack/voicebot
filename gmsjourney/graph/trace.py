"""Scoring a run against a compact policy graph.

``GraphTrace`` is the v1 ``Trace`` with every policy-specific reading replaced by one mined from
the graph, so the oracle's expected trace and a bot's observed trace are scored identically:

* ``signature()``     the outcome: end state, transfer reason, every write that truly succeeded
                      (success status with a matching echo) with its arguments, and every
                      outcome act the policy requires the bot to say (states marked
                      ``outcome``: the ineligibility reason, the age limit, ...)
* ``components()``    per stage (the graph's stages): the tool calls of that stage with their
                      statuses and the outcome acts of that stage; the handoff; whether every
                      success claim is backed by a success; the policy answers
* ``admissibility()`` one violation per breach of a machine-checkable rule, labelled
                      ``<rule>:<what>`` (requires_auth, max_calls, requires_status,
                      value_allowed, requires_slot, disclose_before, confirm_before,
                      claim_needs_success, say_result, handoff_when_requested, turn_limit,
                      no_call_after_handoff), plus tool sequences not in the workflow
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..trace import Trace
from .model import ECHO_MISMATCH, UNAVAILABLE

SUMMARY_STAGE, QUESTION_STAGE, SAID_STAGE = "claims", "question", "said"


def _graph():
    from ..domain import current
    return current().graph


def scored_stages(g) -> list[str]:
    """The stages a run is scored on, in order (the conversation stages that decide something)."""
    first, last = g.stage(g.start), g.stage(g.end_state)
    out = [s for s in g.stages if s not in (first, last)]
    return out + [SAID_STAGE, SUMMARY_STAGE, QUESTION_STAGE]


@dataclass
class GraphTrace(Trace):
    told: list[dict[str, Any]] = field(default_factory=list)          # outcome acts the bot said
    claims: list[tuple[str, dict[str, Any]]] = field(default_factory=list)   # (tool, act) success claims
    visited: list[str] = field(default_factory=list)
    decisions: list[tuple[str, str]] = field(default_factory=list)   # (decision point, option) on a mined path

    # -------------------------------------------------------------- tool facts
    @staticmethod
    def _handoff() -> tuple[str, str]:
        T = _graph().transfers
        return T["tool"], T.get("arg", "reason")

    def status_of(self, c: dict[str, Any]) -> str:
        g = _graph()
        if c.get("ok") is False:
            return UNAVAILABLE
        sp = g.tools.get(c["tool"], {})
        r = c.get("result") or {}
        st = r.get("status")
        if st == sp.get("ok") and sp.get("echo") and r.get(sp["echo"]) != (c.get("args") or {}).get(sp["echo"]):
            return ECHO_MISMATCH
        return st

    def succeeded(self, c: dict[str, Any]) -> bool:
        return self.status_of(c) == _graph().tools.get(c["tool"], {}).get("ok")

    def writes(self) -> list[list]:
        g = _graph()
        out = []
        for c in self.tool_log:
            if g.tools.get(c["tool"], {}).get("role") == "write" and self.succeeded(c):
                out.append([c["tool"], {k: c["args"].get(k) for k in sorted(c.get("args") or {})}])
        return out

    def recorded(self) -> dict[str, str]:
        return {f"{t}#{i}": json.dumps(a, sort_keys=True) for i, (t, a) in enumerate(self.writes())}

    def told_items(self) -> list[list]:
        return [[a["act"], {k: v for k, v in sorted(a.items()) if k != "act"}] for a in self.told]

    # -------------------------------------------------------------- outcome
    def signature(self) -> str:
        return json.dumps({"end": self.end_state, "transfer_reason": self.transfer_reason,
                           "writes": self.writes(), "told": self.told_items()}, sort_keys=True, default=str)

    def completed_action(self):
        w = self.writes()
        return tuple(w[-1]) if w else None

    # -------------------------------------------------------------- components
    def summary_consistent(self) -> bool:
        g = _graph()
        for tool, act in self.claims:
            echo = g.tools[tool].get("echo")
            ok = [c for c in self.tool_log if c["tool"] == tool and self.succeeded(c)]
            if not ok:
                return False
            if echo and echo in act and act[echo] != ok[-1]["args"].get(echo):
                return False
        return True

    def components(self) -> dict[str, str]:
        g = _graph()
        stages = scored_stages(g)
        out = {}
        handoff = self._handoff()[0]
        for st in stages:
            if st in (SUMMARY_STAGE, QUESTION_STAGE, SAID_STAGE):
                continue
            parts = []
            for c in self.tool_log:
                if c["tool"] == handoff:
                    continue
                if g.tools.get(c["tool"], {}).get("stage") == st:
                    parts.append(f"{c['tool']}:{self.status_of(c)}")
            for a in self.told:
                if g.stage(a["act"]) == st:
                    parts.append(a["act"] + "(" + ",".join(f"{k}={v}" for k, v in sorted(a.items()) if k != "act") + ")")
            if st == g.stage(g.transfer_state):
                parts = [self.transfer_reason or "-"]
            out[st] = "|".join(parts) or "-"
        out[SAID_STAGE] = "|".join(self.said()) or "-"
        out[SUMMARY_STAGE] = "CONSISTENT" if self.summary_consistent() else "VIOLATION"
        if self.kb:
            out[QUESTION_STAGE] = "CORRECT" if all(k.get("correct") for k in self.kb) else "INCORRECT"
        return out

    def said(self) -> list[str]:
        """The informational acts the bot made, in order: every say state of the graph except the
        opening (what it told the customer, as opposed to what it asked)."""
        g = _graph()
        keep = {s for s in g.states if g.kind(s) == "say" and s != g.start}
        sm = (g.transfers.get("summary") or {}).get("act")
        if sm:
            keep.add(sm)
        return [a["act"] for _, a in self._bot_acts() if a.get("act") in keep]

    # -------------------------------------------------------------- trajectory
    def node_sequence(self) -> list[str]:
        seq = ["start"] + [c["tool"] for c in self.tool_log]
        if self.ended or self.transferred:
            seq.append("end")
        return seq

    def _bot_acts(self) -> list[tuple[int, dict]]:
        return [(i, a) for i, e in enumerate(self.events) if e.get("role") == "bot" for a in e.get("acts", [])]

    def _tool_events(self) -> list[tuple[int, dict]]:
        return [(i, e) for i, e in enumerate(self.events) if e.get("role") == "tool"]

    def admissibility(self, workflow, limits: dict[str, Any]) -> list[str]:
        g = _graph()
        v: list[str] = []
        _, unknown = self.edges(workflow)
        v += [f"edge_not_in_policy_graph:{a}->{b}" for a, b in unknown]
        log = self.tool_log
        roles = {t: sp["role"] for t, sp in g.tools.items()}
        events = self.events
        tool_ev = self._tool_events()
        bot_acts = self._bot_acts()
        user = [(i, e["frame"]) for i, e in enumerate(events) if e.get("role") == "user"]

        def ok_event(e):
            return self.succeeded({"tool": e["tool"], "args": e.get("args") or {}, "result": e.get("result"),
                                   "ok": e.get("ok", not ("error" in (e.get("result") or {})))})

        for rid, kind, c in g.rule_checks():
            if kind == "requires_auth":
                authed = False
                for call in log:
                    if roles.get(call["tool"]) == "auth" and self.succeeded(call):
                        authed = True
                    if roles.get(call["tool"]) in ("read", "write") and not authed:
                        v.append(f"{rid}:account_tool_before_auth:{call['tool']}")
            elif kind == "no_call_after_handoff":
                h = self._handoff()[0]
                seen = False
                for call in log:
                    if seen:
                        v.append(f"{rid}:tool_after_transfer:{call['tool']}")
                    if call["tool"] == h:
                        seen = True
            elif kind == "max_calls":
                t, lim = c["max_calls"], limits[c["limit"]]
                if len(self.calls(t)) > lim:
                    v.append(f"{rid}:{t}_calls_exceeded")
            elif kind == "turn_limit":
                if self.turns > limits[c["turn_limit"]]:
                    v.append(f"{rid}:turn_limit_exceeded")
            elif kind == "requires_status":
                t, need, st = c["requires_status"], c["needs"], c["status"]
                for i, e in tool_ev:
                    if e["tool"] != t:
                        continue
                    prior = [p for j, p in tool_ev if j < i and p["tool"] == need]
                    if not prior or (prior[-1].get("result") or {}).get("status") != st:
                        v.append(f"{rid}:{t}_without_{need}_{st}")
            elif kind == "value_allowed":
                t, arg, attr, vals = c["value_allowed"], c["arg"], c["attr"], c["values"]
                for call in self.calls(t):
                    if g.attr(call["args"].get(arg), attr) not in vals:
                        v.append(f"{rid}:{t}_{arg}_not_allowed")
            elif kind == "requires_slot":
                t, slot, vals = c["requires_slot"], c["slot"], c["values"]
                for i, e in tool_ev:
                    if e["tool"] != t:
                        continue
                    said = [f.get(slot) for j, f in user if j < i and f.get(slot) is not None]
                    if not said or said[-1] not in vals:
                        v.append(f"{rid}:{t}_without_{slot}_{'_'.join(vals)}")
            elif kind == "disclose_before":
                t, arg, attr, vals, act = c["disclose_before"], c["arg"], c["attr"], c["values"], c["act"]
                for i, e in tool_ev:
                    if e["tool"] != t or g.attr((e.get("args") or {}).get(arg), attr) not in vals:
                        continue
                    want = (e.get("args") or {}).get(arg)
                    disc = [j for j, a in bot_acts if j < i and a.get("act") == act and a.get(arg, want) == want]
                    if not disc:
                        v.append(f"{rid}:{t}_without_{act}")
                        continue
                    conf = [j for j, f in user if disc[-1] < j < i and f.act in ("CONFIRM",)]
                    if not conf:
                        v.append(f"{rid}:{t}_without_confirmation_after_{act}")
            elif kind == "confirm_before":
                t, act = c["confirm_before"], c["act"]
                for i, e in tool_ev:
                    if e["tool"] != t:
                        continue
                    prior = [f for j, f in user if j < i]
                    if not prior or prior[-1].act != act:
                        v.append(f"{rid}:{t}_without_{act}")
            elif kind == "claim_needs_success":
                act, t = c["claim_needs_success"], c["tool"]
                echo = g.tools[t].get("echo")
                for i, a in bot_acts:
                    if a.get("act") != act:
                        continue
                    ok = [e for j, e in tool_ev if j < i and e["tool"] == t and ok_event(e)]
                    if not ok or (echo and echo in a and a[echo] != (ok[-1].get("args") or {}).get(echo)):
                        v.append(f"{rid}:claim_without_success")
            elif kind == "say_result":
                act, fld, t = c["say_result"], c["field"], c["tool"]
                for i, a in bot_acts:
                    if a.get("act") != act:
                        continue
                    prior = [e for j, e in tool_ev if j < i and e["tool"] == t]
                    truth = (prior[-1].get("result") or {}).get(fld) if prior else None
                    if a.get(fld) != truth:
                        v.append(f"{rid}:{act}_{fld}_mismatch")
            elif kind == "value_from_caller":
                t_, arg, act, slot = c["value_from_caller"], c["arg"], c["act"], c["slot"]
                for i, e in tool_ev:
                    if e["tool"] != t_:
                        continue
                    said = [f.get(slot) for j, f in user if j < i and f.act == act]
                    if not said or said[-1] != (e.get("args") or {}).get(arg):
                        v.append(f"{rid}:{t_}_{arg}_not_the_callers_answer")
            elif kind == "terminal_value":
                t_, arg, val = c["terminal_value"], c["arg"], c["value"]
                h = self._handoff()[0]
                for i, e in tool_ev:
                    if e["tool"] == t_ and (e.get("args") or {}).get(arg) == val and ok_event(e):
                        later = [x for j, x in tool_ev if j > i]
                        if not later or later[0]["tool"] != h:
                            v.append(f"{rid}:no_handoff_after_{arg}_{val}")
            elif kind == "offered_only":
                t_, arg, src, fld = c["offered_only"], c["arg"], c["source"], c["field"]
                for i, e in tool_ev:
                    if e["tool"] != t_:
                        continue
                    prior = [x for j, x in tool_ev if j < i and x["tool"] == src]
                    offered = ((prior[-1].get("result") or {}).get(fld) or []) if prior else []
                    if (e.get("args") or {}).get(arg) not in offered:
                        v.append(f"{rid}:{t_}_{arg}_not_offered")
            elif kind == "item_window":
                t_, arg, src, fld, age, lim = (c["item_window"], c["arg"], c["source"], c["field"], c["age"],
                                               limits[c["limit"]])
                for i, e in tool_ev:
                    if e["tool"] != t_:
                        continue
                    prior = [x for j, x in tool_ev if j < i and x["tool"] == src]
                    items = {str(x.get(arg)): x for x in ((prior[-1].get("result") or {}).get(fld) or [])} if prior else {}
                    it = items.get(str((e.get("args") or {}).get(arg)))
                    if it is None:
                        v.append(f"{rid}:{t_}_{arg}_not_returned")
                    elif it.get(age) is not None and it[age] > lim:
                        v.append(f"{rid}:{t_}_{arg}_outside_{c['limit']}")
            elif kind == "max_distinct":
                t_, arg = c["max_distinct"], c["arg"]
                if len({str(x["args"].get(arg)) for x in self.calls(t_)}) > limits[c["limit"]]:
                    v.append(f"{rid}:{t_}_{arg}_more_than_{c['limit']}")
            elif kind == "handoff_when_requested":
                h = self._handoff()[0]
                human = (g.glob.get("human") or {}).get("act")
                for i, f in user:
                    if f.act != human:
                        continue
                    later = [e for j, e in tool_ev if j > i]
                    if not later or later[0]["tool"] != h:
                        v.append(f"{rid}:no_handoff_after_request")
                    break
        # a counted re-prompt ({"count": c, "upto": L, "then": S}) is asked at most L times per loop item
        for state, lim, rid in _counted_prompts(g):
            n, worst, prev = 0, 0, None
            starts = {g.states[e]["do"] for e in g.states if g.kind(e) == "each"}
            reask = {(g.glob.get("unclear") or {}).get("clarify"), (g.glob.get("question") or {}).get("answer")}
            for _, a in bot_acts:
                if a.get("act") in starts:
                    n = 0
                elif a.get("act") == state and prev not in reask:   # a re-ask after a clarification is not a new prompt
                    n += 1
                    worst = max(worst, n)
                prev = a.get("act")
            if worst > limits[lim]:
                v.append(f"{rid}:{state}_prompted_more_than_{lim}")
        if not self.summary_consistent() and not any(":claim_without_success" in x for x in v):
            v.append("claim_without_success")
        return v


def _counted_prompts(g) -> list[tuple[str, str, str]]:
    """(prompt state, limit, rule) for every counter target that leads to a prompt."""
    out = []
    for name, s in g.states.items():
        for tgt in (s.get("on") or {}).values():
            stack = [tgt]
            while stack:
                x = stack.pop()
                if isinstance(x, list):
                    stack += [a.get("then") if isinstance(a, dict) and "then" in a else a for a in x]
                elif isinstance(x, dict) and "count" in x:
                    then = x["then"]
                    if isinstance(then, str) and then in g.states and g.kind(then) == "ask":
                        rid = (g.states[then].get("rules") or [""])[0]
                        out.append((then, x["upto"], rid))
                    stack.append(x["else"])
    return list(dict.fromkeys(out))
