"""A reference bot for a compact policy graph, and the defects (mutants) mined from the graph.

``GraphBot`` follows the graph with the same interpreter as the oracle (graph/machine.py), one
turn at a time, against the live tools it is given. It understands customers with its own
``KeywordReader``: the graph's canonical names and customer synonyms only, bare yes / no words
and a 4-digit code. That reader is deliberately simpler than the frame reader that proves the
invariance (no typo or speech repair, no wording from the lexicon), so on enriched wording the
reference bot fails where a plain keyword bot would.

Mutants are derived from the graph, one per thing a bot could get wrong:

  status_as_ok:<tool>        treat a failure status as success (skip auth, ignore a decline)
  ignore_echo:<tool>         accept a success that echoes another value
  skip_say:<state>           never say this (a disclosure, a limit explanation, a reason)
  skip_ask:<state>           act without asking (submit without a confirmation)
  ignore_guard:<state>       always take the default branch of a guarded choice
  limit_plus_one:<limit>     an off-by-one on a limit
  wrong_field:<state>.<f>    say another value than the one the policy gives
  no_human                   ignore a request for a person
  no_turn_limit              ignore the turn limit
  clarify_forever            never give up on an unclear answer
"""
from __future__ import annotations

import re
from typing import Any

from ..ontology import SemanticFrame as F
from ..sut.reference_bot import BotTurn
from .machine import GraphOracle, _Stop
from .model import ECHO_MISMATCH, NEXT, UNAVAILABLE

_YES = r"\b(yes|yeah|yep|sure|go ahead|please do|correct|ok|okay|proceed)\b"
_NO = r"\b(no|nope|don't|dont|cancel|not now)\b"
_END = r"\b(that's all|thats all|nothing else|no thanks|all set|that's everything|im done|i'm done)\b"
_HUMAN = r"\b(human|real person|representative|an agent|live agent|transfer me|speak to someone|a person)\b"


class KeywordReader:
    """The reference bot's own understanding: the domain's value and act cues (the pack's lexicon,
    as a bot built for this domain would know them) plus the graph's synonyms, bare yes / no and
    a 4-digit code - but none of the proof reader's normalisation: no typo or speech-recognition
    repair, no number words, no reported speech, no correction markers."""

    def __init__(self, g):
        self.g = g
        from ..enrich.lexicon import Lexicon
        lx = Lexicon.current()
        self.cues: dict[str, list[tuple[str, str]]] = {}
        for vs, members in g.values.items():
            rows = []
            for m, spec in members.items():
                for a in [m.lower().replace("_", " ")] + [a.lower() for a in spec["aliases"]]:
                    rows.append((r"\b" + re.escape(a.replace("'", "")) + r"\b", m))
                for c in (lx.values.get(m) or {}).get("cues", []):
                    rows.append((r"\b(" + c + r")\b", m))
            self.cues[vs] = rows
        self.act_cues = {a: r"\b(" + "|".join(sp["cues"]) + r")\b" for a, sp in lx.acts.items() if sp.get("cues")}

    def find(self, vs: str, text: str) -> str | None:
        """The member named last in the text."""
        found = sorted(((mm.start(), mm.end(), m) for pat, m in self.cues.get(vs, []) for mm in re.finditer(pat, text)),
                       key=lambda x: (x[0], -(x[1] - x[0])))
        kept, end = [], -1
        for s, e, m in found:                      # earliest-longest, non-overlapping
            if s >= end:
                kept.append(m)
                end = e
        return kept[-1] if kept else None

    def read(self, text: str, state: str) -> F:
        g = self.g
        t = (text or "").lower().replace("’", "'").replace("'", "")
        human = (g.glob.get("human") or {}).get("act")
        if human in self.act_cues and re.search(self.act_cues[human], t):
            return F.make(human)
        G = g.glob
        if G.get("human") and re.search(_HUMAN, t):
            return F.make(G["human"]["act"])
        s = g.states.get(state) or {}
        on = s.get("on") or {}
        if G.get("question") and "?" in t:
            q = self._question(t)
            if q is not None:
                return q
        for act in on:
            specs = g.slot_specs(act)
            if not specs:
                continue
            kw = {}
            for sl, sp in specs.items():
                if sp["set"] == "credential_code":
                    m = re.findall(r"\b(\d{4})\b", t)
                    if m:
                        kw[sl] = m[-1]
                elif not sp.get("correction_of"):
                    v = self.find(sp["set"], t)
                    if v is not None:
                        kw[sl] = v
            req = [sl for sl, sp in specs.items() if not sp.get("optional")]
            if kw and all(r in kw for r in req):
                return F.make(act, **kw)
        if "END" in on and (re.search(_END, t) or ("END" in self.act_cues and re.search(self.act_cues["END"], t))):
            return F.make("END")
        y, n = g.interprets(state)
        if re.search(_YES, t) and not re.search(_NO, t):
            return self._as(y or "CONFIRM")
        if re.search(_NO, t):
            return self._as(n or "DENY")
        return F.make(G["unclear"]["act"] if G.get("unclear") else "UNCLEAR")

    def _as(self, target: str) -> F:
        vs = self.g.member_set(target)
        if vs:
            for a in self.g.acts:
                for sl, s in self.g.acts[a].items():
                    if s == vs:
                        return F.make(a, **{sl: target})
        return F.make(target)

    def _question(self, t: str) -> F | None:
        g = self.g
        from ..enrich.lexicon import Lexicon
        kbq = Lexicon.current().kb
        if kbq.get("attributes") or kbq.get("limits"):          # the domain's question wording
            has = lambda cues, s: any(re.search(r"\b(" + c + r")\b", s) for c in cues)
            num = g.numeric_attributes()
            for attr, spec in (kbq.get("attributes") or {}).items():
                vs = (num.get(attr) or {}).get("set")
                ent = self.find(vs, t) if vs else None
                if ent is None:
                    continue
                rest = t
                for pat, m in self.cues.get(vs, []):
                    if m == ent:
                        rest = re.sub(pat, " ", rest)
                if has(spec["cues"], rest):
                    return F.make("ASK_KB", intent="value", attr=attr, entity=ent)
            for key, spec in (kbq.get("limits") or {}).items():
                if has(spec["cues"], t):
                    return F.make("ASK_KB", intent="limit", key=key)
            return None
        num = g.numeric_attributes()
        for attr, spec in num.items():
            ent = self.find(spec["set"], t)
            if ent is None:
                continue
            money = attr.endswith("_usd")
            if money and re.search(r"\b(fee|cost|how much|price|pay)\b", t):
                return F.make("ASK_KB", intent="value", attr=attr, entity=ent)
            if not money and re.search(r"\b(how long|days|arrive|when)\b", t):
                return F.make("ASK_KB", intent="value", attr=attr, entity=ent)
        for k, lv in g.limits.items():
            if lv.get("internal"):
                continue
            words = [w for w in k.split("_") if len(w) > 3 and w not in ("max", "min", "per")]
            if words and any(re.search(r"\b" + w.rstrip("s"), t) for w in words):
                return F.make("ASK_KB", intent="limit", key=k)
        return None


def mutants(g) -> list[str]:
    """Every defect the graph makes possible."""
    out = []
    for name, s in g.states.items():
        k = g.kind(name)
        if k == "call":
            t = s["call"]
            if g.tools[t].get("fail") and g.tools[t]["role"] != "handoff":
                out.append(f"status_as_ok:{t}")
            if g.tools[t].get("echo"):
                out.append(f"ignore_echo:{t}")
        elif k == "say" and (s.get("rules") or s.get("outcome")) and name not in (g.start,):
            out.append(f"skip_say:{name}")
        elif k == "ask":
            if "CONFIRM" in s["on"]:
                out.append(f"skip_ask:{name}")
            if any(isinstance(tg, list) for tg in s["on"].values()):
                out.append(f"ignore_guard:{name}")
    for name, s in g.states.items():
        if g.kind(name) == "say" and isinstance(s.get("go"), list):
            out.append(f"ignore_guard:{name}")
    lims = set()
    for e in g.edges():
        try:
            import json
            c = json.loads(e["when"]) if e["when"].startswith("{") else {}
        except ValueError:
            c = {}
        lims |= {c[x] for x in ("below", "at_least", "count_upto", "count_over") if x in c}
    for s in g.states.values():
        if s.get("cap"):
            lims.add(s["cap"])
        for w in (s.get("where") or {}).values():
            if w.get("max"):
                lims.add(w["max"])
    out += [f"limit_plus_one:{x}" for x in sorted(lims)]
    for rid, kind, c in g.rule_checks():
        if kind == "say_result":
            out.append(f"wrong_field:{c['say_result']}.{c['field']}")
    if g.glob.get("human"):
        out.append("no_human")
    if g.glob.get("turn_limit"):
        out.append("no_turn_limit")
    if g.glob.get("unclear"):
        out.append("clarify_forever")
    return list(dict.fromkeys(out))


class GraphBot(GraphOracle):
    """The reference bot: start() / respond(text) -> BotTurn, tools observed at the mock backend."""

    def __init__(self, tools, store, mutations: frozenset = frozenset()):
        super().__init__(store)
        self.m = set(mutations)
        self.reader = KeywordReader(self.g)
        env = tools if hasattr(tools, "log") else getattr(tools, "env", tools)
        self.start_session(tools, env)
        self._state: str | None = None
        self.ended = False

    # -------------------------------------------------------- mutation hooks
    def L(self, name: str) -> Any:
        v = super().L(name)
        if f"limit_plus_one:{name}" in self.m:
            return v + 1
        if "clarify_forever" in self.m and self.g.glob.get("unclear", {}).get("limit") == name:
            return 10 ** 6
        if "no_turn_limit" in self.m and self.g.glob.get("turn_limit", {}).get("limit") == name:
            return 10 ** 6
        return v

    def _call(self, tool: str, args: dict[str, Any]) -> tuple[str, dict]:
        status, res = super()._call(tool, args)
        sp = self.g.tools[tool]
        if f"status_as_ok:{tool}" in self.m and status in sp.get("fail", []):
            status = sp["ok"]
        if f"ignore_echo:{tool}" in self.m and status == ECHO_MISMATCH:
            status = sp["ok"]
        return status, res

    def _target(self, target: Any, state: str) -> str:
        if f"ignore_guard:{state}" in self.m and isinstance(target, list):
            target = target[-1]
        return super()._target(target, state)

    def act_of(self, state: str) -> dict[str, Any]:
        act = super().act_of(state)
        for k in list(act):
            if f"wrong_field:{state}.{k}" in self.m:
                vs = self.g.member_set(act[k]) if isinstance(act[k], str) else None
                others = [m for m in (self.g.members(vs) if vs else []) if m != act[k]]
                act[k] = others[0] if others else None
        return act

    def advance(self, state: str) -> str:
        g = self.g
        while True:
            if state.startswith("!"):
                return super().advance(state)
            k = g.kind(state)
            s = g.states[state]
            if k == "say" and f"skip_say:{state}" in self.m:
                self._tr.visited.append(state)
                state = self._target(s["go"], state)
                if state == NEXT:
                    state = g.loop_of[self._tr.visited[-1]]
                continue
            if k == "ask" and f"skip_ask:{state}" in self.m:
                nxt = self._target(s["on"][next(iter(s["on"]))], state)
                state = g.loop_of.get(state, state) if nxt == NEXT else nxt
                continue
            # run the oracle's advance one step at a time so the hooks above see every state
            return self._advance_one(state)

    def _advance_one(self, state: str) -> str:
        g = self.g
        k = g.kind(state)
        if k in ("ask", "end") or state.startswith("!"):
            return super().advance(state)
        # a say / call / each state: do it, then continue with the hooks
        s = g.states[state]
        tr = self._tr
        tr.visited.append(state)
        tr.rules_fired.update(s.get("rules", []))
        if k == "say":
            act = self.act_of(state)
            self._pending.append(act)
            self._note(state, act)
            nxt = self._target(s["go"], state)
        elif k == "call":
            args = {a: self.resolve(ref) for a, ref in (s.get("args") or {}).items()}
            status, _ = self._call(s["call"], args)
            nxt = self._target(s["on"].get(status, s["on"].get(UNAVAILABLE)), state)
        else:
            nxt = self._each(state)
        if nxt == NEXT:
            nxt = g.loop_of[state]
        return self.advance(nxt)

    def react(self, state: str, f: F) -> str | None:
        G = self.g.glob
        if "no_human" in self.m and G.get("human") and f.act == G["human"]["act"]:
            f = F.make(G["unclear"]["act"]) if G.get("unclear") else f
        return super().react(state, f)

    def _answer_kb(self, frame: F) -> dict[str, Any]:
        """Policy answers from the bot's own copy of the graph (not the store)."""
        g = self.g
        intent = frame.get("intent")
        if intent == "value":
            v = g.attr(frame.get("entity"), frame.get("attr"))
            what = frame.get("attr").replace("_usd", "").replace("_", " ")
            ent = frame.get("entity").replace("_", " ").lower()
            text = f"The {what} for {ent} is " + (f"${v}." if frame.get("attr").endswith("_usd") else f"{v}.")
        elif intent == "limit":
            v = g.limit_value(frame.get("key"))
            text = f"The limit is {v}."
        else:
            v, text = None, "I don't have information on that; a representative can help."
        self._tr.kb.append({"frame": frame, "answer": v, "correct": True})
        return {"act": g.glob["question"]["answer"], "intent": intent, "answer": v, "text": text}

    # -------------------------------------------------------- the bot interface
    def _turn(self, ended: bool) -> BotTurn:
        acts = [a for e in self._tr.events[self._mark:] if e["role"] == "bot" for a in e["acts"]]
        self.ended = self.ended or ended
        return BotTurn(" ".join(_say(a) for a in acts), acts, self.ended)

    def start(self) -> BotTurn:
        self._mark = len(self._tr.events)
        try:
            self._state = self.advance(self.g.start)
            self._emit()
            return self._turn(False)
        except _Stop:
            return self._turn(True)

    def respond(self, text: str) -> BotTurn:
        if self.ended or self._state is None:
            return BotTurn("", [], True)
        self._mark = len(self._tr.events)
        st = self._state
        f = self.reader.read(text, st)
        self.record_user(st, f)
        try:
            nxt = self.react(st, f)
            self._state = self.advance(st if nxt is None else nxt)
            self._emit()
            return self._turn(False)
        except _Stop:
            return self._turn(True)


def _say(a: dict[str, Any]) -> str:
    if a.get("text"):
        return str(a["text"])
    fields = ", ".join(f"{k}={v}" for k, v in a.items() if k != "act")
    return f"[{a['act']}{': ' + fields if fields else ''}]"


def make_graph_factory(store, mutations: frozenset = frozenset()):
    def factory(tools):
        return GraphBot(tools, store, frozenset(mutations))
    return factory
