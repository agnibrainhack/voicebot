"""Plain-English case summaries for the answer sheet: what happened, turn by turn, and why the call ended
the way it did.

Two layers:

1. ``case_facts`` reduces one base case to a small fact sheet, entirely from the oracle's own trace and the
   policy graph: what the customer meant in each turn, every dialog-state hop with its trigger, the tool
   calls and results that decided it, the graph's transition condition and the rule texts, and the expected
   outcome. Nothing in it is written by a model.
2. ``LLMNarrator`` (optional) asks the model on the gateway for a short readable account written from the
   SAME fact sheet, and keeps it only if it passes a check: no figure, rule id or tool name that is not in
   the facts, and the actual ending is named. A reply that fails is retried once with the reason, then
   dropped (the answer sheet's deterministic story stays either way). Replies are cached like every other
   LLM call, so re-building the sheet costs nothing.
   ``template_summary`` is a plain-text deterministic account of the same facts (used outside the sheet).
"""
from __future__ import annotations

import json
import re

_TAG = re.compile(r"<[^>]+>")


def _plain(html: str) -> str:
    import html as _h
    return re.sub(r"\s+", " ", _h.unescape(_TAG.sub("", html or ""))).strip()


# ----------------------------------------------------------------------------- hops
def graph_edge(kg, src: str | None, dst: str | None, on: str | None) -> dict | None:
    """The policy graph's leads_to triple for one hop with the same states AND trigger, or None (a hop whose
    trigger differs from every graph transition between those states is not passed off as one of them)."""
    if kg is None or not src or not dst:
        return None
    cand = [t for t in kg.where("leads_to", head=src) if t["tail"] == dst]
    return next((t for t in cand if (t.get("qualifiers") or {}).get("on") == on), None)


def turn_hops(kg, st: str | None, trigger: str | None, steps: list[tuple[str, dict]]) -> list[dict]:
    """Split a turn's ordered steps (("act", act) | ("tool", event)) into dialog-state hops:
    {src, on, tools, dst, act}. The customer's meaning triggers the first hop; the bot's own moves are 'auto'."""
    known = set(kg.states) if kg is not None else set()
    hops, cur, trig, tools = [], st, (trigger or "call start"), []
    for kind, x in steps:
        if kind == "tool":
            tools.append(x)
        elif x.get("act") in known:
            hops.append({"src": cur, "on": trig, "tools": tools, "dst": x.get("act"), "act": x})
            cur, trig, tools = x.get("act"), "auto", []
    ended = bool(tools and tools[-1].get("tool") == "transfer_to_human") or cur in ("TRANSFER", "END")
    if tools or (ended and cur != "END"):
        hops.append({"src": cur, "on": trig, "tools": tools, "dst": "END" if ended else None, "act": None})
    return hops


def tool_line(ev: dict) -> str:
    args = ", ".join(f"{k}={v}" for k, v in (ev.get("args") or {}).items())
    res = ev.get("result") or {}
    status = res.get("status") or ("error: " + str(res.get("error")) if res.get("error") else "")
    return f"{ev.get('tool')}({args}) -> {status}"


# ----------------------------------------------------------------------------- facts
def case_facts(case_id: str, kind: str, outcome: dict, turns: list[dict], kg, question: dict | None = None) -> dict:
    """turns: [{turn, state, customer_says, customer_means, steps:[(kind, x)], bot_says:[str]}] -> fact sheet."""
    out_turns = []
    for t in turns:
        hops = []
        for h in turn_hops(kg, t.get("state"), t.get("trigger"), t["steps"]):
            e = graph_edge(kg, h["src"], h["dst"], h["on"])
            q = (e or {}).get("qualifiers") or {}
            hops.append({"from": h["src"] or "start", "to": h["dst"] or "(stays)", "trigger": h["on"],
                         "tool_calls": [tool_line(x) for x in h["tools"]],
                         "condition": q.get("when") or ("call begins" if h["src"] is None else None),
                         "rules": [{"id": r, "text": kg.rule_text(r)} for r in q.get("rules") or []] if kg else [],
                         "graph_triple": (e or {}).get("id")})
        out_turns.append({"turn": t["turn"], "state": t.get("state"), "customer_says": t.get("customer_says"),
                          "customer_means": t.get("customer_means"), "transitions": hops,
                          "bot_says": [_plain(x) for x in t.get("bot_says") or []]})
    return {"case": case_id, "kind": kind, "question_under_test": question,
            "expected_outcome": {"end": outcome.get("end"), "transfer_reason": outcome.get("transfer_reason"),
                                 "recorded": outcome.get("recorded")},
            "turns": out_turns}


def _ending(facts: dict) -> str:
    o = facts["expected_outcome"]
    if o.get("end") == "TRANSFERRED":
        return f"transferred to a human (reason: {str(o.get('transfer_reason') or 'unspecified').replace('_', ' ')})"
    return {"COMPLETED": "completed normally", "ENDED": "ended normally"}.get(str(o.get("end")),
                                                                           str(o.get("end") or "ended").lower())


def template_summary(facts: dict) -> str:
    """Deterministic account from the fact sheet (no model)."""
    parts = []
    q = facts.get("question_under_test")
    if q:
        parts.append(f"This case tests one policy question: “{q.get('question')}” (expected answer: {q.get('answer')}).")
    for t in facts["turns"]:
        hs = [h for h in t["transitions"] if h["to"] != "(stays)"]
        if not hs:
            continue
        s = f"Turn {t['turn']}: the customer {t['customer_means']}"
        bits = []
        for h in hs:
            why = h["condition"] or "the policy graph allows it"
            tools = f" ({'; '.join(h['tool_calls'])})" if h["tool_calls"] else ""
            rule = f" [{', '.join(r['id'] for r in h['rules'])}]" if h["rules"] else ""
            bits.append(f"{h['from']} → {h['to']} because {why}{tools}{rule}")
        parts.append(s + "; " + "; then ".join(bits) + ".")
    last = next((h for t in reversed(facts["turns"]) for h in reversed(t["transitions"]) if h["to"] != "(stays)"),
                None)
    end = f"The call is {_ending(facts)}"
    if last and last.get("rules"):
        end += f", as required by {last['rules'][0]['id']}: “{last['rules'][0]['text']}”"
    rec = facts["expected_outcome"].get("recorded")
    if rec:
        end += f"; recorded responses: {', '.join(f'{k} {v}' for k, v in (rec.items() if isinstance(rec, dict) else rec))}"
    parts.append(end + ".")
    return " ".join(parts)


# ----------------------------------------------------------------------------- LLM
NARRATE_SYSTEM = """You explain test cases for a bank's charge-review voice bot to a reader who does not know the
system. You receive a FACT SHEET of one test case: each customer turn (what the customer said and meant), each
dialog-state transition the bot makes (from, to, trigger, the tool calls and results that decided it, the
transition condition and the policy rules) and the expected outcome.
Write a short plain-English account (4 to 8 sentences):
- what happens, turn by turn, in everyday words (group routine turns together);
- above all WHY the call ends the way it does: name the decisive turn, the tool result or customer request
  that caused it, and the policy rule behind it (quote its id in brackets, e.g. [R20]);
- use ONLY facts in the sheet: no other numbers, merchants, tools, states or rule ids; do not judge any bot.
Reply with JSON only: {"summary": "..."}"""


class LLMNarrator:
    """Model-written summaries from the fact sheet, checked against it; falls back to the template."""

    def __init__(self, settings=None, workers: int = 8, log=print):
        from .enrich.llm.client import CachedLLM, LLMSettings
        s = settings or LLMSettings()
        self.llm = CachedLLM(LLMSettings(**{**s.__dict__, "temperature": 0.0, "max_tokens": 450}), tag="narrate")
        self.workers, self.log = workers, log
        self.stats = {"llm": 0, "template_fallback": 0}

    @staticmethod
    def _prompt(facts: dict, feedback: str | None = None) -> tuple[str, str]:
        user = {"fact_sheet": facts}
        if feedback:
            user["previous_summary_rejected_because"] = feedback
        return NARRATE_SYSTEM, json.dumps(user, ensure_ascii=False, indent=1, default=str)

    @staticmethod
    def check(text: str, facts: dict) -> list[str]:
        """Reasons to reject a model summary (empty = acceptable)."""
        blob = json.dumps(facts, ensure_ascii=False, default=str)
        r = []
        ids = re.compile(r"\b(?:R\d{1,2}|T\d{3,5}|W\d{1,2}|J\d{3})\b")      # rule / triple / edge / case ids
        nums = set(re.findall(r"\d+(?:\.\d+)?", ids.sub(" ", blob)))
        extra = set()
        for m in re.finditer(r"(\$\s?)?(\d+(?:\.\d+)?)(\s*(?:dollars?|days?|business days?|%))?", ids.sub(" ", text)):
            n = m.group(2)
            quantity = bool(m.group(1) or m.group(3))
            if n not in nums and (quantity or float(n) > 20):    # small bare counts ("two turns", "turn 3") are fine
                extra.add(n)
        extra = sorted(extra)
        if extra:
            r.append(f"mentions figures not in the facts: {', '.join(extra)}")
        rules = set(re.findall(r"\bR\d{1,2}\b", blob))
        bad_rules = sorted(set(re.findall(r"\bR\d{1,2}\b", text)) - rules)
        if bad_rules:
            r.append(f"cites rules not in the facts: {', '.join(bad_rules)}")
        tools = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b(?=\()", blob))
        bad_tools = sorted({w for w in re.findall(r"\b[a-z]+(?:_[a-z]+){1,4}\b", text)
                            if w.endswith(("_customer", "_transactions", "_response", "_options", "_action", "_human"))
                            and w not in tools})
        if bad_tools:
            r.append(f"names tools not in the facts: {', '.join(bad_tools)}")
        end = str(facts["expected_outcome"].get("end") or "").lower()
        if end == "transferred" and "transfer" not in text.lower():
            r.append("does not say the call ends in a transfer")
        if not text.strip() or len(text) > 2000:
            r.append("empty or too long")
        return r

    def _one(self, facts: dict, live: bool = True) -> tuple[str, str]:
        from .enrich.llm.client import LLMUnavailable, parse_json_object
        feedback = None
        for attempt in range(2):
            system, user = self._prompt(facts, feedback)
            try:
                reply = self.llm.invoke(system, user, salt=f"narrate|{facts['case']}|{attempt}")
            except LLMUnavailable:
                break
            obj = parse_json_object(reply) or {}
            text = str(obj.get("summary") or "").strip()
            why = self.check(text, facts)
            if not why:
                return text, "llm"
            feedback = "; ".join(why)
        return template_summary(facts), "template"

    def summarize_all(self, facts_list: list[dict]) -> dict[str, tuple[str, str]]:
        """case id -> (summary, source); source "llm" (checked) or "template" (the model was unavailable or
        its reply failed the check). First attempts go out concurrently; retries are sequential."""
        if self.llm.available:
            from .enrich.llm.validate import _pmap
            _pmap(lambda f: self.llm.invoke(*self._prompt(f), salt=f"narrate|{f['case']}|0"), facts_list,
                  self.workers, "LLM case summaries")
        out = {}
        for f in facts_list:
            text, src = self._one(f)
            self.stats["llm" if src == "llm" else "template_fallback"] += 1
            out[f["case"]] = (text, src)
        if self.log:
            self.log(f"      case summaries: {self.stats['llm']} written by the LLM, "
                     f"{self.stats['template_fallback']} template (LLM unavailable or reply failed the check)")
        return out
