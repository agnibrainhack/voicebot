"""Case by case: what happened in each run, expected beside actual.

Parts of this file:
  1. Plain-English summaries of a case (what happened turn by turn, why the call ended), optionally by an LLM.
  2. Turn by turn: the oracle's tools and acts for each customer turn next to the bot's.
  3. ``write_case_comparison``: one HTML page per campaign, every run's expected outcome beside the bot's.
  4. ``show_case``: look inside one base case and its reworded variants (for a notebook).
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# =====================================================================================================================
# PART 1 - Plain-English case summaries for the answer sheet: what happened, turn by turn, and why the call ended
#
# the way it did.
#
# Two layers:
#
# 1. ``case_facts`` reduces one base case to a small fact sheet, entirely from the oracle's own trace and the
#    policy graph: what the customer meant in each turn, every dialog-state hop with its trigger, the tool
#    calls and results that decided it, the graph's transition condition and the rule texts, and the expected
#    outcome. Nothing in it is written by a model.
# 2. ``LLMNarrator`` (optional) asks the model on the gateway for a short readable account written from the
#    SAME fact sheet, and keeps it only if it passes a check: no figure, rule id or tool name that is not in
#    the facts, and the actual ending is named. A reply that fails is retried once with the reason, then
#    dropped (the answer sheet's deterministic story stays either way). Replies are cached like every other
#    LLM call, so re-building the sheet costs nothing.
#    ``template_summary`` is a plain-text deterministic account of the same facts (used outside the sheet).
# =====================================================================================================================

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
    from ..s01_packs import names as roles
    from ..s01_packs.names import N
    end = N.state.END
    ended = bool(tools and tools[-1].get("tool") == roles.TRANSFER) or cur in (N.state.TRANSFER, end)
    if tools or (ended and cur != end):
        hops.append({"src": cur, "on": trig, "tools": tools, "dst": end if ended else None, "act": None})
    return hops


def tool_line(ev: dict) -> str:
    """One tool call as a line: ``tool(arg=value, ...) -> status``."""
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
        from ..llm import CachedLLM, LLMSettings
        s = settings or LLMSettings(role="testing")
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
        from ..llm import LLMUnavailable, parse_json_object
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
            from ..s05_rewording.llm_proof import _pmap
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


# =====================================================================================================================
# PART 2 - Turn-by-turn: what the oracle says the bot should do, next to what the bot did.
#
# For one scenario (base or enriched) the oracle's expected run and the bot's actual run
# are cut into customer turns. For every turn the table shows:
#
# * the dialog state the customer is answering (the bot act that prompted the turn) and
#   the customer's words (the text the bot received) with their meaning card;
# * EXPECTED (oracle): the tool calls the bot should make in response, with arguments and
#   results, and the bot acts it should produce (e.g. ``PRESENT_TXN(T2: Starbucks $6.25)``,
#   ``ANSWER_KB(value -> 0)``, ``TRANSFER(card_options_unavailable)``). The oracle is an
#   executable reading of the policy: it decides *what* to do, not the wording;
# * ACTUAL (bot): the tool calls the bot made, the acts it produced (structured acts from the
#   bot, or ``BotActClassifier`` when the bot returns text only) and its text;
# * tools_match (same tools, same key arguments, in order), acts_match (same decision acts:
#   optional acknowledgements such as CONFIRM_RECORDED and a re-presentation of the same
#   charge after a clarification are ignored, see ``normalize_acts``), turn_match (both), and
#   the first turn where the two runs diverge;
# * speech_check: what the speech check (s07_checks/speech_*.py) found in the bot's WORDS that turn -
#   "ok", or each false / unsupported claim and missing or unfaithful act. The oracle comparison
#   above is on tools and acts; this column is where a right act said wrongly shows up.
#
# The customer is simulated reactively (journey.UserSimulator): after the first divergence
# the bot has asked something the oracle did not, so the conversations may legitimately
# differ; those rows are marked ``after_divergence`` and are diagnostic only.
#
# How this relates to grading (evaluation/runner.score): the campaign does not require the
# bot's reply at turn i to equal the oracle's at turn i. It grades the whole run - the
# outcome, the per-stage decisions, and whether every step was admissible - because a bot
# may legitimately take a different but allowed route. This module is the turn-level view of
# the same two runs, to see where and how a failed run went wrong.
# =====================================================================================================================

from ..s06_bots.tools import key_args

ACT_FIELDS = ("txn", "intent", "answer", "reason", "recorded", "action", "delivery", "fee", "disposition")


# ----------------------------------------------------------------------------- rendering
def items_named(fx: dict, *event_lists: list[dict]) -> dict[str, tuple[str, str]]:
    """item id -> (merchant, amount) for every charge the case's bank returns: from the fixture (its ``transactions``
    or a tool's scripted results) and from the tool results in the runs' events."""
    out: dict[str, tuple[str, str]] = {}

    def take(v: Any) -> None:
        if isinstance(v, dict):
            if v.get("transaction_id") and v.get("merchant"):
                out.setdefault(str(v["transaction_id"]), (str(v["merchant"]), str(v.get("amount", ""))))
            for x in v.values():
                take(x)
        elif isinstance(v, list):
            for x in v:
                take(x)
    take(fx)
    for evs in event_lists:
        take([e.get("result") for e in evs or [] if e.get("role") == "tool"])
    return out


def _label(t: Any, merchants: dict | None) -> str:
    """A charge by merchant and amount (Amazon $45.67), or its id when the bank named none."""
    m = (merchants or {}).get(str(t)) if t is not None else None
    if not m:
        return str(t)
    return f"{m[0]} ${m[1]}" if m[1] not in ("", "None") else m[0]


def opening_text(acts: list[dict]) -> str:
    """The oracle's opening acts in the reference wording (the policy's opening statement, then its first question)."""
    from ..s06_bots.reference import _say
    try:
        return " ".join(t for t in (_say(a) for a in acts if not str(a.get("act", "")).startswith("(")) if t)
    except Exception:                                  # noqa: BLE001 - a pack without bot wording
        return ""


def describe_act(a: dict, merchants: dict[str, tuple[str, str]] | None = None) -> str:
    """A bot act as a short readable string: PRESENT_TXN(Starbucks $6.25)."""
    name = a.get("act", "?")
    if name == "OPENING":
        return "OPENING"
    parts = []
    if a.get("txn"):
        parts.append(_label(a["txn"], merchants))
    if name == "ANSWER_KB" and (a.get("intent") or a.get("answer") is not None):
        parts.append(f"{a.get('intent')} -> {json.dumps(a.get('answer'), default=str)}")
    for k in ("reason", "action", "delivery", "disposition"):
        if a.get(k):
            parts.append(str(a[k]))
    if a.get("fee") is not None and name == "DISCLOSE_FEE":
        parts.append(f"${a['fee']}")
    if name == "SUMMARIZE" and a.get("recorded") is not None:
        rec = a["recorded"]
        parts.append(", ".join(f"{_label(k, merchants)} = {v}" for k, v in (rec.items() if isinstance(rec, dict) else rec))
                     or "nothing")
    # a pack's own act fields (the charge-review fields above are rendered their own way)
    parts += [f"{k}={v}" for k, v in a.items()
              if k not in ACT_FIELDS and k not in ("act", "text", "state") and v is not None and not isinstance(v, (dict, list))]
    return f"{name}({'; '.join(parts)})" if parts else name


def describe_tool(e: dict, merchants: dict | None = None) -> str:
    """One tool call as a line, keeping only its key arguments: ``tool(key=value) -> status`` (a charge named by
    merchant and amount)."""
    args = e.get("args") or {}
    keep = {k: v for k, v in args.items() if k in key_args().get(e["tool"], tuple(args))}
    res = e.get("result")
    status = (res.get("status") or res.get("error")) if isinstance(res, dict) else ("error" if e.get("ok") is False else res)
    a = ", ".join(f"{k}={_label(v, merchants) if str(v) in (merchants or {}) else v}" for k, v in keep.items())
    return f"{e['tool']}({a}) -> {status}"


# The turn comparison is shared with the harness (s06_bots/turns.py): acknowledgements and side steps (a policy answer,
# a clarification) never move a turn off the journey.
from ..s06_bots.turns import (OPTIONAL_ACTS, REASK_AFTER, compare_turn, normalize_acts, side_acts,  # noqa: E402,F401
                              split_turns, tool_key as _tool_key)


# ----------------------------------------------------------------------------- segmentation
def segment(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Cut a run into (opening bot acts, per customer turn: prompt state, customer, tools, bot acts/text, and
    ``steps``: the calls and acts in the order they happened; see s06_bots/turns.split_turns)."""
    op, turns = split_turns(events)
    states, last = [], None
    for ev in events:
        if ev.get("role") == "bot" and ev.get("acts"):
            last = ev["acts"][-1].get("act")
        elif ev.get("role") == "user":
            states.append(last)
    for t, st in zip(turns, states):
        t["state"] = st
    opening = [a if k == "say" else {"act": f"(tool {a.get('tool')})"} for k, a in op["steps"]]
    return opening, turns


def steps_of(steps: list, merchants: dict | None = None) -> list[dict]:
    """A turn's steps in order, as [{"k": "calls" | "says", "d": text}] (stored with the turn, drawn by ``_col``)."""
    return [{"k": "calls", "d": describe_tool(x, merchants)} if k == "call" else {"k": "says", "d": describe_act(x, merchants)}
            for k, x in steps]


def speech_by_turn(events: list[dict], speech_turns: list[dict]) -> tuple[str, list[str]]:
    """What the speech check said about the bot's words: (opening, one entry per customer turn).

    Input   the run's trace events and ``extras["speech"]["turns"]`` (one report per bot event, in order).
    Output  "ok", or the turn's problems in words ("contradicted disposition 'recognize' (truth UNRECOGNIZED) ;
            missing PRESENT_TXN: date"); "" where the bot did not speak or speech was not checked."""
    def words(rep: dict | None) -> str:
        if rep is None:
            return ""
        bad = [f"{c['status']} {c['field']} '{c['words']}'" + (f" (truth {c['truth']})" if c.get("truth") is not None else "")
               for c in rep["claims"] if c["status"] in ("contradicted", "unsupported")]
        bad += [f"{f['kind']} {f['act']}: {f['detail']}" for f in rep["findings"]]
        return " ; ".join(bad) or "ok"
    opening, per_turn, k, cur = "", [], 0, None
    for ev in events:
        if ev.get("role") == "user":
            per_turn.append([])
            cur = per_turn[-1]
        elif ev.get("role") == "bot":
            w = words(speech_turns[k] if k < len(speech_turns) else None)
            k += 1
            if cur is None:
                opening = w
            elif w:
                cur.append(w)
    return opening, [" | ".join(x) for x in per_turn]


def reports_by_turn(events: list[dict], speech_turns: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """The speech check's report for every agent reply, cut like ``split_turns``: (the opening's, per customer
    turn)."""
    opening, per_turn, k, cur = [], [], 0, None
    for ev in events:
        if ev.get("role") == "user":
            per_turn.append([])
            cur = per_turn[-1]
        elif ev.get("role") == "bot":
            rep = speech_turns[k] if k < len(speech_turns) else None
            k += 1
            if rep is not None:
                (opening if cur is None else cur).append(rep)
    return opening, per_turn


def says_record(says: list[dict], merchants: dict | None = None) -> dict[str, Any]:
    """The oracle's acts held in an agent reply (s06_bots/oracle_says.py), compact for the page: each expected act
    with whether the words hold it, and what the words said that the oracle did not expect."""
    rows, extra, oracle, sents = [], [], False, []
    for h in says or []:
        oracle = oracle or bool(h.get("oracle"))
        rows += [{"act": describe_act(r.get("label") or {"act": r.get("act")}, merchants), "status": r.get("status"),
                  "how": r.get("how"), "detail": r.get("detail")} for r in h.get("says") or []]
        extra += [{"act": x.get("act"), "how": x.get("how"), "closing": bool(x.get("closing")),
                   "sentence": (h.get("sentences") or [{}] * (x.get("sentence", 0) + 1))[x.get("sentence", 0)].get("text")}
                  for x in h.get("extra") or []]
        ext = {x.get("sentence"): ("closing words" if x.get("closing") else x.get("act")) for x in h.get("extra") or []}
        sents += [{"text": x.get("text"), "act": x.get("act") or ext.get(k)} for k, x in enumerate(h.get("sentences") or [])]
    return {"oracle": oracle, "says": rows, "extra": extra, "sentences": sents}


def claims_record(reports: list[dict], merchants: dict | None = None) -> dict[str, Any]:
    """Every claim card read from an agent reply and its check, and the reply's findings (speech check). A charge is
    named by merchant and amount (``merchants``: id -> (merchant, amount))."""
    claims, finds, ok, core = [], [], True, True
    name = lambda v: _label(v, merchants) if merchants and str(v) in merchants else v          # noqa: E731
    for rep in reports or []:
        ok, core = ok and bool(rep.get("ok", True)), core and bool(rep.get("core_ok", rep.get("ok", True)))
        adv = set(rep.get("advisory") or ["unread"])
        for c in rep.get("claims") or []:
            d = {k: c.get(k) for k in ("field", "value", "words", "about", "mode", "status", "truth", "basis",
                                       "source", "why", "side")}
            d["about"] = name(d["about"])
            if d["field"] == "item":
                d["value"], d["truth"] = name(d["value"]), name(d["truth"])
            for k in ("source", "why"):
                if merchants and isinstance(d.get(k), str):
                    d[k] = re.sub(r"\b(" + "|".join(map(re.escape, sorted(merchants, key=len, reverse=True))) + r")\b",
                                  lambda m: _label(m.group(1), merchants), d[k])
            claims.append(d)
        finds += [{"kind": f.get("kind"), "act": f.get("act"), "detail": f.get("detail"), "side": f.get("side", False),
                   "advisory": f.get("kind") in adv} for f in rep.get("findings") or []]
    return {"checked": bool(reports), "ok": ok, "core_ok": core, "claims": claims, "findings": finds}


# ----------------------------------------------------------------------------- one scenario
def turn_by_turn(runner, oracle, scenario, faults=None, validator=None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expected (oracle) vs actual (bot) for one scenario, one row per customer turn."""
    item = scenario.base
    spec, fx = item.metadata["spec"], item.metadata["fixture"]
    expected = oracle.run(spec, fx)
    path = validator.base_path(item) if validator is not None else None
    res = runner.run(scenario, faults=faults, oracle_path=path)
    actual = res.extras["trace"]
    merchants = items_named(fx, expected.events, actual.events)
    _, exp_turns = segment(expected.events)
    _, act_turns = segment(actual.events)
    ont = runner.ont
    side = side_acts()
    speech_open, speech_turns = speech_by_turn(actual.events, ((res.extras.get("speech") or {}).get("turns") or []))
    rep_open, rep_turns = reports_by_turn(actual.events, ((res.extras.get("speech") or {}).get("turns") or []))
    statuses = {s_["turn"]: s_ for s_ in (res.extras.get("turn_status") or [])}
    stopped = res.extras.get("stopped")
    rows, diverged_at = [], None
    answer_act = ((runner.graph.glob.get("question") or {}).get("answer") if getattr(runner, "graph", None) is not None
                  else "ANSWER_KB")
    ended_at = res.extras.get("bot_ended_at")
    played = set()

    def row_for(i, e, a, line, repeat, status) -> dict:
        st = (e or a)["state"]
        frame = (a or e)["frame"]
        cmp = compare_turn(e["tools"] if e else [], e["acts"] if e else [], a["tools"] if a else [],
                           a["acts"] if a else [], side) if e is not None and a is not None else None
        status = status or (cmp["status"] if cmp else "not_played" if a is None else "differs")
        kb_wrong = []
        if a is not None:                              # this turn's policy answers, each judged where it was given
            for j in a.get("kb") or []:
                if not j.get("correct"):
                    kb_wrong.append(j.get("detail") or "the policy answer is not the policy's")
        if kb_wrong and status in ("same", "side"):
            status = "kb_wrong"
        if a is not None and e is not None and any(x.get("act") == answer_act for x in e["acts"]) \
                and not any(x.get("act") == answer_act for x in a["acts"]):
            kb_wrong.append("the customer's policy question was not answered")
        card = ont.card(frame, st) if frame is not None else {}
        text = (a or {}).get("text") or ""
        if not text and e is not None and path is not None and line is not None and line < len(path):
            f0, ctx = path[line]                      # what the customer would have said on the expected path
            att = scenario.metadata.get("seed_overrides", {}).get(line, 0)
            seed = (f"{scenario.sid}|{line}" + (f"#{att}" if att else "")) if scenario.factor_levels else f"{item.qid}|{line}|base"
            text = "(not played) " + runner.realizer.realize(f0, scenario.factor_levels, seed, ctx)
        e_acts = normalize_acts(e["acts"]) if e else []
        a_acts = normalize_acts(a["acts"]) if a else []
        return {
            "turn": i, "line": line, "repeat": bool(repeat), "state": st,
            "customer_text": text,
            "customer_card": json.dumps({"act": card.get("act"), "slots": card.get("slots")}) if card else "",
            "expected_tools": " ; ".join(describe_tool(x, merchants) for x in (e["tools"] if e else [])) or ("-" if e else "(no turn)"),
            "actual_tools": " ; ".join(describe_tool(x, merchants) for x in (a["tools"] if a else [])) or ("-" if a else "(no turn)"),
            "expected_bot": " ; ".join(describe_act(x, merchants) for x in (e["acts"] if e else [])) or "-",
            "actual_bot_acts": " ; ".join(describe_act(x, merchants) for x in (a["acts"] if a else [])) or "-",
            "expected_steps": steps_of(e["steps"], merchants) if e else [],
            "actual_steps": steps_of(a["steps"], merchants) if a else [],
            "actual_bot_text": " ".join(a["bot_text"]) if a else "",
            "speech_check": speech_turns[i] if a is not None and i < len(speech_turns) else "",
            "actual_says": says_record(a.get("says"), merchants) if a is not None else {},
            "actual_claims": claims_record(rep_turns[i] if i < len(rep_turns) else [], merchants) if a is not None else {},
            "kb_answers": [{"answer": j.get("answer"), "correct": bool(j.get("correct")), "detail": j.get("detail")}
                           for j in (a.get("kb") or [])] if a is not None else [],
            "expected_next_state": e_acts[-1][0] if e_acts else "-",
            "actual_next_state": a_acts[-1][0] if a_acts else "-",
            "tools_match": bool(cmp and cmp["tools_ok"]), "acts_match": bool(cmp and cmp["acts_ok"]),
            "turn_match": status == "same", "turn_status": status, "kb_wrong": " ; ".join(map(str, kb_wrong)),
            "after_divergence": diverged_at is not None and i > diverged_at,
            "after_bot_ended": ended_at is not None and line is not None and line >= ended_at,
        }

    for i, a in enumerate(act_turns):
        line = a.get("line") if a.get("line") is not None else i
        played.add(line)
        e = exp_turns[line] if line < len(exp_turns) else None
        r = row_for(i, e, a, line, a.get("repeat"), (statuses.get(i) or {}).get("status"))
        if r["turn_status"] in ("differs", "asked_again", "wrong_action") and diverged_at is None:
            diverged_at = i
        rows.append(r)
    for line in range(len(exp_turns)):               # lines the run never reached (it stopped, or the bot ended)
        if line not in played and (stopped is not None or line >= len(act_turns)):
            rows.append(row_for(len(rows), exp_turns[line], None, line, False, "not_played"))
    from ..s07_checks.scoring import score
    row = score(scenario, res, oracle.limits, oracle.store.workflow)
    count = lambda st: sum(r["turn_status"] == st for r in rows)          # noqa: E731
    summary = {
        "case_id": scenario.sid, "base_id": item.qid, "turns_expected": len(exp_turns), "turns_actual": len(act_turns),
        "first_divergence_turn": diverged_at, "turns_matching": sum(r["turn_match"] for r in rows),
        "turns_side": count("side"), "turns_asked_again": count("asked_again"), "turns_kb_wrong": count("kb_wrong"),
        "turns_off": count("differs"), "turns_wrong_action": count("wrong_action"),
        "turns_not_played": count("not_played"),
        "stopped_at_turn": (stopped or {}).get("turn"), "stopped_at_line": (stopped or {}).get("line"),
        "stopped_why": (stopped or {}).get("why"),
        "stopped_detail": stopped, "stop_rule": res.extras.get("stop_rule"),
        "items": merchants,                               # charge id -> (merchant, amount), to name charges on the page
        "bot_ended_at": ended_at,
        "expected_outcome": json.loads(item.answer) if isinstance(item.answer, str) else item.answer,
        "actual_outcome": json.loads(res.answer) if isinstance(res.answer, str) else res.answer,
        "outcome_correct": bool(row.get("outcome_correct", row["correct"])),
        "correct": bool(row["correct"]),                         # outcome, and (by default) grounded words
        "workflow_adherent": bool(row["workflow_adherent"]),
        "violations": row["violations"],
        "speech_ok": bool(row.get("c_speech", 1)), "speech_problems": row.get("speech_problems") or [],
        "speech_opening": speech_open,
        "stages_wrong": [k[2:] for k, v in row.items() if k.startswith("c_") and v == 0],
        "expected_opening": " ; ".join(describe_act(x, merchants) for x in segment(expected.events)[0]),
        "actual_opening": " ; ".join(describe_act(x, merchants) for x in segment(actual.events)[0]),
        "expected_opening_steps": steps_of(split_turns(expected.events)[0]["steps"], merchants),
        "actual_opening_steps": steps_of(split_turns(actual.events)[0]["steps"], merchants),
        "actual_opening_text": " ".join(split_turns(actual.events)[0]["bot_text"]),
        "actual_opening_says": says_record(split_turns(actual.events)[0]["says"], merchants),
        "actual_opening_claims": claims_record(rep_open, merchants),
        "expected_opening_text": opening_text(segment(expected.events)[0]),
    }
    return pd.DataFrame(rows), summary


def show_turns(runner, oracle, scenario, validator=None, display=None):
    """Print one case turn by turn (expected vs actual tools, bot acts, whether words were grounded) and return the
    table and its summary.
    """
    df, s = turn_by_turn(runner, oracle, scenario, validator=validator)
    show = display or (lambda d: print(d.to_string(index=False)))
    print(f"{s['case_id']}: outcome {'CORRECT' if s['outcome_correct'] else 'WRONG'} | "
          f"words {'grounded' if s['speech_ok'] else 'NOT grounded'} | "
          f"turns matching {s['turns_matching']}/{max(s['turns_expected'], s['turns_actual'])} | "
          f"first divergence: {s['first_divergence_turn']} | stages wrong: {s['stages_wrong'] or '-'}")
    print("expected outcome:", s["expected_outcome"])
    print("actual outcome:  ", s["actual_outcome"])
    show(df[["turn", "state", "customer_text", "expected_tools", "actual_tools", "expected_bot", "actual_bot_acts",
             "turn_match", "speech_check"]])
    return df, s


# ----------------------------------------------------------------------------- many scenarios
def export_turn_by_turn(out_dir: str | Path, runner, oracle, scenarios, validator=None,
                        show: bool = True, judges=None, kg=None, judge_max: int | None = None,
                        judge_workers: int = 4) -> dict[str, Any]:
    """dataset/turn_by_turn.csv (one row per turn per scenario) + turn_by_turn_summary.csv. With ``judges``
    (s07_checks/llm_judges.Judges) the two LLM judges also read every run (failed runs first, at most ``judge_max``)
    beside the rules' verdict: dataset/llm_judges.jsonl and the judge columns of the summary."""
    from ..s07_checks.scoring import progress_bar
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trows, srows, payloads = [], [], []
    bar = progress_bar(len(scenarios), "turn-by-turn export", show)
    for scn in scenarios:
        if bar is not None:
            bar.update(1)
        df, s = turn_by_turn(runner, oracle, scn, validator=validator)
        for r in df.to_dict(orient="records"):
            trows.append({"case_id": scn.sid, "base_id": scn.base.qid,
                          "case_type": "base" if not scn.factor_levels else "enriched",
                          "arm": scn.metadata.get("arm", "base" if not scn.factor_levels else "template"), **r})
        srows.append({**s, "case_type": "base" if not scn.factor_levels else "enriched",
                      "arm": scn.metadata.get("arm", "base" if not scn.factor_levels else "template")})
        if judges is not None:
            recs = df.to_dict(orient="records")
            from ..s07_checks.llm_judges import rules_of
            payloads.append({"case_id": scn.sid, "turns": recs, "expected_outcome": s["expected_outcome"],
                             "actual_outcome": s["actual_outcome"], "expected_opening": s["expected_opening"],
                             "actual_opening": s["actual_opening"], "stopped": s.get("stopped_detail"),
                             "rules": rules_of(kg, [r.get("state") for r in recs]),
                             "rules_ok": bool(s["correct"] and s["workflow_adherent"])})

    if bar is not None:
        bar.close()
    judged = None
    if judges is not None and payloads:
        from ..s07_checks.llm_judges import summarize
        from .. import progress as _progress
        payloads.sort(key=lambda c: c["rules_ok"])         # failed runs first
        todo = payloads[:judge_max] if judge_max is not None else payloads
        jb = _progress.bar(len(todo), "LLM judges", show=False, key="judges", unit="runs", label="Runs read by the two LLM judges")
        res = []
        for k in range(0, len(todo), max(1, judge_workers) * 2):
            chunk = todo[k:k + max(1, judge_workers) * 2]
            res += judges.judge_many(chunk, workers=judge_workers)
            jb.update(len(chunk))
        jb.close()
        by = {r["case_id"]: r for r in res}
        for s_ in srows:
            r = by.get(s_["case_id"])
            s_["judge_a"] = r["a"]["verdict"] if r else ""
            s_["judge_b"] = r["b"]["verdict"] if r else ""
            s_["judges_status"] = r["combined"]["status"] if r else "not judged"
        with open(out / "llm_judges.jsonl", "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, default=str) + "\n")
        judged = {**summarize(res), "model": judges.model, "not_judged": len(payloads) - len(todo)}

    def write(path, rows):
        if not rows:
            return
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v for k, v in r.items()})

    write(out / "turn_by_turn.csv", trows)
    write(out / "turn_by_turn_summary.csv", srows)
    n = len(srows) or 1
    return {"scenarios": len(srows), "turn_rows": len(trows), **({"llm_judges": judged} if judged else {}),
            "fully_matching_runs": sum(s["turns_matching"] == max(s["turns_expected"], s["turns_actual"]) for s in srows),
            "outcome_correct": sum(s["outcome_correct"] for s in srows),
            "first_divergence_by_state": _divergence_states(trows), "share_outcome_correct": sum(
                s["outcome_correct"] for s in srows) / n}


def _divergence_states(trows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    seen = set()
    for r in trows:
        if r["case_id"] in seen or r["turn_match"]:
            continue
        seen.add(r["case_id"])
        out[r["state"] or "-"] = out.get(r["state"] or "-", 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


# =====================================================================================================================
# PART 3 - Expected vs actual, case by case: one HTML page per campaign, in the answer sheet's format.
#
# For every run of the bot under test (base cases, enriched cases, LLM-arm cases) the page puts what the oracle
# expects next to what the bot did:
#
# * the verdict and exactly what was compared to reach it (the outcome signature: how the call ended, the
#   transfer reason, the responses actually recorded, the card action actually completed; for a policy-question
#   case also the graded policy answer), field by field;
# * the checks reported next to the verdict but not part of it (per-stage decisions, workflow admissibility,
#   runtime invariance, budgets);
# * for a failed run, the reasons in plain words (read off the graded row, nothing re-judged);
# * turn by turn: the customer's words, then EXPECTED (oracle: tool calls and bot acts) beside ACTUAL (bot: tool
#   calls, acts and its words), with the turns that differ marked.
#
# Failed runs are shown by default; buttons show the passed ones, all of them, or one arm only.
# Built from the campaign's own files - dataset/turn_by_turn.csv, dataset/turn_by_turn_summary.csv,
# dataset/case_results.jsonl and report.json - so it can be rebuilt without re-running anything:
#
#     python tools/case_comparison.py --campaign output/campaign_out
# =====================================================================================================================

E = lambda s: html.escape(str(s), quote=True)

STYLE = """
:root{--bg:#eef1f4;--surface:#fff;--ink:#131c26;--muted:#5a6b7d;--line:#d3dbe3;--line-soft:#e3e9ee;
--accent:#a8500a;--accent-soft:#f7e9db;--pass:#1c7a4a;--pass-soft:#e2f2e9;--code-bg:#f2f5f8;
--fail:#b3261e;--fail-soft:#fbe9e7;--wrong:#7a2e8e;--wrong-soft:#f3e7f6;
--display:"IBM Plex Sans Condensed",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10161d;--surface:#161e27;--ink:#e6edf3;
--muted:#93a3b3;--line:#2b3743;--line-soft:#222c36;--accent:#f0a35e;--accent-soft:#33261a;--pass:#5cc98d;
--pass-soft:#16281f;--code-bg:#131a22;--fail:#f2837a;--fail-soft:#2e1a19;--wrong:#d69be6;--wrong-soft:#2a1d2f;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#10161d;--surface:#161e27;--ink:#e6edf3;--muted:#93a3b3;--line:#2b3743;
--line-soft:#222c36;--accent:#f0a35e;--accent-soft:#33261a;--pass:#5cc98d;--pass-soft:#16281f;--code-bg:#131a22;
--fail:#f2837a;--fail-soft:#2e1a19;--wrong:#d69be6;--wrong-soft:#2a1d2f;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.6;padding-inline:16px}
.wrap{max-width:1360px;margin:0 auto;padding-block:clamp(26px,5vw,60px)}
h1,h2,h3{font-family:var(--display);line-height:1.15;margin:0;text-wrap:balance}
h1{font-size:clamp(1.9rem,5vw,2.9rem);font-weight:700}
h2{font-size:1.5rem;font-weight:700;margin-block:1.6rem .4rem}
h3{font-size:1.06rem;font-weight:600;display:flex;flex-wrap:wrap;gap:.5rem;align-items:baseline}
.lede{color:var(--muted);font-size:1.06rem;max-width:72ch}
code{font-family:var(--mono);font-size:.85em;background:var(--code-bg);border:1px solid var(--line-soft);border-radius:4px;padding:.06em .3em;word-break:break-word}
.kicker{display:inline-flex;align-items:center;gap:.5rem;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.16em;color:var(--accent)}
.kicker::before{content:"";width:22px;height:2px;background:var(--accent)}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:clamp(14px,2vw,20px);margin-block:14px}
.panel ol,.panel ul{margin:.3rem 0 .2rem;padding-left:1.3rem}.panel li{margin:.25rem 0}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:1rem 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.6rem .9rem;min-width:150px}
.tile b{display:block;font-family:var(--display);font-size:1.6rem}.tile span{font-size:.8rem;color:var(--muted)}
.case{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:clamp(14px,2vw,20px);margin-block:14px;scroll-margin-top:60px}
.case.fail{border-left:4px solid var(--fail)}.case.pass{border-left:4px solid var(--pass)}
.chead{display:flex;flex-direction:column;gap:.25rem;border-bottom:1px solid var(--line-soft);padding-bottom:.6rem}
.cat{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);background:var(--accent-soft);padding:.12rem .45rem;border-radius:4px}
.kind{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);border:1px solid var(--line);padding:.1rem .45rem;border-radius:4px}
.tag{border-radius:4px;padding:.05rem .45rem;font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.06em}
.tag.ok{background:var(--pass-soft);color:var(--pass)}.tag.no{background:var(--fail-soft);color:var(--fail)}
.tag.sev{border:1px solid var(--line);color:var(--muted)}
.cmeta{font-size:.8rem;color:var(--muted)}
.who{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);display:block}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;vertical-align:top;padding:.4rem .6rem;border-bottom:1px solid var(--line-soft)}
th{font-family:var(--display);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
td.m{white-space:nowrap;font-weight:600}td.m.ok{color:var(--pass)}td.m.no{color:var(--fail)}
tr.sub td{color:var(--muted);font-size:.8rem}tr.grp td{font-family:var(--display);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);background:var(--code-bg)}
.cmp{margin-top:.7rem;overflow-x:auto;border:1px solid var(--line-soft);border-radius:8px}
.why{margin:.7rem 0 .2rem;padding:.55rem .8rem;border-left:3px solid var(--fail);background:var(--fail-soft);border-radius:0 7px 7px 0;font-size:.92rem}
.why ul{margin:.2rem 0 0;padding-left:1.1rem}.why li{margin:.15rem 0}
.turns{display:grid;gap:10px;margin-top:.8rem}
.turn{border:1px solid var(--line-soft);border-radius:9px;padding:.7rem .85rem;min-width:0}
.turn.diff{border-color:var(--fail);background:linear-gradient(90deg,var(--fail-soft),transparent 30%)}
.turn .hd{display:flex;flex-wrap:wrap;gap:.5rem;align-items:baseline;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin-bottom:.4rem}
.turn .hd b{font-family:var(--display);font-size:.92rem;letter-spacing:0;text-transform:none;color:var(--ink)}
.turn .hd span{border-left:1px solid var(--line);padding-left:.5rem}
.say{display:grid;grid-template-columns:6.4rem 1fr;gap:.5rem;margin:.3rem 0}
.ea{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;margin-top:.4rem}
@media (max-width:820px){.ea{grid-template-columns:1fr}}
.col{border-radius:0 7px 7px 0;padding:.45rem .6rem;font-size:.88rem;min-width:0}
.col.exp{border-left:3px solid var(--pass);background:var(--pass-soft)}
.col.act{border-left:3px solid var(--line);background:var(--code-bg)}
.turn.diff .col.act{border-left-color:var(--fail)}
.col ol{margin:.15rem 0 0;padding-left:1.3rem}.col li{margin:.15rem 0}
.sk{display:inline-block;min-width:2.9rem;margin-right:.4rem;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
li.tool .sk{color:var(--accent)}
.words{margin-top:.35rem;font-size:.86rem;color:var(--muted)}
.mk{font-family:var(--mono);font-size:.7rem}.mk.ok{color:var(--pass)}.mk.no{color:var(--fail)}.mk.side{color:var(--accent)}.mk.na{color:var(--muted)}
.turn.side{border-color:var(--accent);background:linear-gradient(90deg,var(--accent-soft),transparent 30%)}
.mk.wrong{color:var(--wrong)}.turn.wrong{border-color:var(--wrong);background:linear-gradient(90deg,var(--wrong-soft),transparent 30%)}
.turn.wrong .col.act{border-left-color:var(--wrong)}
.stopbar{margin:.2rem 0 .6rem;padding:.45rem .8rem;border:1px dashed var(--fail);border-radius:7px;color:var(--fail);font-size:.88rem;font-weight:600}
.stopwhy ol{margin:.3rem 0 .2rem 1.1rem;padding:0}.stopwhy li{margin:.25rem 0}.stopwhy p{margin:.3rem 0}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;margin:.5rem 0 0;font-size:.85rem}
.legend span{display:inline-flex;align-items:center;gap:.4rem}.legend i{width:.8rem;height:.8rem;border-radius:3px;border:2px solid}
.legend i.ok{border-color:var(--pass)}.legend i.side{border-color:var(--accent);background:var(--accent-soft)}
.legend i.wrong{border-color:var(--wrong);background:var(--wrong-soft)}.legend i.no{border-color:var(--fail);background:var(--fail-soft)}
.turn.na{opacity:.7;border-style:dashed}
.note.bad{color:var(--fail)}
.note{font-size:.8rem;color:var(--muted);margin-top:.3rem}
.viewbar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;padding:.5rem 0;background:var(--bg)}
.viewbar button{font:inherit;font-size:.82rem;padding:.3rem .7rem;border-radius:6px;border:1px solid var(--line);background:var(--surface);color:var(--ink);cursor:pointer}
.viewbar button[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
.viewbar .sep{width:1px;height:1.4rem;background:var(--line);margin:0 .3rem}
.viewbar .count{font-size:.8rem;color:var(--muted);margin-left:.4rem}
.toc{columns:3;column-gap:2rem;font-size:.86rem;margin-top:.6rem}
@media (max-width:900px){.toc{columns:1}}
.toc a{display:block;padding:.12rem 0;color:var(--ink);text-decoration:none;border-bottom:1px dashed var(--line-soft);font-family:var(--mono);font-size:.76rem}
.toc a.f::before{content:"✗ ";color:var(--fail)}.toc a.p::before{content:"✓ ";color:var(--pass)}
.small{font-size:.84rem;color:var(--muted)}
.said{margin:.55rem 0 .2rem;padding:.6rem .8rem;background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:0 8px 8px 0}
.said-h{display:block;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);margin-bottom:.2rem}
.said p{margin:0;font-size:1.02rem;line-height:1.55;color:var(--ink);font-weight:500}
.said .sact.x{color:var(--accent);border-color:var(--accent)}
.said .sact{display:inline-block;margin:0 .15rem 0 .35rem;padding:0 .3rem;border-radius:4px;font-family:var(--mono);font-size:.62rem;font-weight:500;vertical-align:.12em;color:var(--muted);background:var(--code-bg);border:1px solid var(--line-soft)}
.sub-h{display:block;margin:.55rem 0 .2rem;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
ul.held{list-style:none;margin:.1rem 0 0;padding:0}ul.held li{display:grid;grid-template-columns:7.2rem minmax(0,1fr);gap:.4rem;margin:.2rem 0}
ul.held .st{font-family:var(--mono);font-size:.7rem;font-weight:500}
.st.ok{color:var(--pass)}.st.no{color:var(--fail)}.st.na{color:var(--muted)}.st.wrong{color:var(--wrong)}.st.side{color:var(--accent)}
table.cards{margin-top:.15rem;font-size:.8rem;background:var(--surface);border:1px solid var(--line-soft);border-radius:7px}
table.cards td{padding:.3rem .45rem;vertical-align:top}table.cards th{padding:.25rem .45rem;font-size:.64rem}
.pill{display:inline-block;border-radius:4px;padding:0 .35rem;font-family:var(--mono);font-size:.68rem}
.pill.supported{background:var(--pass-soft);color:var(--pass)}.pill.contradicted,.pill.unsupported{background:var(--fail-soft);color:var(--fail)}
.pill.unverifiable,.pill.adv{background:var(--code-bg);color:var(--muted);border:1px solid var(--line-soft)}
.pill.side{background:var(--accent-soft);color:var(--accent)}
.rv{margin-top:.35rem;font-size:.8rem}
.stopbar{text-align:center;letter-spacing:.04em}
footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.85rem;color:var(--muted)}
"""

HOW_JUDGED = """
<div class="panel">
<h2 style="margin-top:0">How a case is judged</h2>
<p>Every run is graded by comparing what the agent <b>did</b> - read from the calls it made on the test backend,
not from what it said - with what the oracle (an executable reading of the policy) does in the same situation.</p>
<ol>
<li><b>Correct / incorrect</b> needs the <b>outcome signature</b> to match, all four fields:
<ul>
<li><b>how the call ended</b>: <code>TRANSFERRED</code> (transfer_to_human was called), <code>ENDED</code> (the agent
closed the call), or <code>INCOMPLETE</code> (neither);</li>
<li><b>the transfer reason</b> passed to transfer_to_human (e.g. <code>record_failed</code>, <code>customer_request</code>);</li>
<li><b>the responses actually recorded</b>: every record_charge_response call that returned SUCCESS for the same
charge, as charge → disposition (RECOGNIZED / UNRECOGNIZED / UNSURE);</li>
<li><b>the card action actually completed</b>: a submit_card_action that returned SUCCESS for the requested action,
with its delivery option.</li>
</ul>
and the <b>journey played to its end</b>: a run stopped by the agent (below) is incorrect even when its outcome
happens to match. For a <b>policy-question case</b> the graded policy answer must also be correct: its figure equals
the exact value in the policy store, or the agent abstains where the policy states nothing.</li>
<li><b>Reported next to the verdict, not part of it</b>: the per-stage decisions (authenticate, retrieve, present, kb,
capture, record, card action, transfer, summary), <b>workflow admissibility</b> (no policy violation such as claiming a
record that did not succeed, using an account tool before authentication, submitting without confirmation, skipping the
fee disclosure), runtime invariance of the customer turns, and the latency / length budgets. They set a run's severity
and feed the release gate.</li>
<li><b>Turn by turn</b> (below each case, from the agent's opening statement on): the customer says the base journey's
lines. For each line, the <b>oracle's turn</b> is on the left: its tool calls and what it <b>says</b> (the acts the
policy takes, with their values), in order. On the right is the <b>agent's turn</b>, taken only from what can be
observed: the <b>tool calls</b> it made, with their key arguments and results, and its <b>words</b>. The agent's own
naming of its acts, if it gives any, is not used. Charges are named by merchant and amount. The agent's words are
validated in two ways:
<ul>
<li><b>The oracle's SAYS, checked in the words</b>: each act the oracle says at this turn must be held by the agent's
words. <span class="st ok">✓ held</span>: the words express it (one of the act's cue words, or every field it must say
is stated) and give no other value; <span class="st no">✗ other value</span>: they express it with another value
(noted the charge as RECOGNIZED where the customer said UNRECOGNIZED); <span class="st no">✗ not in the words</span>;
<span class="st na">– not checkable</span>: the act has no cue word and nothing it must say, so it is taken as said when
the calls match. Words that express an act the oracle does not take here are shown as
<span class="st wrong">+ not expected</span>.</li>
<li><b>Claim cards</b>: every value, figure or limit the words state is read into a claim and checked against its
truth: what the customer said (the customer's meaning card), a tool's result or a confirmed write, or the policy store.
Each is <span class="pill supported">supported</span>, <span class="pill contradicted">contradicted</span>,
<span class="pill unsupported">unsupported</span> (nothing backs it), or <span class="pill unverifiable">unverifiable</span>
(nothing to compare yet, never a failure). A policy answer must also give the policy's own figure.</li>
</ul>
Each agent turn is then one of four kinds:
<div class="legend"><span><i class="ok"></i>same as expected</span><span><i class="side"></i>non-material deviation
(side step)</span><span><i class="wrong"></i>incorrect action</span><span><i class="no"></i>off the journey</span></div>
<ul>
<li><b>same as expected</b>: the same tool calls with the same key arguments, in order, and every act the oracle says
held by the words (acknowledgements such as <code>CONFIRM_RECORDED</code> are not decisions: their values are checked
on the tool call and in the claim cards);</li>
<li><b>non-material deviation (side step)</b>: the same calls and decisions, but the policy answer or a clarification
differs, or a policy question is left unanswered or answered wrongly. The turn is flagged and the run goes on (for a
policy-question case the answer is the outcome, so a wrong one fails the case);</li>
<li><b>incorrect action</b>: the right steps done wrongly: another argument, charge or reason;</li>
<li><b>off the journey</b>: a different step: a call or an act the policy does not take here, or one it takes left
out. Asking again what the customer had just answered, with no call, is off the journey too; the customer then says
the line again, once.</li>
</ul>
After one material deviation (incorrect action, off the journey) the agent may get back on the journey at the next
turn. Two in a row, or one that ends the call: <b>Validation Stopped</b>. The lines after it are not played and the
case is incorrect.</li>
</ol>
</div>"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _j(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _b(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _fmt(v: Any) -> str:
    if v is None or v == [] or v == {} or v == "":
        return "none"
    if isinstance(v, list) and v and all(isinstance(x, list) and len(x) == 2 for x in v):
        return ", ".join(f"{a} = {b}" for a, b in v)
    if isinstance(v, (list, tuple)):
        return " / ".join("none" if x is None else str(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k} = {x}" for k, x in v.items()) or "none"
    return str(v)


def _items(s: str) -> list[str]:
    s = (s or "").strip()
    return [] if s in ("", "-", "(no turn)") else [x.strip() for x in s.split(" ; ") if x.strip()]


def _col(kind: str, tools: str, acts: str, words: str = "", steps: Any = None) -> str:
    st = _j(steps) if isinstance(steps, str) else steps
    if st:                                           # calls and acts in the order they happened
        li = [f'<li class="tool"><span class="sk">calls</span><code>{E(x["d"])}</code></li>' if x.get("k") == "calls" else
              f'<li><span class="sk">says</span><code>{E(x["d"])}</code></li>' for x in st]
    else:
        li = [f'<li class="tool"><span class="sk">calls</span><code>{E(t)}</code></li>' for t in _items(tools)]
        li += [f'<li><span class="sk">says</span><code>{E(a)}</code></li>' for a in _items(acts)]
    if (tools or "").strip() == "(no turn)":
        body = "<p class='small'>(no such turn in this run)</p>"
    else:
        body = f"<ol>{''.join(li)}</ol>" if li else "<p class='small'>(nothing)</p>"
    head = "Expected (oracle)" if kind == "exp" else "Actual (agent)"
    return (f'<div class="col {kind}"><span class="who">{head}</span>{body}'
            + (f'<div class="words">“{E(words)}”</div>' if words else "") + "</div>")


HELD = {"held": ("✓ held", "ok"), "contradicted": ("✗ other value", "no"), "not_heard": ("✗ not in the words", "no"),
        "not_checkable": ("– not checkable", "na")}


def _says_html(says: dict) -> str:
    """The oracle's SAYS for a turn, each act marked by whether the agent's words hold it."""
    rows = says.get("says") or []
    if not rows and not says.get("extra"):
        return ""
    li = []
    for r in rows:
        word, cls = HELD.get(r.get("status"), (r.get("status"), "na"))
        why = r.get("how") if r.get("status") == "held" else r.get("detail")
        li.append(f'<li><span class="st {cls}">{E(word)}</span><span><code>{E(r.get("act"))}</code>'
                  + (f' <span class="small">{E(why)}</span>' if why else "") + "</span></li>")
    for x in says.get("extra") or []:
        if x.get("closing"):                         # closing words while the call goes on: flagged, not graded
            li.append(f'<li><span class="st side">! closing words</span><span>“{E(x.get("sentence") or "")}” '
                      f'<span class="small">sounds like the end of the call ({E(x.get("how"))}), but the agent went on: '
                      f'a wording point, not graded as ending the call</span></span></li>')
            continue
        li.append(f'<li><span class="st wrong">+ not expected</span><span><code>{E(x.get("act"))}</code> '
                  f'<span class="small">said by {E(x.get("how"))}; the oracle does not say it here</span></span></li>')
    return '<span class="sub-h">The oracle\'s SAYS, checked in these words</span><ul class="held">' + "".join(li) + "</ul>"


def _claims_html(cr: dict) -> str:
    """Every claim card read from the agent's words, its truth and its verdict, then the reply's findings."""
    if not cr or not cr.get("checked"):
        return ""
    claims, finds = cr.get("claims") or [], [f for f in cr.get("findings") or [] if f.get("kind") != "unread"]
    unread = sum(1 for f in cr.get("findings") or [] if f.get("kind") == "unread")
    out = ['<span class="sub-h">Claim cards: what the words state, checked against the truth</span>']
    if claims:
        tr = []
        for c in claims:
            st = c.get("status") or ""
            tr.append(f'<tr><td><code>{E(c.get("field"))} = {E(_fmt(c.get("value")))}</code>'
                      f'<div class="small">“{E(c.get("words"))}”'
                      + (f' · about {E(c.get("about"))}' if c.get("about") else "")
                      + (f' · {E(c.get("mode"))}' if c.get("mode") and c.get("mode") != "states" else "") + "</div></td>"
                      f'<td><code>{E(_fmt(c.get("truth")))}</code><div class="small">{E(c.get("basis") or "")}'
                      + (f': {E(c.get("source"))}' if c.get("source") and c.get("source") != c.get("basis") else "") + "</div></td>"
                      f'<td><span class="pill {E(st)}">{E(st)}</span>'
                      + (' <span class="pill side">side step</span>' if c.get("side") else "")
                      + (f'<div class="small">{E(c.get("why"))}</div>' if c.get("why") else "") + "</td></tr>")
        out.append('<table class="cards"><thead><tr><th>Claim (the words)</th><th>Truth (where it was read)</th>'
                   '<th>Result</th></tr></thead><tbody>' + "".join(tr) + "</tbody></table>")
    else:
        out.append('<p class="small">No claim: these words state no value, figure or limit to check.</p>')
    for f in finds:
        cls = "adv" if f.get("advisory") else "side" if f.get("side") else "contradicted"
        what = "flow" if f.get("kind") == "not_allowed_here" else f.get("kind")
        out.append(f'<div class="rv"><span class="pill {cls}">{E(what)}</span> {E(f.get("detail"))}</div>')
    verdict = "grounded" if cr.get("ok") else ("fails only in a side step" if cr.get("core_ok") else "not grounded")
    out.append(f'<div class="rv"><span class="pill {"supported" if cr.get("ok") else "side" if cr.get("core_ok") else "contradicted"}">'
               f'{E(verdict)}</span>' + (f' <span class="small">{unread} clause{"s" if unread != 1 else ""} not read '
                                        f'as policy (advisory)</span>' if unread else "") + "</div>")
    return "".join(out)


def _col_agent(tools: str, acts: str, words: str, steps: Any, says: Any, claims: Any) -> str:
    """The agent's side of a turn: the calls it made (with their results), its words, the oracle's SAYS checked in
    those words, and every claim card. A run graded from the agent's own act labels (an older run) shows them."""
    says, claims = _j(says) or {}, _j(claims) or {}
    if not isinstance(says, dict) or not says.get("oracle"):
        return _col("act", tools, acts, words, steps) if not claims else \
            _col("act", tools, acts, words, steps)[:-6] + _claims_html(claims if isinstance(claims, dict) else {}) + "</div>"
    st = _j(steps) if isinstance(steps, str) else steps
    calls = [x["d"] for x in st or [] if x.get("k") == "calls"] if st else _items(tools)
    li = "".join(f'<li class="tool"><span class="sk">calls</span><code>{E(c)}</code></li>' for c in calls)
    return ('<div class="col act"><span class="who">Actual (agent): its calls and its words</span>'
            + (f"<ol>{li}</ol>" if li else "<p class='small'>(no tool call)</p>")
            + _said_html(words, says.get("sentences")) + _says_html(says) + _claims_html(claims) + "</div>")


def _said_html(words: str, sentences: Any = None) -> str:
    """What the agent said, set apart: each sentence with the act it was read as (the oracle's act it expresses)."""
    if not words:
        return '<div class="said"><span class="said-h">The agent said</span><p class="small">(nothing)</p></div>'
    sents = sentences if isinstance(sentences, list) and sentences else None
    body = "".join(f'<span class="sent">{E(x.get("text"))}'
                   + (f'<span class="sact{" x" if x.get("act") == "closing words" else ""}">{E(x.get("act"))}</span>'
                      if x.get("act") else "") + "</span> "
                   for x in sents) if sents else E(words)
    return f'<div class="said"><span class="said-h">The agent said</span><p>“{body.strip()}”</p></div>'


def _speech_rows(row: dict) -> list[str]:
    """The speech check's verdict on the bot's words (s07_checks/speech_*.py), as table rows: one for the words,
    one for the flow (part of the verdict only under --speech-gate strict)."""
    ok, flow = bool(int(row.get("c_speech", 1))), bool(int(row.get("c_flow", 1)))
    n, good = row.get("speech_claims", 0), row.get("speech_supported", 0)
    probs = [p for p in (row.get("speech_problems") or []) if p.get("kind") != "not_allowed_here"]
    detail = ("; ".join(f"{p.get('where') or 'turn ' + str(p.get('turn'))}, {p.get('kind')}: "
                        + (f"“{p.get('words')}” ({p.get('why')})" if p.get("words") else str(p.get("why")))
                        for p in probs[:3]) if probs
              else f"{good} of {n} claims supported, nothing missing")
    flow_probs = [p for p in (row.get("speech_problems") or []) if p.get("kind") == "not_allowed_here"]
    out = [f'<tr><td>What the customer heard (every claim grounded)</td><td colspan="2" class="small">{E(detail)}</td>'
           f'<td class="m {"ok" if ok else "no"}">{"✓ grounded" if ok else "✗ not grounded"}</td></tr>']
    out.append(f'<tr class="sub"><td>Speech flow (each act a graph transition)</td><td colspan="2" class="small">'
               f'{E("; ".join(str(p.get("why", "")) for p in flow_probs[:2]) or "every act follows the graph")}</td>'
               f'<td class="m {"ok" if flow else "no"}">{"✓" if flow else "✗"}</td></tr>')
    return out


def _verdict_table(row: dict, summ: dict) -> str:
    sig = row.get("signature") or row.get("why") or {}
    exp = sig.get("expected") if row else _j(summ.get("expected_outcome"))
    act = sig.get("actual") if row else _j(summ.get("actual_outcome"))
    exp, act = exp or {}, act or {}
    lines = ['<tr class="grp"><td colspan="4">Compared to decide correct / incorrect (all must match)</td></tr>']
    fields = [("How the call ended", "end"), ("Transfer reason", "transfer_reason"),
              ("Responses actually recorded", "recorded"), ("Card action actually completed", "action")]
    for label, k in fields:
        e, a = exp.get(k) if isinstance(exp, dict) else None, act.get(k) if isinstance(act, dict) else None
        ok = e == a
        lines.append(f'<tr><td>{E(label)}</td><td><code>{E(_fmt(e))}</code></td><td><code>{E(_fmt(a))}</code></td>'
                     f'<td class="m {"ok" if ok else "no"}">{"✓ match" if ok else "✗ differs"}</td></tr>')
    if "c_journey" in row:                           # stopped by the agent: incorrect whatever the outcome
        okj = bool(int(row.get("c_journey") or 0))
        lines.append(f'<tr><td>Journey played to its end</td><td><code>every line</code></td><td>'
                     + ("<code>every line</code>" if okj else
                        f'Validation Stopped at turn {E(row.get("stopped_at_turn"))}')
                     + f'</td><td class="m {"ok" if okj else "no"}">{"✓ match" if okj else "✗ stopped"}</td></tr>')
    kb_rows = []
    for k in row.get("kb_detail") or []:
        ok = bool(k.get("correct"))
        kb_rows.append(f'<tr{"" if row.get("kb_case") in (1, "1", True) else " class=sub"}><td>Policy answer to '
                       f'<code>{E(k.get("frame"))}</code></td>'
                       f'<td class="small">{E(k.get("detail"))}</td><td>“{E(k.get("answer"))}”</td>'
                       f'<td class="m {"ok" if ok else "no"}">{"✓ correct" if ok else "✗ wrong"}</td></tr>')
    kb_in = row.get("kb_case") in (1, "1", True)     # a policy-question case: its answer is the outcome
    if kb_in:
        lines += kb_rows
    gate = row.get("speech_gate")
    speech_rows = _speech_rows(row) if "c_speech" in row else []
    if gate in ("fail", "strict"):
        lines += speech_rows
    lines.append('<tr class="grp"><td colspan="4">Also checked, reported next to the verdict (not part of it)</td></tr>')
    if not kb_in:                                    # flagged at its turn (a non-material deviation)
        lines += kb_rows
    if gate not in ("fail", "strict"):
        lines += speech_rows
    stages = (row.get("why") or {}).get("stages") or {}
    wrong = set(_j(summ.get("stages_wrong")) or []) | set(stages)
    graded = sorted(k[2:] for k in row if k.startswith("c_")
                    and k not in ("c_speech", "c_flow", "c_journey", "c_speech_strict", "c_flow_strict"))
    for st in graded:
        if st in stages:
            d = stages[st]
            lines.append(f'<tr class="sub"><td>Stage {E(st)}</td><td><code>{E(_fmt(d.get("expected")))}</code></td>'
                         f'<td><code>{E(_fmt(d.get("actual")))}</code></td><td class="m no">✗ differs</td></tr>')
        else:
            ok = st not in wrong
            lines.append(f'<tr class="sub"><td>Stage {E(st)}</td><td colspan="2"></td>'
                         f'<td class="m {"ok" if ok else "no"}">{"✓ match" if ok else "✗ differs"}</td></tr>')
    viol = row.get("violations") or []
    lines.append('<tr class="sub"><td>Workflow admissibility</td><td colspan="2">'
                 + (", ".join(f"<code>{E(v)}</code>" for v in viol) if viol else "no policy violation")
                 + f'</td><td class="m {"no" if viol else "ok"}">{"✗ violations" if viol else "✓ admissible"}</td></tr>')
    if "invariance_ok" in row:
        ok = bool(row.get("invariance_ok"))
        lines.append(f'<tr class="sub"><td>Runtime invariance (customer turns read back)</td><td colspan="2"></td>'
                     f'<td class="m {"ok" if ok else "no"}">{"✓" if ok else "✗"}</td></tr>')
    if row.get("budget_violations"):
        lines.append(f'<tr class="sub"><td>Budgets</td><td colspan="2">{E("; ".join(map(str, row["budget_violations"])))}'
                     f'</td><td class="m no">✗ over</td></tr>')
    return ('<div class="cmp"><table><thead><tr><th>What is compared</th><th>Expected (oracle)</th>'
            '<th>Actual (agent)</th><th>Result</th></tr></thead><tbody>' + "".join(lines) + "</tbody></table></div>")


# a turn's status -> (words, mark class, turn box class)
TURN_STATUS = {
    "same": ("same as expected", "ok", ""),
    "side": ("non-material deviation (side step)", "side", " side"),
    "kb_wrong": ("non-material deviation (side step): policy answer wrong", "side", " side"),
    "wrong_action": ("incorrect action", "wrong", " wrong"),
    "asked_again": ("off the journey: asked the same question again", "no", " diff"),
    "differs": ("off the journey", "no", " diff"),
    "not_played": ("not played", "na", " na"),
}


def judges_block(j: dict | None) -> str:
    """The two LLM judges beside the rules' verdict (s07_checks/llm_judges.py), for a case page."""
    if not j:
        return ""
    c = j.get("combined") or {}
    st = c.get("status") or ""
    cls = "ok" if st == "agree" else "no" if st in ("judges disagree with the rules", "judges split") else "sev"

    def one(name, v):
        v = v or {}
        verdict = v.get("verdict") or "-"
        tc = "ok" if verdict == "correct" else "no" if verdict == "incorrect" else "sev"
        fw = v.get("first_wrong_turn")
        return (f'<li><b>{E(name)}</b> <span class="tag {tc}">{E(verdict)}</span>'
                + (f' <span class="small">first wrong turn {E(fw)}</span>' if fw is not None else "")
                + f' <span class="small">{E(v.get("reason") or "")}</span></li>')
    return (f'<div class="why judges"><span class="who">LLM judges (beside the rules, which decide)</span>'
            f' <span class="tag {cls}">{E(st)}</span><ul>'
            + one("Judge A, the auditor (expected steps vs the agent's)", j.get("a"))
            + one("Judge B, the reviewer (the call and the policy rules only)", j.get("b")) + "</ul></div>")


def _simple_stop(reason: str, row: dict) -> str:
    """A stopped run's reason, said simply: "Validation Stopped" and where."""
    if reason.startswith("The run stopped"):
        return f"Validation Stopped at turn {E(row.get('stopped_at_turn', ''))}." if row.get("stopped_at_turn") not in (None, "") \
            else "Validation Stopped."
    return reason


def _case(row: dict, summ: dict, turns: list[dict], judges: dict | None = None) -> str:
    from .report import failure_reasons
    sid = row.get("sid") or summ.get("case_id")
    ok = bool(int(row.get("correct", 0))) if row else _b(summ.get("outcome_correct"))
    arm = row.get("arm") or summ.get("arm") or "base"
    sev = row.get("severity") or "none"
    factors = {k[2:]: v for k, v in row.items() if k.startswith("f_")}
    meta = [f"base case <code>{E(row.get('seed') or summ.get('base_id'))}</code>"]
    if summ:
        meta.append(f"customer turns expected <b>{E(summ.get('turns_expected'))}</b>, actual <b>{E(summ.get('turns_actual'))}</b>")
        fd = summ.get("first_divergence_turn")
        meta.append("turns all match" if fd in ("", None) else f"first different turn <b>{E(fd)}</b>")
    if factors:
        meta.append("factor levels <code>" + E(", ".join(f"{k}={v}" for k, v in factors.items())) + "</code>")
    why = ""
    if not ok or not bool(int(row.get("workflow_adherent", 1) or 1)):
        why = ('<div class="why"><span class="who">Why it ' + ("failed" if not ok else "was flagged") + "</span><ul>"
               + "".join(f"<li>{_simple_stop(r, row)}</li>" for r in failure_reasons(row)) + "</ul>"
               + (f"<div class='small'>failure codes: {E(', '.join(row.get('failure_codes') or []))}</div>"
                  if row.get("failure_codes") else "") + "</div>")
    verdict = _verdict_table(row, summ)
    opening = ""
    if summ and (summ.get("expected_opening") or summ.get("actual_opening")):
        same = summ.get("expected_opening") == summ.get("actual_opening")
        opening = (f'<div class="turn{"" if same else " diff"}"><div class="hd"><b>Opening statement</b>'
                   f'<span>the agent speaks first, before any customer turn</span>'
                   f'<span class="mk {"ok" if same else "no"}">{"same as expected" if same else "off the journey"}</span>'
                   f'</div><div class="ea">'
                   + _col("exp", "", summ.get("expected_opening", ""), summ.get("expected_opening_text", ""),
                          steps=summ.get("expected_opening_steps"))
                   + _col_agent("", summ.get("actual_opening", ""), summ.get("actual_opening_text", ""),
                                summ.get("actual_opening_steps"), summ.get("actual_opening_says"),
                                summ.get("actual_opening_claims"))
                   + "</div></div>")
    stop = _j(summ.get("stopped_detail")) if summ else None
    stop = stop if isinstance(stop, dict) else None
    anchor = lambda n: f"{sid}--turn-{n}"                                                # noqa: E731
    tb = []
    for t in turns:
        status = t.get("turn_status") or ("same" if _b(t.get("turn_match")) else "differs")
        word, cls, box = TURN_STATUS.get(status, TURN_STATUS["differs"])
        after = _b(t.get("after_divergence"))
        try:
            tn = int(t.get("turn"))
        except (TypeError, ValueError):
            tn = None
        line = t.get("line")
        flags = (f'<span class="mk {cls}">{E(word)}</span>'
                 + (f"<span>customer line {E(line)}</span>" if line not in (None, "") else "")
                 + ("<span>the customer said it again</span>" if _b(t.get("repeat")) else "")
                 + ("<span>after the first difference</span>" if after and status not in ("not_played",) else ""))
        kbw = f'<div class="note bad">Policy answer wrong: {E(t.get("kb_wrong"))}</div>' if t.get("kb_wrong") else ""
        tb.append(f'<div class="turn{box}" id="{E(anchor(tn))}"><div class="hd"><b>Turn {E(t.get("turn"))}</b>'
                  f'<span>{E(t.get("state"))}</span>{flags}</div>'
                  f'<div class="say"><span class="who">Customer</span><span>“{E(t.get("customer_text"))}”'
                  + (f' <span class="small"><code>{E(t.get("customer_card"))}</code></span>' if t.get("customer_card") else "")
                  + "</span></div><div class='ea'>"
                  + _col("exp", t.get("expected_tools", ""), t.get("expected_bot", ""), steps=t.get("expected_steps"))
                  + (_col_agent(t.get("actual_tools", ""), t.get("actual_bot_acts", ""), t.get("actual_bot_text", ""),
                                t.get("actual_steps"), t.get("actual_says"), t.get("actual_claims"))
                     if status != "not_played" else
                     '<div class="col act"><span class="who">Actual (agent)</span><p class="small">Validation Stopped: '
                     'not played.</p></div>')
                  + "</div>" + kbw + "</div>")
        if stop is not None and tn == int(stop.get("turn", -1)):
            tb.append('<div class="stopbar">Validation Stopped</div>')
    if stop is not None:
        meta.append(f"<b>Validation Stopped</b> at turn {E(stop.get('turn'))}")
    elif summ and summ.get("stopped_at_turn") not in (None, ""):            # an older run folder
        meta.append(f"<b>Validation Stopped</b> at turn {E(summ.get('stopped_at_turn'))}")
    if summ:                                         # the agent's turns by kind of deviation
        n = lambda *ks: sum(int(summ.get(k) or 0) for k in ks)                       # noqa: E731
        kinds = [(n("turns_side", "turns_kb_wrong"), "non-material", "side"),
                 (n("turns_wrong_action"), "incorrect action", "wrong"),
                 (n("turns_off", "turns_asked_again"), "off the journey", "no")]
        if any(k for k, _, _ in kinds):
            meta.append("deviations: " + ", ".join(f'<span class="mk {c}"><b>{k}</b> {w}</span>' for k, w, c in kinds if k))
    items = _j(summ.get("items")) if summ else None
    if isinstance(items, dict) and items:            # charges by merchant and amount in the verdict and the reasons
        pat = re.compile(r"(?<![\w$])(" + "|".join(re.escape(k) for k in sorted(items, key=len, reverse=True)) + r")(?![\w])")
        name = lambda m: E(_label(m.group(1), items))                     # noqa: E731
        # the agent's own words, quoted “...”, stay exactly as said
        outside = lambda h: "".join(pat.sub(name, x) if i % 2 == 0 else x                # noqa: E731
                                    for i, x in enumerate(re.split(r"(“[^”]*”)", h)))
        verdict, why = outside(verdict), outside(why)
    turns_html = (f'<div class="turns">{opening}{"".join(tb)}</div>' if tb or opening else
                  "<p class='small'>(no turn-by-turn record for this run)</p>")
    return (f'<section class="case {"pass" if ok else "fail"}" id="{E(sid)}" data-ok="{int(ok)}" data-arm="{E(arm)}">'
            f'<div class="chead"><h3>{E(sid)} <span class="cat">{E(row.get("base") or "")}</span>'
            f'<span class="kind">{E(arm)}</span><span class="tag {"ok" if ok else "no"}">{"correct" if ok else "incorrect"}</span>'
            + (f'<span class="tag sev">severity {E(sev)}</span>' if sev != "none" else "")
            + f'</h3><div class="cmeta">{" · ".join(meta)}</div></div>'
            + verdict + why + judges_block(judges) + turns_html + "</section>")


def read_judges(ds: Path) -> dict[str, dict]:
    """dataset/llm_judges.jsonl as {case_id: judges}; empty when the run had no LLM judges."""
    out: dict[str, dict] = {}
    p = Path(ds) / "llm_judges.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    out[r["case_id"]] = r
                except (ValueError, KeyError):
                    continue
    return out


def write_speech_playground(out_dir: str | Path) -> Path:
    """Copy the speech check playground (speech_playground.html) into a run folder: served next to the run it reads
    the run's dataset/case_results.jsonl by itself; opened from disk it asks for the folder or the file."""
    import shutil
    dst = Path(out_dir) / "speech_playground.html"
    shutil.copyfile(Path(__file__).with_name("speech_playground.html"), dst)
    return dst


def write_case_comparison(out_dir: str | Path, out_name: str = "case_comparison.html") -> Path:
    """Write ``case_comparison.html`` in a run folder: every case side by side with the expected and actual turns,
    from the run's dataset files.
    """
    out = Path(out_dir)
    ds = out / "dataset"
    rows = {}
    rpath = ds / "case_results.jsonl"
    if rpath.exists():
        for line in rpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["sid"]] = r
    summ = {s["case_id"]: s for s in _read_csv(ds / "turn_by_turn_summary.csv")}
    judged = read_judges(ds)
    turns: dict[str, list[dict]] = {}
    for t in _read_csv(ds / "turn_by_turn.csv"):
        turns.setdefault(t["case_id"], []).append(t)
    R = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    bot = (R.get("results") or {}).get("bot", "the agent under test")
    order = list(rows) + [s for s in summ if s not in rows]
    arm_rank = {"base": 0, "template": 1, "llm": 2}
    order.sort(key=lambda s: (arm_rank.get((rows.get(s) or summ.get(s) or {}).get("arm") or "base", 9), s))
    blocks, toc = [], []
    n_ok = n_bad = 0
    arms: dict[str, int] = {}
    for sid in order:
        row, sm = rows.get(sid, {}), summ.get(sid, {})
        ok = bool(int(row.get("correct", 0))) if row else _b(sm.get("outcome_correct"))
        n_ok += ok
        n_bad += not ok
        arm = row.get("arm") or sm.get("arm") or "base"
        arms[arm] = arms.get(arm, 0) + 1
        blocks.append(_case(row, sm, turns.get(sid, []), judged.get(sid)))
        toc.append(f'<a class="{"p" if ok else "f"}" data-ok="{int(ok)}" data-arm="{E(arm)}" href="#{E(sid)}">{E(sid)}</a>')
    smp = (R.get("results") or {}).get("sample") or {}
    sample = (f" Sample: {smp['base_cases_tested']} of {smp['base_cases_total']} base cases (max_cases={smp['max_cases']})."
              if smp.get("max_cases") is not None else "")
    arm_buttons = "".join(f'<button type="button" data-arm="{E(a)}" aria-pressed="false">{E(a)}</button>' for a in sorted(arms))
    doc = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Expected vs Actual</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{STYLE}</style>
<div class="wrap">
<span class="kicker">Campaign · expected vs actual</span>
<h1>Expected vs actual, case by case</h1>
<p class="lede">System under test: <b>{E(bot)}</b>. Each run of the campaign, with what the oracle expects beside what the
agent did. {len(order)} runs: <b>{n_bad}</b> incorrect, <b>{n_ok}</b> correct.{E(sample)}</p>
<div class="tiles"><div class="tile"><b>{n_bad}</b><span>incorrect</span></div><div class="tile"><b>{n_ok}</b><span>correct</span></div>
{''.join(f'<div class="tile"><b>{n}</b><span>{E(a)} runs</span></div>' for a, n in sorted(arms.items()))}</div>
{HOW_JUDGED}
<h2>Index</h2>
<div class="toc" id="toc">{''.join(toc)}</div>
<h2>Cases</h2>
<div class="viewbar" role="group" aria-label="Show">
  <span class="who" style="display:inline">Show</span>
  <button type="button" data-ok="0" aria-pressed="true">Incorrect</button>
  <button type="button" data-ok="1" aria-pressed="false">Correct</button>
  <button type="button" data-ok="all" aria-pressed="false">All cases</button>
  <span class="sep"></span><span class="who" style="display:inline">Arm</span>
  <button type="button" data-arm="all" aria-pressed="true">all</button>{arm_buttons}
  <span class="count" id="cnt"></span>
</div>
<div id="cases">{''.join(blocks)}</div>
<footer>Generated by <code>gmsjourney/s09_reports/cases.py</code> from the campaign's own files
(<code>dataset/case_results.jsonl</code>, <code>dataset/turn_by_turn.csv</code>, <code>dataset/turn_by_turn_summary.csv</code>,
<code>report.json</code>); rebuild with <code>python tools/case_comparison.py --campaign {E(out)}</code>.
See also <code>report.html</code> (the campaign report).</footer>
</div>
<script>
(function () {{
  var okSel = "0", armSel = "all";
  try {{ okSel = localStorage.getItem("cmp-ok") || "0"; armSel = localStorage.getItem("cmp-arm") || "all"; }} catch (e) {{}}
  function apply() {{
    var n = 0;
    document.querySelectorAll("#cases .case, #toc a").forEach(function (el) {{
      var show = (okSel === "all" || el.dataset.ok === okSel) && (armSel === "all" || el.dataset.arm === armSel);
      el.style.display = show ? "" : "none";
      if (show && el.classList.contains("case")) n++;
    }});
    document.querySelectorAll(".viewbar button[data-ok]").forEach(function (b) {{ b.setAttribute("aria-pressed", String(b.dataset.ok === okSel)); }});
    document.querySelectorAll(".viewbar button[data-arm]").forEach(function (b) {{ b.setAttribute("aria-pressed", String(b.dataset.arm === armSel)); }});
    document.getElementById("cnt").textContent = n + " shown";
    try {{ localStorage.setItem("cmp-ok", okSel); localStorage.setItem("cmp-arm", armSel); }} catch (e) {{}}
  }}
  document.querySelectorAll(".viewbar button[data-ok]").forEach(function (b) {{ b.addEventListener("click", function () {{ okSel = b.dataset.ok; apply(); }}); }});
  document.querySelectorAll(".viewbar button[data-arm]").forEach(function (b) {{ b.addEventListener("click", function () {{ armSel = b.dataset.arm; apply(); }}); }});
  if (location.hash) {{ var t = document.getElementById(location.hash.slice(1)); if (t) {{ okSel = t.dataset.ok; armSel = "all"; }} }}
  apply();
}})();
</script>
"""
    path = out / out_name
    path.write_text(doc, encoding="utf-8")
    return path


# =====================================================================================================================
# PART 4 - Look inside a base (seed) case and any of its enriched variants, turn by turn.
#
# * ``compare_case``  one row per customer turn: the dialog state the turn answers, the seed
#   text and the enriched text, the seed's canonical frame and the canonical frame read
#   back from the enriched text, and whether they are the same meaning.
# * ``case_hrt``      the same turns as Head / Relation / Tail triples, for the seed and for
#   the enriched variant, so you can see that every relation keeps its tail. Optionally
#   adds the store facts that ground each slot value (e.g. expedited_delivery has_fee_usd 15).
# * ``pick``          choose a seed case and one of its enriched variants by id or index.
# * ``show_case``     prints both views for one (case, variant, turn) selection.
#
# Canonical form = the ontology's canonicalisation of a semantic frame (aliases mapped to
# canonical individuals, slots sorted), which is exactly what the invariance proof compares.
# =====================================================================================================================

# ----------------------------------------------------------------------------- selection
def variants_of(scenarios, qid: str) -> list:
    """The reworded versions of one base case."""
    return [s for s in scenarios if s.base.qid == qid]


def pick(bases, scenarios, base: str | int = 0, variant: int = 0):
    """(seed item, enriched scenario) by base qid (e.g. 'J015', 'kb-exact_recall-01') or index, and variant index."""
    item = next((b for b in bases if b.qid == base), None) if isinstance(base, str) else bases[base]
    if item is None:
        raise KeyError(f"no base case {base!r}; ids look like {[b.qid for b in bases[:5]]} ...")
    vs = variants_of(scenarios, item.qid)
    if not vs:
        raise KeyError(f"base {item.qid} has no enriched variants in this design")
    return item, vs[variant % len(vs)]


# ----------------------------------------------------------------------------- canonical
def _canon(ont, frame, state=None) -> str:
    """The state-aware canonical meaning card, as compact JSON."""
    if frame is None:
        return "(unreadable)"
    c = ont.card(frame, state)
    out = {"act": c["act"], "slots": c["slots"]}
    if not c.get("admissible", True):
        out["not_admissible"] = c.get("why_not")
    return json.dumps(out, sort_keys=True)


def compare_case(validator, scenario, turn: int | None = None, proof=None) -> pd.DataFrame:
    """Seed vs enriched, turn by turn, in canonical form."""
    p = proof or validator.prove(scenario)
    ont = validator.ont
    rows = []
    for t in p.turns:
        if turn is not None and t["turn"] != turn:
            continue
        rows.append({"turn": t["turn"], "state": t.get("state"),
                     "seed_text": t["base_text"], "enriched_text": t["enriched_text"],
                     "seed_canonical": _canon(ont, t["frame_obj"], t.get("state")),
                     "enriched_canonical": _canon(ont, t.get("read_obj"), t.get("state")),
                     "same_meaning": t["equivalent"], "surface_cosine": round(t["surface_cosine"], 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- H / R / T
def _values(v) -> list:
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out += _values(x) if not (isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str)) else [f"{x[0]}={x[1]}"]
        return out
    return [v]


def frame_hrt(frame, ont, head: str, state: str | None = None) -> list[tuple[str, str, Any]]:
    """A customer turn as triples: (turn, in_state, state), (turn, has_act, act), (turn, has_<slot>, value).
    The card is canonicalised in its dialog state (a bare yes/no read the way the state reads it)."""
    if frame is None:
        return [(head, "has_act", "(unreadable)")]
    f = ont.canonicalize(frame, state)
    out = [(head, "in_state", state)] if state else []
    out.append((head, "has_act", f.act))
    for k, v in f.slots:
        for x in _values(v):
            out.append((head, f"has_{k}", x))
    return out


def _grounding(store, value, limit: int = 5) -> list[tuple[str, str, Any]]:
    if not isinstance(value, str):
        return []
    facts = sorted(((h, r, t) for h, r, t in getattr(store, "triples", []) if h == value),
                   key=lambda x: (x[1] == "has_alias", x[1], str(x[2])))     # policy facts before aliases
    vs = [(value, "member_of", name) for name, vals in getattr(store, "value_sets", {}).items() if value in vals]
    return (vs + facts)[:limit]


def case_hrt(validator, scenario, turn: int | None = None, grounding: bool = False, proof=None,
             wide: bool = True) -> pd.DataFrame:
    """Head / Relation / Tail for the seed turn and the enriched turn.

    wide=True: one row per (turn, relation) with seed and enriched side by side and a match flag.
    wide=False: long form with a `case` column (seed / enriched) and head, relation, tail."""
    p = proof or validator.prove(scenario)
    ont, store = validator.ont, validator.store
    seed_id, enr_id = scenario.base.qid, scenario.sid
    long_rows, wide_rows = [], []
    for t in p.turns:
        if turn is not None and t["turn"] != turn:
            continue
        i, st = t["turn"], t.get("state")
        seed = frame_hrt(t["frame_obj"], ont, f"{seed_id}/t{i}", st)
        enr = frame_hrt(t.get("read_obj"), ont, f"{enr_id}/t{i}", st)
        for case, trs in (("seed", seed), ("enriched", enr)):
            long_rows += [{"case": case, "turn": i, "state": st, "head": h, "relation": r, "tail": tl}
                          for h, r, tl in trs]
        # side by side: align on relation (multi-valued relations compare as sets)
        rels = list(dict.fromkeys([r for _, r, _ in seed] + [r for _, r, _ in enr]))
        for r in rels:
            a = sorted(str(x) for _, rr, x in seed if rr == r)
            b = sorted(str(x) for _, rr, x in enr if rr == r)
            wide_rows.append({"turn": i, "state": st, "seed_head": f"{seed_id}/t{i}", "relation": r,
                              "seed_tail": ", ".join(a) or "-", "enriched_head": f"{enr_id}/t{i}",
                              "enriched_tail": ", ".join(b) or "-", "match": a == b})
        if grounding and st and getattr(ont, "kg", None) is not None:
            kg = ont.kg                                   # the policy that holds in this state
            for rel, vals in (("in_stage", [kg.stage(st)]), ("applies_rule", kg.rules_for(st)),
                              ("limited_by", [f"{k}={kg.limit(k)}" for k in kg.limits_for(st)]),
                              ("accepts_act", kg.accepted_acts(st))):
                row = {"turn": i, "state": st, "head": st, "relation": rel, "tail": ", ".join(map(str, vals))}
                long_rows.append({"case": "state policy", **row})
                wide_rows.append({"turn": i, "state": st, "seed_head": st, "relation": rel, "seed_tail": row["tail"],
                                  "enriched_head": st, "enriched_tail": row["tail"], "match": "state policy"})
        if grounding:
            for _, r, x in seed:
                if r in ("in_state", "has_act"):
                    continue
                for h, gr, gt in _grounding(store, x):
                    row = {"turn": i, "state": st, "head": h, "relation": gr, "tail": gt}
                    long_rows.append({"case": "store fact", **row})
                    wide_rows.append({"turn": i, "state": st, "seed_head": h, "relation": gr, "seed_tail": gt,
                                      "enriched_head": h, "enriched_tail": gt, "match": "store fact"})
    return pd.DataFrame(wide_rows if wide else long_rows)


# ----------------------------------------------------------------------------- one-call view
def show_case(validator, bases, scenarios, base: str | int = 0, variant: int = 0, turn: int | None = None,
              grounding: bool = True, display=None):
    """Print the seed/enriched comparison and the H/R/T tables for one selection."""
    item, scn = pick(bases, scenarios, base, variant)
    p = validator.prove(scn)
    show = display or (lambda df: print(df.to_string(index=False)))
    print(f"seed case {item.qid}  ->  enriched variant {scn.sid}   (proof accepted: {p.accepted})")
    print("factor levels:", json.dumps(scn.factor_levels))
    print("seed frame signature    ", p.base_signature)
    print("enriched frame signature", p.enriched_signature, "(equal)" if p.base_signature == p.enriched_signature else "(DIFFERENT)")
    print("\n-- canonical comparison --")
    show(compare_case(validator, scn, turn, proof=p))
    print("\n-- Head / Relation / Tail (seed vs enriched) --")
    show(case_hrt(validator, scn, turn, grounding=grounding, proof=p))
    return p
