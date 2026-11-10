"""Two LLM judges beside the rule-based verdict: a second opinion on every run of the bot, never the decider.

The rule-based checks (scoring.py, the speech check, the runtime monitor) decide whether a run passed. Next to them,
two LLM judges read the same run and say whether the bot did and said the right thing, each on its own:

* **Judge A, the auditor**: sees, turn by turn, what the graph says the bot must do (tool calls and acts) next to what
  the bot did and said, and how the call had to end. It is asked whether the differences matter.
* **Judge B, the reviewer**: sees only the conversation (the customer's words, the bot's words and tool calls), the
  policy rules of the dialog states the call went through and how the call had to end - not the expected steps. It
  is asked whether the bot followed the policy.

So the two judges answer the same question from different material and with different prompts: when both say a run
the rules passed is wrong (or the reverse), that run is worth a human look. Each verdict is ``correct``,
``incorrect``, ``unknown`` (a reply that could not be read) or ``unavailable`` (no model), with a one-line reason and
the first turn it found wrong. ``combine`` puts them beside the rules' verdict: ``agree`` (all three the same),
``judges split`` or ``judges disagree with the rules``.

Analogy: the rules are the referee's decision; the judges are two commentators watching the replay from different
angles. The decision stands, and the commentators' disagreements are what the review panel looks at.

The judges run on the testing LLM (role "testing": the gateway or a local model, chosen apart from the rewording LLM), at temperature
0, behind the replay cache like every other LLM call (cache tags ``judge_a`` and ``judge_b``).

    Judges(settings, mode)          the two clients
    judge_case(case)                one test run (the rows of turn_by_turn plus the outcome) -> {"a", "b", ...}
    judge_turn(turn)                one Playground turn -> {"a", "b", ...}
    judge_many(cases, workers)      many runs in parallel
    combine(rules_ok, a, b)         the line shown beside the rules' verdict
    summarize(results)              agreement rates and the runs to look at
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

JUDGES_VERSION = "llm-judges/1"

_REPLY = ('Reply with ONE JSON object and nothing else: {"verdict": "correct" or "incorrect", '
          '"first_wrong_turn": the number of the first turn that is wrong, or null, '
          '"reason": "one short sentence"}')

SYSTEM_A = f"""You audit a customer-service voice bot against its policy graph.
For each customer turn you are given what the policy graph says the bot must do (its tool calls with their key
arguments, and its dialog acts, in order) next to what the bot actually did (tool calls, acts) and said (its words),
and how the call had to end compared with how it ended.
The bot is correct when it made the required tool calls with the right arguments, took the required decisions,
asked and told the customer what the graph requires, said nothing false, and the call ended as required. Different
wording of the same act is fine. A missing, extra or wrong tool call or decision, a wrong ending, or a false or
missing required statement makes it incorrect.
{_REPLY}"""

SYSTEM_B = f"""You are a quality reviewer of customer-service calls handled by a voice bot.
You are given the call as it happened (the customer's words, the bot's words and the bot's tool calls), the written
policy rules that apply in the dialog states the call went through, and how the policy says the call had to end.
Judge only from the policy: did the bot handle the call correctly - the right checks and tool calls, the right
decisions, nothing promised or stated that the policy does not allow, and the required ending?
Polite, differently worded answers are fine; only mistakes against the policy count.
{_REPLY}"""

SYSTEM_TURN_A = f"""You audit one turn of a customer-service voice bot against its policy graph.
You are given the customer's line and its meaning, the dialog state, what the policy graph says the bot must do in
this turn (tool calls with key arguments, and acts) and what the bot did and said.
The turn is correct when the bot made the required tool calls and took the required acts, and its words say what
those acts require and nothing false. Different wording is fine.
{_REPLY}"""

SYSTEM_TURN_B = f"""You are a quality reviewer of one turn of a customer-service call handled by a voice bot.
You are given the conversation so far, the customer's latest line, the policy rules of the current dialog state and
the bot's reply (its words and tool calls). Judge only from the policy whether this reply is correct.
{_REPLY}"""


def _verdict(obj: dict | None, raw: str | None = None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"verdict": "unknown", "reason": "the judge's reply could not be read" + (f": {raw[:120]!r}" if raw else ""),
                "first_wrong_turn": None}
    v = str(obj.get("verdict") or "").strip().lower()
    v = "correct" if v in ("correct", "pass", "yes", "true") else "incorrect" if v in ("incorrect", "fail", "no", "false", "wrong") else "unknown"
    fw = obj.get("first_wrong_turn")
    try:
        fw = int(fw) if fw is not None and str(fw).strip().lower() not in ("", "null", "none") else None
    except (TypeError, ValueError):
        fw = None
    return {"verdict": v, "reason": str(obj.get("reason") or "")[:300], "first_wrong_turn": fw}


def combine(rules_ok: bool | None, a: dict | None, b: dict | None) -> dict[str, Any]:
    """The judges beside the rules' verdict: each judge's agreement with the rules, whether they agree with each
    other, and one status line. Judges that could not answer are left out of the comparison."""
    say = lambda j: (j or {}).get("verdict") if (j or {}).get("verdict") in ("correct", "incorrect") else None  # noqa: E731
    va, vb = say(a), say(b)
    rv = None if rules_ok is None else ("correct" if rules_ok else "incorrect")
    answered = [v for v in (va, vb) if v is not None]
    if not answered:
        status = "judges unavailable"
    elif rv is not None and all(v == rv for v in answered) and len(answered) == 2:
        status = "agree"
    elif len(answered) == 2 and va != vb:
        status = "judges split"
    elif rv is not None and all(v != rv for v in answered):
        status = "judges disagree with the rules"
    else:
        status = "one judge only" if len(answered) == 1 and answered[0] == rv else "judges disagree with the rules"
    return {"rules": rv, "a": va, "b": vb, "status": status,
            "a_agrees": None if va is None or rv is None else va == rv,
            "b_agrees": None if vb is None or rv is None else vb == rv}


class Judges:
    """The two LLM judges (``llm`` clients built from ``settings``; tests pass stand-in clients)."""

    def __init__(self, settings=None, mode: str | None = None, llm_a=None, llm_b=None):
        from ..llm import CachedLLM, LLMSettings
        if llm_a is None or llm_b is None:
            s = settings or LLMSettings(role="testing")
            base = {**s.__dict__, "temperature": 0.0, "max_tokens": 300}
            if mode:
                base["mode"] = mode
            llm_a = llm_a or CachedLLM(LLMSettings(**base), tag="judge_a")
            llm_b = llm_b or CachedLLM(LLMSettings(**base), tag="judge_b")
        self.a, self.b = llm_a, llm_b
        self._down: tuple[float, str] | None = None      # the model could not be reached: (when, why)

    DOWN_FOR = 120.0                                      # seconds a failed connection is not retried

    @property
    def model(self) -> str:
        s = self.a.settings
        return f"{s.model} ({'local model' if getattr(s, 'backend', '') == 'local' else 'Microsoft gateway'})"

    def _ask(self, llm, system: str, user: str, salt: str) -> dict[str, Any]:
        import time
        from ..llm import LLMUnavailable, parse_json_object
        key = f"{JUDGES_VERSION}|{salt}"
        if self._down and time.time() - self._down[0] < self.DOWN_FOR and llm.cached(system, user, key) is None:
            return {"verdict": "unavailable", "reason": self._down[1], "first_wrong_turn": None}
        try:
            raw = llm.invoke(system, user, key)
        except LLMUnavailable as e:
            self._down = (time.time(), str(e)[:200])     # not retried for a while: a judge never stalls the work
            return {"verdict": "unavailable", "reason": str(e)[:200], "first_wrong_turn": None}
        except Exception as e:                            # noqa: BLE001 - a judge never stops a run
            return {"verdict": "unavailable", "reason": f"{type(e).__name__}: {str(e)[:160]}", "first_wrong_turn": None}
        return _verdict(parse_json_object(raw), raw)

    # ------------------------------------------------------------------ one test run
    @staticmethod
    def _case_a(c: dict) -> str:
        lines = [f"Case {c.get('case_id')}"]
        if c.get("expected_opening") or c.get("actual_opening"):
            lines.append(f"Call opening - required: {c.get('expected_opening') or '-'} | bot: {c.get('actual_opening') or '-'}")
        for t in c.get("turns") or []:
            lines.append(f"\nTurn {t.get('turn')} (state {t.get('state')})")
            lines.append(f"  customer said: {t.get('customer_text')}   meaning: {t.get('customer_card')}")
            lines.append(f"  REQUIRED tool calls: {t.get('expected_tools')}")
            lines.append(f"  REQUIRED bot acts: {t.get('expected_bot')}")
            lines.append(f"  bot tool calls: {t.get('actual_tools')}")
            lines.append(f"  bot acts: {t.get('actual_bot_acts')}")
            lines.append(f"  bot said: {t.get('actual_bot_text') or '-'}")
        lines.append(f"\nHow the call had to end: {json.dumps(c.get('expected_outcome'), default=str)}")
        lines.append(f"How the call ended: {json.dumps(c.get('actual_outcome'), default=str)}")
        return "\n".join(lines)

    @staticmethod
    def _case_b(c: dict) -> str:
        lines = [f"Call {c.get('case_id')}", "", "The conversation:"]
        if c.get("actual_opening"):
            lines.append(f"  (bot opens) {c.get('actual_opening')}")
        for t in c.get("turns") or []:
            if str(t.get("customer_text") or "").startswith("(expected path)"):
                continue                                  # the bot had already ended the call
            lines.append(f"  customer: {t.get('customer_text')}")
            tools = t.get("actual_tools")
            if tools and tools not in ("-", "(no turn)"):
                lines.append(f"  bot calls: {tools}")
            lines.append(f"  bot: {t.get('actual_bot_text') or '(' + str(t.get('actual_bot_acts') or '-') + ')'}")
        rules = c.get("rules") or {}
        if rules:
            lines.append("\nPolicy rules of the dialog states this call went through:")
            for rid, text in rules.items():
                lines.append(f"  {rid}: {text}")
        lines.append(f"\nHow the policy says this call had to end: {json.dumps(c.get('expected_outcome'), default=str)}")
        return "\n".join(lines)

    def judge_case(self, c: dict) -> dict[str, Any]:
        """One test run: ``c`` holds case_id, turns (turn_by_turn rows), expected/actual outcome and opening, rules
        (id -> text) and ``rules_ok`` (the rule-based verdict)."""
        a = self._ask(self.a, SYSTEM_A, self._case_a(c), f"case|{c.get('case_id')}")
        b = self._ask(self.b, SYSTEM_B, self._case_b(c), f"case|{c.get('case_id')}")
        return {"case_id": c.get("case_id"), "a": a, "b": b, **{"combined": combine(c.get("rules_ok"), a, b)}}

    def judge_many(self, cases: list[dict], workers: int = 4) -> list[dict[str, Any]]:
        if workers <= 1 or len(cases) <= 1:
            return [self.judge_case(c) for c in cases]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self.judge_case, cases))

    # ------------------------------------------------------------------ one Playground turn
    def judge_turn(self, t: dict) -> dict[str, Any]:
        """One Playground turn: ``t`` holds the state, the customer's line and meaning, the graph's required tools and
        acts, the bot's tools, acts and words, the conversation so far, the state's rules and ``rules_ok``."""
        ua = "\n".join([
            f"Dialog state: {t.get('state')}",
            f"Customer said: {t.get('customer_text')}   meaning: {json.dumps(t.get('customer_card'), default=str)}",
            f"REQUIRED tool calls: {t.get('expected_tools') or '-'}",
            f"REQUIRED bot acts: {t.get('expected_acts') or '-'}",
            f"Bot tool calls: {t.get('actual_tools') or '-'}",
            f"Bot acts: {t.get('actual_acts') or '-'}",
            f"Bot said: {t.get('bot_text') or '-'}"])
        rules = t.get("rules") or {}
        hist = [f"  {x}" for x in (t.get("history") or [])] or ["  (this is the first turn)"]
        ub = "\n".join(["The conversation so far:", *hist,
                        f"\nCustomer's latest line: {t.get('customer_text')}",
                        "\nPolicy rules of the current dialog state:" if rules else "",
                        *[f"  {k}: {v}" for k, v in rules.items()],
                        f"\nBot's tool calls: {t.get('actual_tools') or '-'}",
                        f"Bot's reply: {t.get('bot_text') or '-'}"])
        salt = f"turn|{t.get('key')}"
        with ThreadPoolExecutor(max_workers=2) as ex:     # the two judges at once
            fa = ex.submit(self._ask, self.a, SYSTEM_TURN_A, ua, salt)
            fb = ex.submit(self._ask, self.b, SYSTEM_TURN_B, ub, salt)
            a, b = fa.result(), fb.result()
        return {"a": a, "b": b, "combined": combine(t.get("rules_ok"), a, b), "model": self.model}


def rules_of(kg, states) -> dict[str, str]:
    """The policy rules of the dialog states a call went through (id -> text), in order."""
    out: dict[str, str] = {}
    if kg is None:
        return out
    for st in states:
        if not st:
            continue
        try:
            for r in kg.rules_for(st):
                if r not in out:
                    out[r] = kg.rule_text(r) or ""
        except Exception:                                 # noqa: BLE001 - a state the graph does not know
            continue
    return out


def summarize(results: list[dict]) -> dict[str, Any]:
    """Agreement of each judge with the rules, how often the judges split, and the runs to look at."""
    comb = [r["combined"] for r in results]
    n = len(comb)
    rate = lambda k: (round(sum(1 for c in comb if c[k]) / m, 4) if (m := sum(1 for c in comb if c[k] is not None)) else None)  # noqa: E731
    by = {}
    for c in comb:
        by[c["status"]] = by.get(c["status"], 0) + 1
    look = [{"case_id": r["case_id"], "status": r["combined"]["status"], "rules": r["combined"]["rules"],
             "a": r["a"], "b": r["b"]} for r in results if r["combined"]["status"] in ("judges disagree with the rules", "judges split")]
    look.sort(key=lambda x: x["status"] != "judges disagree with the rules")
    return {"judged": n, "a_agrees_with_rules": rate("a_agrees"), "b_agrees_with_rules": rate("b_agrees"),
            "judges_agree": round(sum(1 for c in comb if c["a"] and c["b"] and c["a"] == c["b"]) / n, 4) if n else None,
            "by_status": by, "to_review": look[:200]}
