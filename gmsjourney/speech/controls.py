"""Checking the checker: negative controls on what the bot says.

A check that never fires proves nothing. Each control below changes ONLY the words of one act in a
real conversation (the act labels, the tool calls and the outcome stay exactly as they were) in a
way that makes the words false, incomplete or unfaithful. The speech check must flag the turn where
the words changed. The detection rate per control is how much the check can be trusted; the
unchanged conversations (the false-alarm baseline) must stay clean.

Input   a runner factory (``bot_factory -> JourneyRunner``), the reference bot factory and scenarios.
Output  {control: {applied, detected, detection_rate, missed: [examples]}} and the false-alarm count.

    from gmsjourney.speech.controls import run_controls
    res = run_controls(lambda f: JourneyRunner(bot_factory=f, **kw), make_reference_factory(store), scenarios)
    res["controls"]["flip_disposition"]["detection_rate"]   # 1.0

The controls (what the customer would hear, against what is true):
  swap_merchant      a charge presented with another returned charge's merchant
  change_amount      a charge's amount one dollar off
  shift_date         a charge's date one day off
  drop_date          a charge presented without its date (R6: merchant, amount and date)
  flip_disposition   a recorded answer said as its opposite
  wrong_fee          a delivery fee ten dollars off
  wrong_days         a delivery time of 9 business days
  wrong_limit        the transaction limit one too many
  claim_done_early   "your card has been replaced" said before any card action succeeded
  assert_fraud       "this means fraud" (R27 says it is not, by itself, proof of fraud)
  drop_summary_item  a summary that leaves out a recorded response (R19)

Policy controls (every pack, built from its graph; speech/policy.py must flag them):
  invented_policy        "Our policy is to always refund the full amount on request." (no such statement)
  wrong_policy_figure    a customer-facing limit restated in other words with the wrong number
  unauthorised_promise   "I'll also waive any fees ..." (a verb no tool of the graph performs)
  internal_disclosure    an internal limit (graph) or a never_volunteer one (project) told to the customer
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..names import N
from .. import roles

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


# ------------------------------------------------------------------ the edits
# Each edit takes (act, its text, ctx) and returns the new text, or None when it does not apply to this act.
# ctx: {"merchants": {transaction_id: merchant} returned so far, "submitted": a card action has succeeded}
def _swap_merchant(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: name another returned charge's merchant (the amount and date stay this charge's)."""
    if act.get("act") != N.state.PRESENT_TXN or len(ctx["merchants"]) < 2:
        return None
    mine = ctx["merchants"].get(act.get("txn"))
    other = next((m for t, m in ctx["merchants"].items() if t != act.get("txn")), None)
    return text.replace(mine, other) if mine and other and mine in text else None


def _change_amount(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN / PROVIDE_CONTEXT: the charge's amount one dollar higher."""
    if act.get("act") not in (N.state.PRESENT_TXN, N.state.PROVIDE_CONTEXT):
        return None
    m = re.search(r"\$(\d+\.\d{2})", text)
    return text[:m.start()] + f"${float(m.group(1)) + 1:.2f}" + text[m.end():] if m else None


def _shift_date(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: the charge's date one day off."""
    m = re.search(r"\b(" + "|".join(MONTHS) + r") (\d{1,2})\b", text)
    if act.get("act") != N.state.PRESENT_TXN or not m:
        return None
    d = int(m.group(2))
    return text[:m.start()] + f"{m.group(1)} {d + 1 if d < 28 else d - 1}" + text[m.end():]


def _drop_date(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: leave the date out (pdf.2B requires merchant, amount and date)."""
    m = re.search(r" on (" + "|".join(MONTHS) + r") \d{1,2}", text)
    return text[:m.start()] + text[m.end():] if act.get("act") == N.state.PRESENT_TXN and m else None


def _flip_disposition(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_RECORDED: say the opposite of the answer that was recorded."""
    if act.get("act") != N.state.CONFIRM_RECORDED or act.get("disposition") not in (N.value.RECOGNIZED, N.value.UNRECOGNIZED):
        return None
    said = "you don't recognize it" if act["disposition"] == N.value.RECOGNIZED else "you recognize it"
    return f"Thanks, I've recorded that {said}."


def _wrong_fee(act: dict, text: str, ctx: dict) -> str | None:
    """DISCLOSE_FEE: the fee ten dollars higher."""
    m = re.search(r"\$(\d+)", text)
    return text[:m.start()] + f"${int(m.group(1)) + 10}" + text[m.end():] if act.get("act") == N.state.DISCLOSE_FEE and m else None


def _wrong_days(act: dict, text: str, ctx: dict) -> str | None:
    """DISCLOSE_FEE: the delivery time as 9 business days."""
    if act.get("act") != N.state.DISCLOSE_FEE:
        return None
    new = re.sub(r"\b\d+(?:-\d+)? business days", "9 business days", text)
    return new if new != text else None


def _wrong_limit(act: dict, text: str, ctx: dict) -> str | None:
    """EXPLAIN_LIMIT: the transaction limit one higher."""
    m = re.search(r"up to (\d+)", text)
    return text[:m.start()] + f"up to {int(m.group(1)) + 1}" + text[m.end():] if act.get("act") == N.state.EXPLAIN_LIMIT and m else None


def _claim_done_early(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_ACTION, before any card action succeeded: claim the card was already replaced (R17)."""
    if act.get("act") != N.state.CONFIRM_ACTION or ctx["submitted"]:
        return None
    return "Your card has been replaced. " + text


def _assert_fraud(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_RECORDED of an unrecognized charge: add "This means fraud." (R27 says it is not proof)."""
    return text + " This means fraud." if act.get("act") == N.state.CONFIRM_RECORDED and act.get("disposition") == N.value.UNRECOGNIZED else None


def _drop_summary_item(act: dict, text: str, ctx: dict) -> str | None:
    """SUMMARIZE with two or more records: leave one recorded answer out (pdf.2F)."""
    if act.get("act") != N.state.SUMMARIZE or len(act.get("recorded") or {}) < 2:
        return None
    m = re.search(r" and the [A-Za-z' ]+ charge as [a-z ]+(?=[.,])", text) or re.search(r", the [A-Za-z' ]+ charge as [a-z ]+", text)
    return text[:m.start()] + text[m.end():] if m else None


CONTROLS: dict[str, Callable[[dict, str, dict], str | None]] = {
    "swap_merchant": _swap_merchant, "change_amount": _change_amount, "shift_date": _shift_date,
    "drop_date": _drop_date, "flip_disposition": _flip_disposition, "wrong_fee": _wrong_fee,
    "wrong_days": _wrong_days, "wrong_limit": _wrong_limit, "claim_done_early": _claim_done_early,
    "assert_fraud": _assert_fraud, "drop_summary_item": _drop_summary_item,
}


# ------------------------------------------------------------------ controls for a pack compiled from a compact graph
def _pack():
    from ..domain import current
    dom = current()
    return dom if dom.graph is not None else None


def _wording(dom, v: str) -> list[str]:
    say = ((dom.get("lexicon.values") or {}).get(v) or {}).get("say") or {}
    return [w for w in (say.get("canonical"), *dom.graph.aliases(v)) if w]


def _value_fields(dom, act: dict) -> list[tuple[str, str, str]]:
    """(field, value, value set) for every field of the act whose value is a member of a value set."""
    g = dom.graph
    return [(k, v, g.member_set(v)) for k, v in act.items()
            if k not in ("act", "text") and isinstance(v, str) and g.member_set(v)]


def _p_swap_value(act, text, ctx):
    """A value the act says, worded as another member of its value set."""
    dom = _pack()
    for _, v, vs in _value_fields(dom, act):
        for w in _wording(dom, v):
            if w.lower() in text.lower():
                other = next((m for m in dom.graph.members(vs) if m != v and _wording(dom, m)), None)
                if other:
                    i = text.lower().index(w.lower())
                    return text[:i] + _wording(dom, other)[0] + text[i + len(w):]
    return None


def _p_wrong_money(act, text, ctx):
    """A money figure ten dollars off."""
    m = re.search(r"\$(\d+)", text)
    return text[:m.start()] + f"${int(m.group(1)) + 10}" + text[m.end():] if m else None


def _p_wrong_limit(act, text, ctx):
    """A limit one off (an act that says a limit)."""
    dom = _pack()
    units = [str(lv.get("unit") or "").rstrip("s") for lv in dom.graph.limits.values() if lv.get("unit")]
    for u in units:
        m = re.search(rf"\b(\d+)(\s+(?:\w+\s+)?{re.escape(u)}s?)\b", text)
        if m:
            return text[:m.start()] + str(int(m.group(1)) + 1) + m.group(2) + text[m.end():]
    return None


def _p_claim_done_early(act, text, ctx):
    """A write said to be done before any write succeeded."""
    if ctx["submitted"]:
        return None
    dom = _pack()
    g = dom.graph
    for _, v, vs in _value_fields(dom, act):
        if any(vs == avs for t, sp in g.tools.items() if sp.get("role") == "write" for avs in (sp.get("args") or {}).values()):
            w = _wording(dom, v)
            if w:
                return text + f" {w[0][:1].upper() + w[0][1:]} has been submitted."
    return None


def _p_drop_required(act, text, ctx):
    """A required value left out of the words."""
    dom = _pack()
    for _, v, _vs in _value_fields(dom, act):
        for w in _wording(dom, v):
            i = text.lower().find(w.lower())
            if i >= 0:
                return " ".join((text[:i] + text[i + len(w):]).split())
    return None


# ------------------------------------------------------------------ policy controls (any pack: built from its graph)
_CORPUS: dict[str, Any] = {}


def _corpus():
    """The active pack's policy corpus (speech/policy.py), built once per pack."""
    from ..domain import current
    dom = current()
    if dom.name not in _CORPUS:
        from .contract import SpeechContract
        from .policy import PolicyCorpus
        kg = dom.kg
        _CORPUS[dom.name] = PolicyCorpus.from_contract(SpeechContract.from_graph(kg), kg, getattr(kg, "graph", None))
    return _CORPUS[dom.name]


def _speaks(act: dict, text: str) -> bool:
    """An act with words to add to (the first one in the conversation the control reaches)."""
    return bool(text.strip())


def _invented_policy(act, text, ctx):
    """A policy the graph does not have, said as policy."""
    return text + " Our policy is to always refund the full amount on request." if _speaks(act, text) else None


def _wrong_policy_figure(act, text, ctx):
    """A customer-facing limit restated in other words with the wrong number."""
    if not _speaks(act, text):
        return None
    f = next((f for f in _corpus().figures if f.attr == "limit" and not f.internal and f.unit), None)
    if f is None:
        return None
    return text + f" Just so you know, the limit is {int(float(f.value)) * 3 + 7} {f.unit.replace('_', ' ')}."


def _unauthorised_promise(act, text, ctx):
    """A promise of an action the graph has no tool for."""
    if not _speaks(act, text):
        return None
    from .policy import stem
    verb = next((v for v in ("waive", "refund", "credit", "reimburse") if stem(v) not in _corpus().action_stems), None)
    return text + f" I'll also {verb} any fees on your account this year." if verb else None


def _internal_disclosure(act, text, ctx):
    """An internal figure (marked internal in the graph, or on the project's never_volunteer list) told to the customer."""
    if not _speaks(act, text):
        return None
    from .policy import settings
    never = set(settings()["never_volunteer"])
    f = next((f for f in _corpus().figures if f.attr == "limit" and (f.internal or f.subject in never) and f.unit), None)
    if f is None:
        return None
    return text + f" I can only go up to {int(float(f.value))} {f.unit.replace('_', ' ')} with you."


POLICY_CONTROLS: dict[str, Callable[[dict, str, dict], str | None]] = {
    "invented_policy": _invented_policy, "wrong_policy_figure": _wrong_policy_figure,
    "unauthorised_promise": _unauthorised_promise, "internal_disclosure": _internal_disclosure,
}
CONTROLS.update(POLICY_CONTROLS)

PACK_CONTROLS: dict[str, Callable[[dict, str, dict], str | None]] = {
    **POLICY_CONTROLS,
    "swap_value": _p_swap_value, "wrong_money": _p_wrong_money, "wrong_limit": _p_wrong_limit,
    "claim_done_early": _p_claim_done_early, "drop_required_value": _p_drop_required,
}


def controls() -> dict[str, Callable]:
    """The controls for the active pack."""
    return PACK_CONTROLS if _pack() is not None else CONTROLS


# ------------------------------------------------------------------ a bot whose words are edited once
@dataclass
class _EditedBot:
    """Wraps a bot: the first act the control applies to gets its words edited (labels untouched)."""
    inner: Any
    tools: Any
    control: str
    turn: int = -1                      # the bot turn whose words were edited (-1: never applied)
    edit: dict = field(default_factory=dict)
    _n: int = 0

    def start(self):
        """The opening turn (edited if the control applies to it)."""
        return self._maybe(self.inner.start())

    def respond(self, text):
        """The bot's reply to ``text`` (edited if the control applies and has not been applied yet)."""
        return self._maybe(self.inner.respond(text))

    def _maybe(self, turn):
        """Edit the first act the control applies to, once per conversation; remember which turn it was."""
        from ..sut.reference_bot import BotTurn
        n, self._n = self._n, self._n + 1
        if self.turn >= 0:
            return turn
        log = getattr(self.tools, "log", None) or getattr(getattr(self.tools, "env", None), "log", []) or []
        ctx = {"merchants": {t["transaction_id"]: t["merchant"] for c in log if roles.ITEMS and c["tool"] == roles.ITEMS
                             for t in (c.get("result") or {}).get("transactions", [])},
               "submitted": any(c["tool"] in roles.WRITE_TOOLS and c.get("ok") for c in log) if _pack() is not None
               else any(c["tool"] == roles.ACTION and c.get("ok") for c in log)}
        acts = [dict(a) for a in turn.acts]
        for a in acts:
            new = controls()[self.control](a, a.get("text", ""), ctx)
            if new is not None and new != a.get("text"):
                self.edit = {"act": a.get("act"), "before": a.get("text"), "after": new}
                a["text"] = new
                self.turn = n
                return BotTurn(" ".join(x.get("text", x["act"]) for x in acts), acts, turn.ended)
        return turn


def run_controls(make_runner: Callable[[Callable], Any], bot_factory: Callable, scenarios: list,
                 controls_: list[str] | None = None, max_examples: int = 3) -> dict[str, Any]:
    """Apply every control to every scenario where it can apply; count the edited turns the speech check flags."""
    out: dict[str, Any] = {"controls": {}}
    for name in controls_ or list(controls()):
        applied = detected = 0
        missed = []
        holder: dict[str, _EditedBot] = {}

        def factory(tools, _name=name):                # the runner makes a fresh bot per conversation
            b = _EditedBot(bot_factory(tools), tools, _name)
            holder["bot"] = b
            return b
        runner = make_runner(factory)
        for scn in scenarios:
            res = runner.run(scn)
            b = holder.get("bot")
            if b is None or b.turn < 0:
                continue
            applied += 1
            turns = (res.extras.get("speech") or {}).get("turns") or []
            rep = turns[b.turn] if b.turn < len(turns) else None
            if rep is not None and not rep["ok"]:
                detected += 1
            elif len(missed) < max_examples:
                missed.append({"sid": scn.sid, **b.edit, "report": rep})
        out["controls"][name] = {"applied": applied, "detected": detected,
                                 "detection_rate": detected / applied if applied else None, "missed": missed}
    rates = [v["detection_rate"] for v in out["controls"].values() if v["detection_rate"] is not None]
    out["overall"] = {"applied": sum(v["applied"] for v in out["controls"].values()),
                      "detected": sum(v["detected"] for v in out["controls"].values()),
                      "min_rate": min(rates) if rates else None}
    return out
