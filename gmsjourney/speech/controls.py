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
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


# ------------------------------------------------------------------ the edits
# Each edit takes (act, its text, ctx) and returns the new text, or None when it does not apply to this act.
# ctx: {"merchants": {transaction_id: merchant} returned so far, "submitted": a card action has succeeded}
def _swap_merchant(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: name another returned charge's merchant (the amount and date stay this charge's)."""
    if act.get("act") != "PRESENT_TXN" or len(ctx["merchants"]) < 2:
        return None
    mine = ctx["merchants"].get(act.get("txn"))
    other = next((m for t, m in ctx["merchants"].items() if t != act.get("txn")), None)
    return text.replace(mine, other) if mine and other and mine in text else None


def _change_amount(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN / PROVIDE_CONTEXT: the charge's amount one dollar higher."""
    if act.get("act") not in ("PRESENT_TXN", "PROVIDE_CONTEXT"):
        return None
    m = re.search(r"\$(\d+\.\d{2})", text)
    return text[:m.start()] + f"${float(m.group(1)) + 1:.2f}" + text[m.end():] if m else None


def _shift_date(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: the charge's date one day off."""
    m = re.search(r"\b(" + "|".join(MONTHS) + r") (\d{1,2})\b", text)
    if act.get("act") != "PRESENT_TXN" or not m:
        return None
    d = int(m.group(2))
    return text[:m.start()] + f"{m.group(1)} {d + 1 if d < 28 else d - 1}" + text[m.end():]


def _drop_date(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: leave the date out (pdf.2B requires merchant, amount and date)."""
    m = re.search(r" on (" + "|".join(MONTHS) + r") \d{1,2}", text)
    return text[:m.start()] + text[m.end():] if act.get("act") == "PRESENT_TXN" and m else None


def _flip_disposition(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_RECORDED: say the opposite of the answer that was recorded."""
    if act.get("act") != "CONFIRM_RECORDED" or act.get("disposition") not in ("RECOGNIZED", "UNRECOGNIZED"):
        return None
    said = "you don't recognize it" if act["disposition"] == "RECOGNIZED" else "you recognize it"
    return f"Thanks, I've recorded that {said}."


def _wrong_fee(act: dict, text: str, ctx: dict) -> str | None:
    """DISCLOSE_FEE: the fee ten dollars higher."""
    m = re.search(r"\$(\d+)", text)
    return text[:m.start()] + f"${int(m.group(1)) + 10}" + text[m.end():] if act.get("act") == "DISCLOSE_FEE" and m else None


def _wrong_days(act: dict, text: str, ctx: dict) -> str | None:
    """DISCLOSE_FEE: the delivery time as 9 business days."""
    if act.get("act") != "DISCLOSE_FEE":
        return None
    new = re.sub(r"\b\d+(?:-\d+)? business days", "9 business days", text)
    return new if new != text else None


def _wrong_limit(act: dict, text: str, ctx: dict) -> str | None:
    """EXPLAIN_LIMIT: the transaction limit one higher."""
    m = re.search(r"up to (\d+)", text)
    return text[:m.start()] + f"up to {int(m.group(1)) + 1}" + text[m.end():] if act.get("act") == "EXPLAIN_LIMIT" and m else None


def _claim_done_early(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_ACTION, before any card action succeeded: claim the card was already replaced (R17)."""
    if act.get("act") != "CONFIRM_ACTION" or ctx["submitted"]:
        return None
    return "Your card has been replaced. " + text


def _assert_fraud(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_RECORDED of an unrecognized charge: add "This means fraud." (R27 says it is not proof)."""
    return text + " This means fraud." if act.get("act") == "CONFIRM_RECORDED" and act.get("disposition") == "UNRECOGNIZED" else None


def _drop_summary_item(act: dict, text: str, ctx: dict) -> str | None:
    """SUMMARIZE with two or more records: leave one recorded answer out (pdf.2F)."""
    if act.get("act") != "SUMMARIZE" or len(act.get("recorded") or {}) < 2:
        return None
    m = re.search(r" and the [A-Za-z' ]+ charge as [a-z ]+(?=[.,])", text) or re.search(r", the [A-Za-z' ]+ charge as [a-z ]+", text)
    return text[:m.start()] + text[m.end():] if m else None


CONTROLS: dict[str, Callable[[dict, str, dict], str | None]] = {
    "swap_merchant": _swap_merchant, "change_amount": _change_amount, "shift_date": _shift_date,
    "drop_date": _drop_date, "flip_disposition": _flip_disposition, "wrong_fee": _wrong_fee,
    "wrong_days": _wrong_days, "wrong_limit": _wrong_limit, "claim_done_early": _claim_done_early,
    "assert_fraud": _assert_fraud, "drop_summary_item": _drop_summary_item,
}


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
        ctx = {"merchants": {t["transaction_id"]: t["merchant"] for c in log if c["tool"] == "get_review_transactions"
                             for t in (c.get("result") or {}).get("transactions", [])},
               "submitted": any(c["tool"] == "submit_card_action" and c.get("ok") for c in log)}
        acts = [dict(a) for a in turn.acts]
        for a in acts:
            new = CONTROLS[self.control](a, a.get("text", ""), ctx)
            if new is not None and new != a.get("text"):
                self.edit = {"act": a.get("act"), "before": a.get("text"), "after": new}
                a["text"] = new
                self.turn = n
                return BotTurn(" ".join(x.get("text", x["act"]) for x in acts), acts, turn.ended)
        return turn


def run_controls(make_runner: Callable[[Callable], Any], bot_factory: Callable, scenarios: list,
                 controls: list[str] | None = None, max_examples: int = 3) -> dict[str, Any]:
    """Apply every control to every scenario where it can apply; count the edited turns the speech check flags."""
    out: dict[str, Any] = {"controls": {}}
    for name in controls or list(CONTROLS):
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
