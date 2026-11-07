"""A scripted stand-in for the gateway model, for tests and demos of the LLM bot without network access.

    GMSJOURNEY_SUT_FAKE_MODEL=fake_sut_model:HappyPath   (with tests/ on PYTHONPATH)

It plays a minimal charge-review bot (authenticate, present the first charge, record the answer, summarise, end)
from the JSON conversation the LLM bot sends, so the console's LLM-bot paths can be exercised end to end. It is not
a good bot on purpose: it reviews only the first charge and records every answer as RECOGNIZED, so the checks have
something to find.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace


def _r(obj):
    return SimpleNamespace(content=json.dumps(obj), usage_metadata={})


class HappyPath:
    def __init__(self):
        self.calls = 0

    def invoke(self, msgs):
        self.calls += 1
        conv = json.loads(msgs[-1].content)["conversation"]
        last = conv[-1]
        say = lambda acts, end=False: _r({"say": acts, "end": end})          # noqa: E731
        if "event" in last:
            return say([{"act": "OPENING", "text": "Before I review account-specific charges, I need to authenticate you."},
                        {"act": "ASK_AUTH", "text": "Please tell me the last four digits of your card."}])
        if "tool" in last:
            n, r = last["tool"], last["result"] if isinstance(last["result"], dict) else {}
            if n == "authenticate_customer":
                if r.get("status") != "AUTHENTICATED":
                    return say([{"act": "ASK_AUTH_RETRY", "text": "That didn't match. Please try again."}])
                return _r({"tool": "get_review_transactions", "args": {}})
            if n == "get_review_transactions":
                ts = r.get("transactions") or []
                if not ts:
                    return say([{"act": "EXPLAIN_NO_TXNS", "text": "There are no charges to review."},
                                {"act": "END", "text": "Goodbye."}], True)
                t = ts[0]
                return say([{"act": "PRESENT_TXN", "txn": t.get("transaction_id"),
                             "text": f"I see a charge at {t.get('merchant')} for ${t.get('amount')}. Do you recognize it?"}])
            if n == "record_charge_response":
                return say([{"act": "CONFIRM_RECORDED", "txn": r.get("transaction_id"), "disposition": r.get("disposition"),
                             "text": "Thanks, I've recorded that."},
                            {"act": "SUMMARIZE", "recorded": {r.get("transaction_id"): r.get("disposition")}, "actions": [],
                             "text": "That's everything I recorded today."},
                            {"act": "ASK_MORE_HELP", "text": "Anything else?"}])
            return say([{"act": "ASK_MORE_HELP", "text": "Is there anything else?"}])
        c = last.get("customer", "").lower()
        prev = [x for x in conv if "bot" in x][-1]["bot"][-1]
        if prev["act"] in ("ASK_AUTH", "ASK_AUTH_RETRY"):
            return _r({"tool": "authenticate_customer", "args": {"credential": "".join(re.findall(r"\d", c))[-4:]}})
        if prev["act"] == "PRESENT_TXN":
            return _r({"tool": "record_charge_response",
                       "args": {"transaction_id": prev.get("txn"), "disposition": "RECOGNIZED"}})
        return say([{"act": "END", "text": "Thank you, goodbye."}], True)
