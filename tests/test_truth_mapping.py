"""Every kind of claim is held to the right truth, and a wrong value fails (the truth-mapping audit as tests)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_turn_grading import _bench  # noqa: E402

TX = {"tool": "get_review_transactions", "args": {}, "result": {"status": "TRANSACTIONS", "transactions": [
    {"transaction_id": "T1", "merchant": "Amazon", "amount": "45.67"}]}}


def rec(d, st="SUCCESS"):
    return {"tool": "record_charge_response", "args": {"transaction_id": "T1", "disposition": d},
            "result": {"status": st, "transaction_id": "T1", "disposition": d}}


def opts(o):
    return {"tool": "get_card_action_options", "args": {}, "result": {"status": "OPTIONS", "options": o}}


def sub(a, dl, st="SUCCESS"):
    return {"tool": "submit_card_action", "args": {"action": a, "delivery": dl},
            "result": {"status": st, "action": a, "delivery": dl}}


def cust(**k):
    from gmsjourney.s04_cases.meaning import SemanticFrame as F
    return [F.make("INFORM_DISPOSITION", **k)]


UNREC = dict(disposition="UNRECOGNIZED")
CHOSE = dict(disposition="UNRECOGNIZED", card_action="REPLACE_CARD", delivery="expedited_delivery")
CASES = [
    # (name, tool log, customer, words, label, every claim as field=value->status)
    ("recorded, the write succeeded", [TX, rec("UNRECOGNIZED")], UNREC, "Thanks, I've noted the Amazon charge as unrecognized.",
     {"act": "CONFIRM_RECORDED", "txn": "T1", "disposition": "UNRECOGNIZED"}, ["disposition=UNRECOGNIZED->supported"]),
    ("recorded, the write FAILED", [TX, rec("UNRECOGNIZED", "FAILED")], UNREC, "Thanks, I've noted the Amazon charge as unrecognized.",
     None, ["disposition=UNRECOGNIZED->unsupported"]),
    ("recorded, no write at all", [TX], UNREC, "Thanks, I've noted the Amazon charge as unrecognized.", None,
     ["disposition=UNRECOGNIZED->unsupported"]),
    ("recorded the wrong value", [TX, rec("RECOGNIZED")], UNREC, "Thanks, I've noted the Amazon charge as recognized.", None,
     ["disposition=RECOGNIZED->contradicted"]),
    ("said back what the customer said", [TX], UNREC, "So you don't recognize the Amazon charge.", None,
     ["disposition=UNRECOGNIZED->supported"]),
    ("options offered, both returned", [TX, opts(["REPLACE_CARD", "SECURE_CARD"])], UNREC,
     "Would you like to replace the card or secure the card?", None,
     ["card_action=REPLACE_CARD->supported", "card_action=SECURE_CARD->supported"]),
    ("an option offered that was not returned", [TX, opts(["REPLACE_CARD"])], UNREC,
     "Would you like to replace the card or secure the card?", None,
     ["card_action=REPLACE_CARD->supported", "card_action=SECURE_CARD->unsupported"]),
    ("done, the submit succeeded", [TX, opts(["REPLACE_CARD"]), sub("REPLACE_CARD", "expedited_delivery")], CHOSE,
     "All done: we will replace the card with expedited delivery.", None,
     ["card_action=REPLACE_CARD->supported", "delivery=expedited_delivery->supported"]),
    ("done, the submit FAILED", [TX, opts(["REPLACE_CARD"]), sub("REPLACE_CARD", "expedited_delivery", "FAILED")], CHOSE,
     "All done: we will replace the card with expedited delivery.", None,
     ["card_action=REPLACE_CARD->unsupported", "delivery=expedited_delivery->unsupported"]),
    ("done with another delivery", [TX, opts(["REPLACE_CARD"]), sub("REPLACE_CARD", "expedited_delivery")], CHOSE,
     "All done: we will replace the card with standard delivery.", None,
     ["card_action=REPLACE_CARD->supported", "delivery=standard_delivery->contradicted"]),
    ("confirming another delivery than the customer's", [TX, opts(["REPLACE_CARD"])], CHOSE,
     "To confirm: you'd like to replace the card with standard delivery. Shall I go ahead?", None,
     ["card_action=REPLACE_CARD->supported", "delivery=standard_delivery->contradicted"]),
    ("the fee, right", [TX], CHOSE, "Expedited delivery has a fee of $15.", None,
     ["delivery=expedited_delivery->supported", "fee_usd=15->supported"]),
    ("the fee, wrong", [TX], CHOSE, "Expedited delivery has a fee of $12.", None,
     ["delivery=expedited_delivery->supported", "fee_usd=12->contradicted"]),
    ("business days, wrong", [TX], UNREC, "Standard delivery takes 1 to 2 business days.", None,
     ["delivery=standard_delivery->unverifiable", "business_days_min=(1, 2)->contradicted"]),
    ("a limit, wrong", [TX], UNREC, "I can only review 6 charges in one call.", None, ["limit=6->contradicted"]),
    ("a count is not a limit", [TX], UNREC, "I found 2 transactions to review.", None, []),
    ("a charge and its amount", [TX], UNREC, "Let's look at the Amazon charge for $45.67.", None,
     ["item=T1->supported", "amount=45.67->supported"]),
    ("a charge with the wrong amount", [TX], UNREC, "Let's look at the Amazon charge for $54.67.", None,
     ["item=T1->supported", "amount=54.67->contradicted"]),
    ("a charge no tool returned", [TX], UNREC, "Let's look at the Walmart charge for $99.10.", None,
     ["item=walmart->unsupported", "amount=99.1->unsupported"]),
]


@pytest.mark.parametrize("name,log,customer,words,label,want", CASES, ids=[c[0] for c in CASES])
def test_each_claim_is_held_to_its_truth(name, log, customer, words, label, want):
    from gmsjourney.s07_checks.speech_reading import SessionFacts
    PG, B = _bench()
    v = B.speech
    f = SessionFacts.from_run(log, cust(**customer), None, v.c.success_values)
    got = [f"{c.field}={c.value}->{v.verify_claim(c, f, label).status}" for c in v.reader.read(words, f).claims]
    if not any(w.startswith(("item=", "amount=")) for w in want):     # the charge's own cards are tested on their own
        got = [g for g in got if not g.startswith(("item=", "amount="))]
    assert sorted(got) == sorted(want), (name, got)
