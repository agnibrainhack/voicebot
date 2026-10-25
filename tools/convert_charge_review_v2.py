"""Convert the v1 charge-review policy graph into a compact policy graph (schema v2).

Everything that is a fact is carried over from the v1 JSON mechanically: limits (value, unit,
wording), value sets with their customer synonyms and attributes, tools with their statuses,
rules with their stance and wording, the knowledge-base sentences, the opening statement and
the domain model. What v1 states in prose - the dialog's branch conditions ("AUTHENTICATED and
transactions returned") - is restated here once, as the machine-readable state machine of v2.

    python tools/convert_charge_review_v2.py            # writes policy/charge_review_graph/charge_review_kg.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "gmsjourney/policy/charge_review/charge_review_kg.json"
DST = ROOT / "gmsjourney/policy/charge_review_graph/charge_review_kg.json"

# the v1 tool roles / stages and what each tool returns (v1 domain.yaml -> tools)
TOOLS = {
    "authenticate_customer": {"role": "auth", "stage": "authenticate", "args": {"credential": "credential_code"},
                              "check": {"arg": "credential", "fixture": "credential"}},
    "get_review_transactions": {"role": "read", "stage": "retrieve",
                                "returns": {"transactions": {"items": "transaction", "id": "transaction_id",
                                                             "age": "age_days"}}},
    "record_charge_response": {"role": "write", "stage": "record",
                               "args": {"transaction_id": "transaction_id", "disposition": "disposition"},
                               "echo": "transaction_id"},
    "get_card_action_options": {"role": "read", "stage": "card_action", "returns": {"options": {"subset_of": "card_action"}}},
    "submit_card_action": {"role": "write", "stage": "card_action", "args": {"action": "card_action", "delivery": "delivery"},
                           "echo": "action"},
    "transfer_to_human": {"role": "handoff", "stage": "transfer", "args": {"reason": "transfer_reason"}},
}

# machine-checkable forms of the rules (what the runtime monitor and the scoring check)
CHECKS = {
    "R2": {"requires_auth": True},
    "R4": {"max_calls": "authenticate_customer", "limit": "max_auth_attempts"},
    "R11": {"terminal_value": "record_charge_response", "arg": "disposition", "value": "UNSURE"},
    "R13": {"claim_needs_success": "CONFIRM_RECORDED", "tool": "record_charge_response"},
    "R15": {"offered_only": "submit_card_action", "arg": "action", "source": "get_card_action_options", "field": "options"},
    "R16": {"confirm_before": "submit_card_action", "act": "CONFIRM"},
    "R17": {"claim_needs_success": "ACTION_DONE", "tool": "submit_card_action"},
    "R20": [{"handoff_when_requested": True}, {"no_call_after_handoff": True}],
    "R21": {"value_from_caller": "record_charge_response", "arg": "disposition", "act": "INFORM_DISPOSITION",
            "slot": "disposition"},
    "R26": {"disclose_before": "submit_card_action", "arg": "delivery", "attr": "requires_fee_confirmation",
            "values": ["yes"], "act": "DISCLOSE_FEE"},
    "R27": {"forbidden_stance": ["unrecognized_transaction", "is_proof_of", "fraud"],
            "pattern": r"\b(is|means|confirms|confirmed|proves?) (fraud|that it was fraud)\b"},
    "R28": {"turn_limit": "max_turns"},
    "R30": {"item_window": "record_charge_response", "arg": "transaction_id", "source": "get_review_transactions",
            "field": "transactions", "age": "age_days", "limit": "lookback_days"},
    "R31": {"max_distinct": "record_charge_response", "arg": "transaction_id", "limit": "max_transactions_per_session"},
}

DISCLOSE = {"card_action": ["REPLACE_CARD"], "delivery.requires_fee_confirmation": ["yes"]}
UNSURE_OR_RECORD = [{"if": {"disposition": ["UNSURE"]},
                     "then": {"count": "unsure", "upto": "max_unsure_retries", "then": "PROVIDE_CONTEXT", "else": "RECORD"}},
                    "RECORD"]

STATES = {
    "OPENING": {"stage": "opening", "say": {"text": "$opening"}, "rules": ["R1"], "go": "ASK_AUTH"},
    "ASK_AUTH": {"stage": "authenticate", "rules": ["R1", "R2"], "yes": "UNCLEAR", "no": "UNCLEAR",
                 "on": {"PROVIDE_CREDENTIAL": "AUTHENTICATE"}},
    "ASK_AUTH_RETRY": {"stage": "authenticate", "rules": ["R3"], "yes": "UNCLEAR", "no": "UNCLEAR",
                       "on": {"PROVIDE_CREDENTIAL": "AUTHENTICATE"}},
    "AUTHENTICATE": {"stage": "authenticate", "call": "authenticate_customer", "args": {"credential": "$credential"},
                     "rules": ["R3"],
                     "on": {"AUTHENTICATED": "FETCH",
                            "FAILED": {"below": "max_auth_attempts", "then": "ASK_AUTH_RETRY", "else": "!auth_failed"},
                            "UNAVAILABLE": "!auth_unavailable"}},
    "FETCH": {"stage": "retrieve", "call": "get_review_transactions", "rules": ["R7", "R24"],
              "on": {"TRANSACTIONS": "REVIEW", "EMPTY": "EXPLAIN_NO_TXNS", "UNAVAILABLE": "!retrieve_unavailable"}},
    "REVIEW": {"stage": "retrieve", "each": "$get_review_transactions.transactions", "as": "txn",
               "where": {"age_days": {"max": "lookback_days"}}, "cap": "max_transactions_per_session",
               "rules": ["R30"], "do": "PRESENT_TXN", "empty": "EXPLAIN_NO_TXNS", "over_cap": "EXPLAIN_LIMIT",
               "done": "SUMMARIZE"},
    "PRESENT_TXN": {"stage": "capture", "say": {"txn": "$txn.transaction_id"}, "rules": ["R6", "R9", "R21", "R22"],
                    "yes": "RECOGNIZED", "no": "UNRECOGNIZED", "on": {"INFORM_DISPOSITION": UNSURE_OR_RECORD}},
    "PROVIDE_CONTEXT": {"stage": "capture", "say": {"txn": "$txn.transaction_id"}, "rules": ["R10"],
                        "yes": "RECOGNIZED", "no": "UNRECOGNIZED", "on": {"INFORM_DISPOSITION": UNSURE_OR_RECORD}},
    "RECORD": {"stage": "record", "call": "record_charge_response",
               "args": {"transaction_id": "$txn.transaction_id", "disposition": "$disposition"}, "rules": ["R12", "R33"],
               "on": {"SUCCESS": "CONFIRM_RECORDED", "FAILED": "!record_failed", "UNAVAILABLE": "!record_failed",
                      "ECHO_MISMATCH": "!record_failed"}},
    "CONFIRM_RECORDED": {"stage": "record", "say": {"txn": "$txn.transaction_id", "disposition": "$disposition"},
                         "claims": "record_charge_response", "rules": ["R13"],
                         "go": [{"if": {"disposition": ["UNSURE"]}, "then": "!unsure_twice"},
                                {"if": {"disposition": ["UNRECOGNIZED"], "card_action": {"not": [None]},
                                        "#get_card_action_options": [0]}, "then": "CARD_OPTIONS"},
                                "@next"]},
    "CARD_OPTIONS": {"stage": "card_action", "call": "get_card_action_options", "rules": ["R15", "R18"],
                     "on": {"OPTIONS": [{"if": [{"card_action": {"in": "$get_card_action_options.options"}}, DISCLOSE],
                                         "then": "DISCLOSE_FEE"},
                                        {"if": {"card_action": {"in": "$get_card_action_options.options"}},
                                         "then": "CONFIRM_ACTION"},
                                        "INFORM_NO_CARD_ACTION"],
                            "NONE": "INFORM_NO_CARD_ACTION", "UNAVAILABLE": "!card_options_unavailable"}},
    "DISCLOSE_FEE": {"stage": "card_action", "say": {"fee": "$delivery.fee_usd", "delivery": "$delivery"},
                     "rules": ["R26"], "go": "CONFIRM_ACTION"},
    "CONFIRM_ACTION": {"stage": "card_action", "say": {"action": "$card_action", "delivery": "$delivery"},
                       "rules": ["R16"], "yes": "CONFIRM", "no": "DENY",
                       "on": {"CONFIRM": "SUBMIT", "DENY": "ACK_DECLINED",
                              "REQUEST_CARD_ACTION": [{"if": DISCLOSE, "then": "DISCLOSE_FEE"}, "CONFIRM_ACTION"]}},
    "SUBMIT": {"stage": "card_action", "call": "submit_card_action",
               "args": {"action": "$card_action", "delivery": "$delivery"}, "rules": ["R16", "R33"],
               "on": {"SUCCESS": "ACTION_DONE", "FAILED": "!submit_failed", "UNAVAILABLE": "!submit_failed",
                      "ECHO_MISMATCH": "!submit_failed"}},
    "ACTION_DONE": {"stage": "card_action", "say": {"action": "$card_action", "delivery": "$delivery"},
                    "claims": "submit_card_action", "rules": ["R17", "R18"], "go": "@next"},
    "ACK_DECLINED": {"stage": "card_action", "say": {}, "rules": ["R18"], "go": "@next"},
    "INFORM_NO_CARD_ACTION": {"stage": "card_action", "say": {}, "rules": ["R15"], "go": "@next"},
    "EXPLAIN_LIMIT": {"stage": "retrieve", "say": {}, "rules": ["R31"], "go": "!transaction_limit"},
    "EXPLAIN_NO_TXNS": {"stage": "retrieve", "say": {}, "outcome": True, "rules": ["R8"], "go": "ASK_MORE_HELP"},
    "SUMMARIZE": {"stage": "finish", "say": {"recorded": {"$writes": "record_charge_response", "key": "transaction_id",
                                                          "value": "disposition"}},
                  "rules": ["R19"], "go": "ASK_MORE_HELP"},
    "ASK_MORE_HELP": {"stage": "finish", "yes": "UNCLEAR", "no": "END",
                      "on": {"END": "END", "CONFIRM": "END", "DENY": "END", "UNCLEAR": "END"}},
    "END": {"stage": "end", "end": True},
}

TRANSFERS = {
    "state": "TRANSFER", "tool": "transfer_to_human", "arg": "reason", "then": "END",
    "summary": {"act": "SUMMARIZE", "field": "recorded", "rule": "R19",
                "value": {"$writes": "record_charge_response", "key": "transaction_id", "value": "disposition"}},
    "reasons": {"auth_failed": "R4", "auth_unavailable": "R5", "retrieve_unavailable": "R14", "record_failed": "R14",
                "unsure_twice": "R11", "ambiguity_persists": "R23", "customer_request": "R20",
                "transaction_limit": "R31", "card_options_unavailable": "R32", "submit_failed": "R29",
                "turn_limit": "R28"},
}

GLOBAL = {
    "human": {"act": "REQUEST_HUMAN", "then": "!customer_request"},
    "question": {"act": "ASK_KB", "answer": "ANSWER_KB", "rule": "R25"},
    "unclear": {"act": "UNCLEAR", "clarify": "ASK_CLARIFY", "limit": "max_clarification_retries", "rule": "R22",
                "else": "!ambiguity_persists"},
    "turn_limit": {"limit": "max_turns", "rule": "R28", "else": "!turn_limit", "wrap_up": "END"},
}

ACTS = {
    "PROVIDE_CREDENTIAL": {"credential": "credential_code"},
    "INFORM_DISPOSITION": {"disposition": "disposition",
                           "card_action": {"set": "card_action", "optional": True},
                           "delivery": {"set": "delivery", "optional": True, "with": {"card_action": ["REPLACE_CARD"]}},
                           "corrected_from": {"set": "disposition", "optional": True, "correction_of": "disposition"}},
    "REQUEST_CARD_ACTION": {"card_action": "card_action",
                            "delivery": {"set": "delivery", "optional": True, "with": {"card_action": ["REPLACE_CARD"]}},
                            "_update": True},
    "CONFIRM": {}, "DENY": {}, "END": {},
}


def convert(src: Path = SRC) -> dict:
    v1 = json.loads(src.read_text())
    T = v1["triples"]
    tails = lambda h, r: [t["tail"] for t in T if t["head"] == h and t["relation"] == r]
    heads = lambda r, tl: [t["head"] for t in T if t["relation"] == r and t["tail"] == tl]
    limits = {}
    for k in heads("is_a", "limit"):
        limits[k] = {"value": tails(k, "has_value")[0], "unit": tails(k, "has_unit")[0], "text": tails(k, "has_text")[0]}
        if k in ("max_clarification_retries", "max_unsure_retries", "max_turns"):
            limits[k]["internal"] = True
    values = {}
    for vs in ("disposition", "card_action", "delivery"):
        values[vs] = {}
        for m in heads("member_of", vs):
            spec = {"aliases": tails(m, "has_alias")}
            for t in T:
                if t["head"] == m and t["relation"] in ("has_fee_usd", "has_business_days_min", "has_business_days_max",
                                                         "requires_fee_confirmation"):
                    attr = t["relation"][4:] if t["relation"].startswith("has_") else t["relation"]
                    spec[attr] = t["tail"]
            values[vs][m] = spec
    statuses = {t: (tails(t, "success_value")[0], [f for f in tails(t, "failure_value") if f != "UNAVAILABLE"])
                for t in heads("is_a", "tool")}
    tools = {}
    for t, sp in TOOLS.items():
        ok, fail = statuses[t]
        text = next((x["evidence"] for x in T if x["head"] == t and x["relation"] == "is_a"), "")
        tools[t] = {**sp, "ok": ok, "fail": fail, "text": text}
    stance = {t["head"]: t["tail"] for t in T if t["relation"] == "has_stance"}
    # the fact each rule states (v1: the triple qualified by the rule), carried over for the rules
    # that have no machine check, so the contradiction primitive compares what the rules say
    statement = {(t.get("qualifiers") or {}).get("rule"): [t["head"], t["relation"], t["tail"]]
                 for t in T if (t.get("qualifiers") or {}).get("rule") and t["kind"] != "trajectory"}
    rules = {}
    for rid in sorted(stance, key=lambda r: int(r[1:])):
        r = [stance[rid], tails(rid, "has_text")[0]]
        if rid in CHECKS:
            r.append(CHECKS[rid])
        elif statement.get(rid) and rid not in TRANSFERS["reasons"].values():   # an escalation rule states
            r.append({"states": statement[rid]})                                 # its graph transfer reason
        rules[rid] = r
    facts = [[t["head"], t["relation"], t["tail"]] for t in T if t["kind"] == "domain"
             and t["relation"] in ("has_session", "reviews", "governed_by", "subject_to", "has_disposition", "limited_by",
                                   "delivered_by", "precedes", "precedes_two")]
    return {
        "schema": "gmsjourney.policy_graph/2", "id": "charge_review", "title": v1["title"], "root": "charge_review_policy",
        "source": {"document": v1["source"]["document"], "converted_from": src.name, "v1_sha256": v1.get("sha256")},
        "opening": tails("opening_statement", "has_text")[0],
        "knowledge": [t["tail"] for t in T if t["relation"] == "states" and t["head"] == "knowledge_base"],
        "limits": limits, "values": values, "acts": ACTS, "tools": tools, "rules": rules, "facts": facts,
        "transfers": TRANSFERS, "global": GLOBAL, "states": STATES,
    }


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from gmsjourney.graph.model import dump_compact
    out = convert()
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(dump_compact(out))
    print(f"wrote {DST.relative_to(ROOT)} ({len(DST.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
