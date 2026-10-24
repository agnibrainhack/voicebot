"""The policy as one clean head / relation / tail graph, with dialog states (book Ch. 2-4).

``policy/charge_review/charge_review_kg.json`` is the single machine-readable source the rest of the
pipeline reads. It is built once from the operating-procedure PDF by
``python -m gmsjourney.knowledge.policy_kg build`` and committed; every later step
(GMS training, the oracle's limits, the state-aware ontology mapper, the exports) reads
the JSON through ``PolicyKG``.

What it holds, as triples ``{head, relation, tail}`` with a ``kind``:

* trajectory  - the tool workflow (``authenticate_customer enables get_review_transactions``,
                with the edge id and condition) and the dialog-state machine
                (``ASK_AUTH leads_to PRESENT_TXN`` with the customer act and condition);
* state       - what holds *in* each dialog state: the stage it belongs to, the customer
                meaning cards it accepts, the rules and limits that apply there, the tools
                the bot may call next, how a bare "yes"/"no" is read there, and which
                escalation triggers can fire there;
* policy      - rules (stance, subject, predicate, object, text), limits (exact values),
                delivery options, tools and their result values, escalation triggers,
                knowledge-base sentences, the required opening;
* vocabulary  - value sets, their members, aliases (curated synonyms), the dialog acts and
                the value set each slot draws from;
* domain      - the domain model (customer has_session review_session, ...).

Each triple carries ``source`` (a PDF section such as ``pdf.2C``, ``program`` for the two
limits the business owner added, or ``lexicon`` for curated synonyms that are not in the
PDF), an ``evidence`` quote that is a verbatim substring of that source, and optional
``qualifiers``. ``gms: true`` marks the triples the GMS is trained on.

The dialog-state layer below is written from PDF sections 2, 3 and 7 and checked against
the policy oracle: every (state, customer act) pair and every state transition the oracle
produces must be declared here (tests/test_selfchecks.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# authoring: build_kg below writes the charge-review graph into its domain pack
POLICY_DIR = Path(__file__).resolve().parents[1] / "policy" / "charge_review"
KG_PATH = POLICY_DIR / "charge_review_kg.json"
TRANSCRIPTION = POLICY_DIR / "charge_review_policy.md"
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ID = "gmsjourney.policy_kg/1"

# --------------------------------------------------------------------------- the brief
PROGRAM_BRIEF = ("At most 4 transactions (charges) can be verified, at most 2 times incorrect authentication "
                 "is allowed and the conversation can go for at max 20 turns.")
LEXICON_NOTE = "Curated customer synonyms for each canonical value (not stated in the PDF)."

# --------------------------------------------------------------------------- dialog acts
# Customer meaning cards: act -> {slot: (value set, required)}
ACT_SCHEMA: dict[str, dict[str, tuple[str, bool]]] = {
    "PROVIDE_CREDENTIAL": {"credential": ("credential_code", True)},
    "INFORM_DISPOSITION": {"disposition": ("disposition", True), "card_action": ("card_action", False),
                           "delivery": ("delivery", False), "corrected_from": ("disposition", False)},
    "REQUEST_CARD_ACTION": {"card_action": ("card_action", True), "delivery": ("delivery", False)},
    "CONFIRM": {}, "DENY": {}, "END": {}, "UNCLEAR": {}, "REQUEST_HUMAN": {},
    "ASK_KB": {"intent": ("kb_intent", True), "attr": ("kb_attribute", False), "entity": ("delivery", False),
               "key": ("limit", False), "value": ("number", False), "options": ("number", False),
               "days": ("number", False), "name": ("option_name", False), "subject": ("card_action", False)},
}
ACT_SOURCE = {"PROVIDE_CREDENTIAL": "pdf.2A", "INFORM_DISPOSITION": "pdf.2C", "REQUEST_CARD_ACTION": "pdf.2E",
              "CONFIRM": "pdf.2E", "DENY": "pdf.2E", "END": "pdf.2F", "UNCLEAR": "pdf.3",
              "REQUEST_HUMAN": "pdf.2F", "ASK_KB": "pdf.3"}
EXTRA_VALUE_SETS = {
    "kb_intent": ["value", "limit", "plausibility", "mcq", "cheaper", "faster", "faster_option_fee",
                  "fee_difference", "count_options", "expedited_total", "needs_confirmation", "window_check",
                  "procedure", "refund", "fraud", "skip_auth", "unknown_option", "unstated_fact"],
    "kb_attribute": ["fee", "days"],
}
ACT_EVIDENCE = {
    "PROVIDE_CREDENTIAL": "Call authenticate_customer.",
    "INFORM_DISPOSITION": "A clear answer maps to RECOGNIZED or UNRECOGNIZED.",
    "REQUEST_CARD_ACTION": "also asks to secure/replace the card",
    "CONFIRM": "confirm the requested action", "DENY": "confirm the requested action",
    "END": "After the available charges are reviewed, summarize the responses",
    "UNCLEAR": "If the bot cannot understand required information or the customer response is ambiguous, ask one "
               "short clarification and make only one additional attempt.",
    "REQUEST_HUMAN": "If the customer requests a human at any point, transfer the interaction.",
    "ASK_KB": "General policy questions may be answered from the knowledge base.",
}
# where the best-matching sentence is not the one that states the rule
RULE_EVIDENCE = {
    "R16": "confirm the requested action, call submit_card_action",
    "R17": "describe it as completed only after SUCCESS",
    "R22": "Do not guess.",
    "R29": "Do not claim an action completed unless its API returns SUCCESS.",
    "R32": "If a required tool fails, transfer to a human.",
    "R33": "The bot must not claim that the response was recorded unless the API returns SUCCESS.",
}
GLOBAL_ACTS = ("REQUEST_HUMAN", "ASK_KB", "UNCLEAR")      # R20, R25, R22: admissible in every prompt state


@dataclass(frozen=True)
class StateSpec:
    stage: str
    prompts: bool                     # does the customer answer in this state?
    source: str
    evidence: str
    accepts: tuple[str, ...] = ()
    yes: str | None = None            # how a bare "yes" is read here
    no: str | None = None             # how a bare "no" is read here
    rules: tuple[str, ...] = ()
    limits: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()       # tools the bot may call next from this state


DISP = ("INFORM_DISPOSITION", "REQUEST_CARD_ACTION")
STATES: dict[str, StateSpec] = {
    "OPENING": StateSpec("opening", False, "pdf.2A", "Play the required opening statement.",
                         rules=("R1", "R2")),
    "ASK_AUTH": StateSpec("authenticate", True, "pdf.2A", "Call authenticate_customer.",
                          accepts=("PROVIDE_CREDENTIAL",), yes="UNCLEAR", no="UNCLEAR",
                          rules=("R2", "R3", "R5", "R20", "R22", "R23", "R25", "R28"),
                          limits=("max_auth_attempts", "max_turns"),
                          tools=("authenticate_customer", "get_review_transactions", "transfer_to_human")),
    "ASK_AUTH_RETRY": StateSpec("authenticate", True, "program", "at most 2 times incorrect authentication",
                                accepts=("PROVIDE_CREDENTIAL",), yes="UNCLEAR", no="UNCLEAR",
                                rules=("R3", "R4", "R5", "R20", "R22", "R23", "R25", "R28"),
                                limits=("max_auth_attempts", "max_turns"),
                                tools=("authenticate_customer", "get_review_transactions", "transfer_to_human")),
    "PRESENT_TXN": StateSpec("capture", True, "pdf.2C", "Ask whether the customer recognizes the transaction.",
                             accepts=DISP, yes="RECOGNIZED", no="UNRECOGNIZED",
                             rules=("R6", "R7", "R9", "R12", "R18", "R20", "R21", "R22", "R24", "R25", "R27",
                                    "R28", "R30", "R31"),
                             limits=("max_transactions_per_session", "lookback_days", "max_turns"),
                             tools=("record_charge_response", "transfer_to_human")),
    "ANSWER_KB": StateSpec("capture", False, "pdf.3",
                           "General policy questions may be answered from the knowledge base.",
                           rules=("R24", "R25", "R26", "R27")),
    "ASK_CLARIFY": StateSpec("capture", True, "pdf.3", "ask one short clarification",
                             accepts=DISP, yes="RECOGNIZED", no="UNRECOGNIZED",
                             rules=("R9", "R20", "R21", "R22", "R23", "R28"),
                             limits=("max_clarification_retries", "max_turns"),
                             tools=("record_charge_response", "transfer_to_human")),
    "PROVIDE_CONTEXT": StateSpec("capture", True, "pdf.2C",
                                 "If the customer is unsure, provide available transaction context and ask once more.",
                                 accepts=DISP, yes="RECOGNIZED", no="UNRECOGNIZED",
                                 rules=("R7", "R10", "R11", "R20", "R21", "R28"),
                                 limits=("max_unsure_retries", "max_turns"),
                                 tools=("record_charge_response", "transfer_to_human")),
    "CONFIRM_RECORDED": StateSpec("record", False, "pdf.2D",
                                  "Call record_charge_response for the current transaction before advancing.",
                                  rules=("R12", "R13", "R14", "R18", "R31", "R32", "R33"),
                                  limits=("max_transactions_per_session",),
                                  tools=("get_card_action_options", "transfer_to_human")),
    "DISCLOSE_FEE": StateSpec("card_action", False, "pdf.5",
                              "Before selecting expedited delivery, the bot must tell the customer the $15 fee",
                              rules=("R26",)),
    "CONFIRM_ACTION": StateSpec("card_action", True, "pdf.2E", "confirm the requested action",
                                accepts=("CONFIRM", "DENY", "REQUEST_CARD_ACTION"), yes="CONFIRM", no="DENY",
                                rules=("R15", "R16", "R18", "R20", "R26", "R28", "R29"), limits=("max_turns",),
                                tools=("submit_card_action", "transfer_to_human")),
    "ACTION_DONE": StateSpec("card_action", False, "pdf.2E", "describe it as completed only after SUCCESS",
                             rules=("R17", "R33")),
    "ACK_DECLINED": StateSpec("card_action", False, "pdf.2E",
                              "This second intent must not erase the charge-review intent.",
                              rules=("R16", "R18")),
    "INFORM_NO_CARD_ACTION": StateSpec("card_action", False, "pdf.4", "Offer only returned options.",
                                       rules=("R15", "R18")),
    "EXPLAIN_LIMIT": StateSpec("retrieve", False, "pdf.5",
                               "The bot may review up to 4 recent transactions in one self-service session.",
                               rules=("R31",), limits=("max_transactions_per_session",),
                               tools=("transfer_to_human",)),
    "EXPLAIN_NO_TXNS": StateSpec("retrieve", False, "pdf.2B",
                                 "If the tool returns no transactions, explain that there are no charges available "
                                 "for review", rules=("R8",)),
    "SUMMARIZE": StateSpec("finish", False, "pdf.2F", "summarize the responses and any account action that "
                           "actually completed", rules=("R13", "R17", "R19"), tools=("transfer_to_human",)),
    "ASK_MORE_HELP": StateSpec("finish", True, "pdf.2B", "ask whether the customer needs other help",
                               accepts=("END",), yes="UNCLEAR", no="END",
                               rules=("R8", "R20", "R25", "R28"), limits=("max_turns",),
                               tools=("transfer_to_human",)),
    "TRANSFER": StateSpec("transfer", False, "pdf.2F",
                          "If the customer requests a human at any point, transfer the interaction.",
                          rules=("R20",), tools=("transfer_to_human",)),
    "END": StateSpec("end", False, "pdf.7", "then end."),
}

# (from, to, on, when, rules, edges, source, evidence); on = customer act or "auto"
TRANSITIONS: list[tuple[str, str, str, str, tuple[str, ...], tuple[str, ...], str, str]] = [
    ("OPENING", "ASK_AUTH", "auto", "always", ("R1", "R2"), ("W1",), "pdf.7",
     "Authenticate customer. On failure, transfer to human. On success, continue."),
    ("ASK_AUTH", "ASK_AUTH_RETRY", "PROVIDE_CREDENTIAL",
     "authenticate_customer FAILED and attempts below max_auth_attempts", ("R3",), ("W2",), "program",
     "at most 2 times incorrect authentication"),
    ("ASK_AUTH", "PRESENT_TXN", "PROVIDE_CREDENTIAL", "AUTHENTICATED and transactions returned",
     ("R3", "R6"), ("W3",), "pdf.2B", "Present one transaction at a time using merchant, amount and date."),
    ("ASK_AUTH", "EXPLAIN_NO_TXNS", "PROVIDE_CREDENTIAL", "AUTHENTICATED and no transactions returned",
     ("R8",), ("W3", "W6"), "pdf.2B", "If the tool returns no transactions"),
    ("ASK_AUTH", "TRANSFER", "PROVIDE_CREDENTIAL", "authenticate_customer or get_review_transactions UNAVAILABLE",
     ("R5", "R14"), ("W4", "W7"), "pdf.2A", "If authentication fails or the tool is unavailable, transfer to a human"),
    ("ASK_AUTH", "ASK_AUTH", "UNCLEAR", "first unclear answer", ("R22",), (), "pdf.3",
     "ask one short clarification"),
    ("ASK_AUTH", "TRANSFER", "UNCLEAR", "still unclear after one retry", ("R23",), (), "pdf.3",
     "transfer to a human representative"),
    ("ASK_AUTH", "TRANSFER", "REQUEST_HUMAN", "customer asks for a human", ("R20",), ("W21",), "pdf.2F",
     "If the customer requests a human at any point, transfer the interaction."),
    ("ASK_AUTH_RETRY", "PRESENT_TXN", "PROVIDE_CREDENTIAL", "AUTHENTICATED and transactions returned",
     ("R3", "R6"), ("W3",), "pdf.7", "On success, continue."),
    ("ASK_AUTH_RETRY", "EXPLAIN_NO_TXNS", "PROVIDE_CREDENTIAL", "AUTHENTICATED and no transactions returned",
     ("R8",), ("W3", "W6"), "pdf.2B", "If the tool returns no transactions"),
    ("ASK_AUTH_RETRY", "TRANSFER", "PROVIDE_CREDENTIAL", "authenticate_customer FAILED at max_auth_attempts",
     ("R4",), ("W4",), "pdf.7", "On failure, transfer to human."),
    ("ASK_AUTH_RETRY", "TRANSFER", "REQUEST_HUMAN", "customer asks for a human", ("R20",), (), "pdf.2F",
     "If the customer requests a human at any point, transfer the interaction."),
    ("PRESENT_TXN", "CONFIRM_RECORDED", "INFORM_DISPOSITION",
     "clear RECOGNIZED/UNRECOGNIZED and record_charge_response SUCCESS", ("R9", "R12"), ("W5",), "pdf.2C",
     "A clear answer maps to RECOGNIZED or UNRECOGNIZED."),
    ("PRESENT_TXN", "PROVIDE_CONTEXT", "INFORM_DISPOSITION", "disposition UNSURE (first time)", ("R10",), (),
     "pdf.2C", "If the customer is unsure, provide available transaction context and ask once more."),
    ("PRESENT_TXN", "ASK_CLARIFY", "UNCLEAR", "answer unclear (first time)", ("R22",), (), "pdf.3",
     "ask one short clarification"),
    ("PRESENT_TXN", "ANSWER_KB", "ASK_KB", "general policy question", ("R25",), (), "pdf.7",
     "General policy questions may trigger a knowledge-base lookup."),
    ("PRESENT_TXN", "TRANSFER", "INFORM_DISPOSITION", "record_charge_response not SUCCESS", ("R14",), ("W11",),
     "pdf.4", "Advance only after SUCCESS; otherwise transfer."),
    ("PRESENT_TXN", "TRANSFER", "REQUEST_HUMAN", "customer asks for a human", ("R20",), (), "pdf.2F",
     "If the customer requests a human at any point, transfer the interaction."),
    ("PRESENT_TXN", "SUMMARIZE", "REQUEST_HUMAN", "customer asks for a human after responses were recorded",
     ("R19", "R20"), (), "pdf.2F", "summarize the responses"),
    ("PRESENT_TXN", "SUMMARIZE", "INFORM_DISPOSITION", "record SUCCESS and no transaction pending",
     ("R12", "R19"), ("W10",), "pdf.2F", "After the available charges are reviewed, summarize the responses"),
    ("ANSWER_KB", "PRESENT_TXN", "auto", "re-ask the pending transaction", ("R6",), (), "pdf.7",
     "Present one transaction and ask whether the customer recognizes it."),
    ("ASK_CLARIFY", "CONFIRM_RECORDED", "INFORM_DISPOSITION", "clear answer after clarification and SUCCESS",
     ("R9", "R12"), ("W5",), "pdf.3", "make only one additional attempt"),
    ("ASK_CLARIFY", "PROVIDE_CONTEXT", "INFORM_DISPOSITION", "disposition UNSURE", ("R10",), (), "pdf.2C",
     "provide available transaction context and ask once more"),
    ("ASK_CLARIFY", "TRANSFER", "UNCLEAR", "still unclear after one retry", ("R23",), (), "pdf.3",
     "If the required information or decision is still unclear after that one retry, transfer to a human "
     "representative."),
    ("ASK_CLARIFY", "TRANSFER", "REQUEST_HUMAN", "customer asks for a human", ("R20",), (), "pdf.2F",
     "If the customer requests a human at any point, transfer the interaction."),
    ("ASK_CLARIFY", "SUMMARIZE", "UNCLEAR", "still unclear after one retry and responses recorded",
     ("R19", "R23"), (), "pdf.3", "transfer to a human representative"),
    ("PROVIDE_CONTEXT", "CONFIRM_RECORDED", "INFORM_DISPOSITION", "clear answer after context and SUCCESS",
     ("R9", "R12"), ("W5",), "pdf.2C", "A clear answer maps to RECOGNIZED or UNRECOGNIZED."),
    ("PROVIDE_CONTEXT", "TRANSFER", "INFORM_DISPOSITION", "still UNSURE: record UNSURE, then transfer",
     ("R11",), ("W11",), "pdf.2C", "If still unsure, record UNSURE and transfer."),
    ("PROVIDE_CONTEXT", "SUMMARIZE", "INFORM_DISPOSITION", "still UNSURE and responses recorded",
     ("R11", "R19"), ("W11",), "pdf.2C", "If still unsure, record UNSURE and transfer."),
    ("PROVIDE_CONTEXT", "TRANSFER", "REQUEST_HUMAN", "customer asks for a human", ("R20",), (), "pdf.2F",
     "If the customer requests a human at any point, transfer the interaction."),
    ("CONFIRM_RECORDED", "PRESENT_TXN", "auto", "SUCCESS and next transaction pending", ("R6", "R12"), ("W8",),
     "pdf.2D", "Call record_charge_response for the current transaction before advancing."),
    ("CONFIRM_RECORDED", "CONFIRM_ACTION", "auto", "UNRECOGNIZED with card intent and an eligible option",
     ("R15", "R16"), ("W9",), "pdf.2E", "If an eligible action is returned, confirm the requested action"),
    ("CONFIRM_RECORDED", "DISCLOSE_FEE", "auto", "expedited delivery requested and eligible", ("R26",), ("W9",),
     "pdf.5", "Before selecting expedited delivery, the bot must tell the customer the $15 fee"),
    ("CONFIRM_RECORDED", "INFORM_NO_CARD_ACTION", "auto", "get_card_action_options returned NONE", ("R15",),
     ("W13", "W14"), "pdf.4", "Offer only returned options."),
    ("CONFIRM_RECORDED", "EXPLAIN_LIMIT", "auto", "more reviewable transactions than max_transactions_per_session",
     ("R31",), (), "pdf.5", "The bot may review up to 4 recent transactions in one self-service session."),
    ("CONFIRM_RECORDED", "SUMMARIZE", "auto", "no transaction pending", ("R19",), ("W10",), "pdf.2F",
     "After the available charges are reviewed, summarize the responses"),
    ("CONFIRM_RECORDED", "TRANSFER", "auto", "get_card_action_options UNAVAILABLE", ("R32",), ("W20",), "pdf.7",
     "If a required tool fails, transfer to a human."),
    ("DISCLOSE_FEE", "CONFIRM_ACTION", "auto", "fee disclosed; ask for clear confirmation", ("R26",), (), "pdf.5",
     "obtain a clear confirmation"),
    ("CONFIRM_ACTION", "ACTION_DONE", "CONFIRM", "submit_card_action SUCCESS", ("R16", "R17"), ("W12",), "pdf.2E",
     "call submit_card_action, and describe it as completed only after SUCCESS"),
    ("CONFIRM_ACTION", "CONFIRM_ACTION", "REQUEST_CARD_ACTION", "customer switches the delivery option",
     ("R16", "R26"), (), "pdf.2E", "confirm the requested action"),
    ("CONFIRM_ACTION", "ACK_DECLINED", "DENY", "customer declines the action", ("R16", "R18"), (), "pdf.2E",
     "This second intent must not erase the charge-review intent."),
    ("CONFIRM_ACTION", "TRANSFER", "CONFIRM", "submit_card_action not SUCCESS", ("R29",), ("W17",), "pdf.7",
     "If a required tool fails, transfer to a human."),
    ("CONFIRM_ACTION", "SUMMARIZE", "CONFIRM", "submit_card_action not SUCCESS and responses recorded",
     ("R19", "R29"), ("W17",), "pdf.7", "Do not claim an action completed unless its API returns SUCCESS."),
    ("CONFIRM_ACTION", "TRANSFER", "REQUEST_HUMAN", "customer asks for a human", ("R20",), (), "pdf.2F",
     "If the customer requests a human at any point, transfer the interaction."),
    ("ACTION_DONE", "PRESENT_TXN", "auto", "SUCCESS and next transaction pending", ("R18",), ("W15",), "pdf.2E",
     "This second intent must not erase the charge-review intent."),
    ("ACTION_DONE", "SUMMARIZE", "auto", "SUCCESS and no transaction pending", ("R19",), ("W16",), "pdf.2F",
     "summarize the responses and any account action that actually completed"),
    ("ACK_DECLINED", "PRESENT_TXN", "auto", "next transaction pending", ("R18",), (), "pdf.2E",
     "This second intent must not erase the charge-review intent."),
    ("ACK_DECLINED", "SUMMARIZE", "auto", "no transaction pending", ("R19",), (), "pdf.2F",
     "After the available charges are reviewed, summarize the responses"),
    ("INFORM_NO_CARD_ACTION", "PRESENT_TXN", "auto", "next transaction pending", ("R18",), ("W13",), "pdf.2E",
     "This second intent must not erase the charge-review intent."),
    ("INFORM_NO_CARD_ACTION", "SUMMARIZE", "auto", "no transaction pending", ("R19",), ("W14",), "pdf.2F",
     "After the available charges are reviewed, summarize the responses"),
    ("EXPLAIN_LIMIT", "SUMMARIZE", "auto", "limit explained; summarize, then transfer the rest", ("R31", "R19"),
     (), "pdf.5", "The bot may review up to 4 recent transactions in one self-service session."),
    ("EXPLAIN_NO_TXNS", "ASK_MORE_HELP", "auto", "no charges to review", ("R8",), ("W6",), "pdf.2B",
     "ask whether the customer needs other help"),
    ("SUMMARIZE", "ASK_MORE_HELP", "auto", "journey complete", ("R19",), ("W18",), "pdf.7",
     "Summarize only recorded responses and account actions that actually completed; then end."),
    ("SUMMARIZE", "TRANSFER", "auto", "a transfer is pending", ("R20",), (), "pdf.2F",
     "transfer the interaction"),
    ("ASK_MORE_HELP", "END", "END", "customer needs nothing else", ("R19",), ("W18",), "pdf.7", "then end."),
    ("ASK_MORE_HELP", "TRANSFER", "REQUEST_HUMAN", "customer asks for a human", ("R20",), (), "pdf.2F",
     "If the customer requests a human at any point, transfer the interaction."),
    ("TRANSFER", "END", "auto", "transfer_to_human TRANSFERRED", ("R20",), ("W19",), "pdf.4",
     "End automated servicing."),
]
# the turn limit can end any prompt state (R28)
TURN_LIMIT_FROM = ("ASK_AUTH", "ASK_AUTH_RETRY", "PRESENT_TXN", "ASK_CLARIFY", "PROVIDE_CONTEXT", "CONFIRM_ACTION",
                   "ASK_MORE_HELP")

# escalation trigger -> states it can fire in
TRIGGER_STATES = {
    "auth_failed": ("ASK_AUTH_RETRY",), "auth_unavailable": ("ASK_AUTH", "ASK_AUTH_RETRY"),
    "retrieve_unavailable": ("ASK_AUTH", "ASK_AUTH_RETRY"),
    "record_failed": ("PRESENT_TXN", "ASK_CLARIFY", "PROVIDE_CONTEXT"),
    "unsure_twice": ("PROVIDE_CONTEXT",), "ambiguity_persists": ("ASK_AUTH", "ASK_CLARIFY"),
    "submit_failed": ("CONFIRM_ACTION",), "customer_request": TURN_LIMIT_FROM,
    "turn_limit": TURN_LIMIT_FROM, "transaction_limit": ("CONFIRM_RECORDED",),
    "card_options_unavailable": ("CONFIRM_RECORDED",),
}

# where each workflow edge is stated in the PDF
WORKFLOW_SOURCES = {
    "W1": ("pdf.2A", "Call authenticate_customer."),
    "W2": ("program", "at most 2 times incorrect authentication"),
    "W3": ("pdf.2B", "After successful authentication, call get_review_transactions."),
    "W4": ("pdf.2A", "If authentication fails or the tool is unavailable, transfer to a human"),
    "W5": ("pdf.4", "After customer gives a clear"),
    "W6": ("pdf.2B", "If the tool returns no transactions, explain that there are no charges available for review"),
    "W7": ("pdf.7", "If a required tool fails, transfer to a human."),
    "W8": ("pdf.2D", "Call record_charge_response for the current transaction before advancing."),
    "W9": ("pdf.2E", "If the customer reports an unrecognized charge and also asks to secure/replace the card, call"),
    "W10": ("pdf.2F", "After the available charges are reviewed, summarize the responses"),
    "W11": ("pdf.4", "Advance only after SUCCESS;"),
    "W12": ("pdf.2E", "If an eligible action is returned, confirm the requested action, call"),
    "W13": ("pdf.4", "Offer only returned options."),
    "W14": ("pdf.4", "Offer only returned options."),
    "W15": ("pdf.2E", "This second intent must not erase the charge-review intent."),
    "W16": ("pdf.2F", "summarize the responses and any account action that actually completed"),
    "W17": ("pdf.7", "If a required tool fails, transfer to a human."),
    "W18": ("pdf.7", "Summarize only recorded responses and account actions that actually completed; then end."),
    "W19": ("pdf.4", "End automated servicing."),
    "W20": ("pdf.7", "If a required tool fails, transfer to a human."),
    "W21": ("pdf.2F", "If the customer requests a human at any point, transfer the interaction."),
}
VALUE_SET_SOURCES = {"disposition": ("pdf.2C", "A clear answer maps to RECOGNIZED or"),
                     "card_action": ("pdf.2E", "also asks to secure/replace the card"),
                     "delivery": ("pdf.5", "Expedited replacement delivery is available"),
                     "tool_result": ("pdf.2D", "unless the API returns SUCCESS")}

# relations the GMS is trained on, beyond those the policy transcription already compiles
GMS_STATE_RELATIONS = ("in_stage", "accepts_act", "leads_to", "applies_rule", "limited_by", "calls_tool",
                       "interprets_yes_as", "interprets_no_as", "fires_in_state", "has_slot")


# =========================================================================== PDF text
_SEC_HEAD = re.compile(r"^(\d)\.\s+(Purpose|Operating Procedure|Voice and|Illustrative Tools|Small Knowledge|"
                       r"Required vs|Illustrative Journey)")
_SUB_HEAD = re.compile(r"^([A-F])\.\s+\S")


def _norm(s: str) -> str:
    s = re.sub(r"[\x7f-•●▪]", " ", s)        # bullet glyphs
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def pdf_text(pdf_path: str | Path) -> str:
    """Layout-preserving text (the tools table keeps its columns)."""
    try:
        import pypdf
        return "\n".join(p.extract_text(extraction_mode="layout") or ""
                         for p in pypdf.PdfReader(str(pdf_path)).pages)
    except ImportError:
        import subprocess
        return subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, text=True,
                              check=True).stdout


def _table_rows(lines: list[str]) -> list[str]:
    """The tools table as one string per row: 'tool | when used | example output | required behaviour'."""
    head = next(i for i, ln in enumerate(lines) if "Tool / API" in ln and "Required bot behavior" in ln)
    h = lines[head]
    cols = [h.index(x) for x in ("Tool / API", "When used", "Example output", "Required bot behavior")]
    rows: list[list[list[str]]] = []
    for ln in lines[head + 1:]:
        if not ln.strip():
            continue
        cells = [ln[a:b].strip() if b else ln[a:].strip() for a, b in zip(cols, cols[1:] + [None])]
        # a cell may start a few characters left of its header: take the leftover of the previous slice
        if cells[0] and re.fullmatch(r"[a-z_]+", cells[0].split()[0]) and len(cells[0].split()) == 1:
            rows.append([[c] if c else [] for c in cells])
        elif rows:
            for i, c in enumerate(cells):
                if c:
                    rows[-1][i].append(c)
    return [" | ".join(" ".join(c) for c in r) for r in rows]


def pdf_sections(text: str) -> dict[str, str]:
    """Split the procedure into sections pdf.1 ... pdf.7 (section 2 into pdf.2A-2F).

    Sentences inside a section are separated by ' ¶ ' where the layout breaks them
    (bullets, table rows), so every evidence quote is one verbatim unit."""
    raw: dict[str, list[str]] = defaultdict(list)
    cur = "pdf.0"
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("Fictional POC Operating Procedure —") or re.fullmatch(r"Page \d+", st):
            continue
        if st.startswith("Fictional POC Operating Procedure") and cur == "pdf.0":
            continue
        m = _SEC_HEAD.match(st)
        if m:
            cur = f"pdf.{m.group(1)}"
            continue
        if cur.startswith("pdf.2"):
            sm = _SUB_HEAD.match(st)
            if sm and len(st) < 60:
                cur = f"pdf.2{sm.group(1)}"
                continue
        if st.startswith("Transcription note"):
            cur = "pdf.note"
        raw[cur].append(line)
    out = {}
    for k, lines in raw.items():
        if k == "pdf.0":
            continue
        if k == "pdf.4":
            out[k] = " ¶ ".join(_norm(r) for r in _table_rows(lines))
            continue
        # a blank line, a bullet or a numbered step starts a new unit
        units, buf = [], []
        for ln in lines:
            st = ln.strip()
            new_unit = (not st) or re.match(r"^([\x7f-•]|\d+[A-C]?\.\s)", st)
            if new_unit and buf:
                units.append(_norm(" ".join(buf)))
                buf = []
            if st:
                buf.append(st)
        if buf:
            units.append(_norm(" ".join(buf)))
        out[k] = " ¶ ".join(u for u in units if u)
    return out


def _sentences(text: str) -> list[str]:
    out = []
    for unit in text.split(" ¶ "):
        out += [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", unit) if p.strip()]
    return out


_STOP = set("the a an and or of to in on for is be by it its at as if any only this that from with not "
            "before after bot customer must".split())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_$]+", s.lower()) if w not in _STOP}


def best_evidence(section_text: str, query: str) -> str:
    """The sentence of the section that shares the most content words with `query`."""
    q = _words(query)
    best, score = "", -1.0
    for s in _sentences(section_text):
        w = _words(s)
        sc = len(q & w) / (len(q | w) ** 0.5 or 1)
        if sc > score:
            best, score = s, sc
    return best


# =========================================================================== build
def build_kg(transcription: str | Path = TRANSCRIPTION, pdf_path: str | Path | None = None) -> dict[str, Any]:
    """Build the JSON graph from the PDF (evidence) and its structured transcription (facts)."""
    from .ingest import _parse_table, _split_sections, ingest_markdown

    md_text = Path(transcription).read_text()
    ing = ingest_markdown(transcription)
    _, sec = _split_sections(md_text)
    if pdf_path is None:
        cand = REPO_ROOT / "fictional_poc_operating_procedure_gpt_readable.pdf"
        pdf_path = cand if cand.exists() else None
    if pdf_path is None:
        raise FileNotFoundError("the operating-procedure PDF is needed to attach evidence")
    raw = pdf_text(pdf_path)
    sections = pdf_sections(raw)
    sections["program"] = PROGRAM_BRIEF
    sections["lexicon"] = LEXICON_NOTE

    triples: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def ev(source: str, quote: str | None, query: str | None = None) -> str:
        text = sections.get(source, "")
        if quote and _norm(quote) in text:
            return _norm(quote)
        return best_evidence(text, query or quote or "") if text else ""

    def add(h, r, t, kind, source, *, evidence=None, query=None, state=None, qualifiers=None, gms=False):
        tail_type = ("number" if isinstance(t, (int, float)) or re.fullmatch(r"-?\d+(\.\d+)?", str(t))
                     else "text" if len(str(t)) > 40 or " " in str(t) and kind not in ("vocabulary",) else "entity")
        if kind == "vocabulary" and r == "has_alias":
            tail_type = "alias"
        key = (kind, h, r, str(t), json.dumps(qualifiers or {}, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        rec = {"id": f"T{len(triples) + 1:04d}", "head": h, "relation": r, "tail": t, "tail_type": tail_type,
               "kind": kind}
        if state:
            rec["state"] = state
        rec["source"] = source
        rec["evidence"] = ev(source, evidence, query)
        if qualifiers:
            rec["qualifiers"] = qualifiers
        rec["gms"] = gms
        triples.append(rec)

    # ---- policy: limits
    for row in _parse_table(sec["limits"]):
        k, src = row["key"], row["source"]
        add(k, "is_a", "limit", "policy", src, query=row["text"])
        add(k, "has_value", int(row["value"]), "policy", src, query=row["text"], qualifiers={"unit": row["unit"]},
            gms=True)
        add(k, "has_unit", row["unit"], "policy", src, query=row["text"])
        add(k, "has_text", row["text"], "policy", src, query=row["text"])
    # ---- policy: delivery options
    for row in _parse_table(sec["replacement delivery options"]):
        o, src = row["option"], row["source"]
        q = f"{o.split('_')[0]} replacement delivery fee business days"
        add(o, "is_a", "delivery_option", "policy", src, query=q)
        add("replace_card", "delivered_by", o, "policy", src, query=q, gms=True)
        add(o, "has_fee_usd", int(row["fee_usd"]), "policy", src, query=f"{o.split('_')[0]} fee $", gms=True)
        add(o, "has_business_days_min", int(row["business_days_min"]), "policy", src,
            query=f"{o.split('_')[0]} arrives business days", gms=True)
        add(o, "has_business_days_max", int(row["business_days_max"]), "policy", src,
            query=f"{o.split('_')[0]} arrives business days", gms=True)
        add(o, "requires_fee_confirmation", row["requires_fee_confirmation"], "policy", src,
            query="tell the customer the $15 fee and obtain a clear confirmation", gms=True)
    # ---- policy: tools
    for row in _parse_table(sec["tools"]):
        tl = row["tool"]
        q = f"{tl} {row['when_used']}"
        add(tl, "is_a", "tool", "policy", "pdf.4", query=tl)
        add(tl, "has_stage", row["stage"], "policy", "pdf.4", query=tl, gms=True)
        add(tl, "when_used", row["when_used"], "policy", "pdf.4", query=q)
        add(tl, "success_value", row["success_value"], "policy", "pdf.4", query=q, gms=True)
        for fv in row["failure_values"].split(";"):
            add(tl, "failure_value", fv.strip(), "policy", "pdf.4", query=q, gms=True)
        add(tl, "on_failure", row["on_failure"], "policy", "pdf.4", query=q, gms=True)
    # ---- trajectory: tool workflow
    for w in ing.workflow:
        src, quote = WORKFLOW_SOURCES.get(w.wid, ("pdf.7", None))
        add(w.src, "enables", w.dst, "trajectory", src, evidence=quote, query=w.condition,
            qualifiers={"edge": w.wid, "condition": w.condition}, gms=True)
    # ---- policy: rules
    rule_src, rule_ev = {}, {}
    for r in ing.rules:
        src = next((row["source"] for row in _parse_table(sec["rules"]) if row["rule_id"] == r.rule_id), "pdf")
        rule_src[r.rule_id] = src
        rule_ev[r.rule_id] = ev(src, RULE_EVIDENCE.get(r.rule_id), r.text)
        e = rule_ev[r.rule_id]
        add(r.rule_id, "is_a", "rule", "policy", src, evidence=e)
        add(r.rule_id, "has_stance", r.stance, "policy", src, evidence=e, gms=True)
        add(r.rule_id, "constrains", r.subject, "policy", src, evidence=e, gms=True)
        add(r.subject, r.predicate, r.obj, "policy", src, evidence=e,
            qualifiers={"rule": r.rule_id, "stance": r.stance}, gms=True)
        add(r.rule_id, "has_text", r.text, "policy", src, evidence=e)
    # ---- vocabulary: value sets + aliases
    for row in _parse_table(sec["dispositions and other value sets"]):
        vs, val = row["value_set"], row["value"]
        src, quote = VALUE_SET_SOURCES.get(vs, ("pdf", None))
        add(val, "member_of", vs, "vocabulary", src, evidence=quote, query=val, gms=True)
        for a in [x.strip() for x in row["aliases"].split(";") if x.strip()]:
            add(val, "has_alias", a, "vocabulary", "lexicon", evidence=LEXICON_NOTE,
                qualifiers={"value_set": vs}, gms=True)
    for vs, vals in EXTRA_VALUE_SETS.items():
        for v in vals:
            add(v, "member_of", vs, "vocabulary", "pdf.3",
                query="General policy questions may be answered from the knowledge base.")
    # ---- vocabulary: dialog acts and their slots
    for act, slots in ACT_SCHEMA.items():
        src = ACT_SOURCE[act]
        add(act, "is_a", "dialog_act", "vocabulary", src, evidence=ACT_EVIDENCE[act])
        for slot, (vs, req) in slots.items():
            add(act, "has_slot", slot, "vocabulary", src, evidence=ACT_EVIDENCE[act],
                qualifiers={"values_from": vs, "required": req}, gms=True)
    # ---- domain model
    for row in _parse_table(sec["domain model"]):
        add(row["head"], row["relation"], row["tail"], "domain", row["source"],
            query=f"{row['head']} {row['relation']} {row['tail']}".replace("_", " "), gms=True)
    # ---- policy: escalation triggers
    for row in _parse_table(sec["escalation triggers"]):
        tg = row["trigger"]
        rs, re_ = rule_src[row["rule"]], rule_ev[row["rule"]]
        add(tg, "is_a", "escalation_trigger", "policy", rs, evidence=re_)
        add(tg, "justified_by", row["rule"], "policy", rs, evidence=re_, gms=True)
        add(tg, "has_condition", row["condition"], "policy", rs, evidence=re_)
    # ---- policy: knowledge base + required language
    for kid, s in ing.kb_sentences:
        add("knowledge_base", "states", s, "policy", "pdf.5", evidence=s, qualifiers={"kb_id": kid})
    if ing.required_language.get("opening"):
        add("opening_statement", "has_text", ing.required_language["opening"], "policy", "pdf.6",
            evidence=ing.required_language["opening"])
    # ---- state layer
    for sid, st in STATES.items():
        add(sid, "is_a", "dialog_state", "state", st.source, evidence=st.evidence, state=sid)
        add(sid, "in_stage", st.stage, "state", st.source, evidence=st.evidence, state=sid, gms=True)
        add(sid, "prompts_customer", "yes" if st.prompts else "no", "state", st.source, evidence=st.evidence,
            state=sid)
        if st.prompts:
            for act in tuple(st.accepts) + GLOBAL_ACTS:
                src = ACT_SOURCE[act] if act in GLOBAL_ACTS else st.source
                add(sid, "accepts_act", act, "state", src,
                    evidence=ACT_EVIDENCE[act] if act in GLOBAL_ACTS else st.evidence, state=sid, gms=True)
            add(sid, "interprets_yes_as", st.yes, "state", st.source, evidence=st.evidence, state=sid, gms=True)
            add(sid, "interprets_no_as", st.no, "state", st.source, evidence=st.evidence, state=sid, gms=True)
        for rid in st.rules:
            add(sid, "applies_rule", rid, "state", rule_src[rid], evidence=rule_ev[rid], state=sid, gms=True)
        for lim in st.limits:
            row = next(x for x in _parse_table(sec["limits"]) if x["key"] == lim)
            add(sid, "limited_by", lim, "state", row["source"], query=row["text"], state=sid, gms=True)
        for tl in st.tools:
            add(sid, "calls_tool", tl, "state", "pdf.4", query=tl, state=sid, gms=True)
    for a, b, on, when, rules, edges, src, quote in TRANSITIONS:
        add(a, "leads_to", b, "trajectory", src, evidence=quote, query=when, state=a,
            qualifiers={"on": on, "when": when, "rules": list(rules), "edges": list(edges)}, gms=True)
    for sid in TURN_LIMIT_FROM:
        add(sid, "leads_to", "TRANSFER", "trajectory", "program", evidence="at max 20 turns", state=sid,
            qualifiers={"on": "any", "when": "max_turns reached before the journey completes", "rules": ["R28"],
                        "edges": []}, gms=True)
    for tg, sts in TRIGGER_STATES.items():
        for sid in sts:
            add(tg, "fires_in_state", sid, "state", STATES[sid].source, evidence=STATES[sid].evidence, state=sid,
                gms=True)

    kg = {
        "schema": SCHEMA_ID,
        "title": ing.title,
        "source": {
            "document": Path(pdf_path).name,
            "sha256": hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest(),
            "transcription": Path(transcription).name,
            "sections": dict(sorted(sections.items())),
        },
        "relations": _relation_docs(),
        "triples": triples,
    }
    kg["sha256"] = _content_hash(kg)
    return kg


def _relation_docs() -> dict[str, str]:
    return {
        "enables": "workflow node -> next workflow node (qualifiers: edge, condition)",
        "leads_to": "dialog state -> next dialog state (qualifiers: on = customer act or auto, when, rules, edges)",
        "in_stage": "dialog state -> journey stage",
        "accepts_act": "dialog state -> customer meaning card (act) admissible in that state",
        "interprets_yes_as": "dialog state -> what a bare 'yes' means there",
        "interprets_no_as": "dialog state -> what a bare 'no' means there",
        "applies_rule": "dialog state -> policy rule that governs it",
        "limited_by": "dialog state -> exact limit that applies there",
        "calls_tool": "dialog state -> tool the bot may call next",
        "fires_in_state": "escalation trigger -> dialog state where it can fire",
        "prompts_customer": "dialog state -> yes/no: does the customer answer in this state",
        "has_slot": "dialog act -> slot (qualifiers: values_from, required)",
        "member_of": "canonical value -> value set", "has_alias": "canonical value -> customer synonym",
        "has_value": "limit -> exact value (qualifier: unit)",
        "has_stance": "rule -> must / must_not / may", "constrains": "rule -> subject",
        "justified_by": "escalation trigger -> rule",
    }


def _content_hash(kg: dict) -> str:
    body = {k: v for k, v in kg.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


# =========================================================================== load + read
class PolicyKG:
    """Read access to the JSON graph; every later step goes through this class."""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self.data = data
        self.path = path
        self.triples: list[dict[str, Any]] = data["triples"]
        self.sha256: str = data.get("sha256") or _content_hash(data)
        self._by: dict[str, list[dict]] = defaultdict(list)
        for t in self.triples:
            self._by[t["relation"]].append(t)

    # ------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | Path | None = None) -> "PolicyKG":
        if path is None:                         # the active domain pack's graph
            from ..domain import current
            path = current().graph_path
        p = Path(path)
        kg = cls(json.loads(p.read_text(encoding="utf-8")), p)
        errs = kg.validate()
        if errs:
            raise ValueError(f"{p.name}: " + "; ".join(errs[:8]))
        return kg

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return p

    # ------------------------------------------------------------- generic
    def where(self, relation: str | None = None, head: str | None = None, tail: Any = None,
              kind: str | None = None) -> list[dict]:
        rows = self._by.get(relation, []) if relation else self.triples
        return [t for t in rows if (head is None or t["head"] == head) and (tail is None or t["tail"] == tail)
                and (kind is None or t["kind"] == kind)]

    def tails(self, head: str, relation: str) -> list:
        return [t["tail"] for t in self.where(relation, head=head)]

    def hrt(self, gms_only: bool = False) -> list[tuple[str, str, str]]:
        out = []
        for t in self.triples:
            if gms_only and not t.get("gms"):
                continue
            tr = (t["head"], t["relation"], str(t["tail"]))
            if tr not in out:
                out.append(tr)
        return out

    # ------------------------------------------------------------- states
    @property
    def states(self) -> list[str]:
        return [t["head"] for t in self.where("is_a", tail="dialog_state")]

    @property
    def prompt_states(self) -> list[str]:
        return [t["head"] for t in self.where("prompts_customer", tail="yes")]

    def stage(self, state: str) -> str | None:
        v = self.tails(state, "in_stage")
        return v[0] if v else None

    def accepts(self, state: str | None, act: str) -> bool:
        if state is None or state not in self.prompt_states:
            return True                                    # outside the dialog-state machine: no restriction
        return act in self.tails(state, "accepts_act")

    def accepted_acts(self, state: str) -> list[str]:
        return self.tails(state, "accepts_act")

    def interprets(self, state: str | None) -> tuple[str | None, str | None]:
        if not state:
            return None, None
        y, n = self.tails(state, "interprets_yes_as"), self.tails(state, "interprets_no_as")
        return (y[0] if y else None), (n[0] if n else None)

    def rules_for(self, state: str) -> list[str]:
        return self.tails(state, "applies_rule")

    def limits_for(self, state: str) -> list[str]:
        return self.tails(state, "limited_by")

    def tools_for(self, state: str) -> list[str]:
        return self.tails(state, "calls_tool")

    def transitions(self, state: str | None = None) -> list[dict]:
        return [dict(src=t["head"], dst=t["tail"], **t.get("qualifiers", {}))
                for t in self.where("leads_to", head=state)]

    def triggers_for(self, state: str) -> list[str]:
        return [t["head"] for t in self.where("fires_in_state", tail=state)]

    def state_triples(self, state: str) -> list[dict]:
        return [t for t in self.triples if t.get("state") == state]

    # ------------------------------------------------------------- vocabulary
    def value_set(self, name: str) -> list[str]:
        return [t["head"] for t in self.where("member_of", tail=name)]

    def aliases(self, value: str) -> list[str]:
        return self.tails(value, "has_alias")

    def slot_schema(self, act: str) -> dict[str, dict]:
        return {t["tail"]: t.get("qualifiers", {}) for t in self.where("has_slot", head=act)}

    def rule_text(self, rule_id: str) -> str:
        v = self.tails(rule_id, "has_text")
        return v[0] if v else ""

    def limit(self, key: str) -> int | float | None:
        v = self.tails(key, "has_value")
        return v[0] if v else None

    # ------------------------------------------------------------- checks
    def validate(self) -> list[str]:
        errs = []
        if self.data.get("schema") != SCHEMA_ID:
            errs.append(f"schema {self.data.get('schema')!r} != {SCHEMA_ID}")
        ids = [t["id"] for t in self.triples]
        if len(ids) != len(set(ids)):
            errs.append("duplicate triple ids")
        for t in self.triples:
            for k in ("head", "relation", "tail", "kind", "source"):
                if k not in t:
                    errs.append(f"{t.get('id')}: missing {k}")
        secs = self.data.get("source", {}).get("sections", {})
        for t in self.triples:
            if t["source"] not in secs:
                errs.append(f"{t['id']}: unknown source {t['source']}")
            elif t.get("evidence") and _norm(t["evidence"]) not in secs[t["source"]]:
                errs.append(f"{t['id']}: evidence not found verbatim in {t['source']}")
        states = set(self.states)
        for tr in self.where("leads_to"):
            for n in (tr["head"], tr["tail"]):
                if n not in states:
                    errs.append(f"{tr['id']}: unknown state {n}")
        acts = {t["head"] for t in self.where("is_a", tail="dialog_act")}
        for tr in self.where("accepts_act"):
            if tr["tail"] not in acts:
                errs.append(f"{tr['id']}: unknown act {tr['tail']}")
        rules = {t["head"] for t in self.where("is_a", tail="rule")}
        for tr in self.where("applies_rule"):
            if tr["tail"] not in rules:
                errs.append(f"{tr['id']}: unknown rule {tr['tail']}")
        reach, todo = {"OPENING"}, ["OPENING"]
        while todo:
            s = todo.pop()
            for tr in self.where("leads_to", head=s):
                if tr["tail"] not in reach:
                    reach.add(tr["tail"])
                    todo.append(tr["tail"])
        if states - reach:
            errs.append(f"unreachable states: {sorted(states - reach)}")
        covered = {t["tail"] for t in self.where("applies_rule")}
        if rules - covered:
            errs.append(f"rules applied in no state: {sorted(rules - covered)}")
        if self.data.get("sha256") and self.data["sha256"] != _content_hash(self.data):
            errs.append("content hash mismatch (file edited without rebuilding its hash)")
        return errs

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = defaultdict(int)
        for t in self.triples:
            kinds[t["kind"]] += 1
        return {"triples": len(self.triples), "by_kind": dict(kinds), "gms_triples": len(self.hrt(gms_only=True)),
                "states": len(self.states), "prompt_states": len(self.prompt_states),
                "transitions": len(self.where("leads_to")), "rules": len(self.where("is_a", tail="rule")),
                "sources": sorted({t["source"] for t in self.triples}), "sha256": self.sha256[:16],
                "document": self.data["source"]["document"]}

    # ------------------------------------------------------------- compile
    def to_markdown(self) -> str:
        """Render the structured policy the compiler ingests (knowledge/ingest.py)."""
        L = []
        row = lambda *c: "| " + " | ".join(str(x) for x in c) + " |"
        L += [f"# {self.data['title']}", "",
              f"Rendered from {self.path.name if self.path else 'the policy graph'} "
              f"(sha256 {self.sha256[:16]}); source document {self.data['source']['document']}.", ""]
        L += ["## 1. Limits", "", row("key", "value", "unit", "source", "text"), row(*["---"] * 5)]
        for t in self.where("is_a", tail="limit"):
            k = t["head"]
            L.append(row(k, self.limit(k), self.tails(k, "has_unit")[0], t["source"], self.tails(k, "has_text")[0]))
        L += ["", "## 2. Replacement delivery options", "",
              row("option", "fee_usd", "business_days_min", "business_days_max", "requires_fee_confirmation",
                  "source"), row(*["---"] * 6)]
        for t in self.where("is_a", tail="delivery_option"):
            o = t["head"]
            L.append(row(o, self.tails(o, "has_fee_usd")[0], self.tails(o, "has_business_days_min")[0],
                         self.tails(o, "has_business_days_max")[0], self.tails(o, "requires_fee_confirmation")[0],
                         t["source"]))
        L += ["", "## 3. Tools", "",
              row("tool", "stage", "when_used", "success_value", "failure_values", "on_failure", "source"),
              row(*["---"] * 7)]
        for t in self.where("is_a", tail="tool"):
            tl = t["head"]
            L.append(row(tl, self.tails(tl, "has_stage")[0], self.tails(tl, "when_used")[0],
                         self.tails(tl, "success_value")[0], ";".join(self.tails(tl, "failure_value")),
                         self.tails(tl, "on_failure")[0], t["source"]))
        L += ["", "## 4. Workflow", ""]
        for t in self.where("enables"):
            q = t["qualifiers"]
            L.append(f"- {q['edge']}: {t['head']} -> {t['tail']} : {q['condition']}")
        L += ["", "## 5. Rules", "", row("rule_id", "stance", "subject", "predicate", "object", "source", "text"),
              row(*["---"] * 7)]
        reified = {t["qualifiers"]["rule"]: t for t in self.triples
                   if t.get("qualifiers", {}).get("rule") and t["kind"] == "policy" and "stance" in t["qualifiers"]}
        for t in self.where("is_a", tail="rule"):
            rid = t["head"]
            f = reified[rid]
            L.append(row(rid, f["qualifiers"]["stance"], f["head"], f["relation"], f["tail"], t["source"],
                         self.rule_text(rid)))
        L += ["", "## 6. Dispositions and other value sets", "", row("value_set", "value", "aliases"),
              row(*["---"] * 3)]
        for t in self.where("member_of"):
            if t["tail"] in EXTRA_VALUE_SETS:
                continue
            L.append(row(t["tail"], t["head"], "; ".join(self.aliases(t["head"]))))
        L += ["", "## 7. Domain model", "", row("head", "relation", "tail", "source"), row(*["---"] * 4)]
        for t in self.triples:
            if t["kind"] == "domain":
                L.append(row(t["head"], t["relation"], t["tail"], t["source"]))
        L += ["", "## 8. Escalation triggers", "", row("trigger", "condition", "rule"), row(*["---"] * 3)]
        for t in self.where("is_a", tail="escalation_trigger"):
            tg = t["head"]
            L.append(row(tg, self.tails(tg, "has_condition")[0], self.tails(tg, "justified_by")[0]))
        L += ["", "## 9. Knowledge base", ""]
        L += [t["tail"] for t in self.where("states", head="knowledge_base")]
        op = self.tails("opening_statement", "has_text")
        L += ["", "## 10. Required language", "", f'Required opening: "{op[0]}"' if op else "", ""]
        return "\n".join(L)

    def entity_types(self) -> dict[str, str]:
        out = {}
        for t in self.where("is_a"):
            out.setdefault(t["head"], t["tail"])
        for t in self.where("in_stage"):
            out.setdefault(t["tail"], "stage")
        for t in self.where("has_slot"):
            out.setdefault(t["tail"], "slot")
        for vs, vals in EXTRA_VALUE_SETS.items():
            for v in vals:
                out.setdefault(v, vs)
        return out

    # ------------------------------------------------------------- consistency
    def consistency(self, store) -> dict[str, Any]:
        """Does the compiled/trained store carry exactly what the graph states?"""
        missing = [tr for tr in self.hrt(gms_only=True) if tr not in store.triples
                   and not (tr[1] == "has_value" and self.limit(tr[0]) is not None)]
        enm = []
        for t in self.where("has_value"):
            got = store.lookup_enm("limit", t["head"])
            enm.append({"key": t["head"], "kg": t["tail"], "store": got, "agree": float(got) == float(t["tail"])})
        for o in [t["head"] for t in self.where("is_a", tail="delivery_option")]:
            for rel, cat in (("has_fee_usd", "delivery_fee_usd"), ("has_business_days_min", "delivery_days_min"),
                             ("has_business_days_max", "delivery_days_max")):
                v = self.tails(o, rel)[0]
                got = store.lookup_enm(cat, o)
                enm.append({"key": f"{cat}/{o}", "kg": v, "store": got, "agree": float(got) == float(v)})
        wf = {(w.wid, w.src, w.dst) for w in store.workflow}
        kgwf = {(t["qualifiers"]["edge"], t["head"], t["tail"]) for t in self.where("enables")}
        rules = set(store.rules) == {t["head"] for t in self.where("is_a", tail="rule")}
        return {"missing_in_store": missing, "enm": enm, "enm_agree": all(e["agree"] for e in enm),
                "workflow_agree": wf == kgwf, "rules_agree": rules,
                "all_agree": not missing and all(e["agree"] for e in enm) and wf == kgwf and rules}


# =========================================================================== source resolution
def resolve_policy(path: str | Path) -> tuple[Path, PolicyKG | None]:
    """A policy path the compiler can read, plus the graph when the source is the JSON.

    The JSON is rendered to the compiler's structured form once, under a name keyed by
    its content hash, so the compiled store (and every cache keyed on it) follows the JSON."""
    p = Path(path)
    if p.suffix.lower() != ".json":
        return p, None
    kg = PolicyKG.load(p)
    from ..backend import CACHE
    out = CACHE / "policy" / f"{p.stem}.{kg.sha256[:16]}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = kg.to_markdown()
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    return out, kg


# =========================================================================== CLI
def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m gmsjourney.knowledge.policy_kg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the policy graph JSON from the PDF and its transcription")
    b.add_argument("--pdf", default=None)
    b.add_argument("--transcription", default=str(TRANSCRIPTION))
    b.add_argument("--out", default=str(KG_PATH))
    c = sub.add_parser("check", help="validate a policy graph JSON")
    c.add_argument("path", nargs="?", default=str(KG_PATH))
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.cmd == "build":
        kg = PolicyKG(build_kg(a.transcription, a.pdf))
        errs = kg.validate()
        kg.save(a.out)
        print(json.dumps(kg.summary(), indent=1))
        print("validation:", "ok" if not errs else errs)
    else:
        kg = PolicyKG.load(a.path)
        print(json.dumps(kg.summary(), indent=1))


if __name__ == "__main__":
    main()
