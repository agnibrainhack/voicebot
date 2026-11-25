"""Block 3 - the scenario simulator: a design row -> synthetic data + an outcome plan -> the correct conversation.

Outcome first: the row fixes what every tool returns at each decision (e.g. 4SSN: INPUT_ERROR, then
PROFILE_SINGLE_MATCH); the simulator derives the customer intent that produces it (malformed digits, then the right
digits) and the walker (graph.Walker) gives the state the bot must be in after each customer turn. So the tools, the
customer's intent and the expected bot state are consistent by construction.

Each turn the customer answers the question of the bot's previous state (its "addressed" input: the tool or metadata
that reads that answer). A tool the customer did not address this turn returns NO_USER_RESPONSE, unless it already
holds a final result (identity results stay, as the real bot's FROZEN results do).

The result is a ``Conversation`` of ``GoldTurn``s: customer intent, tool outputs, expected step (state boxes, hand-off)
and the HRTs that apply (graph.HRTGraph.resolve) - the input to Block 4 (errors) and Block 5 (text).
"""
from __future__ import annotations

import copy
import datetime as dt
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .graph import AOP_KEYS, Context, Flow, HRTGraph, Step, Walker

AOP_NAME = {"UI": "User Identification", "CV": "Charge Verification"}
OPENING = ("Thank you for calling American Express Account Protection. This call may be recorded, monitored, and "
           "analyzed for compliance and to improve our systems. I'm an AI assistant and you can speak to me naturally. "
           "Are you calling in response to a fraud alert on your account?")
HANDOFF_STATE = "HANDOFF"
FINAL_ID = {"PROFILE_SINGLE_MATCH", "PROFILE_MULTI_MATCH", "PROFILE_NO_MATCH", "PROFILE_IDENTIFIED",
            "PROFILE_NOT_IDENTIFIED", "TRANSFER_TO_CCP"}

# The input a bot state asks for: what the customer's next turn answers.
ADDRESSED = [
    (r"UI_CLAUSE_0(76|83)$", "check_fullssn_or_cm15.match"),
    (r"UI_CLAUSE_1(08|11|14)$", "set_4ssn_metadata.match"),
    (r"UI_CLAUSE_119$", "card_confirm"),
    (r"UI_CLAUSE_1(60|63|66)$", "set_fullssn_metadata.match"),
    (r"UI_CLAUSE_171$", "card_confirm"),
    (r"UI_CLAUSE_2(15|17|20|23)$", "set_cm5_metadata.match"),
    (r"UI_CLAUSE_25[036]$", "set_cm15_metadata.match"),
    (r"UI_CLAUSE_28[36]$", "collected_4SSN"),
    (r"UI_CLAUSE_3(07|10)$", "collected_5CM"),
    (r"UI_CLAUSE_3(29|32)$", "collected_zipcode"),
    (r"UI_CLAUSE_5(29|40)$", "check_OTP_channel.match"),
    (r"UI_CLAUSE_", "check_verification.match"),
    (r"CV_", "txn_answer"),
]
CARD_CONFIRM_CONDS = {
    "set_4ssn_metadata": ("the user responds confirming whether or not they are asking about the card you inquired about",
                          "the user confirms they are asking about the card ending in {{ @set_4ssn_metadata.last5AccountNumber }}"),
    "set_fullssn_metadata": ("WAIT_FULL", "the user confirms they're asking about the card ending in {{ @set_fullssn_metadata.last5AccountNumber }}"),
}


def addressed_input(state_boxes: list[str]) -> str | None:
    if not state_boxes:
        return None
    last = state_boxes[-1]
    for rx, key in ADDRESSED:
        if re.search(rx, last):
            return key
    return None


# ---------------------------------------------------------------------------------------------------- synthetic data
FIRST = ["Maria", "James", "Priya", "Chen", "Olivia", "Daniel", "Aisha", "Robert", "Sofia", "Kenji", "Grace", "Luis"]
NICK = {"Maria": "Mari", "James": "Jim", "Priya": "Pri", "Chen": "CJ", "Olivia": "Liv", "Daniel": "Dan",
        "Aisha": "Ash", "Robert": "Bob", "Sofia": "Sofi", "Kenji": "Ken", "Grace": "Gracie", "Luis": "Lou"}
PRODUCTS = ["Platinum Card", "Gold Card", "Green Card", "Blue Cash Preferred Card", "Delta SkyMiles Gold Card",
            "Hilton Honors Card", "Business Platinum Card"]
MERCHANTS = [("Amazon Marketplace", "online retail"), ("Shell Oil 5741", "gas station"), ("Delta Air Lines", "airline"),
             ("Best Buy #1182", "electronics store"), ("Uber Trip", "rideshare"), ("Whole Foods Market", "grocery store"),
             ("Hertz Rent-A-Car", "car rental"), ("Apple.com/Bill", "digital services"), ("Marriott Hotels", "hotel"),
             ("Target T-2214", "department store"), ("DoorDash", "food delivery"), ("Walgreens #4471", "pharmacy")]
CITIES = ["Seattle, WA", "Austin, TX", "Miami, FL", "Chicago, IL", "Denver, CO", "Phoenix, AZ", "Boston, MA"]
FOREIGN = [("EUR", "Paris, France"), ("GBP", "London, UK"), ("MXN", "Cancun, Mexico"), ("JPY", "Tokyo, Japan")]
QUESTIONS = {
    "date_of_birth": "What is your date of birth?",
    "zip_code": "What is the ZIP code on your account?",
    "street_address": "What is the street address on your account?",
    "phone_last4": "What are the last four digits of the phone number on your account?",
}
OTP_QUESTION = "How would you like to receive your one-time passcode: by text message or by email?"


def _digits(rng: random.Random, n: int, lead_zero: bool = False) -> str:
    s = "".join(str(rng.randint(0, 9)) for _ in range(n))
    return ("0" + s[1:]) if lead_zero else (str(rng.randint(1, 9)) + s[1:])


def make_profile(row: dict[str, str], rng: random.Random, today: dt.date) -> dict[str, Any]:
    first = rng.choice(FIRST)
    pref = NICK[first] if row.get("cv_names") == "preferred_differs" else first
    prof = {"first_name": first, "preferred_first_name": pref, "ssn9": _digits(rng, 9), "zip": _digits(rng, 5),
            "dob": f"{rng.choice(['March', 'June', 'October', 'January'])} {rng.randint(1, 28)}, {rng.randint(1955, 2001)}",
            "street": f"{rng.randint(10, 9999)} {rng.choice(['Oak', 'Maple', 'Pine', 'Cedar', 'Elm'])} Street",
            "phone4": _digits(rng, 4), "otp": _digits(rng, 6), "today": today.isoformat()}
    prof["ssn4"] = prof["ssn9"][-4:]
    ncards = int(row["cv_cards"]) if row.get("cv_cards", "n/a") != "n/a" else 1
    nt = int(row["cv_txns_per_card"]) if row.get("cv_txns_per_card", "n/a") != "n/a" else 1
    role = row.get("cv_card_role", "basic_primary")
    cards = []
    for c in range(ncards):
        r_ = role if c == 0 else rng.choice(["basic_primary", "supplementary", "basic_not_primary"])
        rl, basic = {"basic_primary": ("BASIC", True), "supplementary": ("SUPP", False),
                     "basic_not_primary": ("BASIC", False), "n/a": ("BASIC", True)}[r_]
        last5 = _digits(rng, 5, lead_zero=(row.get("cv_leading_zero") == "yes" and c == 0))
        txns = []
        for t in range(nt):
            name, mtype = rng.choice(MERCHANTS)
            status = row.get("cv_status", "declined")
            status = rng.choice(["approved", "declined"]) if status in ("mixed", "n/a") else status
            day = row.get("cv_txn_day", "older")
            day = rng.choice(["today", "yesterday", "older"]) if day in ("mixed", "n/a") else day
            ts = dt.datetime.combine(today, dt.time(rng.randint(6, 22), rng.randint(0, 59)))
            ts -= dt.timedelta(days={"today": 0, "yesterday": 1, "older": rng.randint(2, 20)}[day])
            cur, place = ("USD", rng.choice(CITIES))
            if row.get("cv_currency") == "foreign" and t == 0:
                cur, place = rng.choice(FOREIGN)
            addr = row.get("cv_address_field", "present")
            has_addr = addr == "present" or (addr == "mixed" and t % 2 == 0)
            tx = {"merchantName": name, "merchantType": mtype,
                  "transactionAmount": f"{rng.randint(4, 1900)}.{rng.randint(0, 99):02d}",
                  "transactionCurrency": cur, "transactionStatus": status,
                  "transactionTimestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"), "day": day,
                  "relativeTimeDifference": f"{rng.randint(1, 9)} hours" if day == "today" else None}
            if row.get("cv_mode_field", "present") == "present":
                tx["transactionMode"] = rng.choice(["online", "in person", "contactless", "phone order"])
            if has_addr:
                tx["merchantAddress"] = f"{rng.randint(10, 999)} Main St, {place}"
            txns.append({k: v for k, v in tx.items() if v is not None})
        cards.append({"productName": rng.choice(PRODUCTS), "last5AccountNumber": last5, "role": rl, "isBasic": basic,
                      "transactions": txns})
    prof["cards"] = cards
    prof["cm5"] = cards[0]["last5AccountNumber"]
    prof["cm15"] = _digits(rng, 10) + prof["cm5"]
    q = row.get("ui_question_type", "date_of_birth")
    prof["question"] = QUESTIONS.get(q, QUESTIONS["date_of_birth"])
    prof["question_type"] = q if q in QUESTIONS else "date_of_birth"
    prof["question2"] = QUESTIONS["zip_code" if prof["question_type"] != "zip_code" else "date_of_birth"]
    return prof


# ---------------------------------------------------------------------------------------------------- the plan
def _id_queue(final: str, retry: str) -> list[str]:
    pre = {"none": [], "INPUT_ERROR": ["INPUT_ERROR"], "ATTEMPT_ERROR": ["ATTEMPT_ERROR"],
           "NO_USER_RESPONSE": ["NO_USER_RESPONSE"], "INPUT_then_ATTEMPT": ["INPUT_ERROR", "ATTEMPT_ERROR"],
           "n/a": []}[retry]
    return pre + [final]


def make_plan(row: dict[str, str], rng: random.Random | None = None) -> dict[str, Any]:
    """Queues of outcomes per addressed input, from the design row."""
    q: dict[str, list[Any]] = {}
    p: dict[str, Any] = {"queues": q}
    if row["aop"] == "UI":
        p["ani"] = row["ui_ani"]
        retry = row.get("ui_id_retry", "none")
        first = ("4ssn" if row["ui_ani"] == "PROFILE_MATCH" else
                 "full" if row.get("ui_id_choice") in ("FULL_SSN", "NO_PREFERENCE_SHARED") else
                 "cm15" if row.get("ui_id_choice") == "CM_15" else None)
        if row.get("ui_4ssn_result", "n/a") != "n/a":
            q["set_4ssn_metadata.match"] = _id_queue(row["ui_4ssn_result"], retry if first == "4ssn" else "none")
        if row.get("ui_id_choice", "n/a") != "n/a":
            c = row["ui_id_choice"]
            q["check_fullssn_or_cm15.match"] = ["NO_PREFERENCE_SHARED", "FULL_SSN"] if c == "NO_PREFERENCE_SHARED" else [c]
        if row.get("ui_fullssn_result", "n/a") != "n/a":
            q["set_fullssn_metadata.match"] = _id_queue(row["ui_fullssn_result"], retry if first == "full" else "none")
        if row.get("ui_card_confirm", "n/a") != "n/a":
            q["card_confirm"] = [row["ui_card_confirm"] == "confirms"]
        p["confirm_identification"] = row.get("ui_confirm_identification", "PROFILE_IDENTIFIED")
        cm5 = row.get("ui_cm5", "n/a")
        if cm5 != "n/a":
            q["set_cm5_metadata.match"] = {"SEARCHING_then_IDENTIFIED": ["SEARCHING", "PROFILE_IDENTIFIED"],
                                           "INPUT_ERROR_then_IDENTIFIED": ["INPUT_ERROR", "PROFILE_IDENTIFIED"],
                                           "ATTEMPT_ERROR_then_IDENTIFIED": ["ATTEMPT_ERROR", "PROFILE_IDENTIFIED"]
                                           }.get(cm5, [cm5])
        if row.get("ui_cm15", "n/a") != "n/a":
            q["set_cm15_metadata.match"] = _id_queue(row["ui_cm15"], retry if first == "cm15" else "none")
        k14 = row.get("ui_14key", "n/a")
        if k14 != "n/a":
            q["collected_4SSN"] = (["INPUT_ERROR"] if k14 == "ssn_INPUT_ERROR_then_IDENTIFIED" else []) + ["COLLECTED"]
            q["collected_5CM"] = (["INPUT_ERROR"] if k14 == "cm5_INPUT_ERROR_then_IDENTIFIED" else []) + ["COLLECTED"]
            q["collected_zipcode"] = (["INPUT_ERROR"] if k14 == "zip_INPUT_ERROR_then_IDENTIFIED" else []) + ["COLLECTED"]
            p["k14_final"] = {"PROFILE_NOT_IDENTIFIED": "PROFILE_NOT_IDENTIFIED", "TRANSFER_TO_CCP": "TRANSFER_TO_CCP"
                              }.get(k14, "PROFILE_IDENTIFIED")
        v = row.get("ui_verification", "pass_first")
        p["ask_response"] = {"PASSED_AT_ASK": "VERIFICATION_PASSED", "TRANSFER_AT_ASK": "TRANSFER_TO_CCP"}.get(v, "ASK_QUESTION")
        PASS = ("VALIDATE", "VERIFICATION_PASSED")
        direct = (rng or random.Random(0)).random() < 0.5     # the tool answers itself, or through VALIDATE
        q["check_verification.match"] = {
            "pass_first": [PASS], "repeat_then_pass": [("repeatQuestion", None) if direct else ("VALIDATE", "repeatQuestion"), PASS],
            "next_question_then_pass": [("nextQuestion", None) if direct else ("VALIDATE", "nextQuestion"), PASS],
            "input_error_then_pass": [("INPUT_USER_ERROR", None), PASS],
            "no_response_then_pass": [("NO_USER_RESPONSE", None), PASS],
            "otp_sms": [("nextQuestion-OTP", None), PASS], "otp_email": [("nextQuestion-OTP", None), PASS],
            "otp_no_preference_then_sms": [("nextQuestion-OTP", None), PASS],
            "otp_unknown_transfer": [("nextQuestion-OTP", None)],
            "fails_transfer": [("VALIDATE", "TRANSFER_TO_CCP")]}.get(v, [PASS])
        q["check_OTP_channel.match"] = {"otp_sms": ["SMS"], "otp_email": ["EMAIL"],
                                        "otp_no_preference_then_sms": ["NO_PREFERENCE_SHARED", "SMS"],
                                        "otp_unknown_transfer": ["UNKNOWN"]}.get(v, [])
    else:
        p["fetch"] = row["cv_fetch"]
        ans = row.get("cv_answers", "all_attempted")
        p["answers"] = ans
        p["disclosure_retry"] = row.get("cv_disclosure_retry") == "yes"
        p["next_route"] = row.get("cv_next_txn_route", "via_next_action")
        p["final"] = row.get("cv_final", "ALL_CHARGES_VERIFIED")
        p["declined_non_fraud"] = row.get("cv_declined_non_fraud") == "yes"
    return p


# ---------------------------------------------------------------------------------------------------- the simulator
@dataclass
class GoldTurn:
    index: int
    customer: dict[str, Any]          # intent: addressed input, outcome, act, flags, interjection, correction
    tools: dict[str, Any]             # tool outputs observed on this turn
    step: dict[str, Any]              # expected state: boxes, outcome, hand-off, sections, path
    context: dict[str, Any]           # flags, current card / transaction, metadata snapshot
    hrts: dict[str, Any]              # resolved branches / policies / atoms (as dicts)

    @property
    def state(self) -> str:
        return self.step["state"]


@dataclass
class Conversation:
    conv_id: str
    row: dict[str, str]
    profile: dict[str, Any]
    aop: str
    turns: list[GoldTurn] = field(default_factory=list)
    end: str = ""
    notes: list[str] = field(default_factory=list)


class _TurnOracle:
    """graph.TurnInput for one simulated turn (lazy: a tool's output is decided when the walker asks for it)."""

    def __init__(self, sim: "Simulator", addressed: str | None, outcome: Any, flags: set[str]):
        self.sim, self.addressed, self.outcome, self.flags = sim, addressed, outcome, flags
        self.seen: dict[str, Any] = {}

    def tool(self, name, fld):
        key = f"{name}.{fld}" if fld else name
        if key not in self.seen:
            self.seen[key] = self.sim._tool(name, fld, self)
        return self.seen[key]

    def meta(self, key):
        return self.sim._meta(key, self)

    def semantic(self, condition):
        return self.sim._semantic(condition, self)


class Simulator:
    def __init__(self, flow: Flow, hrt: HRTGraph, row: dict[str, str], conv_id: str, seed: int,
                 today: dt.date | None = None, max_turns: int = 40, include_handoff_turns: bool = True):
        self.flow, self.hrt, self.row, self.conv_id = flow, hrt, row, conv_id
        self.rng = random.Random(seed)
        self.aop = AOP_NAME[row["aop"]]
        self.profile = make_profile(row, self.rng, today or dt.date(2026, 11, 20))
        self.plan = make_plan(row, self.rng)
        self.queues = {k: list(v) for k, v in self.plan["queues"].items()}
        self.final: dict[str, Any] = {}          # sticky final results per tool key
        self.metadata: dict[str, Any] = {"journey": "Fraud", "channel_type": "voice",
                                         "first_name": self.profile["first_name"],
                                         "preferred_first_name": self.profile["preferred_first_name"]}
        if self.aop == "User Identification":
            self.metadata["checkpoint"] = "checkANI"
        self.walker = Walker(flow, self.aop, self.metadata)
        self.max_turns, self.include_handoff = max_turns, include_handoff_turns
        # CV pointers
        self.card_i, self.txn_i = 0, 0
        self.readout_started = False
        self.disclosure_pending = bool(self.plan.get("disclosure_retry"))
        self.answer_turns = 0
        self.id_requests = 0
        self.last_failed_identifier: str | None = None

    # ------------------------------------------------------------------ tool and metadata values
    def _card(self) -> dict[str, Any]:
        return self.profile["cards"][min(self.card_i, len(self.profile["cards"]) - 1)]

    def _txn(self) -> dict[str, Any]:
        c = self._card()
        return c["transactions"][min(self.txn_i, len(c["transactions"]) - 1)]

    def _input(self, key: str, o: _TurnOracle) -> Any:
        if o.addressed == key and o.outcome is not None:
            return o.outcome
        return self.final.get(key, "NO_USER_RESPONSE")

    def _tool(self, name: str, fld: str | None, o: _TurnOracle) -> Any:
        key = f"{name}.{fld}" if fld else name
        p = self.plan
        if name == "initialize_session":
            return p.get("ani")
        if key in ("set_4ssn_metadata.match", "set_fullssn_metadata.match", "set_cm15_metadata.match",
                   "check_fullssn_or_cm15.match", "check_OTP_channel.match"):
            v = self._input(key, o)
            if v in FINAL_ID or v in ("FULL_SSN", "CM_15", "UNKNOWN", "SMS", "EMAIL"):
                self.final[key] = v
            return v
        if key in ("set_4ssn_metadata.last5AccountNumber", "set_fullssn_metadata.last5AccountNumber"):
            return self.profile["cm5"]
        if key == "set_cm5_metadata.match":
            v = self._input(key, o)
            if v == "SEARCHING":
                return "NO_USER_RESPONSE"
            if v in FINAL_ID:
                self.final[key] = v
            return v
        if key == "check_cm5.result":
            return "SEARCHING" if (o.addressed == "set_cm5_metadata.match" and o.outcome == "SEARCHING") else "OK"
        if name == "confirm_identification":
            return p.get("confirm_identification", "PROFILE_IDENTIFIED")
        if key == "set_14key_metadata.match":
            if o.addressed == "collected_zipcode" and o.outcome == "COLLECTED":
                self.final[key] = p.get("k14_final", "PROFILE_IDENTIFIED")
            return self.final.get(key, "NO_USER_RESPONSE")
        if name == "set_ask_verification_metadata":
            if fld == "response":
                return p.get("ask_response", "ASK_QUESTION")
            if fld in ("question", "questionText"):
                return self.metadata.get("questionText", self.profile["question"])
        if key == "check_verification.match":
            v = self._input(key, o)
            v = v[0] if isinstance(v, tuple) else v
            if v == "nextQuestion-OTP":
                self.final[key] = v                       # holds until the OTP itself is answered
                self.metadata["questionText"] = OTP_QUESTION
            elif o.addressed == key:
                self.final.pop(key, None)
            if v == "nextQuestion":
                self.metadata["questionText"] = self.profile["question2"]
            return v
        if key == "process_check_verification.match":
            v = o.outcome if o.addressed == "check_verification.match" else None
            return v[1] if isinstance(v, tuple) else "NO_USER_RESPONSE"
        if name == "fetch_fraud_charges":
            return p.get("fetch")
        if key == "transaction_readout.started":
            return "True" if self.readout_started else "False"
        if name == "check_transaction_disclosure":
            if o.addressed == "txn_answer" and self.disclosure_pending:
                self.disclosure_pending = False
                return False
            return True
        if key == "update_transaction_verification_status.action":
            if o.addressed != "txn_answer":
                return "TRANSACTION_NOT_VERIFIED"
            v = o.outcome
            if v == "SEND_TO_AMEX" and self.plan.get("next_route") == "read_next_transaction" and self._more_on_card():
                self._advance_txn()
                return "READ_NEXT_TRANSACTION"
            return v
        if key == "update_concerns_get_next_action.action":
            if self._more_on_card():
                self._advance_txn()
                return "VERIFY_ADDITIONAL_TRANSACTIONS"
            if self.card_i + 1 < len(self.profile["cards"]):
                self.card_i, self.txn_i = self.card_i + 1, 0
                return "VERIFY_NEXT_CARD"
            f = p.get("final", "ALL_CHARGES_VERIFIED")
            return f if f in ("ALL_CHARGES_VERIFIED", "REPLACE_CARD", "TRANSFER_TO_CCP") else "TRANSFER_TO_CCP"
        if key == "set_update_concern_get_next_action_metadata.reason":
            f = p.get("final", "")
            return f if f not in ("ALL_CHARGES_VERIFIED", "REPLACE_CARD", "TRANSFER_TO_CCP") else "other"
        return None

    def _meta(self, key: str, o: _TurnOracle) -> Any:
        if key in ("collected_4SSN", "collected_5CM", "collected_zipcode"):
            v = self._input(key, o)
            if v == "COLLECTED":
                self.final[key] = v
            return v
        if key == "_4ssn_api_response":
            return self.final.get("set_4ssn_metadata.match")
        if key == "full_ssn_api_response":
            return self.final.get("set_fullssn_metadata.match")
        if key == "_5cm_api_response":
            return self.final.get("set_cm5_metadata.match")
        if key == "_15cm_api_response":
            return self.final.get("set_cm15_metadata.match")
        if key == "num_transactions":
            return len(self._card()["transactions"])
        if key == "current_transaction_index":
            return self.txn_i + 1
        if key == "has_declined_non_fraud":
            return True if self.plan.get("declined_non_fraud") else None
        if key == "questionText":
            return self.metadata.get("questionText", self.profile["question"])
        return None

    def _semantic(self, cond: str, o: _TurnOracle) -> bool | None:
        if o.addressed != "card_confirm":
            if "responds confirming" in cond or cond.startswith("the user indicates they are asking"):
                return False
            return None
        if "responds confirming" in cond or cond.startswith("the user indicates they are asking"):
            return o.outcome is not None
        if "confirms they" in cond:
            return bool(o.outcome)
        return None

    def _more_on_card(self) -> bool:
        return self.txn_i + 1 < len(self._card()["transactions"])

    def _advance_txn(self) -> None:
        self.txn_i += 1

    # ------------------------------------------------------------------ the conversation
    def _context(self, flags: set[str]) -> Context:
        txn = self._txn() if self.aop == "Charge Verification" else {}
        return Context(metadata=dict(self.metadata), flags=set(flags), txn=txn, card=self._card(),
                       txn_index=self.txn_i + 1, card_txn_count=len(self._card()["transactions"]),
                       txn_day=txn.get("day", "older"))

    def _handoff_hrts(self, handoff: str, flags: set[str]) -> dict[str, Any]:
        """A hand-off has no box in this AOP: the response only has to say the caller is being transferred and must
        not carry on with identification or charge review."""
        atoms = [{"ref": "handoff/C1", "box": HANDOFF_STATE, "branch": "A", "kind": "convey", "head": "Transfer notice",
                  "tail": "tell the caller they are being connected to a specialist / agent",
                  "description": "Say the caller is being transferred", "group": "AND"},
                 {"ref": "handoff/N1", "box": HANDOFF_STATE, "branch": "A", "kind": "forbid", "head": "Flow continued",
                  "tail": "ask for identifiers or review charges", "description": "Do not continue the flow",
                  "group": "AND"}]
        if "Card Replacement" in handoff:
            atoms[0]["tail"] = "move on to replacing the card"
        if "end_call" in handoff:
            atoms[0]["tail"] = "say the call will end"
        return {"boxes": [HANDOFF_STATE], "branches": [f"{HANDOFF_STATE}/A"], "policies": [], "atoms": atoms,
                "handoff": handoff}

    def plan_customer(self, addressed: str | None, turn_index: int) -> tuple[Any, dict[str, Any]]:
        """The outcome this customer turn produces for its addressed input, and the intent behind it."""
        if addressed == "txn_answer":
            return self._txn_outcome()
        if addressed and self.queues.get(addressed):
            return self.queues[addressed].pop(0), {}
        if addressed in ("set_4ssn_metadata.match", "set_fullssn_metadata.match", "set_cm15_metadata.match",
                         "set_cm5_metadata.match", "collected_4SSN", "collected_5CM", "collected_zipcode"):
            return ("COLLECTED" if addressed.startswith("collected") else "PROFILE_IDENTIFIED"), {"note": "unplanned"}
        if addressed == "check_verification.match":
            return ("VALIDATE", "VERIFICATION_PASSED"), {"note": "unplanned"}
        return None, {}

    def _txn_outcome(self) -> tuple[Any, dict[str, Any]]:
        a = self.plan.get("answers", "all_attempted")
        self.answer_turns += 1
        key = (self.card_i, self.txn_i)
        seen = getattr(self, "_txn_seen", {})
        n = seen.get(key, 0)
        seen[key] = n + 1
        self._txn_seen = seen
        first_txn = key == (0, 0)
        if a == "unmatched_transfer" and first_txn:
            return "TRANSFER_TO_CCP", {"attempted": None, "act": "talks_about_unknown_transaction"}
        if a == "unsure_then_answer" and first_txn and n == 0:
            return "PROVIDE_MORE_INFO", {"attempted": None, "act": "unsure"}
        if a == "non_answer_then_answer" and first_txn and n == 0:
            return "TRANSACTION_NOT_VERIFIED", {"attempted": None, "act": "non_answer"}
        attempted = {"all_attempted": True, "none_attempted": False}.get(a, (self.card_i + self.txn_i) % 2 == 0)
        return "SEND_TO_AMEX", {"attempted": attempted, "act": "answers"}

    def id_flags(self, step: Step, prev_failed: bool) -> set[str]:
        """first / later identifier request, and switching identifier after a failure (global policies 036/037)."""
        f: set[str] = set()
        asks_id = any(re.search(r"UI_CLAUSE_(076|083|108|111|114|160|163|166|217|220|223|250|253|256|283|286|307|310|329|332)$", b)
                      for b in step.speech)
        if asks_id:
            self.id_requests += 1
            f.add("first_id_request" if self.id_requests == 1 else "later_id_request")
            if prev_failed and any(re.search(r"UI_CLAUSE_(160|217|250|283|307|329)$", b) for b in step.speech):
                f.add("identifier_switch_after_fail")
        return f

    def walk(self, addressed: str | None, outcome: Any, flags: set[str]) -> tuple[Step, dict[str, Any]]:
        o = _TurnOracle(self, addressed, outcome, flags)
        if "asks_human" in flags:
            st = Step(self.aop, [], [], "SWITCH Connect to Agent", ["global/ROUTING_OVERRIDE"], [],
                      "SWITCH Connect to Agent")
            return st, o.seen
        st = self.walker.run(o)
        if st.speech and self.aop == "Charge Verification" and any(
                re.search(r"CV_CLAUSE_03[6-9]|CV_CLAUSE_04[0-4]|CV_LOOP_DISCLOSE", b) for b in st.speech):
            self.readout_started = True
        return st, o.seen

    def run(self, interjection: str = "none", interjection_turn: int | None = None,
            correction: str = "none", correction_turn: int | None = None) -> Conversation:
        conv = Conversation(self.conv_id, self.row, self.profile, self.aop)
        prev_state: list[str] = []
        self.snaps: list[dict[str, Any]] = []
        for k in range(self.max_turns):
            flags: set[str] = {"idv_complete"} if self.aop == "Charge Verification" else set()
            if k == 0:
                addressed = "opening" if self.aop == "User Identification" else "check_verification.match"
                outcome, intent = (None, {"act": "answers_opening"}) if addressed == "opening" else \
                    (("VALIDATE", "VERIFICATION_PASSED"), {"act": "answers_verification", "correct": True})
            else:
                addressed = addressed_input(prev_state)
                outcome, intent = self.plan_customer(addressed, k)
            inter = interjection if interjection_turn == k else "none"
            if inter != "none":
                flags.add(inter if inter != "multi_demand" else "off_topic")
                if inter not in ("multi_demand", "asks_human"):
                    # the customer does not answer this turn: put the planned outcome back for the next turn
                    if addressed in self.queues and outcome is not None and addressed != "txn_answer":
                        self.queues[addressed].insert(0, outcome)
                    outcome = "TRANSACTION_NOT_VERIFIED" if addressed == "txn_answer" else None
                    if inter == "next_txn_request":
                        flags.add("next_txn_request")
                    if addressed == "txn_answer":
                        intent = {"act": "non_answer"}
            corr = correction if correction_turn == k else "none"
            self.snaps.append(self.state())
            if self.aop == "User Identification" and k == 0:
                st = self.walker.run(_TurnOracle(self, None, None, flags))
                tools = {}
            else:
                st, tools = self.walk(addressed, outcome, flags)
            failed_now = str(outcome) in ("PROFILE_NO_MATCH", "PROFILE_NOT_IDENTIFIED")
            flags |= self.id_flags(st, failed_now)
            if st.handoff and "SWITCH Charge Verification" in st.handoff and not st.speech:
                conv.end = st.handoff                     # the next words belong to the Charge Verification AOP
                break
            customer = {"addressed": addressed, "outcome": outcome, "intent": intent, "interjection": inter,
                        "correction": corr}
            if st.speech:
                hr = self.hrt.resolve(self.aop, st.speech, self._context(flags))
                hrts = {**{k2: v for k2, v in hr.items() if k2 != "atoms"}, "atoms": [a.as_dict() for a in hr["atoms"]]}
                state = st.state
            elif st.handoff or "escalate" in st.outcome or "end_call" in st.outcome:
                if not self.include_handoff:
                    conv.end = st.outcome
                    break
                hrts = self._handoff_hrts(st.outcome, flags)
                state = HANDOFF_STATE
            else:
                conv.end = st.outcome
                conv.notes.append(f"turn {k}: no speech ({st.outcome})")
                break
            ctx = self._context(flags)
            conv.turns.append(GoldTurn(k, customer, {**tools}, {
                "state": state, "boxes": list(st.speech), "outcome": st.outcome, "handoff": st.handoff,
                "sections": st.sections, "path": st.path},
                {"flags": sorted(flags), "card": {k2: v for k2, v in ctx.card.items() if k2 != "transactions"},
                 "txn": ctx.txn, "txn_index": ctx.txn_index, "card_index": self.card_i + 1,
                 "metadata": {k2: v for k2, v in self.metadata.items() if isinstance(v, (str, int, bool))}}, hrts))
            if st.handoff or state == HANDOFF_STATE:
                conv.end = st.outcome
                break
            prev_state = st.speech
        else:
            conv.end = "max_turns"
        return conv

    # ------------------------------------------------------------------ counterfactual turns (Block 4)
    _STATE = ("queues", "final", "metadata", "card_i", "txn_i", "readout_started", "disclosure_pending",
              "answer_turns", "id_requests", "_txn_seen")

    def state(self) -> dict[str, Any]:
        return copy.deepcopy({k: getattr(self, k) for k in self._STATE if hasattr(self, k)})

    def restore(self, st: dict[str, Any]) -> None:
        for k, v in copy.deepcopy(st).items():
            setattr(self, k, v)
        self.walker.md = self.metadata

    def counterfactual(self, k: int, addressed: str | None, outcome: Any, flags: set[str]) -> dict[str, Any] | None:
        """Turn k walked again with another outcome / flags: the state the bot would be in, and its HRTs."""
        if k >= len(self.snaps):
            return None
        cur = self.state()
        self.restore(self.snaps[k])
        try:
            st, _tools = self.walk(addressed, outcome, set(flags))
            if st.speech:
                hr = self.hrt.resolve(self.aop, st.speech, self._context(flags))
                res = {"state": st.state, "boxes": list(st.speech), "outcome": st.outcome,
                       "hrts": {**{x: y for x, y in hr.items() if x != "atoms"}, "atoms": [a.as_dict() for a in hr["atoms"]]}}
            elif st.handoff or "escalate" in st.outcome or "end_call" in st.outcome:
                res = {"state": HANDOFF_STATE, "boxes": [HANDOFF_STATE], "outcome": st.outcome,
                       "hrts": self._handoff_hrts(st.outcome, flags)}
            else:
                res = None
        except Exception:                                   # noqa: BLE001 - an impossible counterfactual
            res = None
        finally:
            self.restore(cur)
        return res


def simulate(flow: Flow, hrt: HRTGraph, row: dict[str, str], conv_id: str, seed: int, **kw) -> tuple[Conversation, "Simulator"]:
    """The correct conversation for a design row (interjection / correction placed by the row's position factors)."""
    sim = Simulator(flow, hrt, row, conv_id, seed, **kw)
    probe = copy.deepcopy(sim).run()                       # length of the plain conversation
    n = max(1, len(probe.turns))
    pos = {"early": 0.2, "middle": 0.5, "late": 0.8}
    inter = row.get("interjection", "none")
    it = None
    if inter != "none":
        it = max(1, min(n - 1, round(n * pos.get(row.get("interjection_position", "middle"), 0.5))))
    corr = row.get("correction", "none")
    ct = None
    if corr != "none":
        ct = max(1, min(n - 1, round(n * 0.5) + (1 if it is not None and round(n * 0.5) == it else 0)))
    conv = sim.run(inter, it, corr, ct)
    return conv, sim
