"""Journey content: what a test case *is*, before any presentation factor.

``JourneySpec`` holds the content dimensions of one base case. Every dimension
here changes the correct behaviour of the bot, which is the book's test for
"base, not factor" (Ch. 1.2): a knob that would move the answer belongs to the
base. Tool unavailability, ambiguity that persists, a customer correction that
flips the answer and the 20-turn limit all live here for that reason, even
though a generic catalog files some of them as presentation factors.

``build_fixture`` turns a spec into a deterministic backend fixture.
``UserSimulator`` answers each bot act with the next semantic frame the spec
dictates. It is reactive, so a test does not break when a bot phrases or orders
its prompts differently; it only needs to know what the bot is asking for.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .ontology import SemanticFrame as F

MERCHANTS = [("Amazon", "45.67"), ("Starbucks", "6.25"), ("Shell", "52.10"),
             ("Netflix", "15.99"), ("Uber", "23.40"), ("Target", "88.05")]
REVIEW_AGES = [3, 8, 15, 22, 27]
CREDENTIAL = "4821"

# The content dimensions and their levels (N/A marks "irrelevant in this case").
DIMENSIONS: dict[str, list[Any]] = {
    "auth": ["ok", "retry_ok", "fail_twice", "unavailable", "human_at_auth"],
    "retrieve": ["ok", "unavailable"],
    "n_txn": [0, 1, 2, 4, 5],
    "window": ["none", "day30", "day31"],
    "disposition": ["all_recognized", "mixed", "unsure_resolved", "unsure_twice",
                    "ambiguous_resolved", "ambiguous_twice", "correction"],
    "record": ["ok", "fail_first"],
    "card_intent": ["none", "secure", "replace_standard", "replace_expedited"],
    "card_options": ["eligible", "none", "unavailable"],
    "expedited_confirm": ["confirm", "switch_standard"],
    "submit": ["ok", "fail"],
    "human_mid": ["no", "yes"],
    "kb_mid": ["none", "ask"],
    "turn_pressure": ["normal", "at_limit", "over_limit"],
}
NA = "N/A"


@dataclass(frozen=True)
class JourneySpec:
    auth: str = "ok"
    retrieve: str = "ok"
    n_txn: int = 1
    window: str = "none"
    disposition: str = "all_recognized"
    record: str = "ok"
    card_intent: str = "none"
    card_options: str = "eligible"
    expedited_confirm: str = "confirm"
    submit: str = "ok"
    human_mid: str = "no"
    kb_mid: str = "none"
    turn_pressure: str = "normal"
    # derived/attached content (not covering dimensions)
    kb_frames: tuple[F, ...] = ()            # KB questions asked before the first answer
    extra_kb: int = 0                        # padding questions used to hit a turn target

    def key(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in ("kb_frames",)}
        d["kb_frames"] = [f.canonical() for f in self.kb_frames]
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def dims(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in DIMENSIONS}

    def with_(self, **kw: Any) -> "JourneySpec":
        return replace(self, **kw)


def relevant_dims(spec: JourneySpec) -> dict[str, Any]:
    """The dimensions that actually influence ground truth for this spec (others are N/A)."""
    d = spec.dims()
    out: dict[str, Any] = {"auth": d["auth"]}
    if d["auth"] in ("fail_twice", "unavailable", "human_at_auth"):
        return {**{k: NA for k in DIMENSIONS}, **out}
    out["retrieve"] = d["retrieve"]
    if d["retrieve"] == "unavailable":
        return {**{k: NA for k in DIMENSIONS}, **out}
    out["n_txn"], out["window"] = d["n_txn"], d["window"]
    if d["n_txn"] == 0:
        return {**{k: NA for k in DIMENSIONS}, **out}
    out["disposition"], out["record"] = d["disposition"], d["record"]
    out["kb_mid"], out["turn_pressure"] = d["kb_mid"], d["turn_pressure"]
    early_stop = d["record"] == "fail_first" or d["disposition"] in ("unsure_twice", "ambiguous_twice")
    out["human_mid"] = d["human_mid"] if (not early_stop or d["n_txn"] == 1) else NA
    if early_stop:
        out["card_intent"] = out["card_options"] = out["expedited_confirm"] = out["submit"] = NA
        return {**{k: NA for k in DIMENSIONS}, **out}
    out["card_intent"] = d["card_intent"]
    if d["card_intent"] == "none":
        out["card_options"] = out["expedited_confirm"] = out["submit"] = NA
    else:
        out["card_options"] = d["card_options"]
        eligible = d["card_options"] == "eligible"
        out["expedited_confirm"] = d["expedited_confirm"] if eligible and d["card_intent"] == "replace_expedited" else NA
        out["submit"] = d["submit"] if eligible else NA
    return {**{k: NA for k in DIMENSIONS}, **out}


def is_valid(spec: JourneySpec) -> bool:
    """Static constraints between content dimensions."""
    if spec.n_txn == 0 and spec.window == "day30":
        return False                       # day30 would make a transaction reviewable
    if spec.card_intent != "none" and spec.disposition not in ("mixed", "ambiguous_resolved"):
        return False                       # card intent rides on an UNRECOGNIZED report (pdf 2E)
    if spec.human_mid == "yes" and spec.n_txn == 0:
        return False
    return True


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------
def build_fixture(spec: JourneySpec) -> dict[str, Any]:
    auth_results = {"ok": ["AUTHENTICATED"], "retry_ok": ["FAILED", "AUTHENTICATED"],
                    "fail_twice": ["FAILED", "FAILED", "AUTHENTICATED"],
                    "unavailable": ["UNAVAILABLE"], "human_at_auth": ["AUTHENTICATED"]}[spec.auth]
    txns = []
    for i in range(spec.n_txn):
        m, amt = MERCHANTS[i]
        age = REVIEW_AGES[i]
        if spec.window == "day30" and i == spec.n_txn - 1:
            age = 30
        txns.append({"transaction_id": f"T{i + 1}", "merchant": m, "amount": amt, "age_days": age})
    if spec.window == "day31":
        m, amt = MERCHANTS[5]
        old = {"transaction_id": "T9", "merchant": m, "amount": amt, "age_days": 31}
        txns.insert(min(1, len(txns)), old)       # placed among reviewable ones on purpose
    fx: dict[str, Any] = {
        "credential": CREDENTIAL, "auth_results": auth_results, "transactions": txns,
        "card_options": ["SECURE_CARD", "REPLACE_CARD"] if spec.card_options == "eligible" else [],
        "submit_result": "SUCCESS" if spec.submit == "ok" else "FAILED",
    }
    if spec.retrieve == "unavailable":
        fx["transactions_status"] = "UNAVAILABLE"
    if spec.record == "fail_first":
        fx["record_fail_on_call"] = 1
    if spec.card_options == "unavailable":
        fx["card_options_status"] = "UNAVAILABLE"
    return fx


def reviewable_ids(fixture: dict[str, Any], lookback: int) -> list[str]:
    return [t["transaction_id"] for t in fixture["transactions"] if t["age_days"] <= lookback]


# --------------------------------------------------------------------------
# User simulator
# --------------------------------------------------------------------------
def _card_slots(spec: JourneySpec) -> dict[str, Any]:
    return {"secure": {"card_action": "SECURE_CARD"},
            "replace_standard": {"card_action": "REPLACE_CARD", "delivery": "standard_delivery"},
            "replace_expedited": {"card_action": "REPLACE_CARD", "delivery": "expedited_delivery"},
            }.get(spec.card_intent, {})


def disposition_script(spec: JourneySpec, index: int) -> list[F]:
    """Frames the customer gives for the index-th presented transaction."""
    D = lambda d, **kw: F.make("INFORM_DISPOSITION", disposition=d, **kw)
    if index > 0:
        return [D("RECOGNIZED")]
    card = _card_slots(spec)
    p = spec.disposition
    if p == "all_recognized":
        return [D("RECOGNIZED")]
    if p == "mixed":
        return [D("UNRECOGNIZED", **card)]
    if p == "unsure_resolved":
        return [D("UNSURE"), D("RECOGNIZED")]
    if p == "unsure_twice":
        return [D("UNSURE"), D("UNSURE")]
    if p == "ambiguous_resolved":
        return [F.make("UNCLEAR"), D("UNRECOGNIZED", **card)]
    if p == "ambiguous_twice":
        return [F.make("UNCLEAR"), F.make("UNCLEAR")]
    if p == "correction":
        return [D("RECOGNIZED", corrected_from="UNRECOGNIZED")]
    raise ValueError(p)


@dataclass
class UserSimulator:
    spec: JourneySpec
    fixture: dict[str, Any]
    lookback: int = 30
    queues: dict[str, list[F]] = field(default_factory=dict)
    switched: bool = False
    emitted: list[F] = field(default_factory=list)
    _last_disp: dict[str, F] = field(default_factory=dict)
    _human_used: bool = False

    def __post_init__(self) -> None:
        ids = reviewable_ids(self.fixture, self.lookback)
        human_at = min(1, len(ids) - 1) if (self.spec.human_mid == "yes" and ids) else None
        for i, tid in enumerate(ids):
            q: list[F] = []
            if i == 0:
                q += list(self.spec.kb_frames)
            if human_at == i:
                q.append(F.make("REQUEST_HUMAN"))
            q += disposition_script(self.spec, i)
            self.queues[tid] = q

    def _emit(self, f: F) -> F:
        self.emitted.append(f)
        return f

    def respond(self, bot_act: dict[str, Any]) -> F | None:
        act = bot_act.get("act")
        if act in ("ASK_AUTH", "ASK_AUTH_RETRY"):
            if self.spec.auth == "human_at_auth":
                return self._emit(F.make("REQUEST_HUMAN"))
            return self._emit(F.make("PROVIDE_CREDENTIAL", credential=CREDENTIAL))
        if act in ("PRESENT_TXN", "PROVIDE_CONTEXT", "ASK_CLARIFY", "ANSWER_KB"):
            tid = bot_act.get("txn") or bot_act.get("current_txn")
            q = self.queues.get(tid)
            if q is None:                       # a transaction the spec never scripted
                return self._emit(F.make("INFORM_DISPOSITION", disposition="RECOGNIZED"))
            if q:
                f = q.pop(0)
                if f.act == "INFORM_DISPOSITION":
                    self._last_disp[tid] = f
                return self._emit(f)
            # queue exhausted but the bot keeps asking: repeat the latest clear answer
            return self._emit(self._last_disp.get(tid, F.make("UNCLEAR")))
        if act in ("CONFIRM_ACTION", "DISCLOSE_FEE"):
            if (bot_act.get("delivery") == "expedited_delivery" and self.spec.expedited_confirm == "switch_standard"
                    and not self.switched):
                self.switched = True
                return self._emit(F.make("REQUEST_CARD_ACTION", card_action="REPLACE_CARD",
                                         delivery="standard_delivery"))
            return self._emit(F.make("CONFIRM"))
        if act in ("ASK_MORE_HELP", "EXPLAIN_NO_TXNS"):
            return self._emit(F.make("END"))
        return None                              # terminal or unknown: customer stays silent
