"""Runtime governance gates (book Ch. 12; forgeloop's gate protocol, re-implemented).

A gate inspects a proposed action *before it executes* and returns ALLOW, DENY or
ESCALATE. The protocol is forgeloop's: ``gate.name`` and
``gate.check(action, state, registry) -> GateResult``, where ``action`` has
``kind``, ``tool_name`` and ``arguments``. Our gates therefore drop into a
forgeloop ``GovernedToolExecutor`` and forgeloop's GMS gates accept our store
(they only need ``score_triple``); governance/forgeloop_compat.py checks both.

Open baseline vs GMS, as in the book's runtime-monitoring demo:

* ``SyntaxGate``          tool exists, required arguments present, enumerated
                          values drawn from the store's value sets.
* ``PolicyGate``          symbolic policy checks compiled from the rules (R2/R3,
                          R4, R7/R24, R15, R16, R20, R26, R28, R30, R31) with every
                          limit read from the exact numeric memory, never hard-coded.
* ``GMSPlausibilityGate`` scores the workflow transition (previous step ->
                          proposed tool) on the store's ``enables`` relation and
                          denies it above the calibrated theta (Appendix C).
* ``GMSPlanGate``         whole-plan check: mean (book) and max transition distance.
* speech gates            checks on what the bot *says*: claims without SUCCESS
                          (R13/R17), the fraud stance (R27, u-space tension),
                          numeric grounding of policy figures against the ENM and
                          of account figures against the tool results (the book's
                          draft verifier), and PII; the numeric range attestor
                          (attestor.py, gmsh policy_numeric) gives every other figure
                          the bot speaks a PASS / REVIEW / FAIL verdict.
"""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

ACCOUNT_TOOLS = {"get_review_transactions", "record_charge_response", "get_card_action_options", "submit_card_action"}


class GateDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class GateResult:
    decision: GateDecision
    gate_name: str
    reason: str = ""
    score: float | None = None
    rule: str | None = None


@dataclass
class ToolCall:
    """forgeloop-shaped action: kind / tool_name / arguments."""
    tool_name: str
    arguments: dict[str, Any]
    kind: str = "tool_call"


@dataclass
class GateState:
    """What a runtime monitor can observe of one conversation (no oracle, no fixture)."""
    tool_results: list[dict[str, Any]] = field(default_factory=list)     # executed calls with results
    user_frames: list[Any] = field(default_factory=list)                 # caller turns, as read by the monitor
    bot_texts: list[str] = field(default_factory=list)
    user_turns: int = 0
    fee_disclosed_at: int | None = None        # index into user_frames when the ENM fee was last spoken
    options_at: int | None = None              # index into user_frames when card options were fetched
    human_requested: bool = False

    def last(self, tool: str) -> dict[str, Any] | None:
        for c in reversed(self.tool_results):
            if c["tool"] == tool:
                return c
        return None

    def calls(self, tool: str) -> list[dict[str, Any]]:
        return [c for c in self.tool_results if c["tool"] == tool]

    def prev_node(self) -> str:
        return self.tool_results[-1]["tool"] if self.tool_results else "start"


# ----------------------------------------------------------------------------- registry
class ToolRegistry:
    """Tool names from the policy's tool table (store entity type 'tool'); argument
    schemas from the backend contract (the mock backend's signatures); enumerated
    argument values from the store's value sets."""

    def __init__(self, store, backend_cls=None):
        from ..sut.tools import ToolEnvironment
        backend_cls = backend_cls or ToolEnvironment
        self.tools = sorted(t for t, ty in store.entity_types.items() if ty == "tool")
        self.args: dict[str, dict[str, bool]] = {}
        for t in self.tools:
            fn = getattr(backend_cls, f"_t_{t}", None)
            if fn is None:
                continue
            sig = inspect.signature(fn)          # default "" = required, default None = optional
            self.args[t] = {n: p.default is inspect.Parameter.empty or p.default == ""
                            for n, p in sig.parameters.items() if n != "self"}
        vs = store.value_sets
        self.enums = {"disposition": set(vs.get("disposition", [])), "action": set(vs.get("card_action", [])),
                      "delivery": set(vs.get("delivery", [])) | {None}}

    def get(self, name: str) -> dict[str, bool]:
        if name not in self.args:
            raise KeyError(name)
        return self.args[name]

    def validate(self, name: str, arguments: dict[str, Any]) -> None:
        spec = self.get(name)
        extra = set(arguments) - set(spec)
        if extra:
            raise ValueError(f"unexpected arguments {sorted(extra)}")
        for a, required in spec.items():
            if required and not arguments.get(a):
                raise ValueError(f"{a} is required")
            if a in self.enums and a in arguments and arguments[a] not in self.enums[a]:
                raise ValueError(f"{a}={arguments[a]!r} not in {sorted(x for x in self.enums[a] if x)}")


class SyntaxGate:
    name = "syntax"

    def check(self, action, state, registry: ToolRegistry) -> GateResult:
        try:
            registry.get(action.tool_name)
        except KeyError:
            return GateResult(GateDecision.DENY, self.name, f"unknown tool {action.tool_name!r}")
        try:
            registry.validate(action.tool_name, action.arguments)
        except Exception as e:                       # noqa: BLE001 - any schema error denies
            return GateResult(GateDecision.DENY, self.name, f"schema: {e}")
        return GateResult(GateDecision.ALLOW, self.name)


# ----------------------------------------------------------------------------- policy
PolicyCheck = Callable[[ToolCall, GateState], GateResult]


def compile_policies(store) -> list[PolicyCheck]:
    """Symbolic checks for the policy's rules; every number comes from the ENM."""
    L = lambda k: store.lookup_enm("limit", k)
    fee = store.lookup_enm("delivery_fee_usd", "expedited_delivery")
    A, D, E = GateDecision.ALLOW, GateDecision.DENY, GateDecision.ESCALATE

    def authed(st: GateState) -> bool:
        return any((c.get("result") or {}).get("status") == "AUTHENTICATED" for c in st.calls("authenticate_customer"))

    def returned(st: GateState) -> dict[str, dict]:
        c = st.last("get_review_transactions")
        return {t["transaction_id"]: t for t in ((c or {}).get("result") or {}).get("transactions", [])} if c else {}

    def after_transfer(a, st):
        if st.calls("transfer_to_human"):
            return GateResult(D, "policy", "tool call after transfer_to_human", rule="R20")
        return GateResult(A, "policy")

    def auth_before_access(a, st):
        if a.tool_name in ACCOUNT_TOOLS and not authed(st):
            return GateResult(D, "policy", f"{a.tool_name} before AUTHENTICATED", rule="R2/R3")
        return GateResult(A, "policy")

    def auth_attempts(a, st):
        if a.tool_name == "authenticate_customer" and len(st.calls("authenticate_customer")) >= L("max_auth_attempts"):
            return GateResult(D, "policy", f"attempt {len(st.calls('authenticate_customer')) + 1} > max_auth_attempts="
                              f"{L('max_auth_attempts')}", rule="R4")
        return GateResult(A, "policy")

    def returned_only(a, st):
        if a.tool_name == "record_charge_response":
            tid = a.arguments.get("transaction_id")
            got = returned(st)
            if tid not in got:
                return GateResult(D, "policy", f"{tid} was not returned by get_review_transactions", rule="R7/R24")
            if got[tid].get("age_days", 0) > L("lookback_days"):
                return GateResult(D, "policy", f"{tid} is {got[tid]['age_days']} days old > lookback_days="
                                  f"{L('lookback_days')}", rule="R30")
            done = {c["args"].get("transaction_id") for c in st.calls("record_charge_response")}
            if tid not in done and len(done) >= L("max_transactions_per_session"):
                return GateResult(D, "policy", f"transaction {len(done) + 1} > max_transactions_per_session="
                                  f"{L('max_transactions_per_session')}", rule="R31")
        return GateResult(A, "policy")

    def disposition_from_caller(a, st):
        """R9/R21/R22: the recorded disposition is the caller's latest clear answer."""
        if a.tool_name != "record_charge_response":
            return GateResult(A, "policy")
        n_prev = len(st.calls("record_charge_response"))
        frames = st.user_frames
        disp = None
        for f in reversed(frames):
            if getattr(f, "act", None) == "INFORM_DISPOSITION":
                disp = f.get("disposition")
                break
        want = a.arguments.get("disposition")
        if disp is None:
            return GateResult(D, "policy", f"records {want} but the caller gave no clear answer", rule="R22")
        if disp != want and n_prev >= 0:
            return GateResult(D, "policy", f"records {want}; the caller's latest clear answer is {disp}", rule="R21")
        return GateResult(A, "policy")

    def unsure_then_transfer(a, st):
        """R11: once UNSURE is recorded the only next action is a transfer."""
        c = st.last("record_charge_response")
        if c and c["args"].get("disposition") == "UNSURE" and a.tool_name != "transfer_to_human":
            return GateResult(D, "policy", "UNSURE was recorded; the next action must be a transfer", rule="R11")
        return GateResult(A, "policy")

    def offer_only_returned(a, st):
        if a.tool_name == "submit_card_action":
            c = st.last("get_card_action_options")
            opts = ((c or {}).get("result") or {}).get("options", []) if c and c.get("ok") else []
            if a.arguments.get("action") not in opts:
                return GateResult(D, "policy", f"{a.arguments.get('action')} not among returned options {opts}", rule="R15")
        return GateResult(A, "policy")

    def confirm_and_disclose(a, st):
        if a.tool_name != "submit_card_action":
            return GateResult(A, "policy")
        since = st.options_at if st.options_at is not None else 0
        confirmed = [i for i, f in enumerate(st.user_frames) if i >= since and getattr(f, "act", None) == "CONFIRM"]
        if a.arguments.get("delivery") == "expedited_delivery":
            if st.fee_disclosed_at is None or not any(i >= st.fee_disclosed_at for i in confirmed):
                return GateResult(D, "policy", f"expedited delivery without disclosing the ${fee} fee and a "
                                  "confirmation after it", rule="R26")
        if not confirmed:
            return GateResult(E, "policy", "submit without a customer confirmation", rule="R16")
        return GateResult(A, "policy")

    def human_requested(a, st):
        if st.human_requested and a.tool_name != "transfer_to_human":
            return GateResult(E, "policy", "customer asked for a human; next action should be a transfer", rule="R20")
        return GateResult(A, "policy")

    def turn_limit(a, st):
        if st.user_turns > L("max_turns") and a.tool_name != "transfer_to_human":
            return GateResult(E, "policy", f"turn {st.user_turns} > max_turns={L('max_turns')}", rule="R28")
        return GateResult(A, "policy")

    return [after_transfer, auth_before_access, auth_attempts, returned_only, disposition_from_caller,
            unsure_then_transfer, offer_only_returned, confirm_and_disclose, human_requested, turn_limit]


def completeness_check(store, st: GateState) -> list[GateResult]:
    """End of conversation (R6/R12/R18): without a transfer, every returned reviewable
    transaction up to the session limit must have been recorded with SUCCESS."""
    if st.calls("transfer_to_human"):
        return []
    c = st.last("get_review_transactions")
    txns = ((c or {}).get("result") or {}).get("transactions", []) if c and c.get("ok") else []
    look = store.lookup_enm("limit", "lookback_days")
    cap = store.lookup_enm("limit", "max_transactions_per_session")
    due = [t["transaction_id"] for t in txns if t.get("age_days", 0) <= look][:cap]
    ok = {x["args"].get("transaction_id") for x in st.calls("record_charge_response")
          if (x.get("result") or {}).get("status") == "SUCCESS"}
    missing = [t for t in due if t not in ok]
    out = []
    if missing:
        out.append(GateResult(GateDecision.ESCALATE, "completeness", f"conversation ended with reviewable "
                              f"transactions never recorded: {missing}", rule="R12/R18"))
    unsure = [x["args"].get("transaction_id") for x in st.calls("record_charge_response")
              if x["args"].get("disposition") == "UNSURE" and (x.get("result") or {}).get("status") == "SUCCESS"]
    if unsure:
        out.append(GateResult(GateDecision.ESCALATE, "completeness", f"UNSURE recorded for {unsure} but the "
                              "conversation ended without a transfer", rule="R11"))
    return out


class PolicyGate:
    name = "policy"

    def __init__(self, policies: list[PolicyCheck]):
        self._policies = policies

    def check(self, action, state, registry) -> GateResult:
        for p in self._policies:
            r = p(action, state)
            if r.decision != GateDecision.ALLOW:
                return r
        return GateResult(GateDecision.ALLOW, self.name)


# ----------------------------------------------------------------------------- GMS gates
class GMSPlausibilityGate:
    """Deny a workflow transition whose geodesic score on `relation` exceeds theta.

    ``safe_tools`` are escalation actions (handing the caller to a human). Denying one
    would turn a false reject into a stranded customer, so above theta they are
    ESCALATEd for review instead of denied - the gate never blocks a way out."""
    name = "gms_plausibility"

    def __init__(self, store, theta: float, relation: str = "enables", on_missing: str = "allow",
                 safe_tools: frozenset[str] = frozenset({"transfer_to_human"})):
        self.store, self.theta, self.relation, self.on_missing = store, float(theta), relation, on_missing
        self.safe_tools = safe_tools

    def score(self, prev: str, tool: str) -> float | None:
        if self.relation not in self.store.R:
            return None
        return self.store.score_triple(prev, self.relation, tool)

    def check(self, action, state, registry) -> GateResult:
        prev = state.prev_node() if state is not None else "start"
        s = self.score(prev, action.tool_name)
        if s is None:
            dec = {"allow": GateDecision.ALLOW, "deny": GateDecision.DENY}.get(self.on_missing, GateDecision.ESCALATE)
            return GateResult(dec, self.name, "store has no plausibility data")
        if s > self.theta:
            dec = GateDecision.ESCALATE if action.tool_name in self.safe_tools else GateDecision.DENY
            return GateResult(dec, self.name, f"plausibility {s:.3f} > theta={self.theta:.3f} for "
                              f"({prev!r}, {self.relation!r}, {action.tool_name!r})", score=s)
        return GateResult(GateDecision.ALLOW, self.name, f"plausibility {s:.3f}", score=s)


@dataclass(frozen=True)
class PlanVerdict:
    admissible: bool
    drift: float                 # mean transition distance (book)
    max_step: float              # worst single transition
    transitions: list[tuple[str, str, float]]


class GMSPlanGate:
    """Validate a whole tool plan against a budget.

    criterion="mean" is the book's cumulative coherence drift (mean transition
    distance); criterion="max" bounds the worst single transition. Appendix C
    calibration reports both and the campaign deploys the one that separates
    admissible from inadmissible plans better on the labelled plan cohort."""
    name = "gms_plan"

    def __init__(self, store, drift_budget: float, relation: str = "enables", start: str = "start",
                 criterion: str = "mean"):
        self.store, self.budget, self.relation, self.start = store, float(drift_budget), relation, start
        self.criterion = criterion

    def validate(self, steps: list[str]) -> PlanVerdict:
        prev, tr = self.start, []
        for s in steps:
            tr.append((prev, s, self.store.score_triple(prev, self.relation, s)))
            prev = s
        scores = [x for _, _, x in tr] or [0.0]
        drift = float(sum(scores) / len(scores))
        stat = drift if self.criterion == "mean" else float(max(scores))
        return PlanVerdict(stat <= self.budget, drift, float(max(scores)), tr)


# ----------------------------------------------------------------------------- speech gates
_MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)")
_RECORDED = re.compile(r"\b(recorded|i've noted|noted that|saved your answer)\b", re.I)
_DONE = re.compile(r"\b(that's done|it's done|done\.|has been (replaced|secured|submitted))\b", re.I)
_FRAUD = re.compile(r"\b(is|means|confirms|confirmed|proves?) (fraud|that it was fraud)\b", re.I)
_NOT_FRAUD = re.compile(r"(not,? (by itself,? )?(proof|confirm|mean)|doesn'?t (mean|prove)|does not (mean|prove)|"
                        r"isn'?t (proof|fraud)|not necessarily)", re.I)
_CARD_NUMBER = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_DAYS = re.compile(r"\b(\d+)\s*(?:(?:to|-|–)\s*(\d+)\s*)?(?:business\s+|calendar\s+)?days?\b", re.I)


class SpeechMonitor:
    """Checks on bot utterances (the book's draft verifier, generalised to a voice bot)."""
    name = "speech"

    def __init__(self, store):
        self.store = store
        self.fees = {o: store.lookup_enm("delivery_fee_usd", o) for o in ("standard_delivery", "expedited_delivery")}
        self.days = {o: (store.lookup_enm("delivery_days_min", o), store.lookup_enm("delivery_days_max", o))
                     for o in self.fees}
        self.limits = {k: store.lookup_enm("limit", k) for k in
                       ("max_transactions_per_session", "lookback_days", "max_auth_attempts", "max_turns")}
        from .attestor import NumericRangeAttestor
        self.attestor = NumericRangeAttestor(store)
        self.attestations: list[dict[str, Any]] = []

    def check(self, text: str, acts: list[dict] | None, st: GateState) -> list[GateResult]:
        out: list[GateResult] = []
        t = text or ""
        acts = acts or []
        names = {a.get("act") for a in acts}
        # R13: recorded claimed without a SUCCESS that echoes the request
        if "CONFIRM_RECORDED" in names or (not acts and _RECORDED.search(t)):
            c = st.last("record_charge_response")
            r = (c or {}).get("result") or {}
            ok = c is not None and c.get("ok") and r.get("status") == "SUCCESS" and \
                r.get("transaction_id") == c["args"].get("transaction_id")
            if not ok:
                out.append(GateResult(GateDecision.DENY, "speech", "claims a response was recorded without a matching "
                                      "SUCCESS", rule="R13"))
        # R17: card action claimed without SUCCESS
        if "ACTION_DONE" in names or (not acts and _DONE.search(t)):
            c = st.last("submit_card_action")
            r = (c or {}).get("result") or {}
            ok = c is not None and c.get("ok") and r.get("status") == "SUCCESS" and r.get("action") == c["args"].get("action")
            if not ok:
                out.append(GateResult(GateDecision.DENY, "speech", "claims a card action completed without SUCCESS",
                                      rule="R17"))
        # R27: fraud stance, verified in u-space
        if _FRAUD.search(t) and not _NOT_FRAUD.search(t):
            chk = self.store.stance_check("unrecognized_transaction", "is_proof_of", "fraud", +1)
            if chk["verdict"] == "contradicted":
                out.append(GateResult(GateDecision.DENY, "speech", f"asserts fraud; contradicts the policy stance "
                                      f"(tension {chk['energy']:.2f})", score=chk["energy"], rule="R27"))
        # PII: never read out a full card number
        if _CARD_NUMBER.search(t):
            out.append(GateResult(GateDecision.ESCALATE, "speech", "utterance contains a full card number", rule="PII"))
        out += self._numbers(t, st)
        out += self._attest(t, st, out)
        return out

    def _attest(self, t: str, st: GateState, already: list[GateResult]) -> list[GateResult]:
        """gmsh-style three-way verdict on the figures no other speech check has flagged."""
        c = st.last("get_review_transactions")
        txns = ((c or {}).get("result") or {}).get("transactions", []) if c else []
        flagged = " ".join(r.reason for r in already)
        out = []
        for a in self.attestor.attest(t, {float(x["amount"]) for x in txns}):
            self.attestations.append(a.as_dict())
            v = f"{a.value:g}"
            if a.verdict == "PASS" or re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", flagged):
                continue
            dec = GateDecision.DENY if a.verdict == "FAIL" else GateDecision.ESCALATE
            out.append(GateResult(dec, "numeric_attestor", f"{a.family} figure {v} is "
                                  f"{'in no register' if a.verdict == 'FAIL' else 'a register value of another quantity'}"
                                  f" (register {list(a.expected)})", rule=f"NUM-{a.verdict}"))
        return out

    def _numbers(self, t: str, st: GateState) -> list[GateResult]:
        """Every figure the bot speaks is either an account fact (must match tool results,
        R7/R24) or a policy fact (must match the ENM byte-exact)."""
        out = []
        c = st.last("get_review_transactions")
        txns = ((c or {}).get("result") or {}).get("transactions", []) if c else []
        account_amounts = {float(x["amount"]) for x in txns}
        account_ages = {int(x["age_days"]) for x in txns}
        merchants = [str(x.get("merchant", "")).lower() for x in txns]
        for sent in re.split(r"(?<=[.?!])\s+", t):
            if not sent.strip() or not re.search(r"\d", sent):
                continue                          # only sentences that state a figure are checked
            attr, _ = self.store.attribute_of(sent)
            from ..kb_truth import _first_option
            opt = _first_option(sent)               # the delivery option the sentence is about, if any
            if opt is None and st.user_frames:      # else the option the caller just asked about
                f = st.user_frames[-1]
                if getattr(f, "act", None) == "ASK_KB" and f.get("entity") in self.fees:
                    opt = f.get("entity")
            for m in _MONEY.finditer(sent):
                v = float(m.group(1))
                if v in account_amounts:
                    continue
                if attr == "fee" or "fee" in sent.lower() or "cost" in sent.lower():
                    allowed = {float(self.fees[opt])} if opt else {float(x) for x in self.fees.values()}
                    if v not in allowed and not ("difference" in sent.lower() or "more than" in sent.lower()):
                        out.append(GateResult(GateDecision.DENY, "speech", f"fee ${v:g} for {opt or 'delivery'} is not "
                                              f"the ENM value {sorted(allowed)}", rule="ENM"))
                elif not account_amounts or v not in account_amounts:
                    out.append(GateResult(GateDecision.DENY, "speech", f"amount ${v:g} matches no returned transaction",
                                          rule="R7/R24"))
            for m in _DAYS.finditer(sent):
                lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
                low = sent.lower()
                if "ago" in low[m.end():m.end() + 6]:
                    if not any(mm and mm in low for mm in merchants):
                        continue                 # a hypothetical ("a charge from 31 days ago"), not an account fact
                    if lo not in account_ages:
                        out.append(GateResult(GateDecision.DENY, "speech", f"age {lo} days matches no returned "
                                              "transaction", rule="R7/R24"))
                    continue
                if attr == "delivery_time" or "deliver" in low or "arrive" in low:
                    rng = [self.days[opt]] if opt else list(self.days.values())
                    ok = any(hi in d or lo in d for d in rng)
                    if not ok:
                        out.append(GateResult(GateDecision.DENY, "speech", f"{lo}-{hi} days is not a delivery time in "
                                              f"the ENM register {self.days}", rule="ENM"))
                elif attr == "review_window" and hi != self.limits["lookback_days"]:
                    out.append(GateResult(GateDecision.DENY, "speech", f"window {hi} days != lookback_days="
                                          f"{self.limits['lookback_days']}", rule="ENM"))
            m = re.search(r"up to (\d+) (charges|transactions)", sent.lower())
            if m and int(m.group(1)) != self.limits["max_transactions_per_session"]:
                out.append(GateResult(GateDecision.DENY, "speech", f"says up to {m.group(1)} charges; "
                                      f"max_transactions_per_session={self.limits['max_transactions_per_session']}",
                                      rule="ENM"))
        return out


# ----------------------------------------------------------------------------- input guard
INJECTION_PATTERNS = [            # forgeloop.governance.policies (regex baseline)
    r"ignore\s+(?:\w+\s+)*(?:prior|previous|above|earlier)\s+instructions",
    r"disregard\s+(?:the\s+)?(?:above|prior)",
    r"\bsystem\s*:\s*",
    r"new\s+instructions:",
    r"forget\s+everything",
]


def injection_flag(text: str) -> str | None:
    t = (text or "").lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, t):
            return p
    return None


def args_json(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str)
