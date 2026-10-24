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
* GMS plausibility    forgeloop's ``GMSPlausibilityGate`` (reuse/governance.py) scores the
                          workflow transition (previous step -> proposed tool) on the
                          knowlytix store's ``has_enables`` relation and denies it above the
                          calibrated theta (Appendix C).
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

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable



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
    """Tool names from the policy's tool table (store entity type 'tool'); argument schemas
    from the backend contract (domain.yaml -> tools.*.args: default "" = required, null =
    optional); enumerated argument values from the store's value sets (tools.*.arg_values)."""

    def __init__(self, store, specs: dict[str, dict] | None = None):
        from ..sut.tools import tool_specs
        specs = specs if specs is not None else tool_specs()
        self.tools = sorted(t for t, ty in store.entity_types.items() if ty == "tool")
        self.args: dict[str, dict[str, bool]] = {}
        for t in self.tools:
            if t not in specs:
                continue
            self.args[t] = {n: d == "" for n, d in (specs[t].get("args") or {}).items()}
        vs = store.value_sets
        self.enums: dict[str, set] = {}
        for sp in specs.values():
            for arg, vset in (sp.get("arg_values") or {}).items():
                optional = (sp.get("args") or {}).get(arg, "") is None
                self.enums.setdefault(arg, set()).update(vs.get(vset, []))
                if optional:
                    self.enums[arg].add(None)

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
    """The policy's call checks, compiled from domain.yaml -> rules.calls (governance/rules.py);
    every number comes from the ENM."""
    from .rules import compile_call_checks
    return compile_call_checks(store)


def completeness_check(store, st: GateState) -> list[GateResult]:
    """End of conversation (domain.yaml -> rules.completion): without a handoff, every returned
    item due for review must have been written with success, and no terminal value left behind."""
    from .rules import completion_results
    return completion_results(store, st)


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
_DAYS = re.compile(r"\b(\d+)\s*(?:(?:to|-|–)\s*(\d+)\s*)?(?:business\s+|calendar\s+)?days?\b", re.I)


class SpeechMonitor:
    """Checks on bot utterances (the book's draft verifier, generalised to a voice bot). The
    checks, their patterns and the figures they compare with come from domain.yaml -> rules
    (speech, spoken_figures); every figure is read from exact numeric memory."""
    name = "speech"

    def __init__(self, store):
        from ..domain import current
        from .rules import figure
        self.store = store
        dom = current()
        self.cfg = dom.section("rules")
        sf = self.cfg["spoken_figures"]
        fee_cat = dom.get(f"figures.relations.{sf['options']['relation']}.enm_category")
        opts = list(store.value_sets.get(sf["options"]["over"], []))
        self.option_set = sf["options"]["over"]
        self.fees = {o: store.lookup_enm(fee_cat, o) for o in opts}
        dt = sf["delivery_time"]
        self.days = {o: (store.lookup_enm(dt["min"], o), store.lookup_enm(dt["max"], o)) for o in self.fees}
        self.window = (sf["review_window"]["limit"], figure(store, sf["review_window"]["limit"]))
        self.cap = (re.compile(sf["cap"]["pattern"]), sf["cap"]["limit"], figure(store, sf["cap"]["limit"]))
        self.checks = [{**c, "_re": re.compile(c["pattern"], re.I),
                        "_unless": re.compile(c["unless"], re.I) if c.get("unless") else None}
                       for c in self.cfg.get("speech", [])]
        from .attestor import NumericRangeAttestor
        self.attestor = NumericRangeAttestor(store)
        self.attestations: list[dict[str, Any]] = []

    def check(self, text: str, acts: list[dict] | None, st: GateState) -> list[GateResult]:
        from ..sut.tools import tool_specs
        out: list[GateResult] = []
        t = text or ""
        acts = acts or []
        names = {a.get("act") for a in acts}
        for c in self.checks:
            kind = c["check"]
            if kind == "claim_after_success":            # a claim only after a SUCCESS echoing the request
                if c["act"] in names or (not acts and c["_re"].search(t)):
                    last = st.last(c["tool"])
                    r = (last or {}).get("result") or {}
                    ok_status = (tool_specs()[c["tool"]].get("status") or {}).get("ok")
                    ok = last is not None and last.get("ok") and r.get("status") == ok_status and \
                        r.get(c["echo"]) == last["args"].get(c["echo"])
                    if not ok:
                        out.append(GateResult(GateDecision.DENY, "speech", c["message"], rule=c["rule"]))
            elif kind == "forbidden_stance":             # verified in u-space against the policy stance
                if c["_re"].search(t) and not (c["_unless"] and c["_unless"].search(t)):
                    chk = self.store.stance_check(*c["stance"], +1)
                    if chk["verdict"] == "contradicted":
                        out.append(GateResult(GateDecision.DENY, "speech", f"{c['message']} "
                                              f"(tension {chk['energy']:.2f})", score=chk["energy"], rule=c["rule"]))
            elif kind == "pii":
                if c["_re"].search(t):
                    out.append(GateResult(GateDecision.ESCALATE, "speech", c["message"], rule=c["rule"]))
            else:
                raise ValueError(f"rules.speech: unknown check {kind!r}")
        out += self._numbers(t, st)
        out += self._attest(t, st, out)
        return out

    def _items(self, st: GateState) -> list[dict]:
        it = self.cfg["spoken_figures"]["items"]
        c = st.last(it["source"])
        return ((c or {}).get("result") or {}).get(it["key"], []) if c else []

    def _attest(self, t: str, st: GateState, already: list[GateResult]) -> list[GateResult]:
        """gmsh-style three-way verdict on the figures no other speech check has flagged."""
        it = self.cfg["spoken_figures"]["items"]
        flagged = " ".join(r.reason for r in already)
        out = []
        for a in self.attestor.attest(t, {float(x[it["amount"]]) for x in self._items(st)}):
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
        """Every figure the bot speaks is either an account fact (must match the read tool's
        items) or a policy fact (must match the ENM byte-exact)."""
        sf = self.cfg["spoken_figures"]
        it, fee, dt, rw = sf["items"], sf["fee"], sf["delivery_time"], sf["review_window"]
        out = []
        txns = self._items(st)
        account_amounts = {float(x[it["amount"]]) for x in txns}
        account_ages = {int(x[it["age"]]) for x in txns}
        names = [str(x.get(it["name"], "")).lower() for x in txns]
        noun = it.get("noun", "transaction")
        for sent in re.split(r"(?<=[.?!])\s+", t):
            if not sent.strip() or not re.search(r"\d", sent):
                continue                          # only sentences that state a figure are checked
            attr, _ = self.store.attribute_of(sent)
            from ..kb_truth import _first_option
            opt = _first_option(sent)               # the option the sentence is about, if any
            if opt is None and st.user_frames:      # else the option the caller just asked about
                f = st.user_frames[-1]
                if getattr(f, "act", None) == "ASK_KB" and f.get("entity") in self.fees:
                    opt = f.get("entity")
            low = sent.lower()
            for m in _MONEY.finditer(sent):
                v = float(m.group(1))
                if v in account_amounts:
                    continue
                if attr == fee["attribute"] or any(w in low for w in fee["cues"]):
                    allowed = {float(self.fees[opt])} if opt else {float(x) for x in self.fees.values()}
                    if v not in allowed and not any(w in low for w in fee["exempt_cues"]):
                        out.append(GateResult(GateDecision.DENY, "speech", f"fee ${v:g} for {opt or self.option_set} "
                                              f"is not the ENM value {sorted(allowed)}", rule="ENM"))
                elif not account_amounts or v not in account_amounts:
                    out.append(GateResult(GateDecision.DENY, "speech", f"amount ${v:g} matches no returned {noun}",
                                          rule="R7/R24"))
            for m in _DAYS.finditer(sent):
                lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
                if "ago" in low[m.end():m.end() + 6]:
                    if not any(mm and mm in low for mm in names):
                        continue                 # a hypothetical ("a charge from 31 days ago"), not an account fact
                    if lo not in account_ages:
                        out.append(GateResult(GateDecision.DENY, "speech", f"age {lo} days matches no returned "
                                              f"{noun}", rule="R7/R24"))
                    continue
                if attr == dt["attribute"] or any(w in low for w in dt["cues"]):
                    rng = [self.days[opt]] if opt else list(self.days.values())
                    ok = any(hi in d or lo in d for d in rng)
                    if not ok:
                        out.append(GateResult(GateDecision.DENY, "speech", f"{lo}-{hi} days is not a delivery time in "
                                              f"the ENM register {self.days}", rule="ENM"))
                elif attr == rw["attribute"] and hi != self.window[1]:
                    out.append(GateResult(GateDecision.DENY, "speech", f"window {hi} days != {self.window[0]}="
                                          f"{self.window[1]}", rule="ENM"))
            cap_re, cap_name, cap_val = self.cap
            m = cap_re.search(low)
            if m and int(m.group(1)) != cap_val:
                out.append(GateResult(GateDecision.DENY, "speech", f"says up to {m.group(1)} charges; "
                                      f"{cap_name}={cap_val}", rule="ENM"))
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

