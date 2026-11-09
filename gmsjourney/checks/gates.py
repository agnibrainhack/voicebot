"""The runtime gates and their calibration.

Parts of this file:
  1. The gates (forgeloop's protocol, re-implemented): ``SyntaxGate``, ``PolicyGate`` (compiled rules),
     ``GMSPlanGate`` (the whole plan scored on the store), ``SpeechMonitor``; ``GateDecision`` ALLOW / ESCALATE /
     DENY, first DENY wins.
  2. Calibration (Appendix C): the cohort of allowed and disallowed transitions scored by the store, the regulated
     operating point (zero false admits, then fewest false rejects), the logistic P=0.5 point, held-out accuracy,
     bootstrap intervals, a multi-threshold design, and recalibration triggers (PSI drift, store or policy change).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np


# =====================================================================================================================
# PART 1 - Runtime governance gates (book Ch. 12; forgeloop's gate protocol, re-implemented).
#
# A gate inspects a proposed action *before it executes* and returns ALLOW, DENY or
# ESCALATE. The protocol is forgeloop's: ``gate.name`` and
# ``gate.check(action, state, registry) -> GateResult``, where ``action`` has
# ``kind``, ``tool_name`` and ``arguments``. Our gates therefore drop into a
# forgeloop ``GovernedToolExecutor`` and forgeloop's GMS gates accept our store
# (they only need ``score_triple``); checks/runtime.py (part 5) adapts both.
#
# Open baseline vs GMS, as in the book's runtime-monitoring demo:
#
# * ``SyntaxGate``          tool exists, required arguments present, enumerated
#                           values drawn from the store's value sets.
# * ``PolicyGate``          symbolic policy checks compiled from the rules (R2/R3,
#                           R4, R7/R24, R15, R16, R20, R26, R28, R30, R31) with every
#                           limit read from the exact numeric memory, never hard-coded.
# * GMS plausibility    forgeloop's ``GMSPlausibilityGate`` (checks/runtime.py) scores the
#                           workflow transition (previous step -> proposed tool) on the
#                           knowlytix store's ``has_enables`` relation and denies it above the
#                           calibrated theta (Appendix C).
# * ``GMSPlanGate``         whole-plan check: mean (book) and max transition distance.
# * speech gates            checks on what the bot *says*: claims without SUCCESS
#                           (R13/R17), the fraud stance (R27, u-space tension),
#                           numeric grounding of policy figures against the ENM and
#                           of account figures against the tool results (the book's
#                           draft verifier), and PII; the numeric range attestor
#                           (checks/runtime.py part 2, after gmsh policy_numeric) gives every other figure
#                           the bot speaks a PASS / REVIEW / FAIL verdict.
# =====================================================================================================================

from ..packs.names import N



class GateDecision(str, Enum):
    """What a gate decides about a proposed step: let it through, stop it, or let it through and raise an alarm."""
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class GateResult:
    """One gate's decision on one proposed step, with its reason (and the score and rule when there is one)."""
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
        from ..bots.tools import tool_specs
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
    """The first gate: the tool exists and its arguments match its schema."""
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
    """The policy's call checks, compiled from domain.yaml -> rules.calls (checks/runtime.py);
    every number comes from the ENM."""
    from .runtime import compile_call_checks
    return compile_call_checks(store)


def completeness_check(store, st: GateState) -> list[GateResult]:
    """End of conversation (domain.yaml -> rules.completion): without a handoff, every returned
    item due for review must have been written with success, and no terminal value left behind."""
    from .runtime import completion_results
    return completion_results(store, st)


class PolicyGate:
    """The policy gate: the compiled rule checks in order; the first that does not allow decides."""
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
    """A whole plan's verdict: admissible or not, its mean transition distance and its worst single step."""
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
        from ..packs.loading import current
        from .runtime import figure
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
        from .runtime import NumericRangeAttestor
        self.attestor = NumericRangeAttestor(store)
        self.attestations: list[dict[str, Any]] = []

    def check(self, text: str, acts: list[dict] | None, st: GateState) -> list[GateResult]:
        from ..bots.tools import tool_specs
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
            from ..knowledge.questions import _first_option
            opt = _first_option(sent)               # the option the sentence is about, if any
            if opt is None and st.user_frames:      # else the option the caller just asked about
                f = st.user_frames[-1]
                if getattr(f, "act", None) == N.act.ASK_KB and f.get("entity") in self.fees:
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
    """The first prompt-injection pattern the text matches (forgeloop's regex baseline), else None."""
    t = (text or "").lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, t):
            return p
    return None


# =====================================================================================================================
# PART 2 - Calibrating the runtime gates (book Appendix C; beyond-ship-and-pray appendix_C notebook).
#
# * ``workflow_cohort``  labelled transitions: every workflow edge is ``admit``, every
#   other (source, target) pair over the workflow's nodes is ``block``. The cohort
#   comes from the policy graph, not from the agent under test (the appendix's first
#   anti-pattern is calibrating on the agent's own traces).
# * ``sweep``            false admits / false rejects / accuracy per candidate theta,
#   on the book's grid and on every midpoint between observed scores.
# * ``operating_points`` the regulated point - zero false admits, fewest false
#   rejects (the appendix: in a regulated setting a false admit dominates) - and the
#   maximum-accuracy point the appendix's code selects.
# * ``plan_cohort`` / ``calibrate_plan`` the same for the whole-plan drift budget, with
#   admissible plans taken from the oracle's journeys and inadmissible ones made by
#   skipping, swapping or prepending a step.
# * ``multi_threshold_design`` a balanced design (proofloop's ``balanced_design``) over
#   the four thresholds that matter, each row scored on its own labelled cohort.
# * ``recalibration_triggers`` store retrained, score drift (PSI), false-reject creep,
#   policy changed.
#
# Comparability with gmsh (GMSJudge / plausibility_calibration):
#
# * ``logistic_p50``     gmsh's operating point: a one-feature L2 logistic fit of P(admit | score)
#   and the score where P = 0.5, reported next to the regulated point;
# * ``holdout_eval``     stratified k-fold: every operating-point rule is fitted on the other
#   folds and scored on the held-out one (gmsh reports training-set accuracy only);
# * ``bootstrap_thetas`` 200 stratified resamples of the cohort -> mean, sd and 95% interval of
#   each rule's threshold (gmsh plausibility_calibration's bootstrap, CI kept for the audit);
# * ``tau_bootstrap``    the same for every relation's cut on the store's own facts vs
#   counterfactual tails, plus the realised true/false accept rates at the deployed cut.
# =====================================================================================================================


BOOK_GRID = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]


@dataclass
class LabeledEdge:
    """One calibration example: (context, relation, tail) labelled admit (a policy edge) or block."""
    context: str
    relation: str
    tail: str
    label: str          # admit | block


def workflow_cohort(store, relation: str = "enables") -> list[LabeledEdge]:
    """The calibration cohort: every workflow edge of the policy (admit) and every other possible
    transition between its tools (block).
    """
    edges = {(w.src, w.dst) for w in store.workflow}
    nodes = sorted({w.src for w in store.workflow} | {w.dst for w in store.workflow})
    heads = sorted({w.src for w in store.workflow})
    out = [LabeledEdge(a, relation, b, "admit") for a, b in sorted(edges)]
    out += [LabeledEdge(a, relation, b, "block") for a in heads for b in nodes
            if b != "start" and (a, b) not in edges]
    return out


def sweep(scores: list[tuple[float, str]], thetas: list[float]) -> list[dict[str, Any]]:
    """False admits, false rejects and accuracy of the rule "admit when score <= theta" at each theta."""
    rows = []
    for th in thetas:
        fa = sum(1 for s, l in scores if s <= th and l == "block")
        fr = sum(1 for s, l in scores if s > th and l == "admit")
        tp = sum(1 for s, l in scores if s <= th and l == "admit")
        tn = sum(1 for s, l in scores if s > th and l == "block")
        rows.append({"theta": round(th, 4), "false_admits": fa, "false_rejects": fr,
                     "accuracy": (tp + tn) / max(1, len(scores))})
    return rows


def operating_points(scores: list[tuple[float, str]]) -> dict[str, Any]:
    """The candidate thresholds (midpoints between observed scores) and the two operating points: the
    regulated one (zero false admits, then fewest false rejects) and the most accurate one; plus AUC and separability.
    """
    vals = sorted({s for s, _ in scores})
    cands = [vals[0] - 1e-6] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [vals[-1] + 1e-6]
    rows = sweep(scores, cands)
    zero_fa = [r for r in rows if r["false_admits"] == 0]
    regulated = min(zero_fa, key=lambda r: (r["false_rejects"], -r["theta"]))
    best_acc = max(rows, key=lambda r: (r["accuracy"], -r["false_admits"], -r["false_rejects"]))
    pos = [s for s, l in scores if l == "admit"]
    neg = [s for s, l in scores if l == "block"]
    auc = float(np.mean([1.0 if p < n else 0.5 if p == n else 0.0 for p in pos for n in neg])) if pos and neg else float("nan")
    return {"regulated": regulated, "max_accuracy": best_acc, "auc": auc,
            "separable": max(pos) < min(neg) if pos and neg else None,
            "max_admit_score": max(pos) if pos else None, "min_block_score": min(neg) if neg else None}


# ----------------------------------------------------------------------------- gmsh comparability
def logistic_p50(scores: list[tuple[float, str]], l2: float = 1.0, iters: int = 100) -> dict[str, Any]:
    """One-feature logistic P(admit | score) (L2 on the slope, like sklearn's default C=1) and the
    score where P = 0.5 - gmsh GMSJudge's operating-point rule - with its errors on the cohort."""
    x = np.array([s for s, _ in scores], dtype=float)
    y = np.array([1.0 if l == "admit" else 0.0 for _, l in scores])
    if len(set(y)) < 2:
        return {"theta": float("nan"), "note": "one class only"}
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    X = np.c_[np.ones_like(x), (x - mu) / sd]
    w = np.zeros(2)
    R = np.diag([0.0, l2])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (y - p) - R @ w
        H = X.T @ (X * (p * (1 - p) + 1e-9)[:, None]) + R
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    theta = float(mu + sd * (-w[0] / w[1])) if abs(w[1]) > 1e-12 else float("nan")
    at = sweep(scores, [theta])[0] if theta == theta else {}
    return {"theta": theta, "coef": [float(w[0]), float(w[1])], **{k: v for k, v in at.items() if k != "theta"},
            "rule": "P(admit|score) = 0.5 (gmsh GMSJudge)"}


def _rules(scores: list[tuple[float, str]]) -> dict[str, float]:
    ops = operating_points(scores)
    return {"regulated": ops["regulated"]["theta"], "max_accuracy": ops["max_accuracy"]["theta"],
            "p50": logistic_p50(scores)["theta"]}


def holdout_eval(scores: list[tuple[float, str]], k: int = 5, seed: int = 0) -> dict[str, Any]:
    """Stratified k-fold: fit each rule on k-1 folds, count its errors on the held-out fold."""
    rng = np.random.default_rng(seed)
    pos = [i for i, (_, l) in enumerate(scores) if l == "admit"]
    neg = [i for i, (_, l) in enumerate(scores) if l == "block"]
    k = max(2, min(k, len(pos), len(neg)))
    rng.shuffle(pos)
    rng.shuffle(neg)
    fold = {i: j % k for j, i in enumerate(pos)} | {i: j % k for j, i in enumerate(neg)}
    tot = {r: {"false_admits": 0, "false_rejects": 0, "n": 0} for r in ("regulated", "max_accuracy", "p50")}
    for f in range(k):
        train = [scores[i] for i in range(len(scores)) if fold[i] != f]
        test = [scores[i] for i in range(len(scores)) if fold[i] == f]
        for r, th in _rules(train).items():
            if th != th:
                continue
            sw = sweep(test, [th])[0]
            tot[r]["false_admits"] += sw["false_admits"]
            tot[r]["false_rejects"] += sw["false_rejects"]
            tot[r]["n"] += len(test)
    return {"k": k, **{r: {**v, "heldout_accuracy": 1 - (v["false_admits"] + v["false_rejects"]) / max(1, v["n"])}
                       for r, v in tot.items()}}


def bootstrap_thetas(scores: list[tuple[float, str]], B: int = 200, seed: int = 0) -> dict[str, Any]:
    """Stratified bootstrap of every rule's threshold: mean, sd and 95% interval."""
    rng = np.random.default_rng(seed)
    pos = [s for s in scores if s[1] == "admit"]
    neg = [s for s in scores if s[1] == "block"]
    draws: dict[str, list[float]] = {"regulated": [], "max_accuracy": [], "p50": []}
    for _ in range(B):
        sample = [pos[i] for i in rng.integers(0, len(pos), len(pos))] + [neg[i] for i in rng.integers(0, len(neg), len(neg))]
        for r, th in _rules(sample).items():
            if th == th:
                draws[r].append(th)
    out = {}
    for r, v in draws.items():
        if v:
            a = np.array(v)
            out[r] = {"mean": float(a.mean()), "sd": float(a.std()), "ci95": [float(np.percentile(a, 2.5)),
                                                                             float(np.percentile(a, 97.5))], "n": len(v)}
    return {"B": B, **out}


def tau_bootstrap(store, B: int = 200, seed: int = 7, max_relations: int = 60) -> dict[str, Any]:
    """Per relation: the cut re-fitted on bootstrap resamples of (true facts, counterfactual tails)
    with the store's own operating-point rule, and the realised TAR/FAR at the deployed cut."""
    import random as _random
    from ..knowledge.store import GMSStore
    rng_py = _random.Random(seed)
    rng = np.random.default_rng(seed)
    out = {}
    for rel, trs in sorted(store.by_r.items())[:max_relations]:
        try:
            tau = float(store.tau(rel))
        except (KeyError, ValueError):
            continue
        trs = sorted(trs)
        pos = [float(store.score_triple(h, r, t)) for h, r, t in trs]
        neg = [float(store.score_triple(h, r, c)) for h, r, t in trs for c in store.negatives_for(h, r, t, rng_py)]
        if not pos or not neg:
            continue
        taus, tars, fars = [], [], []
        for _ in range(B):
            P = [pos[i] for i in rng.integers(0, len(pos), len(pos))]
            N = [neg[i] for i in rng.integers(0, len(neg), len(neg))]
            op = GMSStore._operating_point(P, N)
            if op["tau"] == op["tau"]:
                taus.append(op["tau"])
            tars.append(float(np.mean([x <= tau for x in P])))
            fars.append(float(np.mean([x <= tau for x in N])))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
        out[rel] = {"tau_deployed": tau, "n_pos": len(pos), "n_neg": len(neg),
                    "refit_tau_ci95": ci(taus), "tar_ci95": ci(tars), "far_ci95": ci(fars)}
    return {"B": B, "relations": out,
            "note": "knowlytix cuts are learned cap radii (not re-fitted at run time); the refit interval shows how "
                    "stable a data-driven cut would be on this cohort, the TAR/FAR intervals how stable the deployed one is"}


def calibrate_plausibility(store, relation: str = "enables") -> dict[str, Any]:
    """Calibrate the transition-plausibility gate on the workflow cohort scored by the store."""
    cohort = workflow_cohort(store, relation)
    scored = [(store.score_triple(e.context, e.relation, e.tail), e.label) for e in cohort]
    ops = operating_points(scored)
    false_rejects = [{"edge": f"{e.context}->{e.tail}", "score": round(s, 4)}
                     for e, (s, _) in zip(cohort, scored) if e.label == "admit" and s > ops["regulated"]["theta"]]
    return {"relation": relation, "n_admit": sum(e.label == "admit" for e in cohort),
            "n_block": sum(e.label == "block" for e in cohort),
            "book_grid": sweep(scored, BOOK_GRID), **ops, "theta": ops["regulated"]["theta"],
            "p50": logistic_p50(scored), "holdout": holdout_eval(scored), "bootstrap": bootstrap_thetas(scored),
            "false_rejects_at_theta": false_rejects,
            "cohort_scores": [{"edge": f"{e.context}->{e.tail}", "label": e.label, "score": round(s, 4)}
                              for e, (s, _) in zip(cohort, scored)]}


# ----------------------------------------------------------------------------- plans
def plan_cohort(store, journeys: list[list[str]], seed: int = 0) -> tuple[list[list[str]], list[list[str]]]:
    """Admissible tool plans (oracle journeys) and inadmissible perturbations of them."""
    edges = {(w.src, w.dst) for w in store.workflow}

    def admissible(seq):
        return all((a, b) in edges for a, b in zip(["start"] + seq, seq))

    pos = sorted({tuple(s) for s in journeys if s and admissible(s)})
    neg = set()
    for s in pos:
        s = list(s)
        for i in range(len(s)):                                  # skip a step
            t = s[:i] + s[i + 1:]
            if t and not admissible(t):
                neg.add(tuple(t))
        for i in range(len(s) - 1):                              # swap adjacent steps
            if s[i] != s[i + 1]:
                t = s[:i] + [s[i + 1], s[i]] + s[i + 2:]
                if not admissible(t):
                    neg.add(tuple(t))
        from ..packs import names as roles
        early = ([roles.ITEMS] + sorted(roles.WRITE_TOOLS)) if roles.ITEMS else sorted(roles.ACCOUNT_TOOLS)
        for tool in early:
            t = [tool] + s                                        # act before authenticating
            if not admissible(t):
                neg.add(tuple(t))
    neg = sorted(neg - set(pos))
    return [list(p) for p in pos], [list(n) for n in neg]


def calibrate_plan(store, journeys: list[list[str]], relation: str = "enables") -> dict[str, Any]:
    """Calibrate the plan gate: admissible against inadmissible plans, scored by mean drift and by worst
    step; deploy the criterion with fewer false rejects at zero false admits.
    """
    pos, neg = plan_cohort(store, journeys)
    g = GMSPlanGate(store, 0.0, relation)
    mean_scores = [(g.validate(p).drift, "admit") for p in pos] + [(g.validate(n).drift, "block") for n in neg]
    max_scores = [(g.validate(p).max_step, "admit") for p in pos] + [(g.validate(n).max_step, "block") for n in neg]
    mean_ops, max_ops = operating_points(mean_scores), operating_points(max_scores)
    # deploy the criterion with fewer false rejects at zero false admits (ties: the book's mean)
    use_max = max_ops["regulated"]["false_rejects"] < mean_ops["regulated"]["false_rejects"]
    chosen = max_ops if use_max else mean_ops
    chosen_scores = max_scores if use_max else mean_scores
    return {"n_admissible_plans": len(pos), "n_inadmissible_plans": len(neg),
            "criterion": "max" if use_max else "mean", "drift_budget": chosen["regulated"]["theta"],
            "p50": logistic_p50(chosen_scores), "holdout": holdout_eval(chosen_scores),
            "bootstrap": bootstrap_thetas(chosen_scores),
            "mean_drift": mean_ops, "max_step": max_ops,     # book criterion / per-step criterion
            "examples": {"admissible": pos[:3], "inadmissible": neg[:5]}}


# ----------------------------------------------------------------------------- multi-threshold DoE
def multi_threshold_design(store, theta_cal: float, n_runs: int = 12, seed: int = 42) -> dict[str, Any]:
    """The appendix's balanced design over the thresholds that matter, each row scored."""
    from ..cases.design import balanced_design, coverage_report
    from ..knowledge.store import Rule
    tens = store.calibration["tension"]
    hol = store.calibration.get("holonomy") or {}
    factors = {
        "theta_plausibility": [round(theta_cal * 0.8, 3), round(theta_cal, 3), round(theta_cal * 1.2, 3), 1.5],
        "tau_contra": [round(tens["tau_contra"], 3), 1.5, 1.7, 1.9],
        "numeric_tolerance": [0.005, 0.01, 0.02],
    }
    if hol.get("tau_path") is not None:                  # a policy with path rules (charge review's precedes)
        factors["holonomy_threshold"] = [round(hol["tau_path"], 3), 0.3, 0.5]
    design = balanced_design(factors, n_runs, seed)
    cohort = workflow_cohort(store)
    scored = [(store.score_triple(e.context, e.relation, e.tail), e.label) for e in cohort]
    planted = dict(store.rules)
    base = (store.rules.get(N.rule.R13) if "R13" in N.rule else None) or \
        next(r for r in store.rules.values() if r.stance == "must_not")
    planted[f"{base.rule_id}x"] = Rule(f"{base.rule_id}x", "must" if base.stance == "must_not" else "must_not",
                                       base.subject, base.predicate, base.obj, "reversed copy", base.source)
    enm_vals = [e.as_number() for e in store.enm.values()]
    def _defect(x):                                  # a bare score (built-in cohort) or a record (graph compositions)
        return x.get("defect") if isinstance(x, dict) else x
    hp = [_defect(x) for x in hol.get("positives") or []]
    hn = [_defect(x) for x in hol.get("negatives") or []]
    rows = []
    saved = dict(tens)
    try:
        for row in design:
            th = row["theta_plausibility"]
            pl = sweep(scored, [th])[0]
            store.calibration["tension"] = {**saved, "tau_contra": row["tau_contra"]}
            c_err = int(store.find_contradictions() != []) + int(len(store.find_contradictions(planted)) == 0)
            tol = row["numeric_tolerance"]
            n_err = 0
            for v in enm_vals:          # a stated figure is accepted iff it is within tolerance of the register
                for claim, should in ((v, True), (float(f"{v:.1f}"), True), (v * 1.01 + (0.01 if v == 0 else 0), False),
                                      (v * 1.05 + (1 if v == 0 else 0), False), (v + 1, False)):
                    ok = abs(claim - v) <= tol * max(abs(v), 1.0)
                    n_err += ok != should
            h_err = sum(p > row["holonomy_threshold"] for p in hp) + sum(n <= row["holonomy_threshold"] for n in hn) \
                if "holonomy_threshold" in row else 0
            rows.append({**row, "plausibility_errors": pl["false_admits"] + pl["false_rejects"],
                         "contradiction_errors": c_err, "numeric_errors": n_err, "holonomy_errors": h_err,
                         "total_errors": pl["false_admits"] + pl["false_rejects"] + c_err + n_err + h_err})
    finally:
        store.calibration["tension"] = saved
    cov = coverage_report(design, factors)
    balanced = all(max(c.values()) - min(c.values()) <= 1 for c in cov.values())
    return {"factors": factors, "rows": rows, "coverage": {k: {str(a): b for a, b in v.items()} for k, v in cov.items()},
            "balanced": balanced, "best": min(rows, key=lambda r: r["total_errors"])}


# ----------------------------------------------------------------------------- recalibration
def psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Population stability index between two score samples (quantile bins of `expected`)."""
    if not expected or not actual:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(min(expected + actual), max(expected + actual) + 1e-9, bins + 1)
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def store_fingerprint(store) -> str:
    """A short hash of the store's facts and training settings (a change triggers recalibration)."""
    facts = json.dumps(sorted(list(t) for t in store.triples)).encode()
    trn = json.dumps(store.calibration.get("training", {}), sort_keys=True, default=str).encode()
    return hashlib.sha256(facts + trn).hexdigest()[:16]


def policy_fingerprint(store) -> str:
    """A short hash of the policy's rules (a change triggers recalibration)."""
    rules = json.dumps(sorted((r.rule_id, r.stance, r.subject, r.predicate, r.obj) for r in store.rules.values()))
    return hashlib.sha256(rules.encode()).hexdigest()[:16]


def recalibration_triggers(record: dict[str, Any], store, runtime_scores: list[float],
                           runtime_false_reject_rate: float, psi_limit: float = 0.2,
                           fr_margin: float = 0.05) -> dict[str, Any]:
    """record = what was stored at calibration time (fingerprints, reference scores, FR)."""
    p = psi(record["reference_scores"], runtime_scores)
    trig = {
        "store_retrained": store_fingerprint(store) != record["store_fingerprint"],
        "score_drift_psi": (not math.isnan(p)) and p > psi_limit,
        "false_reject_creep": runtime_false_reject_rate > record["false_reject_rate"] + fr_margin,
        "policy_changed": policy_fingerprint(store) != record["policy_fingerprint"],
    }
    return {"psi": p, "psi_limit": psi_limit, "runtime_false_reject_rate": runtime_false_reject_rate,
            "calibrated_false_reject_rate": record["false_reject_rate"], "triggers": trig,
            "recalibrate": any(trig.values())}


# ----------------------------------------------------------------------------- probe sweep
def gate_probe_sweep(store, theta: float, relation: str = "enables") -> dict[str, Any]:
    """Book 12.5.1: every (previous step, proposed tool) pair through the geometric gate,
    compared with the policy graph. Escalation actions are safe-harboured (never denied)."""
    edges = {(w.src, w.dst) for w in store.workflow}
    from ..packs.loading import current
    wf = current().get("flow.workflow")
    tools = sorted({w.dst for w in store.workflow} - {wf["summary"], wf["end"]})
    prevs = [wf["start"]] + tools
    rows, conf = [], {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for p in prevs:
        for t in tools:
            s = store.score_triple(p, relation, t)
            admit = s <= theta
            from ..packs import names as roles
            decision = "allow" if admit else ("escalate" if t == roles.TRANSFER else "deny")
            truth = (p, t) in edges
            conf["tp" if admit and truth else "fp" if admit else "fn" if truth else "tn"] += 1
            rows.append({"prev": p, "proposed": t, "score": round(s, 4), "decision": decision, "in_policy_graph": truth})
    return {"theta": theta, "confusion": conf, "rows": rows,
            "false_denies": [r for r in rows if r["in_policy_graph"] and r["decision"] == "deny"],
            "false_allows": [r for r in rows if not r["in_policy_graph"] and r["decision"] == "allow"]}
