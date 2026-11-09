"""The runtime monitor: the checks that would run next to the bot in production.

Parts of this file:
  1. Rule checks as generic templates, compiled from ``domain.yaml`` -> rules (authentication before account
     tools, attempt limits, confirmation before a write ...).
  2. ``NumericRangeAttestor``: every figure the bot speaks, against the exact register.
  3. ``RuntimeMonitor``: one conversation, every proposed tool call through the gates in order (syntax, policy,
     GMS plausibility), every utterance through the speech gates, completeness at the end, an audit hash chain.
  4. ``GraphRuntimeMonitor``: the same for a compact-graph pack.
  5. Adapters onto forgeloop's plausibility gate and audit logger.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# =====================================================================================================================
# PART 1 - Runtime rule checks as generic templates (domain.yaml -> rules).
#
# Every policy rule the runtime monitor enforces is one of a small, closed set of check
# templates; a domain pack only supplies parameters (which tool, which argument, which limit,
# which rule id). Limits and figures are read from exact numeric memory when the checks are
# compiled, never from the config.
#
# Call checks (before every tool call, first non-allow decides):
#   no_call_after_handoff  nothing after the handoff tool
#   requires_auth          read / write tools only after the auth tool succeeded
#   max_calls              a tool is called at most <limit> times
#   item_scope             the item a tool writes was returned by the read tool, is inside the
#                          window and within the per-session cap
#   value_from_caller      the value written is the caller's latest clear answer for that slot
#   terminal_value         once <value> is written the next action must be the handoff
#   offered_only           the value chosen was among the options the read tool returned
#   disclose_then_confirm  a figure was spoken and a confirmation followed before the write
#   handoff_when_requested the caller asked for a person: the next action should be the handoff
#   turn_limit             no further work past the turn limit
# Completion checks (end of a conversation without a handoff):
#   all_items_written, terminal_value_handed_off
# =====================================================================================================================

from .gates import GateDecision, GateResult, GateState, ToolCall

A, D, E = GateDecision.ALLOW, GateDecision.DENY, GateDecision.ESCALATE
CallCheck = Callable[[ToolCall, GateState], GateResult]


def _domain():
    from ..packs.loading import current
    return current()


def _tools() -> dict[str, dict]:
    return _domain().section("tools")


def _role_tool(role: str) -> str | None:
    return next((t for t, sp in _tools().items() if sp.get("role") == role), None)


def _ok_status(tool: str) -> str:
    sp = _tools()[tool]
    if sp.get("check"):
        return sp["check"]["success"]
    return (sp.get("status") or {}).get("ok") or (sp.get("status") or {}).get("some")


def figure(store, name: str) -> Any:
    """A limit or named figure from exact numeric memory."""
    dom = _domain()
    named = dom.get(f"figures.named.{name}")
    if named:
        return store.lookup_enm(named["enm_category"], named["key"])
    return store.lookup_enm(dom.get("figures.limits.enm_category", "limit"), name)


def _items(st: GateState, source: str, ok_only: bool = False) -> list[dict]:
    c = st.last(source)
    if not c or (ok_only and not c.get("ok")):
        return []
    return ((c.get("result") or {}).get(_tools()[source]["key"], []))


# ---------------------------------------------------------------------- call checks
def compile_call_checks(store, specs: list[dict] | None = None) -> list[CallCheck]:
    """Compile the pack's call rules (domain.yaml -> rules.calls) into checks on each proposed tool call:
    verified before account tools, attempt limits, one write per item, confirmation before a write, nothing after a handoff.
    """
    specs = specs if specs is not None else (_domain().get("rules.calls") or [])
    handoff = _role_tool("handoff")
    auth = _role_tool("auth")
    from ..bots.tools import account_tools
    acct = account_tools()

    def authed(st: GateState) -> bool:
        ok = _ok_status(auth)
        return any((c.get("result") or {}).get("status") == ok for c in st.calls(auth))

    out: list[CallCheck] = []
    for sp in specs:
        kind = sp["check"]
        rule = sp.get("rule")

        if kind == "no_call_after_handoff":
            def chk(a, st, rule=rule):
                if st.calls(handoff):
                    return GateResult(D, "policy", f"tool call after {handoff}", rule=rule)
                return GateResult(A, "policy")

        elif kind == "requires_auth":
            def chk(a, st, rule=rule):
                if a.tool_name in acct and not authed(st):
                    return GateResult(D, "policy", f"{a.tool_name} before {_ok_status(auth)}", rule=rule)
                return GateResult(A, "policy")

        elif kind == "max_calls":
            def chk(a, st, sp=sp, rule=rule):
                L = figure(store, sp["limit"])
                if a.tool_name == sp["tool"] and len(st.calls(sp["tool"])) >= L:
                    return GateResult(D, "policy", f"attempt {len(st.calls(sp['tool'])) + 1} > {sp['limit']}={L}",
                                      rule=rule)
                return GateResult(A, "policy")

        elif kind == "item_scope":
            def chk(a, st, sp=sp, rule=rule):
                if a.tool_name != sp["tool"]:
                    return GateResult(A, "policy")
                arg, win, cap = sp["arg"], sp.get("window"), sp.get("cap")
                tid = a.arguments.get(arg)
                got = {t[arg]: t for t in _items(st, sp["source"])} if st.last(sp["source"]) else {}
                if tid not in got:
                    return GateResult(D, "policy", f"{tid} was not returned by {sp['source']}", rule=rule)
                if win:
                    L = figure(store, win["limit"])
                    if got[tid].get(win["field"], 0) > L:
                        return GateResult(D, "policy", f"{tid} is {got[tid][win['field']]} days old > "
                                          f"{win['limit']}={L}", rule=win["rule"])
                if cap:
                    L = figure(store, cap["limit"])
                    done = {c["args"].get(arg) for c in st.calls(sp["tool"])}
                    if tid not in done and len(done) >= L:
                        return GateResult(D, "policy", f"{cap['noun']} {len(done) + 1} > {cap['limit']}={L}",
                                          rule=cap["rule"])
                return GateResult(A, "policy")

        elif kind == "value_from_caller":
            def chk(a, st, sp=sp):
                if a.tool_name != sp["tool"]:
                    return GateResult(A, "policy")
                said = None
                for f in reversed(st.user_frames):
                    if getattr(f, "act", None) == sp["act"]:
                        said = f.get(sp["slot"])
                        break
                want = a.arguments.get(sp["arg"])
                if said is None:
                    return GateResult(D, "policy", f"records {want} but the caller gave no clear answer",
                                      rule=sp["rule_missing"])
                if said != want:
                    return GateResult(D, "policy", f"records {want}; the caller's latest clear answer is {said}",
                                      rule=sp["rule_mismatch"])
                return GateResult(A, "policy")

        elif kind == "terminal_value":
            def chk(a, st, sp=sp, rule=rule):
                c = st.last(sp["tool"])
                if c and c["args"].get(sp["arg"]) == sp["value"] and a.tool_name != handoff:
                    return GateResult(D, "policy", f"{sp['value']} was recorded; the next action must be a transfer",
                                      rule=rule)
                return GateResult(A, "policy")

        elif kind == "offered_only":
            def chk(a, st, sp=sp, rule=rule):
                if a.tool_name == sp["tool"]:
                    c = st.last(sp["source"])
                    opts = ((c or {}).get("result") or {}).get(sp["key"], []) if c and c.get("ok") else []
                    if a.arguments.get(sp["arg"]) not in opts:
                        return GateResult(D, "policy", f"{a.arguments.get(sp['arg'])} not among returned options {opts}",
                                          rule=rule)
                return GateResult(A, "policy")

        elif kind == "disclose_then_confirm":
            def chk(a, st, sp=sp, rule=rule):
                if a.tool_name != sp["tool"]:
                    return GateResult(A, "policy")
                since = st.options_at if st.options_at is not None else 0
                confirmed = [i for i, f in enumerate(st.user_frames)
                             if i >= since and getattr(f, "act", None) == sp["confirm_act"]]
                if all(a.arguments.get(k) == v for k, v in sp["when"].items()):
                    if st.fee_disclosed_at is None or not any(i >= st.fee_disclosed_at for i in confirmed):
                        what = " ".join(str(v).replace("_", " ") for v in sp["when"].values())
                        return GateResult(D, "policy", f"{what} without disclosing the ${figure(store, sp['figure'])} "
                                          "fee and a confirmation after it", rule=rule)
                if not confirmed:
                    return GateResult(E, "policy", "submit without a customer confirmation", rule=sp["confirm_rule"])
                return GateResult(A, "policy")

        elif kind == "handoff_when_requested":
            def chk(a, st, rule=rule):
                if st.human_requested and a.tool_name != handoff:
                    return GateResult(E, "policy", "customer asked for a human; next action should be a transfer",
                                      rule=rule)
                return GateResult(A, "policy")

        elif kind == "turn_limit":
            def chk(a, st, sp=sp, rule=rule):
                L = figure(store, sp["limit"])
                if st.user_turns > L and a.tool_name != handoff:
                    return GateResult(E, "policy", f"turn {st.user_turns} > {sp['limit']}={L}", rule=rule)
                return GateResult(A, "policy")

        elif kind.startswith("graph_"):                  # a compact policy graph's checks (graph/compact.py)
            chk = _graph_check(kind, sp, rule)

        else:
            raise ValueError(f"rules.calls: unknown check {kind!r}")
        chk.__name__ = kind
        out.append(chk)
    return out


def _graph_check(kind: str, sp: dict, rule: str | None) -> CallCheck:
    """Call checks of a compact policy graph's machine-checkable rules (the same rules the trace's
    admissibility scores): requires_status, value_allowed, requires_slot, confirm_before,
    disclose_before. The monitor sees the caller's frames and the bot's acts, never the oracle."""
    from ..packs.loading import current
    g = current().graph
    k = kind[len("graph_"):]
    tool = sp[k]

    def chk(a, st, sp=sp, rule=rule):
        if a.tool_name != tool:
            return GateResult(A, "policy")
        if k == "requires_status":
            c = st.last(sp["needs"])
            if not c or (c.get("result") or {}).get("status") != sp["status"]:
                return GateResult(D, "policy", f"{tool} without {sp['needs']} {sp['status']}", rule=rule)
        elif k == "value_allowed":
            v = a.arguments.get(sp["arg"])
            if g.attr(v, sp["attr"]) not in sp["values"]:
                return GateResult(D, "policy", f"{tool} for {v}: {sp['attr']} is {g.attr(v, sp['attr'])}", rule=rule)
        elif k == "requires_slot":
            said = [f.get(sp["slot"]) for f in st.user_frames if getattr(f, "get", None) and f.get(sp["slot"]) is not None]
            if not said or said[-1] not in sp["values"]:
                return GateResult(D, "policy", f"{tool} without the caller saying {sp['slot']} in {sp['values']}", rule=rule)
        elif k == "confirm_before":
            last = st.user_frames[-1] if st.user_frames else None
            if getattr(last, "act", None) != sp["act"]:
                return GateResult(E, "policy", f"{tool} without a caller {sp['act']} just before", rule=rule)
        elif k == "disclose_before":
            v = a.arguments.get(sp["arg"])
            if g.attr(v, sp["attr"]) in sp["values"]:
                acts = getattr(st, "bot_acts", [])
                said = [i for i, x in acts if x.get("act") == sp["act"] and x.get(sp["arg"], v) == v]
                conf = [i for i, f in enumerate(st.user_frames) if said and i >= said[-1] and getattr(f, "act", None) == "CONFIRM"]
                if not said or not conf:
                    return GateResult(D, "policy", f"{tool} for {v} without {sp['act']} and a confirmation after it",
                                      rule=rule)
        return GateResult(A, "policy")
    return chk


# ---------------------------------------------------------------------- completion checks
def completion_results(store, st: GateState, specs: list[dict] | None = None) -> list[GateResult]:
    """End-of-call completeness checks (domain.yaml -> rules.completion), e.g. every reviewed item written."""
    specs = specs if specs is not None else (_domain().get("rules.completion") or [])
    if st.calls(_role_tool("handoff")):
        return []
    out: list[GateResult] = []
    for sp in specs:
        kind = sp["check"]
        ok = _ok_status(sp["tool"])
        if kind == "all_items_written":
            arg, win = sp["arg"], sp["window"]
            look, cap = figure(store, win["limit"]), figure(store, sp["cap"])
            due = [t[arg] for t in _items(st, sp["source"], ok_only=True) if t.get(win["field"], 0) <= look][:cap]
            done = {x["args"].get(arg) for x in st.calls(sp["tool"]) if (x.get("result") or {}).get("status") == ok}
            missing = [t for t in due if t not in done]
            if missing:
                out.append(GateResult(E, "completeness", f"conversation ended with reviewable "
                                      f"transactions never recorded: {missing}", rule=sp["rule"]))
        elif kind == "terminal_value_handed_off":
            item_arg = next((c["arg"] for c in specs if c["check"] == "all_items_written"), "transaction_id")
            hits = [x["args"].get(item_arg) for x in st.calls(sp["tool"])
                    if x["args"].get(sp["arg"]) == sp["value"] and (x.get("result") or {}).get("status") == ok]
            if hits:
                out.append(GateResult(E, "completeness", f"{sp['value']} recorded for {hits} but the "
                                      "conversation ended without a transfer", rule=sp["rule"]))
        else:
            raise ValueError(f"rules.completion: unknown check {kind!r}")
    return out


# =====================================================================================================================
# PART 2 - Numeric range attestor for figures the bot speaks (gmsh harness/governance/policy_numeric.py).
#
# gmsh's attestor gives every number a three-way verdict: the rule fails -> BLOCK; the
# rule passes and the value sits inside the calibrated range -> PASS; the rule passes but
# the value is outside -> REVIEW. Here the "rule" and the "range" come from the exact
# numeric memory (ENM) of the store, per family of figure a caller can hear:
#
#   auth_attempts         "you have N attempts / tries"            register: max_auth_attempts
#   transactions          "N transactions / charges"                register: max_transactions_per_session
#   review_window         "N days" about how far back / the window  register: lookback_days
#   delivery_time         "N(-M) business days" about delivery      register: delivery_days_min/max of every option
#   fee                   "$N" / "N dollars" not an account amount  register: delivery fees of every option
#
# Verdicts:
#   PASS    the figure is the register value for its family;
#   REVIEW  the figure is a register value of a *related* family (another limit, another
#           option's days) - a plausible-sounding but possibly misattributed figure: a
#           human should look (ESCALATE);
#   FAIL    the figure is in no register at all - an invented number (DENY).
#
# The speech monitor already denies wrong fees and delivery times where it can tell which
# option a sentence is about; the attestor covers the figures it does not check (limits
# phrased freely, days without a delivery keyword) and never re-flags a figure another
# speech check already flagged.
# =====================================================================================================================

_DAYS = re.compile(r"\b(\d+)\s*(?:(?:to|-|–)\s*(\d+)\s*)?(?:business\s+|calendar\s+)?days?\b(?!\s+(?:ago|old|back))", re.I)
_MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s*dollars\b", re.I)


@dataclass(frozen=True)
class Attestation:
    """One spoken figure's verdict against the exact register: PASS, REVIEW or FAIL, with the expected values."""
    family: str
    value: float
    verdict: str           # PASS | REVIEW | FAIL
    expected: tuple

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "value": self.value, "verdict": self.verdict, "expected": list(self.expected)}


class NumericRangeAttestor:
    """The families, their spoken patterns, registers and related figures come from
    domain.yaml -> rules.attestor; every register value is read from exact numeric memory."""

    def __init__(self, store, cfg: dict | None = None):
        if cfg is None:
            from ..packs.loading import current
            cfg = current().get("rules.attestor")
        self.families = cfg["families"]

        def values(ref: str) -> set[float]:
            cat, key = ref.split("/", 1)
            if key == "*":                     # every key of the category, read from the register
                return {float(store.lookup_enm(cat, k)) for (c, k) in store.enm if c == cat}
            return {float(store.lookup_enm(cat, key))}

        self.register = {f: set().union(*(values(r) for r in sp["register"])) for f, sp in self.families.items()}
        limits = {float(store.lookup_enm("limit", k)) for k in cfg["limits"]}
        self.related = {f: set().union(*(limits if r == "limits" else self.register[r] for r in sp["related"]))
                        for f, sp in self.families.items()}
        self._pat = {f: re.compile(sp["pattern"], re.I) for f, sp in self.families.items() if sp.get("pattern")}

    def verdict(self, family: str, value: float) -> Attestation:
        exp = tuple(sorted(self.register[family]))
        if value in self.register[family]:
            return Attestation(family, value, "PASS", exp)
        if value in self.related[family]:
            return Attestation(family, value, "REVIEW", exp)
        return Attestation(family, value, "FAIL", exp)

    def attest(self, text: str, account_amounts: set[float] | None = None) -> list[Attestation]:
        out: list[Attestation] = []
        account_amounts = account_amounts or set()
        fams = self.families
        day_fams = [f for f, sp in fams.items() if sp.get("days")]
        money_fams = [f for f, sp in fams.items() if sp.get("money")]
        for sent in re.split(r"(?<=[.?!])\s+", text or ""):
            low = sent.lower()
            if not re.search(r"\d", low):
                continue
            for f, pat in self._pat.items():          # families with their own pattern
                if fams[f].get("only_if") and not re.search(fams[f]["only_if"], low):
                    continue                          # e.g. a limit statement, not a count of what was found
                for m in pat.finditer(low):
                    out.append(self.verdict(f, float(m.group(1))))
            for m in _DAYS.finditer(low):             # a day count: the first family whose cue the sentence has
                fam = next((f for f in day_fams if re.search(fams[f]["cue"], low)), None)
                if fam is None:
                    continue
                for g in (m.group(1), m.group(2)):
                    if g:
                        out.append(self.verdict(fam, float(g)))
            for m in _MONEY.finditer(low):
                v = float(m.group(1) or m.group(2))
                if v in account_amounts:
                    continue
                for f in money_fams:
                    if re.search(fams[f]["cue"], low):
                        out.append(self.verdict(f, v))
        return out


# =====================================================================================================================
# PART 3 - Runtime monitoring of one conversation (book Ch. 12.5; forgeloop GovernanceHarness re-done
#
# for a voice bot whose tool client we wrap).
#
# ``RuntimeMonitor`` keeps the observable state of a conversation (tool calls and
# results, the caller's turns as read by the FrameReader, the bot's utterances) and
# asks every gate before each tool call executes. ``GovernedTools`` is the drop-in
# tool client: in ``monitor`` mode it only records (the test outcome is unchanged), in
# ``enforce`` mode a DENY blocks the call and the bot receives a tool error. Every
# decision is sealed into a hash-chained audit log.
#
# Decision rule (``GovernanceConfig.decision_rule``):
#
# * ``strict``  any gate's DENY denies the call (the default; unchanged behaviour);
# * ``quorum``  gmsh's ``check_with_quorum``: a schema (syntax) error always denies; two or
#   more independent layers (policy, GMS plausibility) denying -> DENY; exactly one ->
#   REVIEW, which is routed onto the existing escalation path (the call proceeds, an
#   escalation is raised for a human); none -> ALLOW.
# Both outcomes are computed for every call and reported, so the cost of either rule
# (false denies vs. reviews) is visible whichever one is deployed.
# =====================================================================================================================

from .gates import (GMSPlanGate, PolicyGate, SpeechMonitor, SyntaxGate, ToolRegistry, compile_policies, completeness_check, injection_flag)
from ..packs.names import N



def state_hash(obj: Any) -> str:
    """A short, stable hash of any JSON-able state (used to seal audit events)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

@dataclass
class GovernanceConfig:
    """The runtime monitor's settings: the calibrated plausibility cut and plan budget, monitor or enforce,
    and the decision rule (strict: first DENY wins; quorum: one blocking layer goes to review).
    """
    theta: float                      # plausibility gate cut (Appendix C)
    drift_budget: float               # plan gate budget (Appendix C)
    relation: str = "enables"
    plan_criterion: str = "mean"      # mean (book) | max
    mode: str = "monitor"             # monitor | enforce
    decision_rule: str = "strict"     # strict | quorum (gmsh check_with_quorum; one blocking layer -> review)


class RuntimeMonitor:
    """Watches one conversation as it happens: every proposed tool call through the gates, every bot
    utterance through the speech gates, completeness at the end; every decision sealed in the audit chain.
    """
    def __init__(self, store, reader, cfg: GovernanceConfig, run_id: str = "run"):
        self.store, self.reader, self.cfg, self.run_id = store, reader, cfg, run_id
        self.registry = ToolRegistry(store)
        self.engine = "forgeloop"
        self.gates = [SyntaxGate(), PolicyGate(compile_policies(store)), ForgeloopPlausibilityGate(store.kx, cfg.theta)]
        self.speech = SpeechMonitor(store)
        self.plan_gate = GMSPlanGate(store, cfg.drift_budget, cfg.relation, criterion=cfg.plan_criterion)
        self.state = GateState()
        self.audit = ForgeloopAudit(run_id)
        self._scores: list[float] = []
        self.events: list[dict[str, Any]] = []
        self.input_flags: list[dict[str, Any]] = []
        self._seq = 0
        self.decisions: list[dict[str, Any]] = []
        self.last_outcome = "allowed"
        from ..packs.loading import current
        trk = current().get("rules.tracking")
        fee = figure(store, trk["disclosed_figure"])
        self._fee_text = {f"${fee}", f"{fee} dollars"}
        self._options_tool = trk["options_tool"]
        self._handoff_act = trk.get("handoff_request_act") or N.act.REQUEST_HUMAN

    # ------------------------------------------------------------------ observers
    def _seal(self, kind: str, proposed: dict, results: list[GateResult], outcome: str) -> None:
        dec = [{"gate": r.gate_name, "decision": r.decision.value, "reason": r.reason, "rule": r.rule,
                "score": r.score} for r in results]
        sh = state_hash([c["tool"] for c in self.state.tool_results] + [self.state.user_turns])
        self.audit.log(self._seq, kind, sh, proposed, dec, outcome)
        self._scores += [r.score for r in results if r.gate_name == "gms_plausibility" and r.score is not None]
        self._seq += 1
        for r in results:
            if r.decision != GateDecision.ALLOW:
                self.events.append({"kind": kind, "gate": r.gate_name, "decision": r.decision.value,
                                    "rule": r.rule, "reason": r.reason, "proposed": proposed,
                                    "turn": self.state.user_turns})

    def on_user(self, text: str, state: str | None = None) -> None:
        frame = self.reader.read(text, state)
        if getattr(self.store, "kg", None) is not None:
            if not hasattr(self, "_ont"):
                from ..cases.meaning import Ontology
                self._ont = Ontology.from_store(self.store)
            frame = self._ont.canonicalize(frame, state)          # read in the state it answers
        self.state.user_frames.append(frame)
        self.state.user_turns += 1
        if getattr(frame, "act", None) == self._handoff_act:
            self.state.human_requested = True
        pat = injection_flag(text)
        if pat:
            self.input_flags.append({"turn": self.state.user_turns, "pattern": pat, "text": text})
            self._seal("input", {"text": text}, [GateResult(GateDecision.ESCALATE, "input_guard",
                                                            f"matched {pat!r}", rule="injection")], "flagged")

    def on_bot(self, text: str, acts: list[dict] | None) -> None:
        self.state.bot_texts.append(text)
        if any(f in (text or "") for f in self._fee_text):
            self.state.fee_disclosed_at = len(self.state.user_frames)
        res = self.speech.check(text, acts, self.state)
        if res:
            self._seal("speech", {"text": text}, res, "flagged")

    def on_speech(self, report) -> None:
        """Seal the speech check's verdict on one bot turn (gmsjourney/speech, a ``TurnReport``) into the audit,
        as the gate ``speech_grounding``.

        Input   the turn's report (None or an ok report: nothing is sealed)
        Effect  a claim that is false or has nothing behind it (contradicted / unsupported) -> DENY;
                an act that leaves out required information or says something other than its label -> ESCALATE.
                Flow findings are not sealed: they are graded on their own (``c_flow``), not by default.

        ``SpeechMonitor`` above is the guard for a few things that must never be said; this layer is the full
        reading of every claim against the policy graph, the GMS and the session (it catches, for example,
        "I've recorded that you recognize it" when UNRECOGNIZED was recorded)."""
        if report is None or report.ok:
            return
        res = [GateResult(GateDecision.DENY, "speech_grounding",
                          f"{v.status}: {v.claim.words!r} ({v.why or v.source})", rule=f"claim:{v.claim.field}")
               for v in report.verdicts if v.failed]
        res += [GateResult(GateDecision.ESCALATE, "speech_grounding", f.detail, rule=f"{f.kind}:{f.act}")
                for f in report.findings if f.kind != "not_allowed_here"]
        self._seal("speech", {"text": report.text}, res, "flagged")

    @staticmethod
    def outcomes(results: list[GateResult]) -> tuple[str, str]:
        """(strict outcome, quorum outcome) for one proposed call."""
        deny_layers = {r.gate_name for r in results if r.decision == GateDecision.DENY}
        escalate = any(r.decision == GateDecision.ESCALATE for r in results)
        strict = "denied" if deny_layers else "escalated" if escalate else "allowed"
        if "syntax" in deny_layers or len(deny_layers) >= 2:
            quorum = "denied"
        elif deny_layers:
            quorum = "review"                       # one blocking layer: a human looks, the call is not denied
        else:
            quorum = "escalated" if escalate else "allowed"
        return strict, quorum

    def before_tool(self, tool: str, args: dict) -> list[GateResult]:
        action = ToolCall(tool, dict(args))
        results = [g.check(action, self.state, self.registry) for g in self.gates]
        strict, quorum = self.outcomes(results)
        worst = quorum if self.cfg.decision_rule == "quorum" else strict
        self.last_outcome = worst
        self.decisions.append({"tool": tool, "strict": strict, "quorum": quorum,
                               "layers": sorted({r.gate_name for r in results if r.decision != GateDecision.ALLOW})})
        if worst == "review":
            results = results + [GateResult(GateDecision.ESCALATE, "quorum", "one blocking layer: routed to review",
                                            rule="quorum")]
        self._seal("tool", {"tool": tool, "args": dict(args), "prev": self.state.prev_node()}, results, worst)
        return results

    def after_tool(self, tool: str, args: dict, result: Any, ok: bool) -> None:
        self.state.tool_results.append({"tool": tool, "args": dict(args), "result": copy.deepcopy(result), "ok": ok})
        if tool == self._options_tool:
            self.state.options_at = len(self.state.user_frames)

    # ------------------------------------------------------------------ end of run
    def finish(self) -> dict[str, Any]:
        steps = [c["tool"] for c in self.state.tool_results]
        comp = completeness_check(self.store, self.state)
        if comp:
            self._seal("end", {"steps": steps}, comp, "flagged")
        plan = self.plan_gate.validate(steps)
        if not plan.admissible:
            stat = plan.drift if self.plan_gate.criterion == "mean" else plan.max_step
            self._seal("plan", {"steps": steps}, [GateResult(GateDecision.ESCALATE, "gms_plan",
                       f"{self.plan_gate.criterion} transition distance {stat:.3f} > budget {self.plan_gate.budget:.3f}",
                       score=stat, rule="plan")], "flagged")
        by: dict[str, int] = {}
        for e in self.events:
            by[f"{e['gate']}:{e['rule'] or e['decision']}"] = by.get(f"{e['gate']}:{e['rule'] or e['decision']}", 0) + 1
        scores = list(self._scores)
        return {"alarms": len(self.events), "deny": sum(e["decision"] == "deny" for e in self.events),
                "escalate": sum(e["decision"] == "escalate" for e in self.events),
                "by_gate_rule": by, "events": self.events, "input_flags": self.input_flags,
                "plan": {"admissible": plan.admissible, "drift": plan.drift, "max_step": plan.max_step},
                "plausibility_scores": scores,
                "decision_rule": self.cfg.decision_rule,
                "numeric_attestations": {k: sum(1 for x in self.speech.attestations if x["verdict"] == k)
                                         for k in ("PASS", "REVIEW", "FAIL")},
                "quorum": {"calls": len(self.decisions),
                           "strict_denied": sum(d["strict"] == "denied" for d in self.decisions),
                           "quorum_denied": sum(d["quorum"] == "denied" for d in self.decisions),
                           "routed_to_review": sum(d["quorum"] == "review" for d in self.decisions),
                           "disagreements": [d for d in self.decisions if (d["strict"] == "denied") != (d["quorum"] == "denied")]},
                "audit": {"n_events": len(self.audit.events), "head": self.audit.head(),
                          "verified": self.audit.verify(), "engine": self.engine}}


@dataclass
class GovernedTools:
    """Tool client wrapper: gates before every call; records results for the monitor."""
    inner: Any
    monitor: RuntimeMonitor
    blocked: list[dict[str, Any]] = field(default_factory=list)

    @property
    def log(self):
        return self.inner.log

    @property
    def fixture(self):
        return self.inner.fixture

    @property
    def injected(self):
        return getattr(self.inner, "injected", [])

    def call(self, tool: str, **args: Any) -> Any:
        from ..bots.tools import ToolError
        res = self.monitor.before_tool(tool, args)
        if self.monitor.cfg.mode == "enforce" and self.monitor.last_outcome == "denied":
            reason = "; ".join(f"{r.gate_name}: {r.reason}" for r in res if r.decision == GateDecision.DENY)
            self.blocked.append({"tool": tool, "args": dict(args), "reason": reason})
            self.inner.log.append({"tool": tool, "args": dict(args), "result": {"error": f"blocked: {reason}"},
                                   "ok": False, "blocked": True})
            self.monitor.after_tool(tool, args, {"error": "blocked"}, False)
            raise ToolError(f"blocked by governance: {reason}")
        try:
            out = self.inner.call(tool, **args)
        except Exception as e:
            self.monitor.after_tool(tool, args, {"error": str(e)}, False)
            raise
        self.monitor.after_tool(tool, args, out, True)
        return out


# =====================================================================================================================
# PART 4 - The runtime monitor for a compact policy graph (book Ch. 12.5).
#
# The same gates as the v1 monitor - schema (syntax), the policy's call checks compiled from the
# graph's machine-checkable rules, knowlytix/forgeloop GMS plausibility and the plan gate - with a
# speech monitor that reads the bot's structured acts: a success claim (a state that ``claims`` a
# tool) needs a success echoing the request, a stance the policy contradicts is denied in u-space,
# and every money figure the bot speaks must be a register value.
# =====================================================================================================================


_DOLLARS = re.compile(r"\$\s?(\d+(?:\.\d+)?)")


class GraphSpeechMonitor:
    """The speech gates for a compact-graph pack: the pack's speech rules (patterns) and every spoken
    money figure against the graph's figures.
    """
    name = "speech"

    def __init__(self, store):
        from ..packs.loading import current
        self.store = store
        dom = current()
        self.g = dom.graph
        self.checks = [{**c, "_re": re.compile(c["pattern"], re.I)} for c in dom.get("rules.speech") or []]
        num = self.g.numeric_attributes()
        self.money = {float(store.lookup_enm(a, m)) for a, sp in num.items() if a.endswith("_usd") for m in sp["members"]}
        self.attestations: list[dict[str, Any]] = []

    def check(self, text: str, acts: list[dict] | None, st: GateState) -> list[GateResult]:
        out: list[GateResult] = []
        names = {a.get("act") for a in (acts or [])}
        t = text or ""
        for c in self.checks:
            if c["check"] == "claim_after_success":
                if c["act"] in names or (not acts and c["_re"].search(t)):
                    sp = self.g.tools[c["tool"]]
                    last = st.last(c["tool"])
                    r = (last or {}).get("result") or {}
                    ok = last is not None and last.get("ok") and r.get("status") == sp["ok"] and \
                        (not c.get("echo") or r.get(c["echo"]) == last["args"].get(c["echo"]))
                    if not ok:
                        out.append(GateResult(GateDecision.DENY, "speech", c["message"], rule=c["rule"]))
            elif c["check"] == "forbidden_stance" and c["_re"].search(t):
                try:
                    chk = self.store.stance_check(*c["stance"], +1)
                except Exception:                     # noqa: BLE001 - a store without the u-space
                    continue
                if chk["verdict"] == "contradicted":
                    out.append(GateResult(GateDecision.DENY, "speech", c["message"], score=chk["energy"], rule=c["rule"]))
        for m in _DOLLARS.finditer(t):
            v = float(m.group(1))
            verdict = "PASS" if v in self.money else "FAIL"
            self.attestations.append({"value": v, "verdict": verdict, "family": "money"})
            if verdict == "FAIL":
                out.append(GateResult(GateDecision.DENY, "numeric_attestor", f"money figure {v:g} is in no register",
                                      rule="NUM-FAIL"))
        return out


class GraphRuntimeMonitor(RuntimeMonitor):
    """The runtime monitor for a compact-graph pack (the same gates, the graph's speech monitor)."""
    def __init__(self, store, reader, cfg: GovernanceConfig, run_id: str = "run"):
        self.store, self.reader, self.cfg, self.run_id = store, reader, cfg, run_id
        self.registry = ToolRegistry(store)
        self.engine = "forgeloop"
        self.gates = [SyntaxGate(), PolicyGate(compile_policies(store)), ForgeloopPlausibilityGate(store.kx, cfg.theta)]
        self.speech = GraphSpeechMonitor(store)
        self.plan_gate = GMSPlanGate(store, cfg.drift_budget, cfg.relation, criterion=cfg.plan_criterion)
        self.state = GateState()
        self.state.bot_acts = []
        self.audit = ForgeloopAudit(run_id)
        self._scores: list[float] = []
        self.events: list[dict[str, Any]] = []
        self.input_flags: list[dict[str, Any]] = []
        self._seq = 0
        self.decisions: list[dict[str, Any]] = []
        self.last_outcome = "allowed"
        self._fee_text: set[str] = set()
        self._options_tool = None
        self._handoff_act = (self.store.kg.graph.glob.get("human") or {}).get("act", "REQUEST_HUMAN")

    def on_bot(self, text: str, acts: list[dict] | None) -> None:
        for a in acts or []:
            self.state.bot_acts.append((len(self.state.user_frames), dict(a)))
        super().on_bot(text, acts)


# =====================================================================================================================
# PART 5 - forgeloop's governance pieces driving our runtime monitor.
#
# * ``forgeloop.gms_backend.GMSPlausibilityGate`` scores workflow transitions on the
#   knowlytix store's ``has_enables`` relation (it needs nothing but ``score_triple``); an
#   adapter converts its results into the monitor's GateResult and applies the
#   safe-harbour rule for escalation actions. (The whole-plan check is ours:
#   checks/gates.py GMSPlanGate, on the same store.)
# * ``forgeloop.audit.AuditLogger`` (SHA-256 hash chain) seals every decision.
# * ``forgeloop.governance.policies`` (PII, prompt injection) - the regex baseline.
# =====================================================================================================================

from ..packs import names as roles


class ForgeloopPlausibilityGate:
    """forgeloop's GMS plausibility gate behind our gate protocol: scores (previous tool, enables, next
    tool) on the store; the safe handoff tool is escalated, never denied, so the bot can always hand over.
    """
    name = "gms_plausibility"

    def __init__(self, kx_store, theta: float, safe_tools=None):
        from forgeloop.gms_backend import GMSPlausibilityGate
        self.theta = float(theta)
        self.safe_tools = safe_tools if safe_tools is not None else frozenset({roles.TRANSFER})
        self.impl = GMSPlausibilityGate(kx_store, theta=theta, relation="has_enables", on_missing="allow",
                                        context=lambda action, state: (state.prev_node() if state is not None
                                                                       else "start").lower())

    def check(self, action, state, registry) -> GateResult:
        r = self.impl.check(action, state, registry)
        dec = GateDecision(r.decision.value if hasattr(r.decision, "value") else str(r.decision))
        score = None
        if "plausibility" in r.reason:
            try:
                score = float(r.reason.split("plausibility ")[1].split()[0])
            except (IndexError, ValueError):
                score = None
        if dec == GateDecision.DENY and action.tool_name in self.safe_tools:
            dec = GateDecision.ESCALATE
        return GateResult(dec, self.name, f"forgeloop: {r.reason}", score=score)


class ForgeloopAudit:
    """forgeloop's AuditLogger with a logical clock (deterministic, replayable chains)."""

    def __init__(self, run_id: str):
        from forgeloop.audit import AuditLogger
        self.run_id, self.log_ = run_id, AuditLogger()

    def log(self, seq: int, kind: str, state_hash: str, proposed: dict, decisions: list[dict], outcome: str):
        from forgeloop.audit import AuditEvent
        self.log_.log(AuditEvent(run_id=self.run_id, step=seq, timestamp=float(seq), state_hash=state_hash,
                                 proposed_action={"kind": kind, **proposed},
                                 observation={"decisions": decisions}, final_state_status=outcome))

    @property
    def events(self):
        return self.log_.events

    def head(self) -> str:
        return self.log_.head()

    def verify(self) -> bool:
        return self.log_.verify()
