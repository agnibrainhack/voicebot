"""Scoring a run: was it correct, how bad is a failure, may the bot be released.

Parts of this file:
  1. Per-turn constraint budgets (latency, words per turn, extra tool calls): reported, not failed.
  2. Severity and the release gate: every failure code has a severity; action-taking journeys promote it; each
     risk tier has an accuracy floor and failure budgets (``release_gate``).
  3. ``run`` / ``score``: run scenarios through the bot and score each against the oracle (outcome, per-stage
     components, trajectory adherence, the speech check), then summarise.
  4. ``GraphTrace``: the same scoring for a compact-graph pack (every successful write, every outcome act, every
     machine-checkable rule).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


# =====================================================================================================================
# PART 1 - Per-turn constraint budgets (gmsh ConstraintOracle style, applied to a voice bot).
#
# A voice bot can be right and still unusable: an answer that arrives after three
# seconds or runs to eighty words is a failed turn on a phone line. These budgets are
# checked on every bot turn of every run and reported next to correctness - never folded
# into ``correct`` (so pass rates stay comparable with earlier campaigns) - and feed the
# severity model as LOW-severity failure codes.
#
# * ``max_bot_latency_ms``     worst single bot turn (wall clock around start()/respond())
# * ``p95_bot_latency_ms``     95th percentile over the run's bot turns
# * ``max_bot_words``          longest bot utterance, in words
# * ``max_tool_calls``         absolute tool-call budget per run (None = off)
# * ``tool_call_slack``        tool calls allowed above the oracle's count for the same case
# =====================================================================================================================

@dataclass
class ConstraintBudget:
    """Per-turn budgets: bot latency (max and p95), words per bot turn, tool calls over the oracle's count."""
    max_bot_latency_ms: float | None = 2500.0
    p95_bot_latency_ms: float | None = 1500.0
    max_bot_words: int | None = 60
    max_tool_calls: int | None = None
    tool_call_slack: int | None = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return float(s[k])


def check_constraints(bot_turns: list[dict[str, Any]], tool_calls: int, oracle_tool_calls: int | None,
                      budget: ConstraintBudget | None) -> dict[str, Any]:
    """Measure a run against the budgets; returns the measurements and the violations (reported, not failed)."""
    lat = [float(t.get("latency_ms", 0.0)) for t in bot_turns]
    words = [int(t.get("words", 0)) for t in bot_turns]
    m = {"bot_latency_ms_max": round(max(lat), 2) if lat else 0.0, "bot_latency_ms_p95": round(_p95(lat), 2),
         "bot_words_max": max(words) if words else 0, "bot_turns": len(bot_turns)}
    v: list[str] = []
    if budget is not None:
        b = budget
        if b.max_bot_latency_ms is not None and m["bot_latency_ms_max"] > b.max_bot_latency_ms:
            v.append(f"latency_max:{m['bot_latency_ms_max']:.0f}ms>{b.max_bot_latency_ms:.0f}ms")
        if b.p95_bot_latency_ms is not None and m["bot_latency_ms_p95"] > b.p95_bot_latency_ms:
            v.append(f"latency_p95:{m['bot_latency_ms_p95']:.0f}ms>{b.p95_bot_latency_ms:.0f}ms")
        if b.max_bot_words is not None and m["bot_words_max"] > b.max_bot_words:
            v.append(f"words:{m['bot_words_max']}>{b.max_bot_words}")
        if b.max_tool_calls is not None and tool_calls > b.max_tool_calls:
            v.append(f"tool_calls:{tool_calls}>{b.max_tool_calls}")
        if b.tool_call_slack is not None and oracle_tool_calls is not None and \
                tool_calls > oracle_tool_calls + b.tool_call_slack:
            v.append(f"tool_calls_over_oracle:{tool_calls}>{oracle_tool_calls}+{b.tool_call_slack}")
    return {**m, "within_budget": int(not v), "budget_violations": v}


# =====================================================================================================================
# PART 2 - Severity, risk tiers and the tiered release gate (gmsh testing/severity.py + audit.py, for this bot).
#
# Every graded run is reduced to failure codes, each with a base severity, the way gmsh's
# ``CODE_SEVERITY_MAP`` does; the run's risk tier then promotes severities exactly as gmsh's
# ``_TIER_ESCALATION`` does (an action-taking journey promotes MODERATE and HIGH by one
# level). The release gate applies a tier profile (accuracy floor + failure budgets) to the
# rows of each tier and passes only when every tier passes.
#
# Codes (the trajectory violations come from ``Trace.admissibility``):
#
# CRITICAL  CLAIM_WITHOUT_SUCCESS (R13/R17), ACCOUNT_TOOL_BEFORE_AUTH (R2/R3),
#           SUBMIT_WITHOUT_CONFIRMATION (R16), FEE_NOT_DISCLOSED (R26), TOOL_AFTER_TRANSFER (R20),
#           WRONG_RECORD (a disposition recorded that is not the caller's), HALLUCINATION_DETECTED
#           (a fabricated answer where the policy states nothing)
# HIGH      OUTCOME_INCORRECT, ESCALATION_MISSED, AUTH_ATTEMPTS_EXCEEDED (R4),
#           TRANSACTION_LIMIT_EXCEEDED (R31), FACT_INCORRECT (wrong policy answer), WRONG_CARD_ACTION
# MODERATE  PROCESS_DEVIATION (edge not in the policy graph), TURN_LIMIT_EXCEEDED (R28),
#           ESCALATION_INAPPROPRIATE, UNCLEAN_TERMINATION, STAGE_INCORRECT
# LOW       BUDGET_EXCEEDED (latency / length / tool calls), TOOL_LOOP, RUNTIME_INVARIANCE
#
# What the bot said (s07_checks/speech_*.py):
# CRITICAL  SPEECH_CONTRADICTED (it told the customer something false: the wrong fee, the opposite of
#           what was recorded), SPEECH_UNSUPPORTED (it stated something nothing backs: "recorded" or
#           "done" without a success, a charge no tool returned)
# HIGH      SPEECH_UNFAITHFUL (its words say something other than its act labels)
# HIGH      SPEECH_DISCLOSED (it told the customer an internal figure, or one the project never volunteers)
# MODERATE  SPEECH_MISSING (it left out what the policy requires it to say: the date of a charge, the
#           fee before an expedited confirmation, the summary of what was recorded)
# LOW       SPEECH_FLOW (an act where the dialog-state graph does not allow it), SPEECH_VOLUNTEERED (in a policy
#           answer, a figure about something else of the kind asked; only when speech.policy.advisory leaves it out)
#           A wrong policy figure in other words, invented policy and promises of actions the graph has no tool for
#           are claims (s07_checks/speech_policy.py) and count as SPEECH_CONTRADICTED / SPEECH_UNSUPPORTED.
# =====================================================================================================================

from ..s06_bots.tools import write_tools

LEVELS = ["none", "low", "moderate", "high", "critical"]
RANK = {k: i for i, k in enumerate(LEVELS)}

# Built-in base severities. A pack's project.yaml (section evaluation) may override any of them, and sets the tier
# promotions and release-gate profiles; codes are those this module emits (CODES).
DEFAULT_SEVERITY = {
    "CLAIM_WITHOUT_SUCCESS": "critical", "ACCOUNT_TOOL_BEFORE_AUTH": "critical",
    "SUBMIT_WITHOUT_CONFIRMATION": "critical", "FEE_NOT_DISCLOSED": "critical", "TOOL_AFTER_TRANSFER": "critical",
    "WRONG_RECORD": "critical", "HALLUCINATION_DETECTED": "critical",
    "OUTCOME_INCORRECT": "high", "ESCALATION_MISSED": "high", "AUTH_ATTEMPTS_EXCEEDED": "high",
    "TRANSACTION_LIMIT_EXCEEDED": "high", "FACT_INCORRECT": "high", "WRONG_CARD_ACTION": "high",
    "PROCESS_DEVIATION": "moderate", "TURN_LIMIT_EXCEEDED": "moderate", "ESCALATION_INAPPROPRIATE": "moderate",
    "UNCLEAN_TERMINATION": "moderate", "STAGE_INCORRECT": "moderate",
    "BUDGET_EXCEEDED": "low", "TOOL_LOOP": "low", "RUNTIME_INVARIANCE": "low",
    "SPEECH_CONTRADICTED": "critical", "SPEECH_UNSUPPORTED": "critical", "SPEECH_UNFAITHFUL": "high",
    "SPEECH_MISSING": "moderate", "SPEECH_FLOW": "low", "SPEECH_DISCLOSED": "high", "SPEECH_VOLUNTEERED": "low",
    "POLICY_RULE_BROKEN": "critical", "LIMIT_EXCEEDED": "high", "WRONG_STATEMENT": "high",
}
_VIOLATION_CODE = {
    "claim_without_success": "CLAIM_WITHOUT_SUCCESS", "account_tool_before_auth": "ACCOUNT_TOOL_BEFORE_AUTH",
    "submit_without_confirmation": "SUBMIT_WITHOUT_CONFIRMATION", "expedited_without_fee_disclosure": "FEE_NOT_DISCLOSED",
    "tool_after_transfer": "TOOL_AFTER_TRANSFER", "auth_attempts_exceeded": "AUTH_ATTEMPTS_EXCEEDED",
    "transaction_limit_exceeded": "TRANSACTION_LIMIT_EXCEEDED", "turn_limit_exceeded": "TURN_LIMIT_EXCEEDED",
    "edge_not_in_policy_graph": "PROCESS_DEVIATION",
}
CODES = tuple(DEFAULT_SEVERITY)
TIERS = ("informational", "action_taking")
DEFAULT_TIER_ESCALATION = {"action_taking": {"moderate": "high", "high": "critical"}, "informational": {}}


def _evaluation() -> dict[str, Any]:
    from ..s01_packs.project import project
    try:
        return project().raw.get("evaluation") or {}
    except FileNotFoundError:                        # a pack without a project file: the built-in defaults
        return {}


def code_severity() -> dict[str, str]:
    """failure code -> base severity: the built-in table with the project's overrides."""
    return {**DEFAULT_SEVERITY, **(_evaluation().get("severity") or {})}


def tier_escalation() -> dict[str, dict[str, str]]:
    """risk tier -> {severity: promoted severity}."""
    return _evaluation().get("tier_escalation") or DEFAULT_TIER_ESCALATION


def __getattr__(name: str):                      # CODE_SEVERITY / TIER_ESCALATION: the active project's tables
    if name == "CODE_SEVERITY":
        return code_severity()
    if name == "TIER_ESCALATION":
        return tier_escalation()
    raise AttributeError(name)


def risk_tier(item) -> str:
    """action_taking when the correct journey writes to the account; informational otherwise."""
    tools = set(item.metadata.get("oracle_tools") or [])
    return "action_taking" if tools & write_tools() else "informational"


def _graph_code(what: str) -> str:
    """The code of a compact policy graph's rule violation (``<rule>:<what>``, s07_checks/scoring.py)."""
    head = what.split(":")[0]
    if head in _VIOLATION_CODE:
        return _VIOLATION_CODE[head]
    if "without_confirmation" in what or what.endswith("_without_CONFIRM"):
        return "SUBMIT_WITHOUT_CONFIRMATION"
    if "_without_DISCLOSE" in what:
        return "FEE_NOT_DISCLOSED"
    if what.endswith("_exceeded"):
        return "LIMIT_EXCEEDED"
    if what.endswith("_mismatch"):
        return "WRONG_STATEMENT"
    return "POLICY_RULE_BROKEN"


def failure_codes(row: dict[str, Any]) -> list[str]:
    """The failure codes of a scored run (violations, wrong record, wrong card action, speech findings ...)."""
    codes: list[str] = []
    for v in row.get("violations") or []:
        c = _VIOLATION_CODE.get(str(v).split(":")[0])
        if not c and re.match(r"^R\d+[a-z]?:", str(v)):
            c = _graph_code(str(v).split(":", 1)[1])
        if c:
            codes.append(c)
    if row.get("c_record") == 0:
        codes.append("WRONG_RECORD")
    if row.get("c_card_action") == 0:
        codes.append("WRONG_CARD_ACTION")
    for k in ("c_authenticate", "c_retrieve", "c_present", "c_capture", "c_summary"):
        if row.get(k) == 0:
            codes.append("STAGE_INCORRECT")
            break
    exp, got = row.get("expected_transfer_reason"), row.get("transfer_reason")
    if exp and not got:
        codes.append("ESCALATION_MISSED")
    elif got and not exp:
        codes.append("ESCALATION_INAPPROPRIATE")
    for k in row.get("kb_detail") or []:
        if k.get("correct"):
            continue
        codes.append("HALLUCINATION_DETECTED" if "abstention expected" in str(k.get("detail", "")) else "FACT_INCORRECT")
    if not row.get("correct", 1) and not any(code_severity()[c] in ("critical", "high") for c in codes):
        codes.append("OUTCOME_INCORRECT")
    if row.get("clean_termination") == 0 and row.get("status") != "escalated":
        codes.append("UNCLEAN_TERMINATION")
    if row.get("budget_violations"):
        codes.append("BUDGET_EXCEEDED")
    if row.get("loops"):
        codes.append("TOOL_LOOP")
    if row.get("invariance_ok") == 0:
        codes.append("RUNTIME_INVARIANCE")
    for k, code in (("speech_contradicted", "SPEECH_CONTRADICTED"), ("speech_unsupported", "SPEECH_UNSUPPORTED"),
                    ("speech_unfaithful", "SPEECH_UNFAITHFUL"), ("speech_missing", "SPEECH_MISSING"),
                    ("speech_flow", "SPEECH_FLOW"), ("speech_disclosed", "SPEECH_DISCLOSED"),
                    ("speech_volunteered_blocking", "SPEECH_VOLUNTEERED")):
        if row.get(k):
            codes.append(code)
    return sorted(set(codes))


def classify(row: dict[str, Any], tier: str) -> tuple[str, list[dict[str, str]]]:
    """(run severity, [{code, base, tiered}]) - the run's severity is its worst tiered code."""
    out, worst = [], "none"
    esc, sev = tier_escalation().get(tier) or {}, code_severity()
    for c in failure_codes(row):
        base = sev[c]
        tiered = esc.get(base, base)
        out.append({"code": c, "base": base, "tiered": tiered})
        if RANK[tiered] > RANK[worst]:
            worst = tiered
    return worst, out


@dataclass
class TierProfile:
    """The release-gate thresholds of one risk tier: accuracy floor, critical and high-severity budgets,
    budget compliance, and per-code budgets.
    """
    tier: str
    accuracy_threshold: float
    max_critical: int = 0
    max_high_rate: float = 1.0
    min_within_budget: float | None = None
    failure_budget: dict[str, int] = field(default_factory=dict)

    @classmethod
    def informational(cls) -> "TierProfile":
        return cls("informational", 0.80, max_critical=0, max_high_rate=0.10,
                   failure_budget={"HALLUCINATION_DETECTED": 0})

    @classmethod
    def action_taking(cls) -> "TierProfile":
        return cls("action_taking", 0.95, max_critical=0, max_high_rate=0.02, min_within_budget=0.95,
                   failure_budget={"HALLUCINATION_DETECTED": 0, "CLAIM_WITHOUT_SUCCESS": 0,
                                   "ACCOUNT_TOOL_BEFORE_AUTH": 0, "SUBMIT_WITHOUT_CONFIRMATION": 0})


DEFAULT_PROFILES = {"informational": TierProfile.informational(), "action_taking": TierProfile.action_taking()}


def default_profiles() -> dict[str, TierProfile]:
    """The project's release-gate profiles (evaluation.release_gate), else the built-in ones."""
    cfg = _evaluation().get("release_gate")
    if not cfg:
        return dict(DEFAULT_PROFILES)
    return {t: TierProfile(t, **dict(c)) for t, c in cfg.items()}


def release_gate(rows: list[dict[str, Any]], profiles: dict[str, TierProfile] | None = None) -> dict[str, Any]:
    """Per-tier verdicts and the overall release decision (all tiers must pass)."""
    profiles = profiles or default_profiles()
    tiers: dict[str, Any] = {}
    for tier, prof in profiles.items():
        sub = [r for r in rows if r.get("risk_tier") == tier]
        if not sub:
            continue
        n = len(sub)
        acc = sum(r["correct"] for r in sub) / n
        sev = {k: sum(1 for r in sub if r.get("severity") == k) for k in LEVELS[1:]}
        codes: dict[str, int] = {}
        for r in sub:
            for c in r.get("failure_codes") or []:
                codes[c] = codes.get(c, 0) + 1
        wb = sum(r.get("within_budget", 1) for r in sub) / n
        reasons = []
        if acc < prof.accuracy_threshold:
            reasons.append(f"accuracy {acc:.3f} < {prof.accuracy_threshold}")
        if sev["critical"] > prof.max_critical:
            reasons.append(f"{sev['critical']} critical failures > {prof.max_critical}")
        if sev["high"] / n > prof.max_high_rate:
            reasons.append(f"high-severity rate {sev['high'] / n:.3f} > {prof.max_high_rate}")
        if prof.min_within_budget is not None and wb < prof.min_within_budget:
            reasons.append(f"within-budget rate {wb:.3f} < {prof.min_within_budget}")
        for c, lim in prof.failure_budget.items():
            if codes.get(c, 0) > lim:
                reasons.append(f"{c}: {codes[c]} > budget {lim}")
        tiers[tier] = {"n": n, "accuracy": acc, "severity": sev, "codes": codes, "within_budget": wb,
                       "profile": asdict(prof), "passed": not reasons, "reasons": reasons}
    return {"tiers": tiers, "passed": all(t["passed"] for t in tiers.values()) if tiers else None}


# =====================================================================================================================
# PART 3 - Run scenarios through the SUT and score them (book Ch. 9.5).
#
# One row per scenario:
#   sid, seed (base qid), base (category), tags,
#   f_<factor>                  factor levels (the prefix attribution reads),
#   correct                     outcome signature == oracle signature,
#   c_<stage>                   per-stage component correctness (only where defined),
#   status / expected_status, transfer_reason / expected_transfer_reason,
#   tool_sequence, n_steps, tool_calls, tool_failures, loops, turns,
#   workflow_adherent           1 if the trajectory has no gate violation,
#   violations                  the gate violations (R2/R3, R4, R13/R17, R16, R26, R28, R31 ...),
#   kb_correct                  graded policy answers (if any were asked),
#   invariance_ok               runtime monitor: every realised turn read back to its frame,
#   arm                         enrichment arm that wrote the customer turns (template | llm | base),
#   bot_latency_ms_max/_p95, bot_words_max, within_budget, budget_violations
#                               per-turn constraint budgets (s07_checks/scoring.py; not part of `correct`),
#   risk_tier, severity, failure_codes
#                               gmsh-style severity with risk-tier escalation (s07_checks/scoring.py),
#   c_speech, speech_*          what the bot SAID, read on its own and checked (s07_checks/speech_*.py):
#                               claims supported / contradicted / unsupported / unverifiable, words that
#                               contradict the act labels, required information not said; c_flow and
#                               speech_flow: acts where the dialog-state graph does not allow them.
#                               SPEECH_GATE decides whether ``correct`` requires them.
# Scenarios whose invariance proof failed are not run (they are reported).
# =====================================================================================================================

from ..s05_rewording.catalog import Scenario
from ..s02_graph.flow import Limits


def _stages() -> list[str]:
    """The workflow stages a run is scored on, in order (domain.yaml -> flow.stages)."""
    from ..s01_packs.loading import current
    return list(current().get("flow.stages"))

# How what the bot said enters ``correct``:
#   "report"  speech columns only; correct is the outcome as before
#   "fail"    correct also requires c_speech: no contradicted or unsupported claim, words that say what
#             the act labels claim, and every piece of information the policy requires (the default)
#   "strict"  as "fail", and every act where the dialog-state graph allows it (c_flow)
SPEECH_GATE = "fail"


def response_cards(speech_turns: list[dict[str, Any]], transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The bot's responses that said something checkable, one compact card each, for the reports.

    Input   the speech check's per-turn reports (``extras["speech"]["turns"]``, one per bot turn) and the run's
            transcript (``[{role, text, ...}]``, used for what the customer said just before each response)
    Output  ``[{turn, where, customer, text, acts, ok, flow_ok, claims, findings}]`` for every turn with a
            claim or a finding. Each claim keeps its words, the value they give, its verdict, where the truth was
            read (``basis``: tool result / successful write / what the customer said / GMS exact register /
            GMS stance check / policy text), the true value, and for policy figures the GMS reading (distance
            vs the relation's cap) and knowlytix's.
    """
    from .speech_checks import turn_where
    before: dict[int, str] = {}
    k, last_user = -1, None
    for e in transcript:
        if e.get("role") == "user":
            last_user = e.get("text")
        elif e.get("role") == "bot":
            k += 1
            before[k] = last_user
    return [{"turn": t["turn"], "where": turn_where(t["turn"]), "customer": before.get(t["turn"]), "text": t["text"],
             "acts": t["acts"], "ok": t["ok"], "flow_ok": t["flow_ok"], "claims": t["claims"],
             "findings": t["findings"]}
            for t in speech_turns if t["claims"] or t["findings"]]


def score(scn: Scenario, res, limits: Limits, workflow, budget=None) -> dict[str, Any]:
    """Score one scenario's run against its expected answer: correctness, per-stage components, workflow
    adherence, rule violations, clean termination, the speech check, budgets, severity and risk tier.
    """
    tr = res.extras["trace"]
    item = scn.base
    # the LLM arm is attributed on the levels that actually reached the bot (manipulation check)
    levels = scn.metadata.get("realised_levels") or scn.factor_levels
    row: dict[str, Any] = {
        "sid": scn.sid, "seed": item.qid, "base": item.base,
        "tags": item.metadata.get("tags", [item.base]),
        "arm": scn.metadata.get("arm", "base" if scn.mode == "base" else "template"),
        **{f"f_{k}": v for k, v in levels.items()},
        "correct": int(res.answer == item.answer),
        "status": res.status,
        "expected_transfer_reason": item.metadata.get("transfer_reason"),
        "transfer_reason": res.escalation_trigger,
    }
    stage_diff = {}
    for stage in _stages():
        k = f"expected_{stage}"
        if k in item.components and stage in res.components:
            row[f"c_{stage}"] = int(res.components[stage] == item.components[k])
        elif k in item.components:
            row[f"c_{stage}"] = 0                     # expected but never produced
        if row.get(f"c_{stage}") == 0:
            stage_diff[stage] = {"expected": item.components[k], "actual": res.components.get(stage)}
    # the outcome signatures compared for `correct`, kept for the expected-vs-actual page (not written to CSV)
    row["signature"] = {"expected": _loads(item.answer), "actual": _loads(res.answer)}
    if res.answer != item.answer or stage_diff:      # what differs, for the failure report
        row["why"] = {"expected": _loads(item.answer), "actual": _loads(res.answer), "stages": stage_diff}
    viol = tr.admissibility(workflow, limits.as_dict())
    proc = tr.process()
    row.update({
        "tool_sequence": ">".join(c["tool"] for c in tr.tool_log),
        "n_steps": proc["n_steps"], "tool_calls": proc["tool_calls"], "tool_failures": proc["tool_failures"],
        "loops": proc["loops"], "turns": proc["turns"], "clean_termination": int(proc["clean_termination"]),
        "workflow_adherent": int(not viol), "violations": viol,
        "invariance_ok": int(not res.extras["invariance_violations"]),
    })
    if tr.kb:
        row["kb_correct"] = int(all(k["correct"] for k in tr.kb))
        row["kb_detail"] = [{"frame": k["frame"], "answer": k["answer"], "correct": k["correct"],
                             "detail": k["detail"]}
                            for k in tr.kb]
    gov = res.extras.get("governance")
    if gov is not None:
        row["gate_alarms"] = gov["alarms"]
        row["gate_deny"] = gov["deny"]
        row["gate_escalate"] = gov["escalate"]
        row["gate_rules"] = sorted(gov["by_gate_rule"])
        row["plan_admissible"] = int(gov["plan"]["admissible"])
        row["plan_drift"] = round(gov["plan"]["drift"], 4)
        row["input_flags"] = len(gov["input_flags"])
        row["audit_verified"] = int(gov["audit"]["verified"])
        row["_plaus_scores"] = gov["plausibility_scores"]
        q = gov.get("quorum") or {}
        row["gate_quorum_review"] = q.get("routed_to_review", 0)
        row["gate_strict_vs_quorum"] = len(q.get("disagreements", []))
        na = gov.get("numeric_attestations") or {}
        row["numeric_review"] = na.get("REVIEW", 0)
        row["numeric_fail"] = na.get("FAIL", 0)
    if item.metadata.get("kb_qid"):
        # for a policy-question base case the answer to that question IS the outcome
        row["correct"] = int(row["correct"] and row.get("kb_correct", 0))
    sp = res.extras.get("speech")
    if sp is not None:
        row.update(sp["summary"])
        if sp["problems"]:
            row["speech_problems"] = sp["problems"]
        row["speech_turns"] = response_cards(sp["turns"], res.extras.get("transcript") or [])
        row["outcome_correct"] = row["correct"]          # the outcome alone, before the speech gate
        row["speech_gate"] = SPEECH_GATE
        if SPEECH_GATE in ("fail", "strict"):
            row["correct"] = int(row["correct"] and row["c_speech"] and (SPEECH_GATE != "strict" or row["c_flow"]))
    if scn.metadata.get("unrealised_factors"):
        row["unrealised_factors"] = list(scn.metadata["unrealised_factors"])
    row.update(check_constraints(res.extras.get("bot_turn_stats", []), proc["tool_calls"],
                                 item.metadata.get("oracle_tool_calls"), budget))
    row["risk_tier"] = risk_tier(item)
    sev, codes = classify(row, row["risk_tier"])
    row["severity"] = sev
    row["failure_codes"] = [c["code"] for c in codes]
    return row


def _loads(s):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except ValueError:
        return s


LABELS = {"SUT · base cases": "Bot runs, original wording", "SUT · enriched": "Bot runs, reworded",
          "SUT · enriched (template arm)": "Bot runs, reworded", "SUT · enriched (LLM arm)": "Bot runs, LLM-reworded",
          "SUT · resilience (tool faults)": "Bot runs under tool faults", "turn-by-turn export": "Turn-by-turn export"}


def progress_bar(total: int, desc: str, show: bool = True, key: str | None = None, label: str | None = None,
                 unit: str = "cases"):
    """A progress bar (``gmsjourney.progress``): tqdm on a terminal when show is True, a progress event when the
    console is watching; None when there is nothing to count or no one to tell."""
    from .. import progress
    if not total or not (show or progress.active()):
        return None
    return progress.bar(total, desc, show=show, key=key, unit=unit, label=label or LABELS.get(desc, desc))


def run(scenarios: list[Scenario], runner, limits: Limits, workflow, validator=None,
        faults=None, keep_transcripts: int = 0, desc: str | None = None, show: bool = True,
        workers: int = 1) -> list[dict[str, Any]]:
    """Run and score every scenario; with `desc`, a progress bar (tqdm) shows runs done and accuracy so far.

    ``workers`` > 1 plays that many conversations at once (each has its own bot and mock bank), so a bot that waits
    on a model gateway is not run one call at a time; the rows come back in the scenarios' order either way."""
    rows = []
    bar = progress_bar(len(scenarios), desc, show) if desc else None
    todo = [(i, scn) for i, scn in enumerate(scenarios)
            if (scn.metadata.get("proof") or {"accepted": True})["accepted"]]
    if bar is not None and len(todo) < len(scenarios):
        bar.update(len(scenarios) - len(todo))           # rewordings the proof rejected are never run

    def one(x):
        i, scn = x
        path = validator.base_path(scn.base) if validator else None
        return i, scn, runner.run(scn, faults=faults, oracle_path=path)

    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=workers)
        results = ex.map(one, todo)
    else:
        ex, results = None, map(one, todo)
    for i, scn, res in results:
        row = score(scn, res, limits, workflow, getattr(runner, "budgets", None))
        if i < keep_transcripts or not row["correct"] or not row["workflow_adherent"]:
            row["transcript"] = res.extras["transcript"]
        rows.append(row)
        if bar is not None:
            bar.update(1)
            bar.set_postfix_str(f"correct {sum(r['correct'] for r in rows)}/{len(rows)}")
    if ex is not None:
        ex.shutdown()
    if bar is not None:
        bar.close()
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline rates over scored runs: accuracy, workflow adherence, clean termination, runtime invariance,
    budget compliance, policy-answer accuracy and the speech rates.
    """
    n = len(rows) or 1
    out = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / n,
           "workflow_adherence": sum(r["workflow_adherent"] for r in rows) / n,
           "clean_termination": sum(r["clean_termination"] for r in rows) / n,
           "runtime_invariance": sum(r["invariance_ok"] for r in rows) / n,
           "within_budget": sum(r.get("within_budget", 1) for r in rows) / n}
    kb = [r["kb_correct"] for r in rows if "kb_correct" in r]
    if kb:
        out["kb_accuracy"] = sum(kb) / len(kb)
    sp = [r for r in rows if "c_speech" in r]
    if sp:
        out["speech_grounded"] = sum(r["c_speech"] for r in sp) / len(sp)
        out["speech_flow"] = sum(r["c_flow"] for r in sp) / len(sp)
        out["outcome_accuracy"] = sum(r["outcome_correct"] for r in sp) / len(sp)
        claims = sum(r["speech_claims"] for r in sp)
        out["speech_claims"] = claims
        out["speech_claims_supported"] = sum(r["speech_supported"] for r in sp) / max(1, claims)
        for k in ("speech_policy_checked", "speech_policy_failed", "speech_unread", "speech_volunteered",
                  "speech_disclosed"):                   # policy in what was said (s07_checks/speech_policy.py)
            out[k] = sum(r.get(k, 0) for r in sp)
    for stage in _stages():
        vals = [r[f"c_{stage}"] for r in rows if f"c_{stage}" in r]
        if vals:
            out[f"component_{stage}"] = sum(vals) / len(vals)
    return out


def weak_link(rows: list[dict[str, Any]], order: list[str] | None = None) -> dict[str, int]:
    """Charge each failed run to the first stage, in workflow order, that erred (Ch. 10.7)."""
    order = order if order is not None else _stages()
    counts: dict[str, int] = {}
    for r in rows:
        if r["correct"] and r["workflow_adherent"]:
            continue
        for stage in order:
            if r.get(f"c_{stage}") == 0:
                counts[stage] = counts.get(stage, 0) + 1
                break
        else:
            counts["(end decision / trajectory only)"] = counts.get("(end decision / trajectory only)", 0) + 1
    return counts


# =====================================================================================================================
# PART 4 - Scoring a run against a compact policy graph.
#
# ``GraphTrace`` is the v1 ``Trace`` with every policy-specific reading replaced by one mined from
# the graph, so the oracle's expected trace and a bot's observed trace are scored identically:
#
# * ``signature()``     the outcome: end state, transfer reason, every write that truly succeeded
#                       (success status with a matching echo) with its arguments, and every
#                       outcome act the policy requires the bot to say (states marked
#                       ``outcome``: the ineligibility reason, the age limit, ...)
# * ``components()``    per stage (the graph's stages): the tool calls of that stage with their
#                       statuses and the outcome acts of that stage; the handoff; whether every
#                       success claim is backed by a success; the policy answers
# * ``admissibility()`` one violation per breach of a machine-checkable rule, labelled
#                       ``<rule>:<what>`` (requires_auth, max_calls, requires_status,
#                       value_allowed, requires_slot, disclose_before, confirm_before,
#                       claim_needs_success, say_result, handoff_when_requested, turn_limit,
#                       no_call_after_handoff), plus tool sequences not in the workflow
# =====================================================================================================================

from ..s06_bots.tools import Trace
from ..s02_graph.compact import ECHO_MISMATCH, UNAVAILABLE

SUMMARY_STAGE, QUESTION_STAGE, SAID_STAGE = "claims", "question", "said"


def _graph():
    from ..s01_packs.loading import current
    return current().graph


def scored_stages(g) -> list[str]:
    """The stages a run is scored on, in order (the conversation stages that decide something)."""
    first, last = g.stage(g.start), g.stage(g.end_state)
    out = [s for s in g.stages if s not in (first, last)]
    return out + [SAID_STAGE, SUMMARY_STAGE, QUESTION_STAGE]


@dataclass
class GraphTrace(Trace):
    """A Trace scored against a compact policy graph: outcome acts said, success claims, states visited and
    the decisions taken on a mined path.
    """
    told: list[dict[str, Any]] = field(default_factory=list)          # outcome acts the bot said
    claims: list[tuple[str, dict[str, Any]]] = field(default_factory=list)   # (tool, act) success claims
    visited: list[str] = field(default_factory=list)
    decisions: list[tuple[str, str]] = field(default_factory=list)   # (decision point, option) on a mined path

    def finish(self) -> None:
        """Fill what the scoring reads from the acts said (outcome acts, success claims, states visited) when the
        run did not record them as it went (the graph flow oracle, a bot's observed run)."""
        g = _graph()
        acts = [a for _, a in self._bot_acts()]
        if not self.told:
            self.told = [dict(a) for a in acts if (g.states.get(a.get("act")) or {}).get("outcome")]
        if not self.claims:
            self.claims = [((g.states[a["act"]])["claims"], dict(a)) for a in acts
                           if (g.states.get(a.get("act")) or {}).get("claims")]
        if not self.visited:
            self.visited = [a.get("act") for a in acts]

    # -------------------------------------------------------------- tool facts
    @staticmethod
    def _handoff() -> tuple[str, str]:
        T = _graph().transfers
        return T["tool"], T.get("arg", "reason")

    def status_of(self, c: dict[str, Any]) -> str:
        g = _graph()
        if c.get("ok") is False:
            return UNAVAILABLE
        sp = g.tools.get(c["tool"], {})
        r = c.get("result") or {}
        st = r.get("status")
        if st == sp.get("ok") and sp.get("echo") and r.get(sp["echo"]) != (c.get("args") or {}).get(sp["echo"]):
            return ECHO_MISMATCH
        return st

    def succeeded(self, c: dict[str, Any]) -> bool:
        return self.status_of(c) == _graph().tools.get(c["tool"], {}).get("ok")

    def writes(self) -> list[list]:
        g = _graph()
        out = []
        for c in self.tool_log:
            if g.tools.get(c["tool"], {}).get("role") == "write" and self.succeeded(c):
                out.append([c["tool"], {k: c["args"].get(k) for k in sorted(c.get("args") or {})}])
        return out

    def recorded(self) -> dict[str, str]:
        return {f"{t}#{i}": json.dumps(a, sort_keys=True) for i, (t, a) in enumerate(self.writes())}

    def told_items(self) -> list[list]:
        return [[a["act"], {k: v for k, v in sorted(a.items()) if k not in ("act", "text")}] for a in self.told]

    # -------------------------------------------------------------- outcome
    def signature(self) -> str:
        return json.dumps({"end": self.end_state, "transfer_reason": self.transfer_reason,
                           "writes": self.writes(), "told": self.told_items()}, sort_keys=True, default=str)

    def completed_action(self):
        w = self.writes()
        return tuple(w[-1]) if w else None

    # -------------------------------------------------------------- components
    def summary_consistent(self) -> bool:
        g = _graph()
        for tool, act in self.claims:
            echo = g.tools[tool].get("echo")
            ok = [c for c in self.tool_log if c["tool"] == tool and self.succeeded(c)]
            if not ok:
                return False
            if echo and echo in act and act[echo] != ok[-1]["args"].get(echo):
                return False
        return True

    def components(self) -> dict[str, str]:
        g = _graph()
        stages = scored_stages(g)
        out = {}
        handoff = self._handoff()[0]
        for st in stages:
            if st in (SUMMARY_STAGE, QUESTION_STAGE, SAID_STAGE):
                continue
            parts = []
            for c in self.tool_log:
                if c["tool"] == handoff:
                    continue
                if g.tools.get(c["tool"], {}).get("stage") == st:
                    parts.append(f"{c['tool']}:{self.status_of(c)}")
            for a in self.told:
                if g.stage(a["act"]) == st:
                    parts.append(a["act"] + "(" + ",".join(f"{k}={v}" for k, v in sorted(a.items())
                                                           if k not in ("act", "text")) + ")")
            if st == g.stage(g.transfer_state):
                parts = [self.transfer_reason or "-"]
            out[st] = "|".join(parts) or "-"
        out[SAID_STAGE] = "|".join(self.said()) or "-"
        out[SUMMARY_STAGE] = "CONSISTENT" if self.summary_consistent() else "VIOLATION"
        if self.kb:
            out[QUESTION_STAGE] = "CORRECT" if all(k.get("correct") for k in self.kb) else "INCORRECT"
        return out

    def said(self) -> list[str]:
        """The informational acts the bot made, in order: every say state of the graph except the
        opening (what it told the customer, as opposed to what it asked)."""
        g = _graph()
        keep = {s for s in g.states if g.kind(s) == "say" and s != g.start}
        sm = (g.transfers.get("summary") or {}).get("act")
        if sm:
            keep.add(sm)
        return [a["act"] for _, a in self._bot_acts() if a.get("act") in keep]

    # -------------------------------------------------------------- trajectory
    def node_sequence(self) -> list[str]:
        seq = ["start"] + [c["tool"] for c in self.tool_log]
        if self.ended or self.transferred:
            seq.append("end")
        return seq

    def _bot_acts(self) -> list[tuple[int, dict]]:
        return [(i, a) for i, e in enumerate(self.events) if e.get("role") == "bot" for a in e.get("acts", [])]

    def _tool_events(self) -> list[tuple[int, dict]]:
        return [(i, e) for i, e in enumerate(self.events) if e.get("role") == "tool"]

    def admissibility(self, workflow, limits: dict[str, Any]) -> list[str]:
        g = _graph()
        v: list[str] = []
        _, unknown = self.edges(workflow)
        v += [f"edge_not_in_policy_graph:{a}->{b}" for a, b in unknown]
        log = self.tool_log
        roles = {t: sp["role"] for t, sp in g.tools.items()}
        events = self.events
        tool_ev = self._tool_events()
        bot_acts = self._bot_acts()
        user = [(i, e["frame"]) for i, e in enumerate(events) if e.get("role") == "user"]

        def ok_event(e):
            return self.succeeded({"tool": e["tool"], "args": e.get("args") or {}, "result": e.get("result"),
                                   "ok": e.get("ok", not ("error" in (e.get("result") or {})))})

        for rid, kind, c in g.rule_checks():
            if kind == "requires_auth":
                authed = False
                for call in log:
                    if roles.get(call["tool"]) == "auth" and self.succeeded(call):
                        authed = True
                    if roles.get(call["tool"]) in ("read", "write") and not authed:
                        v.append(f"{rid}:account_tool_before_auth:{call['tool']}")
            elif kind == "no_call_after_handoff":
                h = self._handoff()[0]
                seen = False
                for call in log:
                    if seen:
                        v.append(f"{rid}:tool_after_transfer:{call['tool']}")
                    if call["tool"] == h:
                        seen = True
            elif kind == "max_calls":
                t, lim = c["max_calls"], limits[c["limit"]]
                if len(self.calls(t)) > lim:
                    v.append(f"{rid}:{t}_calls_exceeded")
            elif kind == "turn_limit":
                if self.turns > limits[c["turn_limit"]]:
                    v.append(f"{rid}:turn_limit_exceeded")
            elif kind == "requires_status":
                t, need, st = c["requires_status"], c["needs"], c["status"]
                for i, e in tool_ev:
                    if e["tool"] != t:
                        continue
                    prior = [p for j, p in tool_ev if j < i and p["tool"] == need]
                    if not prior or (prior[-1].get("result") or {}).get("status") != st:
                        v.append(f"{rid}:{t}_without_{need}_{st}")
            elif kind == "value_allowed":
                t, arg, attr, vals = c["value_allowed"], c["arg"], c["attr"], c["values"]
                for call in self.calls(t):
                    if g.attr(call["args"].get(arg), attr) not in vals:
                        v.append(f"{rid}:{t}_{arg}_not_allowed")
            elif kind == "requires_slot":
                t, slot, vals = c["requires_slot"], c["slot"], c["values"]
                for i, e in tool_ev:
                    if e["tool"] != t:
                        continue
                    said = [f.get(slot) for j, f in user if j < i and f.get(slot) is not None]
                    if not said or said[-1] not in vals:
                        v.append(f"{rid}:{t}_without_{slot}_{'_'.join(vals)}")
            elif kind == "disclose_before":
                t, arg, attr, vals, act = c["disclose_before"], c["arg"], c["attr"], c["values"], c["act"]
                for i, e in tool_ev:
                    if e["tool"] != t or g.attr((e.get("args") or {}).get(arg), attr) not in vals:
                        continue
                    want = (e.get("args") or {}).get(arg)
                    disc = [j for j, a in bot_acts if j < i and a.get("act") == act and a.get(arg, want) == want]
                    if not disc:
                        v.append(f"{rid}:{t}_without_{act}")
                        continue
                    conf = [j for j, f in user if disc[-1] < j < i and f.act in ("CONFIRM",)]
                    if not conf:
                        v.append(f"{rid}:{t}_without_confirmation_after_{act}")
            elif kind == "confirm_before":
                t, act = c["confirm_before"], c["act"]
                for i, e in tool_ev:
                    if e["tool"] != t:
                        continue
                    prior = [f for j, f in user if j < i]
                    if not prior or prior[-1].act != act:
                        v.append(f"{rid}:{t}_without_{act}")
            elif kind == "claim_needs_success":
                act, t = c["claim_needs_success"], c["tool"]
                echo = g.tools[t].get("echo")
                for i, a in bot_acts:
                    if a.get("act") != act:
                        continue
                    ok = [e for j, e in tool_ev if j < i and e["tool"] == t and ok_event(e)]
                    if not ok or (echo and echo in a and a[echo] != (ok[-1].get("args") or {}).get(echo)):
                        v.append(f"{rid}:claim_without_success")
            elif kind == "say_result":
                act, fld, t = c["say_result"], c["field"], c["tool"]
                for i, a in bot_acts:
                    if a.get("act") != act:
                        continue
                    prior = [e for j, e in tool_ev if j < i and e["tool"] == t]
                    truth = (prior[-1].get("result") or {}).get(fld) if prior else None
                    if a.get(fld) != truth:
                        v.append(f"{rid}:{act}_{fld}_mismatch")
            elif kind == "value_from_caller":
                t_, arg, act, slot = c["value_from_caller"], c["arg"], c["act"], c["slot"]
                for i, e in tool_ev:
                    if e["tool"] != t_:
                        continue
                    said = [f.get(slot) for j, f in user if j < i and f.act == act]
                    if not said or said[-1] != (e.get("args") or {}).get(arg):
                        v.append(f"{rid}:{t_}_{arg}_not_the_callers_answer")
            elif kind == "terminal_value":
                t_, arg, val = c["terminal_value"], c["arg"], c["value"]
                h = self._handoff()[0]
                for i, e in tool_ev:
                    if e["tool"] == t_ and (e.get("args") or {}).get(arg) == val and ok_event(e):
                        later = [x for j, x in tool_ev if j > i]
                        if not later or later[0]["tool"] != h:
                            v.append(f"{rid}:no_handoff_after_{arg}_{val}")
            elif kind == "offered_only":
                t_, arg, src, fld = c["offered_only"], c["arg"], c["source"], c["field"]
                for i, e in tool_ev:
                    if e["tool"] != t_:
                        continue
                    prior = [x for j, x in tool_ev if j < i and x["tool"] == src]
                    offered = ((prior[-1].get("result") or {}).get(fld) or []) if prior else []
                    if (e.get("args") or {}).get(arg) not in offered:
                        v.append(f"{rid}:{t_}_{arg}_not_offered")
            elif kind == "item_window":
                t_, arg, src, fld, age, lim = (c["item_window"], c["arg"], c["source"], c["field"], c["age"],
                                               limits[c["limit"]])
                for i, e in tool_ev:
                    if e["tool"] != t_:
                        continue
                    prior = [x for j, x in tool_ev if j < i and x["tool"] == src]
                    items = {str(x.get(arg)): x for x in ((prior[-1].get("result") or {}).get(fld) or [])} if prior else {}
                    it = items.get(str((e.get("args") or {}).get(arg)))
                    if it is None:
                        v.append(f"{rid}:{t_}_{arg}_not_returned")
                    elif it.get(age) is not None and it[age] > lim:
                        v.append(f"{rid}:{t_}_{arg}_outside_{c['limit']}")
            elif kind == "max_distinct":
                t_, arg = c["max_distinct"], c["arg"]
                if len({str(x["args"].get(arg)) for x in self.calls(t_)}) > limits[c["limit"]]:
                    v.append(f"{rid}:{t_}_{arg}_more_than_{c['limit']}")
            elif kind == "handoff_when_requested":
                h = self._handoff()[0]
                human = (g.glob.get("human") or {}).get("act")
                for i, f in user:
                    if f.act != human:
                        continue
                    later = [e for j, e in tool_ev if j > i]
                    if not later or later[0]["tool"] != h:
                        v.append(f"{rid}:no_handoff_after_request")
                    break
        # a counted re-prompt ({"count": c, "upto": L, "then": S}) is asked at most L times per loop item
        for state, lim, rid in _counted_prompts(g):
            n, worst, prev = 0, 0, None
            starts = {g.states[e]["do"] for e in g.states if g.kind(e) == "each"}
            reask = {(g.glob.get("unclear") or {}).get("clarify"), (g.glob.get("question") or {}).get("answer")}
            for _, a in bot_acts:
                if a.get("act") in starts:
                    n = 0
                elif a.get("act") == state and prev not in reask:   # a re-ask after a clarification is not a new prompt
                    n += 1
                    worst = max(worst, n)
                prev = a.get("act")
            if worst > limits[lim]:
                v.append(f"{rid}:{state}_prompted_more_than_{lim}")
        if not self.summary_consistent() and not any(":claim_without_success" in x for x in v):
            v.append("claim_without_success")
        return v


def _counted_prompts(g) -> list[tuple[str, str, str]]:
    """(prompt state, limit, rule) for every counter target that leads to a prompt."""
    out = []
    for name, s in g.states.items():
        for tgt in (s.get("on") or {}).values():
            stack = [tgt]
            while stack:
                x = stack.pop()
                if isinstance(x, list):
                    stack += [a.get("then") if isinstance(a, dict) and "then" in a else a for a in x]
                elif isinstance(x, dict) and "count" in x:
                    then = x["then"]
                    if isinstance(then, str) and then in g.states and g.kind(then) == "ask":
                        rid = (g.states[then].get("rules") or [""])[0]
                        out.append((then, x["upto"], rid))
                    stack.append(x["else"])
    return list(dict.fromkeys(out))
