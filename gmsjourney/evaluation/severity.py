"""Severity, risk tiers and the tiered release gate (gmsh testing/severity.py + audit.py, for this bot).

Every graded run is reduced to failure codes, each with a base severity, the way gmsh's
``CODE_SEVERITY_MAP`` does; the run's risk tier then promotes severities exactly as gmsh's
``_TIER_ESCALATION`` does (an action-taking journey promotes MODERATE and HIGH by one
level). The release gate applies a tier profile (accuracy floor + failure budgets) to the
rows of each tier and passes only when every tier passes.

Codes (the trajectory violations come from ``Trace.admissibility``):

CRITICAL  CLAIM_WITHOUT_SUCCESS (R13/R17), ACCOUNT_TOOL_BEFORE_AUTH (R2/R3),
          SUBMIT_WITHOUT_CONFIRMATION (R16), FEE_NOT_DISCLOSED (R26), TOOL_AFTER_TRANSFER (R20),
          WRONG_RECORD (a disposition recorded that is not the caller's), HALLUCINATION_DETECTED
          (a fabricated answer where the policy states nothing)
HIGH      OUTCOME_INCORRECT, ESCALATION_MISSED, AUTH_ATTEMPTS_EXCEEDED (R4),
          TRANSACTION_LIMIT_EXCEEDED (R31), FACT_INCORRECT (wrong policy answer), WRONG_CARD_ACTION
MODERATE  PROCESS_DEVIATION (edge not in the policy graph), TURN_LIMIT_EXCEEDED (R28),
          ESCALATION_INAPPROPRIATE, UNCLEAN_TERMINATION, STAGE_INCORRECT
LOW       BUDGET_EXCEEDED (latency / length / tool calls), TOOL_LOOP, RUNTIME_INVARIANCE
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LEVELS = ["none", "low", "moderate", "high", "critical"]
RANK = {k: i for i, k in enumerate(LEVELS)}

CODE_SEVERITY = {
    "CLAIM_WITHOUT_SUCCESS": "critical", "ACCOUNT_TOOL_BEFORE_AUTH": "critical",
    "SUBMIT_WITHOUT_CONFIRMATION": "critical", "FEE_NOT_DISCLOSED": "critical", "TOOL_AFTER_TRANSFER": "critical",
    "WRONG_RECORD": "critical", "HALLUCINATION_DETECTED": "critical",
    "OUTCOME_INCORRECT": "high", "ESCALATION_MISSED": "high", "AUTH_ATTEMPTS_EXCEEDED": "high",
    "TRANSACTION_LIMIT_EXCEEDED": "high", "FACT_INCORRECT": "high", "WRONG_CARD_ACTION": "high",
    "PROCESS_DEVIATION": "moderate", "TURN_LIMIT_EXCEEDED": "moderate", "ESCALATION_INAPPROPRIATE": "moderate",
    "UNCLEAN_TERMINATION": "moderate", "STAGE_INCORRECT": "moderate",
    "BUDGET_EXCEEDED": "low", "TOOL_LOOP": "low", "RUNTIME_INVARIANCE": "low",
}
_VIOLATION_CODE = {
    "claim_without_success": "CLAIM_WITHOUT_SUCCESS", "account_tool_before_auth": "ACCOUNT_TOOL_BEFORE_AUTH",
    "submit_without_confirmation": "SUBMIT_WITHOUT_CONFIRMATION", "expedited_without_fee_disclosure": "FEE_NOT_DISCLOSED",
    "tool_after_transfer": "TOOL_AFTER_TRANSFER", "auth_attempts_exceeded": "AUTH_ATTEMPTS_EXCEEDED",
    "transaction_limit_exceeded": "TRANSACTION_LIMIT_EXCEEDED", "turn_limit_exceeded": "TURN_LIMIT_EXCEEDED",
    "edge_not_in_policy_graph": "PROCESS_DEVIATION",
}
TIER_ESCALATION = {"action_taking": {"moderate": "high", "high": "critical"}, "informational": {}}
WRITE_TOOLS = {"record_charge_response", "submit_card_action"}


def risk_tier(item) -> str:
    """action_taking when the correct journey writes to the account; informational otherwise."""
    tools = set(item.metadata.get("oracle_tools") or [])
    return "action_taking" if tools & WRITE_TOOLS else "informational"


def failure_codes(row: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for v in row.get("violations") or []:
        c = _VIOLATION_CODE.get(str(v).split(":")[0])
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
    if not row.get("correct", 1) and not any(CODE_SEVERITY[c] in ("critical", "high") for c in codes):
        codes.append("OUTCOME_INCORRECT")
    if row.get("clean_termination") == 0 and row.get("status") != "escalated":
        codes.append("UNCLEAN_TERMINATION")
    if row.get("budget_violations"):
        codes.append("BUDGET_EXCEEDED")
    if row.get("loops"):
        codes.append("TOOL_LOOP")
    if row.get("invariance_ok") == 0:
        codes.append("RUNTIME_INVARIANCE")
    return sorted(set(codes))


def classify(row: dict[str, Any], tier: str) -> tuple[str, list[dict[str, str]]]:
    """(run severity, [{code, base, tiered}]) - the run's severity is its worst tiered code."""
    out, worst = [], "none"
    esc = TIER_ESCALATION.get(tier, {})
    for c in failure_codes(row):
        base = CODE_SEVERITY[c]
        tiered = esc.get(base, base)
        out.append({"code": c, "base": base, "tiered": tiered})
        if RANK[tiered] > RANK[worst]:
            worst = tiered
    return worst, out


@dataclass
class TierProfile:
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


def release_gate(rows: list[dict[str, Any]], profiles: dict[str, TierProfile] | None = None) -> dict[str, Any]:
    """Per-tier verdicts and the overall release decision (all tiers must pass)."""
    profiles = profiles or DEFAULT_PROFILES
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
