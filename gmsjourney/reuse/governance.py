"""forgeloop's governance pieces driving our runtime monitor.

* ``forgeloop.gms_backend.GMSPlausibilityGate`` scores workflow transitions on the
  knowlytix store's ``has_enables`` relation (it needs nothing but ``score_triple``); an
  adapter converts its results into the monitor's GateResult and applies the
  safe-harbour rule for escalation actions. (The whole-plan check is ours:
  governance/gates.py GMSPlanGate, on the same store.)
* ``forgeloop.audit.AuditLogger`` (SHA-256 hash chain) seals every decision.
* ``forgeloop.governance.policies`` (PII, prompt injection) - the regex baseline.
"""
from __future__ import annotations

from ..governance.gates import GateDecision, GateResult


class ForgeloopPlausibilityGate:
    name = "gms_plausibility"

    def __init__(self, kx_store, theta: float, safe_tools=frozenset({"transfer_to_human"})):
        from forgeloop.gms_backend import GMSPlausibilityGate
        self.theta = float(theta)
        self.safe_tools = safe_tools
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
