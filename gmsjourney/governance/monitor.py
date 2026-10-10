"""Runtime monitoring of one conversation (book Ch. 12.5; forgeloop GovernanceHarness re-done
for a voice bot whose tool client we wrap).

``RuntimeMonitor`` keeps the observable state of a conversation (tool calls and
results, the caller's turns as read by the FrameReader, the bot's utterances) and
asks every gate before each tool call executes. ``GovernedTools`` is the drop-in
tool client: in ``monitor`` mode it only records (the test outcome is unchanged), in
``enforce`` mode a DENY blocks the call and the bot receives a tool error. Every
decision is sealed into a hash-chained audit log.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditEvent, AuditLogger, state_hash
from .gates import (GateDecision, GateResult, GateState, GMSPlanGate, GMSPlausibilityGate, PolicyGate,
                    SpeechMonitor, SyntaxGate, ToolCall, ToolRegistry, compile_policies, completeness_check,
                    injection_flag)


@dataclass
class GovernanceConfig:
    theta: float                      # plausibility gate cut (Appendix C)
    drift_budget: float               # plan gate budget (Appendix C)
    relation: str = "enables"
    plan_criterion: str = "mean"      # mean (book) | max
    mode: str = "monitor"             # monitor | enforce


class RuntimeMonitor:
    def __init__(self, store, reader, cfg: GovernanceConfig, run_id: str = "run"):
        self.store, self.reader, self.cfg, self.run_id = store, reader, cfg, run_id
        self.registry = ToolRegistry(store)
        from ..reuse import governance as fl
        self.engine = "forgeloop" if (fl.available() and hasattr(store, "kx")) else "built-in"
        geo = (fl.ForgeloopPlausibilityGate(store.kx, cfg.theta) if self.engine == "forgeloop"
               else GMSPlausibilityGate(store, cfg.theta, cfg.relation))
        self.gates = [SyntaxGate(), PolicyGate(compile_policies(store)), geo]
        self.speech = SpeechMonitor(store)
        self.plan_gate = GMSPlanGate(store, cfg.drift_budget, cfg.relation, criterion=cfg.plan_criterion)
        self.state = GateState()
        self.audit = fl.ForgeloopAudit(run_id) if self.engine == "forgeloop" else AuditLogger()
        self._scores: list[float] = []
        self.events: list[dict[str, Any]] = []
        self.input_flags: list[dict[str, Any]] = []
        self._seq = 0
        self._fee_text = {f"${store.lookup_enm('delivery_fee_usd', 'expedited_delivery')}",
                          f"{store.lookup_enm('delivery_fee_usd', 'expedited_delivery')} dollars"}

    # ------------------------------------------------------------------ observers
    def _seal(self, kind: str, proposed: dict, results: list[GateResult], outcome: str) -> None:
        dec = [{"gate": r.gate_name, "decision": r.decision.value, "reason": r.reason, "rule": r.rule,
                "score": r.score} for r in results]
        sh = state_hash([c["tool"] for c in self.state.tool_results] + [self.state.user_turns])
        if self.engine == "forgeloop":
            self.audit.log(self._seq, kind, sh, proposed, dec, outcome)
        else:
            self.audit.log(AuditEvent(self.run_id, self._seq, kind, sh, proposed, dec, outcome))
        self._scores += [r.score for r in results if r.gate_name == "gms_plausibility" and r.score is not None]
        self._seq += 1
        for r in results:
            if r.decision != GateDecision.ALLOW:
                self.events.append({"kind": kind, "gate": r.gate_name, "decision": r.decision.value,
                                    "rule": r.rule, "reason": r.reason, "proposed": proposed,
                                    "turn": self.state.user_turns})

    def on_user(self, text: str) -> None:
        frame = self.reader.read(text)
        self.state.user_frames.append(frame)
        self.state.user_turns += 1
        if getattr(frame, "act", None) == "REQUEST_HUMAN":
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

    def before_tool(self, tool: str, args: dict) -> list[GateResult]:
        action = ToolCall(tool, dict(args))
        results = [g.check(action, self.state, self.registry) for g in self.gates]
        worst = "denied" if any(r.decision == GateDecision.DENY for r in results) else \
            "escalated" if any(r.decision == GateDecision.ESCALATE for r in results) else "allowed"
        self._seal("tool", {"tool": tool, "args": dict(args), "prev": self.state.prev_node()}, results, worst)
        return results

    def after_tool(self, tool: str, args: dict, result: Any, ok: bool) -> None:
        self.state.tool_results.append({"tool": tool, "args": dict(args), "result": copy.deepcopy(result), "ok": ok})
        if tool == "get_card_action_options":
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
        from ..sut.tools import ToolError
        res = self.monitor.before_tool(tool, args)
        if self.monitor.cfg.mode == "enforce" and any(r.decision == GateDecision.DENY for r in res):
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
