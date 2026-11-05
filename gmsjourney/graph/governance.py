"""The runtime monitor for a compact policy graph (book Ch. 12.5).

The same gates as the v1 monitor - schema (syntax), the policy's call checks compiled from the
graph's machine-checkable rules, knowlytix/forgeloop GMS plausibility and the plan gate - with a
speech monitor that reads the bot's structured acts: a success claim (a state that ``claims`` a
tool) needs a success echoing the request, a stance the policy contradicts is denied in u-space,
and every money figure the bot speaks must be a register value.
"""
from __future__ import annotations

import re
from typing import Any

from ..governance.gates import (GateDecision, GateResult, GateState, GMSPlanGate, PolicyGate, SyntaxGate,
                                ToolRegistry, compile_policies)
from ..governance.monitor import GovernanceConfig, RuntimeMonitor   # noqa: F401 - re-exported

_MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)")


class GraphSpeechMonitor:
    name = "speech"

    def __init__(self, store):
        from ..domain import current
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
        for m in _MONEY.finditer(t):
            v = float(m.group(1))
            verdict = "PASS" if v in self.money else "FAIL"
            self.attestations.append({"value": v, "verdict": verdict, "family": "money"})
            if verdict == "FAIL":
                out.append(GateResult(GateDecision.DENY, "numeric_attestor", f"money figure {v:g} is in no register",
                                      rule="NUM-FAIL"))
        return out


class GraphRuntimeMonitor(RuntimeMonitor):
    def __init__(self, store, reader, cfg: GovernanceConfig, run_id: str = "run"):
        from ..reuse import governance as fl
        self.store, self.reader, self.cfg, self.run_id = store, reader, cfg, run_id
        self.registry = ToolRegistry(store)
        self.engine = "forgeloop"
        self.gates = [SyntaxGate(), PolicyGate(compile_policies(store)), fl.ForgeloopPlausibilityGate(store.kx, cfg.theta)]
        self.speech = GraphSpeechMonitor(store)
        self.plan_gate = GMSPlanGate(store, cfg.drift_budget, cfg.relation, criterion=cfg.plan_criterion)
        self.state = GateState()
        self.state.bot_acts = []
        self.audit = fl.ForgeloopAudit(run_id)
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
