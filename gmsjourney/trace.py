"""One journey's record, for both the oracle (expected) and the bot (observed).

A ``Trace`` is the book's unit of observation for an agent (Ch. 9.2): the ordered
record of bot acts, user frames and tool observations, ending in completion,
escalation (transfer) or failure. Everything scored later derives from it:

* ``signature()``      - the outcome: end state, transfer reason, what was
                         truly recorded (SUCCESS with matching echo) and which
                         card action truly completed.
* ``components()``     - per-stage decisions in workflow order, the per-tool
                         ground truth used by component scoring and weak-link.
* ``edges()``          - workflow edges traversed (W-ids from the store).
* ``admissibility()``  - per-step gate checks against the policy graph.
* ``process()``        - process health: steps, tool calls, failures, loops.

Because components are derived from the tool log (observable from the mock
backend for any bot) and from bot acts, a real bot is scored exactly as the
oracle's own trace is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .ontology import SemanticFrame

WORKFLOW_STAGES = ["authenticate", "retrieve", "present", "kb", "capture", "record",
                   "card_action", "transfer", "summary"]
ACCOUNT_TOOLS = {"get_review_transactions", "record_charge_response",
                 "get_card_action_options", "submit_card_action"}


@dataclass
class Trace:
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    user_frames: list[SemanticFrame] = field(default_factory=list)
    presented: list[str] = field(default_factory=list)
    claims_recorded: dict[str, str] = field(default_factory=dict)
    claims_actions: list[tuple[str, str | None]] = field(default_factory=list)
    kb: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    ended: bool = False
    rules_fired: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- tool facts
    def calls(self, tool: str) -> list[dict[str, Any]]:
        return [c for c in self.tool_log if c["tool"] == tool]

    @property
    def transferred(self) -> bool:
        return bool(self.calls("transfer_to_human"))

    @property
    def transfer_reason(self) -> str | None:
        c = self.calls("transfer_to_human")
        return c[0]["args"].get("reason") if c else None

    @property
    def end_state(self) -> str:
        if self.transferred:
            return "TRANSFERRED"
        return "ENDED" if self.ended else "INCOMPLETE"

    def record_calls(self) -> list[tuple[str, str, str]]:
        out = []
        for c in self.calls("record_charge_response"):
            a, r = c["args"], c.get("result") or {}
            ok = c.get("ok") and r.get("status") == "SUCCESS" and r.get("transaction_id") == a.get("transaction_id")
            out.append((a.get("transaction_id"), a.get("disposition"), "SUCCESS" if ok else "FAILED"))
        return out

    def recorded(self) -> dict[str, str]:
        return {t: d for t, d, s in self.record_calls() if s == "SUCCESS"}

    def completed_action(self) -> tuple[str, str | None] | None:
        for c in self.calls("submit_card_action"):
            a, r = c["args"], c.get("result") or {}
            if c.get("ok") and r.get("status") == "SUCCESS" and r.get("action") == a.get("action"):
                return (a.get("action"), a.get("delivery"))
        return None

    # -------------------------------------------------------------- outcome
    def signature(self) -> str:
        act = self.completed_action()
        return json.dumps({
            "end": self.end_state,
            "transfer_reason": self.transfer_reason,
            "recorded": sorted(self.recorded().items()),
            "action": list(act) if act else None,
        }, sort_keys=True)

    # -------------------------------------------------------------- components
    def summary_consistent(self) -> bool:
        rec = self.recorded()
        for t, d in self.claims_recorded.items():
            if rec.get(t) != d:
                return False                       # R13: claimed without SUCCESS
        done = self.completed_action()
        for a in self.claims_actions:
            if done is None or a[0] != done[0]:
                return False                       # R17: completion claimed without SUCCESS
        return True

    def components(self) -> dict[str, str]:
        auth = self.calls("authenticate_customer")
        authed = any((c.get("result") or {}).get("status") == "AUTHENTICATED" for c in auth)
        if not auth:
            a = "0:" + ("TRANSFER" if self.transferred else "END")
        else:
            a = f"{len(auth)}:" + ("PROCEED" if authed and self.calls("get_review_transactions")
                                   else "TRANSFER" if self.transferred else "STOP")
        ret = self.calls("get_review_transactions")
        if not ret:
            r = "NOT_CALLED"
        elif not ret[0].get("ok"):
            r = "UNAVAILABLE:" + ("TRANSFER" if self.transferred else "CONTINUE")
        else:
            r = "CALLED"
        card = self.calls("submit_card_action")
        opts = self.calls("get_card_action_options")
        if card:
            done = self.completed_action()
            sa = card[-1]["args"]
            c = f"{sa.get('action')}/{sa.get('delivery')}:" + ("SUCCESS" if done else "FAILED")
        elif opts:
            c = "OPTIONS_ONLY"
        else:
            c = "-"
        comps = {
            "authenticate": a,
            "retrieve": r,
            "present": "|".join(self.presented) or "-",
            "capture": "|".join(f"{t}={d}" for t, d, _ in self.record_calls()) or "-",
            "record": "|".join(f"{t}={d}:{s}" for t, d, s in self.record_calls()) or "-",
            "card_action": c,
            "transfer": self.transfer_reason or "-",
            "summary": "CONSISTENT" if self.summary_consistent() else "VIOLATION",
        }
        if self.kb:
            comps["kb"] = "CORRECT" if all(k.get("correct") for k in self.kb) else "INCORRECT"
        return comps

    # -------------------------------------------------------------- trajectory
    def node_sequence(self) -> list[str]:
        seq = ["start"] + [c["tool"] for c in self.tool_log]
        if not self.transferred and self.ended:
            seq.append("summarize")
        seq.append("end")
        return seq

    def edges(self, workflow) -> tuple[set[str], list[tuple[str, str]]]:
        pairs = {}
        for w in workflow:
            pairs.setdefault((w.src, w.dst), []).append(w.wid)
        seq = self.node_sequence()
        hit, unknown = set(), []
        for a, b in zip(seq, seq[1:]):
            if (a, b) in pairs:
                hit.update(pairs[(a, b)])
            else:
                unknown.append((a, b))
        return hit, unknown

    def admissibility(self, workflow, limits: dict[str, int]) -> list[str]:
        """Per-step gate violations (empty list = admissible trajectory)."""
        v: list[str] = []
        _, unknown = self.edges(workflow)
        v += [f"edge_not_in_policy_graph:{a}->{b}" for a, b in unknown]
        authed = False
        transferred = False
        for c in self.tool_log:
            if transferred:
                v.append(f"tool_after_transfer:{c['tool']}")
            if c["tool"] == "authenticate_customer" and (c.get("result") or {}).get("status") == "AUTHENTICATED":
                authed = True
            if c["tool"] in ACCOUNT_TOOLS and not authed:
                v.append(f"account_tool_before_auth:{c['tool']}")       # R2/R3
            if c["tool"] == "transfer_to_human":
                transferred = True
        if len(self.calls("authenticate_customer")) > limits["max_auth_attempts"]:
            v.append("auth_attempts_exceeded")                          # R4
        if len(self.presented) > limits["max_transactions_per_session"]:
            v.append("transaction_limit_exceeded")                     # R31
        if not self.summary_consistent():
            v.append("claim_without_success")                          # R13/R17
        # confirmation (R16) and fee disclosure (R26) before submit
        for i, ev in enumerate(self.events):
            if ev.get("role") == "tool" and ev.get("tool") == "submit_card_action":
                prior_user = [e for e in self.events[:i] if e.get("role") == "user"]
                if not prior_user or prior_user[-1]["frame"].act != "CONFIRM":
                    v.append("submit_without_confirmation")
                if ev.get("args", {}).get("delivery") == "expedited_delivery":
                    disclosed = any(e.get("role") == "bot" and any(a.get("act") == "DISCLOSE_FEE" for a in e.get("acts", []))
                                    for e in self.events[:i])
                    if not disclosed:
                        v.append("expedited_without_fee_disclosure")
        if self.turns > limits["max_turns"]:
            v.append("turn_limit_exceeded")                            # R28
        return v

    def process(self) -> dict[str, Any]:
        bot_turns = sum(1 for e in self.events if e.get("role") == "bot")
        fails = sum(1 for c in self.tool_log if c.get("ok") is False)
        loops = 0
        run = 1
        for a, b in zip(self.tool_log, self.tool_log[1:]):
            same = a["tool"] == b["tool"] and a["args"] == b["args"]
            run = run + 1 if same else 1
            if run == 3:
                loops += 1
        return {"n_steps": bot_turns, "tool_calls": len(self.tool_log), "tool_failures": fails,
                "loops": loops, "clean_termination": self.ended or self.transferred,
                "turns": self.turns}
