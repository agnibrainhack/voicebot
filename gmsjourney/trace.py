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
from .sut.tools import account_tools

def _flow() -> dict[str, Any]:
    from .domain import current
    return current().section("flow")


def _tools() -> dict[str, Any]:
    from .domain import current
    return current().section("tools")


def _ok(tool: str) -> str:
    sp = _tools()[tool]
    return sp["check"]["success"] if sp.get("check") else sp["status"]["ok"]


def __getattr__(name: str):                       # backward-compatible module constant
    if name == "WORKFLOW_STAGES":
        return list(_flow()["stages"])
    raise AttributeError(name)


@dataclass
class Trace:
    """What happened in one run. Which tools authenticate, read, write and hand over, and which
    acts and limits matter, come from the domain pack (domain.yaml -> flow, tools)."""
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

    @staticmethod
    def _handoff() -> tuple[str, str]:
        h = _flow()["handoff"]["tool"]
        return h, next(iter(_tools()[h].get("args") or {"reason": ""}))

    @property
    def transferred(self) -> bool:
        return bool(self.calls(self._handoff()[0]))

    @property
    def transfer_reason(self) -> str | None:
        tool, arg = self._handoff()
        c = self.calls(tool)
        return c[0]["args"].get(arg) if c else None

    @property
    def end_state(self) -> str:
        if self.transferred:
            return "TRANSFERRED"
        return "ENDED" if self.ended else "INCOMPLETE"

    def record_calls(self) -> list[tuple[str, str, str]]:
        """Every write: (item, value, SUCCESS | FAILED); a success must echo the item."""
        W = _flow()["per_item"]["write"]
        out = []
        for c in self.calls(W["tool"]):
            a, r = c["args"], c.get("result") or {}
            ok = c.get("ok") and r.get("status") == _ok(W["tool"]) and r.get(W["echo"]) == a.get(W["echo"])
            out.append((a.get(W["args"]["item"]), a.get(W["args"]["value"]), "SUCCESS" if ok else "FAILED"))
        return out

    def recorded(self) -> dict[str, str]:
        return {t: d for t, d, s in self.record_calls() if s == "SUCCESS"}

    def completed_action(self) -> tuple[str, str | None] | None:
        """The follow-up action that truly completed (success echoing the action), if any."""
        S = (_flow()["per_item"].get("follow_up") or {}).get("submit")
        if not S:
            return None
        for c in self.calls(S["tool"]):
            a, r = c["args"], c.get("result") or {}
            if c.get("ok") and r.get("status") == _ok(S["tool"]) and r.get(S["echo"]) == a.get(S["echo"]):
                return (a.get(S["args"]["action"]), a.get(S["args"]["option"]))
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
        fl = _flow()
        names = fl["stage_names"]
        A, Fe = fl["authenticate"], fl["fetch"]
        auth = self.calls(A["tool"])
        authed = any((c.get("result") or {}).get("status") == _ok(A["tool"]) for c in auth)
        if not auth:
            a = "0:" + ("TRANSFER" if self.transferred else "END")
        else:
            a = f"{len(auth)}:" + ("PROCEED" if authed and self.calls(Fe["tool"])
                                   else "TRANSFER" if self.transferred else "STOP")
        ret = self.calls(Fe["tool"])
        if not ret:
            r = "NOT_CALLED"
        elif not ret[0].get("ok"):
            r = "UNAVAILABLE:" + ("TRANSFER" if self.transferred else "CONTINUE")
        else:
            r = "CALLED"
        comps = {names["authenticate"]: a, names["fetch"]: r,
                 names["present"]: "|".join(self.presented) or "-",
                 names["capture"]: "|".join(f"{t}={d}" for t, d, _ in self.record_calls()) or "-",
                 names["write"]: "|".join(f"{t}={d}:{s}" for t, d, s in self.record_calls()) or "-"}
        FU = fl["per_item"].get("follow_up")
        if FU:
            S = FU["submit"]
            sub, opts = self.calls(S["tool"]), self.calls(FU["options"]["tool"])
            if sub:
                done = self.completed_action()
                sa = sub[-1]["args"]
                c = f"{sa.get(S['args']['action'])}/{sa.get(S['args']['option'])}:" + ("SUCCESS" if done else "FAILED")
            elif opts:
                c = "OPTIONS_ONLY"
            else:
                c = "-"
            comps[names["follow_up"]] = c
        comps[names["handoff"]] = self.transfer_reason or "-"
        comps[names["summary"]] = "CONSISTENT" if self.summary_consistent() else "VIOLATION"
        if self.kb:
            comps[names["question"]] = "CORRECT" if all(k.get("correct") for k in self.kb) else "INCORRECT"
        return comps

    # -------------------------------------------------------------- trajectory
    def node_sequence(self) -> list[str]:
        wf = _flow()["workflow"]
        seq = [wf["start"]] + [c["tool"] for c in self.tool_log]
        if not self.transferred and self.ended:
            seq.append(wf["summary"])
        seq.append(wf["end"])
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
        fl = _flow()
        A, P = fl["authenticate"], fl["per_item"]
        handoff = self._handoff()[0]
        v: list[str] = []
        _, unknown = self.edges(workflow)
        v += [f"edge_not_in_policy_graph:{a}->{b}" for a, b in unknown]
        authed = False
        transferred = False
        acct = account_tools()
        for c in self.tool_log:
            if transferred:
                v.append(f"tool_after_transfer:{c['tool']}")
            if c["tool"] == A["tool"] and (c.get("result") or {}).get("status") == _ok(A["tool"]):
                authed = True
            if c["tool"] in acct and not authed:
                v.append(f"account_tool_before_auth:{c['tool']}")
            if c["tool"] == handoff:
                transferred = True
        if len(self.calls(A["tool"])) > limits[A["limit"]]:
            v.append("auth_attempts_exceeded")
        if len(self.presented) > limits[P["cap"]["limit"]]:
            v.append("transaction_limit_exceeded")
        if not self.summary_consistent():
            v.append("claim_without_success")
        FU = P.get("follow_up")
        if FU:                                   # confirmation, and disclosure, before the follow-up write
            S, D, C = FU["submit"], FU["disclose"], FU["confirm"]
            disclose_arg = {FU["action_slot"]: S["args"]["action"], FU["option_slot"]: S["args"]["option"]}
            for i, ev in enumerate(self.events):
                if ev.get("role") == "tool" and ev.get("tool") == S["tool"]:
                    prior_user = [e for e in self.events[:i] if e.get("role") == "user"]
                    if not prior_user or prior_user[-1]["frame"].act != C["accept"]:
                        v.append("submit_without_confirmation")
                    opt_cond = {disclose_arg[k]: val for k, val in D["when"].items() if k == FU["option_slot"]}
                    if opt_cond and all(ev.get("args", {}).get(k) == val for k, val in opt_cond.items()):
                        disclosed = any(e.get("role") == "bot" and any(a.get("act") == D["act"] for a in e.get("acts", []))
                                        for e in self.events[:i])
                        if not disclosed:
                            v.append("expedited_without_fee_disclosure")
        if self.turns > limits[fl["turn_limit"]["limit"]]:
            v.append("turn_limit_exceeded")
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
