"""The mock bank the bot talks to, and the record of one conversation.

Parts of this file:
  1. ``ToolEnvironment``: the simulated backend tools, answering from the case's fixture (a status per call,
     results, echoes); ``ToolGateway`` / ``FaultProfile`` inject faults (error, timeout, stale data, a plausible
     wrong answer) for the resilience probe.
  2. ``Trace``: what happened in one run - the tools called with arguments and results, the bot acts, the
     customer turns, the outcome - the same record for the oracle (expected) and the bot (observed).
"""
from __future__ import annotations

import copy
import fnmatch
import json
import random
from dataclasses import dataclass, field
from typing import Any


# =====================================================================================================================
# PART 1 - Simulated backend tools and the fault-injection gateway (book Ch. 11).
#
# ``ToolEnvironment`` serves one journey's backend fixture: authentication
# results per attempt, the transaction list, recording results, card options and
# the submit result. Every call is logged with its arguments and result.
#
# ``ToolGateway`` wraps an environment and injects faults per tool pattern:
#
# * error          - the call raises ``ToolError`` (outright failure)
# * timeout        - the call raises ``ToolTimeout``
# * stale          - the call returns the previous response for that tool
#                    (well-formed but out of date)
# * plausible_wrong - the call returns a well-formed SUCCESS that echoes the
#                    wrong transaction/action; only a verifier that compares the
#                    echo against the request can catch it
#
# The same classes back the oracle and the system under test, so the expected
# behaviour under a fault is computed from exactly the same backend.
# =====================================================================================================================

# ------------------------------------------------------------------- the tool registry
# Every tool, its role, arguments and mock behaviour come from the domain pack
# (domain.yaml -> tools); nothing here names a tool.
def tool_specs() -> dict[str, dict[str, Any]]:
    """The active pack's tool specifications (domain.yaml -> tools, mined from a compact graph)."""
    from ..packs.loading import current
    return current().section("tools")


def tool_names() -> list[str]:
    """The names of the active pack's tools."""
    return list(tool_specs())


def tools_with_role(*roles: str) -> set[str]:
    """The tools whose role is one of ``roles`` (auth, read, write, handoff ...)."""
    return {t for t, sp in tool_specs().items() if sp.get("role") in roles}


def account_tools() -> set[str]:
    """Tools that touch the account: only callable after the auth tool succeeded."""
    return tools_with_role("read", "write")


def write_tools() -> set[str]:
    """The tools that change the account."""
    return tools_with_role("write")


def key_args() -> dict[str, tuple[str, ...]]:
    """The arguments that identify a call when comparing two runs (tools without any: all args)."""
    return {t: tuple(sp["key_args"]) for t, sp in tool_specs().items() if sp.get("key_args")}


def _tools_getattr(name: str):                     # backward-compatible module constants
    if name == "TOOLS":
        return tool_names()
    if name == "PLAUSIBLE_WRONG_TOOLS":
        return {t for t, sp in tool_specs().items() if sp.get("plausible_wrong")}
    raise AttributeError(name)


class ToolError(RuntimeError):
    """A tool call failed (the mock bank's equivalent of a service error)."""
    pass


class ToolTimeout(ToolError):
    """A tool call timed out."""
    pass


@dataclass
class ToolEnvironment:
    """One journey's backend: every tool is served from the fixture by its declared behaviour."""
    fixture: dict[str, Any]
    log: list[dict[str, Any]] = field(default_factory=list)
    specs: dict[str, dict[str, Any]] | None = None
    _calls: dict[str, int] = field(default_factory=dict)
    explore: bool = False             # graph packs: an unscripted call asks the journey miner for a choice

    def __post_init__(self) -> None:
        if self.specs is None:
            self.specs = tool_specs()

    def call(self, tool: str, **args: Any) -> Any:
        spec = self.specs.get(tool)
        if spec is None:
            raise ToolError(f"unknown tool {tool}")
        entry = {"tool": tool, "args": dict(args)}
        try:
            res = self._serve(tool, spec, args)
            entry.update(result=copy.deepcopy(res), ok=True)
            return res
        except ToolError as e:
            entry.update(result={"error": str(e)}, ok=False)
            raise
        finally:
            self.log.append(entry)

    # -- behaviours ---------------------------------------------------------------
    def _serve(self, tool: str, spec: dict, args: dict) -> dict:
        declared = spec.get("args") or {}
        for k in args:
            if k not in declared:          # the same failure a bad keyword would give a Python stub
                raise TypeError(f"ToolEnvironment._t_{tool}() got an unexpected keyword argument '{k}'")
        a = {**declared, **args}
        n = self._calls.get(tool, 0)
        self._calls[tool] = n + 1
        fx = self.fixture
        kind = spec["behaviour"]
        if kind == "script":                 # a graph pack: the n-th call returns the n-th scripted result
            results = fx.get(spec["results_field"])
            if self.explore and (results is None or n >= len(results)):
                from ..graph.interpreter import NeedChoice
                raise NeedChoice("tool", tool, n)
            results = results or spec.get("default_results")
            r = copy.deepcopy(results[min(n, len(results) - 1)])
            r = {"status": r} if isinstance(r, str) else r
            if r.get("status") == spec.get("unavailable"):
                raise ToolError(f"{tool} unavailable")
            chk = spec.get("check")
            if chk and r.get("status") == chk["success"] and a.get(chk["arg"]) != fx.get(chk["fixture"]):
                r["status"] = chk["else"]        # a wrong credential never authenticates
            if spec.get("echo") and r.get("status") == spec["status"]["ok"]:
                e = spec["echo"]
                if r.get(e) == "@other":         # echo a value other than the one sent (a mismatch, whatever was sent)
                    from ..packs.loading import current
                    vs = (spec.get("arg_values") or {}).get(e)
                    r[e] = next((m for m in current().graph.members(vs) if m != a.get(e)), None) if vs else None
                r.setdefault(e, a.get(e))
            return r
        if kind == "attempts":
            results = fx.get(spec["results_field"], spec.get("default_results"))
            r = results[min(n, len(results) - 1)]
            if r == spec.get("unavailable"):
                raise ToolError(f"{tool} unavailable")
            chk = spec.get("check")
            if chk and r == chk["success"] and a.get(chk["arg"]) != fx.get(chk["fixture"]):
                r = chk["else"]              # a wrong credential never authenticates
            return {"status": r}
        if spec.get("unavailable_flag") and fx.get(spec["unavailable_flag"]) == "UNAVAILABLE":
            raise ToolError(f"{tool} unavailable")
        if kind == "list":
            items = copy.deepcopy(fx.get(spec["field"], []))
            return {"status": spec["status"]["some"] if items else spec["status"]["none"], spec["key"]: items}
        if kind == "echo":
            if spec.get("result_field"):
                status = fx.get(spec["result_field"], spec["status"]["ok"])
            else:
                fail_on = fx.get(spec["fail_on_call_field"]) if spec.get("fail_on_call_field") else None
                status = spec["status"]["fail"] if fail_on is not None and n + 1 == fail_on else spec["status"]["ok"]
            return {"status": status, **{k: a[k] for k in declared}}
        raise ValueError(f"{tool}: unknown behaviour {kind!r}")


@dataclass(frozen=True)
class FaultProfile:
    """A fault to inject: which tools (a pattern), what kind (error, timeout, stale, plausible_wrong) and how often.
    """
    tool_pattern: str = "*"
    kind: str = "error"            # error | timeout | stale | plausible_wrong
    error_rate: float = 1.0
    error_message: str = "Simulated tool failure"


@dataclass
class ToolGateway:
    """Intercepts every tool call of an environment and applies fault profiles."""
    env: ToolEnvironment
    profiles: list[FaultProfile] = field(default_factory=list)
    seed: int = 0
    injected: list[dict[str, Any]] = field(default_factory=list)
    _last: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @property
    def log(self) -> list[dict[str, Any]]:
        return self.env.log

    @property
    def fixture(self) -> dict[str, Any]:
        return self.env.fixture

    def _match(self, tool: str) -> FaultProfile | None:
        for p in self.profiles:
            if fnmatch.fnmatch(tool, p.tool_pattern) and self._rng.random() < p.error_rate:
                return p
        return None

    def call(self, tool: str, **args: Any) -> Any:
        specs = self.env.specs or tool_specs()
        spec = specs.get(tool) or {}
        prof = self._match(tool) if spec.get("role") != "handoff" else None
        if prof is None:
            res = self.env.call(tool, **args)
            self._last[tool] = copy.deepcopy(res)
            return res
        if prof.kind == "stale" and tool not in self._last:
            return self.call_passthrough(tool, args)      # nothing cached yet: behaves normally
        if prof.kind == "plausible_wrong" and not spec.get("plausible_wrong"):
            return self.call_passthrough(tool, args)
        self.injected.append({"tool": tool, "kind": prof.kind, "args": dict(args)})
        if prof.kind in ("error", "timeout"):
            self.env.log.append({"tool": tool, "args": dict(args),
                                 "result": {"error": prof.error_message}, "ok": False,
                                 "injected": prof.kind})
            raise (ToolTimeout if prof.kind == "timeout" else ToolError)(prof.error_message)
        if prof.kind == "stale":
            res = copy.deepcopy(self._last[tool])
            self.env.log.append({"tool": tool, "args": dict(args), "result": res,
                                 "ok": True, "injected": "stale"})
            return res
        if prof.kind == "plausible_wrong":
            res = copy.deepcopy(self.env.call(tool, **args))
            pw = spec["plausible_wrong"]
            for k, v in (pw.get("set") or {}).items():
                res[k] = copy.deepcopy(v)
            fl = pw.get("flip")
            if fl:
                res[fl["field"]] = fl["map"].get(args.get(fl["from_arg"]), fl["default"])
            self.env.log[-1]["result"] = copy.deepcopy(res)
            self.env.log[-1]["injected"] = "plausible_wrong"
            return res
        raise ValueError(f"unknown fault kind {prof.kind}")

    def call_passthrough(self, tool: str, args: dict) -> Any:
        res = self.env.call(tool, **args)
        self._last[tool] = copy.deepcopy(res)
        return res


# =====================================================================================================================
# PART 2 - One journey's record, for both the oracle (expected) and the bot (observed).
#
# A ``Trace`` is the book's unit of observation for an agent (Ch. 9.2): the ordered
# record of bot acts, user frames and tool observations, ending in completion,
# escalation (transfer) or failure. Everything scored later derives from it:
#
# * ``signature()``      - the outcome: end state, transfer reason, what was
#                          truly recorded (SUCCESS with matching echo) and which
#                          card action truly completed.
# * ``components()``     - per-stage decisions in workflow order, the per-tool
#                          ground truth used by component scoring and weak-link.
# * ``edges()``          - workflow edges traversed (W-ids from the store).
# * ``admissibility()``  - per-step gate checks against the policy graph.
# * ``process()``        - process health: steps, tool calls, failures, loops.
#
# Because components are derived from the tool log (observable from the mock
# backend for any bot) and from bot acts, a real bot is scored exactly as the
# oracle's own trace is.
# =====================================================================================================================

from ..cases.meaning import SemanticFrame

def _flow() -> dict[str, Any]:
    from ..packs.loading import current
    return current().section("flow")


def _tools() -> dict[str, Any]:
    from ..packs.loading import current
    return current().section("tools")


def _ok(tool: str) -> str:
    sp = _tools()[tool]
    return sp["check"]["success"] if sp.get("check") else sp["status"]["ok"]


def _trace_getattr(name: str):                       # backward-compatible module constant
    if name == "WORKFLOW_STAGES":
        return list(_flow()["stages"])
    raise AttributeError(name)


def __getattr__(name: str):
    """Module constants computed from the active pack on every access (TOOLS, PLAUSIBLE_WRONG_TOOLS,
    WORKFLOW_STAGES): the old module-level names, kept for code that imports them."""
    for lookup in (_tools_getattr, _trace_getattr):
        try:
            return lookup(name)
        except AttributeError:
            pass
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
    flow_path: list[tuple[str, str, str]] = field(default_factory=list)   # graph transitions taken (oracle only)

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


def new_trace() -> "Trace":
    """A trace scored the way the active pack is: the graph-generic scoring (checks/scoring.py) for a pack written as
    a compact graph, else the per-item scoring configured in domain.yaml (flow)."""
    from ..packs.loading import current
    if current().graph is not None:
        from ..checks.scoring import GraphTrace
        return GraphTrace()
    return Trace()
