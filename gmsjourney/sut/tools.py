"""Simulated backend tools and the fault-injection gateway (book Ch. 11).

``ToolEnvironment`` serves one journey's backend fixture: authentication
results per attempt, the transaction list, recording results, card options and
the submit result. Every call is logged with its arguments and result.

``ToolGateway`` wraps an environment and injects faults per tool pattern:

* error          - the call raises ``ToolError`` (outright failure)
* timeout        - the call raises ``ToolTimeout``
* stale          - the call returns the previous response for that tool
                   (well-formed but out of date)
* plausible_wrong - the call returns a well-formed SUCCESS that echoes the
                   wrong transaction/action; only a verifier that compares the
                   echo against the request can catch it

The same classes back the oracle and the system under test, so the expected
behaviour under a fault is computed from exactly the same backend.
"""
from __future__ import annotations

import copy
import fnmatch
import random
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------------- the tool registry
# Every tool, its role, arguments and mock behaviour come from the domain pack
# (domain.yaml -> tools); nothing here names a tool.
def tool_specs() -> dict[str, dict[str, Any]]:
    from ..domain import current
    return current().section("tools")


def tool_names() -> list[str]:
    return list(tool_specs())


def tools_with_role(*roles: str) -> set[str]:
    return {t for t, sp in tool_specs().items() if sp.get("role") in roles}


def account_tools() -> set[str]:
    """Tools that touch the account: only callable after the auth tool succeeded."""
    return tools_with_role("read", "write")


def write_tools() -> set[str]:
    return tools_with_role("write")


def key_args() -> dict[str, tuple[str, ...]]:
    """The arguments that identify a call when comparing two runs (tools without any: all args)."""
    return {t: tuple(sp["key_args"]) for t, sp in tool_specs().items() if sp.get("key_args")}


def __getattr__(name: str):                     # backward-compatible module constants
    if name == "TOOLS":
        return tool_names()
    if name == "PLAUSIBLE_WRONG_TOOLS":
        return {t for t, sp in tool_specs().items() if sp.get("plausible_wrong")}
    raise AttributeError(name)


class ToolError(RuntimeError):
    pass


class ToolTimeout(ToolError):
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
                from ..graph.machine import NeedChoice
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
                r.setdefault(spec["echo"], a.get(spec["echo"]))
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
