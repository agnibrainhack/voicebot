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

TOOLS = ["authenticate_customer", "get_review_transactions", "record_charge_response",
         "get_card_action_options", "submit_card_action", "transfer_to_human"]


PLAUSIBLE_WRONG_TOOLS = {"record_charge_response", "submit_card_action", "authenticate_customer",
                         "get_card_action_options"}


class ToolError(RuntimeError):
    pass


class ToolTimeout(ToolError):
    pass


@dataclass
class ToolEnvironment:
    fixture: dict[str, Any]
    log: list[dict[str, Any]] = field(default_factory=list)
    _auth_calls: int = 0
    _record_calls: int = 0

    def call(self, tool: str, **args: Any) -> Any:
        fn = getattr(self, f"_t_{tool}", None)
        if fn is None:
            raise ToolError(f"unknown tool {tool}")
        entry = {"tool": tool, "args": dict(args)}
        try:
            res = fn(**args)
            entry.update(result=copy.deepcopy(res), ok=True)
            return res
        except ToolError as e:
            entry.update(result={"error": str(e)}, ok=False)
            raise
        finally:
            self.log.append(entry)

    # -- tool implementations ---------------------------------------------------
    def _t_authenticate_customer(self, credential: str = "") -> dict:
        results = self.fixture.get("auth_results", ["AUTHENTICATED"])
        i = min(self._auth_calls, len(results) - 1)
        self._auth_calls += 1
        r = results[i]
        if r == "UNAVAILABLE":
            raise ToolError("authenticate_customer unavailable")
        if r == "AUTHENTICATED" and credential != self.fixture.get("credential"):
            r = "FAILED"                 # a wrong credential never authenticates
        return {"status": r}

    def _t_get_review_transactions(self) -> dict:
        if self.fixture.get("transactions_status") == "UNAVAILABLE":
            raise ToolError("get_review_transactions unavailable")
        txns = copy.deepcopy(self.fixture.get("transactions", []))
        return {"status": "TRANSACTIONS" if txns else "EMPTY", "transactions": txns}

    def _t_record_charge_response(self, transaction_id: str = "", disposition: str = "") -> dict:
        self._record_calls += 1
        if self.fixture.get("record_status") == "UNAVAILABLE":
            raise ToolError("record_charge_response unavailable")
        fail_on = self.fixture.get("record_fail_on_call")
        status = "FAILED" if fail_on is not None and self._record_calls == fail_on else "SUCCESS"
        return {"status": status, "transaction_id": transaction_id, "disposition": disposition}

    def _t_get_card_action_options(self) -> dict:
        if self.fixture.get("card_options_status") == "UNAVAILABLE":
            raise ToolError("get_card_action_options unavailable")
        opts = list(self.fixture.get("card_options", []))
        return {"status": "OPTIONS" if opts else "NONE", "options": opts}

    def _t_submit_card_action(self, action: str = "", delivery: str | None = None) -> dict:
        if self.fixture.get("submit_status") == "UNAVAILABLE":
            raise ToolError("submit_card_action unavailable")
        return {"status": self.fixture.get("submit_result", "SUCCESS"), "action": action,
                "delivery": delivery}

    def _t_transfer_to_human(self, reason: str = "") -> dict:
        return {"status": "TRANSFERRED", "reason": reason}


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
        prof = self._match(tool) if tool != "transfer_to_human" else None
        if prof is None:
            res = self.env.call(tool, **args)
            self._last[tool] = copy.deepcopy(res)
            return res
        if prof.kind == "stale" and tool not in self._last:
            return self.call_passthrough(tool, args)      # nothing cached yet: behaves normally
        if prof.kind == "plausible_wrong" and tool not in PLAUSIBLE_WRONG_TOOLS:
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
            res = self.env.call(tool, **args)
            res = copy.deepcopy(res)
            if tool == "record_charge_response":
                res["transaction_id"] = "T999"
            elif tool == "submit_card_action":
                res["action"] = "SECURE_CARD" if args.get("action") == "REPLACE_CARD" else "REPLACE_CARD"
            elif tool == "authenticate_customer":
                res["status"] = "AUTHENTICATED"
            elif tool == "get_card_action_options":
                res["options"] = ["SECURE_CARD", "REPLACE_CARD"]
            self.env.log[-1]["result"] = copy.deepcopy(res)
            self.env.log[-1]["injected"] = "plausible_wrong"
            return res
        raise ValueError(f"unknown fault kind {prof.kind}")

    def call_passthrough(self, tool: str, args: dict) -> Any:
        res = self.env.call(tool, **args)
        self._last[tool] = copy.deepcopy(res)
        return res
