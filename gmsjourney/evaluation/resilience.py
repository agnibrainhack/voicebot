"""Resilience under tool faults: does the bot fail loud? (book Ch. 11, 12.8)

For each tool and fault kind, a ToolGateway injects the fault on every call to
that tool while the other tools behave normally. Over the runs that actually
reached the faulted tool:

  expected_loud  runs where the oracle, under the same fault, escalates;
  detected       of those, runs where the bot also escalated and made no claim that
                 rests on the faulted result;
  silent         expected_loud - detected: the bot carried on as if nothing happened;
  undetectable   reached - expected_loud: faults no bot can see at its boundary (a
                 forced AUTHENTICATED carries no echo to verify); the oracle proceeds too;
  agrees         the bot's outcome equals the oracle's outcome under the same fault.

Outright errors and timeouts are the easy case. Stale and plausible-but-wrong
results are well formed, so only an echo/verifier check catches them (R33).
"""
from __future__ import annotations

from typing import Any

from ..sut.tools import FaultProfile, ToolGateway

FAULT_TOOLS = ["authenticate_customer", "get_review_transactions", "record_charge_response",
               "get_card_action_options", "submit_card_action"]
KINDS = ["error", "timeout", "stale", "plausible_wrong"]


def probe(runner, oracle, scenarios, tools=FAULT_TOOLS, kinds=KINDS) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        for kind in kinds:
            prof = [FaultProfile(tool_pattern=tool, kind=kind, error_message=f"{tool} unavailable (injected)")]
            reached = detected = agrees = expected_loud = 0
            examples = []
            for scn in scenarios:
                res = runner.run(scn, faults=prof)
                if not any(i["tool"] == tool for i in res.extras["injected"]):
                    continue
                reached += 1
                tr = res.extras["trace"]
                exp = oracle.run(scn.base.metadata["spec"], scn.base.metadata["fixture"],
                                 gateway_factory=lambda env: ToolGateway(env, list(prof)))
                ok = res.answer == exp.signature()
                agrees += int(ok)
                if not exp.transferred:
                    continue
                expected_loud += 1
                loud = tr.transferred and tr.summary_consistent()
                detected += int(loud)
                if not loud and len(examples) < 2:
                    examples.append({"sid": scn.sid, "status": res.status, "claims": tr.claims_recorded,
                                     "expected": exp.signature(), "got": res.answer})
            if reached:
                out.append({"tool": tool, "kind": kind, "reached": reached, "expected_loud": expected_loud,
                            "detected": detected, "silent": expected_loud - detected,
                            "undetectable": reached - expected_loud,
                            "detection_rate": detected / expected_loud if expected_loud else None,
                            "oracle_agreement": agrees / reached, "silent_examples": examples})
    return out
