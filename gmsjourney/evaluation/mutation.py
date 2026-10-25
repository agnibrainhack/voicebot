"""Mutation testing: does the suite catch real bugs?

A suite "truly tests the bot in all aspects" only if seeded defects make it
fail. Each mutant in ``reference_bot.MUTATIONS`` is a realistic bug. A mutant
is killed by a scenario when the reference bot passes that scenario (outcome,
trajectory and claims all correct) and the mutant does not, or, for fault
probes, when the mutant's outcome under an injected fault diverges from the
reference bot's.

Comparing suites at the same size isolates what the design buys: the covering-
array + closure suite against a random suite of equal size, with and without
enrichment, with and without fault probes.
"""
from __future__ import annotations

from typing import Any

from ..sut.adapter import JourneyRunner, make_reference_factory
from ..sut.reference_bot import MUTATIONS
from ..sut.tools import FaultProfile
from .runner import run


def _ok(r: dict) -> bool:
    return bool(r["correct"] and r["workflow_adherent"] and r.get("c_summary", 1) and r.get("c_claims", 1)
                and r.get("c_said", 1))


def mutation_analysis(store, runner_kwargs: dict, suites: dict[str, list], limits, workflow, validator,
                      fault_scenarios: list | None = None,
                      fault_plan: tuple = (("record_charge_response", "error"), ("record_charge_response", "plausible_wrong"),
                                           ("get_card_action_options", "error"), ("submit_card_action", "error"),
                                           ("submit_card_action", "plausible_wrong"), ("get_review_transactions", "error")),
                      mutants: dict[str, str] = MUTATIONS, governance=None,
                      runner_cls=JourneyRunner, factory=make_reference_factory) -> dict[str, Any]:
    """governance: a GovernanceConfig; when given, every run is also watched by the runtime
    monitor and a mutant counts as *flagged at runtime* when the monitor raises an alarm
    (deny/escalate) on a scenario where it raised none of that kind for the reference bot -
    detection without an oracle, as in production."""
    runner_kwargs = dict(runner_kwargs)
    if governance is not None:
        runner_kwargs["governance"] = governance
    ref = runner_cls(bot_factory=factory(store), **runner_kwargs)
    ref_rows = {name: {r["sid"]: r for r in run(scns, ref, limits, workflow, validator)} for name, scns in suites.items()}
    ref_fault = {}
    for tool, kind in (fault_plan if fault_scenarios else ()):
        prof = [FaultProfile(tool, kind)]
        for scn in fault_scenarios:
            ref_fault[(tool, kind, scn.sid)] = ref.run(scn, faults=prof)

    results: dict[str, Any] = {}
    for m, desc in mutants.items():
        runner = runner_cls(bot_factory=factory(store, frozenset({m})), **runner_kwargs)
        entry: dict[str, Any] = {"description": desc, "killed_by": {}}
        for name, scns in suites.items():
            rows = {r["sid"]: r for r in run(scns, runner, limits, workflow, validator)}
            killers = [sid for sid, r in rows.items() if sid in ref_rows[name] and _ok(ref_rows[name][sid]) and not _ok(r)]
            if governance is not None:
                new = sorted({g for sid, r in rows.items() if sid in ref_rows[name]
                              for g in set(r.get("gate_rules", [])) - set(ref_rows[name][sid].get("gate_rules", []))})
                fl = entry.setdefault("runtime_monitor", {})
                fl[name] = new
                entry["flagged_at_runtime"] = entry.get("flagged_at_runtime", False) or bool(new)
            ex = None
            if killers:
                r = rows[killers[0]]
                ex = {"sid": killers[0], "category": r["base"], "violations": r["violations"],
                      "expected_transfer": r["expected_transfer_reason"], "got_transfer": r["transfer_reason"]}
            entry["killed_by"][name] = {"killed": bool(killers), "n_killing_scenarios": len(killers),
                                        "example": ex}
        if fault_scenarios:
            fk = []
            for tool, kind in fault_plan:
                prof = [FaultProfile(tool, kind)]
                for scn in fault_scenarios:
                    base = ref_fault[(tool, kind, scn.sid)]
                    if not base.extras["injected"]:
                        continue
                    res = runner.run(scn, faults=prof)
                    bt, mt = base.extras["trace"], res.extras["trace"]
                    if res.answer != base.answer or (bt.summary_consistent() and not mt.summary_consistent()):
                        fk.append({"tool": tool, "kind": kind, "sid": scn.sid})
                    if governance is not None:
                        g0 = set((base.extras.get("governance") or {}).get("by_gate_rule", {}))
                        g1 = set((res.extras.get("governance") or {}).get("by_gate_rule", {}))
                        if g1 - g0:
                            fl = entry.setdefault("runtime_monitor", {})
                            fl.setdefault("fault_probes", [])
                            fl["fault_probes"] = sorted(set(fl["fault_probes"]) | (g1 - g0))
                            entry["flagged_at_runtime"] = True
            entry["killed_by"]["fault_probes"] = {"killed": bool(fk), "n_killing_scenarios": len(fk),
                                                  "example": fk[0] if fk else None}
        entry["killed"] = any(v["killed"] for v in entry["killed_by"].values())
        results[m] = entry
    names = list(suites) + (["fault_probes"] if fault_scenarios else [])
    scores = {n: sum(results[m]["killed_by"].get(n, {}).get("killed", False) for m in results) / len(results)
              for n in names}
    scores["all_suites_combined"] = sum(results[m]["killed"] for m in results) / len(results)
    out = {"mutants": results, "mutation_score": scores, "n_mutants": len(results)}
    if governance is not None:
        out["runtime_monitor_catch_rate"] = sum(bool(results[m].get("flagged_at_runtime")) for m in results) / len(results)
    return out
