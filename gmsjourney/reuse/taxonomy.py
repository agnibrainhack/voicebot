"""proofloop's failure-mode taxonomy mapped onto the probes this campaign runs.

proofloop.evaluation.failure_modes.FailureMode names nine agent failure modes. A
suite only covers a mode if some probe exercises it; this table records which
seeded defect, injected fault, factor level or gate probes each mode, and whether
the campaign caught it (mutation kill / resilience / gate detection).
"""
from __future__ import annotations

from typing import Any

PROBES = {
    "wrong_tool": {"mutants": ["present_all_returned"], "faults": [], "factors": [],
                   "gates": ["gate probe sweep: every out-of-workflow transition"]},
    "malformed_call": {"mutants": [], "faults": [], "factors": [], "gates": ["SyntaxGate schema probes"]},
    "hallucinated_citation": {"mutants": ["kb_first_amount", "fraud_claim"], "faults": [],
                              "factors": ["answer_anchoring=wrong_suggestion"], "gates": ["speech ENM verifier", "R27 stance"]},
    "prompt_injection": {"mutants": [], "faults": [], "factors": ["instruction_conflict=authority_override",
                                                                 "instruction_conflict=contradictory_hint"],
                         "gates": ["input guard (forgeloop regex baseline)"]},
    "over_delegation": {"mutants": ["ignore_human_mid_review"], "faults": [], "factors": [], "gates": ["R20 policy"]},
    "infinite_loop": {"mutants": ["no_turn_limit"], "faults": [], "factors": [], "gates": ["R28 policy"]},
    "premature_stop": {"mutants": ["card_erases_review", "unsure_no_transfer"], "faults": [], "factors": [],
                       "gates": ["completeness R12/R18", "R11 policy"]},
    "stale_memory": {"mutants": ["first_answer_wins"], "faults": ["stale"], "factors": ["self_correction=false_start"],
                     "gates": ["R21 policy"]},
    "irrelevant_retrieval": {"mutants": ["kb_first_amount"], "faults": [], "factors": ["history=distracting_history"],
                             "gates": ["speech ENM verifier"]},
}


def coverage(mutation: dict | None, resilience: dict | None, flagged_by_monitor: dict | None) -> list[dict[str, Any]]:
    from proofloop.evaluation.failure_modes import FailureMode
    rows = []
    for mode in FailureMode:
        p = PROBES.get(mode.value, {})
        killed = [m for m in p.get("mutants", []) if mutation and mutation["mutants"].get(m, {}).get("killed")]
        flagged = [m for m in p.get("mutants", []) if flagged_by_monitor and flagged_by_monitor.get(m)]
        rows.append({"failure_mode": mode.value, **p, "mutants_killed": killed, "mutants_flagged_at_runtime": flagged,
                     "exercised": bool(p.get("mutants") or p.get("faults") or p.get("factors") or p.get("gates"))})
    return rows

