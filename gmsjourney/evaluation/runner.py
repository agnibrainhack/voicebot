"""Run scenarios through the SUT and score them (book Ch. 9.5).

One row per scenario:
  sid, seed (base qid), base (category), tags,
  f_<factor>                  factor levels (the prefix attribution reads),
  correct                     outcome signature == oracle signature,
  c_<stage>                   per-stage component correctness (only where defined),
  status / expected_status, transfer_reason / expected_transfer_reason,
  tool_sequence, n_steps, tool_calls, tool_failures, loops, turns,
  workflow_adherent           1 if the trajectory has no gate violation,
  violations                  the gate violations (R2/R3, R4, R13/R17, R16, R26, R28, R31 ...),
  kb_correct                  graded policy answers (if any were asked),
  invariance_ok               runtime monitor: every realised turn read back to its frame.
Scenarios whose invariance proof failed are not run (they are reported).
"""
from __future__ import annotations

from typing import Any

from ..core import Scenario
from ..oracle import Limits
from ..trace import WORKFLOW_STAGES


def score(scn: Scenario, res, limits: Limits, workflow) -> dict[str, Any]:
    tr = res.extras["trace"]
    item = scn.base
    row: dict[str, Any] = {
        "sid": scn.sid, "seed": item.qid, "base": item.base,
        "tags": item.metadata.get("tags", [item.base]),
        **{f"f_{k}": v for k, v in scn.factor_levels.items()},
        "correct": int(res.answer == item.answer),
        "status": res.status,
        "expected_transfer_reason": item.metadata.get("transfer_reason"),
        "transfer_reason": res.escalation_trigger,
    }
    for stage in WORKFLOW_STAGES:
        k = f"expected_{stage}"
        if k in item.components and stage in res.components:
            row[f"c_{stage}"] = int(res.components[stage] == item.components[k])
        elif k in item.components:
            row[f"c_{stage}"] = 0                     # expected but never produced
    viol = tr.admissibility(workflow, limits.as_dict())
    proc = tr.process()
    row.update({
        "tool_sequence": ">".join(c["tool"] for c in tr.tool_log),
        "n_steps": proc["n_steps"], "tool_calls": proc["tool_calls"], "tool_failures": proc["tool_failures"],
        "loops": proc["loops"], "turns": proc["turns"], "clean_termination": int(proc["clean_termination"]),
        "workflow_adherent": int(not viol), "violations": viol,
        "invariance_ok": int(not res.extras["invariance_violations"]),
    })
    if tr.kb:
        row["kb_correct"] = int(all(k["correct"] for k in tr.kb))
        row["kb_detail"] = [{"frame": k["frame"], "answer": k["answer"], "correct": k["correct"],
                             "detail": k["detail"], **{x: k[x] for x in ("geo_label", "geo_distance") if x in k}}
                            for k in tr.kb]
    gov = res.extras.get("governance")
    if gov is not None:
        row["gate_alarms"] = gov["alarms"]
        row["gate_deny"] = gov["deny"]
        row["gate_escalate"] = gov["escalate"]
        row["gate_rules"] = sorted(gov["by_gate_rule"])
        row["plan_admissible"] = int(gov["plan"]["admissible"])
        row["plan_drift"] = round(gov["plan"]["drift"], 4)
        row["input_flags"] = len(gov["input_flags"])
        row["audit_verified"] = int(gov["audit"]["verified"])
        row["_plaus_scores"] = gov["plausibility_scores"]
    if item.metadata.get("kb_qid"):
        # for a policy-question base case the answer to that question IS the outcome
        row["correct"] = int(row["correct"] and row.get("kb_correct", 0))
    return row


def run(scenarios: list[Scenario], runner, limits: Limits, workflow, validator=None,
        faults=None, keep_transcripts: int = 0) -> list[dict[str, Any]]:
    rows = []
    for i, scn in enumerate(scenarios):
        proof = scn.metadata.get("proof")
        if proof is not None and not proof["accepted"]:
            continue
        path = validator.base_path(scn.base) if validator else None
        res = runner.run(scn, faults=faults, oracle_path=path)
        row = score(scn, res, limits, workflow)
        if i < keep_transcripts or not row["correct"]:
            row["transcript"] = res.extras["transcript"]
        rows.append(row)
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    out = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / n,
           "workflow_adherence": sum(r["workflow_adherent"] for r in rows) / n,
           "clean_termination": sum(r["clean_termination"] for r in rows) / n,
           "runtime_invariance": sum(r["invariance_ok"] for r in rows) / n}
    kb = [r["kb_correct"] for r in rows if "kb_correct" in r]
    if kb:
        out["kb_accuracy"] = sum(kb) / len(kb)
    for stage in WORKFLOW_STAGES:
        vals = [r[f"c_{stage}"] for r in rows if f"c_{stage}" in r]
        if vals:
            out[f"component_{stage}"] = sum(vals) / len(vals)
    return out


def weak_link(rows: list[dict[str, Any]], order: list[str] = WORKFLOW_STAGES) -> dict[str, int]:
    """Charge each failed run to the first stage, in workflow order, that erred (Ch. 10.7)."""
    counts: dict[str, int] = {}
    for r in rows:
        if r["correct"] and r["workflow_adherent"]:
            continue
        for stage in order:
            if r.get(f"c_{stage}") == 0:
                counts[stage] = counts.get(stage, 0) + 1
                break
        else:
            counts["(end decision / trajectory only)"] = counts.get("(end decision / trajectory only)", 0) + 1
    return counts
