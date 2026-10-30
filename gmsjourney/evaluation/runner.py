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
  invariance_ok               runtime monitor: every realised turn read back to its frame,
  arm                         enrichment arm that wrote the customer turns (template | llm | base),
  bot_latency_ms_max/_p95, bot_words_max, within_budget, budget_violations
                              per-turn constraint budgets (evaluation/constraints.py; not part of `correct`),
  risk_tier, severity, failure_codes
                              gmsh-style severity with risk-tier escalation (evaluation/severity.py),
  c_speech, speech_*          what the bot SAID, read on its own and checked (gmsjourney/speech):
                              claims supported / contradicted / unsupported / unverifiable, words that
                              contradict the act labels, required information not said; c_flow and
                              speech_flow: acts where the dialog-state graph does not allow them.
                              SPEECH_GATE decides whether ``correct`` requires them.
Scenarios whose invariance proof failed are not run (they are reported).
"""
from __future__ import annotations

import json
from typing import Any

from ..core import Scenario
from ..oracle import Limits
from ..trace import WORKFLOW_STAGES
from .constraints import check_constraints
from .severity import classify, risk_tier

# How what the bot said enters ``correct``:
#   "report"  speech columns only; correct is the outcome as before
#   "fail"    correct also requires c_speech: no contradicted or unsupported claim, words that say what
#             the act labels claim, and every piece of information the policy requires (the default)
#   "strict"  as "fail", and every act where the dialog-state graph allows it (c_flow)
SPEECH_GATE = "fail"


def response_cards(speech_turns: list[dict[str, Any]], transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The bot's responses that said something checkable, one compact card each, for the reports.

    Input   the speech check's per-turn reports (``extras["speech"]["turns"]``, one per bot turn) and the run's
            transcript (``[{role, text, ...}]``, used for what the customer said just before each response)
    Output  ``[{turn, where, customer, text, acts, ok, flow_ok, claims, findings}]`` for every turn with a
            claim or a finding. Each claim keeps its words, the value they give, its verdict, where the truth was
            read (``basis``: tool result / successful write / what the customer said / GMS exact register /
            GMS stance check / policy text), the true value, and for policy figures the GMS reading (distance
            vs the relation's cap) and knowlytix's.
    """
    from ..speech.verify import turn_where
    before: dict[int, str] = {}
    k, last_user = -1, None
    for e in transcript:
        if e.get("role") == "user":
            last_user = e.get("text")
        elif e.get("role") == "bot":
            k += 1
            before[k] = last_user
    return [{"turn": t["turn"], "where": turn_where(t["turn"]), "customer": before.get(t["turn"]), "text": t["text"],
             "acts": t["acts"], "ok": t["ok"], "flow_ok": t["flow_ok"], "claims": t["claims"],
             "findings": t["findings"]}
            for t in speech_turns if t["claims"] or t["findings"]]


def score(scn: Scenario, res, limits: Limits, workflow, budget=None) -> dict[str, Any]:
    tr = res.extras["trace"]
    item = scn.base
    # the LLM arm is attributed on the levels that actually reached the bot (manipulation check)
    levels = scn.metadata.get("realised_levels") or scn.factor_levels
    row: dict[str, Any] = {
        "sid": scn.sid, "seed": item.qid, "base": item.base,
        "tags": item.metadata.get("tags", [item.base]),
        "arm": scn.metadata.get("arm", "base" if scn.mode == "base" else "template"),
        **{f"f_{k}": v for k, v in levels.items()},
        "correct": int(res.answer == item.answer),
        "status": res.status,
        "expected_transfer_reason": item.metadata.get("transfer_reason"),
        "transfer_reason": res.escalation_trigger,
    }
    stage_diff = {}
    for stage in WORKFLOW_STAGES:
        k = f"expected_{stage}"
        if k in item.components and stage in res.components:
            row[f"c_{stage}"] = int(res.components[stage] == item.components[k])
        elif k in item.components:
            row[f"c_{stage}"] = 0                     # expected but never produced
        if row.get(f"c_{stage}") == 0:
            stage_diff[stage] = {"expected": item.components[k], "actual": res.components.get(stage)}
    # the outcome signatures compared for `correct`, kept for the expected-vs-actual page (not written to CSV)
    row["signature"] = {"expected": _loads(item.answer), "actual": _loads(res.answer)}
    if res.answer != item.answer or stage_diff:      # what differs, for the failure report
        row["why"] = {"expected": _loads(item.answer), "actual": _loads(res.answer), "stages": stage_diff}
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
                             "detail": k["detail"]}
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
        q = gov.get("quorum") or {}
        row["gate_quorum_review"] = q.get("routed_to_review", 0)
        row["gate_strict_vs_quorum"] = len(q.get("disagreements", []))
        na = gov.get("numeric_attestations") or {}
        row["numeric_review"] = na.get("REVIEW", 0)
        row["numeric_fail"] = na.get("FAIL", 0)
    if item.metadata.get("kb_qid"):
        # for a policy-question base case the answer to that question IS the outcome
        row["correct"] = int(row["correct"] and row.get("kb_correct", 0))
    sp = res.extras.get("speech")
    if sp is not None:
        row.update(sp["summary"])
        if sp["problems"]:
            row["speech_problems"] = sp["problems"]
        row["speech_turns"] = response_cards(sp["turns"], res.extras.get("transcript") or [])
        row["outcome_correct"] = row["correct"]          # the outcome alone, before the speech gate
        row["speech_gate"] = SPEECH_GATE
        if SPEECH_GATE in ("fail", "strict"):
            row["correct"] = int(row["correct"] and row["c_speech"] and (SPEECH_GATE != "strict" or row["c_flow"]))
    if scn.metadata.get("unrealised_factors"):
        row["unrealised_factors"] = list(scn.metadata["unrealised_factors"])
    row.update(check_constraints(res.extras.get("bot_turn_stats", []), proc["tool_calls"],
                                 item.metadata.get("oracle_tool_calls"), budget))
    row["risk_tier"] = risk_tier(item)
    sev, codes = classify(row, row["risk_tier"])
    row["severity"] = sev
    row["failure_codes"] = [c["code"] for c in codes]
    return row


def _loads(s):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except ValueError:
        return s


def progress_bar(total: int, desc: str, show: bool = True):
    """A tqdm bar (notebook-aware) when tqdm is installed and show is True, else None."""
    if not show or not total:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True)


def run(scenarios: list[Scenario], runner, limits: Limits, workflow, validator=None,
        faults=None, keep_transcripts: int = 0, desc: str | None = None, show: bool = True) -> list[dict[str, Any]]:
    """Run and score every scenario; with `desc`, a progress bar (tqdm) shows runs done and accuracy so far."""
    rows = []
    bar = progress_bar(len(scenarios), desc, show) if desc else None
    for i, scn in enumerate(scenarios):
        proof = scn.metadata.get("proof")
        if proof is not None and not proof["accepted"]:
            if bar is not None:
                bar.update(1)
            continue
        path = validator.base_path(scn.base) if validator else None
        res = runner.run(scn, faults=faults, oracle_path=path)
        row = score(scn, res, limits, workflow, getattr(runner, "budgets", None))
        if i < keep_transcripts or not row["correct"] or not row["workflow_adherent"]:
            row["transcript"] = res.extras["transcript"]
        rows.append(row)
        if bar is not None:
            bar.update(1)
            bar.set_postfix_str(f"correct {sum(r['correct'] for r in rows)}/{len(rows)}")
    if bar is not None:
        bar.close()
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    out = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / n,
           "workflow_adherence": sum(r["workflow_adherent"] for r in rows) / n,
           "clean_termination": sum(r["clean_termination"] for r in rows) / n,
           "runtime_invariance": sum(r["invariance_ok"] for r in rows) / n,
           "within_budget": sum(r.get("within_budget", 1) for r in rows) / n}
    kb = [r["kb_correct"] for r in rows if "kb_correct" in r]
    if kb:
        out["kb_accuracy"] = sum(kb) / len(kb)
    sp = [r for r in rows if "c_speech" in r]
    if sp:
        out["speech_grounded"] = sum(r["c_speech"] for r in sp) / len(sp)
        out["speech_flow"] = sum(r["c_flow"] for r in sp) / len(sp)
        out["outcome_accuracy"] = sum(r["outcome_correct"] for r in sp) / len(sp)
        claims = sum(r["speech_claims"] for r in sp)
        out["speech_claims"] = claims
        out["speech_claims_supported"] = sum(r["speech_supported"] for r in sp) / max(1, claims)
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
