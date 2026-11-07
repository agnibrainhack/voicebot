"""The generated cases of a finished run, for the console's case explorer (step 11's data, read back).

``index(out)``   every case (one row per base case): category, tags, turns, its journey factor levels, how many
                 reworded versions it has and how many were proved, and the bot's result on the original and the
                 reworded wording where the case was run; plus the wording-factor counts, the coverage (factor-level
                 pairs, rules, transfer triggers) and the journey factors' levels, so the browser can draw any grid.
``detail(out, base)``  one case: its meaning cards, the original words of every customer turn next to each reworded
                 version (the wording factors applied, the proof's verdict per turn: the card read back, equivalent
                 or not, surface similarity), the expected run, and per tested version the bot's result and its
                 turn-by-turn comparison with the graph (expected vs actual tools and acts, the speech check).

Everything comes from files the run wrote (dataset/test_cases.jsonl, case_results.jsonl, turn_by_turn.csv,
scenarios.jsonl, failures.jsonl, report.json); nothing is recomputed.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@lru_cache(maxsize=4)
def _load(out: str, stamp: float) -> dict[str, Any]:
    o = Path(out)
    R = json.loads((o / "report.json").read_text(encoding="utf-8")) if (o / "report.json").exists() else {}
    cases = _jsonl(o / "dataset" / "test_cases.jsonl")
    results = {r["sid"]: r for r in _jsonl(o / "dataset" / "case_results.jsonl")}
    failures = {r["sid"]: r for r in _jsonl(o / "failures.jsonl")}
    proofs = {r["sid"]: r for r in _jsonl(o / "scenarios.jsonl")}
    tbt: dict[str, list[dict]] = defaultdict(list)
    if (o / "dataset" / "turn_by_turn.csv").exists():
        with open(o / "dataset" / "turn_by_turn.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                tbt[r["case_id"]].append(r)
    return {"R": R, "cases": cases, "results": results, "failures": failures, "proofs": proofs, "tbt": dict(tbt)}


def load(out: str | Path) -> dict[str, Any]:
    o = Path(out)
    if not (o / "dataset" / "test_cases.jsonl").exists():
        raise KeyError("this run has no generated cases yet (step 11 writes them)")
    return _load(str(o), (o / "dataset" / "test_cases.jsonl").stat().st_mtime)


def _result(r: dict | None) -> dict | None:
    if r is None:
        return None
    return {"sid": r["sid"], "correct": bool(r.get("correct")), "status": r.get("status"),
            "severity": r.get("severity"), "codes": r.get("failure_codes") or [],
            "transfer_reason": r.get("transfer_reason"), "expected_transfer_reason": r.get("expected_transfer_reason"),
            "outcome_correct": bool(r.get("outcome_correct")), "workflow_adherent": bool(r.get("workflow_adherent")),
            "speech_ok": bool(r.get("c_speech", 1)), "turns": r.get("turns"), "tools": r.get("tool_sequence") or ""}


def index(out: str | Path) -> dict[str, Any]:
    D = load(out)
    R, results = D["R"], D["results"]
    bases = {b["qid"]: b for b in R.get("bases") or []}
    by_base: dict[str, list[dict]] = defaultdict(list)
    for c in D["cases"]:
        by_base[c["base_id"]].append(c)
    wording: dict[str, Counter] = defaultdict(Counter)
    rows = []
    for bid, cs in by_base.items():
        base = next((c for c in cs if c["case_type"] == "base"), cs[0])
        rew = [c for c in cs if c["case_type"] == "enriched"]
        for c in rew:
            for k, v in (c.get("enrichment") or {}).items():
                wording[k][str(v)] += 1
        res_base = _result(results.get(f"{bid}~base"))
        res_rew = [_result(results[c["case_id"]]) for c in rew if c["case_id"] in results]
        b = bases.get(bid) or {}
        rows.append({"id": bid, "category": base.get("category"), "tags": base.get("tags") or [],
                     "turns": base.get("n_turns"), "question": base.get("policy_question"),
                     "first": ((base.get("base_case") or "").split(" || ")[0].split("] ", 1)[-1])[:160],
                     "levels": b.get("relevant") or {}, "rules": b.get("rules") or base.get("rules_fired") or [],
                     "end": (base.get("expected_outcome") or {}).get("end"),
                     "transfer": (base.get("expected_outcome") or {}).get("transfer_reason"),
                     "reworded": len(rew), "proved": sum(bool(c.get("proof_accepted")) for c in rew),
                     "arms": sorted({c["arm"] for c in rew}),
                     "tested": res_base is not None or bool(res_rew),
                     "base_result": res_base, "reworded_results": res_rew})
    rows.sort(key=lambda r: r["id"])
    cov = R.get("coverage") or {}
    levels: dict[str, list] = defaultdict(list)
    for r in rows:
        for k, v in r["levels"].items():
            if v not in levels[k]:
                levels[k].append(v)
    tested = [r for r in rows if r["tested"]]

    def acc(xs):
        xs = [x for x in xs if x is not None]
        return (sum(x["correct"] for x in xs) / len(xs)) if xs else None
    return {"run_bot": (R.get("results") or {}).get("bot"), "cases": rows, "levels": levels,
            "wording": {k: dict(v) for k, v in wording.items()},
            "coverage": {k: cov.get(k) for k in ("pairs_required", "pairs_covered", "pairs_infeasible", "rules_covered",
                                                 "rules_missing", "triggers_covered", "triggers_missing",
                                                 "boundaries_covered", "n_covering", "n_closure")},
            "totals": {"cases": len(rows), "reworded": sum(r["reworded"] for r in rows),
                       "proved": sum(r["proved"] for r in rows), "tested": len(tested),
                       "accuracy_original": acc([r["base_result"] for r in tested]),
                       "accuracy_reworded": acc([x for r in tested for x in r["reworded_results"]])}}


def _turns(case: dict) -> list[dict]:
    """The customer turns of a case: state, card and the words of this version."""
    out = []
    for s in case.get("trajectory") or []:
        if s.get("actor") == "customer":
            out.append({"turn": s.get("turn"), "state": s.get("state"), "card": s.get("card"), "text": s.get("text")})
    return out


def detail(out: str | Path, base_id: str) -> dict[str, Any]:
    D = load(out)
    cs = [c for c in D["cases"] if c["base_id"] == base_id]
    if not cs:
        raise KeyError(base_id)
    base = next((c for c in cs if c["case_type"] == "base"), cs[0])

    def version(c: dict) -> dict:
        proof = D["proofs"].get(c["case_id"]) or {}
        pt = {t.get("turn"): t for t in proof.get("turns") or []}
        turns = []
        for t in _turns(c):
            p = pt.get(t["turn"]) or {}
            turns.append({**t, "equivalent": p.get("equivalent"), "read_back": p.get("read_back"),
                          "similarity": p.get("surface_cosine")})
        res = D["results"].get(c["case_id"])
        fail = D["failures"].get(c["case_id"]) or {}
        tbt = [{k: r.get(k) for k in ("turn", "state", "customer_text", "expected_tools", "actual_tools", "expected_bot",
                                      "actual_bot_acts", "actual_bot_text", "speech_check", "turn_match", "after_divergence")}
               for r in D["tbt"].get(c["case_id"], [])]
        return {"id": c["case_id"], "type": c["case_type"], "arm": c["arm"], "enrichment": c.get("enrichment") or {},
                "changed": [x.split("=")[0] for x in (c.get("enrichment_changed") or "").split("; ") if "=" in x],
                "proof": {"status": c.get("proof_status"), "accepted": c.get("proof_accepted"), "layers": c.get("proof_layers"),
                          "similarity": c.get("surface_cosine_mean")},
                "turns": turns, "result": _result(res), "why": fail.get("why"), "transcript": fail.get("transcript"),
                "turn_by_turn": tbt}
    traj = []
    for s in base.get("trajectory") or []:
        if s["actor"] == "bot":
            traj.append({"actor": "bot", "acts": s.get("acts")})
        elif s["actor"] == "tool":
            res = s.get("result")
            traj.append({"actor": "tool", "tool": s.get("tool"), "args": s.get("args"),
                         "status": res.get("status") if isinstance(res, dict) else res})
        else:
            traj.append({"actor": "customer", "turn": s.get("turn"), "state": s.get("state"), "card": s.get("card")})
    return {"id": base_id, "category": base.get("category"), "tags": base.get("tags") or [],
            "question": base.get("policy_question"), "answer": base.get("policy_answer"),
            "expected": base.get("expected_outcome"), "rules": base.get("rules_fired") or [],
            "states": base.get("states_visited") or [], "cards": base.get("meaning_cards") or [],
            "trajectory": traj, "versions": [version(c) for c in sorted(cs, key=lambda c: (c["case_type"] != "base", c["case_id"]))]}
