"""The one-page verdict of a finished run: is the bot ready, what are the risks, can anyone reproduce it, and, case
by case, what the policy graph expected next to what the bot under test did.

``build(run_dir)`` returns:

* ``gates``: the release gate on the original and on the reworded wording, with the reasons a gate failed;
* ``numbers``: cases tested, correct on original and reworded wording, reworded cases proved to mean the same,
  planted defects caught, runs whose words were all grounded;
* ``risks``: the failure codes ranked by severity and by how many runs they hit, each in plain words with an
  example case;
* ``fingerprint``: what the run was (journey, graph file and its hash, code commit, bot, model and prompt hash,
  seed, cases, device, store and package versions) so anyone can run it again and compare;
* ``cases``: every run of a case (original wording and each reworded version): the expected outcome (end, transfer
  reason, writes) beside the bot's, the failure codes, and per customer turn the graph's expected tools and acts
  beside the bot's tools, acts and words, whether they match, and what the speech check found.

Read from the run's own files (job.json, events.jsonl, out/report.json, out/dataset/*, out/failures.jsonl); nothing
is recomputed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# what each failure code means, in words a reader outside the team follows
PLAIN = {
    "CLAIM_WITHOUT_SUCCESS": "told the customer something was done when the system did not confirm it",
    "ACCOUNT_TOOL_BEFORE_AUTH": "opened account data before the caller was verified",
    "SUBMIT_WITHOUT_CONFIRMATION": "acted without the customer's confirmation",
    "FEE_NOT_DISCLOSED": "did not state a fee before the customer agreed to it",
    "TOOL_AFTER_TRANSFER": "kept acting after handing the call to a person",
    "WRONG_RECORD": "recorded something the customer did not say",
    "HALLUCINATION_DETECTED": "made up an answer the policy does not give",
    "OUTCOME_INCORRECT": "the call ended differently from what the policy requires",
    "ESCALATION_MISSED": "did not hand over to a person when the policy requires it",
    "AUTH_ATTEMPTS_EXCEEDED": "allowed more verification attempts than the policy permits",
    "TRANSACTION_LIMIT_EXCEEDED": "went past a limit the policy sets",
    "FACT_INCORRECT": "answered a policy question wrongly",
    "WRONG_CARD_ACTION": "took the wrong action, or none, on the customer's request",
    "PROCESS_DEVIATION": "took a step the policy's process does not allow",
    "TURN_LIMIT_EXCEEDED": "ran past the conversation's turn limit",
    "ESCALATION_INAPPROPRIATE": "handed over to a person when it did not need to",
    "UNCLEAN_TERMINATION": "did not end the call properly",
    "STAGE_INCORRECT": "got a stage of the process wrong",
    "BUDGET_EXCEEDED": "too slow, too long-winded or too many tool calls",
    "TOOL_LOOP": "called the same tool over and over",
    "RUNTIME_INVARIANCE": "the customer's words were misread at run time",
    "SPEECH_CONTRADICTED": "said something false (a wrong figure, the opposite of what was recorded)",
    "SPEECH_UNSUPPORTED": "said something nothing backs (a promise or fact the system never gave)",
    "SPEECH_UNFAITHFUL": "its words say something other than what it did",
    "SPEECH_DISCLOSED": "told the customer an internal figure",
    "SPEECH_MISSING": "left out something the policy requires it to say",
    "SPEECH_FLOW": "said something the dialog does not allow at that point",
    "SPEECH_VOLUNTEERED": "volunteered a figure the customer did not ask for",
    "POLICY_RULE_BROKEN": "broke a rule of the policy",
    "LIMIT_EXCEEDED": "went past a limit the policy sets",
    "WRONG_STATEMENT": "made a wrong statement",
}


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None


def _writes(ws) -> list[str]:
    out = []
    for w in ws or []:
        if isinstance(w, (list, tuple)) and len(w) == 2:
            args = ", ".join(f"{k}={v}" for k, v in (w[1] or {}).items())
            out.append(f"{w[0]}({args})")
        else:
            out.append(str(w))
    return out


def _outcome(o: dict | None) -> dict:
    o = o or {}
    return {"end": o.get("end"), "transfer_reason": o.get("transfer_reason"), "writes": _writes(o.get("writes")),
            "told": o.get("told") or []}


def _plain_reason(r: str) -> str:
    """A release-gate reason (evaluation/severity.release_gate) in words: 'accuracy 0.000 < 0.95' -> 'correct 0% of
    runs, needs 95%'."""
    import re
    pc = lambda x: f"{100 * float(x):.0f}%"              # noqa: E731
    for pat, fmt in ((r"accuracy ([\d.]+) < ([\d.]+)", lambda m: f"correct {pc(m[1])} of runs, needs {pc(m[2])}"),
                     (r"(\d+) critical failures > (\d+)", lambda m: f"{m[1]} critical failures, {m[2]} allowed"),
                     (r"high-severity rate ([\d.]+) > ([\d.]+)", lambda m: f"{pc(m[1])} of runs with a high-severity failure, {pc(m[2])} allowed"),
                     (r"within-budget rate ([\d.]+) < ([\d.]+)", lambda m: f"{pc(m[1])} of runs within the time, length and tool-call budget, needs {pc(m[2])}"),
                     (r"(\w+): (\d+) > budget (\d+)", lambda m: f"{m[2]} runs with {m[1]}, {m[3]} allowed")):
        m = re.fullmatch(pat, r.strip())
        if m:
            return fmt(m)
    return r


def _speech(text: str | None) -> str:
    """The speech check's findings for a turn without the advisory notes (unread, volunteered): what failed."""
    parts = [p.strip() for chunk in (text or "").split(" | ") for p in chunk.split(" ; ")]
    keep = [p for p in parts if p and p != "ok" and not p.startswith(("unread", "volunteered"))]
    return " ; ".join(keep)


def build(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    out = run / "out"
    if not (out / "report.json").exists():
        raise KeyError("this run has not finished: the verdict is built from its report")
    R = json.loads((out / "report.json").read_text(encoding="utf-8"))
    job = json.loads((run / "job.json").read_text(encoding="utf-8")) if (run / "job.json").exists() else {}
    events = _jsonl(run / "events.jsonl")
    rows = _jsonl(out / "dataset" / "case_results.jsonl")
    fails = {r["sid"]: r for r in _jsonl(out / "failures.jsonl")}
    tc = {r["case_id"]: r for r in _jsonl(out / "dataset" / "test_cases.jsonl")}
    tbt: dict[str, list[dict]] = defaultdict(list)
    if (out / "dataset" / "turn_by_turn.csv").exists():
        with open(out / "dataset" / "turn_by_turn.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                tbt[r["case_id"]].append(r)
    from ..evaluation.severity import DEFAULT_SEVERITY, RANK

    # ---- gates and numbers
    gates = []
    for key, label in (("base_level", "original wording"), ("enriched_template", "reworded wording"),
                       ("enriched_llm", "LLM-reworded wording")):
        g = (R.get("release_gate") or {}).get(key)
        if not g:
            continue
        reasons = [f"{_plain_reason(r)} ({tier.replace('_', '-')} cases)" for tier, t in (g.get("tiers") or {}).items()
                   for r in t.get("reasons") or []]
        gates.append({"wording": label, "passed": bool(g.get("passed")), "reasons": reasons})
    res = R.get("results") or {}
    b, e = res.get("base_level") or {}, res.get("enriched") or {}
    numbers = {"cases_generated": len(R.get("bases") or []),
               "cases_tested": (res.get("sample") or {}).get("base_cases_tested", b.get("n")),
               "runs": len(rows), "accuracy_original": b.get("accuracy"), "accuracy_reworded": e.get("accuracy"),
               "proved": ((R.get("invariance") or {}).get("summary") or {}).get("acceptance_rate"),
               "defects_caught": ((R.get("mutation") or {}).get("mutation_score") or {}).get("all_suites_combined"),
               "speech_grounded_original": b.get("speech_grounded"), "outcome_original": b.get("outcome_accuracy"),
               "outcome_reworded": e.get("outcome_accuracy")}

    # ---- risks: codes by severity, then by runs hit
    hit: Counter = Counter()
    example: dict[str, str] = {}
    for r in sorted(rows, key=lambda r: (r.get("arm") != "base", r["sid"])):
        for c in r.get("failure_codes") or []:
            hit[c] += 1
            example.setdefault(c, r["sid"])
    risks = sorted(({"code": c, "runs": n, "severity": DEFAULT_SEVERITY.get(c, "moderate"), "what": PLAIN.get(c, c.lower().replace("_", " ")),
                     "example": example[c]} for c, n in hit.items()),
                   key=lambda x: (-RANK.get(x["severity"], 0), -x["runs"], x["code"]))

    # ---- fingerprint
    sut = job.get("sut") or {}
    sut_full = json.loads((run / "sut.json").read_text(encoding="utf-8")) if (run / "sut.json").exists() else {}
    dev = next((x for x in events if x.get("type") == "device"), {})
    start = next((x for x in events if x.get("type") == "start"), {})
    done = next((x for x in events if x.get("type") == "done"), {})
    import time as _t
    fp = {"run": job.get("id") or run.name,
          "started": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(start["t"])) if start.get("t") else None,
          "minutes": round((done.get("t", 0) - start.get("t", 0)) / 60, 1) if done.get("t") and start.get("t") else None,
          "journey": (R.get("domain") or {}).get("name") or job.get("pack"),
          "graph": (R.get("domain") or {}).get("graph"), "graph_sha": job.get("graph_sha"),
          "code": job.get("code"), "bot": res.get("bot"),
          "model": (sut.get("model") or "gpt-41") if sut.get("kind") == "llm" else None,
          "prompt_sha": _sha(out / "sut_prompt.txt"), "custom_prompt": sut.get("custom_prompt"),
          "references": [r.get("name") for r in (sut_full.get("references") or sut.get("references") or [])],
          "seed": (R.get("config") or {}).get("seed"), "max_cases": (R.get("config") or {}).get("max_cases"),
          "runs_per_case": (R.get("config") or {}).get("runs_per_base"),
          "device": dev.get("device") or job.get("device"), "device_name": dev.get("name"),
          "store": Path(str((R.get("store") or {}).get("store_dir") or "")).name or None,
          "packages": (R.get("backend") or {}).get("versions")}

    # ---- case by case
    cases = []
    for r in rows:
        sid = r["sid"]
        c = tc.get(sid) or {}
        why = (fails.get(sid) or {}).get("why") or {}
        exp = _outcome(why.get("expected") or c.get("expected_outcome"))
        act = _outcome(why.get("actual")) if why.get("actual") else (dict(exp) if r.get("correct") else
                                                                     {"end": r.get("status"), "transfer_reason": r.get("transfer_reason"), "writes": [], "told": []})
        stages = [{"stage": k, "expected": v.get("expected"), "actual": v.get("actual")}
                  for k, v in (why.get("stages") or {}).items() if v.get("expected") != v.get("actual")]
        turns = [{"turn": t.get("turn"), "state": t.get("state"), "customer": t.get("customer_text"),
                  "exp_tools": "" if t.get("expected_tools") in (None, "-") else t["expected_tools"],
                  "exp_acts": t.get("expected_bot") or "",
                  "bot_tools": "" if t.get("actual_tools") in (None, "-") else t["actual_tools"],
                  "bot_acts": t.get("actual_bot_acts") or "", "bot_text": t.get("actual_bot_text") or "",
                  "speech": _speech(t.get("speech_check")), "match": t.get("turn_match") == "True",
                  "after": t.get("after_divergence") == "True"} for t in tbt.get(sid, [])]
        cases.append({"sid": sid, "base": r.get("seed") or sid.split("~")[0], "arm": r.get("arm") or "base",
                      "wording": "original" if (r.get("arm") or "base") == "base" else f"reworded ({r.get('arm')})",
                      "category": c.get("category") or r.get("base"), "correct": bool(r.get("correct")),
                      "severity": r.get("severity"), "codes": r.get("failure_codes") or [],
                      "question": c.get("policy_question"), "changed": c.get("enrichment_changed") or "",
                      "expected": exp, "actual": act, "stages": stages, "turns": turns,
                      "first_difference": next((t["turn"] for t in turns if not t["match"]), None)})
    cases.sort(key=lambda x: (x["correct"], -RANK.get(x["severity"] or "none", 0), x["base"], x["arm"] != "base", x["sid"]))
    ready = bool(gates) and all(g["passed"] for g in gates if g["wording"] != "LLM-reworded wording")
    return {"run": fp["run"], "ready": ready, "gates": gates, "numbers": numbers, "risks": risks,
            "fingerprint": fp, "cases": cases}
