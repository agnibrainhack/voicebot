"""Campaign runs from the console: start, follow, summarise, read back.

Parts of this file:
  1. ``summarize`` / ``headline``: what each of the 13 steps produced, as metrics, tables and notes.
  2. ``index`` / ``detail``: a finished run's generated cases, for the case explorer.
  3. ``Runs``: start a campaign in its own process, read its events, list past runs, cancel one; devices.
  4. ``main``: the worker that runs one campaign (``python -m gmsjourney.s10_console.runs <run_dir>``), writing
     ``events.jsonl`` as it goes.
"""
from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


# =====================================================================================================================
# PART 1 - What each campaign step produced, in one shape the console renders: headline metrics, tables and notes.
#
# ``summarize(step, R)`` reads the campaign's results dict (``pipeline.run_campaign``) after a step has finished and
# returns ``{"metrics": [{label, value, hint?}], "tables": [{title, columns, rows}], "notes": [str]}``; large tables
# are cut to a few hundred rows. Every value is JSON-ready.
# =====================================================================================================================

STEPS = [
    ("1", "Knowledge store", "Load or build the knowlytix store from the journey's policy graph", ("store",)),
    ("2", "Policy questions", "Questions whose answers the graph proves (two-way admission)", ("kb",)),
    ("3", "Journey cases", "Cases from the journey factors: every feasible pair of levels, then closure", ("bases", "coverage")),
    ("4", "Rewording design", "How each case's customer turns are reworded (the enrichment factors)", ("design",)),
    ("5", "Meaning proof", "Every reworded case must read back to the same meaning and expected outcome", ("invariance",)),
    ("6", "Runtime gates", "Thresholds of the runtime monitor, calibrated on the graph", ("governance",)),
    ("7", "Bot under test", "Every case run against the bot; every utterance checked", ("results",)),
    ("8", "Attribution", "Which factors and stages explain the failures", ("diagnosis",)),
    ("9", "Tool faults", "What the bot does when a tool errors or returns a plausible wrong answer", ("resilience",)),
    ("10", "Planted defects", "Mutant bots each suite must catch", ("mutation",)),
    ("11", "Training data", "Corpora emitted from the proved cases", ("sft",)),
    ("12", "Outputs", "Report, release gate, datasets", ("dataset", "bot_validation", "release_gate")),
    ("13", "Verdict", "One page: release decision, top risks, how to reproduce; every case, expected beside actual", ("verdict",)),
]
MAX_ROWS = 300


def _pct(x: Any) -> str:
    return "–" if x is None else f"{100 * float(x):.1f}%"


def _m(label: str, value: Any, hint: str = "") -> dict[str, Any]:
    return {"label": label, "value": value, **({"hint": hint} if hint else {})}


def _t(title: str, columns: list[str], rows: list[list[Any]], note: str = "") -> dict[str, Any]:
    return {"title": title, "columns": columns, "rows": [[_cell(c) for c in r] for r in rows[:MAX_ROWS]],
            "total": len(rows), **({"note": note} if note else {})}


def _cell(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)


def done_steps(R: dict[str, Any]) -> list[str]:
    """The steps whose results are in ``R``."""
    return [k for k, _, _, keys in STEPS if any(x in R for x in keys)]


def summarize(step: str, R: dict[str, Any]) -> dict[str, Any]:
    """What one campaign step produced (metrics, tables, notes), from the campaign's result dict; a summary never
    breaks the run.
    """
    fn = globals().get(f"_s{step}")
    try:
        return clean(fn(R)) if fn else {"metrics": [], "tables": [], "notes": []}
    except Exception as e:                       # a summary never breaks the run
        return {"metrics": [], "tables": [], "notes": [f"could not summarise this step: {type(e).__name__}: {e}"]}


def _out(metrics=(), tables=(), notes=()) -> dict[str, Any]:
    return {"metrics": [m for m in metrics if m], "tables": list(tables), "notes": [n for n in notes if n]}


# ----------------------------------------------------------------------------- per step
def _s1(R):
    st, b = R.get("store") or {}, (R.get("store") or {}).get("build") or {}
    kg = R.get("policy_kg") or {}
    geode, store, emb = b.get("geode") or {}, b.get("store") or {}, b.get("embed_loop") or {}
    metrics = [_m("Graph triples", kg.get("triples")), _m("Dialog states", kg.get("states")),
               _m("Transitions", kg.get("transitions")), _m("Rules", kg.get("rules")),
               _m("Store entities", st.get("kx_entities") or st.get("entities")),
               _m("Store triples", st.get("kx_triples") or st.get("triples")),
               _m("Exact figures (ENM)", st.get("kx_enm") or st.get("enm_entries"))]
    rows = []
    if geode:
        rows.append(["GEODE self-correction", f"{geode.get('n_proposed')} proposed, {geode.get('n_kept_fact_triples')} kept; "
                     f"converged={geode.get('converged')} in {geode.get('iterations')} iteration(s)"])
    if emb:
        h = emb.get("history") or [{}]
        rows.append(["Embedding SFT", f"{emb.get('iterations')} iteration(s), held-out accuracy {_pct(h[-1].get('heldout_acc'))}"])
    if store:
        rows.append(["Production store", f"{store.get('n_entities')} entities, {store.get('n_triples')} triples, "
                     f"{store.get('epochs')} epochs, geometry {store.get('geometry')}"])
    enc = b.get("encoders") or {}
    if enc:
        rows.append(["Encoders", str(enc.get("kind") or enc.get("source") or enc)[:200]])
    rows.append(["Store directory", st.get("store_dir")])
    by_kind = kg.get("by_kind") or {}
    return _out(metrics, [_t("How the store was made", ["Stage", "Result"], rows),
                          _t("Graph facts by kind", ["Kind", "Triples"], [[k, v] for k, v in by_kind.items()])])


def _s2(R):
    kb = R.get("kb") or {}
    items = kb.get("items") or []
    rows = [[i.get("qid"), i.get("category"), i.get("frame"), i.get("answer"), i.get("primitive"), i.get("admission")]
            for i in items]
    st = kb.get("self_tests") or {}
    return _out([_m("Questions mined", len(items)),
                 _m("Graph answers its own questions", _pct((kb.get("precondition") or {}).get("graph_baseline_accuracy"))),
                 _m("Store self-tests", "passed" if st.get("passed") else "failed")],
                [_t("Policy questions", ["Id", "Category", "Meaning", "Answer", "Primitive", "Admission"], rows)])


def _s3(R):
    bases = [b for b in R.get("bases") or [] if not str(b.get("qid", "")).startswith("kb")]
    cov = R.get("coverage") or {}
    factors = sorted({f for b in bases for f in (b.get("relevant") or {})})
    rows = []
    for b in bases:
        rel = b.get("relevant") or {}
        exp = b.get("expected") or {}
        ending = exp.get("transfer_reason") or exp.get("end") or ""
        rows.append([b.get("qid"), b.get("category"), b.get("turns"), ending,
                     len(exp.get("writes") or []), *[rel.get(f, "") for f in factors]])
    from collections import Counter
    cats = Counter(b.get("category") for b in bases)
    return _out([_m("Journey cases", len(bases)), _m("Pairs covered", f"{cov.get('pairs_covered')}/{cov.get('pairs_required')}"),
                 _m("Rules reached", f"{len(cov.get('rules_covered') or [])}/{len(cov.get('rules_covered') or []) + len(cov.get('rules_missing') or [])}"),
                 _m("Escalation triggers", f"{len(cov.get('triggers_covered') or [])}/{len(cov.get('triggers_covered') or []) + len(cov.get('triggers_missing') or [])}"),
                 _m("Closure cases added", cov.get("n_closure"))],
                [_t("Cases and their factor levels (N/A: the level cannot matter in this case)",
                    ["Case", "Category", "Turns", "Ending", "Writes", *factors], rows),
                 _t("Cases by category", ["Category", "Cases"], [[k, v] for k, v in cats.most_common()])],
                [f"Rules no case reaches: {', '.join(cov['rules_missing'])}" if cov.get("rules_missing") else ""])


def _s4(R):
    d = R.get("design") or {}
    lv = d.get("levels") or {}
    prov = {p.get("factor"): p.get("origin") for p in d.get("level_provenance") or []}
    return _out([_m("Rewording factors", len(lv)), _m("Levels", sum(len(v) for v in lv.values()))],
                [_t("Rewording factors and levels", ["Factor", "Levels", "From"],
                    [[k, ", ".join(map(str, v)), prov.get(k, "")] for k, v in lv.items()])],
                [f"Catalog problems: {d['catalog_errors']}" if d.get("catalog_errors") else ""])


def _s5(R):
    inv = R.get("invariance") or {}
    s = inv.get("summary") or {}
    ctl = inv.get("negative_controls") or {}
    return _out([_m("Scenarios proved", f"{s.get('accepted')}/{s.get('scenarios')}"), _m("Acceptance", _pct(s.get("acceptance_rate"))),
                 _m("Turns checked", s.get("turns_checked")), _m("Mean surface similarity", s.get("surface_cosine_mean"),
                                                                  "lower: the rewording changed the words more")],
                [_t("Proof layers", ["Layer", "Scenarios"], [["Ground truth identical", s.get("L1_gt_identical")],
                                                              ["Meaning cards equivalent", s.get("L2_frames_equivalent")],
                                                              ["Oracle re-derived the same outcome", s.get("L3_oracle_rederived")]]),
                 _t("By rewording level", ["Level", "Scenarios", "Acceptance", "Mean surface similarity"],
                    [[k, v.get("scenarios"), _pct(v.get("acceptance")), v.get("mean_surface_cosine")]
                     for k, v in (s.get("by_level") or {}).items()]),
                 _t("Meaning-changing edits (must be rejected)", ["Edit", "Tried", "Rejected"],
                    [[k, v.get("n"), _pct(v.get("detection_rate"))] for k, v in ctl.items()
                     if isinstance(v, dict) and v.get("n")])])


def _s6(R):
    g = R.get("governance") or {}
    pc = g.get("plausibility_calibration") or {}
    best = (g.get("multi_threshold_design") or {}).get("best") or {}
    rows = [[k, v] for k, v in best.items() if not isinstance(v, (dict, list))]
    return _out([_m("Plausibility threshold", pc.get("theta")), _m("Separable", pc.get("separable")), _m("AUC", pc.get("auc"))],
                [_t("Chosen thresholds (best row of the design)", ["Setting", "Value"], rows)])


def _s7(R):
    res = R.get("results") or {}
    rows = []
    for arm in ("base_level", "enriched"):
        a = res.get(arm) or {}
        rows.append(["Original wording" if arm == "base_level" else "Reworded", a.get("n"), _pct(a.get("accuracy")),
                     _pct(a.get("outcome_accuracy")), _pct(a.get("speech_grounded")), _pct(a.get("speech_flow")),
                     _pct(a.get("kb_accuracy"))])
    comps = [(k[len("component_"):], v) for k, v in (res.get("base_level") or {}).items() if k.startswith("component_")]
    ecomps = res.get("enriched") or {}
    gate = R.get("release_gate") or {}
    spr = R.get("speech") or {}
    ctl = (spr.get("controls") or {}).get("controls") or {}
    pol = []
    for arm, label in (("base_level", "Original wording"), ("enriched", "Reworded")):
        a = spr.get(arm) or {}
        if a:
            pol.append([label, a.get("policy_checked"), a.get("policy_failed"), a.get("disclosed"), a.get("volunteered"),
                        a.get("unread")])
    ex = []
    for kind, xs in ((spr.get("base_level") or {}).get("advisory_examples") or {}).items():
        ex += [[kind, x.get("sid"), x.get("where"), x.get("why")] for x in xs[:6]]
    return _out([_m("Bot", res.get("bot")), _m("Accuracy, original", _pct((res.get("base_level") or {}).get("accuracy"))),
                 _m("Accuracy, reworded", _pct((res.get("enriched") or {}).get("accuracy"))),
                 _m("Release gate, original", "passed" if (gate.get("base_level") or {}).get("passed") else "not passed"),
                 _m("Release gate, reworded", "passed" if (gate.get("enriched_template") or {}).get("passed") else "not passed")
                 if gate.get("enriched_template") else None],
                [_t("Results", ["Wording", "Cases", "Correct", "Outcome right", "Words grounded", "Flow right", "Policy answers"], rows),
                 _t("Per stage", ["Stage", "Original", "Reworded"],
                    [[k, _pct(v), _pct(ecomps.get(f"component_{k}"))] for k, v in comps]),
                 _t("Policy in what the bot said", ["Wording", "Policy clauses checked", "Failed", "Disclosed",
                                                    "Volunteered (advisory)", "Unread (advisory)"], pol) if pol else None,
                 _t("Planted wording errors the speech check must flag", ["Control", "Flagged", "Applied", "Rate"],
                    [[k, v.get("detected"), v.get("applied"), _pct(v.get("detection_rate"))] for k, v in ctl.items()])
                 if ctl else None,
                 _t("Advisory findings (examples)", ["Kind", "Case", "Where", "What"], ex) if ex else None],
                [])


def _s8(R):
    d = R.get("diagnosis") or {}
    rows = [[a.get("factor"), a.get("deviance"), a.get("p_value_adj"), a.get("significant_adj")]
            for a in d.get("outcome_attribution") or []]
    wl = d.get("weak_link") or {}
    return _out([_m("Runs", d.get("n")), _m("Failures", d.get("failures"))],
                [_t("Failures by stage (weak link)", ["Stage", "Failures"], [[k, v] for k, v in wl.items()]) if isinstance(wl, dict) else None,
                 _t("Factors that explain failures", ["Factor", "Deviance", "p (adjusted)", "Significant"], rows)])


def _s9(R):
    rs = R.get("resilience") or []
    rows = [[r.get("tool"), r.get("kind"), r.get("reached"), r.get("detected"), r.get("silent"), _pct(r.get("detection_rate"))]
            for r in rs]
    return _out([_m("Faults probed", len(rs)), _m("Silent failures", sum(r.get("silent") or 0 for r in rs))],
                [_t("Faults", ["Tool", "Fault", "Reached", "Detected", "Silent", "Detection"], rows)])


def _s10(R):
    m = R.get("mutation") or {}
    sc = m.get("mutation_score") or {}
    rows = [[k, "caught" if v.get("killed") else "survived", "yes" if v.get("flagged_at_runtime") else "no"]
            for k, v in (m.get("mutants") or {}).items()]
    return _out([_m("Planted defects", m.get("n_mutants")), _m("Caught, all suites", _pct(sc.get("all_suites_combined"))),
                 _m("Flagged at runtime", _pct(m.get("runtime_monitor_catch_rate")))],
                [_t("Defects", ["Defect", "Tests", "Runtime monitor"], rows),
                 _t("Caught by suite", ["Suite", "Caught"], [[k, _pct(v)] for k, v in sc.items()])])


def _s11(R):
    s = R.get("sft") or {}
    rows = [[k, ", ".join(f"{a}={b}" for a, b in v.items() if not isinstance(b, (dict, list)))] for k, v in s.items()
            if isinstance(v, dict)]
    return _out([], [_t("Corpora", ["Corpus", "Counts"], rows)])


def _s12(R):
    ds = R.get("dataset") or {}
    gate = R.get("release_gate") or {}
    rows = [[t, "passed" if v.get("passed") else "not passed"] for t, v in ((gate.get("base_level") or {}).get("tiers") or {}).items()]
    return _out([_m("Test cases exported", ds.get("test_cases")), _m("Turn rows", ds.get("turn_rows")),
                 _m("Elapsed", f"{R.get('elapsed_seconds')} s" if R.get("elapsed_seconds") else "–")],
                [_t("Release gate by risk tier", ["Tier", "Verdict"], rows),
                 _t("Files", ["File"], [[f] for f in ds.get("files") or []])])


def headline(R: dict[str, Any]) -> dict[str, Any]:
    """The run's results at a glance (for the run list)."""
    res = R.get("results") or {}
    m = R.get("mutation") or {}
    smp = res.get("sample") or {}
    return {"bases": len(R.get("bases") or []), "tested": smp.get("base_cases_tested"),
            "accuracy_original": (res.get("base_level") or {}).get("accuracy"),
            "accuracy_reworded": (res.get("enriched") or {}).get("accuracy"),
            "proved": ((R.get("invariance") or {}).get("summary") or {}).get("acceptance_rate"),
            "defects_caught": (m.get("mutation_score") or {}).get("all_suites_combined"),
            "gate": ((R.get("release_gate") or {}).get("base_level") or {}).get("passed"),
            "gate_reworded": ((R.get("release_gate") or {}).get("enriched_template") or {}).get("passed")}


def clean(x: Any) -> Any:
    """Drop the None tables the summaries skip."""
    if isinstance(x, dict) and "tables" in x:
        x["tables"] = [t for t in x["tables"] if t]
    return x


def _s13(R: dict[str, Any]) -> dict[str, Any]:
    V = R.get("verdict") or {}
    if V.get("error"):
        return _out(notes=[f"The verdict could not be written: {V['error']}"])
    return _out([_m("Release decision", "Ready to release" if V.get("ready") else "Not ready to release"),
                 _m("Runs", V.get("runs")), _m("Failed runs", V.get("failed"))],
                [_t("Release gates", ["Wording", "Passed", "Why not"], [[g["wording"], g["passed"], "; ".join(g["reasons"])] for g in V.get("gates") or []]),
                 _t("Top risks", ["Severity", "What the bot did", "Code", "Runs", "Example"],
                    [[r["severity"], r["what"], r["code"], r["runs"], r["example"]] for r in V.get("risks") or []]) if V.get("risks") else None],
                ["Open the verdict for the one page and every case with the expected run beside the bot's; out/verdict.html is the same page as a file."])


# =====================================================================================================================
# PART 2 - The generated cases of a finished run, for the console's case explorer (step 11's data, read back).
#
# ``index(out)``   every case (one row per base case): category, tags, turns, its journey factor levels, how many
#                  reworded versions it has and how many were proved, and the bot's result on the original and the
#                  reworded wording where the case was run; plus the wording-factor counts, the coverage (factor-level
#                  pairs, rules, transfer triggers) and the journey factors' levels, so the browser can draw any grid.
# ``detail(out, base)``  one case: its meaning cards, the original words of every customer turn next to each reworded
#                  version (the wording factors applied, the proof's verdict per turn: the card read back, equivalent
#                  or not, surface similarity), the expected run, and per tested version the bot's result and its
#                  turn-by-turn comparison with the graph (expected vs actual tools and acts, the speech check).
#
# Everything comes from files the run wrote (dataset/test_cases.jsonl, case_results.jsonl, turn_by_turn.csv,
# scenarios.jsonl, failures.jsonl, report.json); nothing is recomputed.
# =====================================================================================================================

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
    """A finished run's cases, proofs and results, read from its output folder (cached by file time)."""
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
    """The case explorer's list: every base case with its reworded versions, wording factors and results."""
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
    """One base case in full: each version's turns (original and reworded words, the meaning read back, the proof's
    verdict) and the bot's result with its turn-by-turn comparison.
    """
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


# =====================================================================================================================
# PART 3 - The console's campaign runs: start one in its own process, read its events, list past runs, cancel.
#
# Each run is a folder ``<root>/<id>/`` holding ``job.json``, ``events.jsonl`` (the worker, part 4), ``console.log`` (everything
# the process printed), ``headline.json`` when it finished, and the campaign outputs under ``out/``.
# =====================================================================================================================

ROOT = Path(__file__).resolve().parents[2]


def devices() -> dict[str, Any]:
    """The compute devices this machine offers: cpu, and each CUDA GPU torch can see (with its name and memory)."""
    out: dict[str, Any] = {"options": [{"id": "auto", "label": "Auto (GPU when available)", "available": True},
                                       {"id": "cpu", "label": "CPU", "available": True}], "cuda": 0, "torch": None}
    try:
        import torch
        out["torch"] = torch.__version__
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        out["cuda"] = n
        for i in range(max(n, 4)):                 # cuda:0-3 always listed; unavailable ones are marked
            if i < n:
                p = torch.cuda.get_device_properties(i)
                out["options"].append({"id": f"cuda:{i}", "label": f"cuda:{i} · {p.name} · {p.total_memory / 2**30:.0f} GB",
                                       "available": True})
            else:
                out["options"].append({"id": f"cuda:{i}", "label": f"cuda:{i}", "available": False})
    except Exception as e:                         # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["options"] += [{"id": f"cuda:{i}", "label": f"cuda:{i}", "available": False} for i in range(4)]
    return out


def _provenance(pack: str) -> dict[str, Any]:
    from ..s09_reports.verdict import provenance
    return provenance(pack)


def check_device(dev: str | None) -> str | None:
    """None when the device can be used, else why not."""
    if not dev or dev in ("auto", "cpu"):
        return None
    if not re.fullmatch(r"cuda(:\d+)?", dev):
        return f"unknown device {dev!r}: use auto, cpu or cuda:N"
    opts = {o["id"]: o for o in devices()["options"]}
    if dev == "cuda":
        return None if devices()["cuda"] else "no CUDA GPU is visible to torch on this machine"
    if not (opts.get(dev) or {}).get("available"):
        return f"{dev} is not available on this machine ({devices()['cuda']} CUDA GPU(s) visible)"
    return None


class Runs:
    """The console's campaign runs: each starts as its own process (``python -m gmsjourney.s10_console.runs <dir>``), one
    at a time; events are read from ``events.jsonl``; a run outlives the server that started it.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ start / cancel
    def active(self) -> str | None:
        """The run in progress: one this server started, or one still alive from before a restart (its worker's
        pid is in its start event)."""
        with self.lock:
            for rid, p in list(self.procs.items()):
                if p.poll() is None:
                    return rid
        for run in sorted((d for d in self.root.iterdir() if d.is_dir()), reverse=True)[:5]:
            if self.status(run.name) == "running":
                return run.name
        return None

    def start(self, job: dict[str, Any]) -> str:
        busy = self.active()
        if busy:
            raise RuntimeError(f"run {busy} is still going: wait for it or cancel it first")
        why = check_device(job.get("device"))
        if why:
            raise ValueError(why)
        if (job.get("sut") or {}).get("provider") == "local" and (job.get("sut") or {}).get("kind") == "llm":
            from ..s06_bots.llm_bot import status as local_status
            st = local_status(job["sut"].get("local_path"), job["sut"].get("vllm_url"))
            if not st["usable"]:
                raise ValueError(f"the local model cannot be used: {st['detail']}")
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^a-z0-9_]+", "-", job["pack"].lower())
        run = self.root / rid
        run.mkdir(parents=True)
        sut = dict(job.get("sut") or {})
        if sut.get("kind") == "llm":                # the prompt and attachments live beside the job, not in it
            (run / "sut.json").write_text(json.dumps(sut, indent=1, ensure_ascii=False), encoding="utf-8")
            if sut.get("provider") == "local":
                from ..s06_bots.llm_bot import status as local_status
                sut["model"] = f"local-{local_status(sut.get('local_path'), sut.get('vllm_url'))['name']}"
                (run / "sut.json").write_text(json.dumps(sut, indent=1, ensure_ascii=False), encoding="utf-8")
            job = {**job, "sut": {"kind": "llm", "provider": sut.get("provider") or "gateway", "model": sut.get("model") or "",
                                  "mode": sut.get("mode") or "live",
                                  "local_path": str(sut.get("local_path") or "") if sut.get("provider") == "local" else "",
                                  "vllm_url": str(sut.get("vllm_url") or "") if sut.get("provider") == "local" else "",
                                  "critique": __import__("gmsjourney.s10_console.bots", fromlist=["x"]).critique_on(sut),
                                  "critic_model": str(sut.get("critic_model") or ""),
                                  "custom_prompt": bool(sut.get("custom_prompt")),
                                  "references": [{"name": r.get("name"), "chars": len(r.get("text") or "")}
                                                 for r in sut.get("references") or []]}}
        job = {**job, **_provenance(job["pack"])}
        (run / "job.json").write_text(json.dumps({**job, "id": rid, "created": time.time()}, indent=1))
        env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        if job.get("device") and job["device"] != "auto":
            env["GMSJOURNEY_DEVICE"] = job["device"]
        log = open(run / "console.log", "w")
        # its own process group, so cancelling stops everything the campaign started
        p = subprocess.Popen([sys.executable, "-m", "gmsjourney.s10_console.runs", str(run)], cwd=str(ROOT), env=env,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with self.lock:
            self.procs[rid] = p
        return rid

    def cancel(self, rid: str) -> bool:
        """Stop a run and every process it started (its process group); works for a run started before a restart."""
        with self.lock:
            p = self.procs.get(rid)
        pid = p.pid if p is not None and p.poll() is None else (self._pid(rid) if self.status(rid) == "running" else None)
        if pid is None:
            return False
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                break
            for _ in range(50):
                if not _alive(pid) or (p is not None and p.poll() is not None):
                    break
                time.sleep(0.2)
            if not _alive(pid) or (p is not None and p.poll() is not None):
                break
        with open(self.root / rid / "events.jsonl", "a") as f:
            f.write(json.dumps({"type": "cancelled", "t": time.time()}) + "\n")
        return True

    # ------------------------------------------------------------------ reading
    def _dir(self, rid: str) -> Path:
        if not re.fullmatch(r"[0-9A-Za-z_\-]+", rid) or not (self.root / rid).is_dir():
            raise KeyError(rid)
        return self.root / rid

    def events(self, rid: str, since: int = 0) -> dict[str, Any]:
        run = self._dir(rid)
        lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if (run / "events.jsonl").exists() else []
        evs = []
        for ln in lines[since:]:
            try:
                evs.append(json.loads(ln))
            except json.JSONDecodeError:              # a line still being written
                break
        return {"id": rid, "status": self.status(rid, lines), "events": evs, "next": since + len(evs)}

    def status(self, rid: str, lines: list[str] | None = None) -> str:
        run = self._dir(rid)
        if lines is None:
            lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if (run / "events.jsonl").exists() else []
        last = json.loads(lines[-1]).get("type") if lines else None
        for ln in reversed(lines):
            t = json.loads(ln).get("type")
            if t in ("done", "error", "cancelled"):
                return {"done": "finished", "error": "failed", "cancelled": "cancelled"}[t]
        with self.lock:
            p = self.procs.get(rid)
        if p is not None and p.poll() is None:
            return "running"
        pid = self._pid(rid, lines)
        if p is None and pid and _alive(pid):
            return "running"                       # started before this server (a kernel restart): still going
        return "stopped" if last else "starting"

    def _pid(self, rid: str, lines: list[str] | None = None) -> int | None:
        """The worker's pid, from the run's start event."""
        if lines is None:
            f = self.root / rid / "events.jsonl"
            lines = f.read_text(encoding="utf-8").splitlines()[:1] if f.exists() else []
        for ln in lines[:1]:
            try:
                return int(json.loads(ln).get("pid") or 0) or None
            except (ValueError, TypeError):
                return None
        return None

    def list(self) -> list[dict[str, Any]]:
        out = []
        for run in sorted((d for d in self.root.iterdir() if d.is_dir() and (d / "job.json").exists()), reverse=True):
            job = json.loads((run / "job.json").read_text())
            head = json.loads((run / "headline.json").read_text()) if (run / "headline.json").exists() else None
            if head is not None and "gate_reworded" not in head and (run / "out" / "report.json").exists():
                try:                                       # runs from before the reworded gate was shown: once
                    rep = json.loads((run / "out" / "report.json").read_text())
                    head["gate_reworded"] = ((rep.get("release_gate") or {}).get("enriched_template") or {}).get("passed")
                    (run / "headline.json").write_text(json.dumps(head, default=str))
                except (OSError, ValueError):
                    pass
            files = [f for f in ("report.html", "bot_validation.html", "case_comparison.html") if (run / "out" / f).exists()]
            out.append({"id": run.name, "job": job, "status": self.status(run.name), "headline": head, "files": files,
                        "elapsed": _elapsed(run / "events.jsonl")})
        return out

    def file(self, rid: str, rel: str) -> Path:
        """A file of the run's outputs (no escaping the run folder)."""
        base = (self._dir(rid) / "out").resolve()
        p = (base / rel).resolve()
        if base not in p.parents and p != base or not p.is_file():
            raise KeyError(rel)
        return p


# =====================================================================================================================
# Comparing two test runs: what changed between them, case by case.
#
# Each run keeps one row per case played (out/results_base_level.csv: original wording; out/results_enriched.csv:
# reworded). A case is identified by its sid, which depends only on the journey, the project and the seed, so two runs
# of the same journey with the same project and seed play the same cases and can be compared one to one. The headline
# numbers and the per-stage scores are compared as well, and the comparison says plainly when the two runs are not
# like for like (another journey, another graph version, other cases).
# =====================================================================================================================

STAGE_COLUMNS_PREFIX = "c_"


def _case_rows(run: Path) -> dict[str, dict]:
    out = {}
    for fname, wording in (("results_base_level.csv", "original"), ("results_enriched.csv", "reworded")):
        f = run / "out" / fname
        if not f.exists():
            continue
        import csv
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("sid"):
                    out[r["sid"]] = {**r, "_wording": wording}
    return out


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def compare(root: Path, a: str, b: str) -> dict[str, Any]:
    """What changed from run ``a`` to run ``b``: headline numbers, per-stage scores, and every case that flipped."""
    runs = {}
    for rid in (a, b):
        d = Path(root) / rid
        if not re.fullmatch(r"[0-9A-Za-z_\-]+", rid) or not (d / "job.json").exists():
            raise KeyError(rid)
        job = json.loads((d / "job.json").read_text())
        head = json.loads((d / "headline.json").read_text()) if (d / "headline.json").exists() else {}
        prov = {}
        try:
            prov = json.loads((d / "out" / "report.json").read_text()).get("provenance") or {}
        except (OSError, ValueError):
            pass
        runs[rid] = {"id": rid, "job": job, "headline": head, "graph_sha": prov.get("graph_sha"),
                     "commit": (prov.get("code") or {}).get("commit"), "rows": _case_rows(d)}
    A, B = runs[a], runs[b]
    if not A["rows"] or not B["rows"]:
        raise ValueError("both runs need case results (finished runs; a run that stopped early has none)")
    caveats = []
    if A["job"].get("pack") != B["job"].get("pack"):
        caveats.append(f"different journeys ({A['job'].get('pack')} and {B['job'].get('pack')}): few cases can match")
    if A["graph_sha"] and B["graph_sha"] and A["graph_sha"] != B["graph_sha"]:
        caveats.append(f"the policy graph changed between the runs ({A['graph_sha']} to {B['graph_sha']}): a flip can come "
                       "from the graph, not the bot")
    if A["job"].get("seed", 42) != B["job"].get("seed", 42):
        caveats.append("different seeds: the reworded cases differ")
    common = sorted(set(A["rows"]) & set(B["rows"]))
    only_a, only_b = sorted(set(A["rows"]) - set(B["rows"])), sorted(set(B["rows"]) - set(A["rows"]))
    if only_a or only_b:
        caveats.append(f"{len(only_a)} case(s) only in {a} and {len(only_b)} only in {b} (other case limits or another "
                       "project); they are left out of the case-by-case comparison")

    def case(sid: str) -> dict:
        ra, rb = A["rows"][sid], B["rows"][sid]
        return {"sid": sid, "base": rb.get("base") or ra.get("base"), "wording": rb["_wording"],
                "codes_a": ra.get("failure_codes") or "", "codes_b": rb.get("failure_codes") or "",
                "severity": rb.get("severity") or ra.get("severity") or ""}
    flips = {"now_failing": [], "now_passing": [], "still_failing": [], "still_passing": 0}
    for sid in common:
        pa, pb = _truthy(A["rows"][sid].get("correct")), _truthy(B["rows"][sid].get("correct"))
        if pa and not pb:
            flips["now_failing"].append(case(sid))
        elif pb and not pa:
            flips["now_passing"].append(case(sid))
        elif not pa:
            flips["still_failing"].append(case(sid))
        else:
            flips["still_passing"] += 1

    def rate(R: dict, col: str, wording: str | None = None, sids=None) -> float | None:
        vals = [r.get(col) for sid, r in R["rows"].items() if (sids is None or sid in sids) and (wording is None or r["_wording"] == wording)
                and r.get(col) not in (None, "")]
        try:
            vals = [float(_truthy(v)) if str(v).lower() in ("true", "false") else float(v) for v in vals]
        except ValueError:
            return None
        return sum(vals) / len(vals) if vals else None
    cs = set(common)
    stage_cols = sorted({k for R in (A, B) for r in R["rows"].values() for k in r if k.startswith(STAGE_COLUMNS_PREFIX)})
    stages = [{"stage": c[len(STAGE_COLUMNS_PREFIX):], "a": rate(A, c, sids=cs), "b": rate(B, c, sids=cs)} for c in stage_cols]
    stages = [x for x in stages if x["a"] is not None or x["b"] is not None]
    metrics = []
    for key, label in (("accuracy_original", "Correct, original wording"), ("accuracy_reworded", "Correct, reworded"),
                       ("proved", "Reworded cases proved"), ("defects_caught", "Planted defects caught")):
        metrics.append({"key": key, "label": label, "a": A["headline"].get(key), "b": B["headline"].get(key)})
    metrics += [{"key": "same_cases_original", "label": "Correct on the shared cases, original", "a": rate(A, "correct", "original", cs), "b": rate(B, "correct", "original", cs)},
                {"key": "same_cases_reworded", "label": "Correct on the shared cases, reworded", "a": rate(A, "correct", "reworded", cs), "b": rate(B, "correct", "reworded", cs)}]
    gates = [{"label": "Release gate, original", "a": A["headline"].get("gate"), "b": B["headline"].get("gate")},
             {"label": "Release gate, reworded", "a": A["headline"].get("gate_reworded"), "b": B["headline"].get("gate_reworded")}]
    if A["headline"].get("saved_total") or B["headline"].get("saved_total"):
        metrics.append({"key": "saved", "label": "Saved conversations without a deviation",
                        "a": (A["headline"].get("saved_passed") or 0) / A["headline"]["saved_total"] if A["headline"].get("saved_total") else None,
                        "b": (B["headline"].get("saved_passed") or 0) / B["headline"]["saved_total"] if B["headline"].get("saved_total") else None})
    side = lambda R: {"id": R["id"], "pack": R["job"].get("pack"), "bot": R["headline"].get("bot"), "sut": R["job"].get("sut"),
                      "graph_sha": R["graph_sha"], "commit": R["commit"], "cases": len(R["rows"]), "max_cases": R["job"].get("max_cases"),
                      "seed": R["job"].get("seed", 42)}
    return {"a": side(A), "b": side(B), "caveats": caveats, "shared": len(common), "only_a": len(only_a), "only_b": len(only_b),
            "metrics": metrics, "gates": gates, "stages": stages,
            "now_failing": flips["now_failing"], "now_passing": flips["now_passing"],
            "still_failing": flips["still_failing"], "still_passing": flips["still_passing"]}


def _alive(pid: int) -> bool:
    """True while a process with this pid exists (and is not a zombie)."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _elapsed(events) -> float | None:
    """Seconds from a run's first event to its last (for the run list and time estimates)."""
    try:
        lines = events.read_text(encoding="utf-8").splitlines()
        return round(json.loads(lines[-1])["t"] - json.loads(lines[0])["t"], 1) if len(lines) > 1 else None
    except (OSError, ValueError, KeyError, IndexError):
        return None


# =====================================================================================================================
# PART 4 - One campaign in its own process: ``python -m gmsjourney.s10_console.runs <run_dir>``.
#
# Reads ``<run_dir>/job.json`` (the journey, the device and the campaign settings), runs ``pipeline.run_campaign``
# and appends one JSON event per line to ``<run_dir>/events.jsonl`` as it goes:
#
#     start        the job as given
#     step_start   a step began (step, title)
#     step_done    a step finished, with what it produced (summaries.summarize: metrics, tables, notes)
#     log          a line the pipeline logged
#     progress     how far a long part has got (gmsjourney.progress): key, label, done, total, unit and details;
#                  the store's training phases, cases played, proofs, planted defects, LLM calls (at most two a
#                  second per bar, plus the last one)
#     saved        the journey's saved Playground conversations, replayed against this run's bot (bots.py part 4)
#     done         the run finished (headline numbers); the outputs are under <run_dir>/out
#     error        the run stopped (the traceback)
#
# A separate process keeps the console responsive, lets a run be cancelled, and frees the GPU when it ends.
# =====================================================================================================================

OPTIONS = ("max_cases", "runs_per_base", "seed", "run_resilience", "run_mutation", "run_power", "run_governance",
           "speech_controls", "calibration_bootstrap", "negative_controls_per_type", "llm_arm", "llm_mode", "llm_model",
           "rebuild_store", "profile")


def main(run_dir: str) -> int:
    """The worker: run the campaign of ``<run_dir>/job.json`` and append its events to ``events.jsonl``."""
    run = Path(run_dir)
    job = json.loads((run / "job.json").read_text())
    if job.get("device") and job["device"] != "auto":
        os.environ["GMSJOURNEY_DEVICE"] = job["device"]
    events = open(run / "events.jsonl", "a", buffering=1, encoding="utf-8")

    def emit(**e) -> None:
        e["t"] = round(time.time(), 3)
        events.write(json.dumps(e, default=str) + "\n")

    emit(type="start", job=job, pid=os.getpid())
    from .. import progress
    sent_at: dict[str, float] = {}

    def on_progress(**p) -> None:
        now = time.time()
        k = p.get("key", "")
        last = sent_at.get(k, 0.0)
        if not (p.get("final") or p.get("phase") or p.get("done") == p.get("total") or now - last >= 0.5):
            return
        sent_at[k] = now
        emit(type="progress", **p)
    progress.set_sink(on_progress)
    try:
        from ..s03_knowledge.build import device as _device
        dev = _device()                            # what "auto" resolved to on this machine
        name = None
        if dev.startswith("cuda"):
            import torch
            idx = int(dev.split(":")[1]) if ":" in dev else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
        emit(type="device", device=dev, name=name)
    except Exception as e:                         # noqa: BLE001 - the run reports it and carries on
        emit(type="log", text=f"could not resolve the device: {type(e).__name__}: {e}")
    try:
        from ..s08_campaign.pipeline import CampaignConfig, run_campaign
        sent: set[str] = set()

        def finished(R, before: str | None = None) -> None:
            # a later step's keys can appear early (step 7 already holds the release gate): only steps before the one
            # starting count as done, and every step counts once the campaign has returned
            lim = int(before) if before else 99
            for k in done_steps(R):
                if k not in sent and int(k) < lim:
                    sent.add(k)
                    emit(type="step_done", step=k, details=summarize(k, R))

        def on_step(step: str, title: str, R) -> None:
            progress.llm_flush()                            # the LLM counts so far, before the step changes
            finished(R, re.match(r"\d+", step).group(0))     # everything before this step is done
            emit(type="step_start", step=step, title=title)

        def log(msg, *a) -> None:
            emit(type="log", text=" ".join(str(x) for x in (msg, *a)))

        opts = {k: job[k] for k in OPTIONS if job.get(k) is not None}
        sut = job.get("sut") or {}
        if (run / "sut.json").exists():            # the full prompt and attachments (runs.Runs.start)
            sut = json.loads((run / "sut.json").read_text(encoding="utf-8"))
        fac = None
        if sut.get("kind") == "llm":               # the tester's LLM bot instead of the journey's reference bot
            from ..s01_packs.loading import set_current
            from ..s06_bots.llm_bot import CRITIC_VERSION
            from .bots import make_factory
            set_current(job["pack"])
            fac, name, _ = make_factory(sut)
            (run / "out").mkdir(parents=True, exist_ok=True)
            (run / "out" / "sut_prompt.txt").write_text(fac.system_prompt, encoding="utf-8")
            opts.update(bot_factory=fac, bot_name=name)
            refs = [r.get("name") for r in sut.get("references") or [] if r.get("text")]
            log(f"bot under test: {name}; prompt {len(fac.system_prompt):,} characters"
                + (f", with {', '.join(refs)} attached" if refs else "") + " (out/sut_prompt.txt); "
                + ("critique layer on: a second call reviews every step and may correct it (out/sut_critic_prompt.txt); "
                   if fac.critique else "")
                + 
                "planted defects (step 10) need the reference bot and are skipped")
        cfg = CampaignConfig(domain=job["pack"], out_dir=str(run / "out"), device=job.get("device") or None,
                             on_step=on_step, progress=False, **opts)
        try:
            R = run_campaign(cfg, log=log)
        finally:
            if fac is not None:                    # what the model was asked and answered, even when the run stops
                (run / "out" / "sut_llm.json").write_text(json.dumps(
                    {"bot": fac.stats.as_dict(), "llm": fac.llm.report(),
                     "critique": {"on": fac.critique, "version": CRITIC_VERSION, "revisions": fac.stats.revisions()}},
                    indent=1, default=str))
                if fac.critique:
                    (run / "out" / "sut_critic_prompt.txt").write_text(fac.critic_prompt, encoding="utf-8")
        finished(R)
        saved = _replay_saved(job, fac, (opts.get("bot_name") if fac is not None else None), run, log)
        if saved is not None:
            (run / "out" / "saved_conversations.json").write_text(json.dumps(saved, indent=1, default=str))
            emit(type="saved", results=saved)
        progress.llm_flush()
        head = headline(R)
        head["bot"] = (R.get("results") or {}).get("bot")
        if saved:
            head["saved_passed"], head["saved_total"] = sum(r["passed"] for r in saved), len(saved)
        (run / "headline.json").write_text(json.dumps(head, default=str))
        emit(type="done", headline=head)
        return 0
    except Exception:                              # noqa: BLE001
        emit(type="error", text=traceback.format_exc())
        return 1
    finally:
        events.close()


def _replay_saved(job: dict, fac, name: str | None, run: Path, log) -> list[dict] | None:
    """Replay the journey's saved Playground conversations against this run's bot (None when there are none)."""
    try:
        from ..s01_packs.loading import set_current
        from .. import progress
        from . import bots as PG
        set_current(job["pack"])
        if not PG.saved_list():
            return None
        log(f"replaying {len(PG.saved_list())} saved Playground conversation(s) against the bot under test")
        bar = progress.bar(len(PG.saved_list()), "saved conversations", show=False, key="saved", unit="conversations",
                           label="Saved conversations replayed")
        out: list[dict] = []

        def each(i, n, r):
            out.append(r)
            bar.update(1)
            bar.set_postfix_str(f"passed {sum(1 for x in out if x['passed'])}/{i}")
        PG.replay_all(None, run.parent, fac=fac, name=name, on_each=each)
        bar.close()
        log(f"saved conversations: {sum(r['passed'] for r in out)} of {len(out)} replayed without a deviation")
        return out
    except Exception as e:                         # noqa: BLE001 - the test run's results stand without them
        log(f"could not replay the saved conversations: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
