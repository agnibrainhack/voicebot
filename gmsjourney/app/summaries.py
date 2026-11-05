"""What each campaign step produced, in one shape the console renders: headline metrics, tables and notes.

``summarize(step, R)`` reads the campaign's results dict (``pipeline.run_campaign``) after a step has finished and
returns ``{"metrics": [{label, value, hint?}], "tables": [{title, columns, rows}], "notes": [str]}``; large tables
are cut to a few hundred rows. Every value is JSON-ready.
"""
from __future__ import annotations

from typing import Any

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
    fn = globals().get(f"_s{step}")
    try:
        return clean(fn(R)) if fn else {"metrics": [], "tables": [], "notes": []}
    except Exception as e:                       # a summary never breaks the run
        return {"metrics": [], "tables": [], "notes": [f"could not summarise this step: {type(e).__name__}: {e}"]}


def _out(metrics=(), tables=(), notes=()) -> dict[str, Any]:
    return {"metrics": list(metrics), "tables": list(tables), "notes": [n for n in notes if n]}


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
    sp = (R.get("speech") or {}).get("controls") or {}
    return _out([_m("Bot", res.get("bot")), _m("Accuracy, original", _pct((res.get("base_level") or {}).get("accuracy"))),
                 _m("Accuracy, reworded", _pct((res.get("enriched") or {}).get("accuracy"))),
                 _m("Release gate", "passed" if (gate.get("base_level") or {}).get("passed") else "not passed")],
                [_t("Results", ["Wording", "Cases", "Correct", "Outcome right", "Words grounded", "Flow right", "Policy answers"], rows),
                 _t("Per stage", ["Stage", "Original", "Reworded"],
                    [[k, _pct(v), _pct(ecomps.get(f"component_{k}"))] for k, v in comps])],
                [f"Speech check controls: {sp}" if sp and not isinstance(sp, dict) else ""])


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
            "gate": ((R.get("release_gate") or {}).get("base_level") or {}).get("passed")}


def clean(x: Any) -> Any:
    """Drop the None tables the summaries skip."""
    if isinstance(x, dict) and "tables" in x:
        x["tables"] = [t for t in x["tables"] if t]
    return x
