"""The campaign for a compact policy graph: the twelve steps of pipeline.run_campaign, with every
domain-specific component mined from the graph.

 1  GMS: the graph compiled (graph/compile.py), then the knowlytix build of pipeline.build_gms
 2  policy-question bases mined from the graph's figures, two-way admission (graph/kb.py)
 3  journey bases mined from the state machine: paths + failure perturbations (graph/journeys.py)
 4  catalogs + application layer; design                       (shared)
 5  invariance proof + negative controls mined from the graph  (graph/invariance.py)
 6  runtime gates calibrated, the graph's rules as call checks (graph/governance.py)
 7  run the bot under test (default: the graph's reference bot, graph/bot.py) with the monitor on
 8  attribution                                                (shared)
 9  resilience under tool faults on every tool of the graph    (shared probe)
10  mutation analysis with the mutants mined from the graph    (graph/bot.py)
11  training corpora                                           (not yet for graph packs)
12  outputs
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from ..core import Scenario
from ..journey import NA


def run_graph_campaign(cfg, log: Callable[[str], None] = print) -> dict[str, Any]:
    from .. import domain as _domain
    from ..catalog import Catalog, level_provenance_table, resolve, resolve_profile, validate
    from ..design.doe import compare_designs, compose, per_base_dispersion, power_curve
    from ..enrich.reader import FrameReader
    from ..enrich.realizers import Realizer
    from ..evaluation import attribution, mutation, resilience
    from ..evaluation.constraints import ConstraintBudget
    from ..evaluation.runner import run, summary
    from ..evaluation.severity import release_gate
    from ..knowledge.geode import build_store, oracle_property, to_dot
    from ..ontology import Ontology
    from ..pipeline import _failure_modes, _runtime_summary, _write_rows, build_gms, failed_case_records, sample_bases
    from .bot import make_graph_factory, mutants
    from .invariance import GraphInvarianceValidator
    from .journeys import kb_base, mine_paths, perturbations, to_item
    from .kb import mine_kb_items
    from .machine import GraphOracle
    from .runner import GraphRunner

    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dom = _domain.current()
    g = dom.graph
    R: dict[str, Any] = {"config": {k: v for k, v in asdict(cfg).items() if k != "bot_factory"},
                         "domain": {"name": dom.name, "dir": str(dom.dir), "graph": dom.graph_path.name, "engine": "graph"},
                         "skipped": {}}
    policy_path = cfg.policy_path or str(dom.graph_path)

    # 1 -------------------------------------------------------------- store
    store, build_report, R["backend"] = build_gms(policy_path, kx_store_dir=cfg.kx_store_dir, rebuild=cfg.rebuild_store,
                                                  seed=cfg.seed, epochs=cfg.kx_epochs, geode_epochs=cfg.kx_geode_epochs,
                                                  max_iters=cfg.kx_geode_max_iters,
                                                  embed_trainer_epochs=cfg.kx_embed_trainer_epochs,
                                                  embed_max_iters=cfg.kx_embed_max_iters, sft_epochs=cfg.kx_sft_epochs,
                                                  contradiction_epochs=cfg.kx_contradiction_epochs,
                                                  dedup_mode=cfg.kx_dedup_mode, v_sft_mode=cfg.kx_v_sft_mode,
                                                  store_config=cfg.kx_store_config, log=log)
    demo = build_store(policy_path, inject_misread=True)
    sdir = Path(build_report["store_dir"])
    (out / "gms_build_report.json").write_text(json.dumps(build_report, indent=1, default=str))
    (out / "policy.gms.md").write_text((sdir / "policy.gms.md").read_text())
    (out / "store_graph.dot").write_text(to_dot(store, {"enables"}))
    lk = next(k for k, lv in g.limits.items() if not lv.get("internal"))
    v = store.lookup_enm("limit", lk)
    vs0 = next(iter(g.values))
    m0 = g.members(vs0)
    R["store"] = {**store.summary(), "store_dir": str(sdir), "build": build_report,
                  "injected_misread_caught": any(c.action == "drop_conflict" and "misread" in str(c.source)
                                                 for c in demo.corrections),
                  "compiler_corrections": [{"action": c.action, "fact": c.fact, "reason": c.reason} for c in demo.corrections],
                  "oracle_property": oracle_property(store),
                  "primitive_examples": {
                      f"lookup_enm(limit, {lk})": v,
                      f"score_triple({g.root}, has_{lk}, {v})": store.score_triple(g.root, f"has_{lk}", str(v)),
                      f"score_triple({g.root}, has_{lk}, {v * 7 + 3})": store.score_triple(g.root, f"has_{lk}", str(v * 7 + 3)),
                      f"tau(has_{lk})": store.tau(f"has_{lk}"),
                      f"tension({m0[0]}, {m0[1]})": store.tension_energy(m0[0], m0[1]) if len(m0) > 1 else None},
                  "calibration": {"tension": store.calibration.get("tension"),
                                  "per_relation": store.calibration.get("score_triple")}}
    try:
        from ..governance.calibrate import tau_bootstrap
        R["store"]["tau_bootstrap"] = tau_bootstrap(store, B=cfg.calibration_bootstrap)
    except Exception as e:                                   # noqa: BLE001
        R["skipped"]["tau_bootstrap"] = repr(e)[:200]
    kg = store.kg
    R["policy_kg"] = {**kg.summary(), "consistency": {k: v for k, v in kg.consistency(store).items() if k != "enm"}}
    kg.save(out / "policy_kg.json")
    ont = Ontology.from_store(store)
    oracle = GraphOracle(store)

    # 2 -------------------------------------------------------------- policy questions
    log("[2/12] mining policy-question bases from the graph's figures (two-way admission)")
    kb_items, kb_info = mine_kb_items(store, cfg.seed)
    if not kb_info["self_tests"]["passed"]:
        msg = ("the store failed its self-tests (book Ch. 4.3), so nothing can be scored against it: "
               + json.dumps(kb_info["self_tests"], default=str)[:1500])
        if not getattr(cfg, "allow_failed_self_tests", False):
            (out / "self_tests_failed.json").write_text(json.dumps(kb_info["self_tests"], indent=1, default=str))
            raise RuntimeError(msg)
        log("      WARNING: " + msg[:300] + " (continuing: allow_failed_self_tests=True)")
    admitted = [k for k in kb_items if k.admission != "rejected"]
    R["kb"] = {"items": [{"qid": k.qid, "category": k.category, "frame": str(k.frame), "answer": k.answer,
                          "primitive": k.primitive, "admission": k.admission, "recovery": k.recovery,
                          "recovery_detail": k.recovery_detail} for k in kb_items], **kb_info}

    # 3 -------------------------------------------------------------- journeys mined from the machine
    from . import factors as FA
    fcfg = FA.load(dom)
    factor_info = None
    if fcfg:
        # the pack's base factors: a covering array over them, closed on the graph
        log(f"[3/12] base cases from base_factors.yaml: a strength-{fcfg.get('strength', 2)} covering array over "
            f"{len(fcfg['factors'])} factors, closed on the graph")
        errs = FA.check(fcfg)
        if errs:
            raise _domain.DomainError("base_factors.yaml:\n  - " + "\n  - ".join(errs))
        cases, factor_info = FA.generate(oracle, fcfg, [k.frame for k in admitted], seed=cfg.seed)
        closure_paths, cinfo = FA.closure(oracle, cases, fcfg.get("closure_budget", 3))
        factor_info["closure"] = cinfo
        bases = []
        firsts = {f: next(iter(fcfg["factors"][f]["levels"])) for f in fcfg["factors"]}
        for i, c in enumerate(cases):
            it = to_item(c.built, oracle, f"J{i:03d}", category=c.category())
            it.metadata["relevant"] = {f: (lv if c.rel.get(f) else NA) for f, lv in c.levels.items()}
            it.metadata["tags"] = [f"{f}={lv}" for f, lv in c.levels.items() if c.rel.get(f) and lv != firsts[f]] or ["default_path"]
            bases.append(it)
        for j, (sp, _) in enumerate(closure_paths):
            it = to_item(sp, oracle, f"C{j:03d}", category="closure")
            bases.append(it)
        paths, closure, pert = [], closure_paths, []
        kb_paths = [(cases[0].built, cases[0].trace)]
    else:
        log("[3/12] mining journey bases from the state machine (paths + failure perturbations)")
        paths = mine_paths(oracle, max_deviations=2)
        from .journeys import coverage_closure
        closure = coverage_closure(oracle, paths)
        paths = paths + closure
        pert = perturbations(oracle, paths, [k.frame for k in admitted])
        specs = [sp for sp, _ in paths] + [sp for sp, _ in pert]
        specs = list({s.key(): s for s in specs}.values())
        bases = [to_item(sp, oracle, f"J{i:03d}") for i, sp in enumerate(specs)]
        kb_paths = paths
    for k in admitted:
        bases.append(to_item(kb_base(kb_paths, k.frame), oracle, k.qid, category=k.category, kb_item=k))
    edges_all = {e for b in bases for e in b.metadata["edges"]}
    rules_all = {r for b in bases for r in b.metadata["rules_fired"]}
    states_all = {s for b in bases for s in b.metadata["states"]}
    checked = {rid for rid, _, _ in g.rule_checks()}          # machine-checked on every run (admissibility)
    R["coverage"] = {"source": "base_factors.yaml" if fcfg else "mined paths", "factors": factor_info,
                     "paths": len(paths), "closure_paths": len(closure), "perturbations": len(pert), "bases": len(bases),
                     "states": {"covered": len(states_all), "total": len(g.states),
                                "missing": sorted(set(g.states) - states_all)},
                     "workflow_edges": {"covered": len(edges_all), "total": len(store.workflow),
                                        "missing": sorted({w.wid for w in store.workflow} - edges_all)},
                     "rules": {"covered": len(rules_all | checked), "total": len(g.rules),
                               "fired": len(rules_all), "checked_every_run": sorted(checked - rules_all),
                               "missing": sorted(set(g.rules) - rules_all - checked)},
                     "transfer_reasons": sorted({b.metadata["transfer_reason"] for b in bases if b.metadata["transfer_reason"]}),
                     "categories": {c: sum(b.base == c for b in bases) for c in sorted({b.base for b in bases})}}
    R["bases"] = [{"qid": b.qid, "category": b.base, "tags": b.metadata["tags"], "query": b.query,
                   "relevant": b.metadata["relevant"], "expected": json.loads(b.answer),
                   "turns": b.metadata["oracle_turns"], "rules": b.metadata["rules_fired"],
                   "edges": b.metadata["edges"]} for b in bases]
    if fcfg:
        log(f"      factor pairs covered {factor_info['covered']}/{factor_info['feasible']} "
            f"({factor_info['infeasible']} pairs the policy never allows together)")
    log(f"      {len(bases) - len(admitted)} journey bases ({len(paths)} paths + {len(pert)} perturbations) + "
        f"{len(admitted)} policy-question bases; states {len(states_all)}/{len(g.states)}, "
        f"rules {len(rules_all | checked)}/{len(g.rules)}, workflow edges {len(edges_all)}/{len(store.workflow)}")

    # 4-5 ------------------------------------------------------------ suite + design
    log("[4/12] resolving the factor suite (official catalogs + application layer) and composing the design")
    cat = Catalog.load()
    errs = validate(cat)
    suite = resolve_profile(cat, cfg.profile)
    ls = suite.level_specs()
    n_runs = cfg.runs_per_base * len(bases)
    scenarios = compose(ls, bases, n_runs, cfg.seed, mode="embedded", assign="dispersed")
    shuffled = compose(ls, bases, n_runs, cfg.seed, mode="embedded", assign="shuffle")
    R["design"] = {"catalog": cat.summary(), "catalog_errors": errs, "suite": suite.summary(),
                   "official_catalogs": cat.raw.get("official_origin"), "level_provenance": level_provenance_table(cat),
                   "factors": suite.factor_names, "levels": dict(ls),
                   "dropped": resolve(cat, ["*"], list(cat.factors)).dropped_factors,
                   "moved_to_base": cat.app["moved_to_base"], "n_runs": n_runs,
                   "quality": compare_designs(ls, n_runs, cfg.seed),
                   "per_base_dispersion": {"dispersed": per_base_dispersion(scenarios, ls),
                                           "shuffle": per_base_dispersion(shuffled, ls)}}
    if cfg.run_power:
        R["design"]["power"] = power_curve(ls, [n_runs // 4, n_runs // 2, n_runs], reps=120)

    # 5 -------------------------------------------------------------- invariance proof
    log("[5/12] proving base/enriched equivalence for every scenario")
    realizer, reader = Realizer(), FrameReader(store)
    validator = GraphInvarianceValidator(store, oracle, reader, realizer)
    proofs = [validator.prove(s) for s in scenarios]
    controls = validator.negative_controls(scenarios, per_type=cfg.negative_controls_per_type)
    R["invariance"] = {"summary": validator.summarize(proofs, suite.factor_names), "negative_controls": controls,
                       "rejected": [p.sid for p in proofs if not p.accepted],
                       "examples": [{"sid": p.sid, "levels": dict(p.levels), "turns": p.turns} for p in proofs[:12]]}
    items = {b.qid: b for b in bases}
    (out / "ontology.ttl").write_text(ont.to_turtle(validator.evidence_triples(proofs, items)))
    if cfg.llm_arm:
        R["skipped"]["llm_arm"] = "the LLM enrichment arm is not yet wired for graph packs"

    # 6 -------------------------------------------------------------- runtime gates
    gcfg = None
    if cfg.run_governance:
        log("[6/12] calibrating the runtime gates (Appendix C)")
        try:
            from ..governance.calibrate import calibrate_plan, calibrate_plausibility, policy_fingerprint, store_fingerprint
            from ..governance.monitor import GovernanceConfig
            cal = calibrate_plausibility(store)
            journeys = [[c["tool"] for c in oracle.run(b.metadata["spec"], b.metadata["fixture"]).tool_log] for b in bases]
            pc = calibrate_plan(store, journeys)
            gcfg = GovernanceConfig(theta=cal["theta"], drift_budget=pc["drift_budget"], plan_criterion=pc["criterion"],
                                    decision_rule=cfg.decision_rule)
            R["governance"] = {"plausibility_calibration": cal, "plan_calibration": pc,
                               "call_checks": dom.get("rules.calls"), "speech_checks": dom.get("rules.speech"),
                               "calibration_record": {"store_fingerprint": store_fingerprint(store),
                                                      "policy_fingerprint": policy_fingerprint(store),
                                                      "reference_scores": [],
                                                      "false_reject_rate": cal["regulated"]["false_rejects"] / max(1, cal["n_admit"])},
                               "engine": "forgeloop GMSPlausibilityGate + AuditLogger on the knowlytix store"}
        except Exception as e:                               # noqa: BLE001
            R["skipped"]["governance"] = traceback.format_exc()[-800:]
            log(f"      runtime gates skipped: {e!r}")
            gcfg = None

    # 7 -------------------------------------------------------------- run the bot under test
    log("[7/12] running the system under test (runtime monitor on)" if gcfg else "[7/12] running the system under test")
    factory = cfg.bot_factory or make_graph_factory(store)
    rk = {"store": store, "realizer": realizer, "reader": reader}
    budget = ConstraintBudget(**(cfg.budgets or {}))
    runner = GraphRunner(bot_factory=factory, governance=gcfg, budgets=budget, **rk)
    tested = sample_bases(bases, cfg.max_cases)
    ids = {b.qid for b in tested}
    base_scns = [Scenario(f"{b.qid}~base", b, {}, "base") for b in tested]
    run_scns = [s for s in scenarios if s.base.qid in ids]
    pb = cfg.progress
    base_rows = run(base_scns, runner, oracle.limits, store.workflow, None, desc="SUT · base cases", show=pb)
    rows = run(run_scns, runner, oracle.limits, store.workflow, validator, desc="SUT · enriched", show=pb)
    bot_name = cfg.bot_name if cfg.bot_factory else "GraphBot (reference bot mined from the graph)"
    R["results"] = {"bot": bot_name, "base_level": summary(base_rows), "enriched": summary(rows), "budgets": budget.as_dict(),
                    "sample": {"max_cases": cfg.max_cases, "base_cases_tested": len(tested), "base_cases_total": len(bases),
                               "enriched_tested": len(run_scns), "enriched_total": len(scenarios)}}
    R["release_gate"] = {"base_level": release_gate(base_rows), "enriched_template": release_gate(rows)}
    if gcfg is not None:
        good = lambda r: r["correct"] and r["workflow_adherent"] and r.get("kb_correct", 1)
        R["governance"]["calibration_record"]["reference_scores"] = [x for r in base_rows if good(r)
                                                                     for x in r.get("_plaus_scores", [])]

    # 8 -------------------------------------------------------------- attribution
    log("[8/12] attributing failures (deviance tests; per component, weak link)")
    R["diagnosis"] = attribution.diagnose(rows, suite.factor_names)

    # 9 -------------------------------------------------------------- resilience
    probe_scns = [s for s in base_scns if not s.base.metadata.get("kb_qid") and s.base.metadata["oracle_tool_calls"] > 1]
    fault_tools = [t for t, sp in g.tools.items() if sp["role"] != "handoff"]
    if cfg.run_resilience:
        log("[9/12] probing resilience under tool faults")
        R["resilience"] = resilience.probe(runner, oracle, probe_scns[:40], tools=fault_tools, show=cfg.progress)

    # 10 ------------------------------------------------------------- mutation analysis
    if cfg.run_mutation and cfg.bot_factory is None:
        ms = mutants(g)
        log(f"[10/12] mutation analysis ({len(ms)} defects mined from the graph; runtime monitor watching)")
        journey_base = [s for s in base_scns if not s.base.metadata.get("kb_qid")]
        suites = {"mined_base": journey_base, "kb_bases": [s for s in base_scns if s.base.metadata.get("kb_qid")],
                  "enriched_design": run_scns}
        plan = tuple((t, k) for t in fault_tools for k in ("error", "plausible_wrong"))
        R["mutation"] = mutation.mutation_analysis(store, rk, suites, oracle.limits, store.workflow, validator,
                                                   fault_scenarios=probe_scns[:12], fault_plan=plan,
                                                   mutants={m: m.replace("_", " ") for m in ms}, governance=gcfg,
                                                   runner_cls=GraphRunner, factory=make_graph_factory)
        # a surviving defect that behaves exactly like the reference bot on every mined journey is equivalent:
        # no conversation the policy allows can tell them apart
        ref = GraphRunner(bot_factory=make_graph_factory(store), **rk)
        every = [Scenario(f"{b.qid}~base", b, {}, "base") for b in bases if not b.metadata.get("kb_qid")]
        ref_out = {s.sid: ref.run(s) for s in every}
        sig = lambda res: (res.answer, json.dumps(res.components, sort_keys=True), [c["tool"] for c in res.trajectory],
                           [a for e in res.extras["trace"].events if e["role"] == "bot" for a in
                            [json.dumps(x, sort_keys=True, default=str) for x in e["acts"]]])
        equivalent = []
        for m, e in R["mutation"]["mutants"].items():
            if e["killed"]:
                continue
            mr = GraphRunner(bot_factory=make_graph_factory(store, frozenset({m})), **rk)
            if all(sig(mr.run(s)) == sig(ref_out[s.sid]) for s in every):        # on every mined journey
                e["equivalent"] = True
                equivalent.append(m)
        n = len(R["mutation"]["mutants"])
        killed = sum(e["killed"] for e in R["mutation"]["mutants"].values())
        R["mutation"]["equivalent"] = equivalent
        R["mutation"]["mutation_score"]["non_equivalent"] = killed / max(1, n - len(equivalent))
    if gcfg is not None:
        try:
            R["governance"]["runtime"] = _runtime_summary(rows, R["governance"]["calibration_record"], store)
        except Exception as e:                               # noqa: BLE001
            R["skipped"]["runtime_summary"] = repr(e)[:200]

    # 11-12 ---------------------------------------------------------- outputs
    R["skipped"]["training_corpora"] = "the proofloop emitters are not yet wired for graph packs"
    log("[12/12] writing outputs")
    try:
        R["failure_modes"] = _failure_modes(R)
    except Exception as e:                                   # noqa: BLE001
        R["skipped"]["failure_modes"] = repr(e)[:200]
    _write_rows(out / "results_enriched.csv", rows)
    _write_rows(out / "results_base_level.csv", base_rows)
    with open(out / "scenarios.jsonl", "w") as f:
        for p in proofs:
            turns = [{k: v for k, v in t.items() if k not in ("frame_obj", "read_obj")} for t in p.turns]
            f.write(json.dumps({"sid": p.sid, "base": p.qid, "levels": p.levels, "accepted": p.accepted,
                                "signature": p.enriched_signature, "turns": turns}, default=str) + "\n")
    failed = [r for r in base_rows + rows if not r["correct"] or not r["workflow_adherent"]]
    with open(out / "failures.jsonl", "w") as f:
        for r in failed:
            f.write(json.dumps({**r, "arm": r.get("arm") or "base"}, default=str) + "\n")
    R["failed_cases"] = failed_case_records(failed, cfg.max_failed_cases_in_report)
    R["failed_total"] = len(failed)
    (out / "dataset").mkdir(exist_ok=True)
    with open(out / "dataset" / "case_results.jsonl", "w") as f:
        for r in base_rows + rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_plaus_scores"}, default=str) + "\n")
    with open(out / "bases.jsonl", "w") as f:
        for b in bases:
            tr = oracle.run(b.metadata["spec"], b.metadata["fixture"])
            f.write(json.dumps({"qid": b.qid, "category": b.base, "decisions": b.metadata["relevant"],
                                "expected": json.loads(b.answer), "components": b.components,
                                "events": [{k: (str(v) if k == "frame" else v) for k, v in e.items()} for e in tr.events]},
                               default=str) + "\n")
    R["elapsed_seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str))
    try:
        from .report import render_graph_html
        (out / "report.html").write_text(render_graph_html(R))
    except Exception:                                        # noqa: BLE001
        R["skipped"]["report_html"] = traceback.format_exc()[-1500:]
        (out / "report.json").write_text(json.dumps(R, indent=1, default=str))
    log(f"done in {R['elapsed_seconds']} s -> {out}")
    return R
