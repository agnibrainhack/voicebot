"""End-to-end campaign: document -> oracle -> bases -> enrichment -> proof -> run -> diagnosis.

Each step maps to the book; where a component exists in the knowlytix / proofloop /
forgeloop packages it is used directly (backend.available()), and the built-in
implementation is the open-core fallback:

 1  GMS: knowlytix GEODE loop, embed loop (v SFT), contradiction SFT (u), cap-trained
    store, GMSJudge calibration (built-in: our compiler + Procrustes store)       Ch. 2-4
 2  policy-question bases; two-way admission via knowlytix QuestionGenerator;
    knowlytix production generators as the substrate self-test                     Ch. 5, nb 25
 3  journey bases: covering array + rule/edge/trigger/boundary closure            Ch. 5, 12.3
 4  catalogs (official, from proofloop) + application layer; design              Ch. 6-7
 5  invariance proof for every scenario + negative controls                       Ch. 1.2, 8.1.1
 6  runtime gates calibrated (Appendix C) with forgeloop's GMS gates              App. C, Ch. 12
 7  run the SUT under the runtime monitor; KB answers judged by GMSJudge          Ch. 9, 12.5
 8  attribution: knowlytix DOEAnalyzer + per-component / weak link                Ch. 10
 9  resilience under tool faults                                                  Ch. 11
10  mutation analysis (+ runtime-monitor catch rate)                              (beyond the book)
11  training corpora with proofloop's emitters                                    Ch. 8
12  failure-mode coverage (proofloop taxonomy), report                            Ch. 11

gmsh parity additions (0.4): GEODE dedup mode and v-SFT A/B (step 1); hallucination probes
(step 2); an optional LLM enrichment arm (the Microsoft model gateway) proved separately from the
template arm (step 5b); gmsh-style P=0.5 operating points, held-out accuracy and bootstrap
intervals next to the regulated ones (step 6); quorum REVIEW and the numeric range attestor in
the runtime monitor; per-turn latency/length budgets, severity with risk tiers and a tiered
release gate on every run (steps 7-8).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import backend
from .bases.journey import JourneyBaseGenerator, primary_category, to_item
from .bases.kb import mine_kb_items
from .catalog import Catalog, level_provenance_table, resolve, resolve_profile, validate
from .core import Scenario
from .design.doe import compare_designs, compose, per_base_dispersion, phi_p_design, power_curve
from .enrich.invariance import InvarianceValidator
from .enrich.reader import FrameReader
from .enrich.realizers import Realizer
from .evaluation import attribution, mutation, resilience
from .evaluation.constraints import ConstraintBudget
from .evaluation.runner import run, summary
from .evaluation.severity import release_gate
from .journey import NA, JourneySpec, relevant_dims
from .knowledge.geode import build_store, oracle_property, to_dot
from .ontology import Ontology
from .oracle import PolicyOracle
from .sut.adapter import JourneyRunner, make_reference_factory

POLICY = Path(__file__).resolve().parent / "policy" / "charge_review_kg.json"          # the policy graph (source)
POLICY_MD = Path(__file__).resolve().parent / "policy" / "charge_review_policy.md"   # its structured transcription


@dataclass
class CampaignConfig:
    policy_path: str = str(POLICY)
    out_dir: str = "output/campaign_out"
    profile: str = "voice_full"
    runs_per_base: int = 6
    seed: int = 42
    negative_controls_per_type: int = 40
    run_mutation: bool = True
    run_resilience: bool = True
    run_power: bool = True
    run_governance: bool = True
    bot_factory: Callable | None = None          # default: the reference bot
    bot_name: str = "ReferenceBot"
    backend: str = "auto"                        # auto | knowlytix | builtin
    kx_epochs: int = 150                         # production store epochs (licence cap 200)
    kx_geode_epochs: int = 80                    # GEODE correction-loop trainer epochs
    kx_store_dir: str | None = None              # default: ~/.cache/gmsjourney/stores/<hash>
    rebuild_store: bool = False
    kx_dedup_mode: str = "off"                   # off | audit | accept   (knowledge/kx_store.py)
    kx_v_sft_mode: str = "ab"                    # rotation | full | ab
    # runtime checks
    budgets: dict | None = None                  # ConstraintBudget fields (None = defaults)
    decision_rule: str = "strict"                # strict | quorum   (runtime gates)
    calibration_bootstrap: int = 200             # bootstrap resamples for theta / tau intervals
    # LLM enrichment arm (enrich/llm): a chat model writes the turns, proved separately
    llm_arm: bool = False
    llm_mode: str = "live"                       # live (call on cache miss) | replay (cache only)
    llm_model: str = ""                          # gateway deployment; empty: gpt-41
    llm_temperature: float = 0.7
    llm_share: float = 1.0                       # share of design rows also realised by the LLM (paired)
    llm_dual_read: str = "gate"                  # gate | report | off   (independent LLM reader)
    llm_max_attempts: int = 3
    llm_adversarial_controls: int = 20
    llm_cache_dir: str | None = None
    llm_workers: int = 8                         # concurrent gateway calls (prefetch); 1 = sequential


def build_gms(policy_path: str | Path = POLICY, *, backend_choice: str = "auto", kx_store_dir: str | None = None,
              rebuild: bool = False, epochs: int = 150, geode_epochs: int = 80, seed: int = 42,
              dedup_mode: str = "off", v_sft_mode: str = "ab", log: Callable[[str], None] = print):
    """The GMS for a policy: knowlytix (built once, then reloaded from its directory) when the
    package is installed and licensed, else the built-in store. Returns (store, report, backend)."""
    use_kx = backend_choice != "builtin" and backend.available()
    if backend_choice == "knowlytix" and not use_kx:
        backend.require()
    info = {"selected": "knowlytix" if use_kx else "builtin", **backend.describe()}
    if not use_kx:
        log("[1/12] building the built-in knowledge store (ingest -> self-correction -> train -> calibrate)")
        b = build_store(policy_path)
        return b.store, b.report(), info
    from .knowledge.kx_store import build_kx_store, load_kx_store
    # the store is keyed by everything that changes it: the policy source, the training settings,
    # the package versions and which encoders it is warm-started from
    key = hashlib.sha256(json.dumps([Path(policy_path).read_text(), epochs, geode_epochs, seed,
                                     info.get("versions"), backend.encoder_kind(), dedup_mode,
                                     v_sft_mode]).encode()).hexdigest()[:16]
    sdir = Path(kx_store_dir) if kx_store_dir else backend.CACHE / "stores" / key
    if (sdir / "gms_build_report.json").exists() and (sdir / "kx" / "model.pt").exists() and not rebuild:
        log(f"[1/12] loading the knowlytix GMS store ({sdir})")
        kb = load_kx_store(policy_path, sdir)
    else:
        log(f"[1/12] building the knowlytix GMS store: GEODE loop -> embed loop -> contradiction SFT -> "
            f"cap-trained store -> GMSJudge calibration ({sdir})")
        kb = build_kx_store(policy_path, sdir, epochs=epochs, geode_epochs=geode_epochs, seed=seed,
                            dedup_mode=dedup_mode, v_sft_mode=v_sft_mode, log=log)
    return kb.store, {**kb.report, "store_dir": str(sdir)}, info


def tags_for(spec: JourneySpec) -> list[str]:
    rel = relevant_dims(spec)
    tags = {primary_category(spec), "procedural_ordering"}
    if rel["auth"] in ("retry_ok", "fail_twice") or rel.get("disposition") in ("unsure_resolved", "unsure_twice",
                                                                                 "ambiguous_resolved", "ambiguous_twice"):
        tags.add("conditional_rule")
    if rel.get("n_txn") in (4, 5) or rel.get("window") in ("day30", "day31") or \
            rel.get("turn_pressure") in ("at_limit", "over_limit") or rel["auth"] == "fail_twice":
        tags.add("boundary")
    if rel["auth"] in ("unavailable",) or rel.get("retrieve") == "unavailable" or rel.get("record") == "fail_first" or \
            rel.get("card_options") == "unavailable" or rel.get("submit") == "fail" or \
            rel.get("disposition") in ("ambiguous_resolved", "ambiguous_twice", "unsure_twice"):
        tags.add("recovery")
    if rel["auth"] == "human_at_auth" or rel.get("human_mid") == "yes" or rel.get("card_intent") not in (NA, "none") \
            or rel.get("record") == "fail_first":
        tags.add("governance_policy")
    return sorted(tags)


def run_campaign(cfg: CampaignConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    R: dict[str, Any] = {"config": {k: v for k, v in asdict(cfg).items() if k != "bot_factory"}}

    # 1 -------------------------------------------------------------- store
    store, build_report, R["backend"] = build_gms(cfg.policy_path, backend_choice=cfg.backend,
                                                  kx_store_dir=cfg.kx_store_dir, rebuild=cfg.rebuild_store,
                                                  epochs=cfg.kx_epochs, geode_epochs=cfg.kx_geode_epochs,
                                                  seed=cfg.seed, dedup_mode=cfg.kx_dedup_mode,
                                                  v_sft_mode=cfg.kx_v_sft_mode, log=log)
    use_kx = R["backend"]["selected"] == "knowlytix"
    demo = build_store(cfg.policy_path, inject_misread=True, train=False)
    caught = any(c.fact == ("expedited_delivery", "has_fee_usd", "150") and c.action.startswith("drop")
                 for c in demo.corrections)
    if use_kx:
        sdir = Path(build_report["store_dir"])
        (out / "gms_build_report.json").write_text(json.dumps(build_report, indent=1, default=str))
        (out / "policy.gms.md").write_text((sdir / "policy.gms.md").read_text())
        R["store"] = {**store.summary(), "store_dir": str(sdir), "build": build_report}
    else:
        store.save(out / "gms_store")
        (out / "geode_build_report.json").write_text(json.dumps(build_report, indent=1, default=str))
        R["store"] = {**store.summary(), "candidates": build_report["n_candidates"],
                      "corrections": build_report["corrections_by_action"], "build": build_report}
    (out / "store_graph.dot").write_text(to_dot(store, {"enables", "precedes", "delivered_by", "has_fee_usd"}))
    R["store"].update({
        "injected_misread_caught": caught,
        "oracle_property": oracle_property(store),
        "primitive_examples": {
            "lookup_enm(limit, max_transactions_per_session)": store.lookup_enm("limit", "max_transactions_per_session"),
            "score_triple(policy, has_max_transactions_per_session, 4)": store.score_triple("charge_review_policy", "has_max_transactions_per_session", "4"),
            "score_triple(policy, has_max_transactions_per_session, 30)": store.score_triple("charge_review_policy", "has_max_transactions_per_session", "30"),
            "tau(has_max_transactions_per_session)": store.tau("has_max_transactions_per_session"),
            "tension(RECOGNIZED, UNRECOGNIZED)": store.tension_energy("RECOGNIZED", "UNRECOGNIZED"),
            "check_holonomy([precedes, precedes], precedes_two)": store.check_holonomy(["precedes", "precedes"], "precedes_two"),
            "check_holonomy([precedes, precedes], has_fee_usd)": store.check_holonomy(["precedes", "precedes"], "has_fee_usd"),
            "stance(unrecognized is proof of fraud)": store.stance_check("unrecognized_transaction", "is_proof_of", "fraud", +1)["verdict"],
        },
        "calibration": {"tension": store.calibration["tension"], "holonomy_tau": store.calibration["holonomy"]["tau_path"],
                        "per_relation": store.calibration["score_triple"]}})
    from .governance.calibrate import tau_bootstrap
    R["store"]["tau_bootstrap"] = tau_bootstrap(store, B=cfg.calibration_bootstrap)
    kg = getattr(store, "kg", None)
    if kg is not None:
        R["policy_kg"] = {**kg.summary(), "consistency": {k: v for k, v in kg.consistency(store).items()
                                                          if k != "enm"}}
        kg.save(out / "policy_kg.json")
    ont = Ontology.from_store(store)
    oracle = PolicyOracle(store)

    # 2 -------------------------------------------------------------- KB bases
    log("[2/12] mining policy-question bases (two-way admission" + ("; knowlytix substrate generators)" if use_kx else ")"))
    kb_items, kb_info = mine_kb_items(store, cfg.seed)
    admitted = [k for k in kb_items if k.admission != "rejected"]
    R["kb"] = {"items": [{"qid": k.qid, "category": k.category, "frame": str(k.frame), "answer": k.answer,
                          "primitive": k.primitive, "admission": k.admission, "recovery": k.recovery,
                          "recovery_detail": k.recovery_detail} for k in kb_items], **kb_info}

    # 3 -------------------------------------------------------------- journey bases
    log("[3/12] generating journey bases (covering array + closure)")
    gen = JourneyBaseGenerator(oracle, kb_pool=[k.frame for k in admitted], seed=cfg.seed)
    cover, rep = gen.covering()
    kb_traces = [oracle.run(JourneySpec(n_txn=1, kb_mid="ask", kb_frames=(k.frame,))) for k in admitted]
    extra = gen.closure(cover, store, rep, extra_traces=kb_traces)
    specs = list({s.key(): s for s in cover + extra}.values())
    bases = []
    for i, sp in enumerate(specs):
        it = to_item(sp, oracle, f"J{i:03d}")
        it.metadata["tags"] = tags_for(sp)
        bases.append(it)
    for k in admitted:
        it = to_item(JourneySpec(n_txn=1, kb_mid="ask", kb_frames=(k.frame,)), oracle, k.qid, category=k.category, kb_item=k)
        it.metadata["tags"] = [k.category]
        bases.append(it)
    R["coverage"] = asdict(rep)
    R["bases"] = [{"qid": b.qid, "category": b.base, "tags": b.metadata["tags"], "query": b.query,
                   "relevant": {k: v for k, v in b.metadata["relevant"].items() if v != NA},
                   "expected": json.loads(b.answer), "turns": b.metadata["oracle_turns"],
                   "rules": b.metadata["rules_fired"], "edges": b.metadata["edges"]} for b in bases]
    log(f"      {len(specs)} journey bases + {len(admitted)} policy-question bases")

    # 4-5 ------------------------------------------------------------ suite + design
    log("[4/12] resolving the factor suite (official catalogs + application layer) and composing the design")
    cat = Catalog.load()
    errs = validate(cat)
    suite = resolve_profile(cat, cfg.profile)
    ls = suite.level_specs()
    n_runs = cfg.runs_per_base * len(bases)
    scenarios = compose(ls, bases, n_runs, cfg.seed, mode="embedded", assign="dispersed")
    shuffled = compose(ls, bases, n_runs, cfg.seed, mode="embedded", assign="shuffle")
    quality = compare_designs(ls, n_runs, cfg.seed)
    if use_kx:
        quality["knowlytix DesignMatrix sobol+refine"] = _kx_design_quality(ls, n_runs, cfg.seed)
    R["design"] = {"catalog": cat.summary(), "catalog_errors": errs, "suite": suite.summary(),
                   "official_catalogs": cat.raw.get("official_origin"), "level_provenance": level_provenance_table(cat),
                   "factors": suite.factor_names, "levels": dict(ls),
                   "dropped": resolve(cat, ["*"], list(cat.factors)).dropped_factors,
                   "moved_to_base": cat.app["moved_to_base"],
                   "n_runs": n_runs, "quality": quality,
                   "per_base_dispersion": {"dispersed": per_base_dispersion(scenarios, ls),
                                           "shuffle": per_base_dispersion(shuffled, ls)}}
    if cfg.run_power:
        R["design"]["power"] = power_curve(ls, [n_runs // 4, n_runs // 2, n_runs], reps=120)

    # 6 -------------------------------------------------------------- invariance proof
    log("[5/12] proving base/enriched equivalence for every scenario")
    realizer, reader = Realizer(), FrameReader(store)
    validator = InvarianceValidator(store, oracle, reader, realizer)
    proofs = [validator.prove(s) for s in scenarios]
    controls = validator.negative_controls(scenarios, per_type=cfg.negative_controls_per_type)
    R["invariance"] = {"summary": validator.summarize(proofs, suite.factor_names), "negative_controls": controls,
                       "rejected": [p.sid for p in proofs if not p.accepted],
                       "examples": [{"sid": p.sid, "levels": {k: v for k, v in p.levels.items()},
                                     "turns": p.turns} for p in proofs[:12]]}
    items = {b.qid: b for b in bases}
    (out / "ontology.ttl").write_text(ont.to_turtle(validator.evidence_triples(proofs, items)))

    # 5b ------------------------------------------------------------- LLM enrichment arm
    llm_scns, llm_proofs, llm_realizer, llm_validator, llm_controls = [], [], None, None, None
    if cfg.llm_arm:
        log("[5b/12] LLM enrichment arm (Microsoft gateway): generate blind, gate every turn, prove "
            "separately, controls on LLM text")
        from .enrich.llm import LLMSettings, build_llm_arm
        settings = LLMSettings(model=cfg.llm_model or "", temperature=cfg.llm_temperature, mode=cfg.llm_mode,
                               cache_dir=cfg.llm_cache_dir)
        llm_realizer, llm_validator, clients = build_llm_arm(
            store, oracle, reader, validator, settings, dual_read=cfg.llm_dual_read,
            adversarial=cfg.llm_adversarial_controls > 0, max_attempts=cfg.llm_max_attempts)
        why = clients["rephrase"].unavailable_reason()
        if why:
            log(f"      LLM not callable ({why}); only cached generations can be used")
        else:
            log(f"      LLM: Microsoft gateway · model {settings.model}")
        pick = random.Random(cfg.seed + 7)
        chosen = [s for s in scenarios if pick.random() < cfg.llm_share]
        llm_scns = [Scenario(f"{s.sid}~llm", s.base, dict(s.factor_levels), s.mode,
                             {"arm": "llm", "paired_with": s.sid}) for s in chosen]
        if cfg.llm_workers > 1:
            llm_validator.prefetch(llm_scns, workers=cfg.llm_workers, log=log)
        llm_proofs = llm_validator.prove_all(llm_scns, workers=cfg.llm_workers)
        if cfg.llm_workers > 1:
            llm_validator.prefetch_controls(llm_proofs, llm_scns, per_type=cfg.negative_controls_per_type,
                                            workers=cfg.llm_workers)
        llm_controls = llm_validator.negative_controls(llm_proofs, llm_scns, per_type=cfg.negative_controls_per_type,
                                                       adversarial=cfg.llm_adversarial_controls)
        R["invariance_llm"] = {
            "summary": llm_validator.summarize_llm(llm_proofs, suite.factor_names),
            "negative_controls": {k: v for k, v in llm_controls.items() if k != "_audit"},
            "controls_audit_n": len(llm_controls.get("_audit", [])),
            "rejected": [p.sid for p in llm_proofs if not p.accepted],
            "examples": [{"sid": p.sid, "levels": p.levels, "accepted": p.accepted,
                          "turns": [{k: v for k, v in t.items() if k not in ("frame_obj", "read_obj", "attempts")}
                                    for t in p.turns]} for p in llm_proofs[:8]],
            "settings": settings.public(), "unavailable_reason": why,
            "clients": {k: c.report() for k, c in clients.items() if c is not None}}
        a = R["invariance_llm"]["summary"]
        log(f"      LLM arm: {a['accepted']}/{a['scenarios']} scenarios proved; controls rejected "
            f"{llm_controls['_overall']['detection_rate']:.1%} of {llm_controls['_overall']['n']}")

    # 6 -------------------------------------------------------------- runtime gates (Appendix C)
    gcfg = None
    if cfg.run_governance:
        log("[6/12] calibrating the runtime gates (Appendix C) and sweeping the gate probes")
        from .governance.calibrate import (calibrate_plan, calibrate_plausibility, gate_probe_sweep,
                                           multi_threshold_design, policy_fingerprint, store_fingerprint)
        from .governance.monitor import GovernanceConfig
        cal = calibrate_plausibility(store)
        journeys = [[c["tool"] for c in oracle.run(sp).tool_log] for sp in specs]
        pc = calibrate_plan(store, journeys)
        gcfg = GovernanceConfig(theta=cal["theta"], drift_budget=pc["drift_budget"], plan_criterion=pc["criterion"],
                                decision_rule=cfg.decision_rule)
        R["governance"] = {"plausibility_calibration": cal, "plan_calibration": pc,
                           "probe_sweep": gate_probe_sweep(store, cal["theta"]),
                           "multi_threshold_design": multi_threshold_design(store, cal["theta"]),
                           "calibration_record": {"store_fingerprint": store_fingerprint(store),
                                                  "policy_fingerprint": policy_fingerprint(store),
                                                  "reference_scores": [],      # filled from the base-level runs
                                                  "false_reject_rate": cal["regulated"]["false_rejects"] / max(1, cal["n_admit"])},
                           "engine": "forgeloop GMSPlausibilityGate + AuditLogger on the knowlytix store"
                           if use_kx else "built-in gates"}

    # 7 -------------------------------------------------------------- run SUT
    log("[7/12] running the system under test (runtime monitor on)")
    factory = cfg.bot_factory or make_reference_factory(store)
    rk = {"store": store, "realizer": realizer, "reader": reader}
    budget = ConstraintBudget(**(cfg.budgets or {}))
    arms = {"llm": llm_realizer} if llm_realizer is not None else None
    runner = JourneyRunner(bot_factory=factory, governance=gcfg, arm_realizers=arms, budgets=budget, **rk)
    base_scns = [Scenario(f"{b.qid}~base", b, {}, "base") for b in bases]
    base_rows = run(base_scns, runner, oracle.limits, store.workflow, None)
    rows = run(scenarios, runner, oracle.limits, store.workflow, validator)
    llm_rows = run(llm_scns, runner, oracle.limits, store.workflow, llm_validator) if llm_scns else []
    R["results"] = {"bot": cfg.bot_name, "base_level": summary(base_rows), "enriched": summary(rows),
                    "budgets": budget.as_dict()}
    if llm_scns:
        R["results"]["enriched_llm"] = summary(llm_rows)
        R["arms"] = _arm_comparison(rows, llm_rows)
        if llm_realizer is not None:
            R["invariance_llm"]["runtime_realizer"] = dict(llm_realizer.stats)
    R["release_gate"] = {"base_level": release_gate(base_rows), "enriched_template": release_gate(rows)}
    if llm_rows:
        R["release_gate"]["enriched_llm"] = release_gate(llm_rows)
    if gcfg is not None:
        # drift reference: the gate scores of the bot's own base-level traffic (calibration time);
        # the enriched runs play the role of production traffic
        good = lambda r: r["correct"] and r["workflow_adherent"] and r.get("kb_correct", 1)
        R["governance"]["calibration_record"]["reference_scores"] = [x for r in base_rows if good(r)
                                                                     for x in r.get("_plaus_scores", [])]
        R["governance"]["calibration_record"]["reference"] = "gate scores on the base-level runs of the bot under test"

    # 8 -------------------------------------------------------------- attribution
    log("[8/12] attributing failures (" + ("knowlytix DOEAnalyzer + " if use_kx else "") + "deviance tests; per component, weak link)")
    R["diagnosis"] = attribution.diagnose(rows, suite.factor_names)
    if use_kx:
        from .reuse.attribution import kx_attribution
        R["diagnosis"]["knowlytix"] = kx_attribution(rows, suite.factor_names)
    if llm_rows:
        # the LLM arm is attributed on the factor levels that actually reached the bot (realised levels)
        R["diagnosis_llm"] = attribution.diagnose(llm_rows, suite.factor_names)

    # 9 -------------------------------------------------------------- resilience
    probe_scns = [s for s in base_scns if s.base.metadata["relevant"].get("n_txn") not in (NA, 0)
                  and not s.base.metadata.get("kb_qid")]
    if cfg.run_resilience:
        log("[9/12] probing resilience under tool faults")
        R["resilience"] = resilience.probe(runner, oracle, probe_scns)

    # 10 ------------------------------------------------------------- mutation analysis
    if cfg.run_mutation and cfg.bot_factory is None:
        from .sut.reference_bot import MUTATIONS
        log(f"[10/12] mutation analysis ({len(MUTATIONS)} seeded defects; runtime monitor watching)")
        rgen = JourneyBaseGenerator(oracle, kb_pool=[k.frame for k in admitted], seed=cfg.seed + 1)
        rand_specs = []
        while len(rand_specs) < len(specs):
            sp = rgen.finalize(rgen.random_spec())
            if sp is not None:
                rand_specs.append(sp)
        rand = [to_item(sp, oracle, f"RND{i:03d}") for i, sp in enumerate(rand_specs)]
        journey_base_scns = [s for s in base_scns if not s.base.metadata.get("kb_qid")]
        suites = {"covering_base": journey_base_scns,
                  "random_base_same_size": [Scenario(f"{b.qid}~base", b, {}, "base") for b in rand],
                  "kb_bases": [s for s in base_scns if s.base.metadata.get("kb_qid")],
                  "enriched_design": scenarios}
        R["mutation"] = mutation.mutation_analysis(store, rk, suites, oracle.limits, store.workflow, validator,
                                                   fault_scenarios=probe_scns, governance=gcfg)

    # runtime monitoring summary ------------------------------------------------
    if gcfg is not None:
        R["governance"]["runtime"] = _runtime_summary(rows, R["governance"]["calibration_record"], store)

    # 11 ------------------------------------------------------------- training corpora
    log("[11/12] emitting training corpora (proofloop emitters)" if importlib.util.find_spec("proofloop") is not None else "[11/12] training corpora skipped (proofloop not installed)")
    R["sft"] = _training_corpora(store, proofs, reader, realizer, admitted, out)

    # 12 ------------------------------------------------------------- outputs
    R["failure_modes"] = _failure_modes(R)
    log("[12/12] writing outputs")
    R["state_mapping"] = _state_mapping_summary(proofs, kg)
    # turn by turn: what the oracle says the bot should do at each customer turn, next to what
    # the bot did (diagnostic view of the same runs; grading above is outcome + stages + admissibility)
    from .turn_compare import export_turn_by_turn
    diag_runner = runner if (gcfg is not None and getattr(gcfg, "mode", "observe") == "enforce") else \
        JourneyRunner(bot_factory=factory, governance=None, arm_realizers=arms, budgets=budget, **rk)
    llm_ok = [s for s in llm_scns if (s.metadata.get("proof") or {}).get("accepted")]
    R["turn_by_turn"] = export_turn_by_turn(out / "dataset", diag_runner, oracle, base_scns + scenarios + llm_ok,
                                            validator=validator)
    if llm_scns:
        from .enrich.llm.export import write_llm_exports
        R["llm_exports"] = write_llm_exports(out / "dataset", llm_proofs, llm_controls,
                                             R.get("arms", {}).get("pairs"))
    if kg is not None:
        from .export_dataset import write_dataset
        from .knowledge.canonicals import export_canonicals
        canon = export_canonicals(store, kg, R.get("backend"))
        (out / "gms_canonicals.json").write_text(json.dumps(canon, indent=1, ensure_ascii=False, default=str))
        R["canonicals"] = canon["summary"]
        R["dataset"] = write_dataset(out / "dataset", store=store, kg=kg, oracle=oracle, validator=validator,
                                     bases=bases, scenarios=scenarios, proofs=proofs, kb_items=admitted)
    _write_rows(out / "results_enriched.csv", rows)
    _write_rows(out / "results_base_level.csv", base_rows)
    if llm_rows:
        _write_rows(out / "results_enriched_llm.csv", llm_rows)
        with open(out / "scenarios_llm.jsonl", "w") as f:
            for p in llm_proofs:
                turns = [{k: v for k, v in t.items() if k not in ("frame_obj", "read_obj")} for t in p.turns]
                f.write(json.dumps({"sid": p.sid, "base": p.qid, "levels": p.levels, "accepted": p.accepted,
                                    "signature": p.enriched_signature, "extra": p.extra, "turns": turns},
                                   default=str) + "\n")
    with open(out / "scenarios.jsonl", "w") as f:
        for p in proofs:
            turns = [{k: v for k, v in t.items() if k not in ("frame_obj", "read_obj")} for t in p.turns]
            f.write(json.dumps({"sid": p.sid, "base": p.qid, "levels": p.levels, "accepted": p.accepted,
                                "signature": p.enriched_signature, "turns": turns}, default=str) + "\n")
    with open(out / "failures.jsonl", "w") as f:
        for r in rows + llm_rows:
            if not r["correct"] or not r["workflow_adherent"]:
                f.write(json.dumps(r, default=str) + "\n")
    R["elapsed_seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str))
    from .report import render_html
    (out / "report.html").write_text(render_html(R))
    log(f"done in {R['elapsed_seconds']} s -> {out}")
    return R


def _arm_comparison(rows: list[dict], llm_rows: list[dict]) -> dict[str, Any]:
    """Paired template-vs-LLM comparison on the same design rows (same base, same assigned levels)."""
    from math import comb
    t = {r["sid"]: r for r in rows}
    pairs = []
    for r in llm_rows:
        sid_t = r["sid"].removesuffix("~llm")
        if sid_t in t:
            a, b = t[sid_t], r
            pairs.append({"base": a["seed"], "template_sid": sid_t, "llm_sid": b["sid"],
                          "template_correct": a["correct"], "llm_correct": b["correct"],
                          "template_adherent": a["workflow_adherent"], "llm_adherent": b["workflow_adherent"],
                          "unrealised_factors": b.get("unrealised_factors", [])})
    both = sum(p["template_correct"] and p["llm_correct"] for p in pairs)
    t_only = sum(p["template_correct"] and not p["llm_correct"] for p in pairs)
    l_only = sum(p["llm_correct"] and not p["template_correct"] for p in pairs)
    none = len(pairs) - both - t_only - l_only
    n = t_only + l_only
    k = min(t_only, l_only)
    p_mcnemar = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"pairs_n": len(pairs), "both_correct": both, "template_only_correct": t_only,
            "llm_only_correct": l_only, "both_wrong": none,
            "template_accuracy_on_pairs": (both + t_only) / max(1, len(pairs)),
            "llm_accuracy_on_pairs": (both + l_only) / max(1, len(pairs)),
            "mcnemar_exact_p": p_mcnemar, "pairs": pairs}


def _state_mapping_summary(proofs, kg) -> dict[str, Any]:
    """How the state-aware mapper read every customer turn of the proof."""
    by_state: dict[str, dict[str, int]] = {}
    for p in proofs:
        for t in p.turns:
            st = t.get("state") or "-"
            d = by_state.setdefault(st, {"turns": 0, "equivalent": 0, "admissible": 0})
            d["turns"] += 1
            d["equivalent"] += int(t["equivalent"])
            d["admissible"] += int(t.get("read_card", {}).get("admissible", True))
    return {"states": by_state, "policy_graph": kg is not None,
            "accepted_acts": {s: kg.accepted_acts(s) for s in kg.prompt_states} if kg is not None else {}}


def _kx_design_quality(level_specs, n_runs: int, seed: int) -> dict[str, Any]:
    from .design.doe import design_quality
    from .knowledge.kx_store import _quiet
    with _quiet():
        from knowlytix.harness.graphdoe import DesignMatrix
        dm = DesignMatrix([{"name": n, "type": "categorical", "categories": list(l)} for n, l in level_specs],
                          method="sobol+refine", n_runs=n_runs, seed=seed)
        rows = dm.generate().to_dict(orient="records")
        kq = dm.quality()
    q = design_quality(rows, level_specs)
    q["knowlytix_quality"] = {k: v for k, v in kq.items() if isinstance(v, (int, float))}
    return q


def _runtime_summary(rows: list[dict], record: dict, store) -> dict[str, Any]:
    from .governance.calibrate import recalibration_triggers
    good = lambda r: r["correct"] and r["workflow_adherent"] and r.get("kb_correct", 1)   # every graded part right
    ok = [r for r in rows if good(r)]
    bad = [r for r in rows if not good(r)]
    fa = [r for r in ok if r.get("gate_alarms")]
    by_rule: dict[str, int] = {}
    for r in rows:
        for g in r.get("gate_rules", []):
            by_rule[g] = by_rule.get(g, 0) + 1
    ic = {}
    for lv in ("none", "authority_override", "contradictory_hint"):
        sel = [r for r in rows if r.get("f_instruction_conflict") == lv]
        ic[lv] = {"runs": len(sel), "flagged_by_regex_guard": sum(1 for r in sel if r.get("input_flags"))}
    scores = [s for r in ok for s in r.get("_plaus_scores", [])]
    fr_rate = sum(1 for r in ok if any(g.startswith("gms_plausibility") for g in r.get("gate_rules", []))) / max(1, len(ok))
    quorum = {"calls_routed_to_review": sum(r.get("gate_quorum_review", 0) for r in rows),
              "calls_where_strict_and_quorum_disagree": sum(r.get("gate_strict_vs_quorum", 0) for r in rows),
              "runs_with_numeric_review": sum(1 for r in rows if r.get("numeric_review")),
              "runs_with_numeric_fail": sum(1 for r in rows if r.get("numeric_fail")),
              "correct_runs_with_numeric_alarm": sum(1 for r in ok if r.get("numeric_review") or r.get("numeric_fail"))}
    return {"runs": len(rows), "correct_runs": len(ok), "false_alarm_runs": len(fa), "quorum_and_attestor": quorum,
            "false_alarm_rate": len(fa) / max(1, len(ok)),
            "false_alarm_rules": sorted({g for r in fa for g in r.get("gate_rules", [])}),
            "failed_runs": len(bad), "failed_runs_flagged": sum(1 for r in bad if r.get("gate_alarms")),
            "alarms_by_gate_rule": by_rule, "audit_chains_verified": all(r.get("audit_verified", 1) for r in rows),
            "plan_inadmissible_correct_runs": sum(1 - r.get("plan_admissible", 1) for r in ok),
            "injection_guard_by_level": ic,
            "recalibration": recalibration_triggers(record, store, scores, fr_rate)}


def _training_corpora(store, proofs, reader, realizer, admitted, out: Path) -> dict[str, Any]:
    try:
        from proofloop.suite.emit import to_jsonl
        from .reuse.sft import classifier_corpus, draft_corpus
    except ImportError:
        return {"skipped": "proofloop not installed"}
    from .sut.reference_bot import DEFAULT_CONFIG, KBAnswerer
    ont = Ontology.from_store(store)
    cls = classifier_corpus(proofs, reader, ont)
    to_jsonl(cls.pop("records"), out / "sft_classifier.jsonl")
    sents = [s for _, s in store.kb_sentences]
    res = {"classifier": cls}
    for name, muts in (("reference_bot", set()), ("mutant_kb_first_amount", {"kb_first_amount"}),
                       ("mutant_fraud_claim", {"fraud_claim"})):
        ans = KBAnswerer(sents, dict(DEFAULT_CONFIG), muts)
        d = draft_corpus(store, admitted, lambda fr, q, a=ans: a.answer(q), realizer)
        if name == "reference_bot":
            to_jsonl(d["records"], out / "sft_draft.jsonl")
        d.pop("records")
        d["dropped_examples"] = [{"policy_id": x["policy_id"], "assistant": x["assistant"]} for x in d.pop("dropped_records")[:4]]
        res[f"draft_{name}"] = d
    return res


def _failure_modes(R: dict) -> Any:
    try:
        from .reuse.taxonomy import coverage
    except ImportError:
        return None
    flagged = {m: e.get("flagged_at_runtime") for m, e in (R.get("mutation") or {}).get("mutants", {}).items()}
    try:
        return coverage(R.get("mutation"), R.get("resilience"), flagged)
    except ImportError:
        return None


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r if k not in ("transcript", "kb_detail", "_plaus_scores")})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
