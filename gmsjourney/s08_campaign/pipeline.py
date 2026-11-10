"""The campaign: all thirteen steps, from the policy graph to the verdict.

``run_campaign(CampaignConfig(...))`` is the one entry point the notebook, the console and the command line use:
  1  build or reload the knowledge store          8  attribute failures to factors
  2  policy-question bases (two-way admitted)       9  resilience under tool faults
  3  journey bases (covering array + closure)      10  mutation analysis (planted defects)
  4  enrichment factors and the design             11  training corpora (optional)
  5  realise every scenario and prove invariance   12  write the outputs and the report
  6  run the bot, check every turn                 13  the one-page verdict
  7  score and summarise

Parts of this file:
  1. ``CampaignConfig``, ``build_gms`` and ``run_campaign`` (with the helpers that pick, sample and report).
  2. ``run_graph_campaign``: the older campaign path for a compact graph, kept for its tests.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


# =====================================================================================================================
# PART 1 - End-to-end campaign: document -> oracle -> bases -> enrichment -> proof -> run -> diagnosis.
#
# Each step maps to the book. The GMS and every geometric read come from knowlytix, with
# proofloop and forgeloop alongside (requirements.txt; backend.require()):
#
#  1  GMS: our compiler (symbolic facts, exact numbers), then knowlytix: GEODE loop,
#     embed loop (v SFT), contradiction SFT (u), cap-trained store, GMSJudge calibration  Ch. 2-4
#  2  policy-question bases; two-way admission via knowlytix QuestionGenerator;
#     knowlytix production generators as the substrate self-test                     Ch. 5, nb 25
#  3  journey bases: covering array + rule/edge/trigger/boundary closure            Ch. 5, 12.3
#  4  catalogs (official, from proofloop) + application layer; design              Ch. 6-7
#  5  invariance proof for every scenario + negative controls                       Ch. 1.2, 8.1.1
#  6  runtime gates calibrated (Appendix C) with forgeloop's GMS gates              App. C, Ch. 12
#  7  run the SUT under the runtime monitor; KB answers judged exactly             Ch. 9, 12.5
#  8  attribution: deviance tests per factor, per component, weak link             Ch. 10
#  9  resilience under tool faults                                                  Ch. 11
# 10  mutation analysis (+ runtime-monitor catch rate)                              (beyond the book)
# 11  training corpora with proofloop's emitters                                    Ch. 8
# 12  failure-mode coverage (proofloop taxonomy), report                            Ch. 11
#
# gmsh parity additions (0.4): GEODE dedup mode and v-SFT A/B (step 1); hallucination probes
# (step 2); an optional LLM enrichment arm (the Microsoft model gateway) proved separately from the
# template arm (step 5b); gmsh-style P=0.5 operating points, held-out accuracy and bootstrap
# intervals next to the regulated ones (step 6); quorum REVIEW and the numeric range attestor in
# the runtime monitor; per-turn latency/length budgets, severity with risk tiers and a tiered
# release gate on every run (steps 7-8).
# =====================================================================================================================

from .. import progress as _progress
from ..s03_knowledge import build as backend
from ..s04_cases.design import JourneyBaseGenerator, primary_category, to_item
from ..s03_knowledge.questions import mine_kb_items
from ..s05_rewording.catalog import Catalog, level_provenance_table, resolve, resolve_profile, validate
from ..s05_rewording.catalog import Scenario
from ..s04_cases.design import compare_designs, compose, per_base_dispersion, power_curve
from ..s05_rewording.proof import InvarianceValidator
from ..s05_rewording.wording import FrameReader
from ..s05_rewording.wording import Realizer
from ..s07_checks import diagnosis
from ..s07_checks.scoring import ConstraintBudget
from ..s07_checks.scoring import run, summary
from ..s07_checks.scoring import release_gate
from ..s01_packs.project import NA
from ..s04_cases.journey import JourneySpec, policy_question_case, relevant_dims, tags_for
from ..s03_knowledge.build import build_store, oracle_property, to_dot
from ..s04_cases.meaning import Ontology
from ..s02_graph.flow import GraphOracle as PolicyOracle
from ..s06_bots.harness import JourneyRunner, make_reference_factory

from ..s01_packs import loading as _domain


def __getattr__(name: str):                       # the active domain pack's policy graph and transcription
    if name == "POLICY":
        return _domain.current().graph_path
    if name == "POLICY_MD":
        return _domain.current().transcription_path
    raise AttributeError(name)


@dataclass
class CampaignConfig:
    """Every setting of a campaign: the pack, output folder, sizes, seed, which checks run, the device, the
    bot under test, the LLM arm and the store build parameters.
    """
    domain: str | None = None                    # domain pack: name or folder (default $GMSJOURNEY_DOMAIN / charge_review)
    policy_path: str = ""                        # default: the domain pack's policy graph
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
    device: str | None = None                    # torch device for the knowlytix store: cpu, cuda, cuda:0 ... (None: auto)
    on_step: Callable | None = None              # on_step(step, title, R) when a step starts (R: the results so far)
    bot_name: str = "ReferenceBot"
    # "scripted": in every bot run the customer says the base journey's lines in order, whatever the bot replies (a
    # deviation never changes the case); "adaptive": the customer answers what the bot actually asked
    customer: str = "scripted"
    # the meaning proof (step 5) takes a reworded case's proof from an earlier run when nothing it depends on has
    # changed (the case, the journey's files, the code; s05_rewording/proof_cache.py); proofs are always filed
    reuse_proofs: bool = False
    proof_cache_dir: str | None = None           # None: <GMSJOURNEY_CACHE>/proofs
    # how many conversations with the bot under test are played at once (each with its own bot and mock bank): an
    # LLM bot spends most of a conversation waiting on its model, so a few at once cut the wall time (the number of
    # model calls is the same; replies are cached, so the results do not depend on it)
    bot_workers: int = 1
    # knowlytix build (defaults: the reference build_calibrated_rag_store; s03_knowledge/build.py)
    kx_epochs: int | None = None                 # production store epochs (None: 150; shortcut for "train.epochs")
    kx_geode_epochs: int = 300                   # GEODE correction-loop trainer epochs
    kx_geode_max_iters: int = 8                  # GEODE correction rounds
    kx_embed_trainer_epochs: int = 150           # trainer inside the GEODE embed loop
    kx_embed_max_iters: int = 2                  # embed-loop rounds
    kx_sft_epochs: int | None = None             # v-encoder SFT epochs (None: the package default, 100)
    kx_contradiction_epochs: int = 400           # u-encoder contradiction SFT epochs
    kx_store_config: dict | None = None          # any DocGMSConfig field, dotted keys ({"cap.margin": 0.3, ...})
    kx_store_dir: str | None = None              # default: ~/.cache/gmsjourney/stores/<hash>
    rebuild_store: bool = False
    # 20+ turn conversations (turn-limit tests): None = turn_pressure is an ordinary covering dimension
    # (~26 such bases); an int = normal-length covering array + at most this many turn-limit bases
    max_turn_limit_cases: int | None = 2
    kb_questions: str = "standard"               # extended: more policy questions over the same facts (s03_knowledge/questions.py)
    # how many base cases the bot under test is run on (each with its enriched variants); None = all.
    # Picked round-robin over the base categories, so a small sample still covers every kind of case.
    max_cases: int | None = None
    progress: bool = True                        # tqdm progress bars for the bot runs (when tqdm is installed)
    allow_failed_self_tests: bool = False        # run on even if the store fails its self-tests
    speech_gate: str = "fail"                    # report | fail | strict: does `correct` require what the bot SAID
                                                 # to be grounded and complete (and, strict, in the right place)
    speech_controls: bool = True                 # check the speech check with word-only negative controls
    max_failed_cases_in_report: int = 400        # failed runs listed one by one in the report (all in failures.jsonl)
    kx_dedup_mode: str = "off"                   # off | audit | accept   (s03_knowledge/build.py)
    kx_v_sft_mode: str = "full"                  # full (reference) | rotation | ab (train both, keep the better)
    # runtime checks
    budgets: dict | None = None                  # ConstraintBudget fields (None = defaults)
    decision_rule: str = "strict"                # strict | quorum   (runtime gates)
    calibration_bootstrap: int = 200             # bootstrap resamples for theta / tau intervals
    # where the enrichment LLM runs (the LLM arm: rewording, dual read, adversary): "gateway" (the Microsoft gateway,
    # deployment llm_model) or "local" (a vLLM model: llm_local_path in this process, or the server at llm_vllm_url).
    # None: the environment's choice (llm.backend_env("enrichment")). The bot under test chooses its own model.
    llm_backend: str | None = None
    llm_local_path: str | None = None
    llm_vllm_url: str | None = None
    # where the testing LLM runs (the two judges, the speech check's policy judge, the case narration), chosen
    # separately: the same choices, judges_model being the gateway deployment. None: llm.backend_env("testing")
    judges_backend: str | None = None
    judges_model: str | None = None
    judges_local_path: str | None = None
    judges_vllm_url: str | None = None
    # two independent LLM judges read every run of the bot beside the rule-based verdict (which still decides);
    # their verdicts and any disagreement with the rules are reported (s07_checks/llm_judges.py)
    llm_judges: bool = False
    llm_judges_mode: str = "live"                # live | replay (cache only)
    llm_judges_max_runs: int | None = None       # judge at most this many runs (failed ones first); None: all
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
    # cost controls: how many design rows the LLM rewrites, and how many LLM calls the controls use
    llm_max_scenarios: int | None = None         # cap after llm_share; spread round-robin over the bases
    llm_controls_per_type: int | None = None     # deterministic controls on LLM text (None: negative_controls_per_type)
    # a full LLMSettings (timeouts, logging, retries, ...); when given it replaces llm_model/llm_temperature/
    # llm_mode/llm_cache_dir
    llm_settings: Any = None
    # quality: a small population, proved correctly
    llm_target_accepted: int | None = None       # keep proving candidates until this many are accepted
    llm_max_candidates: int | None = None        # ... trying at most this many rows (default 3 x target)
    llm_asides: str = "verbatim"                 # verbatim: template asides around the LLM line | paraphrase
    # which design rows the LLM may rewrite, and which turns keep template wording inside them
    llm_exclude: tuple = ("policy_questions", "turn_padding")   # rows left to the template arm (see llm_pool)
    llm_template_acts: tuple = ("ASK_KB",)       # turns of these acts keep the proved template wording
    llm_turn_fallback: str = "template"          # template: a turn the LLM cannot write is the proved template
                                                 # turn (the scenario survives) | none: reject the scenario


def build_gms(policy_path: str | Path | None = None, *, kx_store_dir: str | None = None, rebuild: bool = False,
              seed: int = 42, epochs: int | None = None, geode_epochs: int = 300, max_iters: int = 8,
              embed_trainer_epochs: int = 150, embed_max_iters: int = 2, sft_epochs: int | None = None,
              contradiction_epochs: int = 400, dedup_mode: str = "off", v_sft_mode: str = "full",
              store_config: dict | None = None, log: Callable[[str], None] = print):
    """The knowlytix GMS for a policy: built once, then reloaded from its directory.
    Requires knowlytix and a licence. Returns (store, report, backend info).

    The build parameters default to knowlytix's reference build (build_calibrated_rag_store);
    see s03_knowledge/build.py for the three stated differences. ``store_config`` sets any field of the
    production store's DocGMSConfig (dotted keys, e.g. {"train.epochs": 200, "cap.margin": 0.3});
    ``epochs`` is a shortcut for "train.epochs"."""
    backend.require()
    policy_path = policy_path or _domain.current().graph_path
    info = {"selected": "knowlytix", **backend.describe()}
    from ..s03_knowledge.build import DEFAULT_STORE_CONFIG, build_kx_store, load_kx_store
    overrides = dict(store_config or {})
    if epochs is not None:
        overrides["train.epochs"] = epochs
    params = {"geode_epochs": geode_epochs, "max_iters": max_iters, "embed_trainer_epochs": embed_trainer_epochs,
              "embed_max_iters": embed_max_iters, "sft_epochs": sft_epochs,
              "contradiction_epochs": contradiction_epochs, "dedup_mode": dedup_mode, "v_sft_mode": v_sft_mode}
    # the store is keyed by everything that changes it: what it is trained on (the GMS view of the
    # compiled policy, so an edit to the graph that the store never sees - the speech layer, evidence
    # quotes - reuses it), every build parameter, the production store's config, the package versions
    # and which encoders it is warm-started from
    from ..s03_knowledge.build import build_store
    from ..s03_knowledge.store import render_gms_markdown
    trained_on = render_gms_markdown(build_store(policy_path).store)
    key = hashlib.sha256(json.dumps([trained_on, seed, params,
                                     sorted({**DEFAULT_STORE_CONFIG, **overrides}.items()),
                                     info.get("versions"), backend.encoder_kind()],
                                    default=str).encode()).hexdigest()[:16]
    sdir = Path(kx_store_dir) if kx_store_dir else backend.CACHE / "stores" / key
    if (sdir / "gms_build_report.json").exists() and (sdir / "kx" / "model.pt").exists() and not rebuild:
        log(f"[1/13] loading the knowlytix GMS store ({sdir})")
        _progress.phase("gms", 0, 1, "Load the trained store (built before, nothing to train)",
                        phases=["Load the trained store"])
        kb = load_kx_store(policy_path, sdir)
        _progress.phase("gms", 1, 1, "Store loaded")
    else:
        log(f"[1/13] building the knowlytix GMS store: GEODE loop -> embed loop -> contradiction SFT -> "
            f"production store -> GMSJudge calibration ({sdir})")
        kb = build_kx_store(policy_path, sdir, seed=seed, store_overrides=overrides, log=log, **params)
    return kb.store, {**kb.report, "store_dir": str(sdir)}, info


def speech_section(store, base_rows: list[dict], rows: list[dict], llm_rows: list[dict], gate: str) -> dict[str, Any]:
    """What the bot said, across the campaign: the claims checked and their verdicts, the problems by kind
    with examples, and how many runs the speech check changed (outcome right, words wrong)."""
    from collections import Counter

    def part(rs: list[dict]) -> dict[str, Any]:
        sp = [r for r in rs if "c_speech" in r]
        kinds: Counter = Counter()
        examples: dict[str, dict] = {}
        for r in sp:
            for p in r.get("speech_problems", []):
                k = f"{p['kind']}:{p.get('field') or p.get('act')}"
                kinds[k] += 1
                examples.setdefault(k, {"sid": r["sid"], **p})
        tot = lambda k: sum(r.get(k, 0) for r in sp)  # noqa: E731
        advisory: dict[str, list] = {"unread": [], "volunteered": []}
        for r in sp:                                   # advisory findings are not problems: keep a few to read
            for t in r.get("speech_turns") or []:
                for f in t.get("findings") or []:
                    if f.get("kind") in advisory and len(advisory[f["kind"]]) < 12:
                        advisory[f["kind"]].append({"sid": r["sid"], "where": t.get("where"), "why": f.get("detail")})
        return {"runs": len(sp), "grounded_runs": sum(r["c_speech"] for r in sp),
                "policy_checked": tot("speech_policy_checked"), "policy_failed": tot("speech_policy_failed"),
                "unread": tot("speech_unread"), "volunteered": tot("speech_volunteered"),
                "disclosed": tot("speech_disclosed"), "advisory_examples": advisory,
                "outcome_right_words_wrong": sum(r["outcome_correct"] and not r["c_speech"] for r in sp),
                "claims": tot("speech_claims"), "supported": tot("speech_supported"),
                "contradicted": tot("speech_contradicted"), "unsupported": tot("speech_unsupported"),
                "unverifiable": tot("speech_unverifiable"), "unfaithful": tot("speech_unfaithful"),
                "missing": tot("speech_missing"), "flow": tot("speech_flow"),
                "knowlytix_checked": tot("speech_kx_checked"), "knowlytix_agree": tot("speech_kx_agree"),
                "problems": dict(kinds.most_common()), "examples": examples}
    kg = store.kg
    out = {"gate": gate, "contract": {"acts_with_content": sum(1 for s in kg.states if kg.says(s)),
                                      "fields": len(kg.where("says")), "cues": len(kg.where("spoken_cue")),
                                      "phrasings": len(kg.where("spoken_as"))},
           "base_level": part(base_rows), "enriched": part(rows)}
    if llm_rows:
        out["enriched_llm"] = part(llm_rows)
    return out


def run_campaign(cfg: CampaignConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run a whole campaign (the thirteen steps) and return its result dict; outputs go to ``cfg.out_dir``."""
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    R: dict[str, Any] = {"config": {k: v for k, v in asdict(cfg).items() if k not in ("bot_factory", "on_step")}}
    from ..llm import use_backend                 # where each role's LLM runs, from here on
    if cfg.llm_backend:
        use_backend(cfg.llm_backend, cfg.llm_local_path, cfg.llm_vllm_url, role="enrichment",
                    model=cfg.llm_model or None)
    if cfg.judges_backend:
        use_backend(cfg.judges_backend, cfg.judges_local_path, cfg.judges_vllm_url, role="testing",
                    model=cfg.judges_model or None)
    if cfg.device:                                # the store's encoders and training run here
        from ..s03_knowledge import build as kx_store
        kx_store.DEVICE = cfg.device
    if cfg.on_step is not None:                   # tell the caller as each step starts, with what is done so far
        _log, _hook = log, cfg.on_step

        def log(msg: str, *a) -> None:            # noqa: F811
            _log(msg, *a)
            m = re.match(r"\[(\d+[a-z]?)/1[23]\]\s*(.*)", str(msg))
            if m:
                _hook(m.group(1), m.group(2), R)
    dom = _domain.set_current(cfg.domain) if cfg.domain else _domain.current()
    problems = _domain.lint(dom)
    if problems:
        raise _domain.DomainError(f"domain pack {dom.name} does not match its graph:\n  - " + "\n  - ".join(problems))
    policy_path = cfg.policy_path or str(dom.graph_path)
    R["domain"] = {"name": dom.name, "dir": str(dom.dir), "graph": dom.graph_path.name}
    from ..s09_reports.verdict import provenance
    R["provenance"] = provenance(dom.name)          # code commit and graph hash, for the verdict's reproduction line

    # 1 -------------------------------------------------------------- store
    store, build_report, R["backend"] = build_gms(policy_path,
                                                  kx_store_dir=cfg.kx_store_dir, rebuild=cfg.rebuild_store,
                                                  seed=cfg.seed, epochs=cfg.kx_epochs,
                                                  geode_epochs=cfg.kx_geode_epochs, max_iters=cfg.kx_geode_max_iters,
                                                  embed_trainer_epochs=cfg.kx_embed_trainer_epochs,
                                                  embed_max_iters=cfg.kx_embed_max_iters,
                                                  sft_epochs=cfg.kx_sft_epochs,
                                                  contradiction_epochs=cfg.kx_contradiction_epochs,
                                                  dedup_mode=cfg.kx_dedup_mode, v_sft_mode=cfg.kx_v_sft_mode,
                                                  store_config=cfg.kx_store_config, log=log)
    demo = build_store(policy_path, inject_misread=True)
    if dom.graph is None:
        caught = any(c.fact == ("expedited_delivery", "has_fee_usd", "150") and c.action.startswith("drop")
                     for c in demo.corrections)
    else:
        caught = any(c.action == "drop_conflict" and "misread" in str(c.source) for c in demo.corrections)
    sdir = Path(build_report["store_dir"])
    (out / "gms_build_report.json").write_text(json.dumps(build_report, indent=1, default=str))
    (out / "policy.gms.md").write_text((sdir / "policy.gms.md").read_text())
    R["store"] = {**store.summary(), "store_dir": str(sdir), "build": build_report}
    (out / "store_graph.dot").write_text(to_dot(store, {"enables", "precedes", "delivered_by", "has_fee_usd"}))
    R["store"].update({
        "injected_misread_caught": caught,
        "oracle_property": oracle_property(store),
        "primitive_examples": _primitive_examples(store, dom) if dom.graph is not None else {
            "lookup_enm(limit, max_transactions_per_session)": store.lookup_enm("limit", "max_transactions_per_session"),
            "score_triple(policy, has_max_transactions_per_session, 4)": store.score_triple("charge_review_policy", "has_max_transactions_per_session", "4"),
            "score_triple(policy, has_max_transactions_per_session, 30)": store.score_triple("charge_review_policy", "has_max_transactions_per_session", "30"),
            "tau(has_max_transactions_per_session)": store.tau("has_max_transactions_per_session"),
            "tension(RECOGNIZED, UNRECOGNIZED)": store.tension_energy("RECOGNIZED", "UNRECOGNIZED"),
            "check_holonomy([precedes, precedes], precedes_two)": store.check_holonomy(["precedes", "precedes"], "precedes_two"),
            "check_holonomy([precedes, precedes], has_fee_usd)": store.check_holonomy(["precedes", "precedes"], "has_fee_usd"),
            "stance(unrecognized is proof of fraud)": store.stance_check("unrecognized_transaction", "is_proof_of", "fraud", +1)["verdict"],
        },
        "calibration": {"tension": store.calibration.get("tension"), "holonomy_tau": (store.calibration.get("holonomy") or {}).get("tau_path"),
                        "per_relation": store.calibration.get("score_triple")}})
    from ..s07_checks.gates import tau_bootstrap
    R["store"]["tau_bootstrap"] = tau_bootstrap(store, B=cfg.calibration_bootstrap)
    kg = getattr(store, "kg", None)
    if kg is not None:
        R["policy_kg"] = {**kg.summary(), "consistency": {k: v for k, v in kg.consistency(store).items()
                                                          if k != "enm"}}
        kg.save(out / "policy_kg.json")
    ont = Ontology.from_store(store)
    oracle = PolicyOracle(store)

    # 2 -------------------------------------------------------------- KB bases
    log("[2/13] mining policy-question bases (two-way admission; knowlytix substrate generators)")
    if dom.graph is not None:                    # questions over the graph's figures (s03_knowledge/questions.py)
        from ..s03_knowledge.questions import mine_graph_kb_items as mine_graph_kb
        kb_items, kb_info = mine_graph_kb(store, cfg.seed)
        if not kb_info["self_tests"]["passed"] and not cfg.allow_failed_self_tests:
            (out / "self_tests_failed.json").write_text(json.dumps(kb_info["self_tests"], indent=1, default=str))
            raise RuntimeError("the store failed its self-tests (book Ch. 4.3), so nothing can be scored against it: "
                               + json.dumps(kb_info["self_tests"], default=str)[:1500])
    else:
        kb_items, kb_info = mine_kb_items(store, cfg.seed, extended=cfg.kb_questions == "extended")
    admitted = [k for k in kb_items if k.admission != "rejected"]
    R["kb"] = {"items": [{"qid": k.qid, "category": k.category, "frame": str(k.frame), "answer": k.answer,
                          "primitive": k.primitive, "admission": k.admission, "recovery": k.recovery,
                          "recovery_detail": k.recovery_detail} for k in kb_items], **kb_info}

    # 3 -------------------------------------------------------------- journey bases
    log("[3/13] generating journey bases (covering array + closure)")
    gen = JourneyBaseGenerator(oracle, kb_pool=[k.frame for k in admitted], seed=cfg.seed,
                               max_turn_limit_cases=cfg.max_turn_limit_cases)
    cover, rep = gen.covering()
    kb_traces = [oracle.run(policy_question_case(k.frame)) for k in admitted]
    extra = gen.closure(cover, store, rep, extra_traces=kb_traces)
    specs = list({s.key(): s for s in cover + extra}.values())
    bases = []
    for i, sp in enumerate(specs):
        it = to_item(sp, oracle, f"J{i:03d}")
        it.metadata["tags"] = tags_for(sp)
        bases.append(it)
    for k in admitted:
        it = to_item(policy_question_case(k.frame), oracle, k.qid, category=k.category, kb_item=k)
        it.metadata["tags"] = [k.category]
        bases.append(it)
    R["coverage"] = asdict(rep)
    with open(out / "case_specs.jsonl", "w") as f:   # every case exactly as generated (the console's playground rebuilds them)
        for b in bases:
            sp = b.metadata["spec"]
            f.write(json.dumps({"qid": b.qid, "category": b.base, "levels": sp.dims(), "extra_kb": sp.extra_kb,
                                "kb_frames": [{"act": fr.act, "slots": dict(fr.slots)} for fr in sp.kb_frames],
                                "signature": json.loads(b.answer)}, default=str) + "\n")
    R["bases"] = [{"qid": b.qid, "category": b.base, "tags": b.metadata["tags"], "query": b.query,
                   "relevant": {k: v for k, v in b.metadata["relevant"].items() if v != NA},
                   "expected": json.loads(b.answer), "turns": b.metadata["oracle_turns"],
                   "rules": b.metadata["rules_fired"], "edges": b.metadata["edges"]} for b in bases]
    log(f"      {len(specs)} journey bases + {len(admitted)} policy-question bases")

    # 4-5 ------------------------------------------------------------ suite + design
    log("[4/13] resolving the factor suite (official catalogs + application layer) and composing the design")
    cat = Catalog.load()
    errs = validate(cat)
    suite = resolve_profile(cat, cfg.profile)
    ls = suite.level_specs()
    n_runs = cfg.runs_per_base * len(bases)
    scenarios = compose(ls, bases, n_runs, cfg.seed, mode="embedded", assign="dispersed")
    shuffled = compose(ls, bases, n_runs, cfg.seed, mode="embedded", assign="shuffle")
    quality = compare_designs(ls, n_runs, cfg.seed)
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
    log("[5/13] proving base/enriched equivalence for every scenario")
    realizer, reader = Realizer(), FrameReader(store)
    if dom.graph is not None:
        from ..s05_rewording.proof import GraphInvarianceValidator
        validator = GraphInvarianceValidator(store, oracle, reader, realizer)
    else:
        validator = InvarianceValidator(store, oracle, reader, realizer)
    from ..s05_rewording.proof_cache import ProofCache
    from ..s03_knowledge.build import CACHE as _CACHE
    pcache = ProofCache(cfg.proof_cache_dir or _CACHE, dom, validator, reuse=cfg.reuse_proofs)
    proofs = [pcache.prove(s) for s in _progress.bar(len(scenarios), "proofs", show=False, iterable=scenarios,
                                                     key="proofs", unit="cases", label="Reworded cases proved")]
    if cfg.reuse_proofs:
        log(f"      {pcache.reused} proofs reused from earlier runs (same case, journey files and code), "
            f"{pcache.proved} proved now")
    controls = validator.negative_controls(scenarios, per_type=cfg.negative_controls_per_type)
    R["invariance"] = {"summary": validator.summarize(proofs, suite.factor_names), "negative_controls": controls,
                       "cache": pcache.stats(),
                       "rejected": [p.sid for p in proofs if not p.accepted],
                       "examples": [{"sid": p.sid, "levels": {k: v for k, v in p.levels.items()},
                                     "turns": p.turns} for p in proofs[:12]]}
    items = {b.qid: b for b in bases}
    (out / "ontology.ttl").write_text(ont.to_turtle(validator.evidence_triples(proofs, items)))

    # 5b ------------------------------------------------------------- LLM enrichment arm
    llm_scns, llm_proofs, llm_realizer, llm_validator, llm_controls = [], [], None, None, None
    if cfg.llm_arm:
        log("[5b/13] LLM enrichment arm: generate blind, gate every turn, prove separately, controls on LLM text")
        from ..llm import LLMSettings, LLMUnavailable
        from ..s05_rewording.llm_proof import build_llm_arm
        settings = cfg.llm_settings or LLMSettings(model=cfg.llm_model or "", temperature=cfg.llm_temperature, role="enrichment",
                                                   mode=cfg.llm_mode, cache_dir=cfg.llm_cache_dir)
        llm_realizer, llm_validator, clients = build_llm_arm(
            store, oracle, reader, validator, settings, dual_read=cfg.llm_dual_read,
            adversarial=cfg.llm_adversarial_controls > 0, max_attempts=cfg.llm_max_attempts,
            asides=cfg.llm_asides, turn_fallback=cfg.llm_turn_fallback, template_acts=cfg.llm_template_acts)
        why = clients["rephrase"].unavailable_reason()
        if why:
            log(f"      LLM not callable ({why}); only cached generations can be used")
        else:
            log(f"      LLM: {settings.public()['where']} · model {settings.model}")
        ctl_n = cfg.llm_controls_per_type if cfg.llm_controls_per_type is not None else cfg.negative_controls_per_type
        if cfg.llm_target_accepted:
            pool = pick_llm_rows(llm_pool(scenarios, cfg.llm_exclude), cfg.llm_share, None, cfg.seed + 7, ordered=True)
            log(f"      LLM: proving candidates until {cfg.llm_target_accepted} are accepted "
                f"(pool {len(pool)} rows, at most {cfg.llm_max_candidates or 3 * cfg.llm_target_accepted} tried)")
            try:
                llm_scns, llm_proofs = llm_validator.prove_target(
                    pool, cfg.llm_target_accepted, cfg.llm_max_candidates, workers=cfg.llm_workers, log=log)
            except LLMUnavailable as e:
                raise RuntimeError(f"LLM not usable ({settings.public()['where']}) - {e}\n"
                                   "check it with:  python tools/check_llm.py   (endpoint, token, CA bundle, proxy)")
        else:
            chosen = pick_llm_rows(llm_pool(scenarios, cfg.llm_exclude), cfg.llm_share, cfg.llm_max_scenarios,
                                   cfg.seed + 7)
            log(f"      LLM rewrites {len(chosen)} of {len(scenarios)} design rows "
                f"({len({s.base.qid for s in chosen})} bases)")
            llm_scns = [Scenario(f"{s.sid}~llm", s.base, dict(s.factor_levels), s.mode,
                                 {"arm": "llm", "paired_with": s.sid}) for s in chosen]
            if cfg.llm_workers > 1:
                pf = llm_validator.prefetch(llm_scns, workers=cfg.llm_workers, log=log)
                if pf.get("stopped"):
                    raise RuntimeError(f"LLM gateway not usable - {pf['stopped']}\n"
                                       "check it with:  python tools/check_llm.py   (endpoint, token, CA bundle, proxy)")
            llm_proofs = llm_validator.prove_all(llm_scns, workers=cfg.llm_workers, log=log)
        if cfg.llm_workers > 1:
            llm_validator.prefetch_controls(llm_proofs, llm_scns, per_type=ctl_n, workers=cfg.llm_workers)
        llm_controls = llm_validator.negative_controls(llm_proofs, llm_scns, per_type=ctl_n,
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

    # the answer sheet: every case, what a correct bot must do, and its proved rewordings (before any bot runs)
    try:
        from ..s09_reports.answer_sheet import write_answer_sheet
        R["answer_sheet"] = write_answer_sheet(out, oracle=oracle, store=store, validator=validator, bases=bases,
                                               proofs=proofs, llm_proofs=llm_proofs, journey=dom.name)
        log(f"      answer sheet: {R['answer_sheet']['cases']} cases, {R['answer_sheet']['rewordings']} reworded "
            f"versions -> {out / 'answer_sheet.html'}")
    except Exception as e:                       # noqa: BLE001 - the sheet never costs the run
        R["answer_sheet"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"      the answer sheet could not be written: {R['answer_sheet']['error']}")

    # 6 -------------------------------------------------------------- runtime gates (Appendix C)
    gcfg = None
    if cfg.run_governance:
        log("[6/13] calibrating the runtime gates (Appendix C) and sweeping the gate probes")
        from ..s07_checks.gates import (calibrate_plan, calibrate_plausibility, gate_probe_sweep, multi_threshold_design, policy_fingerprint, store_fingerprint)
        from ..s07_checks.runtime import GovernanceConfig
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
                           "engine": "forgeloop GMSPlausibilityGate + AuditLogger on the knowlytix store"}

    # 7 -------------------------------------------------------------- run SUT
    log("[7/13] running the system under test (runtime monitor on; every utterance checked)")
    from ..s07_checks import scoring
    scoring.SPEECH_GATE = cfg.speech_gate
    ref_factory, mutants, runner_kw = _reference(dom, store)
    factory = cfg.bot_factory or ref_factory
    rk = {"store": store, "realizer": realizer, "reader": reader, "customer": cfg.customer, "paths": validator}
    budget = ConstraintBudget(**(cfg.budgets or {}))
    arms = {"llm": llm_realizer} if llm_realizer is not None else None
    runner = JourneyRunner(bot_factory=factory, governance=gcfg, arm_realizers=arms, budgets=budget, **rk)
    tested = sample_bases(bases, cfg.max_cases)
    ids = {b.qid for b in tested}
    base_scns = [Scenario(f"{b.qid}~base", b, {}, "base") for b in tested]
    run_scns = [s for s in scenarios if s.base.qid in ids]
    run_llm = [s for s in llm_scns if s.base.qid in ids]
    if cfg.max_cases is not None:
        log(f"      testing {len(tested)} of {len(bases)} base cases (max_cases={cfg.max_cases}): "
            f"{len(base_scns)} base runs, {len(run_scns)} enriched" + (f", {len(run_llm)} LLM-arm" if llm_scns else ""))
    pb = cfg.progress
    bw = max(1, int(cfg.bot_workers or 1))          # conversations played at once (an LLM bot waits on its model)
    base_rows = run(base_scns, runner, oracle.limits, store.workflow, None, desc="SUT · base cases", show=pb, workers=bw)
    rows = run(run_scns, runner, oracle.limits, store.workflow, validator, desc="SUT · enriched (template arm)", show=pb,
               workers=bw)
    llm_rows = run(run_llm, runner, oracle.limits, store.workflow, llm_validator, workers=bw,
                   desc="SUT · enriched (LLM arm)", show=pb) if run_llm else []
    bot = cfg.bot_name if (cfg.bot_factory is not None or dom.graph is None) else "GraphBot (the pack's reference bot)"
    R["results"] = {"bot": bot, "base_level": summary(base_rows), "enriched": summary(rows),
                    "budgets": budget.as_dict(),
                    "sample": {"max_cases": cfg.max_cases, "base_cases_tested": len(tested),
                               "base_cases_total": len(bases), "enriched_tested": len(run_scns),
                               "enriched_total": len(scenarios)}}
    if llm_rows:
        R["results"]["enriched_llm"] = summary(llm_rows)
        R["arms"] = _arm_comparison(rows, llm_rows)
        if llm_realizer is not None:
            R["invariance_llm"]["runtime_realizer"] = dict(llm_realizer.stats)
    R["release_gate"] = {"base_level": release_gate(base_rows), "enriched_template": release_gate(rows)}
    if llm_rows:
        R["release_gate"]["enriched_llm"] = release_gate(llm_rows)
    R["speech"] = speech_section(store, base_rows, rows, llm_rows, cfg.speech_gate)
    if cfg.speech_controls and runner.speech is not None:
        log("      checking the speech check: word-only negative controls on the base cases")
        from ..s07_checks.speech_policy import run_controls
        ctl_kw = {k: v for k, v in rk.items()}
        R["speech"]["controls"] = run_controls(lambda f: JourneyRunner(bot_factory=f, **ctl_kw),
                                               ref_factory, base_scns)
        o = R["speech"]["controls"]["overall"]
        log(f"      {o['detected']}/{o['applied']} word edits flagged (lowest rate per control {o['min_rate']})")
    if gcfg is not None:
        # drift reference: the gate scores of the bot's own base-level traffic (calibration time);
        # the enriched runs play the role of production traffic
        good = lambda r: r["correct"] and r["workflow_adherent"] and r.get("kb_correct", 1)
        R["governance"]["calibration_record"]["reference_scores"] = [x for r in base_rows if good(r)
                                                                     for x in r.get("_plaus_scores", [])]
        R["governance"]["calibration_record"]["reference"] = "gate scores on the base-level runs of the bot under test"

    # 8 -------------------------------------------------------------- attribution
    log("[8/13] attributing failures (deviance tests; per component, weak link)")
    R["diagnosis"] = diagnosis.diagnose(rows, suite.factor_names)
    if llm_rows:
        # the LLM arm is attributed on the factor levels that actually reached the bot (realised levels)
        R["diagnosis_llm"] = diagnosis.diagnose(llm_rows, suite.factor_names)

    # 9 -------------------------------------------------------------- resilience
    if dom.graph is None:
        probe_scns = [s for s in base_scns if s.base.metadata["relevant"].get("n_txn") not in (NA, 0)
                      and not s.base.metadata.get("kb_qid")]
    else:                                        # cases that get past the auth tool
        probe_scns = [s for s in base_scns if not s.base.metadata.get("kb_qid")
                      and s.base.metadata["oracle_tool_calls"] > 2][:40]
    if cfg.run_resilience:
        log("[9/13] probing resilience under tool faults")
        R["resilience"] = diagnosis.probe(runner, oracle, probe_scns, show=cfg.progress)

    # 10 ------------------------------------------------------------- mutation analysis
    if cfg.run_mutation and cfg.bot_factory is None:
        log(f"[10/13] mutation analysis ({len(mutants)} seeded defects; runtime monitor watching)")
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
                  "enriched_design": run_scns}
        R["mutation"] = diagnosis.mutation_analysis(store, rk, suites, oracle.limits, store.workflow, validator,
                                                   fault_scenarios=probe_scns[:12], governance=gcfg, mutants=mutants,
                                                   **runner_kw)

    # runtime monitoring summary ------------------------------------------------
    if gcfg is not None:
        R["governance"]["runtime"] = _runtime_summary(rows, R["governance"]["calibration_record"], store)

    # 11 ------------------------------------------------------------- training corpora
    log("[11/13] emitting training corpora (proofloop emitters)" if importlib.util.find_spec("proofloop") is not None else "[11/13] training corpora skipped (proofloop not installed)")
    R["sft"] = _training_corpora(store, proofs, reader, realizer, admitted, out)

    # 12 ------------------------------------------------------------- outputs
    R["failure_modes"] = _failure_modes(R)
    log("[12/13] writing outputs")
    R["state_mapping"] = _state_mapping_summary(proofs, kg)
    # turn by turn: what the oracle says the bot should do at each customer turn, next to what
    # the bot did (diagnostic view of the same runs; grading above is outcome + stages + admissibility)
    from ..s09_reports.cases import export_turn_by_turn
    diag_runner = runner if (gcfg is not None and getattr(gcfg, "mode", "observe") == "enforce") else \
        JourneyRunner(bot_factory=factory, governance=None, arm_realizers=arms, budgets=budget, **rk)
    llm_ok = [s for s in run_llm if (s.metadata.get("proof") or {}).get("accepted")]
    judges = None
    if cfg.llm_judges:
        from ..s07_checks.llm_judges import Judges
        judges = Judges(mode=cfg.llm_judges_mode)
        why = judges.a.unavailable_reason() if cfg.llm_judges_mode == "live" else None
        if why:
            log(f"      LLM judges skipped: {why}")
            R["llm_judges"] = {"skipped": why, "model": judges.model}
            judges = None
        else:
            log(f"      two LLM judges read every run beside the rules ({judges.model})")
    R["turn_by_turn"] = export_turn_by_turn(out / "dataset", diag_runner, oracle, base_scns + run_scns + llm_ok,
                                            validator=validator, show=cfg.progress, judges=judges, kg=kg,
                                            judge_max=cfg.llm_judges_max_runs, judge_workers=max(1, cfg.llm_workers))
    if judges is not None:
        R["llm_judges"] = R["turn_by_turn"].get("llm_judges")
    if llm_scns:
        from ..s05_rewording.llm_arm import write_llm_exports
        R["llm_exports"] = write_llm_exports(out / "dataset", llm_proofs, llm_controls,
                                             R.get("arms", {}).get("pairs"))
    if kg is not None:
        from .outputs import write_dataset
        from ..s03_knowledge.store import export_canonicals
        canon = export_canonicals(store, kg, R.get("backend"))
        (out / "gms_canonicals.json").write_text(json.dumps(canon, indent=1, ensure_ascii=False, default=str))
        R["canonicals"] = canon["summary"]
        R["dataset"] = write_dataset(out / "dataset", store=store, kg=kg, oracle=oracle, validator=validator,
                                     bases=bases, scenarios=scenarios, proofs=proofs, kb_items=admitted,
                                     llm_scenarios=llm_scns, llm_proofs=llm_proofs)
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
    failed = [r for r in base_rows + rows + llm_rows if not r["correct"] or not r["workflow_adherent"]]
    with open(out / "failures.jsonl", "w") as f:
        for r in failed:
            f.write(json.dumps({**r, "arm": r.get("arm") or "base"}, default=str) + "\n")
    R["failed_cases"] = failed_case_records(failed, cfg.max_failed_cases_in_report)
    R["failed_total"] = len(failed)
    with open(out / "dataset" / "case_results.jsonl", "w") as f:     # every graded run, for case_comparison.html
        for r in base_rows + rows + llm_rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_plaus_scores"}, default=str) + "\n")
    # bot validation: what the bot did and what it said, by what each test case contained (bot_validation.html)
    from ..s07_checks.diagnosis import write_bot_validation
    R["bot_validation"] = write_bot_validation(out, R, json.loads(json.dumps(base_rows + rows + llm_rows, default=str)))
    R["elapsed_seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str))
    from ..s09_reports.report import render_html
    (out / "report.html").write_text(render_html(R))
    from ..s09_reports.cases import write_case_comparison
    write_case_comparison(out)

    # 13 ------------------------------------------------------------- verdict
    log("[13/13] writing the verdict: release decision, top risks, how to reproduce; every case, expected beside actual")
    try:
        from ..s09_reports.verdict import write as write_verdict
        R["verdict"] = write_verdict(out)
    except Exception as e:                       # noqa: BLE001 - the verdict never costs the run its report
        R["verdict"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"      the verdict could not be written: {R['verdict']['error']}")
    R["elapsed_seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(R, indent=1, default=str))
    log(f"done in {R['elapsed_seconds']} s -> {out}")
    return R


def _primitive_examples(store, dom) -> dict[str, Any]:
    """The store's primitives on the pack's own figures (a limit, a substitution, two members of a value set)."""
    g = dom.graph
    lk = next(k for k, lv in g.limits.items() if not lv.get("internal"))
    v = store.lookup_enm("limit", lk)
    vs0 = next(iter(g.values))
    m0 = g.members(vs0)
    return {f"lookup_enm(limit, {lk})": v,
            f"score_triple({g.root}, has_{lk}, {v})": store.score_triple(g.root, f"has_{lk}", str(v)),
            f"score_triple({g.root}, has_{lk}, {v * 7 + 3})": store.score_triple(g.root, f"has_{lk}", str(v * 7 + 3)),
            f"tau(has_{lk})": store.tau(f"has_{lk}"),
            f"tension({m0[0]}, {m0[1]})": store.tension_energy(m0[0], m0[1]) if len(m0) > 1 else None}


def _reference(dom, store):
    """The pack's reference bot factory, its seeded defects and the mutation analysis's runner settings: the
    hand-written reference bot of a pack that has one, else the bot that follows the graph (s06_bots/reference.py), with the
    defects mined from the graph."""
    if dom.graph is None:
        from ..s06_bots.reference import MUTATIONS
        return make_reference_factory(store), MUTATIONS, {}
    from ..s06_bots.reference import make_graph_factory, mutants
    ms = mutants(dom.graph)
    return make_graph_factory(store), {m: m.replace("_", " ") for m in ms}, \
        {"runner_cls": JourneyRunner, "factory": make_graph_factory}


def sample_bases(bases: list, n: int | None) -> list:
    """n base cases spread over the base categories in proportion to their size (all when n is None):
    the i-th case of a category of size k is placed at (i + 0.5) / k and the n earliest are taken, so a
    small sample keeps the mix of the full suite and every category of a reasonable size is represented."""
    if n is None or n >= len(bases):
        return list(bases)
    by_cat: dict[str, list] = {}
    for b in bases:
        by_cat.setdefault(b.base, []).append(b)
    keyed = [((i + 0.5) / len(items), c, i, b) for c, items in sorted(by_cat.items()) for i, b in enumerate(items)]
    keyed.sort(key=lambda t: t[:3])
    return [t[3] for t in keyed[: max(0, n)]]


_FAILED_KEYS = ("sid", "seed", "base", "arm", "correct", "workflow_adherent", "status", "transfer_reason",
                "expected_transfer_reason", "severity", "risk_tier", "failure_codes", "violations", "why",
                "kb_detail", "invariance_ok", "budget_violations", "gate_rules", "turns", "tool_sequence",
                "tags", "transcript", "outcome_correct", "c_speech", "speech_problems")


def failed_case_records(failed: list[dict], limit: int | None = 400) -> list[dict]:
    """Compact records of the failed runs for the report: most severe first, transcripts capped at 80 lines."""
    from ..s07_checks.scoring import RANK
    order = sorted(failed, key=lambda r: (-RANK.get(r.get("severity") or "none", 0), r.get("arm") or "", r["sid"]))
    out = []
    for r in order[: limit if limit is not None else len(order)]:
        rec = {k: r.get(k) for k in _FAILED_KEYS if r.get(k) is not None}
        rec["arm"] = r.get("arm") or "base"
        rec["stages_wrong"] = sorted(k[2:] for k, v in r.items() if k.startswith("c_") and v == 0)
        rec["factors"] = {k[2:]: v for k, v in r.items() if k.startswith("f_")}
        if rec.get("transcript"):
            rec["transcript"] = rec["transcript"][:80]
        out.append(rec)
    return out


def llm_pool(scenarios: list, exclude=("policy_questions", "turn_padding")) -> list:
    """The design rows the LLM may rewrite. Excluded rows are still tested with template wording:
    policy_questions - the single-question policy bases (tested on facts, not on wording);
    turn_padding     - turn-limit conversations padded to 20+ turns with policy questions (they test the
                       limit, not the wording, and are the most expensive rows per gateway call)."""
    ex = set(exclude or ())
    out = []
    for s in scenarios:
        md = s.base.metadata
        if "policy_questions" in ex and md.get("kb_qid"):
            continue
        if "turn_padding" in ex and getattr(md.get("spec"), "extra_kb", 0):
            continue
        out.append(s)
    return out


def pick_llm_rows(scenarios: list, share: float, cap: int | None, seed: int, ordered: bool = False) -> list:
    """The design rows the LLM also rewrites: a ``share`` of them, then at most ``cap``, spread
    round-robin over the bases (every base gets one before any gets two), in design order.
    ordered=True returns them in the round-robin order instead (a candidate pool to try in turn)."""
    pick = random.Random(seed)
    chosen = [s for s in scenarios if pick.random() < share]
    if not ordered and (cap is None or len(chosen) <= cap):
        return chosen
    cap = len(chosen) if cap is None else min(cap, len(chosen))
    rng = random.Random(seed + 1)
    by_base: dict[str, list] = {}
    for s in chosen:
        by_base.setdefault(s.base.qid, []).append(s)
    bases = list(by_base)
    rng.shuffle(bases)
    for b in bases:
        rng.shuffle(by_base[b])
    out, i = [], 0
    while len(out) < cap:
        for b in bases:
            if i < len(by_base[b]) and len(out) < cap:
                out.append(by_base[b][i])
        i += 1
    if ordered:
        return out
    order = {id(s): k for k, s in enumerate(scenarios)}
    return sorted(out, key=lambda s: order[id(s)])


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


def _runtime_summary(rows: list[dict], record: dict, store) -> dict[str, Any]:
    from ..s07_checks.gates import recalibration_triggers
    # every graded part right, the words included (whatever --speech-gate says), so an alarm on a run whose words
    # were wrong is never counted as a false alarm
    good = lambda r: r["correct"] and r["workflow_adherent"] and r.get("kb_correct", 1) and r.get("c_speech", 1)  # noqa: E731
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


def _kb_answerers(store) -> list[tuple[str, Callable]]:
    """(name, answer(frame, question) -> text): the reference bot's policy answers and those of two defects."""
    if _domain.current().graph is None:
        from ..s06_bots.reference import DEFAULT_CONFIG, KBAnswerer
        sents = [s for _, s in store.kb_sentences]
        out = []
        for name, muts in (("reference_bot", set()), ("mutant_kb_first_amount", {"kb_first_amount"}),
                           ("mutant_fraud_claim", {"fraud_claim"})):
            ans = KBAnswerer(sents, dict(DEFAULT_CONFIG), muts)
            out.append((name, lambda fr, q, a=ans: a.answer(q)))
        return out
    from ..s06_bots.reference import GraphBot
    bot = GraphBot(None, store)
    return [("reference_bot", lambda fr, q: bot._answer_kb(fr)["text"])]


def _training_corpora(store, proofs, reader, realizer, admitted, out: Path) -> dict[str, Any]:
    try:
        from proofloop.suite.emit import to_jsonl
        from .outputs import classifier_corpus, draft_corpus
    except ImportError:
        return {"skipped": "proofloop not installed"}
    ont = Ontology.from_store(store)
    cls = classifier_corpus(proofs, reader, ont)
    to_jsonl(cls.pop("records"), out / "sft_classifier.jsonl")
    res = {"classifier": cls}
    for name, answer in _kb_answerers(store):
        d = draft_corpus(store, admitted, answer, realizer)
        if name == "reference_bot":
            to_jsonl(d["records"], out / "sft_draft.jsonl")
        d.pop("records")
        d["dropped_examples"] = [{"policy_id": x["policy_id"], "assistant": x["assistant"]} for x in d.pop("dropped_records")[:4]]
        res[f"draft_{name}"] = d
    return res


def _failure_modes(R: dict) -> Any:
    try:
        from .outputs import coverage
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
    keys = sorted({k for r in rows for k in r if k not in ("transcript", "kb_detail", "_plaus_scores", "signature",
                                                          "speech_turns")})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


# =====================================================================================================================
# PART 2 - The campaign for a compact policy graph: the twelve steps of pipeline.run_campaign, with every
#
# domain-specific component mined from the graph.
#
#  1  GMS: the graph compiled (s02_graph/compile.py), then the knowlytix build of pipeline.build_gms
#  2  policy-question bases mined from the graph's figures, two-way admission (s03_knowledge/questions.py)
#  3  journey bases mined from the state machine: paths + failure perturbations (s04_cases/factors.py)
#  4  catalogs + application layer; design                       (shared)
#  5  invariance proof + negative controls mined from the graph  (s05_rewording/proof.py)
#  6  runtime gates calibrated, the graph's rules as call checks (s07_checks/runtime.py)
#  7  run the bot under test (default: the graph's reference bot, s06_bots/reference.py) with the monitor on
#  8  attribution                                                (shared)
#  9  resilience under tool faults on every tool of the graph    (shared probe)
# 10  mutation analysis with the mutants mined from the graph    (s06_bots/reference.py)
# 11  training corpora                                           (not yet for graph packs)
# 12  outputs
# =====================================================================================================================



def run_graph_campaign(cfg, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The older campaign path for a compact graph (journeys mined by the interpreter), kept for its tests."""
    from ..s01_packs import loading as _domain
    from ..s05_rewording.catalog import Catalog, level_provenance_table, resolve, resolve_profile, validate
    from ..s04_cases.design import compare_designs, compose, per_base_dispersion, power_curve
    from ..s05_rewording.wording import FrameReader
    from ..s05_rewording.wording import Realizer
    from ..s07_checks import diagnosis
    from ..s07_checks.scoring import ConstraintBudget
    from ..s07_checks.scoring import run, summary
    from ..s07_checks.scoring import release_gate
    from ..s03_knowledge.build import build_store, oracle_property, to_dot
    from ..s04_cases.meaning import Ontology
    from ..s06_bots.reference import make_graph_factory, mutants
    from ..s05_rewording.proof import GraphInvarianceValidator
    from ..s04_cases.factors import kb_base, mine_paths, perturbations, to_item
    from ..s03_knowledge.questions import mine_graph_kb_items as mine_kb_items
    from ..s02_graph.interpreter import GraphOracle
    from ..s06_bots.harness import GraphRunner

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
        from ..s07_checks.gates import tau_bootstrap
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
    from ..s04_cases import factors
    fcfg = factors.load(dom)
    factor_info = None
    if fcfg:
        # the pack's base factors: a covering array over them, closed on the graph
        log(f"[3/12] base cases from base_factors.yaml: a strength-{fcfg.get('strength', 2)} covering array over "
            f"{len(fcfg['factors'])} factors, closed on the graph")
        errs = factors.check(fcfg)
        if errs:
            raise _domain.DomainError("base_factors.yaml:\n  - " + "\n  - ".join(errs))
        cases, factor_info = factors.generate(oracle, fcfg, [k.frame for k in admitted], seed=cfg.seed)
        closure_paths, cinfo = factors.closure(oracle, cases, fcfg.get("closure_budget", 3))
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
        from ..s04_cases.factors import coverage_closure
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
    proofs = [validator.prove(s) for s in _progress.bar(len(scenarios), "proofs", show=False, iterable=scenarios,
                                                        key="proofs", unit="cases", label="Reworded cases proved")]
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
            from ..s07_checks.gates import calibrate_plan, calibrate_plausibility, policy_fingerprint, store_fingerprint
            from ..s07_checks.runtime import GovernanceConfig
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
    rk = {"store": store, "realizer": realizer, "reader": reader, "customer": cfg.customer, "paths": validator}
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
    R["diagnosis"] = diagnosis.diagnose(rows, suite.factor_names)

    # 9 -------------------------------------------------------------- resilience
    probe_scns = [s for s in base_scns if not s.base.metadata.get("kb_qid") and s.base.metadata["oracle_tool_calls"] > 1]
    fault_tools = [t for t, sp in g.tools.items() if sp["role"] != "handoff"]
    if cfg.run_resilience:
        log("[9/12] probing resilience under tool faults")
        R["resilience"] = diagnosis.probe(runner, oracle, probe_scns[:40], tools=fault_tools, show=cfg.progress)

    # 10 ------------------------------------------------------------- mutation analysis
    if cfg.run_mutation and cfg.bot_factory is None:
        ms = mutants(g)
        log(f"[10/12] mutation analysis ({len(ms)} defects mined from the graph; runtime monitor watching)")
        journey_base = [s for s in base_scns if not s.base.metadata.get("kb_qid")]
        suites = {"mined_base": journey_base, "kb_bases": [s for s in base_scns if s.base.metadata.get("kb_qid")],
                  "enriched_design": run_scns}
        plan = tuple((t, k) for t in fault_tools for k in ("error", "plausible_wrong"))
        R["mutation"] = diagnosis.mutation_analysis(store, rk, suites, oracle.limits, store.workflow, validator,
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
        from ..s09_reports.report import render_graph_html
        (out / "report.html").write_text(render_graph_html(R))
    except Exception:                                        # noqa: BLE001
        R["skipped"]["report_html"] = traceback.format_exc()[-1500:]
        (out / "report.json").write_text(json.dumps(R, indent=1, default=str))
    log(f"done in {R['elapsed_seconds']} s -> {out}")
    return R
