"""Checks for the workflow-agnostic engine (gmsjourney/graph): compact policy graphs, everything
mined from them. Runs on the symbolic store (no knowlytix licence needed), one pack per process.

Run with `python tests/test_graph_engine.py` (both packs) or `pytest -q tests/test_graph_engine.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PACKS = ("charge_review_graph", "supplementary_card")


@lru_cache(maxsize=None)
def env(pack: str):
    from gmsjourney import domain
    dom = domain.set_current(pack)
    from gmsjourney.graph.compile import compile_graph
    from gmsjourney.graph.journeys import coverage_closure, mine_paths, perturbations
    from gmsjourney.graph.kb import kb_frames
    from gmsjourney.graph.machine import GraphOracle
    store = compile_graph(dom.graph_path).store
    oracle = GraphOracle(store)
    paths = mine_paths(oracle)
    paths += coverage_closure(oracle, paths)
    pert = perturbations(oracle, paths, kb_frames(dom.graph))
    return dom, store, oracle, paths, pert


def _pack() -> str:
    return os.environ.get("GMSJOURNEY_DOMAIN") if os.environ.get("GMSJOURNEY_DOMAIN") in PACKS else PACKS[0]


def test_graph_validates_and_pack_lints():
    from gmsjourney import domain
    dom, *_ = env(_pack())
    assert dom.engine == "graph"
    assert dom.graph.validate() == []
    assert domain.lint(dom) == [], domain.lint(dom)
    print("ok test_graph_validates_and_pack_lints", dom.name)


def test_mined_journeys_cover_the_graph_and_the_oracle_obeys_every_rule():
    dom, store, oracle, paths, pert = env(_pack())
    g = dom.graph
    traces = [tr for _, tr in paths] + [oracle.run(sp) for sp, _ in pert]
    states = {s for tr in traces for s in tr.visited}
    assert states == set(g.states), sorted(set(g.states) - states)
    edges = {e for tr in traces for e in tr.edges(store.workflow)[0]}
    assert edges == {w.wid for w in store.workflow}
    reasons = {tr.transfer_reason for tr in traces if tr.transfer_reason}
    assert reasons >= set(g.transfers["reasons"]) - {"turn_limit"}, set(g.transfers["reasons"]) - reasons
    lim = oracle.limits.as_dict()
    bad = [(sp, tr.admissibility(store.workflow, lim)) for (sp, _), tr in zip(paths + pert, traces)
           if tr.admissibility(store.workflow, lim)]
    assert not bad, bad[:3]
    print(f"ok test_mined_journeys_cover_the_graph_and_the_oracle_obeys_every_rule {dom.name}: {len(paths)} paths, "
          f"{len(pert)} perturbations")


def test_reference_bot_passes_every_mined_base_case():
    from gmsjourney.core import Scenario
    from gmsjourney.enrich.reader import FrameReader
    from gmsjourney.enrich.realizers import Realizer
    from gmsjourney.evaluation.runner import run, summary
    from gmsjourney.graph.bot import make_graph_factory
    from gmsjourney.graph.journeys import to_item
    from gmsjourney.graph.runner import GraphRunner
    dom, store, oracle, paths, pert = env(_pack())
    bases = [to_item(sp, oracle, f"G{i:03d}") for i, sp in enumerate([p for p, _ in paths] + [p for p, _ in pert])]
    runner = GraphRunner(store=store, realizer=Realizer(), reader=FrameReader(store), bot_factory=make_graph_factory(store))
    rows = run([Scenario(f"{b.qid}~base", b, {}, "base") for b in bases], runner, oracle.limits, store.workflow, None, show=False)
    s = summary(rows)
    assert s["accuracy"] == 1.0 and s["workflow_adherence"] == 1.0 and s["runtime_invariance"] == 1.0, s
    print(f"ok test_reference_bot_passes_every_mined_base_case {dom.name}: {len(rows)} runs")


def test_every_mined_defect_is_caught_or_equivalent():
    from gmsjourney.core import Scenario
    from gmsjourney.enrich.reader import FrameReader
    from gmsjourney.enrich.realizers import Realizer
    from gmsjourney.evaluation.runner import run
    from gmsjourney.graph.bot import make_graph_factory, mutants
    from gmsjourney.graph.journeys import to_item
    from gmsjourney.graph.runner import GraphRunner
    dom, store, oracle, paths, pert = env(_pack())
    bases = [to_item(sp, oracle, f"G{i:03d}") for i, sp in enumerate([p for p, _ in paths] + [p for p, _ in pert])]
    scns = [Scenario(f"{b.qid}~base", b, {}, "base") for b in bases]
    kw = dict(store=store, realizer=Realizer(), reader=FrameReader(store))
    ref = {s.sid: GraphRunner(bot_factory=make_graph_factory(store), **kw).run(s) for s in scns}
    sig = lambda r: (r.answer, json.dumps(r.components, sort_keys=True), [a for e in r.extras["trace"].events
                                                                         if e["role"] == "bot" for a in map(str, e["acts"])])
    survivors = []
    for m in mutants(dom.graph):
        rows = run(scns, GraphRunner(bot_factory=make_graph_factory(store, frozenset({m})), **kw), oracle.limits,
                   store.workflow, None, show=False)
        caught = any(not (r["correct"] and r["workflow_adherent"] and r.get("c_said", 1) and r.get("c_claims", 1)) for r in rows)
        if not caught:
            runner = GraphRunner(bot_factory=make_graph_factory(store, frozenset({m})), **kw)
            if any(sig(runner.run(s)) != sig(ref[s.sid]) for s in scns):
                survivors.append(m)
    assert not survivors, f"defects neither caught nor equivalent: {survivors}"
    print(f"ok test_every_mined_defect_is_caught_or_equivalent {dom.name}: {len(mutants(dom.graph))} defects")


def test_base_factor_array_covers_every_feasible_pair_and_the_graph():
    from gmsjourney.graph import factors as FA
    from gmsjourney.graph.kb import kb_frames
    dom, store, oracle, *_ = env(_pack())
    cfg = FA.load(dom)
    assert cfg and FA.check(cfg) == [], FA.check(cfg or {})
    cases, led = FA.generate(oracle, cfg, kb_frames(dom.graph))
    extra, cinfo = FA.closure(oracle, cases, cfg.get("closure_budget", 3))
    assert led["covered"] == led["feasible"] and not led["never_relevant_levels"], led
    assert cinfo["covered_items"] == cinfo["reachable_items"], cinfo["unreached"]
    lim = oracle.limits.as_dict()
    assert not [c for c in cases if c.trace.admissibility(store.workflow, lim)]
    print(f"ok test_base_factor_array_covers_every_feasible_pair_and_the_graph {dom.name}: {len(cases)} cases, "
          f"{led['covered']} pairs, closure +{cinfo['added']}")


def test_converter_reproduces_the_committed_graph():
    if _pack() != "charge_review_graph":
        return
    sys.path.insert(0, str(ROOT / "tools"))
    from convert_charge_review_v2 import DST, convert
    assert json.loads(DST.read_text()) == json.loads(json.dumps(convert())), "re-run tools/convert_charge_review_v2.py"
    print("ok test_converter_reproduces_the_committed_graph")


def test_lexicon_reads_back_policy_questions():
    import random
    from gmsjourney.enrich.reader import FrameReader
    from gmsjourney.enrich.realizers import Realizer
    from gmsjourney.graph.kb import kb_frames
    from gmsjourney.ontology import Ontology
    dom, store, *_ = env(_pack())
    fr, R, ont = FrameReader(store), Realizer(), Ontology.from_store(store)
    st = dom.graph.prompt_states[1]
    rng = random.Random(5)
    levels = dict(paraphrase_depth=["none", "light", "deep"], question_format=["interrogative", "declarative", "imperative"],
                  entity_aliasing=["canonical", "synonym", "colloquial"], style=["formal", "casual", "terse"],
                  noise=["none", "typo", "asr"], disfluency=["none", "fillers", "restart"], clarity=["clear", "indirect"])
    bad = []
    for f in kb_frames(dom.graph):
        for i in range(60):
            lv = {k: rng.choice(v) for k, v in levels.items()}
            if not any(ont.equivalent(f, fr.read(R.realize(f, lv, f"t{i}#{a}"), st), st) for a in range(5)):
                bad.append((str(f), lv))
    assert not bad, bad[:3]
    print(f"ok test_lexicon_reads_back_policy_questions {dom.name}")


def test_base_journey_review_passes_and_catches_a_wrong_expectation():
    from gmsjourney.graph import factors as FA
    from gmsjourney.graph.kb import kb_frames
    dom, store, oracle, *_ = env(_pack())
    pool = kb_frames(dom.graph)
    r = FA.review(oracle, FA.load(dom), pool)
    assert r["ok"], [c for c in r["checks"] if not c["ok"]]
    lv = "authenticate_customer.unavailable"
    cfg, errs = FA.apply_edits(dom, [{"expect": lv, "outcome": {"transfer": None, "writes": [dom.graph.transfers["tool"]]}}],
                               dry_run=True)
    assert not errs, errs
    bad = FA.review(oracle, cfg, pool)
    assert not bad["ok"] and any(lv.replace(".", "=") in d for d in bad["checks"][-1]["detail"]), bad["checks"][-1]
    _, errs = FA.apply_edits(dom, [{"drop_level": "authenticate_customer.nope"}, {"expect": lv, "outcome": {"bogus": 1}}],
                             dry_run=True)
    assert len(errs) == 2, errs
    print(f"ok test_base_journey_review_passes_and_catches_a_wrong_expectation {dom.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 or os.environ.get("GMSJOURNEY_DOMAIN") in PACKS:
        if len(sys.argv) > 1:
            os.environ["GMSJOURNEY_DOMAIN"] = sys.argv[1]
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn()
    else:                                   # one process per pack (the active domain is process-wide)
        for p in PACKS:
            r = subprocess.run([sys.executable, __file__, p], env={**os.environ, "GMSJOURNEY_DOMAIN": p})
            if r.returncode:
                sys.exit(r.returncode)
