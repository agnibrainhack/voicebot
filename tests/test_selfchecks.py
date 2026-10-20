"""Self-checks, one per layer (the book ends each chapter notebook with assertions).

Run with `pytest -q tests/` or `python tests/test_selfchecks.py`.
"""
from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsjourney.bases.journey import JourneyBaseGenerator, to_item  # noqa: E402
from gmsjourney.bases.kb import mine_kb_items  # noqa: E402
from gmsjourney.catalog import Catalog, resolve_profile, validate  # noqa: E402
from gmsjourney.core import Scenario  # noqa: E402
from gmsjourney.design.doe import compare_designs, compose  # noqa: E402
from gmsjourney.enrich.invariance import InvarianceValidator  # noqa: E402
from gmsjourney.enrich.reader import FrameReader  # noqa: E402
from gmsjourney.enrich.realizers import Realizer  # noqa: E402
from gmsjourney.journey import JourneySpec  # noqa: E402
from gmsjourney.knowledge.geode import build_store  # noqa: E402
from gmsjourney.ontology import Ontology, SemanticFrame as F  # noqa: E402
from gmsjourney.oracle import PolicyOracle  # noqa: E402
from gmsjourney.sut.adapter import JourneyRunner, make_reference_factory  # noqa: E402
from gmsjourney.evaluation.runner import run  # noqa: E402
from gmsjourney.evaluation.stats import deviance_test  # noqa: E402


POLICY_KG = Path(__file__).resolve().parents[1] / "gmsjourney" / "policy" / "charge_review_kg.json"
POLICY_MD = Path(__file__).resolve().parents[1] / "gmsjourney" / "policy" / "charge_review_policy.md"


@lru_cache(maxsize=1)
def env():
    store = build_store(POLICY_KG).store                       # the policy graph JSON is the source
    return store, PolicyOracle(store), Realizer(), FrameReader(store)


# Ch. 2-4: the oracle --------------------------------------------------------
def test_exact_memory_and_plausibility():
    store, *_ = env()
    assert store.lookup_enm("limit", "max_transactions_per_session") == 4
    assert store.lookup_enm("delivery_fee_usd", "expedited_delivery") == 15
    committed = store.score_triple("charge_review_policy", "has_max_transactions_per_session", "4")
    fabricated = store.score_triple("charge_review_policy", "has_max_transactions_per_session", "5")
    assert committed < store.tau("has_max_transactions_per_session") < fabricated


def test_tension_and_holonomy():
    store, *_ = env()
    assert store.tension_energy("RECOGNIZED", "UNRECOGNIZED") > store.calibration["tension"]["tau_contra"]
    assert store.is_path_consistent(["precedes", "precedes"], "precedes_two")
    assert not store.is_path_consistent(["precedes", "precedes"], "has_fee_usd")
    assert store.stance_check("unrecognized_transaction", "is_proof_of", "fraud", +1)["verdict"] == "contradicted"


def test_self_correction_drops_misread():
    res = build_store(POLICY_KG,
                      inject_misread=True)
    assert ("expedited_delivery", "has_fee_usd", "150") not in res.store.triples
    assert any(c.action == "drop_conflict" for c in res.corrections)


# Ch. 5, 8: bases --------------------------------------------------------------
def test_kb_bases_are_grounded():
    store, *_ = env()
    items, info = mine_kb_items(store)
    assert info["precondition"]["graph_baseline_accuracy"] == 1.0
    assert info["self_tests"]["passed"]
    assert all(i.admission != "rejected" for i in items)


def test_covering_array_covers_all_feasible_pairs():
    store, oracle, *_ = env()
    gen = JourneyBaseGenerator(oracle, seed=3)
    rows, rep = gen.covering()
    assert rep.pairs_covered == rep.pairs_required
    extra = gen.closure(rows, store, rep)
    assert not rep.edges_missing and not rep.triggers_missing


def test_oracle_policy_limits():
    _, oracle, *_ = env()
    assert oracle.run(JourneySpec(auth="fail_twice")).transfer_reason == "auth_failed"
    t5 = oracle.run(JourneySpec(n_txn=5))
    assert t5.transfer_reason == "transaction_limit" and len(t5.presented) == 4
    assert "T9" not in oracle.run(JourneySpec(n_txn=2, window="day31")).presented
    assert oracle.run(JourneySpec(n_txn=2, record="fail_first")).recorded() == {}


# Ch. 6-7: catalogs and design -------------------------------------------------
def test_catalog_contract_and_design_quality():
    cat = Catalog.load()
    assert validate(cat) == []
    suite = resolve_profile(cat, "voice_full")
    assert "tool_availability" not in suite.factor_names        # moved to base: it changes the answer
    q = compare_designs(suite.level_specs(), 160)
    assert q["sobol+refine"]["log_phi_p"] < q["random"]["log_phi_p"]
    assert q["sobol+refine"]["worst_marginal_chi2_p"] > 0.5


# Invariance: base and enriched cases are the same test ------------------------
def test_round_trip_and_negative_controls():
    store, oracle, R, rd = env()
    ont = Ontology.from_store(store)
    cat = Catalog.load()
    sel = {n: cat.app_factor(n).levels for n in cat.app["selected"]}
    rng = random.Random(0)
    frames = [F.make("INFORM_DISPOSITION", disposition="UNRECOGNIZED", card_action="REPLACE_CARD", delivery="expedited_delivery"),
              F.make("PROVIDE_CREDENTIAL", credential="4821"), F.make("ASK_KB", intent="window_check", days=31)]
    for f in frames:
        for k in range(150):
            lv = {n: rng.choice(l) for n, l in sel.items()}
            assert ont.equivalent(f, rd.read(R.realize(f, lv, f"t{k}", {"merchant": "Amazon", "amount": "45.67"})))
    # a meaning change must be rejected
    flipped = R.realize(F.make("INFORM_DISPOSITION", disposition="RECOGNIZED"), {}, "x")
    assert not ont.equivalent(frames[0], rd.read(flipped))


def test_invariance_proof_on_scenarios():
    store, oracle, R, rd = env()
    bases = [to_item(JourneySpec(n_txn=2, disposition="mixed", card_intent="replace_expedited"), oracle, "A"),
             to_item(JourneySpec(auth="retry_ok", n_txn=4, disposition="correction"), oracle, "B")]
    suite = resolve_profile(Catalog.load(), "voice_full")
    scns = compose(suite.level_specs(), bases, 24, seed=1)
    v = InvarianceValidator(store, oracle, rd, R)
    proofs = [v.prove(s) for s in scns]
    assert all(p.accepted and p.base_signature == p.enriched_signature for p in proofs)
    nc = v.negative_controls(scns, per_type=10)
    assert nc["_overall"]["detection_rate"] >= 0.95


# Ch. 9-10: SUT scoring and attribution -----------------------------------------
def test_reference_bot_base_cases_and_mutant_kill():
    store, oracle, R, rd = env()
    spec = JourneySpec(n_txn=5)
    scn = [Scenario("L~base", to_item(spec, oracle, "L"), {}, "base")]
    kw = {"store": store, "realizer": R, "reader": rd}
    ok = run(scn, JourneyRunner(bot_factory=make_reference_factory(store), **kw), oracle.limits, store.workflow)
    bad = run(scn, JourneyRunner(bot_factory=make_reference_factory(store, frozenset({"limit_off_by_one"})), **kw),
              oracle.limits, store.workflow)
    assert ok[0]["correct"] == 1 and bad[0]["correct"] == 0


def test_logistic_attribution_finds_planted_driver():
    rng = random.Random(4)
    rows = []
    for _ in range(300):
        a, b = rng.choice(["x", "y", "z"]), rng.choice(["p", "q"])
        fail = rng.random() < (0.6 if a == "z" else 0.1)
        rows.append({"a": a, "b": b, "correct": 0 if fail else 1})
    assert deviance_test(rows, "a", "correct")["p_value"] < 1e-6
    assert deviance_test(rows, "b", "correct")["p_value"] > 0.01


# Catalog contract: application levels must trace to the official catalog ----------------
def test_catalog_contract_catches_a_bad_level_alias():
    import copy
    cat = Catalog.load()
    bad = copy.deepcopy(cat)
    bad.app["selected"]["noise"]["level_aliases"]["asr"] = "Not_A_Catalog_Level"
    assert any("Not_A_Catalog_Level" in e for e in validate(bad))
    bad2 = copy.deepcopy(cat)
    del bad2.app["selected"]["question_format"]["unrealised"]
    assert any("Conditional" in e for e in validate(bad2))


# Two-way recovery: the geometry rejects what the symbolic check admits ------------------
def test_two_way_recovery_and_forged_triple():
    store, *_ = env()
    items, info = mine_kb_items(store)
    rec = info["recovery"]
    assert rec["forged"]["corrupted_triple"]["symbolic"] and rec["forged"]["corrupted_triple"]["two_way"] == "rejected"
    assert rec["forged"]["true_fact"]["two_way"] == "two_way"
    mh = rec["multi_hop_ground_truth"]
    assert mh["head"] == "expedited_delivery" and mh["recovery"] == "two_way"


# Runtime governance: calibrated gates, live detection, tamper-evident audit -------------
def test_runtime_gates_and_audit():
    from gmsjourney.governance.audit import AuditEvent, AuditLogger
    from gmsjourney.governance.calibrate import calibrate_plausibility, gate_probe_sweep
    from gmsjourney.governance.monitor import GovernanceConfig
    store, oracle, R, rd = env()
    cal = calibrate_plausibility(store)
    assert cal["regulated"]["false_admits"] == 0
    sw = gate_probe_sweep(store, cal["theta"])
    assert sw["confusion"]["fp"] == 0
    g = GovernanceConfig(theta=cal["theta"], drift_budget=10.0)
    kw = {"store": store, "realizer": R, "reader": rd}
    scn = [Scenario("L~base", to_item(JourneySpec(n_txn=5), oracle, "L"), {}, "base")]
    ok = run(scn, JourneyRunner(bot_factory=make_reference_factory(store), governance=g, **kw), oracle.limits, store.workflow)
    bad = run(scn, JourneyRunner(bot_factory=make_reference_factory(store, frozenset({"limit_off_by_one"})),
                                 governance=g, **kw), oracle.limits, store.workflow)
    assert ok[0]["audit_verified"] == 1 and "policy:R31" not in ok[0]["gate_rules"]
    assert "policy:R31" in bad[0]["gate_rules"]                 # the 5th record is denied before it executes
    log = AuditLogger()
    for i in range(3):
        log.log(AuditEvent("r", i, "tool", "h", {"tool": "x"}, [], "allowed"))
    assert log.verify()
    ev = log.events[1]
    object.__setattr__(ev.event, "outcome", "denied")          # tamper with history
    assert not log.verify()


def test_gms_view_is_table_shaped_for_geode():
    from gmsjourney.knowledge.kx_view import kx_triple, render_gms_markdown
    store, *_ = env()
    md = render_gms_markdown(store)
    for sec in ("## Limits", "## Delivery options", "## Workflow", "## Aliases", "## Opposites"):
        assert sec in md
    assert "| authenticate_customer | get_review_transactions |" in md
    assert kx_triple("charge_review_policy", "has_max_turns", "20") == ("max_turns", "has_value", "20.0")
    assert kx_triple("expedited_delivery", "requires_fee_confirmation", "yes")[2] == "required"


# policy graph JSON + state-aware ontology mapping ---------------------------------------
def test_policy_graph_is_valid_and_is_the_source():
    from gmsjourney.knowledge.ingest import ingest_markdown
    from gmsjourney.knowledge.policy_kg import PolicyKG, resolve_policy
    kg = PolicyKG.load(POLICY_KG)
    assert kg.validate() == []
    sm = kg.summary()
    assert sm["states"] >= 15 and sm["prompt_states"] == 7 and sm["rules"] == 33
    # every triple quotes its source verbatim; every rule applies in some state
    assert all(t["evidence"] for t in kg.triples)
    # the JSON compiles to exactly the facts of the structured transcription ...
    md, _ = resolve_policy(POLICY_KG)
    a, b = ingest_markdown(POLICY_MD), ingest_markdown(md)
    assert {(t.head, t.relation, t.tail) for t in a.triples} == {(t.head, t.relation, t.tail) for t in b.triples}
    assert a.value_sets == b.value_sets and a.aliases == b.aliases and a.entity_types == b.entity_types
    # ... and the trained store carries everything the graph states, state layer included
    store, *_ = env()
    assert store.kg is not None and store.kg.consistency(store)["all_agree"]
    assert ("PRESENT_TXN", "accepts_act", "INFORM_DISPOSITION") in store.triples


def test_state_aware_ontology_mapping():
    store, _oracle, _R, rd = env()
    ont = Ontology.from_store(store)
    card = lambda text, st: ont.card(rd.read(text, st), st)
    assert card("Yes.", "PRESENT_TXN")["slots"] == {"disposition": "RECOGNIZED"}      # "do you recognise it?"
    assert card("No.", "PRESENT_TXN")["slots"] == {"disposition": "UNRECOGNIZED"}
    assert card("Yes.", "CONFIRM_ACTION")["act"] == "CONFIRM"                         # "shall I go ahead?"
    assert card("No.", "CONFIRM_ACTION")["act"] == "DENY"
    assert card("No.", "ASK_MORE_HELP")["act"] == "END"                               # "anything else?"
    # a 4-digit number is a credential only where the state asks for one
    assert card("My card ends 4821 and that one is not mine.", "PRESENT_TXN")["slots"]["disposition"] == "UNRECOGNIZED"
    assert card("My card ends 4821 and that one is not mine.", "ASK_AUTH")["act"] == "PROVIDE_CREDENTIAL"
    c = card("The last four digits of my card are 4821.", "PRESENT_TXN")
    assert c["act"] == "PROVIDE_CREDENTIAL" and not c["admissible"]                  # out of state -> rejected
    f = F.make("PROVIDE_CREDENTIAL", credential="4821")
    assert ont.equivalent(f, f, "ASK_AUTH") and not ont.equivalent(f, f, "PRESENT_TXN")
    # slot values resolve inside the slot's value set
    g = ont.canonicalize(F.make("REQUEST_CARD_ACTION", card_action="new card", delivery="regular"), "CONFIRM_ACTION")
    assert dict(g.slots) == {"card_action": "REPLACE_CARD", "delivery": "standard_delivery"}


def test_oracle_conforms_to_the_state_graph():
    """Every customer turn the oracle produces is admissible in its state, and every state
    transition it makes is declared in the policy graph."""
    store, oracle, _R, _rd = env()
    kg, ont = store.kg, Ontology.from_store(store)
    declared = {(t["head"], t["tail"]) for t in kg.where("leads_to")}
    gen = JourneyBaseGenerator(oracle, kb_pool=[], seed=42)
    cover, _ = gen.covering()
    for sp in cover:
        tr = oracle.run(sp)
        last, bad = None, []
        for ev in tr.events:
            if ev["role"] == "bot":
                for a in ev["acts"]:
                    if last is not None and (last, a["act"]) not in declared:
                        bad.append((last, a["act"]))
                    last = a["act"]
            elif ev["role"] == "user":
                ok, why = ont.admissible(ev["frame"], last)
                assert ok, (last, str(ev["frame"]), why)
        assert not bad, bad


def test_turn_by_turn_oracle_vs_bot():
    """On clean base cases the reference bot does, turn by turn, what the oracle says it
    should; a seeded defect shows up as a divergence at the turn where it acts."""
    from gmsjourney.turn_compare import turn_by_turn
    store, oracle, R, rd = env()
    gen = JourneyBaseGenerator(oracle, kb_pool=[], seed=42)
    cover, _ = gen.covering()
    items = [to_item(sp, oracle, f"J{i:03d}") for i, sp in enumerate(cover[:12])]
    runner = JourneyRunner(bot_factory=make_reference_factory(store), store=store, realizer=R, reader=rd)
    for it in items:
        df, s = turn_by_turn(runner, oracle, Scenario(f"{it.qid}~base", it, {}, "base"))
        assert s["first_divergence_turn"] is None and s["outcome_correct"], (it.qid, s)
    retry = next((to_item(sp, oracle, "JFAIL") for sp in cover if sp.auth == "fail_twice"), None)
    if retry is not None:
        bad = JourneyRunner(bot_factory=make_reference_factory(store, frozenset({"auth_three_tries"})),
                            store=store, realizer=R, reader=rd)
        df, s = turn_by_turn(bad, oracle, Scenario(f"{retry.qid}~base", retry, {}, "base"))
        assert s["first_divergence_turn"] is not None and df.loc[s["first_divergence_turn"], "state"] == "ASK_AUTH_RETRY"


# knowlytix backend (runs only where knowlytix is installed, licensed and a store is built) --
def test_knowlytix_store_if_available():
    import os
    from gmsjourney import backend
    sdir = os.environ.get("GMSJOURNEY_KX_STORE")
    if not (backend.available() and sdir):
        print("    (skipped: set GMSJOURNEY_KX_STORE to a built store and install knowlytix)")
        return
    from gmsjourney.governance.calibrate import calibrate_plausibility
    from gmsjourney.knowledge.geode import oracle_property
    from gmsjourney.knowledge.kx_store import load_kx_store
    b = load_kx_store(POLICY_KG, sdir)
    st = b.store
    assert st.enm_agreement()["all_agree"]
    assert st.lookup_enm("delivery_fee_usd", "expedited_delivery") == 15
    assert all(o["calibrated_ok"] for o in oracle_property(st))
    items, info = mine_kb_items(st)
    assert info["precondition"]["graph_baseline_accuracy"] == 1.0
    assert info["recovery"]["forged"]["corrupted_triple"]["two_way"] == "rejected"
    assert all(i.recovery != "rejected" for i in items)
    assert calibrate_plausibility(st)["regulated"]["false_admits"] == 0


if __name__ == "__main__":
    import inspect as _inspect
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and _inspect.isfunction(f)]
    for n, f in fns:
        f()
        print(f"ok  {n}")
    print(f"{len(fns)} self-checks passed")
