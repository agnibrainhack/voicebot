"""Mining policy-question base cases from the store (book Ch. 5, 8.1.1).

Each base category has a generator that reads the store and emits KB intent
frames whose answers the store computes (kb_truth.kb_answer). Distractors and
counterfactuals are derived from the store too: multiple-choice options are
numeric neighbours of the register value, false plausibility claims come from
``store.negatives_for`` and window checks straddle the register's lookback limit.

Every candidate passes the two-way functionality check before admission
(book Ch. 5; beyond-ship-and-pray nb 25, the production answer-key generators):
  symbolic   re-derive the answer from the register / graph and compare;
  geometric  for every fact triple that carries the answer, the trained manifold
             must rank the true tail first among the relation's non-asserted and
             counterfactual tails AND place it inside the relation's calibrated
             cap radius (the operating point read from the store, never hand-set).
Each item carries a ``_recovery`` tag: ``two_way`` (symbolic and geometry agree),
``symbolic:<reason>`` (an aggregation, absence or stance with no fact triple - the
recusal is recorded with its reason) or ``rejected`` (nothing is admitted
silently). ``multi_hop`` is a genuine two-hop: hop 1 aggregates (the fastest
option by delivery_days_max), hop 2 traverses has_fee_usd, and the ground truth
is (head, value, tails). Two categories that are not customer facing
(contradiction, reasoning-path consistency) run as store self-tests, a forged
question shows the geometry rejecting what the symbolic check admits, and the
precondition of Ch. 4.3 - the store answers its own questions perfectly - is
checked by grading the graph's own answers.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..kb_truth import KBTruth, judge_kb_answer, kb_answer
from ..ontology import SemanticFrame as F

K = lambda intent, **kw: F.make("ASK_KB", intent=intent, **kw)


@dataclass
class KBItem:
    qid: str
    category: str
    frame: F
    answer: Any
    answer_type: str
    primitive: str
    admission: str = "pending"          # doubly_grounded | symbolic_only | rejected
    customer_facing: bool = True
    notes: list[str] = field(default_factory=list)
    recovery: str = "pending"           # two_way | symbolic:<reason> | rejected   (metadata['_recovery'])
    recovery_detail: list[dict] = field(default_factory=list)
    ground_truth: Any = None            # (head, value, tails) for multi_hop


def _kx_two_way(store, truth: KBTruth) -> tuple[str, list[dict]]:
    """knowlytix's own two-way functionality check (benchmark.eval generators, geometric=True):
    the question is wrapped as a ``GeneratedQuestion`` whose ``answer_triples`` are our fact
    triples in the GMS vocabulary, and ``QuestionGenerator._validate`` decides - symbolic
    recompute, then rank-first among non-asserted tails and inside the relation's learned
    ``cap_radius``. The same gate the book's production answer-key generators use."""
    from ..knowledge.kx_store import _quiet
    from ..knowledge.kx_view import kx_triple
    with _quiet():
        from knowlytix.benchmark.eval.generators import ExactRecallGenerator, GeneratedQuestion
    gen = ExactRecallGenerator(geometric=True)
    triples = [kx_triple(*t) for t in truth.answer_triples]
    q = GeneratedQuestion(qid=truth.intent, category="gmsjourney_kb", natural_language=truth.intent,
                          gms_answer_fn=lambda _s: truth.recover(), ground_truth=truth.answer, llm_prompt="",
                          answer_type="str", metadata={"answer_triples": triples})
    with _quiet():
        ok = gen._validate(store.kx, q)
    tag = q.metadata.get("_recovery", "rejected")
    detail = [{"triple": t, "kx_triple": k, "distance": store.kx_score(*t), "cap_radius": store.tau(t[1]),
               "note": gen.last_recovery_note} for t, k in zip(truth.answer_triples, triples)]
    if truth.geometric is not None:
        agree = bool(truth.geometric())
        ok &= agree
        detail.append({"claim_verdict_agrees": agree})
    return ("two_way" if ok and tag == "two_way" else "rejected"), detail


def two_way_check(store, truth: KBTruth, seed: int = 7) -> tuple[str, list[dict]]:
    """The admission gate: symbolic recompute AND geometric recovery of the answer triples."""
    if truth.recover() != truth.answer:
        return "rejected", [{"reason": "symbolic recompute disagrees"}]
    if truth.answer_triples and hasattr(store, "kx"):
        return _kx_two_way(store, truth)
    if truth.recover() != truth.answer:
        return "rejected", [{"reason": "symbolic recompute disagrees"}]
    if not truth.answer_triples:
        return f"symbolic:{truth.symbolic_reason or 'no fact triple'}", []
    rng = random.Random(seed)
    detail, ok = [], True
    for h, r, t in truth.answer_triples:
        d, cap = store.score_triple(h, r, t), store.tau(r)
        competitors = sorted((set(store.relation_range(r)) - set(store.tails(h, r))) |
                             set(store.negatives_for(h, r, t, rng)))
        comp_d = {c: store.score_triple(h, r, c) for c in competitors}
        rank_first = all(d < x for x in comp_d.values())
        within = d <= cap
        ok &= rank_first and within
        detail.append({"triple": (h, r, t), "distance": round(d, 4), "cap_radius": round(cap, 4),
                       "rank_first": rank_first, "within_cap": within, "n_competitors": len(competitors),
                       "nearest_competitor": round(min(comp_d.values()), 4) if comp_d else None})
    if truth.geometric is not None:
        agree = bool(truth.geometric())
        ok &= agree
        detail.append({"claim_verdict_agrees": agree})
    return ("two_way" if ok else "rejected"), detail


def _generators(store, seed: int = 42) -> dict[str, list[F]]:
    rng = random.Random(seed)
    opts = ["standard_delivery", "expedited_delivery"]
    fee = {o: store.lookup_enm("delivery_fee_usd", o) for o in opts}
    L = store.lookup_enm("limit", "lookback_days")
    # plausibility: one committed and one counterfactual claim per option (balanced)
    plaus = []
    for o in opts:
        plaus.append(K("plausibility", attr="fee", entity=o, value=str(fee[o])))
        neg = [n for n in store.negatives_for(o, "has_fee_usd", str(fee[o]), rng)
               if n.replace(".", "").isdigit()]
        plaus.append(K("plausibility", attr="fee", entity=o, value=neg[0]))
    gt = fee["expedited_delivery"]
    mcq_opts = sorted({gt - 10, gt - 5, gt, gt + 5} if gt >= 10 else {gt, gt + 5, gt + 10, gt + 15})
    return {
        "exact_recall": [K("value", attr="fee", entity=o) for o in opts] +
                        [K("value", attr="days", entity=o) for o in opts] +
                        [K("limit", key=k) for k in ("max_transactions_per_session", "lookback_days", "max_auth_attempts")],
        "fact_completion": [K("mcq", attr="fee", entity="expedited_delivery", options=tuple(mcq_opts))],
        "plausibility": plaus + [K("fraud")],
        "absence_detection": [K("refund")],
        "ranking": [K("faster")],
        "comparative_discrimination": [K("cheaper")],
        "counting": [K("count_options")],
        "cross_reference": [K("needs_confirmation")],
        "numeric_computation": [K("fee_difference")],
        "cross_table_aggregation": [K("expedited_total")],
        "multi_hop": [K("faster_option_fee")],
        "conditional_rule": [K("window_check", days=max(1, L // 3)), K("window_check", days=L + 15)],
        "boundary": [K("window_check", days=L), K("window_check", days=L + 1)],
        "procedural_ordering": [K("procedure")],
        "governance_policy": [K("skip_auth")],
    }


def kx_substrate(store, full: bool = False) -> dict[str, Any]:
    """Mine the knowlytix store with the production answer-key generators (book nb 25):
    every category under the two-way geometric guard, with the per-question recovery tag
    and the calibrated cut each generator read. Nothing is admitted silently.

    cross_reference validates every member of every relation-pair overlap against every
    decoy tail (one Cayley rotor per score on CPU); on this graph that is hours, so it
    runs only with full=True and is otherwise reported as skipped with the reason."""
    from ..knowledge.kx_store import _quiet
    with _quiet():
        from knowlytix.benchmark.eval.generators import (ContradictionGenerator, CountingGenerator,
                                                         CrossReferenceGenerator, ExactRecallGenerator,
                                                         MultiHopGenerator, ThresholdGenerator)
    out = {}
    for Cls in (ExactRecallGenerator, CountingGenerator, CrossReferenceGenerator, MultiHopGenerator,
                ContradictionGenerator, ThresholdGenerator):
        if Cls is CrossReferenceGenerator and not full:
            out[Cls.category] = {"skipped": "O(questions x decoys) geometric validations; run kx_substrate(full=True)"}
            continue
        g = Cls(geometric=True)
        try:
            with _quiet():
                qs = g.generate(store.kx)
        except Exception as e:                      # noqa: BLE001 - a generator the store cannot feed
            out[Cls.category] = {"error": str(e)[:200]}
            continue
        tags = Counter(q.metadata.get("_recovery") for q in qs)
        ok = 0
        for q in qs:                                # the graph answers its own questions
            with _quiet():
                a = q.gms_answer_fn(store.kx)
            gt = q.ground_truth
            ok += (abs(a - gt) < 1e-9) if isinstance(gt, float) and a is not None else a == gt
        out[Cls.category] = {"admitted": len(qs), "recovery_stats": dict(g.recovery_stats),
                             "tags": {str(k): v for k, v in tags.items()}, "calibrated_cut": g.last_recovery_note,
                             "graph_answers_own_questions": (ok / len(qs)) if qs else None,
                             "examples": [q.natural_language for q in qs[:2]]}
    return out


def render_truth(truth) -> str:
    """The graph's own answer as text (used for the Ch. 4.3 precondition)."""
    a, at = truth.answer, truth.answer_type
    if at == "money":
        return f"It is ${a:g}."
    if at == "days":
        return f"It takes {a[0]} to {a[1]} business days."
    if at == "number":
        return f"The answer is {a}."
    if at == "bool":
        return "Yes, that is correct." if a else "No, that is not the case."
    if at == "entity":
        return f"{a.replace('_', ' ')} is the one."
    if at == "set":
        return f"{a[0].replace('_', ' ')} is the one."
    if at == "bundle":
        return f"It costs ${a['money']:g}, takes {a['days'][1]} business days, and you must confirm the fee."
    if at == "absent":
        return "I don't have information on that; a representative can help."
    if at == "stance":
        return "An unrecognized charge is not by itself proof of fraud."
    if at == "refuse":
        return "I need to verify your identity first; I can't skip that."
    if at == "sequence":
        return "First we verify you, then review your transactions, record each answer, then any card action."
    return str(a)


def mine_kb_items(store, seed: int = 42) -> tuple[list[KBItem], dict[str, Any]]:
    items: list[KBItem] = []
    recovery_stats: dict[str, Counter] = {}
    cap_notes: dict[str, dict[str, float]] = {}
    for cat, frames in _generators(store, seed).items():
        stats = recovery_stats.setdefault(cat, Counter())
        for j, fr in enumerate(frames):
            truth = kb_answer(store, fr)
            it = KBItem(f"kb-{cat}-{j:02d}", cat, fr, truth.answer, truth.answer_type, truth.primitive,
                        ground_truth=truth.ground_truth)
            tag, detail = two_way_check(store, truth)
            it.recovery, it.recovery_detail = tag, detail
            stats["two_way" if tag == "two_way" else "symbolic" if tag.startswith("symbolic") else "rejected"] += 1
            it.admission = {"two_way": "doubly_grounded", "rejected": "rejected"}.get(tag, "symbolic_only")
            if tag == "rejected":
                it.notes.append(str(detail))
            for dd in detail:
                if "triple" in dd:
                    cap_notes.setdefault(cat, {})[dd["triple"][1]] = dd["cap_radius"]
            items.append(it)

    # the geometry earns its keep: a forged question whose answer recomputes correctly
    # but whose declared fact triple is another option's real fee
    true_q = kb_answer(store, K("value", attr="fee", entity="expedited_delivery"))
    other = str(store.lookup_enm("delivery_fee_usd", "standard_delivery"))
    bad_q = KBTruth(true_q.intent, true_q.answer, true_q.answer_type, true_q.primitive, true_q.recover,
                    answer_triples=[("expedited_delivery", "has_fee_usd", other)])
    symbolic_only = lambda q: q.recover() == q.answer
    forged = {"true_fact": {"symbolic": symbolic_only(true_q), "two_way": two_way_check(store, true_q)[0]},
              "corrupted_triple": {"declared": ("expedited_delivery", "has_fee_usd", other),
                                   "symbolic": symbolic_only(bad_q), "two_way": two_way_check(store, bad_q)[0]}}

    # store self-tests for the non customer-facing categories
    from ..knowledge.ingest import Rule
    planted = dict(store.rules)
    r13 = store.rules["R13"]
    planted["R13x"] = Rule("R13x", "must", r13.subject, r13.predicate, r13.obj, "reversed copy", r13.source)
    self_tests = {
        "contradiction": {
            "policy_consistent": {"expected": [], "got": store.find_contradictions()},
            "planted_reversal": {"expected": "detected", "got": store.find_contradictions(planted)},
        },
        "reasoning_path_consistency": {
            "precedes∘precedes = precedes_two": store.is_path_consistent(["precedes", "precedes"], "precedes_two"),
            "has_session∘reviews∘governed_by = subject_to": store.is_path_consistent(
                ["has_session", "reviews", "governed_by"], "subject_to"),
            "precedes∘precedes = has_fee_usd (must be False)": store.is_path_consistent(
                ["precedes", "precedes"], "has_fee_usd"),
        },
    }
    st = self_tests
    st["passed"] = (st["contradiction"]["policy_consistent"]["got"] == [] and
                    len(st["contradiction"]["planted_reversal"]["got"]) > 0 and
                    st["reasoning_path_consistency"]["precedes∘precedes = precedes_two"] and
                    st["reasoning_path_consistency"]["has_session∘reviews∘governed_by = subject_to"] and
                    not st["reasoning_path_consistency"]["precedes∘precedes = has_fee_usd (must be False)"])

    # precondition: the store answers its own generated questions perfectly
    graded = [judge_kb_answer(store, it.frame, render_truth(kb_answer(store, it.frame)))["correct"]
              for it in items if it.admission != "rejected"]
    precondition = {"n": len(graded), "graph_baseline_accuracy": sum(graded) / max(1, len(graded))}
    mh = next((i for i in items if i.category == "multi_hop"), None)
    recovery = {"gate": "knowlytix QuestionGenerator._validate (geometric=True)" if hasattr(store, "kx")
                else "built-in two-way check",
                "stats": {c: dict(v) for c, v in recovery_stats.items()},
                "cap_radius_read": cap_notes,
                "forged": forged,
                "multi_hop_ground_truth": None if mh is None else
                {"head": mh.ground_truth[0], "hop1_value": mh.ground_truth[1], "tails": sorted(mh.ground_truth[2]),
                 "recovery": mh.recovery}}
    if hasattr(store, "kx"):
        recovery["knowlytix_substrate"] = kx_substrate(store)
    return items, {"self_tests": self_tests, "precondition": precondition, "recovery": recovery}
