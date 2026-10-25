"""Policy questions mined from a compact policy graph.

Every numeric attribute of a value-set member (``card_product.PREMIUM_SUPP.annual_fee_usd``)
and every customer-facing limit (``internal`` limits excluded) is a question whose answer is an
exact register read. Each question passes the same two-way admission gate as the v1 policy
questions (symbolic recompute + knowlytix's geometric recovery of its fact triple, bases/kb.py).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..ontology import SemanticFrame as F
from .model import KB_ACT, PolicyGraph, attr_relation

K = lambda intent, **kw: F.make(KB_ACT, intent=intent, **kw)


def answer_type(attr: str) -> str:
    return "money" if attr.endswith("_usd") else "number"


def kb_frames(g: PolicyGraph) -> list[F]:
    """The askable questions: every numeric attribute the lexicon has wording for (all of them when
    it lists none) on every member, and every customer-facing limit."""
    from ..domain import current
    asked = (current().get("lexicon.kb_questions.attributes") or {})
    out = []
    for attr, spec in g.numeric_attributes().items():
        if asked and attr not in asked:
            continue
        for m in spec["members"]:
            out.append(K("value", attr=attr, entity=m))
    for k, lv in g.limits.items():
        if not lv.get("internal"):
            out.append(K("limit", key=k))
    return out


def kb_answer(store, frame: F):
    """KBTruth for a graph-domain policy question (an exact register read)."""
    from ..domain import current
    from ..kb_truth import KBTruth
    g = current().graph
    intent = frame.get("intent")
    if intent == "value":
        attr, ent = frame.get("attr"), frame.get("entity")
        f = lambda: store.lookup_enm(attr, ent)
        v = f()
        return KBTruth(intent, v, answer_type(attr), "lookup_enm", f, answer_triples=[(ent, attr_relation(attr), str(v))])
    if intent == "limit":
        key = frame.get("key")
        f = lambda: store.lookup_enm("limit", key)
        v = f()
        return KBTruth(intent, v, "number", "lookup_enm", f, answer_triples=[(g.root, f"has_{key}", str(v))])
    return KBTruth(intent, "absent", "absent", "none", lambda: "absent", symbolic_reason="the policy states nothing")


def mine_kb_items(store, seed: int = 42, extended: bool = False) -> tuple[list, dict[str, Any]]:
    from ..bases.kb import KBItem, render_truth, two_way_check
    from ..domain import current
    from ..kb_truth import judge_kb_answer
    g = current().graph
    items, stats = [], {}
    for j, fr in enumerate(kb_frames(g)):
        cat = "exact_recall" if fr.get("intent") == "value" else "conditional_rule"
        truth = kb_answer(store, fr)
        it = KBItem(f"kb-{cat}-{j:02d}", cat, fr, truth.answer, truth.answer_type, truth.primitive)
        tag, detail = two_way_check(store, truth)
        it.recovery, it.recovery_detail = tag, detail
        it.admission = {"two_way": "doubly_grounded", "rejected": "rejected"}.get(tag, "symbolic_only")
        stats.setdefault(cat, Counter())["two_way" if tag == "two_way" else
                                         "symbolic" if tag.startswith("symbolic") else "rejected"] += 1
        items.append(it)
    self_tests = store_self_tests(store)
    graded = [judge_kb_answer(store, it.frame, render_truth(kb_answer(store, it.frame)))["correct"]
              for it in items if it.admission != "rejected"]
    return items, {"self_tests": self_tests,
                   "precondition": {"n": len(graded), "graph_baseline_accuracy": sum(graded) / max(1, len(graded))},
                   "recovery": {"gate": "knowlytix QuestionGenerator._validate (geometric=True)",
                                "stats": {c: dict(v) for c, v in stats.items()}},
                   "extended": {"enabled": extended, "added": [], "dropped": []},
                   "hallucination_probes": {"n": 0, "by_kind": {}, "items": []}}


def store_self_tests(store) -> dict[str, Any]:
    """The store must pass these before anything is scored against it (book Ch. 4.3):

    * contradiction: the policy's rules contain no contradiction, and a planted reversal of a
      ``must`` rule (the same fact, stance ``must_not``) is caught, with the tension cut
      calibrated from this pack's own graph;
    * reasoning-path consistency: every composition the pack's facts close is certified by
      holonomy and the same chain against an unrelated relation is not (not applicable when the
      pack has no composed relations).
    ``passed`` is False when a check fails or cannot run; the campaign stops on it."""
    from ..knowledge.ingest import Rule
    out: dict[str, Any] = {}
    planted = dict(store.rules)
    rid, r = next((k, v) for k, v in store.rules.items() if v.stance == "must")
    planted[f"{rid}x"] = Rule(f"{rid}x", "must_not", r.subject, r.predicate, r.obj, "reversed copy", r.source)
    try:
        consistent, reversal = store.find_contradictions(), store.find_contradictions(planted)
        ok = consistent == [] and any(x.get("a") in (rid, f"{rid}x") and x.get("b") in (rid, f"{rid}x") for x in reversal)
        out["contradiction"] = {"policy_consistent": {"expected": [], "got": consistent},
                                "planted_reversal": {"expected": f"{rid} vs {rid}x", "got": reversal}, "passed": ok}
    except Exception as e:                       # noqa: BLE001 - no usable u-space calibration
        out["contradiction"] = {"unavailable": str(e)[:200], "passed": False}
    hol = (getattr(store, "calibration", None) or {}).get("holonomy") or {}
    if hol.get("status") == "not_applicable" or not hol:
        out["reasoning_path_consistency"] = {"status": "not_applicable", "reason": hol.get("reason", "no holonomy cut"),
                                             "passed": True}
    else:
        rows = [{"path": p["path"], "direct": p["direct"], "expected": True,
                 "got": store.is_path_consistent(p["path"], p["direct"])} for p in hol.get("positives", [])]
        rows += [{"path": n["path"], "direct": n["direct"], "expected": False,
                  "got": store.is_path_consistent(n["path"], n["direct"])} for n in hol.get("negatives", [])]
        out["reasoning_path_consistency"] = {"checks": rows, "passed": all(x["got"] == x["expected"] for x in rows)}
    out["passed"] = all(v.get("passed") for k, v in out.items() if isinstance(v, dict))
    return out
