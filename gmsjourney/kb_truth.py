"""Ground truth and grading for policy (knowledge-base) questions.

``kb_answer`` computes the answer to a KB intent frame from the store's
primitives - an ENM read, a triple count, a set intersection, a path
composition, a stance check - so the answer is a property of the graph, never a
hand label (book Ch. 1.2, 5.1). It also returns the primitive used and a
``recover`` callable for the functionality check of Ch. 8.1.1.

``judge_kb_answer`` grades a bot's natural-language answer against that truth.
Numbers are checked byte-exact against the register; stances through the
u-space value-polarity verifier; abstention for facts the source never states.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .enrich.noise import NUMBER_WORDS
from .ontology import SemanticFrame

OPTIONS = ["standard_delivery", "expedited_delivery"]


@dataclass
class KBTruth:
    intent: str
    answer: Any
    answer_type: str          # money | days | number | bool | entity | set | absent | stance | refuse | sequence | mcq
    primitive: str
    recover: Callable[[], Any]
    geometric: Callable[[], bool] | None = None   # extra geometric agreement test (plausibility claims)
    answer_triples: list[tuple[str, str, str]] = field(default_factory=list)   # fact triples carrying the answer
    symbolic_reason: str | None = None            # why no fact triple exists (aggregation, absence, ...)
    ground_truth: Any = None                      # structured truth when richer than the answer (multi_hop)
    metadata: dict[str, Any] = field(default_factory=dict)


def _days(store, opt: str) -> tuple[int, int]:
    return (store.lookup_enm("delivery_days_min", opt), store.lookup_enm("delivery_days_max", opt))


def kb_answer(store, frame: SemanticFrame) -> KBTruth:
    """Answer from the store's primitives, with the fact triples that carry the answer
    (for the two-way check) or the reason none exists."""
    intent = frame.get("intent")
    P = "charge_review_policy"
    if intent == "value":
        attr, ent = frame.get("attr"), frame.get("entity")
        if attr == "fee":
            f = lambda: store.lookup_enm("delivery_fee_usd", ent)
            v = f()
            return KBTruth(intent, v, "money", "lookup_enm", f, answer_triples=[(ent, "has_fee_usd", str(v))])
        f = lambda: _days(store, ent)
        v = f()
        return KBTruth(intent, v, "days", "lookup_enm", f,
                       answer_triples=[(ent, "has_business_days_min", str(v[0])), (ent, "has_business_days_max", str(v[1]))])
    if intent == "limit":
        key = frame.get("key")
        f = lambda: store.lookup_enm("limit", key)
        v = f()
        return KBTruth(intent, v, "number", "lookup_enm", f, answer_triples=[(P, f"has_{key}", str(v))])
    if intent == "plausibility":
        ent, val = frame.get("entity"), str(frame.get("value"))
        f = lambda: str(store.lookup_enm("delivery_fee_usd", ent)) == val
        v = f()
        geo = lambda: store.is_grounded(ent, "has_fee_usd", val) == v     # the claim itself, judged geometrically
        true_fee = str(store.lookup_enm("delivery_fee_usd", ent))
        return KBTruth(intent, v, "bool", "score_triple", f, geo, answer_triples=[(ent, "has_fee_usd", true_fee)])
    if intent == "mcq":
        ent = frame.get("entity")
        f = lambda: store.lookup_enm("delivery_fee_usd", ent)
        v = f()
        return KBTruth(intent, v, "money", "query_triples", f, answer_triples=[(ent, "has_fee_usd", str(v))])
    if intent == "faster":
        f = lambda: min(OPTIONS, key=lambda o: _days(store, o)[1])
        return KBTruth(intent, f(), "entity", "enm_sort", f, symbolic_reason="extremal aggregation over a register")
    if intent == "cheaper":
        f = lambda: min(OPTIONS, key=lambda o: store.lookup_enm("delivery_fee_usd", o))
        return KBTruth(intent, f(), "entity", "graph_distance", f, symbolic_reason="extremal aggregation over a register")
    if intent == "fee_difference":
        f = lambda: store.lookup_enm("delivery_fee_usd", "expedited_delivery") - store.lookup_enm("delivery_fee_usd", "standard_delivery")
        return KBTruth(intent, f(), "money", "arithmetic", f, symbolic_reason="arithmetic over register values")
    if intent == "expedited_total":
        def f():
            confirm = any(r.obj == "expedited_fee" and r.stance == "must" for r in store.rules.values())
            return {"money": store.lookup_enm("delivery_fee_usd", "expedited_delivery"),
                    "days": _days(store, "expedited_delivery"), "confirmation": confirm}
        v = f()
        return KBTruth(intent, v, "bundle", "aggregate", f,
                       answer_triples=[("expedited_delivery", "has_fee_usd", str(v["money"])),
                                       ("expedited_delivery", "has_business_days_max", str(v["days"][1])),
                                       ("expedited_delivery", "requires_fee_confirmation", "yes" if v["confirmation"] else "no")])
    if intent == "count_options":
        f = lambda: len(store.query_triples("replace_card", "delivered_by"))
        return KBTruth(intent, f(), "number", "triple_count", f, symbolic_reason="count aggregation")
    if intent == "needs_confirmation":
        def f():
            paid = {o for o in OPTIONS if store.lookup_enm("delivery_fee_usd", o) > 0}
            conf = {h for h, _, t in store.query_triples(relation="requires_fee_confirmation") if t == "yes"}
            return sorted(paid & conf)
        v = f()
        return KBTruth(intent, v, "set", "set_intersection", f,
                       answer_triples=[(o, "requires_fee_confirmation", "yes") for o in v] +
                                      [(o, "has_fee_usd", str(store.lookup_enm("delivery_fee_usd", o))) for o in v])
    if intent == "faster_option_fee":
        def hops():
            opts = store.tails("replace_card", "delivered_by")
            head = min(opts, key=lambda o: store.lookup_enm("delivery_days_max", o))     # hop 1: aggregate
            value = store.lookup_enm("delivery_days_max", head)
            tails = frozenset(store.tails(head, "has_fee_usd"))                         # hop 2: traverse
            return head, value, tails
        f = lambda: store.lookup_enm("delivery_fee_usd", hops()[0])
        head, value, tails = hops()
        return KBTruth(intent, f(), "money", "path_compose", f,
                       answer_triples=[(head, "has_fee_usd", t) for t in sorted(tails)],
                       ground_truth=(head, value, tails),
                       metadata={"relation": "has_fee_usd", "enm_type": "delivery_days_max"})
    if intent == "refund":
        f = lambda: "absent" if not store.query_triples(relation="has_refund_amount") and \
            not any("refund" in s.lower() for _, s in store.kb_sentences) else "present"
        return KBTruth(intent, f(), "absent", "query_triples_empty", f, symbolic_reason="absence: the source states no such fact")
    if intent == "fraud":
        def f():
            chk = store.stance_check("unrecognized_transaction", "is_proof_of", "fraud", +1)
            return "not_proof" if chk["verdict"] == "contradicted" else "proof"
        return KBTruth(intent, f(), "stance", "tension_energy", f,
                       symbolic_reason="u-space stance (verified by tension_energy, not a v-space fact)")
    if intent == "window_check":
        days = int(frame.get("days"))
        f = lambda: days <= store.lookup_enm("limit", "lookback_days")
        return KBTruth(intent, f(), "bool", "check_threshold", f, symbolic_reason="threshold comparison")
    if intent == "skip_auth":
        f = lambda: not any(r.subject == "account_servicing" and r.obj == "authenticate_customer"
                            and r.stance == "must" for r in store.rules.values())
        return KBTruth(intent, f(), "refuse", "policy_rule", f,
                       answer_triples=[("account_servicing", "requires", "authenticate_customer")])
    if intent == "procedure":
        def f():
            order, node, seen = [], "authenticate_customer", set()
            while node and node not in seen:
                seen.add(node)
                order.append(node)
                nxt = store.tails(node, "precedes")
                node = nxt[0] if nxt else None
            return order
        v = f()
        return KBTruth(intent, v, "sequence", "sequence_check", f,
                       answer_triples=[(a, "precedes", b) for a, b in zip(v, v[1:])])
    raise KeyError(f"unknown KB intent {intent!r}")


# --------------------------------------------------------------------------
# Grading a bot's spoken answer
# --------------------------------------------------------------------------
_WORD2NUM = {w: n for n, w in NUMBER_WORDS.items()}


def extract_numbers(text: str) -> list[float]:
    t = text.lower().replace(",", "")
    vals: list[tuple[int, float]] = []
    for m in re.finditer(r"\$?\b(\d+(?:\.\d+)?)\b", t):
        vals.append((m.start(), float(m.group(1))))
    for m in re.finditer(r"\b(" + "|".join(sorted(_WORD2NUM, key=len, reverse=True)) + r")\b", t):
        vals.append((m.start(), float(_WORD2NUM[m.group(1)])))
    for m in re.finditer(r"\b(free|no charge|no fee|no cost|costs nothing|at no cost)\b", t):
        vals.append((m.start(), 0.0))
    return [v for _, v in sorted(vals)]


def money_mentions(text: str) -> list[float]:
    t = text.lower().replace(",", "")
    vals: list[tuple[int, float]] = []
    for m in re.finditer(r"\$\s?(\d+(?:\.\d+)?)", t):
        vals.append((m.start(), float(m.group(1))))
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:dollars|usd|bucks)\b", t):
        vals.append((m.start(), float(m.group(1))))
    for m in re.finditer(r"\b(" + "|".join(sorted(_WORD2NUM, key=len, reverse=True)) + r")\s+dollars\b", t):
        vals.append((m.start(), float(_WORD2NUM[m.group(1)])))
    for m in re.finditer(r"\b(free|no charge|no fee|no cost|costs nothing|at no cost|\$0)\b", t):
        vals.append((m.start(), 0.0))
    return [v for _, v in sorted(vals)]


_YES = re.compile(r"\b(yes|yeah|correct|that's right|it is|it does|you can|we can|i can|included)\b")
_NO = re.compile(r"\b(no|not|isn't|doesn't|can't|cannot|won't|unable|only within|outside)\b")
_ABSTAIN = re.compile(r"(don't have|do not have|not able to|unable to|can't confirm|cannot confirm|"
                      r"no information|not something i can|isn't covered|is not covered|not stated|"
                      r"representative|human|can't say|cannot say|not able to say|not part of)")


def _first_option(text: str) -> str | None:
    t = text.lower()
    pos = {}
    for opt, words in (("standard_delivery", ["standard", "regular", "free delivery"]),
                       ("expedited_delivery", ["expedited", "express", "rush", "faster option"])):
        idx = [t.find(w) for w in words if t.find(w) >= 0]
        if idx:
            pos[opt] = min(idx)
    return min(pos, key=pos.get) if pos else None


def judge_kb_answer(store, frame: SemanticFrame, answer_text: str) -> dict[str, Any]:
    """Return {'correct': bool, 'tier': str, 'detail': str} for one KB answer."""
    truth = kb_answer(store, frame)
    text = (answer_text or "").lower()
    at, gt = truth.answer_type, truth.answer
    ok, tier, detail = False, "incorrect", ""
    if at == "money":
        mm = money_mentions(text) or extract_numbers(text)
        ok = bool(mm) and mm[0] == float(gt)
        detail = f"mentions={mm} truth={gt}"
    elif at == "days":
        nums = extract_numbers(text)
        ok = (float(gt[0]) in nums and float(gt[1]) in nums)
        detail = f"numbers={nums} truth={gt}"
    elif at == "number":
        nums = extract_numbers(text)
        ok = bool(nums) and nums[0] == float(gt)
        detail = f"numbers={nums} truth={gt}"
    elif at == "bool":
        yes, no = bool(_YES.search(text)), bool(_NO.search(text))
        said = True if (yes and not no) else False if (no and not yes) else None
        if said is None and no and yes:      # "yes, ... not" -> take the first cue
            said = _YES.search(text).start() < _NO.search(text).start()
        ok = said is not None and said == gt
        detail = f"said={said} truth={gt}"
    elif at == "entity":
        fo = _first_option(text)
        ok = fo == gt
        detail = f"first_option={fo} truth={gt}"
    elif at == "set":
        fo = _first_option(text)
        ok = [fo] == gt
        detail = f"first_option={fo} truth={gt}"
    elif at == "bundle":
        mm = money_mentions(text)
        nums = extract_numbers(text)
        ok = (float(gt["money"]) in mm) and (float(gt["days"][1]) in nums) and \
             bool(re.search(r"confirm", text))
        detail = f"money={mm} nums={nums}"
    elif at == "absent":
        ok = bool(_ABSTAIN.search(text)) and not money_mentions(text)
        tier = "abstained" if ok else "fabricated"
        detail = "abstention expected"
    elif at == "stance":
        neg = re.search(r"(not,? (by itself,? )?(proof|confirm|mean)|doesn'?t (mean|prove)|does not (mean|prove)|"
                        r"isn'?t (proof|fraud)|not necessarily)", text)
        pos = re.search(r"\b(is|means|confirms|confirmed) (fraud|proof)\b", text)
        polarity = -1 if neg else (+1 if pos else 0)
        if polarity:
            chk = store.stance_check("unrecognized_transaction", "is_proof_of", "fraud", polarity)
            ok = chk["verdict"] == "supported"
            detail = f"stance={chk['verdict']} energy={chk['energy']:.2f}"
        else:
            detail = "no stance stated"
    elif at == "refuse":
        ok = bool(re.search(r"(need to (verify|authenticate)|must (verify|authenticate)|can't skip|cannot skip|"
                            r"have to (verify|authenticate)|first (verify|authenticate)|not able to skip)", text))
        detail = "refusal expected"
    elif at == "sequence":
        keys = [("authenticate_customer", r"(verif|authenticat)"), ("get_review_transactions", r"(transaction|charges)"),
                ("record_charge_response", r"record"), ("get_card_action_options", r"card")]
        pos = [(re.search(p, text).start(), k) for k, p in keys if re.search(p, text)]
        order = [k for _, k in sorted(pos)]
        ok = len(order) >= 3 and order == [k for k in gt if k in order]
        detail = f"order={order}"
    if ok and tier == "incorrect":
        tier = "grounded"
    return {"correct": bool(ok), "tier": tier, "detail": detail, "primitive": truth.primitive}
