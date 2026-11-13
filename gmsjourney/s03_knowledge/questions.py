"""Policy questions: their true answers, how an answer is graded, and the questions a campaign asks.

A customer may ask "how much is expedited delivery?" in the middle of a journey. The true answer is computed
from the store's exact figures (never written by hand), and the bot's answer is graded by reading every figure
in it and tying each to its subject: the right figure for the asked option passes, a wrong one fails, a figure
about another option is "volunteered".

Parts of this file:
  1. ``kb_answer`` (the truth for a question frame) and ``judge_kb_answer`` (grade a bot's answer).
  2. Mining policy-question base cases from the store for the 18-category taxonomy (value, limit, plausibility,
     multiple choice, cheaper / faster, window checks, hallucination probes ...), each admitted only when the
     symbolic answer and knowlytix's geometric recovery agree (``two_way_check``); the store's self-tests.
  3. The same for a compact graph: questions over its figures and limits (``kb_frames``, ``graph_kb_answer``).
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


# =====================================================================================================================
# PART 1 - Ground truth and grading for policy (knowledge-base) questions.
#
# ``kb_answer`` computes the answer to a KB intent frame from the store's
# primitives - an ENM read, a triple count, a set intersection, a path
# composition, a stance check - so the answer is a property of the graph, never a
# hand label (book Ch. 1.2, 5.1). It also returns the primitive used and a
# ``recover`` callable for the functionality check of Ch. 8.1.1.
#
# ``judge_kb_answer`` grades a bot's natural-language answer against that truth.
# Numbers are checked byte-exact against the register; stances through the
# u-space value-polarity verifier; abstention for facts the source never states.
# =====================================================================================================================

from ..s05_rewording.wording import NUMBER_WORDS
from ..s04_cases.meaning import SemanticFrame

OPTIONS = ["standard_delivery", "expedited_delivery"]


@dataclass
class KBTruth:
    """The truth of one policy question: the answer, its type, the store primitive that computes it, a
    re-derivation, an optional geometric agreement test, and the fact triples that carry it.
    """
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
    from ..s01_packs.loading import current
    if current().graph is not None:                 # a compact policy graph: questions mined from its figures
        return graph_kb_answer(store, frame)
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
    if intent == "unknown_option":
        name = str(frame.get("name"))

        def f():
            options = set(store.value_sets.get("delivery", [])) | \
                {e for e, ty in store.entity_types.items() if ty in ("delivery", "delivery_option")}
            spoken = name.replace("_", " ")
            alias = any(spoken == a.lower() for o in options for a in store.aliases.get(o, []))
            stated = name in options or alias or any(name in (h, t) for h, _r, t in store.triples)
            return "present" if stated else "absent"
        return KBTruth(intent, f(), "absent", "query_triples_empty", f,
                       symbolic_reason="absence: the policy names no such delivery option (near-miss name)",
                       metadata={"probe": "near_miss", "name": name})
    if intent == "unstated_fact":
        subj, attr = str(frame.get("subject")), frame.get("attr")
        rel = {"fee": "has_fee_usd", "days": "has_business_days_max"}.get(attr, "has_fee_usd")
        f = lambda: "present" if (store.query_triples(subj, rel) or store.query_triples(subj.lower(), rel)) else "absent"
        return KBTruth(intent, f(), "absent", "query_triples_empty", f,
                       symbolic_reason=f"absence: the policy states no {attr} for {subj} (low-coverage entity)",
                       metadata={"probe": "low_coverage", "subject": subj, "relation": rel})
    raise KeyError(f"unknown KB intent {intent!r}")


# --------------------------------------------------------------------------
# Grading a bot's spoken answer
# --------------------------------------------------------------------------
_WORD2NUM = {w: n for n, w in NUMBER_WORDS.items()}


def extract_numbers(text: str) -> list[float]:
    """Every number in a text, in order: digits, number words, and "free" / "no charge" as 0."""
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
    """Every money amount in a text, in order ($N, N dollars, number words + dollars, free as 0)."""
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
    from ..s01_packs.loading import current
    for opt, words in (current().get("rules.bot_option_cues") or {}).items():
        idx = [t.find(w) for w in words if t.find(w) >= 0]
        if idx:
            pos[opt] = min(idx)
    return min(pos, key=pos.get) if pos else None


_READERS: dict[tuple, Any] = {}


def anchored_figures(store, frame: SemanticFrame, answer_text: str) -> list[Any] | None:
    """The figures the answer states about what the question asked, read by the speech reader (which ties each
    figure to the thing named nearest it): "Standard delivery is free, and expedited is $15" answers a question
    about standard delivery with [0], whatever order the options are named in. None when the reader is not
    available for this store or the words tie no figure to the asked thing (the grader then reads the first
    figure, as before)."""
    from ..s01_packs.loading import current
    from ..s07_checks.speech_reading import ANY
    kg = getattr(store, "kg", None)
    subject = frame.get("entity") or frame.get("key")
    if kg is None or not subject:
        return None
    try:
        key = (id(store), id(kg), current().name)
        if key not in _READERS or _READERS[key][0] is not store:     # the store is held, so its id is never reused
            from ..s07_checks.speech_reading import SpeechContract
            from ..s07_checks.speech_checks import UtteranceVerifier
            _READERS[key] = (store, UtteranceVerifier(store, SpeechContract.from_graph(kg), corroborate=False).reader)
        from ..s07_checks.speech_reading import SessionFacts
        read = _READERS[key][1].read(answer_text or "", SessionFacts.from_run([], [], None, {}), question=frame)
    except Exception:                                        # noqa: BLE001 - the plain reading still grades it
        return None
    attr = str(frame.get("attr") or "")
    def fits(field: str) -> bool:
        if frame.get("intent") == "limit":
            return field == "limit"
        return field == attr or (attr == "fee" and field == "fee") or ("days" in attr and "days" in field)
    vals = [c.value for c in read.claims if c.about == subject and c.about != ANY and fits(c.field)]
    return vals or None


def judge_kb_answer(store, frame: SemanticFrame, answer_text: str) -> dict[str, Any]:
    """Return {'correct': bool, 'tier': str, 'detail': str} for one KB answer.

    A figure answer is graded on the figure the words tie to what was asked (``anchored_figures``): every figure
    said about it must be the truth, and figures about other things are not this answer's (the speech check
    grades those). Only when the words tie no figure to the asked thing is the first figure read."""
    truth = kb_answer(store, frame)
    text = (answer_text or "").lower()
    at, gt = truth.answer_type, truth.answer
    ok, tier, detail = False, "incorrect", ""
    anchored = anchored_figures(store, frame, answer_text) if at in ("money", "days", "number") else None
    if anchored is not None:
        attr = str(frame.get("attr") or "")

        def same(v: Any) -> bool:
            if isinstance(gt, (tuple, list)):
                lo, hi = (v if isinstance(v, (tuple, list)) else (v, v))
                return float(gt[0]) <= float(lo) and float(hi) <= float(gt[1])
            if isinstance(v, (tuple, list)):        # "at most 7 business days" reads as (7, 7); "3 to 7" as (3, 7)
                lo, hi = float(v[0]), float(v[1])
                pick = hi if attr.endswith("_max") else lo if attr.endswith("_min") else (hi if lo == hi else None)
                return pick is not None and abs(pick - float(gt)) < 1e-9
            return abs(float(v) - float(gt)) < 1e-9
        ok = all(same(v) for v in anchored)
        detail = f"about {frame.get('entity') or frame.get('key')}: {anchored} truth={gt}"
    elif at == "money":
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


# =====================================================================================================================
# PART 2 - Mining policy-question base cases from the store (book Ch. 5, 8.1.1).
#
# Each base category has a generator that reads the store and emits KB intent
# frames whose answers the store computes (kb_truth.kb_answer). Distractors and
# counterfactuals are derived from the store too: multiple-choice options are
# numeric neighbours of the register value, false plausibility claims come from
# ``store.negatives_for`` and window checks straddle the register's lookback limit.
#
# Every candidate passes the two-way functionality check before admission
# (book Ch. 5; beyond-ship-and-pray nb 25, the production answer-key generators):
#   symbolic   re-derive the answer from the register / graph and compare;
#   geometric  for every fact triple that carries the answer, the trained manifold
#              must rank the true tail first among the relation's non-asserted and
#              counterfactual tails AND place it inside the relation's calibrated
#              cap radius (the operating point read from the store, never hand-set).
# Each item carries a ``_recovery`` tag: ``two_way`` (symbolic and geometry agree),
# ``symbolic:<reason>`` (an aggregation, absence or stance with no fact triple - the
# recusal is recorded with its reason) or ``rejected`` (nothing is admitted
# silently). ``multi_hop`` is a genuine two-hop: hop 1 aggregates (the fastest
# option by delivery_days_max), hop 2 traverses has_fee_usd, and the ground truth
# is (head, value, tails). Two categories that are not customer facing
# (contradiction, reasoning-path consistency) run as store self-tests, a forged
# question shows the geometry rejecting what the symbolic check admits, and the
# precondition of Ch. 4.3 - the store answers its own questions perfectly - is
# checked by grading the graph's own answers.
#
# Hallucination probes (gmsh testing/hallucination.py probe generators, for this policy):
# ``near_miss`` asks for the fee or delivery time of an option the policy does not name
# but a caller plausibly could ("priority delivery", "same day delivery"); ``low_coverage``
# asks for an attribute that a sparsely described entity does not have (a fee to secure
# the card). The truth is absence, admitted symbolically with its reason; the correct bot
# abstains. Each near-miss records how strongly the trained GMS binds the fake name to a
# real option (``fuzzy_match_entity``): a strong binding is exactly where an
# embedding-driven bot would hallucinate a figure.
# =====================================================================================================================

from ..s04_cases.meaning import SemanticFrame as F
from ..s01_packs.names import N
from ..s01_packs import names as roles

K = lambda intent, **kw: F.make(N.act.ASK_KB, intent=intent, **kw)


@dataclass
class KBItem:
    """One policy-question base case: id, category, question frame, true answer, recovery tag and details."""
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
    probe: str | None = None            # near_miss | low_coverage for hallucination probes


def _kx_two_way(store, truth: KBTruth) -> tuple[str, list[dict]]:
    """knowlytix's own two-way functionality check (benchmark.eval generators, geometric=True):
    the question is wrapped as a ``GeneratedQuestion`` whose ``answer_triples`` are our fact
    triples in the GMS vocabulary, and ``QuestionGenerator._validate`` decides - symbolic
    recompute, then rank-first among non-asserted tails and inside the relation's learned
    ``cap_radius``. The same gate the book's production answer-key generators use."""
    from .build import _quiet
    from .store import kx_triple
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
    """The admission gate: symbolic recompute AND knowlytix's geometric recovery of the answer triples."""
    if truth.recover() != truth.answer:
        return "rejected", [{"reason": "symbolic recompute disagrees"}]
    if not truth.answer_triples:
        return f"symbolic:{truth.symbolic_reason or 'no fact triple'}", []
    return _kx_two_way(store, truth)


def _generators(store, seed: int = 42) -> dict[str, list[F]]:
    rng = random.Random(seed)
    opts = [roles.FREE_OPTION, roles.FEE_OPTION]
    fee = {o: store.lookup_enm("delivery_fee_usd", o) for o in opts}
    L = store.lookup_enm("limit", N.limit.lookback_days)
    # plausibility: one committed and one counterfactual claim per option (balanced)
    plaus = []
    for o in opts:
        plaus.append(K("plausibility", attr="fee", entity=o, value=str(fee[o])))
        neg = [n for n in store.negatives_for(o, "has_fee_usd", str(fee[o]), rng)
               if n.replace(".", "").isdigit()]
        plaus.append(K("plausibility", attr="fee", entity=o, value=neg[0]))
    gt = fee[roles.FEE_OPTION]
    mcq_opts = sorted({gt - 10, gt - 5, gt, gt + 5} if gt >= 10 else {gt, gt + 5, gt + 10, gt + 15})
    return {
        "exact_recall": [K("value", attr="fee", entity=o) for o in opts] +
                        [K("value", attr="days", entity=o) for o in opts] +
                        [K("limit", key=k) for k in (N.limit.max_transactions_per_session, N.limit.lookback_days, N.limit.max_auth_attempts)],
        "fact_completion": [K("mcq", attr="fee", entity=roles.FEE_OPTION, options=tuple(mcq_opts))],
        "plausibility": plaus + [K("fraud")],
        "absence_detection": [K("refund")],
        "ranking": [K("faster")],
        "comparative_discrimination": [K("cheaper")],
        "counting": [K(N.value.count_options)],
        "cross_reference": [K(N.value.needs_confirmation)],
        "numeric_computation": [K(N.value.fee_difference)],
        "cross_table_aggregation": [K(N.value.expedited_total)],
        "multi_hop": [K(N.value.faster_option_fee)],
        "conditional_rule": [K(N.value.window_check, days=max(1, L // 3)), K(N.value.window_check, days=L + 15)],
        "boundary": [K(N.value.window_check, days=L), K(N.value.window_check, days=L + 1)],
        "procedural_ordering": [K("procedure")],
        "governance_policy": [K(N.value.skip_auth)],
    }


# near-miss option names a caller might use; none is an option or an alias in the policy
NEAR_MISS_OPTIONS = [("priority_delivery", "fee"), ("same_day_delivery", "days"), ("economy_delivery", "fee")]
# card actions whose 'fee' question the reader can phrase and read back (s05_rewording/wording.py)
def _readable_unstated() -> set[str]:
    return {N.value.SECURE_CARD}


# extended mode: more questions over the same policy facts, with intents the reader, the realizer and
# kb_truth already support - so every answer is still computed from the graph, never hand-labelled
EXTENDED_NEAR_MISS = [("premium_delivery", "fee"), ("next_day_delivery", "days"), ("saver_delivery", "fee"),
                      ("priority_delivery", "days"), ("economy_delivery", "days"), ("same_day_delivery", "fee")]


def _extended_generators(store) -> dict[str, list[F]]:
    opts = list(store.value_sets.get("delivery", [])) or [roles.FREE_OPTION, roles.FEE_OPTION]
    L = int(store.lookup_enm("limit", N.limit.lookback_days))
    out: dict[str, list[F]] = {
        # inside, just inside and far outside the review window (the existing ones: L/3, L, L+1, L+15)
        "conditional_rule": [K(N.value.window_check, days=d) for d in (1, L - 1, 2 * L) if d > 0],
        "plausibility": [K("plausibility", attr="fee", entity=o, value="0") for o in opts],   # "is it free?"
        "fact_completion": [],
    }
    for o in opts:                                   # multiple choice for every option's fee
        gt = int(store.lookup_enm("delivery_fee_usd", o))
        opts4 = sorted({gt - 10, gt - 5, gt, gt + 5} if gt >= 10 else {gt, gt + 5, gt + 10, gt + 15})
        out["fact_completion"].append(K("mcq", attr="fee", entity=o, options=tuple(opts4)))
    return out


def _reads_back(store, frame: F, state: str | None = None) -> bool:
    """The template wording of the question reads back to the same meaning card - at base levels and under
    heavier rewording - so an extended question can be enriched and proved like the others."""
    from ..s01_packs.names import graph_names
    state = state or graph_names().question_state
    from ..s05_rewording.wording import FrameReader
    from ..s05_rewording.wording import Realizer
    from ..s04_cases.meaning import Ontology
    rz, rd, ont = Realizer(), FrameReader(store), Ontology.from_store(store)
    for seed in ("a", "b", "c"):
        for lv in ({}, {"paraphrase_depth": "deep", "entity_aliasing": "colloquial", "numeric_format": "spoken"}):
            if not ont.equivalent(frame, rd.read(rz.realize(frame, lv, seed, {}), state), state):
                return False
    return True


def _probes(store) -> list[tuple[str, F, dict]]:
    """Hallucination probes: (kind, frame, diagnostic) - only probes whose truth is absence."""
    out = []
    options = set(store.value_sets.get("delivery", []))
    for name, attr in NEAR_MISS_OPTIONS:
        fr = K(N.value.unknown_option, attr=attr, name=name)
        if kb_answer(store, fr).answer != "absent":
            continue
        diag = {"name": name}
        try:
            bound, conf = store.fuzzy_match_entity(name.replace("_", " "), mode="frame")
            diag.update({"gms_binds_to": bound, "confidence": None if conf is None else round(float(conf), 4),
                         "binds_to_real_option": bound in options})
        except Exception as e:                      # noqa: BLE001 - diagnostic only
            diag["gms_binding_error"] = str(e)[:120]
        out.append(("near_miss", fr, diag))
    deg = Counter(e for h, _r, t in store.triples for e in (h, t))
    for ca in sorted(store.value_sets.get("card_action", []), key=lambda e: (deg[e], e)):
        if ca not in _readable_unstated():
            continue
        fr = K(N.value.unstated_fact, attr="fee", subject=ca)
        if kb_answer(store, fr).answer == "absent":
            out.append(("low_coverage", fr, {"subject": ca, "facts_about_subject": deg[ca]}))
    return out


def kx_substrate(store, full: bool = False) -> dict[str, Any]:
    """Mine the knowlytix store with the production answer-key generators (book nb 25):
    every category under the two-way geometric guard, with the per-question recovery tag
    and the calibrated cut each generator read. Nothing is admitted silently.

    cross_reference validates every member of every relation-pair overlap against every
    decoy tail (one Cayley rotor per score on CPU); on this graph that is hours, so it
    runs only with full=True and is otherwise reported as skipped with the reason."""
    from .build import _quiet
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


def mine_kb_items(store, seed: int = 42, extended: bool = False) -> tuple[list[KBItem], dict[str, Any]]:
    """extended=True adds more questions over the same policy facts (other days around the review window,
    'is it free?', multiple choice for every option, more near-miss option names). They are appended after
    the standard ones, so the standard question ids do not change, and each is kept only if its answer is
    computed from the graph and its wording reads back to the same meaning card."""
    items: list[KBItem] = []
    recovery_stats: dict[str, Counter] = {}
    cap_notes: dict[str, dict[str, float]] = {}
    gens = _generators(store, seed)
    ext_info = {"enabled": extended, "added": [], "dropped": []}
    if extended:
        seen = {f.canonical() for fs in gens.values() for f in fs}
        for cat, frames in _extended_generators(store).items():
            for f in frames:
                if f.canonical() in seen:
                    continue
                seen.add(f.canonical())
                try:
                    kb_answer(store, f)
                    ok = _reads_back(store, f)
                except Exception as e:              # noqa: BLE001 - a question the graph cannot answer
                    ext_info["dropped"].append({"frame": str(f), "why": str(e)[:120]})
                    continue
                if not ok:
                    ext_info["dropped"].append({"frame": str(f), "why": "wording does not read back"})
                    continue
                gens.setdefault(cat, []).append(f)
                ext_info["added"].append(str(f))
    for cat, frames in gens.items():
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

    # hallucination probes (absence truths; the bot must abstain)
    probe_stats = Counter()
    for j, (kind, fr, diag) in enumerate(_probes(store)):
        truth = kb_answer(store, fr)
        it = KBItem(f"kb-probe-{kind}-{j:02d}", "absence_detection", fr, truth.answer, truth.answer_type,
                    truth.primitive, probe=kind)
        tag, detail = two_way_check(store, truth)
        it.recovery, it.recovery_detail = tag, detail + [diag]
        it.admission = {"two_way": "doubly_grounded", "rejected": "rejected"}.get(tag, "symbolic_only")
        probe_stats[kind] += 1
        items.append(it)

    if extended:                                     # more near-miss names, numbered after the standard probes
        options = set(store.value_sets.get("delivery", []))
        known = {str(i.frame) for i in items}
        for j, (name, attr) in enumerate(EXTENDED_NEAR_MISS):
            fr = K(N.value.unknown_option, attr=attr, name=name)
            if str(fr) in known or name in options:
                continue
            truth = kb_answer(store, fr)
            if truth.answer != "absent" or not _reads_back(store, fr):
                ext_info["dropped"].append({"frame": str(fr), "why": "not absent or does not read back"})
                continue
            it = KBItem(f"kb-probe-near_miss-x{j:02d}", "absence_detection", fr, truth.answer, truth.answer_type,
                        truth.primitive, probe="near_miss")
            tag, detail = two_way_check(store, truth)
            it.recovery, it.recovery_detail = tag, detail + [{"name": name}]
            it.admission = {"two_way": "doubly_grounded", "rejected": "rejected"}.get(tag, "symbolic_only")
            probe_stats["near_miss"] += 1
            items.append(it)
            ext_info["added"].append(str(fr))

    # the geometry earns its keep: a forged question whose answer recomputes correctly
    # but whose declared fact triple is another option's real fee
    true_q = kb_answer(store, K("value", attr="fee", entity=roles.FEE_OPTION))
    other = str(store.lookup_enm("delivery_fee_usd", roles.FREE_OPTION))
    bad_q = KBTruth(true_q.intent, true_q.answer, true_q.answer_type, true_q.primitive, true_q.recover,
                    answer_triples=[(roles.FEE_OPTION, "has_fee_usd", other)])
    symbolic_only = lambda q: q.recover() == q.answer
    forged = {"true_fact": {"symbolic": symbolic_only(true_q), "two_way": two_way_check(store, true_q)[0]},
              "corrupted_triple": {"declared": (roles.FEE_OPTION, "has_fee_usd", other),
                                   "symbolic": symbolic_only(bad_q), "two_way": two_way_check(store, bad_q)[0]}}

    # store self-tests for the non customer-facing categories
    from .store import Rule
    planted = dict(store.rules)
    r13 = store.rules[N.rule.R13]
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
    recovery = {"gate": "knowlytix QuestionGenerator._validate (geometric=True)",
                "stats": {c: dict(v) for c, v in recovery_stats.items()},
                "cap_radius_read": cap_notes,
                "forged": forged,
                "multi_hop_ground_truth": None if mh is None else
                {"head": mh.ground_truth[0], "hop1_value": mh.ground_truth[1], "tails": sorted(mh.ground_truth[2]),
                 "recovery": mh.recovery}}
    recovery["knowlytix_substrate"] = kx_substrate(store)
    probes = [{"qid": i.qid, "kind": i.probe, "frame": str(i.frame), "admission": i.admission,
               "diagnostic": (i.recovery_detail or [{}])[-1]} for i in items if i.probe]
    return items, {"self_tests": self_tests, "precondition": precondition, "recovery": recovery, "extended": ext_info,
                   "hallucination_probes": {"n": len(probes), "by_kind": dict(probe_stats), "items": probes}}


# =====================================================================================================================
# PART 3 - Policy questions mined from a compact policy graph.
#
# Every numeric attribute of a value-set member (``card_product.PREMIUM_SUPP.annual_fee_usd``)
# and every customer-facing limit (``internal`` limits excluded) is a question whose answer is an
# exact register read. Each question passes the same two-way admission gate as the v1 policy
# questions (symbolic recompute + knowlytix's geometric recovery of its fact triple, s03_knowledge/questions.py).
# =====================================================================================================================

from ..s02_graph.compact import KB_ACT, PolicyGraph, attr_relation

_KQ = lambda intent, **kw: F.make(KB_ACT, intent=intent, **kw)


def answer_type(attr: str) -> str:
    """The type of a figure's answer: ``money`` for an attribute ending in ``_usd``, else ``number``."""
    return "money" if attr.endswith("_usd") else "number"


def kb_frames(g: PolicyGraph) -> list[F]:
    """The askable questions: every numeric attribute the lexicon has wording for (all of them when
    it lists none) on every member, and every customer-facing limit."""
    from ..s01_packs.loading import current
    asked = (current().get("lexicon.kb_questions.attributes") or {})
    out = []
    for attr, spec in g.numeric_attributes().items():
        if asked and attr not in asked:
            continue
        for m in spec["members"]:
            out.append(_KQ("value", attr=attr, entity=m))
    for k, lv in g.limits.items():
        if not lv.get("internal"):
            out.append(_KQ("limit", key=k))
    return out


def graph_kb_answer(store, frame: F):
    """KBTruth for a graph-domain policy question (an exact register read)."""
    from ..s01_packs.loading import current
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


def mine_graph_kb_items(store, seed: int = 42, extended: bool = False) -> tuple[list, dict[str, Any]]:
    """Policy-question base cases of a compiled pack: one per question frame of its graph (values and limits), each
    answered by the store and admitted by the two-way check; returns the items and per-category counts.
    """
    from ..s01_packs.loading import current
    g = current().graph
    items, stats = [], {}
    for j, fr in enumerate(kb_frames(g)):
        cat = "exact_recall" if fr.get("intent") == "value" else "conditional_rule"
        truth = graph_kb_answer(store, fr)
        it = KBItem(f"kb-{cat}-{j:02d}", cat, fr, truth.answer, truth.answer_type, truth.primitive)
        tag, detail = two_way_check(store, truth)
        it.recovery, it.recovery_detail = tag, detail
        it.admission = {"two_way": "doubly_grounded", "rejected": "rejected"}.get(tag, "symbolic_only")
        stats.setdefault(cat, Counter())["two_way" if tag == "two_way" else
                                         "symbolic" if tag.startswith("symbolic") else "rejected"] += 1
        items.append(it)
    self_tests = store_self_tests(store)
    graded = [judge_kb_answer(store, it.frame, render_truth(graph_kb_answer(store, it.frame)))["correct"]
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
    from .store import Rule
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
