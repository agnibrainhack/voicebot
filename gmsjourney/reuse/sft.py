"""Training-data consumers of the design (book Ch. 8) with proofloop's emitters.

* ``classifier_corpus``  proofloop.suite.emit.emit_classifier_sft over every caller
  turn of every proven scenario: {message, label, _seed, _factors}. The label is the
  turn's canonical semantic frame; enrichment varies the surface only, so one seed
  turn keeps one label across all its phrasings (checked). Each record is also run
  through the grounding-preserving filter - the FrameReader must read the message
  back to the labelled frame - so nothing that would teach a wrong binding is kept.
* ``draft_corpus``  proofloop.suite.emit.emit_draft_sft over the policy-question
  bases with golden contracts read from the store (citation keywords, the byte-exact
  ENM figure, forbidden unauthorised commitments); a draft that breaks its contract
  is dropped. Run with the reference bot's answerer and with a seeded defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Item:                       # proofloop QAItem shape
    qid: str
    query: str
    answer: Any
    answer_type: str = "label"
    base: str | None = None
    components: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Scn:                        # proofloop Scenario shape
    sid: str
    base: _Item
    factor_levels: dict[str, str]
    mode: str = "embedded"
    metadata: dict[str, Any] = field(default_factory=dict)


def classifier_corpus(proofs, reader, ontology) -> dict[str, Any]:
    from proofloop.suite.emit import emit_classifier_sft
    scns = []
    for p in proofs:
        if not p.accepted:
            continue
        for i, t in enumerate(p.turns):
            label = t.get("frame")
            item = _Item(qid=f"{p.qid}#t{i}", query=label, answer=label, base="journey_turn",
                         components={"expected_classification": label})
            scns.append(_Scn(f"{p.sid}#t{i}", item, dict(p.levels), metadata={"text": t.get("enriched_text"), "frame": t.get("frame_obj")}))
    recs = emit_classifier_sft(scns, materialize_fn=lambda s: s.metadata["text"])
    kept, dropped = [], []
    for r, s in zip(recs, scns):
        fr = s.metadata.get("frame")
        ok = fr is None or ontology.equivalent(fr, reader.read(r["message"]))
        (kept if ok else dropped).append(r)
    by_seed: dict[str, set] = {}
    for r in kept:
        by_seed.setdefault(r["_seed"], set()).add(r["label"])
    return {"records": kept, "n": len(kept), "dropped_by_grounding_filter": len(dropped),
            "schema": sorted(recs[0]) if recs else [], "seeds": len(by_seed),
            "one_label_per_seed": all(len(v) == 1 for v in by_seed.values()),
            "distinct_messages": len({r["message"] for r in kept})}


def goldens_from_store(store) -> dict[str, dict]:
    fee = store.lookup_enm("delivery_fee_usd", "expedited_delivery")
    std = store.lookup_enm("delivery_fee_usd", "standard_delivery")
    dmin, dmax = (store.lookup_enm("delivery_days_min", "expedited_delivery"),
                  store.lookup_enm("delivery_days_max", "expedited_delivery"))
    forbid = ["is fraud", "means fraud", "guarantee", "we will refund", "full refund"]
    L = lambda k: store.lookup_enm("limit", k)
    return {
        "value:fee:expedited_delivery": {"citation_keywords": ["expedited"], "required_numbers": [str(fee)], "forbidden_phrases": forbid},
        "value:fee:standard_delivery": {"citation_keywords": ["standard"], "required_numbers": [str(std)], "forbidden_phrases": forbid},
        "value:days:expedited_delivery": {"citation_keywords": ["expedited"], "required_numbers": [str(dmax)], "forbidden_phrases": forbid},
        "limit:max_transactions_per_session": {"citation_keywords": [], "required_numbers": [str(L("max_transactions_per_session"))], "forbidden_phrases": forbid},
        "limit:lookback_days": {"citation_keywords": [], "required_numbers": [str(L("lookback_days"))], "forbidden_phrases": forbid},
        "fraud": {"citation_keywords": [], "required_numbers": [], "forbidden_phrases": forbid},
        "refund": {"citation_keywords": [], "required_numbers": [], "forbidden_phrases": forbid + ["$"]},
    }


def _policy_id(frame) -> str:
    i = frame.get("intent")
    if i == "value":
        return f"value:{frame.get('attr')}:{frame.get('entity')}"
    if i == "limit":
        return f"limit:{frame.get('key')}"
    return i


def draft_corpus(store, kb_items, answer_fn, realizer) -> dict[str, Any]:
    from proofloop.suite.emit import emit_draft_sft, passes_contract
    goldens = goldens_from_store(store)
    scns = []
    for k in kb_items:
        pid = _policy_id(k.frame)
        q = realizer.realize(k.frame, {}, "draft")
        scns.append(_Scn(k.qid, _Item(k.qid, q, k.answer, metadata={"policy_id": pid, "issue": k.category,
                                                                      "frame": k.frame}), {}))
    kept, dropped = emit_draft_sft(scns, response_fn=lambda s: answer_fn(s.base.metadata["frame"], s.base.query),
                                   goldens=goldens, source="gmsjourney")
    return {"kept": len(kept), "dropped": len(dropped), "records": kept, "dropped_records": dropped,
            "contracts": len(goldens), "covered_by_contract": sum(1 for s in scns if s.base.metadata["policy_id"] in goldens),
            "contract_check": {"faithful": passes_contract("Expedited delivery has a $15 fee.", goldens["value:fee:expedited_delivery"]),
                               "unauthorised": passes_contract("It's fraud, we will refund you.", goldens["fraud"])}}
