"""What a campaign writes besides the report: the dataset tables and the training corpora.

Parts of this file:
  1. proofloop's failure-mode taxonomy mapped onto the probes this campaign runs (which mutants, faults, factors
     and gates exercise each mode).
  2. Training data: the proved utterances as a classifier corpus, and policy answers as a draft corpus, through
     proofloop's emitters.
  3. ``write_dataset``: the generated test suite as clean CSV / JSONL tables (cases, turns, results).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =====================================================================================================================
# PART 1 - proofloop's failure-mode taxonomy mapped onto the probes this campaign runs.
#
# proofloop.evaluation.failure_modes.FailureMode names nine agent failure modes. A
# suite only covers a mode if some probe exercises it; this table records which
# seeded defect, injected fault, factor level or gate probes each mode, and whether
# the campaign caught it (mutation kill / resilience / gate detection).
# =====================================================================================================================

PROBES = {
    "wrong_tool": {"mutants": ["present_all_returned"], "faults": [], "factors": [],
                   "gates": ["gate probe sweep: every out-of-workflow transition"]},
    "malformed_call": {"mutants": [], "faults": [], "factors": [], "gates": ["SyntaxGate schema probes"]},
    "hallucinated_citation": {"mutants": ["kb_first_amount", "fraud_claim"], "faults": [],
                              "factors": ["answer_anchoring=wrong_suggestion"], "gates": ["speech ENM verifier", "R27 stance"]},
    "prompt_injection": {"mutants": [], "faults": [], "factors": ["instruction_conflict=authority_override",
                                                                 "instruction_conflict=contradictory_hint"],
                         "gates": ["input guard (forgeloop regex baseline)"]},
    "over_delegation": {"mutants": ["ignore_human_mid_review"], "faults": [], "factors": [], "gates": ["R20 policy"]},
    "infinite_loop": {"mutants": ["no_turn_limit"], "faults": [], "factors": [], "gates": ["R28 policy"]},
    "premature_stop": {"mutants": ["card_erases_review", "unsure_no_transfer"], "faults": [], "factors": [],
                       "gates": ["completeness R12/R18", "R11 policy"]},
    "stale_memory": {"mutants": ["first_answer_wins"], "faults": ["stale"], "factors": ["self_correction=false_start"],
                     "gates": ["R21 policy"]},
    "irrelevant_retrieval": {"mutants": ["kb_first_amount"], "faults": [], "factors": ["history=distracting_history"],
                             "gates": ["speech ENM verifier"]},
}


def coverage(mutation: dict | None, resilience: dict | None, flagged_by_monitor: dict | None) -> list[dict[str, Any]]:
    """For each proofloop failure mode: the mutants, faults, factors and gates that probe it, and which mutants
    were killed or flagged at run time.
    """
    from proofloop.evaluation.failure_modes import FailureMode
    rows = []
    for mode in FailureMode:
        p = PROBES.get(mode.value, {})
        killed = [m for m in p.get("mutants", []) if mutation and mutation["mutants"].get(m, {}).get("killed")]
        flagged = [m for m in p.get("mutants", []) if flagged_by_monitor and flagged_by_monitor.get(m)]
        rows.append({"failure_mode": mode.value, **p, "mutants_killed": killed, "mutants_flagged_at_runtime": flagged,
                     "exercised": bool(p.get("mutants") or p.get("faults") or p.get("factors") or p.get("gates"))})
    return rows


# =====================================================================================================================
# PART 2 - Training-data consumers of the design (book Ch. 8) with proofloop's emitters.
#
# * ``classifier_corpus``  proofloop.suite.emit.emit_classifier_sft over every caller
#   turn of every proven scenario: {message, label, _seed, _factors}. The label is the
#   turn's canonical semantic frame; enrichment varies the surface only, so one seed
#   turn keeps one label across all its phrasings (checked). Each record is also run
#   through the grounding-preserving filter - the FrameReader must read the message
#   back to the labelled frame - so nothing that would teach a wrong binding is kept.
# * ``draft_corpus``  proofloop.suite.emit.emit_draft_sft over the policy-question
#   bases with golden contracts read from the store (citation keywords, the byte-exact
#   ENM figure, forbidden unauthorised commitments); a draft that breaks its contract
#   is dropped. Run with the reference bot's answerer and with a seeded defect.
# =====================================================================================================================

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
    """Training records for an utterance classifier: every proved customer turn with its meaning, kept only
    when the reader reads it back as that meaning.
    """
    from proofloop.suite.emit import emit_classifier_sft
    scns = []
    for p in proofs:
        if not p.accepted:
            continue
        for i, t in enumerate(p.turns):
            label = t.get("frame")
            item = _Item(qid=f"{p.qid}#t{i}", query=label, answer=label, base="journey_turn",
                         components={"expected_classification": label})
            scns.append(_Scn(f"{p.sid}#t{i}", item, dict(p.levels),
                             metadata={"text": t.get("enriched_text"), "frame": t.get("frame_obj"),
                                       "state": t.get("state")}))
    recs = emit_classifier_sft(scns, materialize_fn=lambda s: s.metadata["text"])
    kept, dropped = [], []
    for r, s in zip(recs, scns):
        fr = s.metadata.get("frame")
        st = s.metadata.get("state")
        ok = fr is None or ontology.equivalent(fr, reader.read(r["message"], st), st)
        (kept if ok else dropped).append(r)
    by_seed: dict[str, set] = {}
    for r in kept:
        by_seed.setdefault(r["_seed"], set()).add(r["label"])
    return {"records": kept, "n": len(kept), "dropped_by_grounding_filter": len(dropped),
            "schema": sorted(recs[0]) if recs else [], "seeds": len(by_seed),
            "one_label_per_seed": all(len(v) == 1 for v in by_seed.values()),
            "distinct_messages": len({r["message"] for r in kept})}


def _pack_goldens(store, dom) -> dict[str, dict]:
    """A contract for every figure of a compiled pack: each numeric attribute of each member (cited by the member's
    wording) and each customer-facing limit."""
    g = dom.graph
    lex = dom.get("lexicon.values") or {}
    forbid = ["guarantee", "we promise"]
    out = {}
    for attr, spec in g.numeric_attributes().items():
        for m in spec["members"]:
            say = (lex.get(m) or {}).get("say") or {}
            words = [w for w in (say.get("canonical"), *g.aliases(m)) if w]
            key = (words[0].split()[-1] if words else m.lower()).strip(".")
            out[f"value:{attr}:{m}"] = {"citation_keywords": [key], "required_numbers": [_n(store.lookup_enm(attr, m))],
                                        "forbidden_phrases": forbid}
    for k, lv in g.limits.items():
        if not lv.get("internal"):
            out[f"limit:{k}"] = {"citation_keywords": [], "required_numbers": [_n(store.lookup_enm("limit", k))],
                                 "forbidden_phrases": forbid}
    return out


def _n(v) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def goldens_from_store(store) -> dict[str, dict]:
    """The contracts a good policy answer must meet: the keywords to cite, the numbers to state, the phrases
    never to say; read from the store.
    """
    from ..s01_packs.loading import current
    if current().graph is not None:
        return _pack_goldens(store, current())
    fee = store.lookup_enm("delivery_fee_usd", "expedited_delivery")
    std = store.lookup_enm("delivery_fee_usd", "standard_delivery")
    dmax = store.lookup_enm("delivery_days_max", "expedited_delivery")
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
    """Training records for policy answers: each admitted question with a bot's answer, kept when the answer
    passes its contract.
    """
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
            "contract_check": _contract_check(goldens, passes_contract)}


def _contract_check(goldens, passes_contract) -> dict[str, bool]:
    """The contract checker on one faithful and one unauthorised answer."""
    if "value:fee:expedited_delivery" in goldens:
        return {"faithful": passes_contract("Expedited delivery has a $15 fee.", goldens["value:fee:expedited_delivery"]),
                "unauthorised": passes_contract("It's fraud, we will refund you.", goldens["fraud"])}
    key, gd = next((k, v) for k, v in goldens.items() if v["required_numbers"])
    cite = " ".join(gd["citation_keywords"])
    return {"faithful": passes_contract(f"The {cite} figure is {gd['required_numbers'][0]}.", gd),
            "unauthorised": passes_contract(f"The {cite} figure is {gd['required_numbers'][0]}, we guarantee it.", gd)}


# =====================================================================================================================
# PART 3 - The generated test suite as clean tables (<out_dir>/dataset/).
#
# ``test_cases.csv``    one row per test case: the base cases and every enriched variant of
#                       both arms (template and, when the LLM arm ran, LLM), accepted or not.
#                       Columns: case_id, case_type (base / enriched), arm (base / template /
#                       llm), base_id,
#                       category, tags, policy_question (+ its meaning card and the answer the
#                       store computes), canonicals (every canonical value and state the case
#                       uses), meaning_cards (the customer's turns as state-aware canonical
#                       cards), base_case (the conversation at base levels), enrichment (the
#                       factor levels; changed factors listed separately), enriched_output (the
#                       enriched conversation), expected_outcome / expected_components (the
#                       oracle's answer key), trajectory (the complete expected run: bot acts,
#                       customer turns with their state and card, tool calls with results),
#                       states_visited, rules_fired, workflow_edges and the proof verdict.
# ``test_turns.csv``    one row per customer turn of every case: dialog state, its stage, the
#                       rules and limits that apply there (policy graph), the cards it accepts,
#                       the meaning card and its head/relation/tail form, base text, enriched
#                       text, the card read back from the enriched text, equivalence,
#                       admissibility, distance, surface cosine, and the tools the oracle calls
#                       right after the turn.
# ``policy_questions.csv``  the 27 policy-question bases: category, question, card, answer,
#                       the store primitive that computes it, admission and recovery tag.
# ``test_cases.jsonl``  the same cases with nested structures (for code).
#
# Nested values in CSV cells are JSON strings.
# =====================================================================================================================

from ..s05_rewording.wording import BASE_LEVELS
from ..s03_knowledge.questions import kb_answer
from ..s01_packs.names import N, graph_names


def _j(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, default=str, sort_keys=False)


def _card_hrt(card: dict, head: str) -> list[list]:
    out = [[head, "in_state", card.get("state")], [head, "has_act", card.get("act")]]
    for k, v in (card.get("slots") or {}).items():
        for x in (v if isinstance(v, list) else [v]):
            out.append([head, f"has_{k}", x])
    return out


def _canonicals(cards: list[dict]) -> list[str]:
    vals = set()
    for c in cards:
        if c.get("state"):
            vals.add(c["state"])
        if c.get("act"):
            vals.add(c["act"])
        for v in (c.get("slots") or {}).values():
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, str):
                    vals.add(x)
    return sorted(vals)


def _trajectory(trace, customer_texts: list[str], cards: list[dict]) -> list[dict]:
    """The complete expected run, with the customer's words for this case."""
    steps, ui = [], 0
    for ev in trace.events:
        if ev["role"] == "bot":
            steps.append({"actor": "bot", "acts": [a.get("act") for a in ev["acts"]],
                          "state": ev["acts"][-1].get("act") if ev["acts"] else None})
        elif ev["role"] == "user":
            steps.append({"actor": "customer", "turn": ui, "state": cards[ui]["state"] if ui < len(cards) else None,
                          "card": {k: cards[ui][k] for k in ("act", "slots")} if ui < len(cards) else None,
                          "text": customer_texts[ui] if ui < len(customer_texts) else None})
            ui += 1
        elif ev["role"] == "tool":
            res = ev.get("result")
            if isinstance(res, dict):
                if isinstance(res.get("transactions"), list):
                    res = {"status": res.get("status"),
                           "transactions": [t.get("transaction_id") for t in res["transactions"]]}
                else:
                    res = {k: v for k, v in res.items() if k in ("status", "transaction_id", "disposition",
                                                                 "options", "action", "reason", "error")} or res
            steps.append({"actor": "tool", "tool": ev["tool"], "args": ev.get("args"), "result": res})
    return steps


def _tools_after_turns(trace) -> list[list[str]]:
    out, cur = [], None
    for ev in trace.events:
        if ev["role"] == "user":
            cur = []
            out.append(cur)
        elif ev["role"] == "tool" and cur is not None:
            cur.append(ev["tool"])
    return out


def write_dataset(out_dir: str | Path, *, store, kg, oracle, validator, bases, scenarios, proofs,
                  kb_items, llm_scenarios=None, llm_proofs=None) -> dict[str, Any]:
    """llm_scenarios / llm_proofs: the LLM arm's variants (proved separately); written as enriched
    cases with arm=llm next to the template arm's (arm=template)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ont = validator.ont
    kb_by_qid = {k.qid: k for k in kb_items}
    proofs_by_sid = {p.sid: p for p in list(proofs) + list(llm_proofs or [])}
    all_scenarios = [(s, "template") for s in scenarios] + [(s, "llm") for s in (llm_scenarios or [])]
    case_rows, turn_rows, jsonl = [], [], []

    def state_policy(st):
        if kg is None or not st:
            return {}
        return {"stage": kg.stage(st), "rules": kg.rules_for(st),
                "limits": {k: kg.limit(k) for k in kg.limits_for(st)}, "accepts": kg.accepted_acts(st)}

    def policy_question(item, path, states, texts):
        rows = []
        for i, (f, _ctx) in enumerate(path):
            if f.act == N.act.ASK_KB:
                truth = kb_answer(store, f)
                rows.append({"turn": i, "text": texts[i], "card": ont.card(f, states[i]),
                             "answer": truth.answer, "primitive": truth.primitive})
        return rows

    for item in bases:
        path = validator.base_path(item)
        states = validator._states.get(item.qid, [None] * len(path))
        trace = oracle.run(item.metadata["spec"], item.metadata["fixture"])
        tools_after = _tools_after_turns(trace)
        base_texts = [validator.realizer.realize(f, {}, f"{item.qid}|{i}|base", ctx) for i, (f, ctx) in enumerate(path)]
        base_cards = [ont.card(f, st) for (f, _), st in zip(path, states)]
        pq = policy_question(item, path, states, base_texts)
        kbi = kb_by_qid.get(item.metadata.get("kb_qid"))
        common = {
            "base_id": item.qid, "category": item.base, "tags": item.metadata.get("tags", []),
            "policy_question": pq[0]["text"] if pq else "",
            "policy_question_count": len(pq),
            "policy_question_card": pq[0]["card"] if pq else None,
            "policy_answer": (kbi.answer if kbi else (pq[0]["answer"] if pq else None)),
            "expected_outcome": json.loads(item.answer) if isinstance(item.answer, str) else item.answer,
            "expected_components": item.components,
            "states_visited": [s for s in dict.fromkeys(states) if s],
            "rules_fired": item.metadata.get("rules_fired"), "workflow_edges": item.metadata.get("edges"),
        }
        conv = lambda texts: " || ".join(f"T{i} [{st}] {t}" for i, (st, t) in enumerate(zip(states, texts)))

        # ---- the base case
        case = {"case_id": f"{item.qid}~base", "case_type": "base", "arm": "base", **common,
                "canonicals": _canonicals(base_cards),
                "meaning_cards": [{"turn": i, **{k: c[k] for k in ("state", "act", "slots")}}
                                  for i, c in enumerate(base_cards)],
                "base_case": conv(base_texts), "enrichment": {}, "enrichment_changed": "",
                "enriched_output": "", "trajectory": _trajectory(trace, base_texts, base_cards),
                "proof_accepted": True, "proof_status": "accepted", "proof_layers": "base", "n_turns": len(path), "surface_cosine_mean": 1.0}
        case_rows.append(case)
        jsonl.append(case)
        for i, ((f, _), st) in enumerate(zip(path, states)):
            sp = state_policy(st)
            turn_rows.append({
                "case_id": case["case_id"], "case_type": "base", "arm": "base", "base_id": item.qid,
                "category": item.base,
                "turn": i, "state": st, "stage": sp.get("stage"), "state_rules": ";".join(sp.get("rules", [])),
                "state_limits": ";".join(f"{k}={v}" for k, v in sp.get("limits", {}).items()),
                "accepted_acts": ";".join(sp.get("accepts", [])),
                "meaning_card": base_cards[i], "canonical_hrt": _card_hrt(base_cards[i], f"{item.qid}/t{i}"),
                "base_text": base_texts[i], "enrichment": {}, "enriched_text": "", "read_back_card": base_cards[i],
                "equivalent": True, "admissible": base_cards[i]["admissible"], "frame_distance": 0.0,
                "surface_cosine": 1.0, "next_tools": ";".join(tools_after[i] if i < len(tools_after) else [])})

        # ---- its enriched variants, both arms
        for scn, arm in [(s, a) for s, a in all_scenarios if s.base.qid == item.qid]:
            p = proofs_by_sid.get(scn.sid)
            if p is None:
                continue
            enr_texts = [t["enriched_text"] for t in p.turns]
            changed = {k: v for k, v in scn.factor_levels.items() if BASE_LEVELS.get(k) not in (None, v)}
            case = {"case_id": scn.sid, "case_type": "enriched", "arm": arm, **common,
                    "canonicals": _canonicals([t["read_card"] for t in p.turns]),
                    "meaning_cards": [{"turn": t["turn"], **{k: t["read_card"][k] for k in ("state", "act", "slots")}}
                                      for t in p.turns],
                    "base_case": conv(base_texts), "enrichment": dict(scn.factor_levels),
                    "enrichment_changed": "; ".join(f"{k}={v}" for k, v in changed.items()),
                    "enriched_output": conv(enr_texts),
                    "trajectory": _trajectory(trace, enr_texts, [t["read_card"] for t in p.turns]),
                    "proof_accepted": p.accepted,
                    # accepted | rejected (a gate's verdict: the meaning changed) | not_generated (a gateway
                    # call failed, so there is no verdict; never run, never counted against invariance)
                    "proof_status": "accepted" if p.accepted else
                                    ("not_generated" if p.extra.get("llm_unavailable") else "rejected"),
                    "proof_layers": f"L1={p.gt_identical} L2={p.frames_equivalent} L3={p.oracle_rederived}",
                    "n_turns": p.n_turns, "surface_cosine_mean": round(p.mean_surface_cosine, 4)}
            if pq:
                case["policy_question_enriched"] = enr_texts[pq[0]["turn"]]
            case_rows.append(case)
            jsonl.append(case)
            for t in p.turns:
                i, st = t["turn"], t["state"]
                sp = state_policy(st)
                turn_rows.append({
                    "case_id": scn.sid, "case_type": "enriched", "arm": arm, "base_id": item.qid,
                    "category": item.base,
                    "turn": i, "state": st, "stage": sp.get("stage"), "state_rules": ";".join(sp.get("rules", [])),
                    "state_limits": ";".join(f"{k}={v}" for k, v in sp.get("limits", {}).items()),
                    "accepted_acts": ";".join(sp.get("accepts", [])),
                    "meaning_card": t["card"], "canonical_hrt": _card_hrt(t["read_card"], f"{scn.sid}/t{i}"),
                    "base_text": t["base_text"], "enrichment": dict(scn.factor_levels),
                    "enriched_text": t["enriched_text"], "read_back_card": t["read_card"],
                    "turn_source": t.get("turn_source") or arm,
                    "equivalent": t["equivalent"], "admissible": t["read_card"]["admissible"],
                    "frame_distance": t["frame_distance"], "surface_cosine": round(t["surface_cosine"], 4),
                    "next_tools": ";".join(tools_after[i] if i < len(tools_after) else [])})

    case_cols = ["case_id", "case_type", "arm", "base_id", "category", "tags", "policy_question", "policy_question_count",
                 "policy_question_enriched", "policy_question_card", "policy_answer", "canonicals", "meaning_cards",
                 "base_case", "enrichment", "enrichment_changed", "enriched_output", "expected_outcome",
                 "expected_components", "trajectory", "states_visited", "rules_fired", "workflow_edges",
                 "proof_accepted", "proof_status", "proof_layers", "n_turns", "surface_cosine_mean"]
    turn_cols = ["case_id", "case_type", "arm", "base_id", "category", "turn", "state", "stage", "state_rules",
                 "state_limits", "accepted_acts", "meaning_card", "canonical_hrt", "base_text", "enrichment",
                 "enriched_text", "turn_source", "read_back_card", "equivalent", "admissible", "frame_distance", "surface_cosine",
                 "next_tools"]

    def write(path, rows, cols):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (_j(v) if isinstance(v, (dict, list, tuple)) or v is None and k.endswith("card")
                                else ("" if v is None else v)) for k, v in r.items()})

    write(out / "test_cases.csv", case_rows, case_cols)
    write(out / "test_turns.csv", turn_rows, turn_cols)
    with open(out / "test_cases.jsonl", "w", encoding="utf-8") as f:
        for r in jsonl:
            f.write(_j(r) + "\n")
    pq_rows = []
    st = graph_names().question_state
    for k in kb_items:
        pq_rows.append({"qid": k.qid, "category": k.category,
                        "question": validator.realizer.realize(k.frame, {}, f"{k.qid}|q|base", {}),
                        "meaning_card": ont.card(k.frame, st), "answer": k.answer, "answer_type": k.answer_type,
                        "primitive": k.primitive, "admission": k.admission, "recovery": k.recovery})
    write(out / "policy_questions.csv", pq_rows, ["qid", "category", "question", "meaning_card", "answer",
                                                   "answer_type", "primitive", "admission", "recovery"])
    return {"dir": str(out), "test_cases": len(case_rows), "base_cases": sum(r["case_type"] == "base" for r in case_rows),
            "enriched_cases": sum(r["case_type"] == "enriched" for r in case_rows),
            "llm_cases": sum(r.get("arm") == "llm" for r in case_rows), "turn_rows": len(turn_rows),
            "policy_questions": len(pq_rows),
            "files": ["test_cases.csv", "test_turns.csv", "test_cases.jsonl", "policy_questions.csv"]}
