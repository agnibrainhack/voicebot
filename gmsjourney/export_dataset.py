"""The generated test suite as clean tables (campaign_out/dataset/).

``test_cases.csv``    one row per test case: the 81 base cases and their 486 enriched
                      variants. Columns: case_id, case_type (base / enriched), base_id,
                      category, tags, policy_question (+ its meaning card and the answer the
                      store computes), canonicals (every canonical value and state the case
                      uses), meaning_cards (the customer's turns as state-aware canonical
                      cards), base_case (the conversation at base levels), enrichment (the
                      factor levels; changed factors listed separately), enriched_output (the
                      enriched conversation), expected_outcome / expected_components (the
                      oracle's answer key), trajectory (the complete expected run: bot acts,
                      customer turns with their state and card, tool calls with results),
                      states_visited, rules_fired, workflow_edges and the proof verdict.
``test_turns.csv``    one row per customer turn of every case: dialog state, its stage, the
                      rules and limits that apply there (policy graph), the cards it accepts,
                      the meaning card and its head/relation/tail form, base text, enriched
                      text, the card read back from the enriched text, equivalence,
                      admissibility, distance, surface cosine, and the tools the oracle calls
                      right after the turn.
``policy_questions.csv``  the 27 policy-question bases: category, question, card, answer,
                      the store primitive that computes it, admission and recovery tag.
``test_cases.jsonl``  the same cases with nested structures (for code).

Nested values in CSV cells are JSON strings.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .enrich.realizers import BASE_LEVELS
from .kb_truth import kb_answer


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
                  kb_items) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ont = validator.ont
    kb_by_qid = {k.qid: k for k in kb_items}
    proofs_by_sid = {p.sid: p for p in proofs}
    case_rows, turn_rows, jsonl = [], [], []

    def state_policy(st):
        if kg is None or not st:
            return {}
        return {"stage": kg.stage(st), "rules": kg.rules_for(st),
                "limits": {k: kg.limit(k) for k in kg.limits_for(st)}, "accepts": kg.accepted_acts(st)}

    def policy_question(item, path, states, texts):
        rows = []
        for i, (f, _ctx) in enumerate(path):
            if f.act == "ASK_KB":
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
        case = {"case_id": f"{item.qid}~base", "case_type": "base", **common,
                "canonicals": _canonicals(base_cards),
                "meaning_cards": [{"turn": i, **{k: c[k] for k in ("state", "act", "slots")}}
                                  for i, c in enumerate(base_cards)],
                "base_case": conv(base_texts), "enrichment": {}, "enrichment_changed": "",
                "enriched_output": "", "trajectory": _trajectory(trace, base_texts, base_cards),
                "proof_accepted": True, "proof_layers": "base", "n_turns": len(path), "surface_cosine_mean": 1.0}
        case_rows.append(case)
        jsonl.append(case)
        for i, ((f, _), st) in enumerate(zip(path, states)):
            sp = state_policy(st)
            turn_rows.append({
                "case_id": case["case_id"], "case_type": "base", "base_id": item.qid, "category": item.base,
                "turn": i, "state": st, "stage": sp.get("stage"), "state_rules": ";".join(sp.get("rules", [])),
                "state_limits": ";".join(f"{k}={v}" for k, v in sp.get("limits", {}).items()),
                "accepted_acts": ";".join(sp.get("accepts", [])),
                "meaning_card": base_cards[i], "canonical_hrt": _card_hrt(base_cards[i], f"{item.qid}/t{i}"),
                "base_text": base_texts[i], "enrichment": {}, "enriched_text": "", "read_back_card": base_cards[i],
                "equivalent": True, "admissible": base_cards[i]["admissible"], "frame_distance": 0.0,
                "surface_cosine": 1.0, "next_tools": ";".join(tools_after[i] if i < len(tools_after) else [])})

        # ---- its enriched variants
        for scn in [s for s in scenarios if s.base.qid == item.qid]:
            p = proofs_by_sid.get(scn.sid)
            if p is None:
                continue
            enr_texts = [t["enriched_text"] for t in p.turns]
            changed = {k: v for k, v in scn.factor_levels.items() if BASE_LEVELS.get(k) not in (None, v)}
            case = {"case_id": scn.sid, "case_type": "enriched", **common,
                    "canonicals": _canonicals([t["read_card"] for t in p.turns]),
                    "meaning_cards": [{"turn": t["turn"], **{k: t["read_card"][k] for k in ("state", "act", "slots")}}
                                      for t in p.turns],
                    "base_case": conv(base_texts), "enrichment": dict(scn.factor_levels),
                    "enrichment_changed": "; ".join(f"{k}={v}" for k, v in changed.items()),
                    "enriched_output": conv(enr_texts),
                    "trajectory": _trajectory(trace, enr_texts, [t["read_card"] for t in p.turns]),
                    "proof_accepted": p.accepted,
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
                    "case_id": scn.sid, "case_type": "enriched", "base_id": item.qid, "category": item.base,
                    "turn": i, "state": st, "stage": sp.get("stage"), "state_rules": ";".join(sp.get("rules", [])),
                    "state_limits": ";".join(f"{k}={v}" for k, v in sp.get("limits", {}).items()),
                    "accepted_acts": ";".join(sp.get("accepts", [])),
                    "meaning_card": t["card"], "canonical_hrt": _card_hrt(t["read_card"], f"{scn.sid}/t{i}"),
                    "base_text": t["base_text"], "enrichment": dict(scn.factor_levels),
                    "enriched_text": t["enriched_text"], "read_back_card": t["read_card"],
                    "equivalent": t["equivalent"], "admissible": t["read_card"]["admissible"],
                    "frame_distance": t["frame_distance"], "surface_cosine": round(t["surface_cosine"], 4),
                    "next_tools": ";".join(tools_after[i] if i < len(tools_after) else [])})

    case_cols = ["case_id", "case_type", "base_id", "category", "tags", "policy_question", "policy_question_count",
                 "policy_question_enriched", "policy_question_card", "policy_answer", "canonicals", "meaning_cards",
                 "base_case", "enrichment", "enrichment_changed", "enriched_output", "expected_outcome",
                 "expected_components", "trajectory", "states_visited", "rules_fired", "workflow_edges",
                 "proof_accepted", "proof_layers", "n_turns", "surface_cosine_mean"]
    turn_cols = ["case_id", "case_type", "base_id", "category", "turn", "state", "stage", "state_rules",
                 "state_limits", "accepted_acts", "meaning_card", "canonical_hrt", "base_text", "enrichment",
                 "enriched_text", "read_back_card", "equivalent", "admissible", "frame_distance", "surface_cosine",
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
    for k in kb_items:
        st = "PRESENT_TXN"
        pq_rows.append({"qid": k.qid, "category": k.category,
                        "question": validator.realizer.realize(k.frame, {}, f"{k.qid}|q|base", {}),
                        "meaning_card": ont.card(k.frame, st), "answer": k.answer, "answer_type": k.answer_type,
                        "primitive": k.primitive, "admission": k.admission, "recovery": k.recovery})
    write(out / "policy_questions.csv", pq_rows, ["qid", "category", "question", "meaning_card", "answer",
                                                   "answer_type", "primitive", "admission", "recovery"])
    return {"dir": str(out), "test_cases": len(case_rows), "base_cases": sum(r["case_type"] == "base" for r in case_rows),
            "enriched_cases": sum(r["case_type"] == "enriched" for r in case_rows), "turn_rows": len(turn_rows),
            "policy_questions": len(pq_rows),
            "files": ["test_cases.csv", "test_turns.csv", "test_cases.jsonl", "policy_questions.csv"]}
