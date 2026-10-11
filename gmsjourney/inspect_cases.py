"""Look inside a base (seed) case and any of its enriched variants, turn by turn.

* ``compare_case``  one row per customer turn: the dialog state the turn answers, the seed
  text and the enriched text, the seed's canonical frame and the canonical frame read
  back from the enriched text, and whether they are the same meaning.
* ``case_hrt``      the same turns as Head / Relation / Tail triples, for the seed and for
  the enriched variant, so you can see that every relation keeps its tail. Optionally
  adds the store facts that ground each slot value (e.g. expedited_delivery has_fee_usd 15).
* ``pick``          choose a seed case and one of its enriched variants by id or index.
* ``show_case``     prints both views for one (case, variant, turn) selection.

Canonical form = the ontology's canonicalisation of a semantic frame (aliases mapped to
canonical individuals, slots sorted), which is exactly what the invariance proof compares.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd


# ----------------------------------------------------------------------------- selection
def variants_of(scenarios, qid: str) -> list:
    return [s for s in scenarios if s.base.qid == qid]


def pick(bases, scenarios, base: str | int = 0, variant: int = 0):
    """(seed item, enriched scenario) by base qid (e.g. 'J015', 'kb-exact_recall-01') or index, and variant index."""
    item = next((b for b in bases if b.qid == base), None) if isinstance(base, str) else bases[base]
    if item is None:
        raise KeyError(f"no base case {base!r}; ids look like {[b.qid for b in bases[:5]]} ...")
    vs = variants_of(scenarios, item.qid)
    if not vs:
        raise KeyError(f"base {item.qid} has no enriched variants in this design")
    return item, vs[variant % len(vs)]


# ----------------------------------------------------------------------------- canonical
def _canon(ont, frame, state=None) -> str:
    """The state-aware canonical meaning card, as compact JSON."""
    if frame is None:
        return "(unreadable)"
    c = ont.card(frame, state)
    out = {"act": c["act"], "slots": c["slots"]}
    if not c.get("admissible", True):
        out["not_admissible"] = c.get("why_not")
    return json.dumps(out, sort_keys=True)


def compare_case(validator, scenario, turn: int | None = None, proof=None) -> pd.DataFrame:
    """Seed vs enriched, turn by turn, in canonical form."""
    p = proof or validator.prove(scenario)
    ont = validator.ont
    rows = []
    for t in p.turns:
        if turn is not None and t["turn"] != turn:
            continue
        rows.append({"turn": t["turn"], "state": t.get("state"),
                     "seed_text": t["base_text"], "enriched_text": t["enriched_text"],
                     "seed_canonical": _canon(ont, t["frame_obj"], t.get("state")),
                     "enriched_canonical": _canon(ont, t.get("read_obj"), t.get("state")),
                     "same_meaning": t["equivalent"], "surface_cosine": round(t["surface_cosine"], 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- H / R / T
def _values(v) -> list:
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out += _values(x) if not (isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str)) else [f"{x[0]}={x[1]}"]
        return out
    return [v]


def frame_hrt(frame, ont, head: str, state: str | None = None) -> list[tuple[str, str, Any]]:
    """A customer turn as triples: (turn, in_state, state), (turn, has_act, act), (turn, has_<slot>, value).
    The card is canonicalised in its dialog state (a bare yes/no read the way the state reads it)."""
    if frame is None:
        return [(head, "has_act", "(unreadable)")]
    f = ont.canonicalize(frame, state)
    out = [(head, "in_state", state)] if state else []
    out.append((head, "has_act", f.act))
    for k, v in f.slots:
        for x in _values(v):
            out.append((head, f"has_{k}", x))
    return out


def _grounding(store, value, limit: int = 5) -> list[tuple[str, str, Any]]:
    if not isinstance(value, str):
        return []
    facts = sorted(((h, r, t) for h, r, t in getattr(store, "triples", []) if h == value),
                   key=lambda x: (x[1] == "has_alias", x[1], str(x[2])))     # policy facts before aliases
    vs = [(value, "member_of", name) for name, vals in getattr(store, "value_sets", {}).items() if value in vals]
    return (vs + facts)[:limit]


def case_hrt(validator, scenario, turn: int | None = None, grounding: bool = False, proof=None,
             wide: bool = True) -> pd.DataFrame:
    """Head / Relation / Tail for the seed turn and the enriched turn.

    wide=True: one row per (turn, relation) with seed and enriched side by side and a match flag.
    wide=False: long form with a `case` column (seed / enriched) and head, relation, tail."""
    p = proof or validator.prove(scenario)
    ont, store = validator.ont, validator.store
    seed_id, enr_id = scenario.base.qid, scenario.sid
    long_rows, wide_rows = [], []
    for t in p.turns:
        if turn is not None and t["turn"] != turn:
            continue
        i, st = t["turn"], t.get("state")
        seed = frame_hrt(t["frame_obj"], ont, f"{seed_id}/t{i}", st)
        enr = frame_hrt(t.get("read_obj"), ont, f"{enr_id}/t{i}", st)
        for case, trs in (("seed", seed), ("enriched", enr)):
            long_rows += [{"case": case, "turn": i, "state": st, "head": h, "relation": r, "tail": tl}
                          for h, r, tl in trs]
        # side by side: align on relation (multi-valued relations compare as sets)
        rels = list(dict.fromkeys([r for _, r, _ in seed] + [r for _, r, _ in enr]))
        for r in rels:
            a = sorted(str(x) for _, rr, x in seed if rr == r)
            b = sorted(str(x) for _, rr, x in enr if rr == r)
            wide_rows.append({"turn": i, "state": st, "seed_head": f"{seed_id}/t{i}", "relation": r,
                              "seed_tail": ", ".join(a) or "-", "enriched_head": f"{enr_id}/t{i}",
                              "enriched_tail": ", ".join(b) or "-", "match": a == b})
        if grounding and st and getattr(ont, "kg", None) is not None:
            kg = ont.kg                                   # the policy that holds in this state
            for rel, vals in (("in_stage", [kg.stage(st)]), ("applies_rule", kg.rules_for(st)),
                              ("limited_by", [f"{k}={kg.limit(k)}" for k in kg.limits_for(st)]),
                              ("accepts_act", kg.accepted_acts(st))):
                row = {"turn": i, "state": st, "head": st, "relation": rel, "tail": ", ".join(map(str, vals))}
                long_rows.append({"case": "state policy", **row})
                wide_rows.append({"turn": i, "state": st, "seed_head": st, "relation": rel, "seed_tail": row["tail"],
                                  "enriched_head": st, "enriched_tail": row["tail"], "match": "state policy"})
        if grounding:
            for _, r, x in seed:
                if r in ("in_state", "has_act"):
                    continue
                for h, gr, gt in _grounding(store, x):
                    row = {"turn": i, "state": st, "head": h, "relation": gr, "tail": gt}
                    long_rows.append({"case": "store fact", **row})
                    wide_rows.append({"turn": i, "state": st, "seed_head": h, "relation": gr, "seed_tail": gt,
                                      "enriched_head": h, "enriched_tail": gt, "match": "store fact"})
    return pd.DataFrame(wide_rows if wide else long_rows)


# ----------------------------------------------------------------------------- one-call view
def show_case(validator, bases, scenarios, base: str | int = 0, variant: int = 0, turn: int | None = None,
              grounding: bool = True, display=None):
    """Print the seed/enriched comparison and the H/R/T tables for one selection."""
    item, scn = pick(bases, scenarios, base, variant)
    p = validator.prove(scn)
    show = display or (lambda df: print(df.to_string(index=False)))
    print(f"seed case {item.qid}  ->  enriched variant {scn.sid}   (proof accepted: {p.accepted})")
    print("factor levels:", json.dumps(scn.factor_levels))
    print("seed frame signature    ", p.base_signature)
    print("enriched frame signature", p.enriched_signature, "(equal)" if p.base_signature == p.enriched_signature else "(DIFFERENT)")
    print("\n-- canonical comparison --")
    show(compare_case(validator, scn, turn, proof=p))
    print("\n-- Head / Relation / Tail (seed vs enriched) --")
    show(case_hrt(validator, scn, turn, grounding=grounding, proof=p))
    return p
