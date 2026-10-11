"""The canonical vocabulary as the *trained* GMS holds it (gms_canonicals.json).

For every canonical individual the export records what the trained store says about it,
next to what the policy graph (policy/charge_review_kg.json) states:

* value sets  - each canonical value, its aliases, and how the trained GMS binds every
                alias (``fuzzy_match_entity(alias, mode="frame")``: the tuned v encoder on
                knowlytix, the v-space adapter on the built-in store) - agreement with the
                graph is reported per alias; the value's membership score and the tension to
                its disjoint siblings (u-space);
* numbers     - every exact figure (ENM) with its unit, integrity digest and the value the
                trained store returns;
* states      - each dialog state with the meaning cards it accepts, how it reads a bare
                yes/no, its rules (id, stance, text), limits (with values), next tools,
                transitions and escalation triggers;
* acts        - the slots of each meaning card and the value set each slot draws from;
* facts       - every triple the GMS is trained on with its geodesic score, the relation's
                calibrated cut, and whether the fact sits inside it;
* relations   - per relation: the cut and the share of its facts inside it.

Nothing here is hand-written: it is read off the trained store and the graph.
"""
from __future__ import annotations

import time
from typing import Any


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:                                     # noqa: BLE001 - a relation may be uncalibrated
        return default


def export_canonicals(store, kg, backend_info: dict | None = None) -> dict[str, Any]:
    is_kx = hasattr(store, "kx")
    meta = {"schema": "gmsjourney.gms_canonicals/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": "knowlytix" if is_kx else "built-in", "policy_graph_sha256": kg.sha256,
            "policy_document": kg.data["source"]["document"],
            "encoders": getattr(store, "encoders", None) or ({"kind": "hashing (built-in)"} if not is_kx else None),
            "versions": (backend_info or {}).get("versions")}

    def score(h, r, t):
        return _safe(lambda: float(store.score_triple(h, r, str(t))))

    def cut(r):
        return _safe(lambda: float(store.tau(r)))

    # ---------------------------------------------------------------- value sets
    value_sets: dict[str, Any] = {}
    for vs, vals in store.value_sets.items():
        members = []
        for v in vals:
            aliases = []
            for a in store.aliases.get(v, []):
                bound, conf = _safe(lambda a=a: store.fuzzy_match_entity(a, mode="frame"), (None, None))
                aliases.append({"alias": a, "gms_binds_to": bound,
                                "confidence": None if conf is None else round(float(conf), 4),
                                "agrees": bound is not None and str(bound).lower() == v.lower()})
            s, c = score(v, "member_of", vs), cut("member_of")
            members.append({
                "canonical": v, "label": v.replace("_", " ").lower(), "aliases": aliases,
                "alias_agreement": (sum(a["agrees"] for a in aliases) / len(aliases)) if aliases else None,
                "member_of_score": s, "member_of_cut": c,
                "member_of_inside": (s is not None and c is not None and s <= c),
                "tension_to_siblings": {o: _safe(lambda o=o: round(float(store.tension_energy(v, o)), 4))
                                        for o in vals if o != v},
                "accepted_in_states": sorted({t["head"] for t in kg.where("accepts_act")
                                              if any(q.get("values_from") == vs
                                                     for q in kg.slot_schema(t["tail"]).values())}),
            })
        value_sets[vs] = {"members": members, "disjoint": True}
    from .policy_kg import EXTRA_VALUE_SETS
    for vs, vals in EXTRA_VALUE_SETS.items():
        value_sets[vs] = {"members": [{"canonical": v, "label": v.replace("_", " ")} for v in vals],
                          "disjoint": True, "note": "policy-question vocabulary (not trained geometry)"}

    # ---------------------------------------------------------------- numbers
    numbers = []
    for (cat, key), e in sorted(store.enm.items()):
        got = _safe(lambda: store.lookup_enm(cat, key))
        numbers.append({"category": cat, "key": key, "value": e.as_number(), "unit": e.unit,
                        "sha256": e.sha256, "trained_store_value": got,
                        "agree": got is not None and float(got) == float(e.as_number())})

    # ---------------------------------------------------------------- states + acts
    states = {}
    for sid in kg.states:
        y, n = kg.interprets(sid)
        rows = [t for t in kg.state_triples(sid) if t.get("gms")]
        checks = []
        for t in rows:
            s, c = score(t["head"], t["relation"], t["tail"]), cut(t["relation"])
            checks.append(s is not None and c is not None and s <= c)
        states[sid] = {
            "stage": kg.stage(sid), "prompts_customer": sid in kg.prompt_states,
            "accepts": kg.accepted_acts(sid), "reads_yes_as": y, "reads_no_as": n,
            "rules": [{"id": r, "stance": (kg.tails(r, "has_stance") or [None])[0], "text": kg.rule_text(r)}
                      for r in kg.rules_for(sid)],
            "limits": [{"key": k, "value": kg.limit(k)} for k in kg.limits_for(sid)],
            "tools": kg.tools_for(sid),
            "transitions": [{"to": t["dst"], "on": t.get("on"), "when": t.get("when"), "rules": t.get("rules")}
                            for t in kg.transitions(sid)],
            "triggers": kg.triggers_for(sid),
            "gms_facts_inside_cut": (sum(checks) / len(checks)) if checks else None,
        }
    acts = {}
    for t in kg.where("is_a", tail="dialog_act"):
        a = t["head"]
        acts[a] = {"slots": {k: {"values_from": q.get("values_from"), "required": q.get("required"),
                                 "values": store.value_sets.get(q.get("values_from"),
                                                                EXTRA_VALUE_SETS.get(q.get("values_from")))}
                             for k, q in kg.slot_schema(a).items()},
                   "accepted_in_states": [s for s in kg.prompt_states if kg.accepts(s, a)]}

    # ---------------------------------------------------------------- facts + relations
    facts, by_rel = [], {}
    for h, r, t in kg.hrt(gms_only=True):
        if r == "has_value":                       # limits are stored as (policy, has_<limit>, v)
            h, r = "charge_review_policy", f"has_{h}"
        s, c = score(h, r, t), cut(r)
        inside = s is not None and c is not None and s <= c
        facts.append({"head": h, "relation": r, "tail": t, "score": None if s is None else round(s, 4),
                      "cut": None if c is None else round(c, 4), "inside": inside})
        by_rel.setdefault(r, []).append(inside)
    relations = {r: {"cut": None if cut(r) is None else round(cut(r), 4), "facts": len(v),
                     "inside_rate": round(sum(v) / len(v), 4)} for r, v in sorted(by_rel.items())}

    entities = [{"id": n, "type": ty, "label": n.replace("_", " ").lower(), "aliases": store.aliases.get(n, [])}
                for n, ty in sorted(store.entity_types.items())]

    summary = {
        "entities": len(entities), "value_sets": len(value_sets), "numbers": len(numbers),
        "numbers_agree": all(x["agree"] for x in numbers), "states": len(states), "acts": len(acts),
        "facts": len(facts), "facts_inside_cut": round(sum(f["inside"] for f in facts) / max(1, len(facts)), 4),
        "alias_agreement": round(
            sum(a["agrees"] for vs in value_sets.values() for m in vs["members"] for a in m.get("aliases", []))
            / max(1, sum(len(m.get("aliases", [])) for vs in value_sets.values() for m in vs["members"])), 4),
    }
    return {"meta": meta, "summary": summary, "value_sets": value_sets, "numbers": numbers, "states": states,
            "acts": acts, "relations": relations, "facts": facts, "entities": entities}
