"""The policy, rendered in the table shape knowlytix's GEODE ingest reads.

``knowlytix.benchmark.ingest.ingest_markdown`` turns every wide table into typed
triples ``(row_entity, has_<column>, cell)`` and every numeric cell into an
integrity-hashed ENM entry ``(type=<section>, id=<row_entity>/<column>)``. The
source policy (policy/charge_review/charge_review_policy.md) mixes tables with a bulleted
workflow, a head/relation/tail domain table and multi-alias cells, which that
parser reifies or flattens. This module renders the *same compiled facts* as one
table per relation, so GEODE proposes the graph the policy states:

* ``## Limits``            -> (max_turns, has_value, 20.0) + ENM limits/max_turns/value
* ``## Delivery options``  -> (expedited_delivery, has_fee_usd, 15.0) + ENM
* ``## Workflow``          -> (authenticate_customer, has_enables, get_review_transactions)
* ``## Aliases``           -> (recognized, has_alias, mine)  - the GEODE embed loop's supervision
* ``## Opposites``         -> (recognized, has_opposite_of, unrecognized) - disjoint values and
                             opposite stances, which knowlytix's tension calibration reads
* one table per domain relation, the rules (reified), escalation triggers.

Every row is generated from a compiled fact that keeps its source span, so the
view is a deterministic function of the policy document: one source, one store.
Boolean yes/no columns are written as words (``required``/``not_required``)
because the parser treats yes/no columns as pass/fail flags.
"""
from __future__ import annotations

from collections import defaultdict


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(title: str, header: list[str], rows: list[list]) -> list[str]:
    if not rows:
        return []
    out = [f"## {title}", "", _row(*header), _row(*["---"] * len(header))]
    out += [_row(*r) for r in rows]
    return out + [""]


# relations the compiled graph carries that are rendered elsewhere (or are rule reifications)
_SKIP = {"has_fee_usd", "has_business_days_min", "has_business_days_max", "requires_fee_confirmation",
         "member_of", "has_alias", "enables", "has_stance", "constrains"}


def render_gms_markdown(store) -> str:
    """Render the compiled policy (a GMSStore built by knowledge.geode) as GEODE input."""
    lines = [f"# {store.title} (GMS view)", "",
             "Generated from the policy document; every row is a compiled fact with source provenance.", ""]
    lim = sorted((e.key, e.value, e.unit) for e in store.enm.values() if e.category == "limit")
    # numeric tables carry no other text column: the parser folds text cells into the ENM key
    lines += _table("Limits", ["limit", "value"], [[k, v] for k, v, _ in lim])
    lines += _table("Limit units", ["limit", "unit"], [[k, u] for k, _, u in lim])
    opts = sorted({k for (c, k) in store.enm if c == "delivery_fee_usd"})
    conf = {h: t for h, r, t in store.triples if r == "requires_fee_confirmation"}
    lines += _table("Delivery options", ["delivery_option", "fee_usd", "business_days_min", "business_days_max"],
                    [[o, store.enm[("delivery_fee_usd", o)].value, store.enm[("delivery_days_min", o)].value,
                      store.enm[("delivery_days_max", o)].value] for o in opts])
    lines += _table("Fee confirmation", ["delivery_option", "fee_confirmation"],
                    [[o, "required" if conf.get(o) == "yes" else "not_required"] for o in opts])
    lines += _table("Workflow", ["step", "enables"], [[w.src, w.dst] for w in store.workflow])
    vs_rows = [[v, vs] for vs, vals in sorted(store.value_sets.items()) for v in vals]
    lines += _table("Value sets", ["value", "member_of"], vs_rows)
    lines += _table("Aliases", ["entity", "alias"],
                    [[c, a] for c, als in sorted(store.aliases.items()) for a in als])
    opp = []
    for vs, vals in sorted(store.value_sets.items()):
        for i, a in enumerate(vals):
            for b in vals[i + 1:]:
                opp.append([a, b])
    opp.append(["must", "must_not"])
    lines += _table("Opposites", ["entity", "opposite_of"], opp)
    by_rel: dict[str, list[list[str]]] = defaultdict(list)
    for h, r, t in sorted(store.triples):
        if r in _SKIP or h.startswith("R") and h[1:].isdigit():
            continue
        if h == "charge_review_policy" and r.startswith("has_") and r[4:] in _LIMIT_KEYS:
            continue                                   # rendered in the Limits table
        by_rel[r].append([h, t])
    for r, rows in sorted(by_rel.items()):
        col = r[4:] if r.startswith("has_") else r
        lines += _table(f"Relation {col}", ["entity", col], rows)
    lines += _table("Rules", ["rule", "stance", "subject", "predicate", "object"],
                    [[r.rule_id, r.stance, r.subject, r.predicate, r.obj] for r in store.rules.values()])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- vocabulary map
def kx_relation(rel: str) -> str:
    """Our compiled relation name -> the relation the knowlytix ingest produces for it."""
    special = {"enables": "has_enables", "requires_fee_confirmation": "has_fee_confirmation",
               "member_of": "has_member_of", "has_alias": "has_alias"}
    if rel in special:
        return special[rel]
    if rel.startswith("has_"):
        return rel
    return f"has_{rel}"


def kx_tail(tail: str) -> str:
    t = str(tail)
    if t in ("yes", "no"):
        return {"yes": "required", "no": "not_required"}[t]
    try:
        return f"{float(t):.1f}" if "." not in t or float(t) == int(float(t)) else str(float(t))
    except ValueError:
        return " ".join(t.split()).lower()


def kx_triple(h: str, r: str, t: str) -> tuple[str, str, str]:
    """Map a fact in our vocabulary onto the knowlytix store's vocabulary.

    Limits are stored as (charge_review_policy, has_<limit>, v) in the compiled
    graph and as (<limit>, has_value, v) in the GMS view."""
    if h == "charge_review_policy" and r.startswith("has_") and r[4:] in _LIMIT_KEYS:
        return r[4:], "has_value", kx_tail(t)
    return " ".join(h.split()).lower(), kx_relation(r), kx_tail(t)


_LIMIT_KEYS = {"max_transactions_per_session", "lookback_days", "max_auth_attempts",
               "max_clarification_retries", "max_unsure_retries", "max_turns"}
