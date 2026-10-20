"""GEODE stage 0 for the policy graph: drop duplicate and contradicting triples before anything is trained.

A policy graph extracted from a document can carry the same fact twice (an exact copy, a differently written
surface - case, spacing, a number written as text, a customer synonym instead of the canonical value) or two
facts that cannot both hold (a single-valued relation with two different tails: two fees for one delivery
option, two stages for one dialog state, two stances for one rule). This pass finds both and repairs the graph;
every decision is logged with the critic that made it. It runs inside ``PolicyKG.load``, so the compiler, the
GEODE loop and the GMS (built-in or knowlytix) are all trained on the repaired graph.

Duplicates
    Two triples are the same fact when head, relation and tail are equal after normalisation (case, spacing,
    ``_``/``-``, numbers as numbers, a customer synonym resolved to its canonical value through the graph's own
    ``has_alias`` lexicon) and their kind and qualifiers are equal (``leads_to`` edges on different customer acts
    are different facts). One copy is kept: the one whose evidence is found verbatim in its source section,
    then the one written in canonical form, then the earliest extracted (lowest id).

Contradictions
    For a single-valued relation (``FUNCTIONAL``; ``leads_to`` per state + customer act + condition, ``enables``
    per workflow edge) every alternative tail is put through the critics, in order:

    provenance   the evidence quote must appear verbatim in the cited source section;
    grounding    the tail must be a known node of the type the relation takes (an unknown rule id, a dialog
                 state that does not exist, fails);
    support      the evidence must state the tail: a number must appear in the quote (as digits or as a word:
                 "once", "two", ...).

    When exactly one alternative passes, the others are dropped (``drop_contradiction``, with the first critic
    each failed). When several pass - both quotes are real, both tails are valid nodes, and nothing in the
    document says which is right (a dialog state given two stages, the quote copied from the true triple) - or
    none does, the evidence cannot decide: the earliest extracted alternative (lowest id) stays so the relation
    remains single-valued, the others are taken out of the graph as ``quarantine_for_review`` and the group is
    listed under ``flags``. That is a stated default, not a judgement; the report keeps the two apart.

What was tried and is not used, because it is not sound on this graph:
    geometry   a provisional GMS trained on the rest of the graph picks the right tail of a one-against-one
               contradiction 4 times in 8 (knowlytix's own duplicate resolution flags a two-way tie rather than
               deciding it); GEODE's geometric critics still run later, on the repaired graph;
    structure  counting two-hop paths from head to tail (state -calls_tool-> tool -has_stage-> stage) agrees
               with the true stage of a state at most half the time, and dropped a correct triple;
    polarity   reading must / must_not / may off the quote disagrees with 6 of the 33 correct rule stances
               (the policy often states a must_not positively: "describe it as completed only after SUCCESS").
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

# single-valued relations: relation -> the qualifier fields that, with the head, identify one value
FUNCTIONAL: dict[str, tuple[str, ...]] = {
    "has_value": (), "has_unit": (), "has_fee_usd": (), "has_business_days_min": (),
    "has_business_days_max": (), "requires_fee_confirmation": (), "has_stance": (), "in_stage": (),
    "interprets_yes_as": (), "interprets_no_as": (), "prompts_customer": (), "has_stage": (),
    "justified_by": (), "has_condition": (), "success_value": (), "on_failure": (), "when_used": (),
    "leads_to": ("on", "when"), "enables": ("edge",),
}
# tails that are free text or customer wording: compared as written (normalised), never alias-resolved
TEXT_RELATIONS = {"has_alias", "has_text", "states", "has_condition", "when_used"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_WORDNUM = {"zero": 0, "one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _numbers_in(text: str) -> set[str]:
    out = {_num(n) for n in _NUM.findall(text.replace(",", ""))}
    out |= {f"{float(v):g}" for w, v in _WORDNUM.items() if re.search(rf"\b{w}\b", text, re.I)}
    return out


def _norm_text(s: str) -> str:
    return " ".join(str(s).replace("–", "-").replace("—", "-").split())


def _num(x: Any) -> str | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return f"{float(x):g}"
    s = str(x).strip().lstrip("$")
    return f"{float(s):g}" if re.fullmatch(r"-?\d+(?:\.\d+)?", s) else None


def _key(x: Any) -> str:
    n = _num(x)
    return n if n is not None else re.sub(r"[\s_\-]+", "_", str(x).strip()).casefold()


def _id_order(i: str) -> tuple:
    m = re.search(r"\d+", i)
    return (int(m.group()) if m else 10 ** 9, i)


def _canonical_form(x: Any) -> bool:
    return not (isinstance(x, str) and (x != x.strip() or "  " in x))


class _Graph:
    def __init__(self, triples: list[dict], sections: dict[str, str]):
        self.triples = triples
        self.sections = sections
        defined = {_key(t["head"]) for t in triples}
        self.alias: dict[str, str] = {}
        for t in triples:
            if t["relation"] == "has_alias":
                a = _key(t["tail"])
                if a not in defined:                  # the synonym is not itself an entity of the graph
                    self.alias.setdefault(a, _key(t["head"]))

    def node(self, x: Any, relation: str | None = None) -> str:
        k = _key(x)
        if relation in TEXT_RELATIONS:
            return _key(_norm_text(x))
        return self.alias.get(k, k)

    def provenance_ok(self, t: dict) -> tuple[bool, str]:
        sec = self.sections.get(t.get("source"))
        if sec is None:
            return False, f"cites an unknown source section {t.get('source')!r}"
        ev = t.get("evidence")
        if not ev:
            return False, "has no evidence quote"
        from .policy_kg import _norm                 # the same check as PolicyKG.validate
        if _norm(ev) not in sec:
            return False, f"evidence is not found in source section {t['source']}"
        return True, ""


def _qual(t: dict, fields: tuple[str, ...] | None = None) -> str:
    q = t.get("qualifiers") or {}
    if fields is not None:
        q = {k: q.get(k) for k in fields}
    return json.dumps(q, sort_keys=True, default=str)


def _describe(t: dict) -> str:
    return f"({t['head']}, {t['relation']}, {t['tail']!r})"


def clean_graph(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The repaired graph and the cleaning report (every drop and flag, with the critic that decided it)."""
    triples = list(data["triples"])
    G = _Graph(triples, data.get("source", {}).get("sections", {}))
    drops: list[dict] = []
    flags: list[dict] = []

    # ---- 1. duplicates
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in triples:
        k = (G.node(t["head"]), t["relation"], G.node(t["tail"], t["relation"]), t.get("kind"), _qual(t))
        groups[k].append(t)
    dropped: set[str] = set()
    for k, ts in groups.items():
        if len(ts) < 2:
            continue
        rank = sorted(ts, key=lambda t: (not G.provenance_ok(t)[0],
                                         not (_canonical_form(t["head"]) and _canonical_form(t["tail"])),
                                         G.node(t["tail"], t["relation"]) != _key(t["tail"]),
                                         isinstance(t["tail"], str) and _num(t["tail"]) is not None,
                                         t["id"]))
        keep = rank[0]
        for t in rank[1:]:
            how = ("exact copy" if {**t, "id": keep["id"]} == keep else
                   "same fact through a customer synonym" if G.node(t["tail"], t["relation"]) != _key(t["tail"]) else
                   "same fact, number written as text" if isinstance(t["tail"], str) and _num(t["tail"]) is not None
                   and not isinstance(keep["tail"], str) else
                   "same fact, different source / evidence" if (t.get("source"), t.get("evidence")) !=
                   (keep.get("source"), keep.get("evidence")) and _key(t["tail"]) == _key(keep["tail"])
                   and str(t["head"]) == str(keep["head"]) else
                   "same fact written differently (case / spacing)")
            drops.append({"id": t["id"], "triple": _describe(t), "action": "drop_duplicate", "critic": "duplicate",
                          "kept_id": keep["id"], "reason": f"{how} of {keep['id']} {_describe(keep)}",
                          "source": t.get("source"), "evidence": t.get("evidence")})
            dropped.add(t["id"])
    survivors = [t for t in triples if t["id"] not in dropped]

    # ---- 2. contradictions in single-valued relations
    types: dict[str, str] = {}
    for t in survivors:
        if t["relation"] == "is_a":
            types.setdefault(G.node(t["head"]), _key(t["tail"]))
        elif t["relation"] == "in_stage":
            types.setdefault(G.node(t["tail"]), "stage")
        elif t["relation"] == "member_of":
            types.setdefault(G.node(t["head"]), "value:" + _key(t["tail"]))
    cgroups: dict[tuple, list[dict]] = defaultdict(list)
    for t in survivors:
        if t["relation"] in FUNCTIONAL:
            cgroups[(G.node(t["head"]), t["relation"], _qual(t, FUNCTIONAL[t["relation"]]))].append(t)
    for (h, rel, _q), ts in cgroups.items():
        tails = {G.node(t["tail"], rel) for t in ts}
        if len(tails) < 2:
            continue
        group_ids = {t["id"] for t in ts}
        others = [t for t in survivors if t["relation"] == rel and t["id"] not in group_ids]
        range_types = {types.get(G.node(t["tail"], rel)) for t in others} - {None}
        verdicts = []
        for t in ts:
            fails = []
            ok, why = G.provenance_ok(t)
            if not ok:
                fails.append(("provenance", why))
            tail = G.node(t["tail"], rel)
            if rel not in TEXT_RELATIONS and _num(t["tail"]) is None:
                elsewhere = any(G.node(o["head"]) == tail or G.node(o["tail"], o["relation"]) == tail
                                for o in survivors if o["id"] not in group_ids)
                if not elsewhere:
                    fails.append(("grounding", f"{t['tail']!r} is not a node anywhere else in the graph"))
                elif range_types and types.get(tail) not in range_types:
                    fails.append(("grounding", f"{t['tail']!r} is a {types.get(tail) or 'untyped node'}, "
                                               f"{rel} takes {' / '.join(sorted(range_types))}"))
            ev = t.get("evidence") or ""
            if _num(t["tail"]) is not None and ev:
                if _num(t["tail"]) not in _numbers_in(ev):
                    fails.append(("support", f"the evidence quote does not state {t['tail']}: “{ev}”"))
            verdicts.append({"t": t, "fails": fails})
        alive = [v for v in verdicts if not v["fails"]]
        decided = len(alive) == 1
        pool = alive if alive else verdicts
        keep = alive[0]["t"] if decided else min((v["t"] for v in pool), key=lambda t: _id_order(t["id"]))
        if not decided:
            flags.append({"group": f"({ts[0]['head']}, {rel}, ?)" + (f" where {json.loads(_q)}" if FUNCTIONAL[rel] else ""),
                          "ids": [v["t"]["id"] for v in verdicts], "tails": [v["t"]["tail"] for v in verdicts],
                          "kept_id": keep["id"],
                          "reason": ("no alternative passes the critics" if not alive else
                                     f"{len(alive)} alternatives pass every critic: the quotes are real and the "
                                     f"tails are valid, the document does not say which is right"),
                          "critics": {v["t"]["id"]: [f"{c}: {w}" for c, w in v["fails"]] or ["passes"]
                                      for v in verdicts}})
        for v in verdicts:
            if v["t"] is keep:
                continue
            if v["fails"]:
                critic, why = v["fails"][0]
                action = "drop_contradiction" if decided or not alive or v not in alive else "quarantine_for_review"
            else:
                critic, why = "undecided", (f"passes every critic, as does {keep['id']}; the earliest extracted "
                                            f"alternative is kept until reviewed")
                action = "quarantine_for_review"
            drops.append({"id": v["t"]["id"], "triple": _describe(v["t"]), "action": action,
                          "critic": critic, "kept_id": keep["id"],
                          "reason": f"contradicts {keep['id']} {_describe(keep)}; {why}",
                          "all_critics_failed": [c for c, _ in v["fails"]],
                          "source": v["t"].get("source"), "evidence": v["t"].get("evidence")})
            dropped.add(v["t"]["id"])

    out = {k: v for k, v in data.items() if k != "sha256"}
    out["triples"] = [t for t in triples if t["id"] not in dropped]
    if drops or flags:
        body = json.dumps(out, sort_keys=True, default=str).encode()
        out["sha256"] = hashlib.sha256(body).hexdigest()
    elif "sha256" in data:
        out["sha256"] = data["sha256"]
    by = defaultdict(int)
    for d in drops:
        by[d["action"]] += 1
    crit = defaultdict(int)
    for d in drops:
        crit[d["critic"]] += 1
    report = {"triples_in": len(triples), "triples_out": len(out["triples"]), "dropped": len(drops),
              "by_action": dict(by), "by_critic": dict(crit), "flagged_groups": len(flags),
              "drops": drops, "flags": flags}
    return out, report


def score_against_answers(report: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """How the cleaning did against a planted-error answer key (tools/make_noisy_kg.py)."""
    planted = {p["id"]: p for p in answers["planted"]}
    dropped = {d["id"]: d for d in report["drops"]}
    caught = [i for i in planted if i in dropped and dropped[i]["action"] == planted[i]["expected_action"]]
    wrong_action = [i for i in planted if i in dropped and i not in caught]
    missed = [i for i in planted if i not in dropped]
    false_drops = [i for i in dropped if i not in planted]
    kept_right = [i for i in caught if dropped[i]["kept_id"] == planted[i]["original_id"]]
    return {"planted": len(planted), "caught": len(caught), "kept_the_original": len(kept_right),
            "missed": missed, "wrong_action": wrong_action, "false_drops": false_drops,
            "precision": round(len(caught) / max(1, len(dropped)), 3),
            "recall": round(len(caught) / max(1, len(planted)), 3)}
