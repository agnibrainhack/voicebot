"""Compile a compact policy graph into the symbolic store and the GMS view knowlytix ingests.

The graph is already structured, so there is no table parsing: every fact is a triple with its
source (``values.card_product.PREMIUM_SUPP``, ``rules.R13``, ...), every number goes to the exact
numeric register (ENM, integrity hashed), and the dialog-state layer comes with it.

The one noisy source is the knowledge-base prose (``knowledge``). As in the v1 compiler, its
figures are proposed as candidate facts and reconciled against the register before anything is
accepted: a figure that matches is corroborated, one that conflicts is dropped (the critic
``inject_misread`` demonstrates on a planted misread), one with nothing to check against is
flagged for the GEODE loop.

``render_gms_view`` writes the same compiled facts in the table shape knowlytix's GEODE ingest
reads (one table per value set with numbers, so the register keys are ``<member>/<attribute>``;
one table per relation otherwise), the way knowledge/kx_view.py does for the v1 graph.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..knowledge.ingest import CandidateTriple, Rule, Source, WorkflowEdge, canonical_number
from ..knowledge.store import ENMEntry, GMSStore
from .model import PolicyGraph, attr_relation, graph_kg, rule_reification, to_triples

# relations kept out of the store (wording, not facts the geometry should learn)
_NOT_IN_STORE = {"has_text", "has_unit", "states"}


def _tail(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return canonical_number(str(v))
    return str(v)


def compile_graph(path: str | Path, *, inject_misread: bool = False):
    """The symbolic store for a v2 graph, plus the self-correction log (geode.BuildResult)."""
    from ..knowledge.geode import BuildResult, Correction
    g = PolicyGraph.load(path)
    kg = graph_kg(g)
    triples = to_triples(g)
    accepted: dict[tuple[str, str, str], list[dict]] = {}
    for t in triples:
        if t["relation"] in _NOT_IN_STORE:
            continue
        tr = (t["head"], t["relation"], _tail(t["tail"]))
        accepted.setdefault(tr, []).append({"section": t["source"], "ref": t["id"], "span": t.get("evidence", ""),
                                            "path": "policy_graph"})
    # exact numeric register
    enm: dict[tuple[str, str], ENMEntry] = {}

    def reg(cat: str, key: str, value: Any, unit: str, src: str):
        v = _tail(value)
        e = enm.get((cat, key)) or ENMEntry(cat, key, v, unit, ENMEntry.digest(cat, key, v, unit))
        if e.value != v:
            raise ValueError(f"conflicting register values for {cat}/{key}: {e.value} vs {v}")
        e.sources.append({"section": src, "ref": key, "span": "", "path": "policy_graph"})
        enm[(cat, key)] = e

    for k, lv in g.limits.items():
        reg("limit", k, lv["value"], lv.get("unit", ""), f"limits.{k}")
    from .model import _unit
    for attr, spec in g.numeric_attributes().items():
        for m, v in spec["members"].items():
            reg(attr, m, v, _unit(attr), f"values.{spec['set']}.{m}")
    # knowledge-base prose: candidate figures reconciled against the register
    corrections: list[Correction] = []
    cands = prose_candidates(g)
    if inject_misread:
        attr, spec = next(iter(g.numeric_attributes().items()))
        m, v = max(spec["members"].items(), key=lambda kv: kv[1])
        cands.append(CandidateTriple(m, attr_relation(attr), _tail(v * 10 + 50 if v else 150),
                                     Source("knowledge base", "KB?", f"{m} ... {v * 10 + 50 if v else 150} (OCR misread)",
                                            "prose")))
    for c in cands:
        src = asdict(c.source)
        backbone = _register_value(g, enm, c)
        if backbone is None:
            corrections.append(Correction(1, "flag", c.key, "no exact register value to check against; left to the "
                                          "GEODE loop", src))
            accepted.setdefault(c.key, []).append(src)
        elif backbone == c.tail:
            corrections.append(Correction(1, "corroborate", c.key, "prose figure matches the exact register", src))
            accepted.setdefault(c.key, []).append(src)
        else:
            corrections.append(Correction(1, "drop_conflict", c.key,
                                          f"prose figure {c.tail} conflicts with register value {backbone}", src))
    rules = {}
    for rid, r in g.rules.items():
        subj, pred, obj = rule_reification(g, rid)
        rules[rid] = Rule(rid, r["stance"], subj, pred, obj, r["text"], Source("rules", rid, r["text"], "rule"))
    workflow = [WorkflowEdge(w["wid"], w["src"], w["dst"], w["condition"],
                             Source("workflow", w["wid"], f"{w['src']} -> {w['dst']}: {w['condition']}", "workflow"))
                for w in g.workflow()]
    value_sets = {vs: list(members) for vs, members in g.values.items()}
    aliases = {m: list(spec["aliases"]) for members in g.values.values() for m, spec in members.items()}
    store = GMSStore(title=g.title, triples=list(accepted), provenance=accepted, enm=enm, rules=rules,
                     workflow=workflow, value_sets=value_sets, aliases=aliases, entity_types=kg.entity_types(),
                     kb_sentences=[(f"KB{i}", s) for i, s in enumerate(g.knowledge, start=1)],
                     required_language={"opening": g.opening} if g.opening else {})
    store.kg = kg
    return BuildResult(store=store, corrections=corrections, iterations=1, n_candidates=len(triples) + len(cands),
                       converged=True, max_iterations=1, doc=Path(path).name)


# ---------------------------------------------------------------------------- prose figures
_MONEY = re.compile(r"\$(\d+(?:\.\d+)?)")
_COUNT = re.compile(r"(?<![\d$.])(\d+)\s+([a-z]+(?: [a-z]+)?)")


def prose_candidates(g: PolicyGraph) -> list[CandidateTriple]:
    """Figures the knowledge-base sentences state, bound to the member they name (naive binding:
    the member mentioned last before the figure), as candidate facts for the critics."""
    out = []
    num = g.numeric_attributes()
    names: dict[str, list[str]] = {}
    for spec in num.values():
        for m in spec["members"]:
            names[m] = [m.lower().replace("_", " "), m.split("_")[0].lower()] + [a.lower() for a in g.aliases(m)]
    for i, sent in enumerate(g.knowledge, start=1):
        low = sent.lower()
        src = Source("knowledge base", f"KB{i}", sent, "prose")

        def member_before(pos: int) -> str | None:
            best, best_pos = None, -1
            for m, ns in names.items():
                for n in ns:
                    for mm in re.finditer(r"\b" + re.escape(n) + r"\b", low):
                        if best_pos < mm.start() < pos:
                            best, best_pos = m, mm.start()
            return best

        for mm in _MONEY.finditer(sent):
            m = member_before(mm.start())
            attr = next((a for a, sp in num.items() if a.endswith("_usd") and (m in sp["members"] if m else True)), None)
            if attr:
                out.append(CandidateTriple(m or "unbound", attr_relation(attr), canonical_number(mm.group(1)), src))
        for mm in _COUNT.finditer(low):
            n, unit = mm.group(1), mm.group(2)
            if "day" in unit:
                m = member_before(mm.start())
                attr = next((a for a, sp in num.items() if "days" in a and (m in sp["members"] if m else True)), None)
                if attr:
                    out.append(CandidateTriple(m or "unbound", attr_relation(attr), canonical_number(n), src))
                continue
            for k, lv in g.limits.items():
                u = str(lv.get("unit", ""))
                if u and (unit.startswith(u.rstrip("s")) or unit.split()[-1].startswith(u.rstrip("s"))):
                    out.append(CandidateTriple(g.root, f"has_{k}", canonical_number(n), src))
                    break
                if k.endswith(unit.split()[-1].rstrip("s") + "s") or unit.split()[-1].rstrip("s") in k:
                    out.append(CandidateTriple(g.root, f"has_{k}", canonical_number(n), src))
                    break
    return out


def _register_value(g: PolicyGraph, enm: dict, c: CandidateTriple) -> str | None:
    if c.head == g.root and c.relation.startswith("has_"):
        e = enm.get(("limit", c.relation[4:]))
        return e.value if e else None
    for attr in g.numeric_attributes():
        if c.relation == attr_relation(attr):
            e = enm.get((attr, c.head))
            return e.value if e else None
    return None


# ---------------------------------------------------------------------------- views
def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(title: str, header: list[str], rows: list[list]) -> list[str]:
    if not rows:
        return []
    return [f"## {title}", "", _row(*header), _row(*["---"] * len(header)), *[_row(*r) for r in rows], ""]


def yes_no_words(attr: str) -> tuple[str, str, str]:
    """(column, word for yes, word for no): the GEODE ingest reads yes/no columns as pass/fail
    flags, so a yes/no attribute is written as words."""
    if attr.startswith("requires_"):
        return attr[len("requires_"):], "required", "not_required"
    if attr.startswith("is_"):
        return attr[3:], attr[3:], f"not_{attr[3:]}"
    return attr, attr, f"not_{attr}"


def render_gms_view(store, g: PolicyGraph) -> str:
    """The compiled policy (a GMSStore from compile_graph) as GEODE input."""
    lines = [f"# {store.title} (GMS view)", "",
             "Generated from the policy graph; every row is a compiled fact with source provenance.", ""]
    lim = sorted((e.key, e.value, e.unit) for e in store.enm.values() if e.category == "limit")
    lines += _table("Limits", ["limit", "value"], [[k, v] for k, v, _ in lim])
    lines += _table("Limit units", ["limit", "unit"], [[k, u] for k, _, u in lim if u])
    rendered = set()
    num = g.numeric_attributes()
    for vs in g.values:
        attrs = [a for a, sp in num.items() if sp["set"] == vs]
        if attrs:
            members = sorted({m for a in attrs for m in num[a]["members"]})
            lines += _table(vs.replace("_", " ").capitalize(), [vs, *attrs],
                            [[m, *[store.enm[(a, m)].value for a in attrs]] for m in members])
            rendered |= {attr_relation(a) for a in attrs}
        for a in g.attributes(vs):
            if a in attrs:
                continue
            vals = {m: spec.get(a) for m, spec in g.values[vs].items() if a in spec}
            if vals and all(v in ("yes", "no") for v in vals.values()):
                col, yes, no = yes_no_words(a)
                lines += _table(col.replace("_", " ").capitalize(), [vs, col],
                                [[m, yes if v == "yes" else no] for m, v in vals.items()])
                rendered.add(attr_relation(a))
    lines += _table("Workflow", ["step", "enables"], [[w.src, w.dst] for w in store.workflow])
    lines += _table("Value sets", ["value", "member_of"],
                    [[v, vs] for vs, vals in sorted(store.value_sets.items()) for v in vals])
    lines += _table("Aliases", ["entity", "alias"], [[c, a] for c, als in sorted(store.aliases.items()) for a in als])
    opp = [[a, b] for vs, vals in sorted(store.value_sets.items()) for i, a in enumerate(vals) for b in vals[i + 1:]]
    opp.append(["must", "must_not"])
    lines += _table("Opposites", ["entity", "opposite_of"], opp)
    skip = rendered | {"member_of", "has_alias", "enables", "has_stance", "constrains"}
    limit_rels = {f"has_{k}" for k in g.limits}
    by_rel: dict[str, list[list[str]]] = defaultdict(list)
    for h, r, t in sorted(store.triples):
        if r in skip or h in store.rules:
            continue
        if h == g.root and r in limit_rels:
            continue
        if h == "bot" and any(rr.subject == h and rr.predicate == r for rr in store.rules.values()):
            continue                                   # rule reifications: in the Rules table
        by_rel[r].append([h, t])
    for r, rows in sorted(by_rel.items()):
        col = r[4:] if r.startswith("has_") else r
        lines += _table(f"Relation {col}", ["entity", col], rows)
    lines += _table("Rules", ["rule", "stance", "subject", "predicate", "object"],
                    [[r.rule_id, r.stance, r.subject, r.predicate, r.obj] for r in store.rules.values()])
    return "\n".join(lines) + "\n"


def render_policy_markdown(g: PolicyGraph) -> str:
    """A readable rendering of the graph (the policy as a document)."""
    L = [f"# {g.title}", "", f"Rendered from {g.path.name if g.path else g.id} (sha256 {g.sha256[:16]}).", ""]
    if g.opening:
        L += ["## Opening", "", f'Required opening: "{g.opening}"', ""]
    L += ["## Limits", "", _row("limit", "value", "unit", "text"), _row(*["---"] * 4)]
    L += [_row(k, lv["value"], lv.get("unit", ""), lv.get("text", "")) for k, lv in g.limits.items()] + [""]
    for vs, members in g.values.items():
        attrs = g.attributes(vs)
        L += [f"## {vs}", "", _row(vs, *attrs, "aliases"), _row(*["---"] * (len(attrs) + 2))]
        L += [_row(m, *[spec.get(a, "") for a in attrs], "; ".join(spec["aliases"])) for m, spec in members.items()]
        L.append("")
    L += ["## Tools", "", _row("tool", "role", "stage", "ok", "fail", "text"), _row(*["---"] * 6)]
    L += [_row(t, sp["role"], sp.get("stage", ""), sp["ok"], ";".join(sp.get("fail", [])), sp.get("text", ""))
          for t, sp in g.tools.items()] + [""]
    L += ["## Rules", "", _row("rule", "stance", "text"), _row(*["---"] * 3)]
    L += [_row(rid, r["stance"], r["text"]) for rid, r in g.rules.items()] + [""]
    L += ["## Dialog", ""]
    for e in g.edges():
        L.append(f"- {e['src']} -> {e['dst']} on {e['on']}" + (f" when {e['when']}" if e["when"] != "always" else "")
                 + (f" (transfer: {e['reason']})" if e.get("reason") else ""))
    L += ["", "## Knowledge base", "", *g.knowledge, ""]
    return "\n".join(L)


def attribute_phrasings(g: PolicyGraph) -> dict[str, list[str]]:
    """Per-relation policy-question phrasings for the u-channel contradiction SFT, rendered by
    the pack's realizer from the graph's figures (labelled by construction)."""
    from ..enrich.realizers import Realizer
    from .kb import kb_frames
    R = Realizer()
    by_rel: dict[str, list] = defaultdict(list)
    for f in kb_frames(g):
        rel = "has_value" if f.get("intent") == "limit" else attr_relation(f.get("attr"))
        by_rel[rel].append(f)
    out = {}
    for rel, frs in by_rel.items():
        ph = set()
        for fr in frs:
            for pd in ("none", "light", "deep"):
                for qf in ("interrogative", "declarative"):
                    ph.add(R.realize(fr, {"paraphrase_depth": pd, "question_format": qf}, "u"))
        out[rel] = sorted(ph)
    return out
