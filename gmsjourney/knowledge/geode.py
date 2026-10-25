"""The policy compiler: ingest -> symbolic self-correction (book Ch. 4, the part before training).

The compiler never trusts a candidate fact on the strength of having been extracted.

propose   every candidate from ingest (table backbone + naive prose).
diagnose  symbolic critics
            - type check against the relation schema (domain/range),
            - numeric reconciliation of every prose candidate against the ENM
              backbone (corroborate / conflict),
            - unbound subjects (prose that never named an entity),
            - dangling workflow endpoints and rules that reference unknown limits.
repair    corroborated prose is merged into the backbone fact (its span added as
          provenance); conflicting or unbound prose is dropped. Every decision is logged.
iterate   until a pass makes no change.

Prose facts with no exact value to check against are kept and logged as ``flag``: the
geometric checks are knowlytix's GEODE loop, which runs on the compiled facts next
(knowledge/kx_store.py: build_kx_store). The result is the symbolic store (rules, workflow,
value sets, the exact-number register, the dialog-state layer) and an auditable correction
log. ``oracle_property`` checks the trained store: true figures inside each relation's cut,
substituted figures outside.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .policy_kg import GMS_STATE_RELATIONS
from .ingest import CandidateTriple, IngestResult, Source, ingest_markdown
from .store import ENMEntry, GMSStore

# Relation schema used by the type critic: relation -> (domain types, range kind)
SCHEMA: dict[str, tuple[set[str], str]] = {
    "has_fee_usd": ({"delivery_option"}, "number"),
    "has_business_days_min": ({"delivery_option"}, "number"),
    "has_business_days_max": ({"delivery_option"}, "number"),
    "requires_fee_confirmation": ({"delivery_option"}, "yesno"),
    "has_max_transactions_per_session": ({"policy"}, "number"),
    "has_lookback_days": ({"policy"}, "number"),
    "has_max_auth_attempts": ({"policy"}, "number"),
    "has_max_clarification_retries": ({"policy"}, "number"),
    "has_max_unsure_retries": ({"policy"}, "number"),
    "has_max_turns": ({"policy"}, "number"),
    "enables": ({"tool", "state"}, "node"),
    "member_of": (set(), "node"),
}

def _enm_for_rel() -> dict[str, str]:
    """Which ENM register entry backs a numeric relation (domain.yaml -> figures.relations)."""
    from ..domain import current
    return current().enm_for_relation()


def __getattr__(name: str):                       # backward-compatible module constant
    if name == "ENM_FOR_REL":
        return _enm_for_rel()
    raise AttributeError(name)


@dataclass
class Correction:
    iteration: int
    action: str            # corroborate | drop_conflict | drop_unbound | drop_type | flag
    fact: tuple[str, str, str]
    reason: str
    source: dict[str, Any]


@dataclass
class BuildResult:
    store: GMSStore
    corrections: list[Correction] = field(default_factory=list)
    iterations: int = 0
    n_candidates: int = 0
    converged: bool = False
    max_iterations: int = 0
    doc: str = ""

    def correction_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.corrections:
            out[c.action] = out.get(c.action, 0) + 1
        return out


def _is_number(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", s))


def _enm_value(ing: IngestResult, cand: CandidateTriple) -> str | None:
    """The authoritative backbone value for a numeric candidate, if any."""
    from ..domain import current
    dom = current()
    h, r = cand.head, cand.relation
    enm_for_rel = dom.enm_for_relation()
    if r in enm_for_rel:
        for e in ing.enm:
            if e.category == enm_for_rel[r] and e.key == h:
                return e.value
    if r.startswith("has_") and h == dom.root:            # a limit stated on the policy root
        key = r[len("has_"):]
        cat = dom.get("figures.limits.enm_category", "limit")
        for e in ing.enm:
            if e.category == cat and e.key == key:
                return e.value
    return None


def build_store(policy_path: str | Path, *, inject_misread: bool = False,
                max_iterations: int = 5) -> BuildResult:
    """Compile the policy into the symbolic store, with the self-correction log.

    inject_misread adds an OCR-style misread candidate (expedited fee "150") so the
    critics can be seen catching it; it never reaches the store.

    policy_path may be the policy graph JSON (policy/charge_review/charge_review_kg.json, the default
    source) or a structured policy markdown. From the JSON, the compiled policy is
    rendered by PolicyKG.to_markdown() and the dialog-state layer (states, accepted
    meaning cards, transitions, per-state rules/limits/tools) is added to the facts;
    the store keeps the graph as ``store.kg``.
    """
    from ..graph.model import is_v2
    if is_v2(policy_path):                       # a compact policy graph (graph/compile.py)
        from ..graph.compile import compile_graph
        return compile_graph(policy_path, inject_misread=inject_misread)
    from .policy_kg import resolve_policy
    source_path = policy_path
    policy_path, kg = resolve_policy(policy_path)
    ing = ingest_markdown(policy_path)
    if kg is not None:
        for name, ty in kg.entity_types().items():
            ing.entity_types.setdefault(name, ty)
    candidates = list(ing.triples)
    if inject_misread:
        candidates.append(CandidateTriple(
            "expedited_delivery", "has_fee_usd", "150",
            Source("knowledge base", "KB3", "Expedited ... for a $150 fee (OCR misread)", "prose")))
    n_candidates = len(candidates)

    corrections: list[Correction] = []
    accepted: dict[tuple[str, str, str], list[dict]] = {}
    pending = candidates
    it = 0
    converged = False
    for it in range(1, max_iterations + 1):
        changed = False
        survivors: list[CandidateTriple] = []
        for c in pending:
            src = asdict(c.source)
            # --- type critic
            if c.relation in SCHEMA:
                dom, rng = SCHEMA[c.relation]
                htype = ing.entity_types.get(c.head)
                if c.head == "unbound":
                    corrections.append(Correction(it, "drop_unbound", c.key,
                                                  "prose sentence names no entity for this figure", src))
                    changed = True
                    continue
                if dom and htype not in dom:
                    corrections.append(Correction(it, "drop_type", c.key,
                                                  f"head type {htype!r} not in domain {sorted(dom)}", src))
                    changed = True
                    continue
                if rng == "number" and not _is_number(c.tail):
                    corrections.append(Correction(it, "drop_type", c.key, "range must be numeric", src))
                    changed = True
                    continue
            # --- numeric reconciliation of prose against the ENM backbone
            if c.source.path == "prose":
                backbone = _enm_value(ing, c)
                if backbone is None:
                    survivors.append(c)       # nothing to reconcile against: flagged, left to GEODE
                    continue
                if backbone == c.tail:
                    corrections.append(Correction(it, "corroborate", c.key,
                                                  "prose figure matches the exact register", src))
                    accepted.setdefault(c.key, []).append(src)
                    changed = True
                else:
                    corrections.append(Correction(it, "drop_conflict", c.key,
                                                  f"prose figure {c.tail} conflicts with register value {backbone}", src))
                    changed = True
                continue
            accepted.setdefault(c.key, []).append(src)
        pending = survivors
        if not changed:
            converged = True
            break

    # --- structural critics on the accepted graph
    for w in ing.workflow:
        for n in (w.src, w.dst):
            if n not in ing.entity_types:
                corrections.append(Correction(it, "flag", (w.src, "enables", w.dst),
                                              f"workflow node {n} has no type", asdict(w.source)))
    limit_keys = {e.key for e in ing.enm if e.category == "limit"}
    for r in ing.rules:
        if r.obj in {"max_transactions_per_session", "lookback_days"} and r.obj not in limit_keys:
            corrections.append(Correction(it, "flag", (r.rule_id, r.predicate, r.obj),
                                          "rule references an unknown limit", asdict(r.source)))

    # --- ENM register (exact numbers, integrity hashed); duplicates must agree
    enm: dict[tuple[str, str], ENMEntry] = {}
    for e in ing.enm:
        k = (e.category, e.key)
        if k in enm and enm[k].value != e.value:
            raise ValueError(f"conflicting ENM values for {k}: {enm[k].value} vs {e.value}")
        entry = enm.get(k) or ENMEntry(e.category, e.key, e.value, e.unit,
                                       ENMEntry.digest(e.category, e.key, e.value, e.unit))
        entry.sources.append(asdict(e.source))
        enm[k] = entry

    # --- prose facts with no exact value to check: kept, left to the GEODE loop's geometric critics
    for c in pending:
        corrections.append(Correction(it, "flag", c.key, "no exact register value to check against; "
                                      "left to the GEODE loop", asdict(c.source)))
        accepted.setdefault(c.key, []).append(asdict(c.source))

    if kg is not None:                     # the dialog-state layer of the policy graph
        for t in kg.triples:
            tr = (t["head"], t["relation"], str(t["tail"]))
            if t.get("gms") and tr[1] in GMS_STATE_RELATIONS and tr not in accepted:
                accepted[tr] = [{"section": t["source"], "ref": t["id"], "span": t.get("evidence", ""),
                                 "path": "policy_kg"}]
    store = GMSStore(
        title=ing.title, triples=list(accepted), provenance=accepted, enm=enm,
        rules={r.rule_id: r for r in ing.rules}, workflow=ing.workflow,
        value_sets=ing.value_sets, aliases=ing.aliases, entity_types=dict(ing.entity_types),
        kb_sentences=ing.kb_sentences, required_language=ing.required_language)
    store.kg = kg
    return BuildResult(store=store, corrections=corrections, iterations=it, n_candidates=n_candidates,
                       converged=converged, max_iterations=max_iterations, doc=Path(source_path).name)


def to_dot(store: GMSStore, relations: set[str] | None = None) -> str:
    """Graphviz rendering of the store (or of selected relations) - the book draws the
    small GEODE store as a knowledge graph; render with `dot -Tsvg`."""
    lines = ["digraph gms {", '  rankdir=LR; node [shape=box, style=rounded, fontsize=10]; edge [fontsize=8];']
    for h, r, t in sorted(store.triples):
        if relations is None or r in relations:
            lines.append(f'  "{h}" -> "{t}" [label="{r}"];')
    lines.append("}")
    return "\n".join(lines)


def oracle_property(store: GMSStore) -> list[dict[str, Any]]:
    """An oracle, not an index (book Ch. 4 capstone): every ENM-backed fact scores inside
    its relation's calibrated band, and a cross-tail substitution - another entity's
    *real* figure placed on the same relation - scores outside it. Also reports the
    book's absolute bands (true < 0.5, fabricated > 1.0) and knowlytix's per-head calibrated
    admissibility (``native_admissible_ok``) for comparison."""
    from ..domain import current
    rows = []
    enm_for_rel, root = _enm_for_rel(), current().root
    numeric_rel = [r for r in store.by_r if r in enm_for_rel or
                   (r.startswith("has_") and any(h == root for h, _, _ in store.by_r[r]))]
    for r in sorted(numeric_rel):
        trs = sorted(store.by_r[r])
        all_vals = sorted({t for _, rr, t in store.triples if rr in numeric_rel})
        for h, _, t in trs:
            same_rel = [x for _, _, x in trs if x != t]                     # another entity's figure
            sibling = [x for hh, rr, x in sorted(store.triples)             # another limit of the same head
                       if hh == h and rr in numeric_rel and rr != r and x != t]
            fake = (same_rel or sibling or [v for v in all_vals if v != t])[0]
            st, sf, tau = store.score_triple(h, r, t), store.score_triple(h, r, fake), store.tau(r)
            row = {"head": h, "relation": r, "true": t, "fabricated": fake,
                   "score_true": round(st, 4), "score_fabricated": round(sf, 4), "tau": round(tau, 4),
                   "calibrated_ok": st <= tau < sf, "book_bands_ok": st < 0.5 and sf > 1.0}
            adm = getattr(store, "admissibility", None)
            if adm is not None:                     # knowlytix's per-head calibrated admissibility, reported
                at, af = adm(h, r, t), adm(h, r, fake)
                row["native_admissible_ok"] = (None if at is None or af is None
                                               else at["admissible"] and not af["admissible"])
            rows.append(row)
    return rows
