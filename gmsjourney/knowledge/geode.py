"""GEODE-style build: ingest -> self-correction -> train -> calibrate (book Ch. 4).

The loop never trusts a candidate fact on the strength of having been extracted.

propose   every candidate from ingest (table backbone + naive prose).
diagnose  symbolic critics
            - type check against the relation schema (domain/range),
            - numeric reconciliation of every prose candidate against the ENM
              backbone (corroborate / conflict),
            - unbound subjects (prose that never named an entity),
            - dangling workflow endpoints and rules that reference unknown limits,
          and a geometric critic: after a provisional training pass, any fact
          that is not corroborated and sits outside its relation's calibrated
          band is flagged as implausible.
repair    corroborated prose is merged into the backbone fact (its span added as
          provenance); conflicting or unbound prose is dropped; implausible
          uncorroborated facts are dropped. Every decision is logged.
iterate   until a pass makes no change, then train and calibrate the final store.

The result is an auditable correction log plus a store whose own generated
questions it must answer perfectly (the precondition of Ch. 4.3, checked in
bases/kb.py).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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

# Which ENM register entry backs a numeric relation, keyed by head type.
ENM_FOR_REL = {
    "has_fee_usd": "delivery_fee_usd",
    "has_business_days_min": "delivery_days_min",
    "has_business_days_max": "delivery_days_max",
}


@dataclass
class Correction:
    iteration: int
    action: str            # corroborate | drop_conflict | drop_unbound | drop_type | drop_implausible | flag
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
    training: dict[str, Any] = field(default_factory=dict)       # store / embed loop / u-channel reports

    def report(self) -> dict[str, Any]:
        """geode_build_report.json (same fields as the book's build report, plus training)."""
        st = self.store
        facts = json.dumps(sorted(list(t) for t in st.triples)).encode()
        return {"doc": self.doc, "converged": self.converged, "iterations": self.iterations,
                "max_iters": self.max_iterations, "n_candidates": self.n_candidates,
                "n_entities": self.n_entities, "n_triples": self.n_triples, "n_enm": self.n_enm,
                "n_corrections": len(self.corrections), "corrections_by_action": self.correction_counts(),
                "triples_sha256": hashlib.sha256(facts).hexdigest(),
                "enm_integrity_ok": all(e.verify() for e in st.enm.values()),
                "training": self.training,
                "oracle_property": oracle_property(st)}

    @property
    def n_entities(self) -> int:
        return len(self.store.entity_types)

    @property
    def n_triples(self) -> int:
        return len(self.store.triples)

    @property
    def n_enm(self) -> int:
        return len(self.store.enm)

    def correction_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.corrections:
            out[c.action] = out.get(c.action, 0) + 1
        return out


def _is_number(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", s))


def _enm_value(ing: IngestResult, cand: CandidateTriple) -> str | None:
    """The authoritative backbone value for a numeric candidate, if any."""
    h, r = cand.head, cand.relation
    if r in ENM_FOR_REL:
        for e in ing.enm:
            if e.category == ENM_FOR_REL[r] and e.key == h:
                return e.value
    if r.startswith("has_") and h == "charge_review_policy":
        key = r[len("has_"):]
        for e in ing.enm:
            if e.category == "limit" and e.key == key:
                return e.value
    return None


def build_store(policy_path: str | Path, *, inject_misread: bool = False,
                max_iterations: int = 5, v_encoder: Any = None, u_encoder: Any = None,
                train: bool = True) -> BuildResult:
    """Build, self-correct, train and calibrate the store from the policy markdown.

    inject_misread adds an OCR-style misread candidate (expedited fee "150") so the
    critics can be seen catching it; it never reaches the final store.
    train runs GMS training (knowledge/training.py): knowledge-graph training of the
    v-space, the GEODE embed loop (v-space rotation SFT) and the u-space contradiction
    SFT. train=False gives the warm-start store (encoder embeddings, Procrustes only).
    """
    ing = ingest_markdown(policy_path)
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
                    survivors.append(c)       # nothing to reconcile against; geometric critic decides
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
    nodes = {w.src for w in ing.workflow} | {w.dst for w in ing.workflow}
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

    def make_store(triples: dict) -> GMSStore:
        return GMSStore(
            title=ing.title, triples=list(triples), provenance=triples, enm=enm,
            rules={r.rule_id: r for r in ing.rules}, workflow=ing.workflow,
            value_sets=ing.value_sets, aliases=ing.aliases, entity_types=dict(ing.entity_types),
            kb_sentences=ing.kb_sentences, required_language=ing.required_language,
            v_encoder=v_encoder, u_encoder=u_encoder)

    # --- geometric critic over prose-only facts that had no backbone to check
    if pending:
        provisional = make_store(accepted).train()
        provisional.calibrate()
        for c in pending:
            d = provisional.score_triple(c.head, c.relation, c.tail)
            rel_cal = provisional.calibration["score_triple"].get(c.relation)
            if rel_cal and d > rel_cal["tau"]:
                corrections.append(Correction(it, "drop_implausible", c.key,
                                              f"geodesic {d:.3f} > calibrated tau {rel_cal['tau']:.3f}",
                                              asdict(c.source)))
            else:
                accepted.setdefault(c.key, []).append(asdict(c.source))

    store = make_store(accepted).train()
    store.calibrate()
    res = BuildResult(store=store, corrections=corrections, iterations=it, n_candidates=n_candidates,
                      converged=converged, max_iterations=max_iterations, doc=Path(policy_path).name)
    if train:
        from .training import GeodeEmbedLoop, train_store, train_u_channel
        rep = train_store(store)
        res.training["store"] = rep.as_dict()
        res.training["embed_loop"] = GeodeEmbedLoop(store).run().as_dict()
        from ..bases.kb import _generators          # KB frames the u-channel corpus is rendered from
        frames = [f for fs in _generators(store).values() for f in fs]
        res.training["u_channel"] = train_u_channel(store, frames).as_dict()
    return res


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
    book's absolute bands (true < 0.5, fabricated > 1.0) for comparison."""
    rows = []
    numeric_rel = [r for r in store.by_r if r in ENM_FOR_REL or
                   (r.startswith("has_") and any(h == "charge_review_policy" for h, _, _ in store.by_r[r]))]
    for r in sorted(numeric_rel):
        trs = sorted(store.by_r[r])
        all_vals = sorted({t for _, rr, t in store.triples if rr in numeric_rel})
        for h, _, t in trs:
            same_rel = [x for _, _, x in trs if x != t]                     # another entity's figure
            sibling = [x for hh, rr, x in sorted(store.triples)             # another limit of the same head
                       if hh == h and rr in numeric_rel and rr != r and x != t]
            fake = (same_rel or sibling or [v for v in all_vals if v != t])[0]
            st, sf, tau = store.score_triple(h, r, t), store.score_triple(h, r, fake), store.tau(r)
            rows.append({"head": h, "relation": r, "true": t, "fabricated": fake,
                         "score_true": round(st, 4), "score_fabricated": round(sf, 4), "tau": round(tau, 4),
                         "calibrated_ok": st <= tau < sf, "book_bands_ok": st < 0.5 and sf > 1.0})
    return rows
