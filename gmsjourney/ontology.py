"""Journey ontology and semantic frames.

The ontology is the contract that makes "base case" and "enriched case" provably
the same test. It has three parts:

1. Domain individuals read from the store: dispositions, card actions, delivery
   options, tools, rules, limits, escalation triggers. Members of one value set
   are pairwise disjoint (owl:AllDifferent), and their aliases are
   skos:altLabel of a single canonical individual.
2. Dialog acts: the closed set of things a customer turn can *mean* in this
   journey (provide a credential, report a disposition, request a card action,
   confirm, deny, ask a policy question, ask for a human, end, or be unclear).
3. SemanticFrame: an act plus slots whose values are canonical individuals.
   The ground truth of a journey is a function only of its frame sequence and
   the backend fixture (oracle.py). If two utterances map to the same frame,
   they are the same test input, whatever their surface form.

``frame_distance`` gives a graded ontological distance between frames (0 means
identical meaning), using u-space tension so that swapping a value for a
disjoint sibling (RECOGNIZED -> UNRECOGNIZED) scores as a contradiction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

CV = "https://example.org/chargereview#"

# ---------------------------------------------------------------- dialog acts
USER_ACTS = {
    "PROVIDE_CREDENTIAL": "Customer supplies the authentication secret (card last four).",
    "INFORM_DISPOSITION": "Customer states whether the presented charge is theirs.",
    "REQUEST_CARD_ACTION": "Customer asks to secure or replace the card / chooses delivery.",
    "CONFIRM": "Customer clearly agrees to the proposed action.",
    "DENY": "Customer clearly declines the proposed action.",
    "ASK_KB": "Customer asks a general policy question.",
    "REQUEST_HUMAN": "Customer asks to speak to a human representative.",
    "END": "Customer has no further needs.",
    "UNCLEAR": "Turn carries no recoverable decision (content-level ambiguity).",
    "CHITCHAT": "Off-task talk with no journey meaning.",
}

BOT_ACTS = [
    "ASK_AUTH", "ASK_AUTH_RETRY", "PRESENT_TXN", "PROVIDE_CONTEXT", "ASK_CLARIFY",
    "ANSWER_KB", "CONFIRM_RECORDED", "OFFER_CARD_OPTIONS", "INFORM_NO_CARD_ACTION",
    "CONFIRM_ACTION", "DISCLOSE_FEE", "ACTION_DONE", "ASK_MORE_HELP", "EXPLAIN_NO_TXNS",
    "EXPLAIN_LIMIT", "SUMMARIZE", "TRANSFER", "END",
]

# KB intents: the meaning of a policy question. Each maps to a base category of
# the book's taxonomy (bases/kb.py) and to a ground-truth computation over the store.
KB_INTENTS = {
    "value": ("attr", "entity"),
    "limit": ("key",),
    "plausibility": ("attr", "entity", "value"),
    "mcq": ("attr", "entity", "options"),
    "faster": (),
    "cheaper": (),
    "fee_difference": (),
    "expedited_total": (),
    "count_options": (),
    "needs_confirmation": (),
    "faster_option_fee": (),
    "refund": (),
    "fraud": (),
    "window_check": ("days",),
    "skip_auth": (),
    "procedure": (),
}


# ---------------------------------------------------------------- frames
@dataclass(frozen=True)
class SemanticFrame:
    act: str
    slots: tuple[tuple[str, Any], ...] = ()

    @staticmethod
    def make(act: str, **slots: Any) -> "SemanticFrame":
        clean = {k: v for k, v in slots.items() if v is not None}
        return SemanticFrame(act, tuple(sorted((k, _freeze(v)) for k, v in clean.items())))

    def get(self, key: str, default: Any = None) -> Any:
        return dict(self.slots).get(key, default)

    def canonical(self) -> str:
        return json.dumps({"act": self.act, "slots": [[k, _thaw(v)] for k, v in self.slots]},
                          sort_keys=True)

    def signature(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={_thaw(v)}" for k, v in self.slots)
        return f"{self.act}({inner})"


def _freeze(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    return v


def _thaw(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_thaw(x) for x in v]
    return v


def sequence_signature(frames: Iterable[SemanticFrame]) -> str:
    """Ontological signature of a whole conversation (order-sensitive)."""
    h = hashlib.sha256()
    for f in frames:
        h.update(f.canonical().encode())
        h.update(b"\x1e")
    return h.hexdigest()[:20]


# ---------------------------------------------------------------- ontology
@dataclass
class Ontology:
    store: Any
    individuals: dict[str, str] = field(default_factory=dict)   # name -> class
    alias_to_canon: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_store(cls, store) -> "Ontology":
        ont = cls(store=store)
        cls_map = {"disposition": "Disposition", "card_action": "CardAction",
                   "delivery": "DeliveryOption", "tool_result": "ToolResult",
                   "tool": "Tool", "rule": "PolicyRule", "limit": "Limit",
                   "escalation_trigger": "EscalationTrigger", "delivery_option": "DeliveryOption",
                   "stage": "WorkflowStage", "policy": "Policy", "state": "WorkflowState",
                   "concept": "DomainConcept"}
        for name, ty in store.entity_types.items():
            ont.individuals[name] = cls_map.get(ty, "Thing")
        for canon, als in store.aliases.items():
            for a in als:
                ont.alias_to_canon[a.lower()] = canon
            ont.alias_to_canon[canon.lower()] = canon
            ont.alias_to_canon[canon.replace("_", " ").lower()] = canon
        return ont

    def canon(self, value: str) -> str:
        return self.alias_to_canon.get(str(value).lower(), value)

    def disjoint(self, a: str, b: str) -> bool:
        for vals in self.store.value_sets.values():
            if a in vals and b in vals and a != b:
                return True
        return False

    def canonicalize(self, frame: SemanticFrame) -> SemanticFrame:
        slots = {}
        for k, v in frame.slots:
            slots[k] = self.canon(v) if isinstance(v, str) else v
        return SemanticFrame.make(frame.act, **{k: _thaw(v) for k, v in slots.items()})

    def equivalent(self, a: SemanticFrame, b: SemanticFrame) -> bool:
        return self.canonicalize(a) == self.canonicalize(b)

    def frame_distance(self, a: SemanticFrame, b: SemanticFrame) -> float:
        """0 = same meaning; 0.5 = slot missing/extra; ~0.9-1.0 = contradiction or act change."""
        a, b = self.canonicalize(a), self.canonicalize(b)
        if a == b:
            return 0.0
        if a.act != b.act:
            return 1.0
        da, db = dict(a.slots), dict(b.slots)
        worst = 0.0
        for k in set(da) | set(db):
            if k not in da or k not in db:
                worst = max(worst, 0.5)
            elif da[k] != db[k]:
                va, vb = da[k], db[k]
                if isinstance(va, str) and isinstance(vb, str) and self.disjoint(va, vb):
                    worst = max(worst, min(1.0, self.store.tension_energy(va, vb) / 2))
                else:
                    worst = max(worst, 0.75)
        return worst

    # ------------------------------------------------------------- export
    def to_turtle(self, extra_triples: Iterable[tuple[str, str, str]] = ()) -> str:
        def iri(x: str) -> str:
            safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(x))
            return f"cv:{safe}"

        def lit(x: str) -> str:
            return json.dumps(str(x))

        lines = [
            f"@prefix cv: <{CV}> .", "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", "",
            "# ---- classes",
        ]
        classes = sorted(set(self.individuals.values()) | {"SemanticFrame", "DialogAct", "KBIntent",
                                                           "BaseCase", "EnrichedCase"})
        for c in classes:
            lines.append(f"cv:{c} a owl:Class .")
        lines += ["cv:EnrichedCase rdfs:subClassOf cv:BaseCase .", "",
                  "# ---- object properties",
                  "cv:hasAct a owl:ObjectProperty ; rdfs:domain cv:SemanticFrame ; rdfs:range cv:DialogAct .",
                  "cv:derivedFrom a owl:ObjectProperty ; rdfs:domain cv:EnrichedCase ; rdfs:range cv:BaseCase .",
                  "cv:frameSignature a owl:DatatypeProperty .",
                  "cv:groundTruthHash a owl:DatatypeProperty .", "",
                  "# ---- dialog acts"]
        for act, desc in USER_ACTS.items():
            lines.append(f"cv:{act} a cv:DialogAct ; rdfs:comment {lit(desc)} .")
        for intent in KB_INTENTS:
            lines.append(f"cv:kb_{intent} a cv:KBIntent .")
        lines += ["", "# ---- individuals"]
        for name, c in sorted(self.individuals.items()):
            lines.append(f"{iri(name)} a cv:{c} ; skos:prefLabel {lit(name.replace('_', ' '))} .")
            for a in self.store.aliases.get(name, []):
                lines.append(f"{iri(name)} skos:altLabel {lit(a)} .")
        lines += ["", "# ---- disjoint value sets"]
        for vs, vals in sorted(self.store.value_sets.items()):
            members = " ".join(iri(v) for v in vals)
            lines.append(f"[] a owl:AllDifferent ; owl:distinctMembers ( {members} ) .  # {vs}")
        lines += ["", "# ---- exact numeric memory"]
        for (cat, key), e in sorted(self.store.enm.items()):
            lines.append(f"{iri(key)} cv:{cat} {lit(e.value)}^^xsd:decimal .  # sha256 {e.sha256[:12]}")
        lines += ["", "# ---- policy rules"]
        for rid, r in sorted(self.store.rules.items()):
            lines.append(f"{iri(rid)} a cv:PolicyRule ; cv:stance {lit(r.stance)} ; "
                         f"cv:subject {iri(r.subject)} ; cv:predicate {lit(r.predicate)} ; "
                         f"cv:object {iri(r.obj)} ; rdfs:comment {lit(r.text)} .")
        lines += ["", "# ---- asserted graph"]
        for h, r, t in sorted(self.store.triples):
            if r in ("has_alias",):
                continue
            obj = lit(t) if t[:1].isdigit() or t in ("yes", "no") else iri(t)
            lines.append(f"{iri(h)} cv:{r} {obj} .")
        extra = list(extra_triples)
        if extra:
            lines += ["", "# ---- invariance evidence"]
            for s, p, o in extra:
                lines.append(f"{s} {p} {o} .")
        return "\n".join(lines) + "\n"
