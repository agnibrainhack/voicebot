"""The compiled policy store: the symbolic layer every step reads (book Ch. 2).

``GMSStore`` holds what the compiler (knowledge/geode.py: build_store) reads from the policy:

1. the asserted knowledge graph (typed triples with provenance),
2. the Exact Numeric Memory register (ENM): every authoritative number keyed by
   (category, key) and protected by a SHA-256 integrity hash, so a figure is
   *read*, never parsed out of prose,
3. the rules, the tool workflow, value sets, customer synonyms, entity types,
   knowledge-base sentences and the required language.

It carries no geometry. The trained GMS is knowlytix's (knowledge/kx_store.py:
``KXPolicyStore`` subclasses this class and answers every geometric read -
score_triple, cap radii, tension, holonomy, the tuned encoders - from the
knowlytix store). Helpers shared by the calibration code live here:
``negatives_for`` (counterfactual tails) and ``_operating_point`` (a threshold under a
false-accept ceiling).
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .ingest import Rule, WorkflowEdge

FALSE_ACCEPT_CEILING = 0.05
_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


class IntegrityError(RuntimeError):
    """Raised when an ENM value no longer matches its integrity hash."""


@dataclass
class ENMEntry:
    category: str
    key: str
    value: str
    unit: str
    sha256: str
    sources: list[dict] = field(default_factory=list)

    @staticmethod
    def digest(category: str, key: str, value: str, unit: str) -> str:
        return hashlib.sha256(f"{category}|{key}|{value}|{unit}".encode()).hexdigest()

    def verify(self) -> bool:
        return self.sha256 == self.digest(self.category, self.key, self.value, self.unit)

    def as_number(self) -> int | float:
        return int(self.value) if re.fullmatch(r"-?\d+", self.value) else float(self.value)


class GMSStore:
    # ------------------------------------------------------------------ init
    def __init__(self, *, title: str, triples: Iterable[tuple[str, str, str]],
                 provenance: dict[tuple[str, str, str], list[dict]],
                 enm: dict[tuple[str, str], ENMEntry], rules: dict[str, Rule],
                 workflow: list[WorkflowEdge], value_sets: dict[str, list[str]],
                 aliases: dict[str, list[str]], entity_types: dict[str, str],
                 kb_sentences: list[tuple[str, str]], required_language: dict[str, str]):
        self.title = title
        self.triples: set[tuple[str, str, str]] = set(triples)
        self.provenance = provenance
        self.enm = enm
        self.rules = rules
        self.workflow = workflow
        self.value_sets = value_sets
        self.aliases = aliases
        self.entity_types = entity_types
        self.kb_sentences = kb_sentences
        self.required_language = required_language
        self.kg = None                                     # the policy graph (knowledge/policy_kg.py)
        self.calibration: dict[str, Any] = {}
        self._reindex()

    def _reindex(self) -> None:
        self.by_h: dict[str, set] = {}
        self.by_r: dict[str, set] = {}
        self.by_t: dict[str, set] = {}
        for tr in self.triples:
            h, r, t = tr
            self.by_h.setdefault(h, set()).add(tr)
            self.by_r.setdefault(r, set()).add(tr)
            self.by_t.setdefault(t, set()).add(tr)

    # ------------------------------------------------------------- exact reads
    def lookup_enm(self, category: str, key: str) -> int | float:
        """Byte-exact numeric read from the ENM register (integrity-checked)."""
        e = self.enm[(category, key)]
        if not e.verify():
            raise IntegrityError(f"ENM integrity failure for {category}/{key}")
        return e.as_number()

    def query_triples(self, head: str | None = None, relation: str | None = None,
                      tail: str | None = None) -> list[tuple[str, str, str]]:
        pool = self.triples
        if head is not None:
            pool = self.by_h.get(head, set())
        if relation is not None:
            pool = pool & self.by_r.get(relation, set()) if head is not None else self.by_r.get(relation, set())
        out = [tr for tr in pool
               if (head is None or tr[0] == head) and (relation is None or tr[1] == relation)
               and (tail is None or tr[2] == tail)]
        return sorted(out)

    def tails(self, head: str, relation: str) -> list[str]:
        return [t for _, _, t in self.query_triples(head, relation)]

    def relation_range(self, relation: str) -> list[str]:
        return sorted({t for _, _, t in self.by_r.get(relation, set())})

    # ----------------------------------------------------------- calibration
    @staticmethod
    def _operating_point(pos: list[float], neg: list[float], ceiling: float = FALSE_ACCEPT_CEILING,
                         accept_low: bool = True) -> dict[str, float]:
        """Threshold maximising accuracy subject to a false-accept ceiling.

        accept_low=True means a score <= tau is accepted (distances)."""
        if not pos:
            return {"tau": float("nan"), "tar": 0.0, "far": 0.0, "n_pos": 0, "n_neg": len(neg)}
        s = (lambda x: x) if accept_low else (lambda x: -x)
        P, N = [s(x) for x in pos], [s(x) for x in neg]
        cands = sorted(set(P + N + [max(P) + 0.05]))
        best = None
        for c in cands:
            tar = sum(p <= c for p in P) / len(P)
            far = (sum(n <= c for n in N) / len(N)) if N else 0.0
            if far > ceiling:
                continue
            acc = (tar * len(P) + (1 - far) * len(N)) / (len(P) + len(N))
            if best is None or acc > best[0] + 1e-12 or (abs(acc - best[0]) < 1e-12 and c < best[1]):
                best = (acc, c, tar, far)
        if best is None:                       # no threshold meets the ceiling
            c = min(P)
            best = (0.0, c, sum(p <= c for p in P) / len(P),
                    (sum(n <= c for n in N) / len(N)) if N else 0.0)
        # place the threshold midway into the gap above the chosen point when possible
        acc, c, tar, far = best
        above = [n for n in N if n > c]
        if above and tar == 1.0:
            c = (c + min(above)) / 2 if min(above) > max(P) else c
        tau = c if accept_low else -c
        return {"tau": float(tau), "tar": float(tar), "far": float(far),
                "n_pos": len(P), "n_neg": len(N)}

    def negatives_for(self, head: str, relation: str, tail: str, rng: random.Random,
                      n: int = 6) -> list[str]:
        """Counterfactual tails: same-range tails the head does not assert, plus numeric perturbations."""
        asserted = set(self.tails(head, relation))
        pool = [t for t in self.relation_range(relation) if t not in asserted]
        if _NUMERIC.match(tail):
            x = float(tail)
            pert = {x + 1, x - 1, x * 2, x + 10, x + 5, round(x / 2)}
            pool += [f"{p:g}" for p in pert if p >= 0 and f"{p:g}" not in asserted]
        else:
            same_type = [e for e, ty in self.entity_types.items()
                         if ty == self.entity_types.get(tail) and e not in asserted]
            pool += same_type
        pool = sorted(set(pool) - {tail})
        rng.shuffle(pool)
        return pool[:n]

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        return {"entities": len(self.entity_types), "triples": len(self.triples),
                "relations": len(self.by_r), "enm_entries": len(self.enm),
                "rules": len(self.rules), "workflow_edges": len(self.workflow)}
