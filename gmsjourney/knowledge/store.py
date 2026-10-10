"""The geometric memory store and its primitives (book Ch. 2-3).

A ``GMSStore`` holds three layers:

1. the asserted knowledge graph (typed triples with provenance),
2. the Exact Numeric Memory register (ENM): every authoritative number keyed by
   (category, key) and protected by a SHA-256 integrity hash, so a figure is
   *read*, never parsed out of prose,
3. a trained geometry on the unit hypersphere:
   - v-space: entity embeddings plus one orthogonal operator per relation, fit
     by orthogonal Procrustes. A rotation preserves norms and distances, which
     is the book's "rotation mode" for v-space (Ch. 3.2).
   - u-space: a contradiction-structured logical channel. Mutually exclusive
     values of the same attribute (disjointWith in the ontology) are placed on a
     regular simplex, and stance facts are signed by polarity, so
     tension_energy reads ~0 for agreement, ~sqrt(2) for unrelated items and up
     to 2 for contradiction (Ch. 3.6).

Primitives: lookup_enm, query_triples, score_triple, link_predict,
tension_energy, stance_check, find_contradictions, check_holonomy,
is_path_consistent, fuzzy_match_entity. Every accept/abstain decision reads a
calibrated operating point fit from the store's own facts under a stated
false-accept ceiling (Ch. 3.7). Nothing uses a hand-set default.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .encoders import HashingEncoder, geodesic, normalize_text
from .ingest import Rule, Source, WorkflowEdge

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


def orthogonal_procrustes(H: np.ndarray, T: np.ndarray, w: np.ndarray | None = None,
                          span: np.ndarray | None = None) -> np.ndarray:
    """Proper rotation R maximising sum_i w_i <R h_i, t_i> (rows of H, T), computed in the
    span of the data and equal to the identity on its orthogonal complement.

    Every R of this form is an exact optimum: M = T^T W H has its row and column spaces
    inside S = span(H, T), so only R's action on S enters the objective. Restricting to
    S turns a d x d SVD into an m x m one (m <= 2n), and leaving the complement fixed is
    the minimal rotation. If the optimum on S is a reflection, the sign flip is moved to
    one direction of the complement, which keeps both the optimum and det(R) = +1.
    """
    d = H.shape[1]
    Hw = H * (w[:, None] if w is not None else 1.0)
    if span is None:                     # distinct head/tail vectors (repeated rows add nothing)
        span = np.concatenate([np.unique(H, axis=0), np.unique(T, axis=0)], axis=0)
    B, _ = np.linalg.qr(span.T)
    Ms = (T @ B).T @ (Hw @ B)
    U, sv, Vt = np.linalg.svd(Ms)
    r = int(np.sum(sv > 1e-10 * max(sv[0], 1e-300)))
    Q = U[:, :r] @ Vt[:r]
    if r < Ms.shape[0]:                  # unconstrained directions inside S: closest-to-identity choice
        U0, V0 = U[:, r:], Vt[r:].T
        Uc, _, Vct = np.linalg.svd(U0.T @ V0)
        Qn = Q + U0 @ (Uc @ Vct) @ V0.T
        if np.linalg.det(Qn) < 0:        # fix the sign inside the free directions (objective unchanged)
            Uc[:, -1] *= -1
            Qn = Q + U0 @ (Uc @ Vct) @ V0.T
        Q = Qn
    m = Q.shape[0]
    R = np.eye(d) + B @ (Q - np.eye(m)) @ B.T
    if np.linalg.det(Q) < 0:
        if m < d:                        # move the reflection to one direction outside S
            j = int(np.argmin(np.sum(B ** 2, axis=1)))
            c = -(B @ B[j])
            c[j] += 1.0
            c /= np.linalg.norm(c)
            R = R - 2.0 * np.outer(c, c)
        else:                            # S is the whole space: flip the weakest direction
            U[:, -1] *= -1
            R = np.eye(d) + B @ (U @ Vt - np.eye(m)) @ B.T
    return R


def _readable(name: str) -> str:
    return name.replace("_", " ")


class GMSStore:
    # ------------------------------------------------------------------ init
    def __init__(self, *, title: str, triples: Iterable[tuple[str, str, str]],
                 provenance: dict[tuple[str, str, str], list[dict]],
                 enm: dict[tuple[str, str], ENMEntry], rules: dict[str, Rule],
                 workflow: list[WorkflowEdge], value_sets: dict[str, list[str]],
                 aliases: dict[str, list[str]], entity_types: dict[str, str],
                 kb_sentences: list[tuple[str, str]], required_language: dict[str, str],
                 v_encoder: Any = None, u_encoder: Any = None):
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
        self.v = v_encoder or HashingEncoder(salt="v")
        self.u = u_encoder or HashingEncoder(salt="u")
        self.R: dict[str, np.ndarray] = {}
        self.calibration: dict[str, Any] = {}
        self._trained_emb: dict[str, np.ndarray] = {}     # set by training.train_store
        self.v_adapter: np.ndarray | None = None          # set by training.finetune_embedding
        self.u_adapter: np.ndarray | None = None          # set by training.finetune_contradiction
        self.u_groups: dict[str, np.ndarray] = {}         # attribute prototypes in the tuned u-space
        self._emb_cache: dict[str, np.ndarray] = {}
        self._u_cache: dict[str, np.ndarray] = {}
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

    def enm_entry(self, category: str, key: str) -> ENMEntry:
        return self.enm[(category, key)]

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

    # ----------------------------------------------------------- embeddings
    def entity_text(self, x: str) -> str:
        if _NUMERIC.match(str(x)):
            return str(x)
        et = self.entity_types.get(x)
        return f"{_readable(x)} {_readable(et)}" if et else _readable(x)

    def emb(self, x: str) -> np.ndarray:
        if x in self._trained_emb:
            return self._trained_emb[x]
        if x not in self._emb_cache:
            self._emb_cache[x] = self.v.encode(self.entity_text(x))
        return self._emb_cache[x]

    def set_entity_embeddings(self, emb: dict[str, np.ndarray]) -> None:
        """Install trained entity embeddings (graph nodes); other strings keep the encoder."""
        self._trained_emb = {k: np.asarray(v, dtype=float) for k, v in emb.items()}

    # ------------------------------------------------------------ training
    def train(self) -> "GMSStore":
        """Fit one orthogonal operator per relation (orthogonal Procrustes)."""
        self.R = {}
        for rel, trs in self.by_r.items():
            H = np.stack([self.emb(h) for h, _, _ in sorted(trs)])
            T = np.stack([self.emb(t) for _, _, t in sorted(trs)])
            nodes = sorted({h for h, _, _ in trs} | {t for _, _, t in trs})
            self.R[rel] = orthogonal_procrustes(H, T, span=np.stack([self.emb(x) for x in nodes]))
        return self

    # ----------------------------------------------------------- plausibility
    def score_triple(self, head: str, relation: str, tail: str) -> float:
        """Geodesic distance of the tail from the relation-mapped head. Lower = more plausible."""
        Rm = self.R.get(relation)
        if Rm is None:
            return math.pi / 2          # no operator: uninformative (orthogonal)
        p = Rm @ self.emb(head)
        p = p / (np.linalg.norm(p) or 1.0)
        return geodesic(p, self.emb(str(tail)))

    def tau(self, relation: str) -> float:
        rel = self.calibration.get("score_triple", {}).get(relation)
        if rel is None:
            raise KeyError(f"relation {relation!r} not calibrated; call calibrate() first")
        return rel["tau"]

    def is_grounded(self, head: str, relation: str, tail: str) -> bool:
        return self.score_triple(head, relation, tail) <= self.tau(relation)

    def link_predict(self, head: str, relation: str, candidates: list[str] | None = None,
                     k: int = 5) -> list[dict[str, Any]]:
        """Asserted tails first (they take precedence), then ranked predictions."""
        asserted = self.tails(head, relation)
        out = [{"tail": t, "distance": self.score_triple(head, relation, t), "asserted": True}
               for t in asserted]
        cands = candidates if candidates is not None else self.relation_range(relation)
        preds = sorted(({"tail": c, "distance": self.score_triple(head, relation, c),
                         "asserted": False} for c in cands if c not in asserted),
                       key=lambda d: d["distance"])
        return out + preds[: max(0, k - len(out))]

    # ------------------------------------------------------------- u-space
    def _orth_basis(self, seed: str, k: int) -> np.ndarray:
        rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
        A = rng.standard_normal((self.u.dim, k))
        Q, _ = np.linalg.qr(A)
        return Q.T  # k x dim, orthonormal rows

    def _value_set_of(self, x: str) -> str | None:
        for vs, vals in self.value_sets.items():
            if x in vals:
                return vs
        return None

    def u_vector(self, x: str) -> np.ndarray:
        """Logical-channel vector. Disjoint values sit on a simplex; rules are polarity-signed."""
        if x in self._u_cache:
            return self._u_cache[x]
        vs = self._value_set_of(x)
        if vs is not None:
            vals = self.value_sets[vs]
            B = self._orth_basis(f"vs:{vs}", len(vals))
            verts = B - B.mean(axis=0, keepdims=True)
            verts /= np.linalg.norm(verts, axis=1, keepdims=True)
            vec = verts[vals.index(x)]
        elif x in self.rules:
            r = self.rules[x]
            vec = self.claim_u(r.subject, r.predicate, r.obj, -1 if r.stance == "must_not" else +1)
        else:
            vec = self.u.encode(self.entity_text(x))
        self._u_cache[x] = vec
        return vec

    def attribute_vector(self, subject: str, predicate: str, obj: str) -> np.ndarray:
        return self.u.encode(f"{_readable(subject)} {_readable(predicate)} {_readable(obj)}")

    def claim_u(self, subject: str, predicate: str, obj: str, polarity: int) -> np.ndarray:
        return float(np.sign(polarity) or 1.0) * self.attribute_vector(subject, predicate, obj)

    def tension_energy(self, a: str | np.ndarray, b: str | np.ndarray) -> float:
        """Chord distance in u-space: 0 agree, ~sqrt(2) unrelated, 2 contradict."""
        ua = a if isinstance(a, np.ndarray) else self.u_vector(a)
        ub = b if isinstance(b, np.ndarray) else self.u_vector(b)
        return float(np.linalg.norm(ua - ub))

    def stance_check(self, subject: str, predicate: str, obj: str,
                     polarity: int) -> dict[str, Any]:
        """Value-polarity verifier (Ch. 11.2): supported / contradicted / uncertain."""
        claim = self.claim_u(subject, predicate, obj, polarity)
        best, best_cos = None, -1.0
        attr = self.attribute_vector(subject, predicate, obj)
        for rid, r in self.rules.items():
            c = float(np.dot(attr, self.attribute_vector(r.subject, r.predicate, r.obj)))
            if c > best_cos:
                best, best_cos = rid, c
        cal = self.calibration.get("tension", {})
        if best is None or best_cos < cal.get("attr_match", 0.8):
            return {"verdict": "uncertain", "energy": None, "rule": None}
        e = self.tension_energy(claim, self.u_vector(best))
        if e <= cal.get("tau_agree", 0.5):
            v = "supported"
        elif e >= cal.get("tau_contra", 1.6):
            v = "contradicted"
        else:
            v = "uncertain"
        return {"verdict": v, "energy": e, "rule": best, "attr_cos": best_cos}

    def find_contradictions(self, rules: dict[str, Rule] | None = None) -> list[dict[str, Any]]:
        """Pairs of rules on the same attribute whose stances conflict (tension above tau)."""
        rules = rules if rules is not None else self.rules
        cal = self.calibration.get("tension", {})
        out = []
        items = sorted(rules.items())
        for i, (ra, a) in enumerate(items):
            for rb, b in items[i + 1:]:
                ca = self.attribute_vector(a.subject, a.predicate, a.obj)
                cb = self.attribute_vector(b.subject, b.predicate, b.obj)
                if float(np.dot(ca, cb)) < cal.get("attr_match", 0.8):
                    continue
                ua = self.claim_u(a.subject, a.predicate, a.obj, -1 if a.stance == "must_not" else 1)
                ub = self.claim_u(b.subject, b.predicate, b.obj, -1 if b.stance == "must_not" else 1)
                e = float(np.linalg.norm(ua - ub))
                if e >= cal.get("tau_contra", 1.6):
                    out.append({"a": ra, "b": rb, "energy": e})
        return out

    # ----------------------------------------------------------- holonomy
    def _path_heads(self, path: list[str]) -> list[str]:
        heads = []
        for h in sorted({h for h, _, _ in self.by_r.get(path[0], set())}):
            frontier = {h}
            ok = True
            for rel in path:
                nxt = {t for x in frontier for t in self.tails(x, rel)}
                if not nxt:
                    ok = False
                    break
                frontier = nxt
            if ok:
                heads.append(h)
        return heads

    def check_holonomy(self, path: list[str], direct: str) -> float:
        """Length-normalised defect between composing `path` and the `direct` relation."""
        if direct not in self.R or any(p not in self.R for p in path):
            return math.pi
        heads = self._path_heads(path)
        if not heads:
            return math.pi
        C = np.eye(self.v.dim)
        for rel in path:
            C = self.R[rel] @ C
        defects = []
        for h in heads:
            a = C @ self.emb(h)
            b = self.R[direct] @ self.emb(h)
            defects.append(geodesic(a / np.linalg.norm(a), b / np.linalg.norm(b)))
        return float(np.mean(defects) / len(path))

    def is_path_consistent(self, path: list[str], direct: str) -> bool:
        return self.check_holonomy(path, direct) <= self.calibration["holonomy"]["tau_path"]

    # ------------------------------------------------------ entity resolution
    def vocabulary(self) -> list[tuple[str, str]]:
        vocab = [(_readable(e), e) for e in self.entity_types]
        for canon, als in self.aliases.items():
            vocab += [(a, canon) for a in als]
        return vocab

    def fuzzy_match_entity(self, surface: str, types: set[str] | None = None,
                           mode: str = "lexicon") -> tuple[str | None, float]:
        """Bind a surface form to a canonical node (abstains below the calibrated threshold).

        mode="lexicon" compares against every name and alias string (an alias table);
        mode="frame" carries the encoded surface into the trained entity frame with the
        v-space adapter (training.finetune_embedding) and reads the nearest entity."""
        if mode == "frame":
            return self._frame_bind(surface, types)
        q = self.v.encode(surface)
        best, best_s = None, -1.0
        for text, canon in self.vocabulary():
            if types and self.entity_types.get(canon) not in types:
                continue
            s = float(np.dot(q, self.v.encode(text)))
            if s > best_s:
                best, best_s = canon, s
        tau = self.calibration.get("fuzzy", {}).get("tau", 0.5)
        return (best, best_s) if best_s >= tau else (None, best_s)

    def bind_candidates(self) -> list[str]:
        return sorted({canon for _, canon in self.vocabulary()})

    def _frame_scores(self, surface: str, types: set[str] | None = None) -> list[tuple[float, str]]:
        R = self.v_adapter if self.v_adapter is not None else np.eye(self.v.dim)
        q = R @ self.v.encode(surface)
        q = q / (np.linalg.norm(q) or 1.0)
        out = []
        for c in self.bind_candidates():
            if types and self.entity_types.get(c) not in types:
                continue
            out.append((float(np.dot(q, self.emb(c))), c))
        return sorted(out, reverse=True)

    def _frame_bind(self, surface: str, types: set[str] | None = None) -> tuple[str | None, float]:
        sc = self._frame_scores(surface, types)
        if not sc:
            return None, -1.0
        s, c = sc[0]
        tau = self.calibration.get("fuzzy_frame", {}).get("tau", -1.0)
        return (c, s) if s >= tau else (None, s)

    # ------------------------------------------------------ u-space attribute routing
    def u_text(self, text: str) -> np.ndarray:
        """Logical-channel vector of free text, through the contradiction-tuned map if present."""
        x = self.u.encode(text)
        if self.u_adapter is not None:
            x = self.u_adapter @ x
            x = x / (np.linalg.norm(x) or 1.0)
        return x

    def attribute_of(self, text: str) -> tuple[str | None, float]:
        """Route a claim to the attribute group it is about (nearest u-space prototype)."""
        if not self.u_groups:
            return None, 0.0
        z = self.u_text(text)
        best = max(self.u_groups.items(), key=lambda kv: float(np.dot(z, kv[1])))
        return best[0], float(np.dot(z, best[1]))

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

    def calibrate(self, seed: int = 7) -> dict[str, Any]:
        rng = random.Random(seed)
        cal: dict[str, Any] = {"false_accept_ceiling": FALSE_ACCEPT_CEILING}
        # -- per-relation score_triple operating points
        per_rel = {}
        for rel, trs in sorted(self.by_r.items()):
            pos = [self.score_triple(h, r, t) for h, r, t in sorted(trs)]
            neg = [self.score_triple(h, r, c) for h, r, t in sorted(trs)
                   for c in self.negatives_for(h, r, t, rng)]
            per_rel[rel] = self._operating_point(pos, neg)
        cal["score_triple"] = per_rel

        # -- fuzzy entity resolution
        from ..enrich.noise import typo  # local import avoids a cycle
        pos, neg = [], []
        vocab = self.vocabulary()
        for text, canon in vocab:
            q_variants = [text, typo(text, random.Random(f"{text}-1")), typo(text, random.Random(f"{text}-2"))]
            for qv in q_variants:
                m, s = self._raw_best(qv)
                pos.append(s if m == canon else -1.0)
        for phrase in _DISTRACTORS:
            neg.append(self._raw_best(phrase)[1])
        cal["fuzzy"] = self._operating_point(pos, neg, accept_low=False)

        # -- holonomy
        pos_paths = [(["precedes", "precedes"], "precedes_two"),
                     (["has_session", "reviews", "governed_by"], "subject_to")]
        unrelated = ["has_fee_usd", "member_of", "has_alias", "has_stage", "is_a", "delivered_by"]
        hp = [self.check_holonomy(p, d) for p, d in pos_paths]
        hn = [self.check_holonomy(p, u) for p, _ in pos_paths for u in unrelated if u in self.R]
        op = self._operating_point(hp, hn)
        cal["holonomy"] = {"tau_path": op["tau"], "positives": hp, "negatives": hn, **op}

        # -- tension energy: agreement, unrelated, contradiction cohorts
        agree, unrel, contra = [], [], []
        for rid, r in self.rules.items():
            pol = -1 if r.stance == "must_not" else 1
            agree.append(self.tension_energy(self.claim_u(r.subject, r.predicate, r.obj, pol), self.u_vector(rid)))
            contra.append(self.tension_energy(self.claim_u(r.subject, r.predicate, r.obj, -pol), self.u_vector(rid)))
        rids = sorted(self.rules)
        for a, b in zip(rids, rids[1:]):
            unrel.append(self.tension_energy(a, b))
        for vs, vals in self.value_sets.items():
            for i, a in enumerate(vals):
                for b in vals[i + 1:]:
                    contra.append(self.tension_energy(a, b))
        attr_self = [float(np.dot(self.attribute_vector(r.subject, r.predicate, r.obj),
                                  self.attribute_vector(r.subject, r.predicate, r.obj))) for r in self.rules.values()]
        attr_cross = [float(np.dot(self.attribute_vector(self.rules[a].subject, self.rules[a].predicate, self.rules[a].obj),
                                   self.attribute_vector(self.rules[b].subject, self.rules[b].predicate, self.rules[b].obj)))
                      for a, b in zip(rids, rids[1:])]
        cal["tension"] = {
            "tau_agree": float((max(agree) + min(unrel)) / 2),
            "tau_contra": float((max(unrel) + min(contra)) / 2),
            "attr_match": float(min(0.8, (max(attr_cross) + min(attr_self)) / 2)),
            "agree_max": float(max(agree)), "unrelated_range": [float(min(unrel)), float(max(unrel))],
            "contra_min": float(min(contra)),
        }
        self.calibration = cal
        return cal

    def _raw_best(self, surface: str) -> tuple[str | None, float]:
        q = self.v.encode(surface)
        best, best_s = None, -1.0
        for text, canon in self.vocabulary():
            s = float(np.dot(q, self.v.encode(text)))
            if s > best_s:
                best, best_s = canon, s
        return best, best_s

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        facts = {
            "title": self.title,
            "triples": sorted(list(t) for t in self.triples),
            "provenance": {"|".join(k): v for k, v in self.provenance.items()},
            "enm": [asdict(e) for e in self.enm.values()],
            "rules": [{**asdict(r), "source": asdict(r.source)} for r in self.rules.values()],
            "workflow": [{**asdict(w), "source": asdict(w.source)} for w in self.workflow],
            "value_sets": self.value_sets, "aliases": self.aliases,
            "entity_types": self.entity_types, "kb_sentences": self.kb_sentences,
            "required_language": self.required_language,
        }
        (p / "facts.json").write_text(json.dumps(facts, indent=1))
        (p / "calibration.json").write_text(json.dumps(self.calibration, indent=1, default=float))
        arrays: dict[str, Any] = {}
        if self._trained_emb:
            names = sorted(self._trained_emb)
            arrays["entity_names"] = np.array(names)
            arrays["entity_matrix"] = np.stack([self._trained_emb[n] for n in names])
        if self.v_adapter is not None:
            arrays["v_adapter"] = self.v_adapter
        if self.u_adapter is not None:
            arrays["u_adapter"] = self.u_adapter
            arrays["u_group_names"] = np.array(sorted(self.u_groups))
            arrays["u_group_matrix"] = np.stack([self.u_groups[g] for g in sorted(self.u_groups)])
        if arrays:
            np.savez_compressed(p / "trained_embeddings.npz", **arrays)
        # Relation operators are not saved: orthogonal Procrustes is deterministic and takes
        # well under a second, so load() re-derives them from the facts (keeps the artifact small).
        return p

    @classmethod
    def load(cls, path: str | Path) -> "GMSStore":
        p = Path(path)
        f = json.loads((p / "facts.json").read_text())
        rules = {}
        for r in f["rules"]:
            src = Source(**r.pop("source"))
            rules[r["rule_id"]] = Rule(**{**r, "source": src})
        wf = []
        for w in f["workflow"]:
            src = Source(**w.pop("source"))
            wf.append(WorkflowEdge(**{**w, "source": src}))
        store = cls(
            title=f["title"], triples=[tuple(t) for t in f["triples"]],
            provenance={tuple(k.split("|")): v for k, v in f["provenance"].items()},
            enm={(e["category"], e["key"]): ENMEntry(**e) for e in f["enm"]},
            rules=rules, workflow=wf, value_sets=f["value_sets"], aliases=f["aliases"],
            entity_types=f["entity_types"], kb_sentences=[tuple(x) for x in f["kb_sentences"]],
            required_language=f["required_language"],
        )
        npz = p / "trained_embeddings.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=False)
            if "entity_names" in z:
                store.set_entity_embeddings(dict(zip([str(n) for n in z["entity_names"]], z["entity_matrix"])))
            if "v_adapter" in z:
                store.v_adapter = z["v_adapter"]
            if "u_adapter" in z:
                store.u_adapter = z["u_adapter"]
                store.u_groups = dict(zip([str(n) for n in z["u_group_names"]], z["u_group_matrix"]))
        store.train()
        store.calibration = json.loads((p / "calibration.json").read_text())
        return store

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        return {"entities": len(self.entity_types), "triples": len(self.triples),
                "relations": len(self.by_r), "enm_entries": len(self.enm),
                "rules": len(self.rules), "workflow_edges": len(self.workflow)}


_DISTRACTORS = [
    "weather tomorrow", "pizza order", "mortgage refinance rate", "my favourite movie",
    "airport parking", "stock price of apple", "gym membership", "birthday party",
    "the football score", "gardening tips", "car insurance quote", "recipe for pasta",
    "hotel booking", "train timetable", "electric bill", "learn spanish",
]
