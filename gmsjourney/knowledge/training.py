"""GMS training on our own core (book Ch. 3-4; beyond-ship-and-pray nb 21-23).

The open companion repository trains its store and two embedding channels through
the licensed ``knowlytix`` package (``build_rag_store(..., epochs)``,
``finetune_embedding(mode='rotation')``, ``finetune_contradiction(mode='full')``).
This module is the open fallback used when knowlytix is not installed: it re-derives each step from the
objective the book states, on top of ``GMSStore``:

1. ``train_store`` - knowledge-graph training of the v-space. Relation operators stay
   orthogonal (the book's rotation mode); entity embeddings, which the plain store
   keeps fixed at their encoder warm start, are refined by block-coordinate descent on

       L = sum_(h,r,t) ||R_r e_h - e_t||^2  +  lam * sum_x ||e_x - e0_x||^2 ,  ||e_x|| = 1

   alternating an exact orthogonal-Procrustes step for every R_r with a closed-form
   step for every e_x (the minimiser of L in e_x with the other blocks fixed). The
   drift term keeps every entity near its warm start, which is what keeps
   numbers, aliases and names readable by the encoder-side primitives. A single
   rotation cannot send one head to several distinct tails; refinement lets the
   tails of a multi-valued relation (workflow successors, escalation triggers) move
   into a common cap, which is why it raises the calibrated true-accept rate.

2. ``finetune_embedding`` (v-space, rotation mode) - an orthogonal adapter that
   carries encoder space into the trained entity frame, fitted by weighted
   orthogonal Procrustes on labelled (surface text -> entity) rows. Canonical names
   are anchor rows (the drift penalty). A rotation preserves every distance between
   encoded texts: it does not separate anything, it aligns coordinates so that a
   surface form can be bound to an entity by reading the store's own frame.
   ``GeodeEmbedLoop`` wraps it in propose -> diagnose -> repair: noisy variants of
   each labelled surface that bind wrongly become new labelled rows until none fail.

3. ``finetune_contradiction`` (u-space, full mode) - a full linear map W on the
   logical channel trained with a supervised contrastive objective and a drift
   penalty ||W - I||^2, so that phrasings of the same attribute (a fee, a duration,
   a window) cluster and different attributes separate. A rotation could not do
   this (it preserves angles), which is why the book requires full mode here.

Every step reports before/after numbers; nothing is assumed to help.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .encoders import geodesic
from .store import orthogonal_procrustes

# ============================================================================ 1
# Store (knowledge-graph) training
# ============================================================================


_procrustes = orthogonal_procrustes


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


@dataclass
class StoreTrainingReport:
    epochs: int
    drift: float
    objective: list[float]
    per_relation: dict[str, dict[str, float]]          # tar/far at the ceiling, before and after
    mean_tar_before: float
    mean_tar_after: float
    drift_geodesic: dict[str, float]                   # how far entities moved from their warm start
    n_nodes: int
    checksum: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _calibrated_tar(store) -> dict[str, dict[str, float]]:
    return {r: {"tar": c["tar"], "far": c["far"], "tau": c["tau"], "n_pos": c["n_pos"], "n_neg": c["n_neg"]}
            for r, c in store.calibration["score_triple"].items()}


def train_store(store, epochs: int = 40, drift: float = 0.75, contrast: float = 0.35) -> StoreTrainingReport:
    """Refine entity embeddings and relation rotations in place, then recalibrate.

    Maximises, by exact block-coordinate ascent,

        J = sum_pos <R_r e_h, e_t> - contrast * sum_neg w <R_r e_h, e_t'> + drift * sum_x <e_x, e0_x>

    over unit e_x and proper rotations R_r. Positives are the asserted triples; the
    negatives of (h, r) are the relation's other tails that h does not assert, weighted
    so that each head's negatives carry the same total weight as its positives. Each
    R_r step is a weighted orthogonal Procrustes problem (negative weights allowed) and
    each e_x step is the normalised gradient - both exact maximisers, so J never
    decreases. contrast=0 is pure fitting; drift is the pull back to the warm start.
    """
    if not store.calibration:
        store.calibrate()
    before = _calibrated_tar(store)
    nodes = sorted({h for h, _, _ in store.triples} | {t for _, _, t in store.triples})
    idx = {x: i for i, x in enumerate(nodes)}
    E0 = np.stack([store.v.encode(store.entity_text(x)) for x in nodes])
    E = E0.copy()
    # weighted pairs per relation: (head, tail, weight)
    pairs: dict[str, list[tuple[int, int, float]]] = {}
    for r, trs in sorted(store.by_r.items()):
        rng_r = sorted({t for _, _, t in trs})
        heads = sorted({h for h, _, _ in trs})
        lst = []
        for h in heads:
            pos = sorted(store.tails(h, r))
            neg = [t for t in rng_r if t not in pos]
            lst += [(idx[h], idx[t], 1.0) for t in pos]
            if neg and contrast > 0:
                w = -contrast * len(pos) / len(neg)
                lst += [(idx[h], idx[t], w) for t in neg]
        pairs[r] = lst
    arr = {r: (np.array([h for h, _, _ in lst]), np.array([t for _, t, _ in lst]),
               np.array([w for _, _, w in lst])) for r, lst in pairs.items()}
    span_idx = {r: np.unique(np.concatenate([hi, ti])) for r, (hi, ti, _) in arr.items()}

    def fit_ops(E):
        return {r: _procrustes(E[hi], E[ti], w, span=E[span_idx[r]]) for r, (hi, ti, w) in arr.items()}

    def objective(E, R):
        j = sum(float(np.sum(w * np.sum((E[hi] @ R[r].T) * E[ti], axis=1))) for r, (hi, ti, w) in arr.items())
        return j + drift * float(np.sum(E * E0))

    R = fit_ops(E)
    obj = [objective(E, R)]
    for _ in range(epochs):
        newE = drift * E0.copy()
        for r, (hi, ti, w) in arr.items():
            np.add.at(newE, ti, w[:, None] * (E[hi] @ R[r].T))     # tail pulled to R e_h (pushed if w < 0)
            np.add.at(newE, hi, w[:, None] * (E[ti] @ R[r]))       # head pulled to R^T e_t
        E = _unit(newE)
        R = fit_ops(E)
        obj.append(objective(E, R))

    store.set_entity_embeddings({x: E[i] for x, i in idx.items()})
    store.train()                       # operators from the installed embeddings (same Procrustes, positives)
    store.calibrate()
    after = _calibrated_tar(store)
    moved = [geodesic(E[i], E0[i]) for i in range(len(nodes))]
    per_rel = {r: {"tar_before": before[r]["tar"], "tar_after": after[r]["tar"],
                   "far_before": before[r]["far"], "far_after": after[r]["far"],
                   "tau_after": after[r]["tau"], "n_pos": after[r]["n_pos"]} for r in after}
    multi = [r for r in per_rel if per_rel[r]["n_pos"] > 1]
    chk = hashlib.sha256(np.round(E, 6).tobytes()).hexdigest()[:16]
    rep = StoreTrainingReport(
        epochs=epochs, drift=drift, objective=[round(o, 4) for o in obj], per_relation=per_rel,
        mean_tar_before=float(np.mean([before[r]["tar"] for r in multi])),
        mean_tar_after=float(np.mean([after[r]["tar"] for r in multi])),
        drift_geodesic={"mean": float(np.mean(moved)), "max": float(np.max(moved)),
                        "median": float(np.median(moved))},
        n_nodes=len(nodes), checksum=chk)
    store.calibration["training"] = {"epochs": epochs, "drift": drift, "contrast": contrast, "checksum": chk,
                                     "objective_first_last": [obj[0], obj[-1]]}
    return rep


DEFAULT_GRID = [(d, c) for d in (0.5, 0.75, 1.0) for c in (0.0, 0.25, 0.35, 0.5)]


def select_training(build_fn, grid=DEFAULT_GRID, epochs: int = 40) -> dict[str, Any]:
    """Model selection for train_store on the store's own operating points.

    For every (drift, contrast) the store is rebuilt with ``build_fn()``, trained and
    recalibrated; a setting is eligible only if the store still passes its structural
    self-tests (both positive holonomy paths consistent, the unrelated one not; the
    rule set consistent and a planted reversal caught; the fraud stance contradicted).
    Among eligible settings the one with the highest mean true-accept rate over
    multi-valued relations (at the 5% false-accept ceiling) wins; ties go to the
    larger drift (the smaller move from the warm start).
    """
    from .ingest import Rule
    rows = []
    for drift, contrast in grid:
        st = build_fn()
        rep = train_store(st, epochs=epochs, drift=drift, contrast=contrast)
        planted = dict(st.rules)
        r13 = st.rules["R13"]
        planted["R13x"] = Rule("R13x", "must", r13.subject, r13.predicate, r13.obj, "reversed copy", r13.source)
        checks = {
            "holonomy_pos": st.is_path_consistent(["precedes", "precedes"], "precedes_two")
            and st.is_path_consistent(["has_session", "reviews", "governed_by"], "subject_to"),
            "holonomy_neg": not st.is_path_consistent(["precedes", "precedes"], "has_fee_usd"),
            "contradiction": st.find_contradictions() == [] and len(st.find_contradictions(planted)) > 0,
            "stance": st.stance_check("unrecognized_transaction", "is_proof_of", "fraud", +1)["verdict"] == "contradicted",
        }
        worst = min(rep.per_relation.values(), key=lambda v: v["tar_after"])
        rows.append({"drift": drift, "contrast": contrast, "mean_tar_multi": round(rep.mean_tar_after, 4),
                     "min_tar": round(worst["tar_after"], 3), "mean_tar_before": round(rep.mean_tar_before, 4),
                     "drift_mean_geodesic": round(rep.drift_geodesic["mean"], 3), "eligible": all(checks.values()),
                     **checks})
    ok = [r for r in rows if r["eligible"]]
    best = max(ok, key=lambda r: (r["mean_tar_multi"], r["drift"])) if ok else None
    return {"grid": rows, "selected": best, "epochs": epochs}


# ============================================================================ 2
# v-space encoder SFT, rotation mode, and the GEODE embed loop
# ============================================================================
@dataclass
class EmbedLoopReport:
    iterations: list[dict[str, Any]]
    converged: bool
    n_rows: int
    eval: dict[str, Any]
    distance_preservation_max_err: float
    leave_one_alias_out: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def finetune_embedding(store, rows: list[tuple[str, str, float]], mode: str = "rotation") -> np.ndarray:
    """Fit the v-space adapter: a proper rotation carrying encoder space into the store's
    trained entity frame (weighted orthogonal Procrustes on (text, entity, weight) rows).
    Only mode='rotation' exists here - the book's v-space mode; a rotation keeps every
    distance between encoded texts, so the adapter aligns frames and separates nothing."""
    if mode != "rotation":
        raise ValueError("the v-space adapter is a rotation (book Ch. 3: rotation mode for v-space)")
    X = np.stack([store.v.encode(r[0]) for r in rows])
    Y = np.stack([store.emb(r[1]) for r in rows])
    w = np.array([r[2] for r in rows], dtype=float)
    R = orthogonal_procrustes(X, Y, w)
    return R.astype(np.float32).astype(float)        # stored as float32; keep the in-memory copy identical


def _noisy(text: str, seed: int) -> str:
    from ..enrich.noise import asr_noise, typo
    rng = random.Random(f"{text}|{seed}")
    return asr_noise(text, rng) if seed % 2 else typo(text, rng, rate=0.2)


def _bind_acc(store, items: list[tuple[str, str]], mode: str) -> float:
    if not items:
        return float("nan")
    ok = 0
    for text, canon in items:
        if mode == "lexicon":
            got = store._raw_best(text)[0]
        else:
            got = store._frame_scores(text)[0][1]
        ok += got == canon
    return ok / len(items)


class GeodeEmbedLoop:
    """propose -> diagnose -> repair for entity binding in the trained frame.

    propose   labelled rows from the store itself: each entity's readable name
              (anchor rows, weight ``anchor``) and every alias the policy lists.
    diagnose  fit the rotation adapter, then bind noisy variants (typo / ASR) of every
              labelled surface; a variant that binds to the wrong entity is a failure.
    repair    failing variants join the rows with their known label; refit.
    Stops when a pass finds no failure or after ``max_iters``. Evaluation uses
    held-out noise seeds the loop never saw.
    """

    def __init__(self, store, anchor: float = 2.0, max_iters: int = 4, probe_seeds=(0, 1, 2, 3),
                 test_seeds=(101, 102, 103, 104, 105, 106)):
        self.store, self.anchor, self.max_iters = store, anchor, max_iters
        self.probe_seeds, self.test_seeds = probe_seeds, test_seeds

    def labelled_surfaces(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        st = self.store
        cands = set(st.bind_candidates())
        names = [(st.entity_text(c), c) for c in sorted(cands) if not c.replace(".", "").isdigit()]
        aliases = [(a, c) for c, als in sorted(st.aliases.items()) if c in cands for a in als]
        return names, aliases

    def run(self) -> EmbedLoopReport:
        st = self.store
        names, aliases = self.labelled_surfaces()
        surfaces = names + aliases
        rows = [(t, c, self.anchor, t) for t, c in names] + [(t, c, 1.0, t) for t, c in aliases]
        probe = [(_noisy(t, s), c, t) for t, c in surfaces for s in self.probe_seeds]
        test = [(_noisy(t, s), c) for t, c in surfaces for s in self.test_seeds]
        seen = {r[0] for r in rows}
        before = {"frame_identity_clean": None, "frame_identity_noisy": None}
        st.v_adapter = None
        before["frame_identity_clean"] = _bind_acc(st, surfaces, "frame")
        before["frame_identity_noisy"] = _bind_acc(st, test, "frame")
        iters, converged = [], False
        for it in range(1, self.max_iters + 1):
            st.v_adapter = finetune_embedding(st, rows)
            fails = [(t, c, o) for t, c, o in probe if t not in seen and st._frame_scores(t)[0][1] != c]
            iters.append({"iteration": it, "rows": len(rows), "probe_acc": 1 - len(fails) / len(probe),
                          "new_rows": len(fails)})
            if not fails:
                converged = True
                break
            for t, c, o in fails:
                rows.append((t, c, 1.0, o))           # origin kept so held-out tests can exclude it
                seen.add(t)
        if not converged:                        # last repair round must be fitted too
            st.v_adapter = finetune_embedding(st, rows)
        # abstention threshold for frame binding, calibrated like the lexicon binder
        pos = [st._frame_scores(t)[0][0] if st._frame_scores(t)[0][1] == c else -1.0 for t, c, _ in probe]
        from .store import _DISTRACTORS
        neg = [st._frame_scores(p)[0][0] for p in _DISTRACTORS]
        st.calibration["fuzzy_frame"] = st._operating_point(pos, neg, accept_low=False)
        ev = {**before,
              "frame_adapter_clean": _bind_acc(st, surfaces, "frame"),
              "frame_adapter_noisy": _bind_acc(st, test, "frame"),
              "lexicon_clean": _bind_acc(st, surfaces, "lexicon"),
              "lexicon_noisy": _bind_acc(st, test, "lexicon"),
              "n_surfaces": len(surfaces), "n_test_variants": len(test),
              "frame_tau": st.calibration["fuzzy_frame"]["tau"],
              "distractor_false_binds": sum(st._frame_bind(p)[0] is not None for p in _DISTRACTORS),
              "n_distractors": len(_DISTRACTORS)}
        # a rotation preserves distances between encoded texts (alignment, not separation)
        rng = np.random.default_rng(0)
        texts = [t for t, _ in surfaces]
        X = np.stack([st.v.encode(t) for t in texts])
        Z = X @ st.v_adapter.T
        i, j = rng.integers(0, len(texts), 200), rng.integers(0, len(texts), 200)
        err = float(np.max(np.abs(np.sum(X[i] * X[j], 1) - np.sum(Z[i] * Z[j], 1))))
        # out-of-vocabulary generalisation: hold each alias out entirely
        hits, n = 0, 0
        R_full = st.v_adapter
        for t, c in aliases:
            st.v_adapter = finetune_embedding(st, [r for r in rows if r[3] != t])   # the alias and its variants
            hits += st._frame_scores(t)[0][1] == c
            n += 1
        st.v_adapter = R_full
        loo = {"n": n, "accuracy": hits / max(1, n),
               "note": "aliases never seen in training; a character-hashing encoder only binds them when they share "
                       "spelling with a known surface - genuinely new synonyms need the alias table or a semantic encoder"}
        return EmbedLoopReport(iters, converged, len(rows), ev, err, loo)


# ============================================================================ 3
# u-space contradiction SFT, full mode
# ============================================================================
@dataclass
class ContradictionReport:
    groups: list[str]
    n_rows: int
    margin_before: float
    margin_after: float
    folds: list[dict[str, Any]]
    routing_acc_before: float
    routing_acc_after: float
    drift_fro: float
    pca_before: list[dict[str, Any]]
    pca_after: list[dict[str, Any]]
    loss: list[float]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _supcon(W: np.ndarray, X: np.ndarray, y: np.ndarray, temp: float, drift: float):
    """Supervised contrastive loss on normalised W x, plus drift ||W - I||^2; returns (loss, grad)."""
    Z = X @ W.T
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    Y = Z / n
    S = (Y @ Y.T) / temp
    np.fill_diagonal(S, -np.inf)
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=1, keepdims=True)
    pos = (y[:, None] == y[None, :]).astype(float)
    np.fill_diagonal(pos, 0.0)
    npos = pos.sum(axis=1, keepdims=True)
    T = np.divide(pos, npos, out=np.zeros_like(pos), where=npos > 0)
    logP = np.log(np.where(P > 0, P, 1.0))
    loss = -float(np.sum(T * logP)) / len(X)
    G = (P - T) / temp / len(X)                     # dL/dS (row-wise softmax cross-entropy)
    dY = (G + G.T) @ Y
    dZ = (dY - Y * np.sum(dY * Y, axis=1, keepdims=True)) / n
    I = np.eye(W.shape[0])
    loss += drift * float(np.sum((W - I) ** 2))
    grad = dZ.T @ X + 2 * drift * (W - I)
    return loss, grad


def finetune_contradiction(X: np.ndarray, y: np.ndarray, mode: str = "full", epochs: int = 50,
                           lr: float = 0.5, temp: float = 0.1, drift: float = 1e-3) -> tuple[np.ndarray, list[float]]:
    """Full linear map on the u-channel trained so attribute groups separate (Adam)."""
    if mode != "full":
        raise ValueError("u-space contradiction tuning needs mode='full' (a rotation cannot move tension)")
    d = X.shape[1]
    W = np.eye(d)
    m, v = np.zeros_like(W), np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    for t in range(1, epochs + 1):
        loss, g = _supcon(W, X, y, temp, drift)
        losses.append(round(loss, 5))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        W -= lr * 1e-2 * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return W.astype(np.float32).astype(float), losses


def _route_acc(W, Xtr, ytr, Xte, yte) -> float:
    Ztr, Zte = _unit(Xtr @ W.T), _unit(Xte @ W.T)
    protos = np.stack([_unit(Ztr[ytr == g].mean(0)) for g in range(int(ytr.max()) + 1)])
    return float(np.mean(np.argmax(Zte @ protos.T, axis=1) == yte))


def _margin(W, X, y) -> float:
    Z = _unit(X @ W.T)
    S = Z @ Z.T
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    diff = y[:, None] != y[None, :]
    return float(np.mean(S[same]) - np.mean(S[diff]))


def pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


ATTRIBUTE_GROUPS = {
    "fee": [("value", {"attr": "fee"}), ("plausibility", {}), ("mcq", {}), ("cheaper", {}),
            ("fee_difference", {}), ("faster_option_fee", {})],
    "delivery_time": [("value", {"attr": "days"}), ("faster", {})],
    "review_window": [("window_check", {}), ("limit", {"key": "lookback_days"})],
    "session_limit": [("limit", {"key": "max_transactions_per_session"})],
    "verification": [("limit", {"key": "max_auth_attempts"}), ("skip_auth", {})],
}


def attribute_corpus(store, frames) -> tuple[list[str], list[str], list[str]]:
    """Phrasings of KB questions grouped by the attribute (register) they ask about.

    Rendered by the application's own realiser at every paraphrase depth, aliasing and
    question format - labelled by construction, nothing hand-written here."""
    from ..enrich.realizers import Realizer
    R = Realizer()
    texts, labels, depth = [], [], []
    for fr in frames:
        grp = None
        for g, specs in ATTRIBUTE_GROUPS.items():
            for intent, cond in specs:
                if fr.get("intent") == intent and all(fr.get(k) == v for k, v in cond.items()):
                    grp = g
        if grp is None:
            continue
        for pd in ("none", "light", "deep"):
            for al in ("canonical", "synonym", "colloquial"):
                for qf in ("interrogative", "declarative", "imperative"):
                    t = R.realize(fr, {"paraphrase_depth": pd, "entity_aliasing": al, "question_format": qf}, "u")
                    if t not in texts:
                        texts.append(t)
                        labels.append(grp)
                        depth.append(pd)
    return texts, labels, depth


def train_u_channel(store, frames, epochs: int = 50) -> ContradictionReport:
    """Train the u-channel adapter on attribute phrasings; hyper-parameters (temperature 0.1,
    drift 1e-3, 50 epochs) were chosen by held-out-paraphrase-depth cross-validation
    (DESIGN.md, 'GMS training'). Inserts the map and the attribute prototypes into the store."""
    texts, labels, depth = attribute_corpus(store, frames)
    groups = sorted(set(labels))
    y = np.array([groups.index(g) for g in labels])
    X = np.stack([store.u.encode(t) for t in texts])
    I = np.eye(X.shape[1])
    folds = []
    for held in ("none", "light", "deep"):                 # generalise to an unseen paraphrase depth
        tr = np.array([d != held for d in depth])
        W, _ = finetune_contradiction(X[tr], y[tr], epochs=epochs)
        folds.append({"held_out_depth": held, "n_test": int((~tr).sum()),
                      "routing_acc_before": _route_acc(I, X[tr], y[tr], X[~tr], y[~tr]),
                      "routing_acc_after": _route_acc(W, X[tr], y[tr], X[~tr], y[~tr])})
    W, losses = finetune_contradiction(X, y, epochs=epochs)
    store.u_adapter = W
    Z = _unit(X @ W.T)
    store.u_groups = {g: _unit(Z[y == k].mean(0)) for k, g in enumerate(groups)}
    pb, pa = pca2(X), pca2(Z)
    show = [k for k, g in enumerate(labels)]
    return ContradictionReport(
        groups=groups, n_rows=len(texts), margin_before=_margin(I, X, y), margin_after=_margin(W, X, y),
        folds=folds,
        routing_acc_before=float(np.mean([f["routing_acc_before"] for f in folds])),
        routing_acc_after=float(np.mean([f["routing_acc_after"] for f in folds])),
        drift_fro=float(np.linalg.norm(W - I)),
        pca_before=[{"x": float(pb[k, 0]), "y": float(pb[k, 1]), "g": labels[k]} for k in show],
        pca_after=[{"x": float(pa[k, 0]), "y": float(pa[k, 1]), "g": labels[k]} for k in show],
        loss=losses[:: max(1, len(losses) // 30)])
