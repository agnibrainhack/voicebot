"""The GMS on the unit sphere: every entity of the policy graph as a point on S^2, every relation as an arc.

Two placements, the first that is available:

* ``store``  the knowledge store's own geometry: the entity vectors a built knowlytix GMS store saved
  (``v_emb.pt``, the tuned v-space), reduced to 3 dimensions by PCA and put on the sphere. Points that sit
  close are entities the store learned to treat as close.
* ``graph``  the graph's relationships alone: a spectral layout of the entity graph (the 2nd-4th eigenvectors
  of its normalised Laplacian), refined on the sphere so linked entities pull together and the rest spread
  out. Deterministic: the same graph gives the same picture.

``build(dom)`` returns nodes (with section, type and degree), edges (with relation and section) and which
placement was used and why.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .graphs import SECTIONS, _types, classify
from .runs import ROOT

_SKIP_TAIL = {"text", "number"}


def _entity_graph(kg) -> tuple[list[str], list[dict], dict[str, str], dict[str, str]]:
    ty = _types(kg)
    sec_of = {}
    for s, ts in classify(kg).items():
        for t in ts:
            sec_of[t.get("id") or id(t)] = s
    nodes: dict[str, Counter] = defaultdict(Counter)
    edges = []
    for t in kg.triples:
        tail = t["tail"]
        if t.get("tail_type", "entity") in _SKIP_TAIL or not isinstance(tail, str) or t["relation"] == "is_a":
            continue
        if t["relation"] in ("has_alias", "spoken_as", "spoken_cue", "says", "has_text", "has_condition", "states",
                             "carries", "when_used", "asserted_by", "denied_by", "prompts_customer"):
            continue                                  # wording, not structure
        s = sec_of.get(t.get("id") or id(t), "other")
        nodes[t["head"]][s] += 1
        nodes[tail][s] += 1
        edges.append({"s": t["head"], "t": tail, "relation": t["relation"], "section": s})
    names = sorted(nodes)
    sec = {n: nodes[n].most_common(1)[0][0] for n in names}
    return names, edges, sec, ty


def _unit(X: np.ndarray, seed: int = 7) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    bad = (n[:, 0] < 1e-9)
    if bad.any():
        rng = np.random.default_rng(seed)
        X = X.copy()
        X[bad] = rng.normal(size=(int(bad.sum()), 3))
        n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / n


def spectral_sphere(names: list[str], edges: list[dict], iters: int = 300) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    A = np.zeros((N, N))
    for e in edges:
        i, j = idx[e["s"]], idx[e["t"]]
        if i != j:
            A[i, j] += 1.0
            A[j, i] += 1.0
    deg = A.sum(1)
    if N < 4:
        return _unit(np.eye(3)[:N] if N else np.zeros((0, 3)))
    d = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    L = np.eye(N) - (d[:, None] * A * d[None, :])
    w, V = np.linalg.eigh(L)
    X = V[:, 1:4] * d[:, None] if (deg > 0).all() else V[:, 1:4]
    X = X - X.mean(0)
    X = _unit(X / (np.abs(X).max() + 1e-12))
    # refine on the sphere: linked entities pull together, every pair pushes apart
    rng = np.random.default_rng(11)
    X = _unit(X + 0.01 * rng.normal(size=X.shape))
    I = np.array([idx[e["s"]] for e in edges if idx[e["s"]] != idx[e["t"]]], dtype=int)
    J = np.array([idx[e["t"]] for e in edges if idx[e["s"]] != idx[e["t"]]], dtype=int)
    step = 0.05
    for k in range(iters):
        D = X[:, None, :] - X[None, :, :]
        dist2 = (D ** 2).sum(-1) + 1e-3
        rep = (D / dist2[..., None] ** 1.5).sum(1) / N
        att = np.zeros_like(X)
        if len(I):
            dv = X[J] - X[I]
            np.add.at(att, I, dv)
            np.add.at(att, J, -dv)
            att /= np.maximum(deg, 1)[:, None]
        F = 0.6 * rep + 1.0 * att
        F -= (F * X).sum(1, keepdims=True) * X        # tangent to the sphere
        X = _unit(X + step * (1 - k / iters) * F)
    return X


def _find_store(dom, runs_root: Path | None, store_dir: str | None) -> Path | None:
    if store_dir:
        p = Path(store_dir).expanduser()
        return p if (p / "v_emb.pt").exists() else None
    root = runs_root or ROOT / "output" / "console_runs"
    best = None
    if root.exists():
        for job in root.glob("*/job.json"):
            try:
                j = json.loads(job.read_text())
                if j.get("pack") != dom.dir.name:
                    continue
                rep = json.loads((job.parent / "out" / "report.json").read_text())
                sd = Path(((rep.get("store") or {}).get("store_dir")) or "")
                if (sd / "v_emb.pt").exists():
                    m = job.stat().st_mtime
                    if best is None or m > best[0]:
                        best = (m, sd)
            except (OSError, ValueError):
                continue
    return best[1] if best else None


def store_vectors(sdir: Path) -> tuple[list[str], np.ndarray]:
    import torch                                       # present wherever a store was built
    obj = torch.load(sdir / "v_emb.pt", map_location="cpu")
    if isinstance(obj, dict) and "names" in obj and ("vectors" in obj or "emb" in obj):
        names, M = list(obj["names"]), obj.get("vectors", obj.get("emb"))
    elif isinstance(obj, dict):
        names = list(obj)
        M = [obj[n] for n in names]
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        names, M = list(obj[0]), obj[1]
    else:
        raise ValueError("v_emb.pt holds no entity names")
    M = np.stack([np.asarray(getattr(v, "detach", lambda: v)().cpu().numpy() if hasattr(v, "cpu") else v, dtype=float).ravel()
                  for v in M]) if isinstance(M, list) else np.asarray(M.detach().cpu().numpy() if hasattr(M, "cpu") else M, dtype=float)
    return names, M


def pca_sphere(M: np.ndarray) -> np.ndarray:
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    C = M - M.mean(0)
    _u, _s, Vt = np.linalg.svd(C, full_matrices=False)
    return _unit(C @ Vt[:3].T)


def build(dom, runs_root: Path | None = None, store_dir: str | None = None) -> dict[str, Any]:
    names, edges, sec, ty = _entity_graph(dom.kg)
    placement, note = "graph", ""
    X = None
    sdir = _find_store(dom, runs_root, store_dir)
    if sdir is not None:
        try:
            snames, M = store_vectors(sdir)
            pos = dict(zip(snames, pca_sphere(M)))
            hit = [n for n in names if n in pos]
            if len(hit) >= max(3, len(names) // 2):
                G = spectral_sphere(names, edges)          # entities the store never embedded keep a graph place
                X = np.array([pos[n] if n in pos else G[i] for i, n in enumerate(names)])
                placement = "store"
                note = (f"{len(hit)} of {len(names)} entities placed from the store's learned vectors ({sdir.name}, PCA to 3D)"
                        + (f"; {len(names) - len(hit)} from the graph" if len(hit) < len(names) else ""))
            else:
                note = f"the store at {sdir} embeds too few of this graph's entities ({len(hit)}); placed from the graph"
        except Exception as e:                             # noqa: BLE001
            note = f"could not read the store's vectors ({type(e).__name__}: {e}); placed from the graph"
    else:
        note = "no built knowledge store found for this journey (run a campaign with knowlytix); placed from the graph's relationships"
    if X is None:
        X = spectral_sphere(names, edges)
    deg = Counter()
    for e in edges:
        deg[e["s"]] += 1
        deg[e["t"]] += 1
    nodes = [{"id": n, "type": ty.get(n, ""), "section": sec[n], "degree": deg[n],
              "x": round(float(X[i, 0]), 4), "y": round(float(X[i, 1]), 4), "z": round(float(X[i, 2]), 4)}
             for i, n in enumerate(names)]
    # how well the placement keeps linked entities together: mean arc length of an edge vs of a random pair
    idx = {n: i for i, n in enumerate(names)}
    ang = lambda a, b: float(np.arccos(np.clip((X[a] * X[b]).sum(-1), -1, 1)).mean()) if len(a) else 0.0
    I = np.array([idx[e["s"]] for e in edges]); J = np.array([idx[e["t"]] for e in edges])
    rng = np.random.default_rng(3)
    R1, R2 = rng.integers(0, len(names), 2000), rng.integers(0, len(names), 2000)
    return {"pack": dom.dir.name, "placement": placement, "note": note, "nodes": nodes, "edges": edges,
            "relations": dict(Counter(e["relation"] for e in edges).most_common()),
            "sections": [{"key": k, "title": t} for k, t, _ in SECTIONS],
            "fit": {"edge_arc": round(ang(I, J), 3), "random_arc": round(ang(R1, R2), 3)}}
