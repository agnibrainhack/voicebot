"""The GMS built, trained, calibrated and judged by knowlytix (book Ch. 2-4, 8).

``build_kx_store`` runs knowlytix's GEODE pipeline - the same steps, calls and default
parameters as ``knowlytix.knowledge.geode.rag.build_calibrated_rag_store`` (GEODE trainer
300 epochs, 8 rounds; embed loop: trainer 150 epochs, 2 rounds, EmbeddingSFTConfig(rank=8,
mode="full", out_dim=d_v); contradiction SFT 400 epochs) - with three stated differences:

* the text encoders are passed by name or local path (``backend.resolve_encoders``) - the
  same models; the packaged call hard-codes HuggingFace hub names for the u channel;
* duplicate resolution runs audit-only by default (``dedup_mode="off"``; the reference
  lets removals stand), because on this graph every removal it proposes is a fact the
  policy states (a dialog state applying several rules, two delivery options);
* the u encoder is trained on per-relation phrasings rendered by the application's
  realiser (``_attribute_phrasings``) plus the relation name for every other relation; the
  reference default is the relation name alone.

The reference takes the production store's ``DocGMSConfig`` from its caller (ingest_mode
default "regex"); ours is ``DEFAULT_STORE_CONFIG`` (loss_mode "cap", ingest_mode "regex",
train.epochs 150) over the DocGMSConfig defaults.

At run time every geometric read is the knowlytix method: ``score_triple`` (an unstated
tail through ``score_triple_emb`` on the tuned v encoder), ``cap_radius``,
``tension_energy``, ``check_holonomy`` and ``is_path_consistent`` (the calibrated
``tau_path`` is written onto ``config.verify``). Grounding is ``score_triple <= cap_radius``;
knowlytix's per-head calibrated admissibility is reported (``admissibility``), not used to
decide (see ``is_grounded``).

Every parameter can be set at run time (``build_gms`` / ``CampaignConfig``), including any
field of the production store's ``DocGMSConfig`` through ``store_config`` (dotted keys,
e.g. ``{"loss_mode": "cap", "train.epochs": 200, "cap.margin": 0.3}``).

  1  self-correct      GeodeLoop(make_default_trainer)            propose -> train -> diagnose -> repair;
                                                                  dedup_mode: off (default: GeodeLoop(dedup=False),
                                                                  so every retrain and the anchor check see the
                                                                  full fact set; duplicate resolution then runs
                                                                  audit-only on the production store) | audit
                                                                  (loop removes, removals of verbatim facts are
                                                                  reinstated afterwards - the 0.3 behaviour) |
                                                                  accept (stock gmsh: removals stand)
  2  tune v            GeodeEmbedLoop + EmbeddingSFTConfig        has_alias supervision + manifold links;
                                                                  name-keyed vectors -> Mode-B warm start;
                                                                  v_sft_mode: full (reference default) |
                                                                  rotation (book) | ab (train both, keep the one
                                                                  with the better held-out noisy-alias binding)
  3  tune u            contradiction_sft (full mode)              per-relation phrasings
  4  relevance cal.    calibrate_relevance_thresholds
  5  train store       store_from_triples (cap admissibility)     GMSExpertStore, saved
  6  coverage          coverage_report / graph_coverage
  7  calibrate         GMSJudge.calibrate -> calibration.json     the operating points every gate reads

``KXPolicyStore`` then exposes the store through the interface the rest of
gmsjourney uses (it subclasses the compiled ``GMSStore`` for the symbolic policy
facts - rules, workflow conditions, value sets - compiled from the same document),
but every geometric or exact-numeric read is answered by knowlytix: ENM lookups,
``score_triple``, cap radii, tension, holonomy, the judge's calibrated cuts and
the tuned encoders. Our compiled facts and knowlytix's ingest are cross-checked
(every ENM value must agree) so the two views of the document cannot drift.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import backend
from .kx_view import kx_relation, kx_triple, render_gms_markdown
from .store import GMSStore

# ============================================================================ device
# The one place the device is set, for building and for loading the store: "cpu", "cuda",
# "cuda:1", "mps" or "auto" (cuda when available, else cpu). The GMSJOURNEY_DEVICE environment
# variable overrides it without editing the file. The device is not part of the store's cache
# key: a store built on a GPU loads on a CPU and vice versa. (knowlytix's contradiction_sft
# takes no device argument and places itself.)
DEVICE = "auto"


def device() -> str:
    d = os.environ.get("GMSJOURNEY_DEVICE") or DEVICE
    if d == "auto":
        import torch
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return d


def _on_device(x):
    """A tensor moved to the store's device (the tuned encoder may return it on the CPU)."""
    return x.to(device()) if hasattr(x, "to") else x


def _enm_map() -> dict[str, tuple[str, str]]:
    """ENM category -> (knowlytix register type, key format), from domain.yaml -> figures."""
    from ..domain import current
    return current().kx_enm_map()


def _attribute_names() -> dict[str, str]:
    """u-encoder class label (a relation) -> the attribute name the runtime checks use."""
    from ..domain import current
    f = current().section("figures")
    out = {r: sp["attribute"] for r, sp in (f.get("relations") or {}).items() if sp.get("attribute")}
    if f.get("limits", {}).get("attribute"):
        out[f["limits"].get("attribute_label", "has_value")] = f["limits"]["attribute"]
    return out


def __getattr__(name: str):                       # backward-compatible module constant
    if name == "ENM_MAP":
        return _enm_map()
    raise AttributeError(name)


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


@dataclass
class KXBuild:
    store: "KXPolicyStore"
    report: dict[str, Any] = field(default_factory=dict)


def _attribute_phrasings() -> dict[str, list[str]]:
    """Per-relation phrasings for the u-channel contradiction SFT, rendered by the
    application's realiser (labelled by construction)."""
    from ..enrich.realizers import Realizer
    from ..ontology import SemanticFrame as F
    R = Realizer()
    K = lambda i, **kw: F.make("ASK_KB", intent=i, **kw)
    frames = {
        "has_fee_usd": [K("value", attr="fee", entity=o) for o in ("standard_delivery", "expedited_delivery")]
                       + [K("cheaper"), K("fee_difference")],
        "has_business_days_max": [K("value", attr="days", entity=o) for o in ("standard_delivery", "expedited_delivery")]
                                 + [K("faster")],
        "has_value": [K("limit", key=k) for k in ("max_transactions_per_session", "lookback_days", "max_auth_attempts")],
        "has_fee_confirmation": [K("needs_confirmation")],
        "has_delivered_by": [K("count_options")],
        "has_precedes": [K("procedure")],
    }
    out = {}
    for rel, frs in frames.items():
        ph = set()
        for fr in frs:
            for pd in ("none", "light", "deep"):
                for qf in ("interrogative", "declarative"):
                    ph.add(R.realize(fr, {"paraphrase_depth": pd, "question_format": qf}, "u"))
        out[rel] = sorted(ph)
    return out


# the production store's DocGMSConfig unless overridden: cap admissibility, regex ingest
DEFAULT_STORE_CONFIG = {"loss_mode": "cap", "ingest_mode": "regex", "train.epochs": 150}


def store_config(store_dir: str | Path, overrides: dict | None = None):
    """The production store's DocGMSConfig: package defaults, then DEFAULT_STORE_CONFIG, then the
    caller's dotted-key overrides (``{"train.epochs": 200, "cap.margin": 0.3, "geometry.d_v": 256}``)."""
    with _quiet():
        from knowlytix.knowledge.config import DocGMSConfig
    cfg = DocGMSConfig(store_path=str(Path(store_dir) / "kx"))
    for key, value in {**DEFAULT_STORE_CONFIG, **(overrides or {})}.items():
        obj, *path = [cfg] + key.split(".")
        for part in path[:-1]:
            obj = getattr(obj, part)
        if not hasattr(obj, path[-1]):
            raise KeyError(f"DocGMSConfig has no field {key!r}")
        setattr(obj, path[-1], value)
    cfg.train.device = device()
    return cfg


def build_kx_store(policy_path: str | Path, store_dir: str | Path, *, geode_epochs: int = 300,
                   max_iters: int = 8, embed_trainer_epochs: int = 150, embed_max_iters: int = 2,
                   sft_epochs: int | None = None, contradiction_epochs: int = 400,
                   store_overrides: dict | None = None, encoder_mode: str | None = None, seed: int = 42,
                   dedup_mode: str = "off", v_sft_mode: str = "full", log=print) -> KXBuild:
    if dedup_mode not in ("off", "audit", "accept"):
        raise ValueError(f"dedup_mode must be off | audit | accept, not {dedup_mode!r}")
    if v_sft_mode not in ("rotation", "full", "ab"):
        raise ValueError(f"v_sft_mode must be rotation | full | ab, not {v_sft_mode!r}")
    backend.require()
    import torch
    with _quiet():
        from knowlytix.embedding import EmbeddingSFTConfig
        from knowlytix.core.config import EmbeddingConfig
        from knowlytix.knowledge.geode import GeodeLoop, make_default_trainer
        from knowlytix.knowledge.geode.embed_loop import EmbedLoopConfig, GeodeEmbedLoop, contradiction_sft
        from knowlytix.knowledge.geode.rag import fact_only_triples, store_from_triples
        from knowlytix.knowledge.rag import coverage_report, graph_coverage
        from knowlytix.knowledge.rag.relevance import calibrate_relevance_thresholds
        from knowlytix.harness.testing.judge import GMSJudge
    from .geode import build_store

    t0 = time.time()
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device())
    timings: dict[str, float] = {}

    # 0. compile the policy (symbolic facts + the GMS view GEODE ingests)
    from .policy_kg import resolve_policy
    compiled = build_store(policy_path)
    view = store_dir / "policy.gms.md"
    view.write_text(render_gms_markdown(compiled.store))
    md_path, _ = resolve_policy(policy_path)
    corpus = Path(md_path).read_text().splitlines() + view.read_text().splitlines()
    for rel, ph in _attribute_phrasings().items():
        corpus += ph
    enc = backend.resolve_encoders(corpus, encoder_mode)
    log(f"      encoders: {enc['kind']} ({enc['reason']})")

    cfg = store_config(store_dir, {"train.seed": seed, **(store_overrides or {})})
    ingest = cfg.ingest_mode
    cfg.embedding = EmbeddingConfig(semantic_model=enc["v"], logical_model=enc["u"], warm_start=True,
                                    freeze_base=True, dim_policy="project", device=device())
    trainer = make_default_trainer(dev, epochs=geode_epochs)

    # 1. GEODE self-correction
    t = time.time()
    with _quiet():
        loop = GeodeLoop(trainer, device=dev, ingest_mode=ingest, max_iters=max_iters,
                         residual_threshold=None, canonicalize=False, dedup=(dedup_mode != "off"))
        res = loop.run(str(view))
    kept = fact_only_triples(res.triples)
    rejected = []
    if dedup_mode == "audit":
        # 0.3 behaviour: GEODE's duplicate resolution assumes one tail per (head, relation) is likely
        # and drops "cap outliers"; a removal of a row the source states verbatim is reinstated after
        # the loop (the loop's later retrains and its anchor check ran on the reduced set).
        for d in res.duplicates_removed:
            tr = tuple(d["triple"])
            if tr not in kept:
                kept.append(tr)
                rejected.append({**d, "decision": "rejected: stated verbatim in the source table"})
    timings["geode_loop"] = time.time() - t
    log(f"      GEODE loop (dedup={dedup_mode}): converged={res.converged} iterations={res.iterations} "
        f"corrections={len(res.corrections)}, anchor violations={len(res.anchor_violations)}, "
        f"dedup removals={len(res.duplicates_removed)}, reinstated={len(rejected)} ({timings['geode_loop']:.0f}s)")

    # 2. tune v: GEODE embed loop (document alias structure + manifold links)
    t = time.time()
    names_and_aliases = [(c, a) for c, als in compiled.store.aliases.items() for a in als]
    probes = _noisy_probes(names_and_aliases)

    def embed(mode: str):
        extra = {"epochs": sft_epochs} if sft_epochs is not None else {}     # else the package default (100)
        sft = EmbeddingSFTConfig(rank=8, mode=mode, encoder=enc["v"], out_dim=cfg.geometry.d_v, device=device(),
                                 **extra)
        with _quiet():
            eloop = GeodeEmbedLoop(make_default_trainer(dev, epochs=embed_trainer_epochs),
                                   EmbedLoopConfig(sft=sft, max_iters=embed_max_iters, use_geometry=True,
                                                   ingest_mode=ingest))
            er = eloop.run(str(view), triples=kept, eval_pairs=[(s, c.lower()) for s, c in probes])
        return er, _binding_eval(er.ft, probes, enc["v"])

    modes = ["rotation", "full"] if v_sft_mode == "ab" else [v_sft_mode]
    trials = {}
    for m in modes:
        t_m = time.time()
        trials[m] = embed(m)
        log(f"      GEODE embed loop ({m}): iterations={trials[m][0].iterations} converged={trials[m][0].converged} "
            f"noisy-alias binding {trials[m][1]['before']:.2f} -> {trials[m][1]['after']:.2f} ({time.time() - t_m:.0f}s)")
    # A/B rule: higher held-out noisy-alias binding wins; a tie keeps rotation (the book's choice)
    chosen = max(modes, key=lambda m: (round(trials[m][1]["after"], 6), m == "rotation"))
    eres, v_eval = trials[chosen]
    v_ab = {"modes": {m: {"before": tr[1]["before"], "after": tr[1]["after"], "iterations": tr[0].iterations,
                          "converged": tr[0].converged} for m, tr in trials.items()},
            "chosen": chosen, "rule": "higher held-out noisy-alias binding; tie -> rotation (book)",
            "gmsh_default": "full"}
    ent_names = sorted({e for h, _r, t2 in kept for e in (h, t2)})
    v_emb = eres.ft.export_vectors(ent_names)
    torch.save(v_emb, store_dir / "v_emb.pt")
    cfg.embedding.v_vectors_path = str(store_dir / "v_emb.pt")
    eres.ft.save(store_dir / "tuned_encoder")
    timings["embed_loop"] = time.time() - t
    log(f"      v-space SFT: kept {chosen} ({timings['embed_loop']:.0f}s)")

    # 3. tune u: contradiction SFT over per-relation phrasings (full mode)
    t = time.time()
    phrasings = _attribute_phrasings()
    content_rels = sorted({r for _h, r, _t in kept if r.startswith("has_") and r != "has_alias"})
    for r in content_rels:
        phrasings.setdefault(r, [(r[4:] if r.startswith("has_") else r).replace("_", " ")])
    with _quiet():
        u_enc = contradiction_sft(phrasings, base_model=enc["u"], epochs=contradiction_epochs)
        u_enc.save(store_dir / "contradiction_encoder")
    u_eval = _routing_eval(phrasings, enc["u"], contradiction_epochs)
    timings["contradiction_sft"] = time.time() - t
    log(f"      u-space contradiction SFT: held-out routing {u_eval['before']:.2f} -> {u_eval['after']:.2f} "
        f"({timings['contradiction_sft']:.0f}s)")

    # 4. relevance calibration
    with _quiet():
        relcal = calibrate_relevance_thresholds(phrasings, eres.ft.encode, u_enc.encode)
    (store_dir / "relevance_calibration.json").write_text(json.dumps(relcal, indent=2, default=float))

    # 5. production store (cap admissibility, Mode-B tuned v vectors)
    t = time.time()
    with _quiet():
        kx = store_from_triples(str(view), kept, cfg, dev, ingest_mode=ingest)
    timings["store_training"] = time.time() - t
    log(f"      production store trained ({timings['store_training']:.0f}s): "
        f"{kx.adapter.num_entities} entities, {len(kx.doc_graph.triples)} triples, {len(kx.doc_graph.enm)} ENM")
    dedup_audit = _dedup_audit(kx, kept) if dedup_mode == "off" else None

    # 6. coverage
    with _quiet():
        cov = coverage_report(kx)
        gcov = graph_coverage(res.triples)
    (store_dir / "coverage_report.json").write_text(json.dumps({**cov.as_dict(), "graph": gcov.as_dict()},
                                                                indent=2, default=str))

    # 7. calibrate the judge's operating points (persisted where the generators read them)
    t = time.time()
    with _quiet():
        judge = GMSJudge(kx)
        judge.calibrate(seed=seed)
        judge.apply_thresholds()
        judge.save(cfg.store_path)
    thresholds = dict(getattr(judge, "_thresholds", {}) or {})
    timings["calibration"] = time.time() - t

    store = KXPolicyStore.from_compiled(compiled.store, kx, judge, eres.ft, u_enc, enc)
    agree = store.enm_agreement()
    report = {
        "backend": backend.describe(), "encoders": enc, "view": str(view.name),
        "geode": {"converged": res.converged, "iterations": res.iterations, "max_iters": max_iters,
                  "n_proposed": len(res.triples), "n_kept_fact_triples": len(kept),
                  "corrections": res.corrections, "anchor_violations": res.anchor_violations,
                  "n_anchor_violations": len(res.anchor_violations),
                  "dedup_mode": dedup_mode,
                  "duplicates_proposed": res.duplicates_removed, "duplicates_rejected": rejected,
                  "duplicates_flagged": res.duplicates_flagged,
                  "dedup_audit_only": dedup_audit},
        "store": {"n_entities": kx.adapter.num_entities, "n_triples": len(kx.doc_graph.triples),
                  "n_enm": len(kx.doc_graph.enm), "loss_mode": cfg.loss_mode, "ingest_mode": ingest,
                  "epochs": cfg.train.epochs,
                  "geometry": {"d_v": cfg.geometry.d_v, "d_u": cfg.geometry.d_u, "m": cfg.geometry.m},
                  "config_overrides": {"train.seed": seed, **(store_overrides or {})}},
        "build_parameters": {"geode_epochs": geode_epochs, "max_iters": max_iters,
                             "embed_trainer_epochs": embed_trainer_epochs, "embed_max_iters": embed_max_iters,
                             "sft_epochs": sft_epochs if sft_epochs is not None else "package default",
                             "v_sft_mode": v_sft_mode, "contradiction_epochs": contradiction_epochs,
                             "dedup_mode": dedup_mode, "reference": "knowlytix build_calibrated_rag_store"},
        "embed_loop": {"iterations": eres.iterations, "converged": eres.converged, "mode": chosen,
                       "history": eres.history, "n_canonicals": len(eres.canonicals), **v_eval, "ab": v_ab},
        "contradiction_sft": {"relations": sorted(phrasings), **u_eval},
        "judge_thresholds": {k: v for k, v in thresholds.items() if not hasattr(v, "predict")},
        "enm_agreement": agree,
        "compiler_corrections": compiled.correction_counts(),
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (store_dir / "gms_build_report.json").write_text(json.dumps(report, indent=1, default=str))
    return KXBuild(store, report)


def load_kx_store(policy_path: str | Path, store_dir: str | Path) -> KXBuild:
    """Reload a built store (knowlytix model, ENM, judge calibration, tuned encoders)."""
    backend.require()
    import torch
    with _quiet():
        from knowlytix.embedding import FineTunedEmbedding
        from knowlytix.harness.testing.judge import GMSJudge
        from knowlytix.knowledge.store import GMSExpertStore
    from .geode import build_store
    store_dir = Path(store_dir)
    report = json.loads((store_dir / "gms_build_report.json").read_text())
    compiled = build_store(policy_path)
    # the same production-store config it was built with (stores built before the overrides were
    # recorded used DEFAULT_STORE_CONFIG)
    cfg = store_config(store_dir, (report.get("store") or {}).get("config_overrides"))
    with _quiet():
        kx = GMSExpertStore(cfg, device=torch.device(device()))
        if not kx.load():
            raise RuntimeError(f"could not load the knowlytix store at {store_dir / 'kx'}")
        judge = GMSJudge(kx)
        judge.load(str(store_dir / "kx"))
        judge.apply_thresholds()
        v_ft = FineTunedEmbedding.load(store_dir / "tuned_encoder")
        u_ft = FineTunedEmbedding.load(store_dir / "contradiction_encoder")
    return KXBuild(KXPolicyStore.from_compiled(compiled.store, kx, judge, v_ft, u_ft, report.get("encoders")), report)


def _dedup_audit(kx, triples) -> dict[str, Any]:
    """GEODE's duplicate resolution run audit-only on the trained production store: what it would
    remove or flag, without removing anything (dedup_mode='off')."""
    try:
        with _quiet():
            from knowlytix.knowledge.geode.dedup import resolve_duplicates
            model = getattr(kx, "model", None) or getattr(kx, "gkg", None)
            _kept, removed, flagged = resolve_duplicates(model, kx.adapter, list(triples))
        return {"would_remove": removed, "would_flag": flagged, "applied": False}
    except Exception as e:                     # noqa: BLE001 - audit only; never fails the build
        return {"error": f"{type(e).__name__}: {str(e)[:200]}", "applied": False}


def _noisy_probes(pairs: list[tuple[str, str]], seeds=(101, 102, 103)) -> list[tuple[str, str]]:
    from ..enrich.noise import asr_noise, typo
    out = []
    for canon, alias in pairs:
        for s in seeds:
            rng = random.Random(f"{alias}|{s}")
            out.append((asr_noise(alias, rng) if s % 2 else typo(alias, rng, rate=0.2), canon))
    return out


def _binding_eval(ft, probes, base_encoder: str) -> dict[str, Any]:
    """Held-out noisy aliases: nearest prototype in raw encoder space vs the tuned encoder."""
    with _quiet():
        from knowlytix.core.graph.encoders import encode_texts
    labels = list(ft.label_order)
    texts = [p for p, _ in probes]
    gold = [c.lower() for _, c in probes]
    with _quiet():
        preds, _ = ft.classify(texts)
    after = float(np.mean([p == g for p, g in zip(preds, gold)]))
    # raw encoder: prototype = encoded humanized canonical name
    with _quiet():
        P = encode_texts([lab.replace("_", " ") for lab in labels], base_encoder, device()).cpu().numpy()
        X = encode_texts(texts, base_encoder, device()).cpu().numpy()
    raw = [labels[int(np.argmax(P @ x))] for x in X]
    before = float(np.mean([p == g for p, g in zip(raw, gold)]))
    return {"before": before, "after": after, "n_probes": len(probes),
            "note": "held-out typo/ASR variants of the policy's aliases; before = raw encoder vs entity name"}


def _routing_eval(phrasings: dict[str, list[str]], base_encoder: str, epochs: int) -> dict[str, Any]:
    """Two-fold held-out phrasings: nearest group centroid, raw u encoder vs contradiction-tuned."""
    with _quiet():
        from knowlytix.core.graph.encoders import encode_texts
        from knowlytix.knowledge.geode.embed_loop import contradiction_sft
    groups = {g: p for g, p in phrasings.items() if len(p) >= 4}
    accs_b, accs_a = [], []
    for fold in (0, 1):
        tr = {g: [x for i, x in enumerate(p) if i % 2 != fold] for g, p in groups.items()}
        te = [(x, g) for g, p in groups.items() for i, x in enumerate(p) if i % 2 == fold]
        with _quiet():
            ft = contradiction_sft(tr, base_model=base_encoder, epochs=epochs)
            preds, _ = ft.classify([x for x, _ in te])
            cen = {g: encode_texts(p, base_encoder, device()).cpu().numpy().mean(0) for g, p in tr.items()}
            X = encode_texts([x for x, _ in te], base_encoder, device()).cpu().numpy()
        names = list(cen)
        C = np.stack([cen[g] / np.linalg.norm(cen[g]) for g in names])
        raw = [names[int(np.argmax(C @ x))] for x in X]
        accs_b.append(np.mean([p == g for p, (_, g) in zip(raw, te)]))
        accs_a.append(np.mean([p == g for p, (_, g) in zip(preds, te)]))
    return {"before": float(np.mean(accs_b)), "after": float(np.mean(accs_a)),
            "groups": sorted(groups), "note": "two-fold held-out phrasings; before = raw u encoder centroids"}


# ============================================================================ facade
class CachedKX:
    """Read-through proxy over a GMSExpertStore that memoises the geometric reads.

    Each knowlytix ``score_triple`` builds the relation's rotor (a Cayley transform)
    afresh; the campaign asks the same questions many times (gates in every run,
    decoys shared across mined questions), so pure reads are cached. Everything else
    is delegated untouched, so the proxy can be handed to knowlytix generators and
    forgeloop gates in place of the store."""

    def __init__(self, kx):
        object.__setattr__(self, "_kx", kx)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name):
        return getattr(self._kx, name)

    def _memo(self, key, fn):
        c = self._cache
        if key not in c:
            with _quiet():
                c[key] = fn()
        return c[key]

    def score_triple(self, head, rel, tail):
        return self._memo(("s", head, rel, tail), lambda: self._kx.score_triple(head, rel, tail))

    def cap_radius(self, rel):
        return self._memo(("c", rel), lambda: self._kx.cap_radius(rel))

    def tension_energy(self, a, b):
        return self._memo(("t", a, b), lambda: self._kx.tension_energy(a, b))

    def check_holonomy(self, path, direct):
        return self._memo(("h", tuple(path), direct), lambda: self._kx.check_holonomy(path, direct))

    def is_path_consistent(self, path, direct):
        tau = self._kx.config.verify.tau_path
        return self._memo(("p", tuple(path), direct, tau), lambda: self._kx.is_path_consistent(path, direct))

    def score_triple_emb(self, head, rel, tail_text, encoder):
        """Native ``score_triple_emb``; the tail vector comes from the store's tuned v encoder.
        Memoised by the tail's text (the tuned encoder reloads its base model on each call)."""
        return self._memo(("e", head, rel, tail_text),
                          lambda: self._kx.score_triple_emb(head, rel, _on_device(encoder.encode([tail_text]))))


class KXPolicyStore(GMSStore):
    """Symbolic policy facts from the compiler; geometry, ENM and calibration from knowlytix."""

    backend_name = "knowlytix"

    @classmethod
    def from_compiled(cls, sym: GMSStore, kx, judge, v_ft=None, u_ft=None, encoders=None) -> "KXPolicyStore":
        obj = cls.__new__(cls)
        obj.__dict__.update(sym.__dict__)               # triples, rules, workflow, value sets, aliases, ...
        obj.kx_raw = kx
        obj.kx, obj.judge, obj.v_ft, obj.u_ft = CachedKX(kx), judge, v_ft, u_ft
        obj.encoders = encoders or {}
        obj._kx_enm = {}
        for (cat, key), e in sym.enm.items():
            v = obj._kx_lookup(cat, key)
            if v is not None:
                obj._kx_enm[(cat, key)] = v
        obj.calibration = obj._calibration_view()
        return obj

    # ------------------------------------------------------------------ exact reads
    def _kx_lookup(self, category: str, key: str):
        enm_map = _enm_map()
        if category not in enm_map:
            return None
        typ, fmt = enm_map[category]
        with _quiet():
            return self.kx.lookup_enm(typ, fmt.format(key=key))

    def lookup_enm(self, category: str, key: str) -> int | float:
        """Exact read from knowlytix's integrity-checked ENM register."""
        v = self._kx_lookup(category, key)
        if v is None:
            raise KeyError(f"ENM entry {category}/{key} not in the knowlytix register")
        return int(v) if float(v).is_integer() else float(v)

    def enm_agreement(self) -> dict[str, Any]:
        rows = []
        for (cat, key), e in sorted(self.enm.items()):
            kv = self._kx_lookup(cat, key)
            rows.append({"entry": f"{cat}/{key}", "compiled": e.value, "knowlytix": kv,
                         "agree": kv is not None and abs(float(kv) - float(e.value)) < 1e-9})
        return {"all_agree": all(r["agree"] for r in rows), "n": len(rows), "rows": rows}

    # ------------------------------------------------------------------ geometry
    def score_triple(self, head: str, relation: str, tail: str) -> float:
        """knowlytix ``score_triple``: geodesic distance to the head-conditioned cap centre.

        A tail the store never trained as an entity (a figure the policy does not state) is
        scored the knowlytix way, ``score_triple_emb`` on the tuned v encoder's embedding of
        the tail text. pi/2 (the edge of the hemisphere) only when the head or the relation
        is not in the store at all."""
        s = self.kx_score(head, relation, tail)
        if s is None:
            h, r, t = kx_triple(head, relation, tail)
            if self.v_ft is not None and self.kx.relation_exists(r) and self.kx.entity_exists(h):
                s = self.kx.score_triple_emb(h, r, t, self.v_ft)
        return float(s) if s is not None else math.pi / 2

    def kx_score(self, head: str, relation: str, tail: str) -> float | None:
        """knowlytix ``score_triple`` as is (None for anything outside the store's vocabulary)."""
        h, r, t = kx_triple(head, relation, tail)
        with _quiet():
            return self.kx.score_triple(h, r, t)

    def tau(self, relation: str) -> float:
        """The relation's learned cap radius; else the judge's calibrated geodesic cut."""
        rel = "has_value" if relation.startswith("has_") and relation[4:] in _LIMITS else relation
        with _quiet():
            rho = self.kx.cap_radius(kx_relation(rel) if rel != "has_value" else rel)
        if rho is not None:
            return float(rho)
        g = (getattr(self.judge, "_thresholds", {}) or {}).get("geodesic")
        if g is None:
            raise KeyError(f"no calibrated cut for {relation!r}")
        return float(g)

    def is_grounded(self, head: str, relation: str, tail: str) -> bool:
        """Inside the relation's learned cap: ``score_triple <= cap_radius``, the admissibility
        test the store is trained for and the one the knowlytix benchmark generators apply.
        knowlytix's per-head calibrated admissibility is reported alongside (``admissibility``):
        on this policy its per-head margins admit cross-tail substitutions on the two-head
        delivery relations, so it is not used to decide."""
        return self.score_triple(head, relation, tail) <= self.tau(relation)

    def admissibility(self, head: str, relation: str, tail: str) -> dict[str, Any] | None:
        """knowlytix's layered calibrated admissibility (``knowlytix.core.graph.admissibility``:
        per-head cap margins AND per-relation tension cut, both at 5% FPR, calibrated once).
        None when the triple is outside the store's vocabulary."""
        with _quiet():
            from knowlytix.core.graph.admissibility import (admissible, calibrate_cap_margins_per_head,
                                                            calibrate_tension_threshold)
        model, adapter = self.kx_raw.model, self.kx_raw.adapter
        cal = self.__dict__.get("_admissibility_cal")
        if cal is None:
            with _quiet():
                cal = (calibrate_cap_margins_per_head(model, adapter, fpr_target=0.05),
                       calibrate_tension_threshold(model, adapter, fpr_target=0.05))
            self.__dict__["_admissibility_cal"] = cal
        h, r, t = kx_triple(head, relation, tail)
        e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
        hi, ri, ti = e2i.get(h), r2i.get(r), e2i.get(t)
        if hi is None or ri is None or ti is None:
            return None
        memo = self.__dict__.setdefault("_admissible_memo", {})
        if (hi, ri, ti) not in memo:
            with _quiet():
                v = admissible(model, adapter, hi, ri, ti, cap_margins=cal[0], tension_tau=cal[1])
            memo[(hi, ri, ti)] = {k: (bool(x) if isinstance(x, bool) else float(x)) for k, x in v.items()}
        return memo[(hi, ri, ti)]

    def tension_energy(self, a, b) -> float:
        with _quiet():
            te = self.kx.tension_energy(str(a).lower(), str(b).lower())
        return float(te) if te is not None else math.sqrt(2)

    def check_holonomy(self, path: list[str], direct: str) -> float:
        with _quiet():
            d = self.kx.check_holonomy([kx_relation(r) for r in path], kx_relation(direct))
        return float(d) if d is not None else math.pi

    def is_path_consistent(self, path: list[str], direct: str) -> bool:
        """knowlytix ``is_path_consistent`` (holonomy defect <= ``config.verify.tau_path``). The
        store ships with no tau_path; the cut written there is this store's calibrated one
        (``calibration["holonomy"]``), the way GMSJudge writes its own. Unknown relations: False."""
        with _quiet():
            ok = self.kx.is_path_consistent([kx_relation(r) for r in path], kx_relation(direct))
        return bool(ok) if ok is not None else False

    def stance_check(self, subject: str, predicate: str, obj: str, polarity: int) -> dict[str, Any]:
        """Value-polarity verifier. The rule is found symbolically; the verdict is read
        from knowlytix's u-space tension between the claimed stance and the rule's stance
        (must / must_not are declared opposite_of in the GMS view)."""
        rule = next((r for r in self.rules.values() if (r.subject, r.predicate, r.obj) == (subject, predicate, obj)), None)
        if rule is None:
            return {"verdict": "uncertain", "energy": None, "rule": None}
        claimed = "must" if polarity > 0 else "must_not"
        stance = "must" if rule.stance == "may" else rule.stance
        e = self.tension_energy(claimed, stance)
        cal = self.calibration["tension"]
        v = "supported" if e <= cal["tau_agree"] else "contradicted" if e >= cal["tau_contra"] else "uncertain"
        return {"verdict": v, "energy": e, "rule": rule.rule_id, "attr_cos": 1.0}

    def find_contradictions(self, rules=None) -> list[dict[str, Any]]:
        rules = rules if rules is not None else self.rules
        out = []
        items = sorted(rules.items())
        for i, (ra, a) in enumerate(items):
            for rb, b in items[i + 1:]:
                if (a.subject, a.predicate, a.obj) != (b.subject, b.predicate, b.obj):
                    continue
                e = self.tension_energy(a.stance if a.stance != "may" else "must", b.stance if b.stance != "may" else "must")
                if e >= self.calibration["tension"]["tau_contra"]:
                    out.append({"a": ra, "b": rb, "energy": e})
        return out

    def attribute_of(self, text: str) -> tuple[str | None, float]:
        """Route a claim to its attribute with the contradiction-tuned u encoder (memoised:
        the tuned encoder reloads its base model on every call, and bot utterances repeat)."""
        if self.u_ft is None:
            return None, 0.0
        memo = self.__dict__.setdefault("_attr_memo", {})
        if text not in memo:
            memo[text] = self._attribute_of(text)
        return memo[text]

    def _attribute_of(self, text: str) -> tuple[str | None, float]:
        with _quiet():
            preds, scores = self.u_ft.classify([text])
        p = preds[0]
        sc = scores[0]
        conf = float(max(sc)) if hasattr(sc, "__iter__") else float(sc)
        return (_attribute_names().get(p, p), conf)

    def fuzzy_match_entity(self, surface: str, types=None, mode: str = "frame"):
        """Bind a surface form to an entity with the GEODE-tuned v encoder (model 1)."""
        if self.v_ft is None:
            return None, 0.0
        with _quiet():
            preds, scores = self.v_ft.classify([surface])
        sc = scores[0]
        return preds[0], float(max(sc)) if hasattr(sc, "__iter__") else float(sc)

    # ------------------------------------------------------------------ calibration view
    def calibrate(self, seed: int = 7) -> dict[str, Any]:
        self.calibration = self._calibration_view(seed)
        return self.calibration

    @property
    def calibration(self) -> dict[str, Any]:
        return self.__dict__["calibration"]

    @calibration.setter
    def calibration(self, cal: dict[str, Any]) -> None:
        # the holonomy cut also goes onto the store's verify config, where knowlytix's own
        # is_path_consistent reads it
        self.__dict__["calibration"] = cal
        self.kx_raw.config.verify.tau_path = float(cal["holonomy"]["tau_path"])

    def _calibration_view(self, seed: int = 7) -> dict[str, Any]:
        """The operating points in the shape the rest of the framework reads, all taken
        from knowlytix: per-relation cap radii (with their realised true/false accept
        rates on the store's own triples and counterfactual tails), the judge's
        calibrated holonomy and tension cuts, and the lexicon matcher's cut."""
        thr = dict(getattr(self.judge, "_thresholds", {}) or {})
        rng = random.Random(seed)
        per_rel = {}
        for rel, trs in sorted(self.by_r.items()):
            h0, r0, t0 = sorted(trs)[0]
            if self.kx_score(h0, r0, t0) is None:      # relation not carried by the GMS view
                continue
            try:
                tau = self.tau(rel)
            except KeyError:
                continue
            pos = [self.score_triple(h, r, t) for h, r, t in sorted(trs)]
            neg = [self.score_triple(h, r, c) for h, r, t in sorted(trs) for c in self.negatives_for(h, r, t, rng)]
            per_rel[rel] = {"tau": tau, "tar": float(np.mean([p <= tau for p in pos])) if pos else 0.0,
                            "far": float(np.mean([n <= tau for n in neg])) if neg else 0.0,
                            "n_pos": len(pos), "n_neg": len(neg), "source": "knowlytix cap_radius"}
        hp = [self.check_holonomy(p, d) for p, d in ((["precedes", "precedes"], "precedes_two"),
                                                     (["has_session", "reviews", "governed_by"], "subject_to"))]
        hn = [self.check_holonomy(["precedes", "precedes"], u) for u in ("has_fee_usd", "delivered_by", "member_of")]
        # The judge's holonomy cut is global over every relation triangle of the graph; the
        # self-test asks a narrower question (does this declared composition close?), so the
        # cut is fitted on the declared compositions vs unrelated relations, and the judge's
        # global cut is reported alongside.
        tau_path = GMSStore._operating_point(hp, hn)["tau"]
        te_same = [self.tension_energy(x, x) for x in ("must", "recognized")]
        te_opp = [self.tension_energy(a, b) for a, b in (("must", "must_not"), ("recognized", "unrecognized"))]
        tau_contra = thr.get("tau_contra")
        tau_ent = thr.get("tau_ent")
        if tau_contra is None:
            tau_contra = (max(te_same) + min(te_opp)) / 2
        if tau_ent is None:
            tau_ent = tau_contra
        return {"false_accept_ceiling": 0.05, "score_triple": per_rel,
                "holonomy": {"tau_path": float(tau_path), "positives": hp, "negatives": hn,
                             "source": "declared-composition cohort", "judge_global_cut": thr.get("holonomy")},
                "tension": {"tau_contra": float(tau_contra), "tau_agree": float(min(tau_ent, tau_contra)),
                            "attr_match": 0.8, "same": te_same, "opposite": te_opp,
                            "source": "GMSJudge" if thr.get("tau_contra") is not None else "store cohort"},
                "judge": {k: v for k, v in thr.items() if isinstance(v, (int, float, str, tuple, list))}}

    def summary(self) -> dict[str, Any]:
        out = GMSStore.summary(self)
        out["geometric_reads_cached"] = len(self.kx._cache)
        out.update({"backend": "knowlytix", "kx_entities": self.kx.adapter.num_entities,
                    "kx_triples": len(self.kx.doc_graph.triples), "kx_enm": len(self.kx.doc_graph.enm)})
        return out


_LIMITS = {"max_transactions_per_session", "lookback_days", "max_auth_attempts", "max_clarification_retries",
           "max_unsure_retries", "max_turns"}
