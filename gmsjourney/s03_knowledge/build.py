"""Building the knowledge store: symbolic self-correction, then the trained GMS (knowlytix).

The GMS (geometric memory store) places every entity of the policy as a point and every relation as a learned
rotation, so a true fact ``(head, relation, tail)`` lands close to where the relation predicts and a false one
lands far away; exact figures are kept separately and verbatim. Building it is like teaching a map: first the
facts are checked against each other (a figure in a prose sentence must equal the register's), then the map is
trained, then its tolerance (how far is "close") is calibrated per relation.

Parts of this file:
  1. ``build_store``: ingest -> symbolic self-correction (type critics, prose figures reconciled against the
     exact register, an injected OCR misread caught) -> the compiled ``GMSStore``; ``oracle_property`` checks that
     true facts score inside the cut and substituted ones outside.
  2. The knowlytix backend: is it installed and licensed (``available`` / ``require``), which text encoders to
     use (Hugging Face models, local paths, or an offline hashing stand-in), on which device.
  3. ``build_kx_store`` / ``load_kx_store`` and ``KXPolicyStore``: GEODE loop, embed loop (v space),
     contradiction SFT (u space), production store, judge calibration; then every geometric read (score_triple,
     cap radius, tension, holonomy, stance checks) answered from the trained store, memoised.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# =====================================================================================================================
# PART 1 - The policy compiler: ingest -> symbolic self-correction (book Ch. 4, the part before training).
#
# The compiler never trusts a candidate fact on the strength of having been extracted.
#
# propose   every candidate from ingest (table backbone + naive prose).
# diagnose  symbolic critics
#             - type check against the relation schema (domain/range),
#             - numeric reconciliation of every prose candidate against the ENM
#               backbone (corroborate / conflict),
#             - unbound subjects (prose that never named an entity),
#             - dangling workflow endpoints and rules that reference unknown limits.
# repair    corroborated prose is merged into the backbone fact (its span added as
#           provenance); conflicting or unbound prose is dropped. Every decision is logged.
# iterate   until a pass makes no change.
#
# Prose facts with no exact value to check against are kept and logged as ``flag``: the
# geometric checks are knowlytix's GEODE loop, which runs on the compiled facts next
# (s03_knowledge/build.py: build_kx_store). The result is the symbolic store (rules, workflow,
# value sets, the exact-number register, the dialog-state layer) and an auditable correction
# log. ``oracle_property`` checks the trained store: true figures inside each relation's cut,
# substituted figures outside.
# =====================================================================================================================

from ..s02_graph.triples import GMS_STATE_RELATIONS
from .store import CandidateTriple, IngestResult, Source, ingest_markdown
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
    from ..s01_packs.loading import current
    return current().enm_for_relation()


def _geode_getattr(name: str):                       # backward-compatible module constant
    if name == "ENM_FOR_REL":
        return _enm_for_rel()
    raise AttributeError(name)


def __getattr__(name: str):
    """Module constants computed from the active pack on every access (ENM_FOR_REL, ENM_MAP)."""
    for lookup in (_geode_getattr, _kx_getattr):
        try:
            return lookup(name)
        except AttributeError:
            pass
    raise AttributeError(name)

@dataclass
class Correction:
    """One self-correction decision on a candidate fact (corroborate, drop a conflict, drop unbound, drop a type
    error, flag), with its reason and source.
    """
    iteration: int
    action: str            # corroborate | drop_conflict | drop_unbound | drop_type | flag
    fact: tuple[str, str, str]
    reason: str
    source: dict[str, Any]


@dataclass
class BuildResult:
    """The compiled symbolic store and the self-correction record (decisions, iterations, convergence)."""
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
    from ..s01_packs.loading import current
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
    from ..s02_graph.compact import is_v2
    if is_v2(policy_path):                       # a compact policy graph (s02_graph/compile.py)
        from ..s02_graph.compile import compile_graph
        return compile_graph(policy_path, inject_misread=inject_misread)
    from ..s02_graph.triples import resolve_policy
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
    from ..s01_packs.loading import current
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


# =====================================================================================================================
# PART 2 - The knowlytix backend: availability, licence and text encoders.
#
# The GMS comes from the ``knowlytix`` package family (pip install knowlytix; Python 3.12),
# with proofloop and forgeloop alongside (requirements.txt). It is required:
# ``require()`` raises a clear error when knowlytix is not installed or no licence is
# configured; ``available()`` answers the same question without raising.
#
# Licence. knowlytix reads its runtime licence from the ``KNOWLYTIX_LICENSE_KEY``
# environment variable. This module never stores, prints or logs the key: if the
# variable is unset it loads it from ``~/.knowlytix/license.key`` (the location the
# beyond-ship-and-pray demos use) into the process environment only.
#
# Encoders. knowlytix warm-starts the v (semantic) and u (logical) channels and fine-
# tunes them from HuggingFace sentence encoders (all-MiniLM-L6-v2, nli-mpnet-base-v2).
# ``resolve_encoders`` uses those when the hub (or a local copy) is reachable. In an
# air-gapped run it builds a local stand-in in HuggingFace format: a WordPiece
# tokenizer trained on the policy text and a one-layer BERT whose word embeddings are
# signed hashes of the sub-word strings and whose attention/feed-forward outputs are
# zeroed, so with knowlytix's mean pooling it is a deterministic bag-of-sub-words
# hashing encoder (typo-robust through sub-words, no semantic prior). Every report
# records which encoders a store was built with.
#
# Local model folders (no download). Point the two encoders at directories that already
# hold the models (a ``sentence-transformers`` or plain HuggingFace save: config.json +
# weights + tokenizer), either by editing ``HF_V_LOCAL`` / ``HF_U_LOCAL`` below or with
# the environment variables ``GMSJOURNEY_HF_V`` / ``GMSJOURNEY_HF_U``. When both folders
# exist they are used as-is: nothing is downloaded, the hub is never contacted and
# ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set for the process.
#
# DeepSpeed. ``transformers.save_pretrained`` (via accelerate) imports DeepSpeed whenever it
# is installed; a DeepSpeed build that does not match the installed torch fails on import
# (``cannot import name '_get_socket_with_port'``). ``guard_deepspeed()`` checks once and,
# if DeepSpeed cannot be imported, tells accelerate/transformers it is absent. Nothing
# here uses DeepSpeed.
# =====================================================================================================================

HF_V = "sentence-transformers/all-MiniLM-L6-v2"           # hub names (used only when no local folder is set)
HF_U = "sentence-transformers/nli-mpnet-base-v2"
# local folders that already contain the models; edit here or set GMSJOURNEY_HF_V / GMSJOURNEY_HF_U
HF_V_LOCAL = os.environ.get("GMSJOURNEY_HF_V", "/aip/amex/nchak14/VoiceBOT/all-MiniLM-L6-v2")
HF_U_LOCAL = os.environ.get("GMSJOURNEY_HF_U", "/aip/amex/nchak14/VoiceBOT/nli-mpnet-base-v2")
LICENSE_FILE = Path.home() / ".knowlytix" / "license.key"
CACHE = Path(os.environ.get("GMSJOURNEY_CACHE", Path.home() / ".cache" / "gmsjourney"))


def _load_license_env() -> bool:
    if os.environ.get("KNOWLYTIX_LICENSE_KEY"):
        return True
    if LICENSE_FILE.exists():
        os.environ["KNOWLYTIX_LICENSE_KEY"] = LICENSE_FILE.read_text().strip()
        return True
    return False


def available() -> bool:
    """True when the knowlytix family (knowlytix, proofloop, forgeloop) is installed and a licence is configured."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    guard_deepspeed()
    if importlib.util.find_spec("knowlytix") is None or importlib.util.find_spec("knowlytix.core") is None:
        return False
    if importlib.util.find_spec("proofloop") is None or importlib.util.find_spec("forgeloop") is None:
        return False
    return _load_license_env()


def require() -> None:
    """Raise a clear error when knowlytix is not installed or not licensed."""
    if not available():
        raise ImportError("knowlytix backend unavailable: `pip install -r requirements.txt` (knowlytix, "
                          "proofloop, forgeloop; Python 3.12) and set KNOWLYTIX_LICENSE_KEY "
                          "(or ~/.knowlytix/license.key)")


def describe() -> dict[str, Any]:
    """What is installed: knowlytix (with versions), proofloop, forgeloop."""
    out = {"knowlytix": False, "proofloop": importlib.util.find_spec("proofloop") is not None,
           "forgeloop": importlib.util.find_spec("forgeloop") is not None}
    if available():
        import io
        import contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            import knowlytix.core as kc
        out["knowlytix"] = True
        from importlib.metadata import version
        out["versions"] = {p: version(p) for p in ("knowlytix-core", "knowlytix-knowledge", "knowlytix-harness",
                                                   "knowlytix-benchmark")}
        out["licence_tier"] = getattr(kc, "IS_DEMO_MODE", None) is False and "licensed" or "demo"
    return out


# ----------------------------------------------------------------------------- environment guards
_DS_CHECKED = False


def guard_deepspeed() -> bool:
    """If DeepSpeed is installed but cannot be imported (version mismatch with torch), make
    accelerate and transformers treat it as absent. Returns True when a patch was applied."""
    global _DS_CHECKED
    if _DS_CHECKED:
        return False
    _DS_CHECKED = True
    if importlib.util.find_spec("deepspeed") is None:
        return False
    try:
        import deepspeed  # noqa: F401
        return False
    except Exception:                      # noqa: BLE001 - any import failure: pretend it is not installed
        pass
    no = lambda *a, **k: False             # noqa: E731
    import sys
    for mod in ("accelerate.utils", "accelerate.utils.imports", "accelerate.utils.other",
                "accelerate.accelerator", "transformers.integrations.deepspeed", "transformers.integrations",
                "transformers.modeling_utils", "transformers.utils.import_utils", "transformers.trainer"):
        try:
            m = sys.modules.get(mod) or importlib.import_module(mod)
        except Exception:                  # noqa: BLE001
            continue
        for name in ("is_deepspeed_available", "is_deepspeed_zero3_enabled"):
            if hasattr(m, name):
                setattr(m, name, no)
    return True


# ----------------------------------------------------------------------------- encoders
def _local_model(path: str | None) -> bool:
    """A folder holding a saved model (HF or sentence-transformers layout)."""
    if not path:
        return False
    p = Path(path).expanduser()
    return p.is_dir() and ((p / "config.json").exists() or (p / "modules.json").exists())


def local_encoders() -> tuple[str, str] | None:
    """(v_path, u_path) when both local model folders exist, else None."""
    if _local_model(HF_V_LOCAL) and _local_model(HF_U_LOCAL):
        return str(Path(HF_V_LOCAL).expanduser()), str(Path(HF_U_LOCAL).expanduser())
    return None


def _force_offline() -> None:
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"


def _hub_reachable(timeout: float = 3.0) -> bool:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            pass
    except OSError:
        return False
    try:                                   # through a proxy a TCP connect is not enough
        import urllib.request
        urllib.request.urlopen("https://huggingface.co/api/models/" + HF_V, timeout=timeout)
        return True
    except Exception:                      # noqa: BLE001
        return False


def _hf_cached(name: str) -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (hub / ("models--" + name.replace("/", "--"))).exists()


def build_offline_encoder(path: Path, corpus: list[str], salt: str, dim: int = 384, vocab_size: int = 3000) -> Path:
    """HuggingFace-format hashing encoder (see module docstring). Deterministic."""
    import numpy as np
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=specials,
                                                              show_progress=False))
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                            special_tokens=[("[CLS]", tok.token_to_id("[CLS]")),
                                                            ("[SEP]", tok.token_to_id("[SEP]"))])
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]", mask_token="[MASK]").save_pretrained(path)
    cfg = BertConfig(vocab_size=tok.get_vocab_size(), hidden_size=dim, num_hidden_layers=1, num_attention_heads=4,
                     intermediate_size=4 * dim, max_position_embeddings=512, layer_norm_eps=1e-12)
    torch.manual_seed(0)
    model = BertModel(cfg, add_pooling_layer=False)
    with torch.no_grad():
        W = torch.zeros(tok.get_vocab_size(), dim)
        for t, i in tok.get_vocab().items():
            if t in specials:
                continue
            seed = int(hashlib.sha256(f"{salt}|{t.lstrip('#')}".encode()).hexdigest()[:8], 16)
            g = np.random.default_rng(seed).standard_normal(dim)
            W[i] = torch.tensor(g / np.linalg.norm(g) * np.sqrt(dim), dtype=torch.float32)
        model.embeddings.word_embeddings.weight.copy_(W)
        model.embeddings.position_embeddings.weight.zero_()
        model.embeddings.token_type_embeddings.weight.zero_()
        for layer in model.encoder.layer:
            layer.attention.output.dense.weight.zero_()
            layer.attention.output.dense.bias.zero_()
            layer.output.dense.weight.zero_()
            layer.output.dense.bias.zero_()
    model.save_pretrained(path)
    (path / "GMSJOURNEY_OFFLINE_ENCODER.json").write_text(json.dumps(
        {"kind": "offline hashing stand-in", "salt": salt, "dim": dim, "vocab": tok.get_vocab_size()}))
    return path


def encoder_kind(mode: str | None = None) -> str:
    """'huggingface' or 'offline-stand-in': what resolve_encoders would pick, without building anything."""
    mode = mode or os.environ.get("GMSJOURNEY_ENCODER", "auto")
    if mode == "offline":
        return "offline-stand-in"
    if local_encoders() is not None:
        return "huggingface"
    if (_hf_cached(HF_V) and _hf_cached(HF_U)) or _hub_reachable():
        return "huggingface"
    return "offline-stand-in"


def resolve_encoders(corpus: list[str], mode: str | None = None) -> dict[str, Any]:
    """{'v': name_or_path, 'u': name_or_path, 'kind': 'huggingface' | 'offline-stand-in', 'reason': ...}"""
    mode = mode or os.environ.get("GMSJOURNEY_ENCODER", "auto")
    guard_deepspeed()
    if mode in ("auto", "hf", "local"):
        loc = local_encoders()
        if loc is not None:
            _force_offline()
            return {"v": loc[0], "u": loc[1], "kind": "huggingface",
                    "reason": f"local model folders ({loc[0]}, {loc[1]}); hub not contacted"}
        if mode == "local":
            raise RuntimeError(f"local encoders requested but not found: v={HF_V_LOCAL!r} u={HF_U_LOCAL!r} "
                               "(each folder needs config.json or modules.json)")
        if _hf_cached(HF_V) and _hf_cached(HF_U):
            return {"v": HF_V, "u": HF_U, "kind": "huggingface", "reason": "found in the local HF cache"}
        if _hub_reachable():
            return {"v": HF_V, "u": HF_U, "kind": "huggingface", "reason": "HF hub reachable"}
        if mode == "hf":
            raise RuntimeError("HuggingFace encoders requested but the hub is unreachable and nothing is cached")
    key = hashlib.sha256("\n".join(corpus).encode()).hexdigest()[:12]
    base = CACHE / "encoders" / key
    v, u = base / "v", base / "u"
    if not (v / "config.json").exists():
        build_offline_encoder(v, corpus, "v")
    if not (u / "config.json").exists():
        build_offline_encoder(u, corpus, "u")
    return {"v": str(v), "u": str(u), "kind": "offline-stand-in",
            "reason": "HuggingFace hub unreachable from this environment; deterministic hashing stand-in in HF format"}


# =====================================================================================================================
# PART 3 - The GMS built, trained, calibrated and judged by knowlytix (book Ch. 2-4, 8).
#
# ``build_kx_store`` runs knowlytix's GEODE pipeline - the same steps, calls and default
# parameters as ``knowlytix.knowledge.geode.rag.build_calibrated_rag_store`` (GEODE trainer
# 300 epochs, 8 rounds; embed loop: trainer 150 epochs, 2 rounds, EmbeddingSFTConfig(rank=8,
# mode="full", out_dim=d_v); contradiction SFT 400 epochs) - with three stated differences:
#
# * the text encoders are passed by name or local path (``backend.resolve_encoders``) - the
#   same models; the packaged call hard-codes HuggingFace hub names for the u channel;
# * duplicate resolution runs audit-only by default (``dedup_mode="off"``; the reference
#   lets removals stand), because on this graph every removal it proposes is a fact the
#   policy states (a dialog state applying several rules, two delivery options);
# * the u encoder is trained on per-relation phrasings rendered by the application's
#   realiser (``_attribute_phrasings``) plus the relation name for every other relation; the
#   reference default is the relation name alone.
#
# The reference takes the production store's ``DocGMSConfig`` from its caller (ingest_mode
# default "regex"); ours is ``DEFAULT_STORE_CONFIG`` (loss_mode "cap", ingest_mode "regex",
# train.epochs 150) over the DocGMSConfig defaults.
#
# At run time every geometric read is the knowlytix method: ``score_triple`` (an unstated
# tail through ``score_triple_emb`` on the tuned v encoder), ``cap_radius``,
# ``tension_energy``, ``check_holonomy`` and ``is_path_consistent`` (the calibrated
# ``tau_path`` is written onto ``config.verify``). Grounding is ``score_triple <= cap_radius``;
# knowlytix's per-head calibrated admissibility is reported (``admissibility``), not used to
# decide (see ``is_grounded``).
#
# Every parameter can be set at run time (``build_gms`` / ``CampaignConfig``), including any
# field of the production store's ``DocGMSConfig`` through ``store_config`` (dotted keys,
# e.g. ``{"loss_mode": "cap", "train.epochs": 200, "cap.margin": 0.3}``).
#
#   1  self-correct      GeodeLoop(make_default_trainer)            propose -> train -> diagnose -> repair;
#                                                                   dedup_mode: off (default: GeodeLoop(dedup=False),
#                                                                   so every retrain and the anchor check see the
#                                                                   full fact set; duplicate resolution then runs
#                                                                   audit-only on the production store) | audit
#                                                                   (loop removes, removals of verbatim facts are
#                                                                   reinstated afterwards - the 0.3 behaviour) |
#                                                                   accept (stock gmsh: removals stand)
#   2  tune v            GeodeEmbedLoop + EmbeddingSFTConfig        has_alias supervision + manifold links;
#                                                                   name-keyed vectors -> Mode-B warm start;
#                                                                   v_sft_mode: full (reference default) |
#                                                                   rotation (book) | ab (train both, keep the one
#                                                                   with the better held-out noisy-alias binding)
#   3  tune u            contradiction_sft (full mode)              per-relation phrasings
#   4  relevance cal.    calibrate_relevance_thresholds
#   5  train store       store_from_triples (cap admissibility)     GMSExpertStore, saved
#   6  coverage          coverage_report / graph_coverage
#   7  calibrate         GMSJudge.calibrate -> calibration.json     the operating points every gate reads
#
# ``KXPolicyStore`` then exposes the store through the interface the rest of
# gmsjourney uses (it subclasses the compiled ``GMSStore`` for the symbolic policy
# facts - rules, workflow conditions, value sets - compiled from the same document),
# but every geometric or exact-numeric read is answered by knowlytix: ENM lookups,
# ``score_triple``, cap radii, tension, holonomy, the judge's calibrated cuts and
# the tuned encoders. Our compiled facts and knowlytix's ingest are cross-checked
# (every ENM value must agree) so the two views of the document cannot drift.
# =====================================================================================================================

backend_describe = describe
backend_require = require
from .store import kx_relation, kx_triple, render_gms_markdown

# ============================================================================ device
# The one place the device is set, for building and for loading the store: "cpu", "cuda",
# "cuda:1", "mps" or "auto" (cuda when available, else cpu). The GMSJOURNEY_DEVICE environment
# variable overrides it without editing the file. The device is not part of the store's cache
# key: a store built on a GPU loads on a CPU and vice versa. (knowlytix's contradiction_sft
# takes no device argument and places itself.)
DEVICE = "auto"


def device() -> str:
    """The compute device: GMSJOURNEY_DEVICE, else the module setting, ``auto`` -> cuda when available."""
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
    from ..s01_packs.loading import current
    return current().kx_enm_map()


def _attribute_names() -> dict[str, str]:
    """u-encoder class label (a relation) -> the attribute name the runtime checks use."""
    from ..s01_packs.loading import current
    f = current().section("figures")
    out = {r: sp["attribute"] for r, sp in (f.get("relations") or {}).items() if sp.get("attribute")}
    if f.get("limits", {}).get("attribute"):
        out[f["limits"].get("attribute_label", "has_value")] = f["limits"]["attribute"]
    return out


def _kx_getattr(name: str):                       # backward-compatible module constant
    if name == "ENM_MAP":
        return _enm_map()
    raise AttributeError(name)


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


@dataclass
class KXBuild:
    """A trained store and its build report."""
    store: "KXPolicyStore"
    report: dict[str, Any] = field(default_factory=dict)


def _attribute_phrasings() -> dict[str, list[str]]:
    """Per-relation phrasings for the u-channel contradiction SFT, rendered by the
    application's realiser (labelled by construction)."""
    from ..s05_rewording.wording import Realizer
    from ..s04_cases.meaning import SemanticFrame as F
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


def _graph():
    from ..s01_packs.loading import current
    return current().graph


def _render_view(store) -> str:
    g = _graph()
    if g is not None:
        from ..s02_graph.compile import render_gms_view
        return render_gms_view(store, g)
    return render_gms_markdown(store)


def _phrasings() -> dict[str, list[str]]:
    g = _graph()
    if g is not None:
        from ..s02_graph.compile import attribute_phrasings
        return attribute_phrasings(g)
    return _attribute_phrasings()


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


# the phases of a store build, as the console's progress bar names them
GMS_PHASES = ["Compile the policy and choose encoders", "GEODE self-correction loop", "Tune the meaning space (embed loop)",
              "Tune the contradiction space (SFT)", "Calibrate relevance", "Train the production store",
              "Measure coverage", "Calibrate the judge"]


def _last_timings(store_dir: Path) -> dict[str, float]:
    """Phase timings of the most recent store built on this machine (for the progress bar's estimate)."""
    best, when = {}, 0.0
    for rep in [store_dir / "gms_build_report.json", *store_dir.parent.glob("*/gms_build_report.json")]:
        try:
            m = rep.stat().st_mtime
            if m > when:
                best, when = json.loads(rep.read_text()).get("timings_s") or {}, m
        except (OSError, ValueError):
            continue
    return best


def build_kx_store(policy_path: str | Path, store_dir: str | Path, *, geode_epochs: int = 300,
                   max_iters: int = 8, embed_trainer_epochs: int = 150, embed_max_iters: int = 2,
                   sft_epochs: int | None = None, contradiction_epochs: int = 400,
                   store_overrides: dict | None = None, encoder_mode: str | None = None, seed: int = 42,
                   dedup_mode: str = "off", v_sft_mode: str = "full", log=print) -> KXBuild:
    """Build the trained GMS with knowlytix (GEODE loop, embed loop, contradiction SFT, production store, judge
    calibration) into ``store_dir``, and return it with its build report.
    """
    if dedup_mode not in ("off", "audit", "accept"):
        raise ValueError(f"dedup_mode must be off | audit | accept, not {dedup_mode!r}")
    if v_sft_mode not in ("rotation", "full", "ab"):
        raise ValueError(f"v_sft_mode must be rotation | full | ab, not {v_sft_mode!r}")
    backend_require()
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

    t0 = time.time()
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    from .. import progress
    last = _last_timings(store_dir)            # how long each phase took the last time a store was built here

    def phase(i: int, timing: str | None = None) -> None:
        progress.phase("gms", i, len(GMS_PHASES), GMS_PHASES[i] if i < len(GMS_PHASES) else "Store built",
                       phases=GMS_PHASES if i == 0 else None, expect_s=last.get(timing) if timing else None)
    steps = progress.StepCounter("gms")
    phase(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device())
    timings: dict[str, float] = {}

    # 0. compile the policy (symbolic facts + the GMS view GEODE ingests)
    from ..s02_graph.triples import resolve_policy
    compiled = build_store(policy_path)
    view = store_dir / "policy.gms.md"
    view.write_text(_render_view(compiled.store))
    md_path, _ = resolve_policy(policy_path)
    corpus = Path(md_path).read_text().splitlines() + view.read_text().splitlines()
    for rel, ph in _phrasings().items():
        corpus += ph
    enc = resolve_encoders(corpus, encoder_mode)
    log(f"      encoders: {enc['kind']} ({enc['reason']})")

    cfg = store_config(store_dir, {"train.seed": seed, **(store_overrides or {})})
    ingest = cfg.ingest_mode
    cfg.embedding = EmbeddingConfig(semantic_model=enc["v"], logical_model=enc["u"], warm_start=True,
                                    freeze_base=True, dim_policy="project", device=device())
    trainer = make_default_trainer(dev, epochs=geode_epochs)

    # 1. GEODE self-correction
    t = time.time()
    phase(1, "geode_loop")
    with _quiet(), steps:
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
    phase(2, "embed_loop")
    for m in modes:
        t_m = time.time()
        with steps:
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
    phrasings = _phrasings()
    content_rels = sorted({r for _h, r, _t in kept if r.startswith("has_") and r != "has_alias"})
    for r in content_rels:
        phrasings.setdefault(r, [(r[4:] if r.startswith("has_") else r).replace("_", " ")])
    phase(3, "contradiction_sft")
    with _quiet(), steps:
        u_enc = contradiction_sft(phrasings, base_model=enc["u"], epochs=contradiction_epochs)
        u_enc.save(store_dir / "contradiction_encoder")
    u_eval = _routing_eval(phrasings, enc["u"], contradiction_epochs)
    timings["contradiction_sft"] = time.time() - t
    log(f"      u-space contradiction SFT: held-out routing {u_eval['before']:.2f} -> {u_eval['after']:.2f} "
        f"({timings['contradiction_sft']:.0f}s)")

    # 4. relevance calibration
    phase(4)
    with _quiet():
        relcal = calibrate_relevance_thresholds(phrasings, eres.ft.encode, u_enc.encode)
    (store_dir / "relevance_calibration.json").write_text(json.dumps(relcal, indent=2, default=float))

    # 5. production store (cap admissibility, Mode-B tuned v vectors)
    t = time.time()
    phase(5, "store_training")
    with _quiet(), steps:
        kx = store_from_triples(str(view), kept, cfg, dev, ingest_mode=ingest)
    timings["store_training"] = time.time() - t
    log(f"      production store trained ({timings['store_training']:.0f}s): "
        f"{kx.adapter.num_entities} entities, {len(kx.doc_graph.triples)} triples, {len(kx.doc_graph.enm)} ENM")
    dedup_audit = _dedup_audit(kx, kept) if dedup_mode == "off" else None

    # 6. coverage
    phase(6)
    with _quiet():
        cov = coverage_report(kx)
        gcov = graph_coverage(res.triples)
    (store_dir / "coverage_report.json").write_text(json.dumps({**cov.as_dict(), "graph": gcov.as_dict()},
                                                                indent=2, default=str))

    # 7. calibrate the judge's operating points (persisted where the generators read them)
    t = time.time()
    phase(7, "calibration")
    with _quiet(), steps:
        judge = GMSJudge(kx)
        judge.calibrate(seed=seed)
        judge.apply_thresholds()
        judge.save(cfg.store_path)
    thresholds = dict(getattr(judge, "_thresholds", {}) or {})
    timings["calibration"] = time.time() - t

    store = KXPolicyStore.from_compiled(compiled.store, kx, judge, eres.ft, u_enc, enc)
    agree = store.enm_agreement()
    report = {
        "backend": backend_describe(), "encoders": enc, "view": str(view.name),
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
    report["entity_points"] = export_entity_points(kx, store_dir)
    (store_dir / "gms_build_report.json").write_text(json.dumps(report, indent=1, default=str))
    phase(len(GMS_PHASES))
    return KXBuild(store, report)


POINTS_FILE = "gms_points.npz"


def entity_points(kx) -> tuple[list[str], Any, str]:
    """The trained production store's own entity points: (entity names in index order, a matrix with one row per
    entity, where it was read from). The GMS places every entity on a hypersphere and measures closeness as the angle
    between two points, so these rows (normalised) are what the store's angular distances are computed from.

    Read from the store's model: an entity-embedding accessor when it has one, else the model's 2-D weight whose rows
    are the entities (``adapter.entity_to_idx``), preferring one named for entities."""
    import numpy as np
    adapter = kx.adapter
    e2i = dict(adapter.entity_to_idx)
    n = len(e2i)
    names = [None] * n
    for name, i in e2i.items():
        if 0 <= int(i) < n:
            names[int(i)] = str(name)
    model = getattr(kx, "model", None) or getattr(kx, "gkg", None)
    to_np = lambda t: np.asarray(t.detach().cpu().float().numpy() if hasattr(t, "detach") else t, dtype=float)  # noqa: E731
    for holder in (model, kx):
        for attr in ("entity_embeddings", "get_entity_embeddings", "entity_vectors", "entity_points"):
            fn = getattr(holder, attr, None)
            if fn is None:
                continue
            try:
                M = to_np(fn() if callable(fn) else fn)
            except Exception:                              # noqa: BLE001 - try the next way
                continue
            if M.ndim == 2 and M.shape[0] == n:
                return names, M, f"{type(holder).__name__}.{attr}"
    cands = []
    sd = model.state_dict() if hasattr(model, "state_dict") else {}
    for key, t in sd.items():
        shape = tuple(getattr(t, "shape", ()))
        if len(shape) == 2 and shape[0] == n and shape[1] >= 3:
            k = key.lower()
            score = (2 if "ent" in k else 0) - (2 if "rel" in k else 0) + (1 if "emb" in k or "weight" in k else 0)
            cands.append((score, key, t))
    if not cands:
        raise ValueError(f"the store's model has no {n}-row entity matrix")
    cands.sort(key=lambda x: -x[0])
    _, key, t = cands[0]
    return names, to_np(t), f"model.{key}"


def export_entity_points(kx, store_dir: str | Path) -> dict[str, Any]:
    """Save the trained store's entity points next to it (``gms_points.npz``: names, vectors, source), so the
    console's GMS sphere is drawn from the store's own angular distances without loading knowlytix."""
    import numpy as np
    try:
        names, M, src = entity_points(kx)
        keep = [i for i, nm in enumerate(names) if nm is not None]
        np.savez_compressed(Path(store_dir) / POINTS_FILE, names=np.array([names[i] for i in keep]), vectors=M[keep],
                            source=np.array(src))
        return {"file": POINTS_FILE, "entities": len(keep), "dim": int(M.shape[1]), "source": src}
    except Exception as e:                                 # noqa: BLE001 - the sphere falls back; the build goes on
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def load_kx_store(policy_path: str | Path, store_dir: str | Path) -> KXBuild:
    """Reload a built store (knowlytix model, ENM, judge calibration, tuned encoders)."""
    backend_require()
    import torch
    with _quiet():
        from knowlytix.embedding import FineTunedEmbedding
        from knowlytix.harness.testing.judge import GMSJudge
        from knowlytix.knowledge.store import GMSExpertStore
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
    if not (store_dir / POINTS_FILE).exists():          # a store built before the points were saved
        report["entity_points"] = export_entity_points(kx, store_dir)
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
    from ..s05_rewording.wording import asr_noise, typo
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
            v = self.kx.lookup_enm(typ, fmt.format(key=key))
            if v is None and key != key.lower():          # the ingest may lower-case row entities
                v = self.kx.lookup_enm(typ, fmt.format(key=key.lower()))
            return v

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
        rel = "has_value" if relation.startswith("has_") and relation[4:] in _limit_names() else relation
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
        if cal.get("status") == "degenerate":
            return {"verdict": "unavailable", "energy": e, "rule": rule.rule_id, "reason": cal.get("reason")}
        v = "supported" if e <= cal["tau_agree"] else "contradicted" if e >= cal["tau_contra"] else "uncertain"
        return {"verdict": v, "energy": e, "rule": rule.rule_id, "attr_cos": 1.0}

    def find_contradictions(self, rules=None) -> list[dict[str, Any]]:
        if self.calibration["tension"].get("status") == "degenerate":
            raise RuntimeError(f"tension calibration is degenerate: {self.calibration['tension'].get('reason')}")
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
        if cal["holonomy"].get("tau_path") is not None:
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
        if _graph() is not None:                        # a compact graph pack: cohorts from its own graph
            return {"false_accept_ceiling": 0.05, "score_triple": per_rel,
                    "holonomy": self._holonomy_from_graph(), "tension": self._tension_from_graph(thr),
                    "judge": {k: v for k, v in thr.items() if isinstance(v, (int, float, str, tuple, list))}}
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

    # ------------------------------------------------------------------ cohorts from the pack's graph
    def _raw_tension(self, a: str, b: str) -> float | None:
        with _quiet():
            te = self.kx.tension_energy(str(a).lower(), str(b).lower())
        return float(te) if te is not None else None

    def _tension_from_graph(self, thr: dict[str, Any]) -> dict[str, Any]:
        """Stance cuts for ``stance_check`` and ``find_contradictions``, from pairs the graph itself
        declares opposite: the stances must / must_not, and every prompt's yes / no reading
        (CONFIRM / DENY, RECOGNIZED / UNRECOGNIZED, ...). Agreement is each of those entities with
        itself. GMSJudge's three-way cuts are used when it could fit them; otherwise the cuts are
        the cohort's own zero-error band: agree up to the largest same-stance energy, contradict
        from the smallest opposite-stance energy, uncertain in between (never collapsed into one
        cut). If the cohort does not separate, the calibration is refused as degenerate (as
        GMSJudge refuses it) and every check that needs it reports itself unavailable."""
        g = _graph()
        unclear = (g.glob.get("unclear") or {}).get("act")
        pairs = [("must", "must_not")]
        for st in g.prompt_states:
            y, n = g.interprets(st)
            if y and n and y != n and unclear not in (y, n) and (y, n) not in pairs and (n, y) not in pairs:
                pairs.append((y, n))           # a yes and a no the prompt reads as opposite answers
        opposite = [[a, b, e] for a, b in pairs if (e := self._raw_tension(a, b)) is not None]
        ents = sorted({x for a, b, _ in opposite for x in (a, b)})
        same = [[x, x, e] for x in ents if (e := self._raw_tension(x, x)) is not None]
        out = {"attr_match": 0.8, "same": same, "opposite": opposite,
               "cohort": "graph: stances must/must_not + every prompt's yes/no reading"}
        if thr.get("tau_contra") is not None and thr.get("tau_ent") is not None and thr["tau_ent"] < thr["tau_contra"]:
            return {**out, "status": "ok", "source": "GMSJudge", "tau_agree": float(thr["tau_ent"]),
                    "tau_contra": float(thr["tau_contra"])}
        if same and opposite and max(e for *_, e in same) < min(e for *_, e in opposite):
            return {**out, "status": "ok", "source": "graph cohort (zero-error band)",
                    "tau_agree": max(e for *_, e in same), "tau_contra": min(e for *_, e in opposite)}
        why = ("no opposite pair is in the store" if not opposite else
               "same-stance and opposite-stance energies overlap: the u-space does not separate them")
        return {**out, "status": "degenerate", "source": "graph cohort", "reason": why,
                "tau_agree": None, "tau_contra": None}

    def _holonomy_from_graph(self, max_hops: int = 3, support: int = 2, negatives: int = 3) -> dict[str, Any]:
        """Path-consistency cut from the compositions the pack's own facts close: a chain of
        relations r1..rk whose every instance (a, c) is also asserted by one direct relation r
        (at least ``support`` instances; r not on the chain, so a type edge such as is_a is not
        mistaken for a rule). Negatives compare the same chain with relations that relate none of
        its instances. A pack with no such composition has nothing for holonomy to certify (book
        Ch. 3.6: it applies only to composed paths), and the cut is reported not applicable."""
        from collections import defaultdict
        known = {}

        def carried(path, direct) -> float | None:
            key = (tuple(path), direct)
            if key not in known:
                with _quiet():
                    d = self.kx.check_holonomy([kx_relation(r) for r in path], kx_relation(direct))
                known[key] = float(d) if d is not None else None
            return known[key]

        rels = {r: {(h, t) for h, _, t in trs} for r, trs in self.by_r.items()}
        by_head = {r: defaultdict(set) for r in rels}
        for r, ps in rels.items():
            for a, b in ps:
                by_head[r][a].add(b)
        found, frontier = [], [((r,), set(ps)) for r, ps in sorted(rels.items())]
        for _ in range(max_hops - 1):
            nxt = []
            for path, pairs in frontier:
                for r2 in sorted(rels):
                    comp = {(a, c) for a, b in pairs for c in by_head[r2].get(b, ()) if a != c}
                    if len(comp) < support:
                        continue
                    chain = path + (r2,)
                    nxt.append((chain, comp))
                    for r3 in sorted(rels):
                        if r3 not in chain and comp <= rels[r3]:
                            found.append((list(chain), r3, comp))
            frontier = nxt
        pos, neg = [], []
        for chain, direct, comp in found:
            d = carried(chain, direct)
            if d is None:
                continue
            pos.append({"path": chain, "direct": direct, "defect": d, "instances": len(comp)})
            unrelated = [r for r in sorted(rels) if r not in chain and r != direct and not (rels[r] & comp)]
            for r in unrelated:
                if sum(1 for x in neg if x["path"] == chain) >= negatives:
                    break
                dn = carried(chain, r)
                if dn is not None:
                    neg.append({"path": chain, "direct": r, "defect": dn})
        if not pos or not neg:
            return {"status": "not_applicable", "tau_path": None, "positives": pos, "negatives": neg,
                    "source": "graph compositions",
                    "reason": "the pack's facts close no chain of relations into a direct one"}
        op = GMSStore._operating_point([p["defect"] for p in pos], [n["defect"] for n in neg])
        status = "ok" if op["tar"] == 1.0 and op["far"] == 0.0 else "degenerate"
        return {"status": status, "tau_path": float(op["tau"]), "tar": op["tar"], "far": op["far"],
                "positives": pos, "negatives": neg, "source": "graph compositions"}

    def summary(self) -> dict[str, Any]:
        out = GMSStore.summary(self)
        out["geometric_reads_cached"] = len(self.kx._cache)
        out.update({"backend": "knowlytix", "kx_entities": self.kx.adapter.num_entities,
                    "kx_triples": len(self.kx.doc_graph.triples), "kx_enm": len(self.kx.doc_graph.enm)})
        return out


_LIMITS = {"max_transactions_per_session", "lookback_days", "max_auth_attempts", "max_clarification_retries",
           "max_unsure_retries", "max_turns"}


def _limit_names() -> set[str]:
    g = _graph()
    return set(g.limits) if g is not None else _LIMITS
