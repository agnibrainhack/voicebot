"""Which GMS backend runs, and with which text encoders.

The GMS core comes from the ``knowlytix`` package family (pip install knowlytix;
Python 3.12). ``available()`` is the same open-core seam as ``forgeloop.gms``:
when ``knowlytix`` is importable *and* licensed, the campaign builds, trains,
calibrates and judges with it; otherwise it falls back to the built-in
implementation (knowledge/store.py), so the open baseline always runs.

Licence. knowlytix reads its runtime licence from the ``KNOWLYTIX_LICENSE_KEY``
environment variable. This module never stores, prints or logs the key: if the
variable is unset it loads it from ``~/.knowlytix/license.key`` (the location the
beyond-ship-and-pray demos use) into the process environment only.

Encoders. knowlytix warm-starts the v (semantic) and u (logical) channels and fine-
tunes them from HuggingFace sentence encoders (all-MiniLM-L6-v2, nli-mpnet-base-v2).
``resolve_encoders`` uses those when the hub (or a local copy) is reachable. In an
air-gapped run it builds a local stand-in in HuggingFace format: a WordPiece
tokenizer trained on the policy text and a one-layer BERT whose word embeddings are
signed hashes of the sub-word strings and whose attention/feed-forward outputs are
zeroed, so with knowlytix's mean pooling it is a deterministic bag-of-sub-words
hashing encoder (typo-robust through sub-words, no semantic prior). Every report
records which encoders a store was built with.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
from pathlib import Path
from typing import Any

HF_V = "sentence-transformers/all-MiniLM-L6-v2"
HF_U = "sentence-transformers/nli-mpnet-base-v2"
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
    """True when the knowlytix family is installed and a licence is configured."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if os.environ.get("GMSJOURNEY_BACKEND", "auto") == "builtin":
        return False
    if importlib.util.find_spec("knowlytix") is None or importlib.util.find_spec("knowlytix.core") is None:
        return False
    return _load_license_env()


def require() -> None:
    if not available():
        raise ImportError("knowlytix backend unavailable: `pip install knowlytix` (Python 3.12) and set "
                          "KNOWLYTIX_LICENSE_KEY (or ~/.knowlytix/license.key)")


def describe() -> dict[str, Any]:
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


# ----------------------------------------------------------------------------- encoders
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
    if (_hf_cached(HF_V) and _hf_cached(HF_U)) or _hub_reachable():
        return "huggingface"
    return "offline-stand-in"


def resolve_encoders(corpus: list[str], mode: str | None = None) -> dict[str, Any]:
    """{'v': name_or_path, 'u': name_or_path, 'kind': 'huggingface' | 'offline-stand-in', 'reason': ...}"""
    mode = mode or os.environ.get("GMSJOURNEY_ENCODER", "auto")
    if mode in ("auto", "hf"):
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
