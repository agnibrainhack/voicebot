"""Embedding channels for the geometric store (book Ch. 3.1).

v-space (semantic) and u-space (logical) need starting vectors. The book warm-
starts both from sentence-transformer encoders ("Mode A"). This module provides

* ``HashingEncoder`` - an offline, deterministic encoder built from signed
  feature hashing of character n-grams and words, with a dedicated numeric block
  so that literals such as "4" and "5" land at a measurable, graded distance.
  Character n-grams make it robust to typos and ASR-style misspellings, which is
  what entity binding needs.
* ``SentenceTransformerEncoder`` - optional Mode A, used automatically when the
  ``sentence-transformers`` package is installed and requested.

Every vector is L2-normalised, so all geometry lives on the unit hypersphere.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

import numpy as np

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _h(token: str, salt: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{salt}|{token}".encode(), digest_size=8).digest(), "little")


def normalize_text(s: str) -> str:
    s = s.replace("_", " ").replace("-", " ").lower()
    s = re.sub(r"[^a-z0-9$.%' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class HashingEncoder:
    """Signed feature-hashing encoder with a numeric block.

    Parameters
    ----------
    dim : total dimension (text block + numeric block).
    num_dim : size of the numeric block.
    salt : channel salt, so v-space and u-space get independent hash families.
    """

    def __init__(self, dim: int = 512, num_dim: int = 64, salt: str = "v"):
        self.dim, self.num_dim, self.salt = dim, num_dim, salt
        self.text_dim = dim - num_dim

    # -- features -----------------------------------------------------------
    def _text_features(self, s: str) -> np.ndarray:
        v = np.zeros(self.text_dim)
        t = normalize_text(s)
        words = t.split()
        feats: list[tuple[str, float]] = []
        for w in words:
            if _NUM_RE.fullmatch(w):
                feats.append(("<num>", 0.3))
                continue
            feats.append((f"w:{w}", 1.0))
            padded = f"#{w}#"
            for n in (3, 4, 5):
                for i in range(max(1, len(padded) - n + 1)):
                    feats.append((f"c{n}:{padded[i:i + n]}", 0.5))
        for a, b in zip(words, words[1:]):
            feats.append((f"b:{a}_{b}", 0.7))
        for tok, w in feats:
            hv = _h(tok, self.salt)
            v[hv % self.text_dim] += w if (hv >> 63) & 1 else -w
        return v

    def _numeric_features(self, s: str) -> np.ndarray:
        v = np.zeros(self.num_dim)
        nums = _NUM_RE.findall(s.replace(",", ""))
        for raw in nums:
            x = float(raw)
            # exact identity: two different numbers are nearly orthogonal here
            hv = _h(f"num={x:g}", self.salt)
            v[hv % self.num_dim] += 2.0 if (hv >> 63) & 1 else -2.0
            # smooth part: log-scaled phase features give a graded distance
            z = math.copysign(math.log1p(abs(x)), x)
            half = self.num_dim // 4
            for k in range(half):
                f = 0.6 * (1.5 ** k)
                v[k] += 0.35 * math.sin(f * z)
                v[half + k] += 0.35 * math.cos(f * z)
        return v

    @lru_cache(maxsize=20000)
    def _encode_cached(self, s: str) -> tuple[float, ...]:
        vec = np.concatenate([self._text_features(s), self._numeric_features(s)])
        n = np.linalg.norm(vec)
        if n == 0:
            hv = _h("<empty>", self.salt)
            vec[hv % self.dim] = 1.0
            n = 1.0
        return tuple(vec / n)

    def encode(self, s: str) -> np.ndarray:
        return np.asarray(self._encode_cached(str(s)))

    def encode_many(self, items: list[str]) -> np.ndarray:
        return np.stack([self.encode(x) for x in items]) if items else np.zeros((0, self.dim))


class SentenceTransformerEncoder:
    """Mode A warm start (book Ch. 3.1). Optional dependency."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, s: str) -> np.ndarray:
        if s not in self._cache:
            v = self.model.encode([str(s).replace("_", " ")], normalize_embeddings=True)[0]
            self._cache[s] = np.asarray(v, dtype=float)
        return self._cache[s]

    def encode_many(self, items: list[str]) -> np.ndarray:
        return np.stack([self.encode(x) for x in items]) if items else np.zeros((0, self.dim))


def make_encoder(kind: str = "hashing", salt: str = "v", model: str | None = None):
    if kind == "sentence-transformers":
        try:
            from ..backend import HF_V, local_encoders
            loc = local_encoders()
            return SentenceTransformerEncoder(model or (loc[0] if loc else HF_V))
        except Exception:  # pragma: no cover - optional path
            pass
    return HashingEncoder(salt=salt)


def geodesic(a: np.ndarray, b: np.ndarray) -> float:
    """Great-circle distance on the unit sphere, in [0, pi]."""
    return float(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
