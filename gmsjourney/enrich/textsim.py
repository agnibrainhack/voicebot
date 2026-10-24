"""Surface similarity of two wordings: a reported diagnostic of the invariance proof.

How far an enriched turn's wording moved from its base wording ("surface cosine") is
reported next to every proof; it never decides anything. It is computed with a small,
deterministic signed feature-hashing encoder (character n-grams, words, bigrams and a
numeric block), so it is cheap enough for every turn of every scenario and gives the
same numbers on every machine. It is not part of the GMS: the store's encoders are
knowlytix's (knowledge/kx_store.py).
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


_ENC = HashingEncoder(salt="v")


def surface_cosine(a: str, b: str) -> float:
    """Cosine similarity of two wordings (1.0 = same surface)."""
    va, vb = _ENC.encode(a), _ENC.encode(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(np.dot(va, vb) / (na * nb)) if na and nb else 0.0
