"""Reusing meaning proofs from earlier runs (campaign step 5, the template arm).

Proving a reworded case reads every reworded turn back into a meaning card and replays the read-back cards through
the oracle. The result depends only on: the case (its spec, mock-bank fixture, answer key and the factor levels of
the rewording), the journey's own files (graph, domain lexicons, project factors) and the code that words, reads and
replays it. ``ProofCache`` keys every proof on exactly those, so a later run with the same inputs takes the proof as
it was, and anything that could change it (an edited graph or lexicon, a new reader, another case) misses the cache
and is proved again.

Analogy: a lab keeps the certificate of every sample it has tested, filed under the sample's full description and
the test method's version; a sample that arrives again under the same description, tested by the same method, gets
the filed certificate instead of a second test.

    ProofCache(root, dom, validator, reuse=True)
    cache.prove(scn)            the filed proof (and its side effects on the scenario), or a new proof filed
    cache.stats()               {"reused", "proved", "dir"}

Each proof is one JSON file under ``<root>/proofs/<context>/``: the context is the validator, the journey's files and
the code; the file name is the scenario's content hash. Proofs are always filed; ``reuse=False`` only stops reading
them (every case is proved again, and the files are refreshed).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..s04_cases.meaning import SemanticFrame, _freeze, _thaw

PROOF_CACHE_VERSION = "proof-cache/1"
_PKG = Path(__file__).resolve().parents[1]
CODE_DIRS = ("s01_packs", "s02_graph", "s03_knowledge", "s04_cases", "s05_rewording")


def _sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\0")
    return h.hexdigest()


def code_fingerprint() -> str:
    """The code that words, reads and replays a case (every module of the packs, graph, knowledge, cases and
    rewording stages)."""
    h = hashlib.sha256()
    for d in CODE_DIRS:
        for f in sorted((_PKG / d).rglob("*.py")):
            h.update(str(f.relative_to(_PKG)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:20]


def pack_fingerprint(dom) -> str:
    """The journey's own files (graph, domain lexicons, project factors), not the conversations saved in it."""
    h = hashlib.sha256()
    root = Path(dom.dir)
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in (".yaml", ".yml", ".json") and f.name != "saved_conversations.json":
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:20]


def scenario_key(scn) -> str:
    """Everything about one reworded case its proof depends on."""
    b = scn.base
    return _sha(PROOF_CACHE_VERSION, scn.sid, dict(scn.factor_levels), b.qid, b.answer, b.components,
                b.metadata.get("spec"), b.metadata.get("fixture"))


# ------------------------------------------------------------------------------------------- the files
def _enc_frame(f) -> Any:
    if f is None:
        return None
    return {"act": f.act, "slots": [[k, _thaw(v)] for k, v in f.slots]}


def _dec_frame(d) -> Any:
    if d is None:
        return None
    return SemanticFrame(d["act"], tuple((k, _freeze(v)) for k, v in d["slots"]))


def encode(res, metadata: dict) -> str:
    """A proof and the scenario metadata it set, as JSON (TypeError when something cannot be filed)."""
    d = {k: getattr(res, k) for k in res.__dataclass_fields__ if k != "turns"}
    d["turns"] = [{**{k: v for k, v in t.items() if k not in ("frame_obj", "read_obj")},
                   "frame_obj": _enc_frame(t.get("frame_obj")), "read_obj": _enc_frame(t.get("read_obj"))}
                  for t in res.turns]
    meta = {k: metadata[k] for k in ("seed_overrides", "proof") if k in metadata}
    meta["seed_overrides"] = {str(k): v for k, v in (meta.get("seed_overrides") or {}).items()}
    return json.dumps({"version": PROOF_CACHE_VERSION, "result": d, "metadata": meta})


def decode(text: str):
    from .proof import ProofResult
    d = json.loads(text)
    r = dict(d["result"])
    r["turns"] = [{**t, "frame_obj": _dec_frame(t.get("frame_obj")), "read_obj": _dec_frame(t.get("read_obj"))}
                  for t in r["turns"]]
    meta = d.get("metadata") or {}
    meta["seed_overrides"] = {int(k): v for k, v in (meta.get("seed_overrides") or {}).items()}
    return ProofResult(**r), meta


class ProofCache:
    """Proofs filed by everything they depend on (see the module docstring)."""

    def __init__(self, root: str | Path, dom, validator, reuse: bool = True):
        self.validator, self.reuse = validator, reuse
        ctx = _sha(PROOF_CACHE_VERSION, type(validator).__name__, dom.dir.name, pack_fingerprint(dom),
                   code_fingerprint())[:24]
        self.dir = Path(root) / "proofs" / ctx
        self.reused = self.proved = 0

    def prove(self, scn):
        f = self.dir / f"{scenario_key(scn)[:40]}.json"
        if self.reuse and f.exists():
            try:
                res, meta = decode(f.read_text(encoding="utf-8"))
                scn.metadata.update(meta)
                self.reused += 1
                return res
            except (ValueError, KeyError, TypeError):
                pass                                   # an unreadable file: prove again and refile it
        res = self.validator.prove(scn)
        self.proved += 1
        try:
            text = encode(res, scn.metadata)
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(f)
        except (TypeError, ValueError, OSError):
            pass                                       # not fileable: this proof is simply not reused
        return res

    def stats(self) -> dict[str, Any]:
        return {"reused": self.reused, "proved": self.proved, "reuse": self.reuse, "dir": str(self.dir)}
