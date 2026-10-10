"""Geometric judging of policy answers with knowlytix's GMSJudge via proofloop's GeometricJudge.

proofloop.testing.judge.GeometricJudge wraps ``knowlytix.harness.testing.judge.GMSJudge``
and maps the mean geodesic of the answer's claims to the book's four hallucination
bands (grounded < 0.5 <= embellishment < 1.0 <= distortion < 1.5 <= fabrication).
Exactness (byte-exact ENM numbers) stays with kb_truth.judge_kb_answer; this adds the
continuous geometric signal next to it.
"""
from __future__ import annotations

from typing import Any


class KBGeometricJudge:
    def __init__(self, store):
        from ..knowledge.kx_store import _quiet
        with _quiet():
            from proofloop.testing.judge import GeometricJudge
            self.impl = GeometricJudge(store.kx)
        self.store = store

    def judge(self, answer: str, truth, ok: bool, detail: str) -> dict[str, Any]:
        from ..knowledge.kx_store import _quiet
        from ..knowledge.kx_view import kx_triple
        meta: dict[str, Any] = {"exact_correct": bool(ok), "exact_detail": detail}
        if truth.answer_triples:
            h, r, t = kx_triple(*truth.answer_triples[0])
            meta.update(head=h, relation=r, tail=t)
        try:
            with _quiet():
                v = self.impl.judge(answer or "", str(truth.answer), meta)
        except Exception as e:                     # noqa: BLE001 - a judge failure never fails a run
            return {"geo_error": str(e)[:160]}
        return {"geo_label": v.label, "geo_distance": round(v.geodesic, 4), "geo_confidence": round(v.confidence, 4),
                "geo_tension": round(v.tension, 4)}
