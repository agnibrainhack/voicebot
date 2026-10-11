"""CSV exports for the LLM enrichment arm (one row per customer turn, controls audit, arm comparison)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _w(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return 0
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})
    return len(rows)


def write_llm_exports(out_dir: str | Path, proofs, controls: dict[str, Any] | None,
                      paired: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    turns = []
    for p in proofs:
        for t in p.turns:
            fr = t.get("factor_realisation") or {}
            turns.append({
                "sid": p.sid, "base": p.qid, "scenario_accepted": p.accepted, "turn": t["turn"], "state": t["state"],
                "meaning_card": t["frame"], "base_text": t.get("base_text"), "template_text": t.get("template_text"),
                "llm_raw": t.get("llm_raw"), "llm_final": t.get("enriched_text"),
                "rule_reader_card": t.get("read_back"), "independent_reader_card": t.get("dual_read"),
                "turn_accepted": t["equivalent"], **{f"gate_{k}": v for k, v in (t.get("gates") or {}).items()},
                "reasons": t.get("reasons"), "attempts": len(t.get("attempts", [])),
                "surface_cosine_to_base": round(t.get("surface_cosine", 0.0), 4),
                "surface_cosine_to_template": round(t.get("template_cosine", 0.0), 4) if "template_cosine" in t else None,
                "factors_requested": {k: v["level"] for k, v in fr.items()},
                "factors_realised": sorted(k for k, v in fr.items() if v["detected"]),
                "factors_not_realised": sorted(k for k, v in fr.items() if v["detected"] is False),
                "levels": p.levels, "realised_levels": p.extra.get("realised_levels"),
            })
    res = {"llm_turns": _w(out / "llm_enrichment_turns.csv", turns)}
    if controls:
        res["llm_controls_audit"] = _w(out / "llm_negative_controls_audit.csv", controls.get("_audit", []))
    if paired:
        res["arm_pairs"] = _w(out / "arm_pairs.csv", paired)
    return res
