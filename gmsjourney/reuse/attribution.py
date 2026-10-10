"""Factor attribution with knowlytix's DOEAnalyzer (book Ch. 10; GraphDOE).

``DOEAnalyzer.from_dataframe`` over the run rows: per-factor logistic analysis of
deviance (G2, p, McFadden pseudo-R2, odds ratios), Benjamini-Hochberg correction,
the joint model (pseudo-R2, AIC, BIC, AUC, Hosmer-Lemeshow), two-way interaction
tests and the failure table - the four tables the book prints.
"""
from __future__ import annotations

from typing import Any


def _records(obj) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict(orient="records")
        except TypeError:
            return obj.to_dict()
    return obj


def kx_attribution(rows: list[dict], factors: list[str], metric: str = "correct", alpha: float = 0.05,
                   top_pairs: int = 6) -> dict[str, Any]:
    import pandas as pd
    from ..knowledge.kx_store import _quiet
    with _quiet():
        from knowlytix.harness.graphdoe import DOEAnalyzer
    df = pd.DataFrame([{**{f: r.get(f"f_{f}") for f in factors}, metric: int(r[metric])} for r in rows])
    out: dict[str, Any] = {"engine": "knowlytix.harness.graphdoe.DOEAnalyzer", "n": len(df)}
    if df[metric].nunique() < 2:
        out["note"] = f"{metric} is constant ({df[metric].iloc[0]}); nothing to attribute"
        return out
    a = DOEAnalyzer.from_dataframe(df, factors=factors)
    with _quiet():
        table = a.run_logistic(metric=metric)
        try:
            table = a.apply_fdr_correction(table, alpha=alpha)
        except Exception as e:                     # noqa: BLE001
            out["fdr_error"] = str(e)[:200]
        out["logistic"] = _records(table)
        try:
            out["joint_model"] = {k: v for k, v in (a.run_joint_model(metric=metric) or {}).items()
                                  if isinstance(v, (int, float, str, bool, type(None)))}
        except Exception as e:                     # noqa: BLE001
            out["joint_model_error"] = str(e)[:200]
        try:
            ranked = sorted(out["logistic"], key=lambda r: r.get("p_value", 1.0))[:4]
            pairs = [(x["factor"], y["factor"]) for i, x in enumerate(ranked) for y in ranked[i + 1:]][:top_pairs]
            out["interactions"] = _records(a.run_interaction_logistic(factor_pairs=pairs, metric=metric)) if pairs else []
        except Exception as e:                     # noqa: BLE001
            out["interactions_error"] = str(e)[:200]
        try:
            out["failure_table"] = _records(a.failure_analysis())
        except Exception as e:                     # noqa: BLE001
            out["failure_table_error"] = str(e)[:200]
    return out
