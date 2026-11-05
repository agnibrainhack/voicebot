"""The policy oracle: expected behaviour, computed rather than annotated (Ch. 12.4).

The oracle is an executable reading of the policy graph: ``PolicyOracle`` is ``flow.GraphOracle``, which
walks the graph's dialog states (``leads_to`` with its executable qualifiers), reads every number from the
exact numeric register (lookup_enm) and plays the journey against the user simulator on the same backend
fixture the bot will face. The resulting ``Trace`` is the ground truth: outcome signature, per-stage
decisions, which rules fired and which workflow edges were traversed.

The oracle consumes semantic frames only. It never reads text, so its answer depends on what the customer
*meant*, which is exactly what the invariance proof (enrich/invariance.py) shows enrichment leaves unchanged.
"""
from __future__ import annotations

from typing import Any


class Limits:
    """Every limit the policy graph declares (``(key, is_a, limit)``), read exactly from the store's ENM.

    Attributes are the graph's limit keys (``limits.max_turns``, ``limits.lookback_days`` ...); an attribute
    may be overridden on a copy (the turn-limit generator plays journeys with ``max_turns`` lifted)."""

    def __init__(self, values: dict[str, Any]):
        self.__dict__.update(values)

    @classmethod
    def from_store(cls, store) -> "Limits":
        kg = getattr(store, "kg", None)
        keys = [t["head"] for t in kg.where("is_a", tail="limit")] if kg is not None else \
            [k for (c, k) in store.enm if c == "limit"]
        return cls({k: store.lookup_enm("limit", k) for k in keys})

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __repr__(self) -> str:
        return f"Limits({self.__dict__})"


from .flow import GraphOracle as PolicyOracle  # noqa: E402,F401  (flow imports Limits from here, lazily)
