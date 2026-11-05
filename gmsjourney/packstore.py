"""The symbolic store of the active domain pack: compiled straight from its graph in about a second, no GMS
training and no licence. Previews, the studio, factor relevance and the parity tests run on it; the campaign
runs on the knowlytix GMS built from the same graph."""
from __future__ import annotations

from typing import Any

_CACHE: dict = {}


def symbolic_store(policy_graph=None, kg=None) -> Any:
    """The store compiled from a policy graph (default: the active pack's), with ``store.kg`` the full graph and a
    symbolic stance check (the claimed stance equals the rule's, or not; the GMS reads tension instead)."""
    from .domain import current
    from .knowledge.geode import build_store
    dom = current()
    path = policy_graph or dom.graph_path
    b = build_store(path)
    store = getattr(b, "store", b)
    if kg is not None:
        store.kg = kg
    elif str(path) == str(dom.graph_path):
        store.kg = dom.kg
    elif getattr(store, "kg", None) is None:
        from .knowledge.policy_kg import PolicyKG
        store.kg = PolicyKG.load(path)

    def stance_check(subject, predicate, obj, polarity):
        rule = next((r for r in store.rules.values() if (r.subject, r.predicate, r.obj) == (subject, predicate, obj)), None)
        if rule is None:
            return {"verdict": "uncertain", "energy": None, "rule": None}
        claimed = "must" if polarity > 0 else "must_not"
        stance = "must" if rule.stance == "may" else rule.stance
        return {"verdict": "supported" if claimed == stance else "contradicted", "energy": None, "rule": rule.rule_id}

    if not hasattr(store, "stance_check"):
        store.stance_check = stance_check
    return store


def pack_store() -> Any:
    """The active pack's symbolic store, built once per pack."""
    from .domain import current
    dom = current()
    key = (str(dom.dir), id(dom))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = symbolic_store()
    return _CACHE[key]


def pack_oracle():
    """The graph oracle on the active pack's symbolic store (cached with it)."""
    store = pack_store()
    if getattr(store, "_flow_oracle", None) is None:
        from .flow import GraphOracle
        store._flow_oracle = GraphOracle(store)
    return store._flow_oracle


def _reset() -> None:
    _CACHE.clear()


from .domain import on_switch as _on_switch  # noqa: E402

_on_switch(_reset)
