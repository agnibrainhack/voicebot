"""The graph-walking oracle (gmsjourney/graph/flow.py) reproduces the hand-written one it replaced.

The hand-written oracle (tests/legacy_oracle.py) is kept only as this reference. Every trace field the
scoring uses must match: bot acts turn by turn, customer frames, tool calls with arguments and results,
rules exercised, what was presented, recorded and claimed, KB answers, turns, end state.
"""
from __future__ import annotations

import json
import os
import random
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmsjourney.cases.journey import DIMENSIONS, JourneySpec, build_fixture, is_valid
from gmsjourney.cases.meaning import SemanticFrame as F
from gmsjourney.graph.flow import GraphOracle as PolicyOracle
from gmsjourney.bots.tools import TOOLS, FaultProfile, ToolGateway

from _store import kx_store
from legacy_oracle import LegacyOracle

KB = [F.make("ASK_KB", intent="value", attr="fee", entity="expedited_delivery"),
      F.make("ASK_KB", intent="limit", key="lookback_days"), F.make("ASK_KB", intent="fraud")]


@lru_cache(maxsize=1)
def oracles():
    store = kx_store()
    return LegacyOracle(store), PolicyOracle(store)


def _norm(tr):
    ev = []
    for e in tr.events:
        if e["role"] == "user":
            ev.append(("user", e["frame"].canonical()))
        elif e["role"] == "bot":
            ev.append(("bot", json.dumps(e["acts"], sort_keys=True, default=str)))
        else:
            ev.append(("tool", e["tool"], json.dumps(e.get("args"), sort_keys=True),
                       json.dumps(e.get("result"), sort_keys=True, default=str), e.get("ok", True)))
    return {"events": ev, "rules": sorted(tr.rules_fired), "presented": tr.presented,
            "recorded": sorted(tr.claims_recorded.items()), "actions": tr.claims_actions,
            "kb": [(k["frame"].canonical(), str(k["answer"])) for k in tr.kb], "turns": tr.turns, "ended": tr.ended,
            "notes": tr.notes, "signature": tr.signature(), "components": tr.components()}


def _specs(n, seed=3):
    rng, out = random.Random(seed), []
    while len(out) < n:
        sp = JourneySpec(**{k: rng.choice(v) for k, v in DIMENSIONS.items()})
        if is_valid(sp):
            if rng.random() < 0.4:
                sp = sp.with_(kb_frames=tuple(rng.sample(KB, rng.randint(1, 2))))
            out.append(sp)
    return out


def test_same_journeys_as_the_hand_written_oracle():
    old, new = oracles()
    for sp in _specs(400):
        assert _norm(old.run(sp)) == _norm(new.run(sp)), sp.dims()


def test_same_journeys_under_tool_faults():
    old, new = oracles()
    for sp in _specs(30, seed=5):
        fx = build_fixture(sp)
        for tool in TOOLS:
            for kind in ("error", "stale", "plausible_wrong"):
                gw = lambda env, t=tool, k=kind: ToolGateway(env, [FaultProfile(tool_pattern=t, kind=k)])  # noqa: E731
                assert _norm(old.run(sp, fx, gateway_factory=gw)) == _norm(new.run(sp, fx, gateway_factory=gw)), \
                    (sp.dims(), tool, kind)


def test_same_journeys_on_replayed_frames():
    old, new = oracles()
    for sp in _specs(25, seed=7):
        fx = build_fixture(sp)
        frames = old.run(sp, fx).user_frames
        for i in range(len(frames)):
            fr = list(frames)
            fr[i] = F.make("UNCLEAR")
            assert _norm(old.run(sp, fx, frames_override=fr)) == _norm(new.run(sp, fx, frames_override=fr))


def test_the_flow_is_read_from_the_graph():
    """Disable one transition in the graph and the walk changes: the oracle has no flow of its own."""
    import copy
    from gmsjourney.graph.flow import FlowSpec, GraphOracle
    from gmsjourney.graph.triples import PolicyKG
    store = kx_store()
    kg = PolicyKG(copy.deepcopy(store.kg.data))
    for t in kg.triples:
        if t["relation"] == "leads_to" and t["head"] == "CONFIRM_RECORDED" and t["tail"] == "EXPLAIN_LIMIT":
            t["qualifiers"]["guard"] = "False"
    changed = GraphOracle(store, spec=FlowSpec.from_kg(kg)).run(JourneySpec(n_txn=5))
    assert oracles()[1].run(JourneySpec(n_txn=5)).transfer_reason == "transaction_limit"
    assert changed.transfer_reason != "transaction_limit"


if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (not only or any(o in name for o in only)):
            fn()
            print("ok", name)
