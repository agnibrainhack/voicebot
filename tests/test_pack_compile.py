"""A compact policy graph compiled into the full graph walks the same journeys as its own interpreter.

For every journey the compact engine's miner finds (``graph/journeys.py``: every path with up to two deviations,
plus failure perturbations and policy questions), the compact interpreter (``graph/machine.py``) plays it with
its simulated customer; the compiled flow (``flow.GraphOracle`` over ``knowledge/pack_compile.py``) replays the
same customer frames against the same tool script. Bot acts, tool calls, rules exercised and the ending must be
identical.

Charge review's compact rewrite (charge_review_graph) loops over the transactions (an each-state: filter by the review
window, cap, per-item counters, @next) and summarises what was recorded before a transfer.

    python tests/test_pack_compile.py [pack ...]          (default: both compact packs)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmsjourney import domain  # noqa: E402


def _events(tr) -> list:
    out = []
    for e in tr.events:
        if e["role"] == "bot":
            out.append(("bot", tuple(sorted((k, str(v)) for a in e["acts"] for k, v in a.items()))))
        elif e["role"] == "user":
            out.append(("user", e["frame"].canonical()))
        elif e["role"] == "tool":
            out.append(("tool", e["tool"], tuple(sorted((e.get("args") or {}).items())),
                        str((e.get("result") or {}).get("status"))))
    return out


def check_pack(pack: str, limit: int | None = None) -> dict:
    dom = domain.set_current(pack)
    from gmsjourney.flow import GraphOracle as FlowOracle
    from gmsjourney.graph.compile import compile_graph
    from gmsjourney.graph.journeys import build_fixture, mine_paths, perturbations
    from gmsjourney.graph.kb import kb_frames
    from gmsjourney.graph.machine import GraphOracle as MachineOracle
    store = compile_graph(dom.graph_path).store
    store.kg = dom.kg
    machine, flow = MachineOracle(store), FlowOracle(store)
    paths = mine_paths(machine)
    specs = [s for s, _ in paths] + [s for s, _ in perturbations(machine, paths, kb_frames(dom.graph))]
    if limit:
        specs = specs[:limit]
    bad = []
    for spec in specs:
        a = machine.run(spec)
        b = flow.run(spec, fixture=build_fixture(spec), frames_override=list(a.user_frames))
        diffs = []
        if _events(a) != _events(b):
            diffs.append("events")
        if a.rules_fired != b.rules_fired:
            diffs.append(f"rules {sorted(a.rules_fired ^ b.rules_fired)}")
        if (a.ended, a.turns) != (b.ended, b.turns):
            diffs.append(f"ended/turns {(a.ended, a.turns)} vs {(b.ended, b.turns)}")
        if [c["tool"] for c in a.tool_log] != [c["tool"] for c in b.tool_log]:
            diffs.append("tool log")
        if diffs:
            bad.append((spec, diffs, a, b))
    return {"pack": pack, "journeys": len(specs), "different": bad}


def _show(bad) -> str:
    out = []
    for spec, diffs, a, b in bad[:3]:
        out.append(f"{spec!r}: {diffs}")
        ea, eb = _events(a), _events(b)
        for i, (x, y) in enumerate(zip(ea + [None] * (len(eb) - len(ea)), eb + [None] * (len(ea) - len(eb)))):
            if x != y:
                out.append(f"   first difference at event {i}:\n     machine: {x}\n     flow:    {y}")
                break
    return "\n".join(out)


def test_supplementary_card_compiles_to_the_same_journeys():
    r = check_pack("supplementary_card")
    assert r["journeys"] > 50, r["journeys"]
    assert not r["different"], f"{len(r['different'])} of {r['journeys']} journeys differ:\n" + _show(r["different"])


def test_a_looped_graph_compiles_to_the_same_journeys():
    r = check_pack("charge_review_graph")
    assert r["journeys"] > 200, r["journeys"]
    assert not r["different"], f"{len(r['different'])} of {r['journeys']} journeys differ:\n" + _show(r["different"])


if __name__ == "__main__":
    packs = sys.argv[1:] or ["supplementary_card", "charge_review_graph"]
    for p in packs:
        r = check_pack(p)
        print(f"{p}: {r['journeys']} journeys, {len(r['different'])} different")
        if r["different"]:
            print(_show(r["different"]))
    domain.set_current("charge_review")
