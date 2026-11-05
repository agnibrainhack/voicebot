"""Loops in journey generation: a compact graph that loops over a returned list (charge_review_graph reviews the
transactions one at a time) gets loop factors in its drafted project, and each level plays as the graph says.

  item count     one, two, at the cap, over the cap (the limit transfer after the cap is reviewed)
  window         the last item on the window's edge (reviewed), one more just outside it (skipped)
  answers        the second item answered differently (relevant only when there is a second item)

    python tests/test_loops.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmsjourney import domain  # noqa: E402

PACK = "charge_review_graph"


def _setup():
    dom = domain.set_current(PACK)
    from gmsjourney import journey as J
    from gmsjourney import project as PJ
    from gmsjourney.packstore import pack_oracle
    PJ.reset_cache()
    proj = PJ.project()
    J.activate(proj)
    return dom, PJ, J, proj, pack_oracle()


def _writes(tr, tool):
    return [(c["args"].get("transaction_id"), c["args"].get("disposition")) for c in tr.tool_log
            if c["tool"] == tool and c.get("ok", True) and (c.get("result") or {}).get("status") == "SUCCESS"]


def test_the_drafted_project_has_loop_factors_and_is_valid():
    dom, PJ, J, proj, _ = _setup()
    from gmsjourney.names import graph_names
    from gmsjourney.project_draft import draft
    raw = draft(dom)
    names = {f["name"]: f for f in raw["factors"]}
    assert {"transactions_count", "transactions_window", "transactions_answers"} <= set(names), list(names)
    assert set(names["transactions_count"]["levels"]) == {"one", "two", "at_cap_4", "over_cap_5"}
    loop_prompts = [s for s, sc in raw["customer"]["scripts"].items() if sc.get("per_item")]
    assert "present_txn" in loop_prompts, loop_prompts
    assert not PJ.check(proj, graph_names()), PJ.check(proj, graph_names())


def test_each_loop_level_plays_as_the_graph_says():
    dom, PJ, J, proj, oracle = _setup()
    rec = "record_charge_response"

    def play(**kw):
        spec = J.JourneySpec(**kw)
        return oracle.run(spec, J.build_fixture(spec), responder=J.UserSimulator(spec, J.build_fixture(spec), None).respond)

    assert _writes(play(), rec) == [("T1", "RECOGNIZED")]
    assert [t for t, _ in _writes(play(transactions_count="two"), rec)] == ["T1", "T2"]
    assert len(_writes(play(transactions_count="at_cap_4"), rec)) == 4
    over = play(transactions_count="over_cap_5")
    assert len(_writes(over, rec)) == 4 and over.transfer_reason == "transaction_limit", over.transfer_reason
    edge = play(transactions_count="two", transactions_window="last_at_30")
    assert [t for t, _ in _writes(edge, rec)] == ["T1", "T2"]                 # 30 days: inside the window
    beyond = play(transactions_count="two", transactions_window="one_more_at_31")
    assert [t for t, _ in _writes(beyond, rec)] == ["T1", "T2"]               # the 31-day one is never reviewed
    mixed = play(transactions_count="two", transactions_answers="second_unrecognized")
    assert _writes(mixed, rec) == [("T1", "RECOGNIZED"), ("T2", "UNRECOGNIZED")]
    every = play(transactions_count="two", present_txn_disposition="unrecognized")
    assert {d for _, d in _writes(every, rec)} == {"UNRECOGNIZED"}           # item: all - every item answered so


def test_a_per_item_level_is_relevant_only_when_its_item_exists():
    dom, PJ, J, proj, _ = _setup()
    one = J.relevant_dims(J.JourneySpec(transactions_answers="second_unrecognized"))
    two = J.relevant_dims(J.JourneySpec(transactions_count="two", transactions_answers="second_unrecognized"))
    assert one["transactions_answers"] == J.NA and two["transactions_answers"] != J.NA


if __name__ == "__main__":
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok", name)
    finally:
        domain.set_current("charge_review")
