"""The configured journey factors (project.yaml + journey.py) reproduce the hand-written ones they replaced.

tests/legacy_journey.py is the hand-written module, kept only as this reference. Checked on random cases:
relevance, validity, the bank fixture, the case key, the customer's answers to fuzzed bot acts (as a real
bot might send them) and the oracle's walk with each customer. Runs on the symbolic store.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import legacy_journey as LJ  # noqa: E402

from gmsjourney.s04_cases import journey as NJ  # noqa: E402
from gmsjourney.s02_graph.flow import GraphOracle  # noqa: E402
from gmsjourney.s04_cases.meaning import SemanticFrame as F  # noqa: E402
from gmsjourney.s08_campaign.pipeline import POLICY  # noqa: E402
from gmsjourney.s03_knowledge.store import symbolic_store  # noqa: E402

KB = [F.make("ASK_KB", intent="value", attr="fee", entity="expedited_delivery"), F.make("ASK_KB", intent="fraud")]


def _cases(n, seed, valid_only=False):
    rng, out = random.Random(seed), []
    while len(out) < n:
        d = {k: rng.choice(v) for k, v in LJ.DIMENSIONS.items()}
        if not valid_only or LJ.is_valid(LJ.JourneySpec(**d)):
            out.append(d)
    return out


def test_same_factors_relevance_validity_fixture_and_key():
    assert LJ.DIMENSIONS == NJ.DIMENSIONS
    for d in _cases(20000, 1):
        o, n = LJ.JourneySpec(**d), NJ.JourneySpec(**d)
        assert LJ.relevant_dims(o) == NJ.relevant_dims(n), d
        assert LJ.is_valid(o) == NJ.is_valid(n), d
        assert LJ.build_fixture(o) == NJ.build_fixture(n), d
        assert o.key() == n.key(), d


def test_same_customer_answers():
    rng = random.Random(2)
    other = ["ASK_AUTH", "ASK_AUTH_RETRY", "ASK_MORE_HELP", "EXPLAIN_NO_TXNS", "END", "OPENING", "TRANSFER"]
    for d in _cases(800, 3, valid_only=True):
        kb = tuple(rng.sample(KB, rng.randint(0, 2)))
        o, n = LJ.JourneySpec(**d, kb_frames=kb), NJ.JourneySpec(**d, kb_frames=kb)
        fx = LJ.build_fixture(o)
        so, sn = LJ.UserSimulator(o, fx, 30), NJ.UserSimulator(n, fx, 30)
        tids = [t["transaction_id"] for t in fx["transactions"]] + [None, "T7"]
        for _ in range(30):
            r = rng.random()
            a = ({"act": rng.choice(["PRESENT_TXN", "PROVIDE_CONTEXT", "ASK_CLARIFY", "ANSWER_KB"]), "txn": rng.choice(tids)}
                 if r < .45 else {"act": rng.choice(["CONFIRM_ACTION", "DISCLOSE_FEE"]),
                                  "delivery": rng.choice(["expedited_delivery", "standard_delivery", None])}
                 if r < .65 else {"act": rng.choice(other)})
            assert so.respond(dict(a)) == sn.respond(dict(a)), (d, a)


def test_same_oracle_walk_with_either_customer():
    oracle = GraphOracle(symbolic_store(POLICY))
    rng = random.Random(4)
    for d in _cases(400, 5, valid_only=True):
        kb = tuple(rng.sample(KB, rng.randint(0, 1)))
        o, n = LJ.JourneySpec(**d, kb_frames=kb), NJ.JourneySpec(**d, kb_frames=kb)
        fx = LJ.build_fixture(o)
        a = oracle.run(o, fx, responder=LJ.UserSimulator(o, fx, 30).respond)
        c = oracle.run(n, NJ.build_fixture(n))
        assert a.signature() == c.signature() and a.rules_fired == c.rules_fired, d
        assert [e for e in a.events if e["role"] == "bot"] == [e for e in c.events if e["role"] == "bot"], d


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
