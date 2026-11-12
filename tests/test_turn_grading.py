"""How a bot turn is graded against the graph's turn, and the order a turn is shown in (s06_bots/turns.py,
s06_bots/harness.py JourneyRunner, s02_graph/pending.py, s07_checks/speech_checks.py).

* a side step (a policy answer, a clarification) that differs flags only its turn ("side"); the run goes on;
* a bot that asks again the question just answered gets the same line once more ("asked_again");
* one turn off the journey is tolerated, two in a row end the run (``extras["stopped"]``);
* a turn's tool calls and acts are kept in the order they happened, for the graph and for the bot;
* a speech failure inside a side step leaves ``c_speech`` alone and fails ``c_speech_strict``;
* "anything else?" answered unclearly is clarified by the graph (R22), not taken as a goodbye.

No knowledge store build, no gateway: the Playground's bench.

    python tests/test_turn_grading.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gmsjourney.s01_packs import loading as domain  # noqa: E402

SIDE = {"ANSWER_KB", "ASK_CLARIFY"}


def _bench():
    from gmsjourney.s10_console import bots as PG
    domain.set_current("charge_review_graph")
    return PG, PG.bench()


def _runner(fac):
    from gmsjourney.s05_rewording.proof import GraphInvarianceValidator
    from gmsjourney.s06_bots.harness import JourneyRunner
    PG, B = _bench()
    val = GraphInvarianceValidator(B.store, B.oracle, B.reader, B.realizer)
    return JourneyRunner(store=B.store, realizer=B.realizer, reader=B.reader, bot_factory=fac, customer="scripted",
                         paths=val)


def _scenario(case_id="default"):
    from gmsjourney.s04_cases.design import to_item
    from gmsjourney.s04_cases.journey import JourneySpec
    from gmsjourney.s05_rewording.catalog import Scenario
    PG, B = _bench()
    cases = {c["id"]: c for c in PG.cases("project", None)["cases"]}
    item = to_item(JourneySpec(**cases[case_id]["levels"]), B.oracle, case_id)
    return Scenario(f"{item.qid}~base", item, {}, "base")


class Fixed:
    """A bot that opens as the graph's bot does and then always says ``acts`` without calling a tool."""

    def __init__(self, tools, acts):
        from gmsjourney.s06_bots.reference import GraphBot
        _, B = _bench()
        self.inner, self.acts = GraphBot(tools, B.store), acts

    def start(self):
        return self.inner.start()

    def respond(self, text):
        from gmsjourney.s06_bots.reference import BotTurn
        return BotTurn("Sorry?", [{"act": a} for a in self.acts])


# ------------------------------------------------------------------ one turn
def test_a_side_step_flags_the_turn_and_a_wrong_exit_is_off_the_journey():
    from gmsjourney.s06_bots.turns import compare_turn
    call = [{"role": "tool", "tool": "authenticate_customer", "args": {}}]
    exp = [{"act": "ASK_CLARIFY"}, {"act": "ASK_MORE_HELP"}]
    assert compare_turn([], exp, [], exp, SIDE)["status"] == "same"
    got = compare_turn([], exp, [], [{"act": "ANSWER_KB", "fact": None}, {"act": "ASK_MORE_HELP"}], SIDE)
    assert got["status"] == "side" and got["core_ok"] and not got["acts_ok"]
    assert compare_turn([], exp, [], [{"act": "END"}], SIDE)["status"] == "differs"
    assert compare_turn(call, [{"act": "PRESENT_TXN"}], [], [{"act": "PRESENT_TXN"}], SIDE)["status"] == "differs"
    # an acknowledgement may be added or left out
    assert compare_turn([], [{"act": "SUMMARIZE"}], [], [{"act": "CONFIRM_RECORDED"}, {"act": "SUMMARIZE"}],
                        SIDE)["status"] == "same"


def test_a_re_ask_is_only_a_re_ask_without_a_call_and_on_the_same_question():
    from gmsjourney.s06_bots.turns import asked_again
    assert asked_again([], [{"act": "ASK_AUTH"}], "ASK_AUTH")
    assert not asked_again([], [{"act": "PRESENT_TXN"}], "ASK_AUTH")
    assert not asked_again([{"tool": "authenticate_customer"}], [{"act": "ASK_AUTH"}], "ASK_AUTH")


# ------------------------------------------------------------------ order
def test_calls_and_acts_are_kept_in_the_order_they_happened():
    from gmsjourney.s06_bots.turns import split_turns
    ev = [{"role": "user", "frame": "X", "text": "x", "line": 0},
          {"role": "tool", "tool": "record"},
          {"role": "tool", "tool": "options"},
          {"role": "bot", "acts": [{"act": "CONFIRM_RECORDED"}, {"act": "PRESENT_OPTIONS"}], "at": [1, 2]}]
    _, turns = split_turns(ev)
    order = [(k, v.get("tool") or v.get("act")) for k, v in turns[0]["steps"]]
    assert order == [("call", "record"), ("say", "CONFIRM_RECORDED"), ("call", "options"), ("say", "PRESENT_OPTIONS")]
    ev[-1].pop("at")                                   # without ``at`` the acts come after the calls
    _, turns = split_turns(ev)
    assert [k for k, _ in turns[0]["steps"]] == ["call", "call", "say", "say"]


def test_the_pending_acts_remember_where_they_were_decided():
    from gmsjourney.s02_graph.pending import Pending
    calls = [0]
    p = Pending(lambda: calls[0])
    p.append({"act": "A"})
    calls[0] = 2
    p.append({"act": "B"})
    p.append({"act": "C"})
    p.pop()
    assert p.event() == {"role": "bot", "acts": [{"act": "A"}, {"act": "B"}], "at": [0, 2]}
    p.clear()
    assert p.at == []


def test_the_graph_and_its_bot_say_a_turn_in_the_same_order():
    from gmsjourney.s06_bots.reference import make_graph_factory
    from gmsjourney.s06_bots.turns import split_turns
    _, B = _bench()
    runner = _runner(make_graph_factory(B.store))
    scn = _scenario("get_card_action_options=none")
    res = runner.run(scn)
    _, got = split_turns(res.extras["trace"].events)
    exp = runner._expected(scn)
    shape = lambda t: [(k, v.get("tool") or v.get("act")) for k, v in t["steps"]]   # noqa: E731
    assert [shape(t) for t in got] == [shape(t) for t in exp]
    # interleaved, not all calls first: some turn says something between two calls
    kinds = lambda t: "".join("c" if k == "call" else "s" for k, _ in shape(t))      # noqa: E731
    assert any("sc" in kinds(t) for t in exp), "expected at least one turn whose acts sit between its calls"
    assert {s["status"] for s in res.extras["turn_status"]} == {"same"} and res.extras["stopped"] is None


# ------------------------------------------------------------------ the run
def test_a_re_ask_gets_the_line_again_once_and_two_off_turns_stop_the_run():
    from gmsjourney.s06_bots.harness import JourneyRunner  # noqa: F401
    runner = _runner(lambda tools: Fixed(tools, ["ASK_AUTH"]))
    res = runner.run(_scenario())
    st = res.extras["turn_status"]
    assert [(s["line"], s["status"], s["repeat"]) for s in st] == [(0, "asked_again", False), (0, "differs", True)]
    assert res.extras["stopped"]["line"] == 0
    users = [e for e in res.extras["trace"].events if e.get("role") == "user"]
    assert [(u["line"], u["repeat"]) for u in users] == [(0, False), (0, True)]


def test_one_turn_off_is_tolerated_two_in_a_row_end_the_run():
    runner = _runner(lambda tools: Fixed(tools, ["PRESENT_TXN"]))
    res = runner.run(_scenario())
    st = res.extras["turn_status"]
    # line 1 answers PRESENT_TXN, so saying it again there is a re-ask: off the journey all the same
    assert [(s["line"], s["status"]) for s in st] == [(0, "differs"), (1, "asked_again")]
    assert res.extras["stopped"] == {"turn": 1, "line": 1, "why": "two bot turns in a row off the journey"}
    script = res.extras["script_lines"]
    assert sum(1 for e in res.extras["trace"].events if e.get("role") == "user") == 2 < script


class OnceOff:
    """The graph's bot, but its first reply adds an act the graph does not say there."""

    def __init__(self, tools):
        from gmsjourney.s06_bots.reference import GraphBot
        _, B = _bench()
        self.inner, self.n = GraphBot(tools, B.store), 0

    def start(self):
        return self.inner.start()

    def respond(self, text):
        t = self.inner.respond(text)
        self.n += 1
        if self.n == 1:
            t.acts = t.acts + [{"act": "SUMMARIZE"}]
            t.calls_before = None
        return t


def test_a_bot_back_on_the_journey_after_one_wrong_turn_plays_on():
    res = _runner(OnceOff).run(_scenario())
    st = [s["status"] for s in res.extras["turn_status"]]
    assert st[0] == "differs" and set(st[1:]) == {"same"} and res.extras["stopped"] is None
    assert len(st) == res.extras["script_lines"]


# ------------------------------------------------------------------ speech
def test_a_speech_failure_in_a_side_step_is_flagged_at_its_turn_only():
    from gmsjourney.s07_checks.speech_checks import Finding, TurnReport
    from gmsjourney.s07_checks.speech_checks import ConversationChecker as Conversation
    domain.set_current("charge_review_graph")
    rep = TurnReport(1, "x", ["ANSWER_KB"], [], [], [Finding("missing", "ANSWER_KB", "no figure")])
    Conversation._mark_side(rep, [{"act": "ANSWER_KB", "text": "x"}])
    assert rep.side["findings"] == {0} and rep.core_ok and not rep.ok
    rep2 = TurnReport(2, "y", ["SUMMARIZE"], [], [], [Finding("missing", "SUMMARIZE", "no summary")])
    Conversation._mark_side(rep2, [{"act": "SUMMARIZE", "text": "y"}])
    assert not rep2.core_ok and rep2.as_dict()["findings"][0].get("side") is None


# ------------------------------------------------------------------ the graph
def test_an_unclear_answer_to_anything_else_is_clarified_then_a_second_one_transfers():
    from gmsjourney.s06_bots.turns import split_turns
    _, B = _bench()
    for cid, want in (("unclear_answer=once_at_ask_more_help", ["ASK_CLARIFY", "ASK_MORE_HELP"]),
                      ("unclear_answer=persists_at_ask_more_help", ["SUMMARIZE", "TRANSFER"])):
        md = _scenario(cid).base.metadata
        _, turns = split_turns(B.oracle.run(md["spec"], md["fixture"]).events)
        unclear = [t for t in turns if str(t["frame"]).startswith("UNCLEAR")]
        assert unclear and [a["act"] for a in unclear[-1]["acts"]][-len(want):] == want, (cid, unclear[-1]["acts"])
    g = domain.current().graph
    assert "UNCLEAR" not in (g.states["ASK_MORE_HELP"].get("on") or {})


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:          # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
