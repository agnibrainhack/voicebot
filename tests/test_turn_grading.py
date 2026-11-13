"""How a bot turn is graded against the graph's turn, and the order a turn is shown in (s06_bots/turns.py,
s06_bots/harness.py JourneyRunner, s02_graph/pending.py, s07_checks/speech_checks.py).

* an immaterial deviation (a policy answer, a clarification) flags only its turn ("side"); the run goes on;
* a bot that asks again the question just answered gets the same line once more ("asked_again");
* one material deviation is tolerated; two in a row, or one that ends the call, stop the run (``extras["stopped"]``)
  and the case is incorrect;
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


def _runner(fac, acts_from="bot"):
    from gmsjourney.s05_rewording.proof import GraphInvarianceValidator
    from gmsjourney.s06_bots.harness import JourneyRunner
    PG, B = _bench()
    val = GraphInvarianceValidator(B.store, B.oracle, B.reader, B.realizer)
    return JourneyRunner(store=B.store, realizer=B.realizer, reader=B.reader, bot_factory=fac, customer="scripted",
                         paths=val, acts_from=acts_from)


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
    # line 1 answers PRESENT_TXN, so saying it again there is a re-ask: a material deviation all the same
    assert [(s["line"], s["status"]) for s in st] == [(0, "differs"), (1, "asked_again")]
    assert res.extras["stopped"] == {"turn": 1, "line": 1, "rule": "two_in_a_row", "why": "two material deviations in a row",
                                     "turns": [{"turn": 0, "line": 0, "status": "differs"},
                                               {"turn": 1, "line": 1, "status": "asked_again"}]}
    assert res.extras["stop_rule"] == {"active": True, "off_because": None}
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


class HangsUp:
    """The graph's bot, but it ends the call after its first reply."""

    def __init__(self, tools):
        from gmsjourney.s06_bots.reference import GraphBot
        _, B = _bench()
        self.inner = GraphBot(tools, B.store)

    def start(self):
        return self.inner.start()

    def respond(self, text):
        from gmsjourney.s06_bots.reference import BotTurn
        t = self.inner.respond(text)
        return BotTurn(t.text, t.acts + [{"act": "END"}], True, None)


def test_a_stopped_case_says_why_on_its_page_in_the_report_and_to_the_judges():
    from gmsjourney.s07_checks.llm_judges import Judges
    from gmsjourney.s07_checks.scoring import failure_codes, score
    from gmsjourney.s08_campaign.pipeline import failed_case_records
    from gmsjourney.s09_reports.cases import _case, turn_by_turn
    from gmsjourney.s09_reports.report import failure_reasons
    PG, B = _bench()
    runner = _runner(lambda tools: Fixed(tools, ["PRESENT_TXN"]))
    scn = _scenario()
    df, summ = turn_by_turn(runner, B.oracle, scn, validator=runner.paths)
    res = runner.run(scn)
    row = score(scn, res, B.oracle.limits, B.oracle.store.workflow)
    assert "JOURNEY_STOPPED" in failure_codes(row) and row["correct"] == 0
    rec = failed_case_records([row])[0]               # what the main report's failure list is built from
    reason = failure_reasons(rec)[0]
    assert "customer line 1" in reason and "turn 0 (customer line 0): off the journey" in reason \
        and "JOURNEY_STOPPED" in reason, reason
    page = _case(row, summ, df.to_dict(orient="records"))
    for want in ("Validation Stopped at turn 1", '<div class="stopbar">Validation Stopped</div>',
                 "Validation Stopped: not played."):
        assert want in page, want
    assert "counted toward the stop" not in page and "Why the run stopped" not in page
    from gmsjourney.s10_console import runs
    assert runs._result(row)["stopped"].startswith("The run stopped after the agent's reply to customer line 1")
    text = Judges._case_a({"turns": df.to_dict(orient="records"), "stopped": summ["stopped_detail"]})
    assert "NOT PLAYED" in text and "The test stopped the call after turn 1" in text


def test_a_run_with_tool_faults_says_the_stop_rule_was_off():
    from gmsjourney.s06_bots.reference import make_graph_factory
    from gmsjourney.s06_bots.tools import FaultProfile
    _, B = _bench()
    res = _runner(make_graph_factory(B.store)).run(_scenario(), faults=[FaultProfile("record_charge_response", "error")])
    assert res.extras["stop_rule"] == {"active": False, "off_because": "tool faults are injected"}
    assert res.extras["stopped"] is None


def test_an_agent_that_ends_the_call_where_the_graph_does_not_stops_the_run_and_fails_the_case():
    from gmsjourney.s07_checks.scoring import score
    PG, B = _bench()
    runner = _runner(HangsUp)
    scn = _scenario()
    res = runner.run(scn)
    assert [s["status"] for s in res.extras["turn_status"]] == ["differs"]
    assert res.extras["stopped"]["why"] == "the agent ended the call where the graph does not"
    assert res.extras["stopped"]["rule"] == "ended_call" and res.extras["stopped"]["ended_with"] == {"act": "END"}
    row = score(scn, res, B.oracle.limits, B.oracle.store.workflow)
    assert row["c_journey"] == 0 and row["correct"] == 0


def test_a_clarification_in_place_of_a_policy_answer_is_an_immaterial_deviation():
    from gmsjourney.s06_bots.turns import compare_turn
    exp = [{"act": "ANSWER_KB", "intent": "value", "answer": 15}, {"act": "PRESENT_TXN", "txn": "T1"}]
    got = [{"act": "ASK_CLARIFY"}, {"act": "PRESENT_TXN", "txn": "T1"}]
    assert compare_turn([], exp, [], got, SIDE)["status"] == "side"


def test_charges_are_named_by_merchant_and_amount_and_the_opening_is_shown():
    from gmsjourney.s06_bots.reference import make_graph_factory
    from gmsjourney.s09_reports.cases import turn_by_turn
    PG, B = _bench()
    runner = _runner(make_graph_factory(B.store))
    df, s = turn_by_turn(runner, B.oracle, _scenario("transactions_count=two"), validator=runner.paths)
    said = " ".join(x["d"] for st in df["expected_steps"] for x in st)
    assert "Amazon $45.67" in said and "Starbucks $6.25" in said and "(T1" not in said
    assert s["actual_opening_text"] and s["expected_opening_steps"][0]["d"] == "OPENING"


def test_policy_answers_carry_their_unit():
    from gmsjourney.s06_bots.reference import limit_sentence, value_sentence
    assert value_sentence("standard_delivery", "business_days_max", 7) == "Standard delivery takes at most 7 business days."
    assert value_sentence("expedited_delivery", "fee_usd", 15) == "The fee for expedited delivery is $15."
    assert limit_sentence("lookback_days", 30, "calendar_days") == "The limit is 30 calendar days."
    assert limit_sentence("max_transactions_per_session", 4, "transactions") == "The limit is 4 transactions per session."


def test_the_customer_asks_for_a_new_card_once_and_gives_a_wrong_code_when_verification_fails():
    from gmsjourney.s04_cases.journey import JourneySpec
    PG, B = _bench()
    c = PG._case(B, "x", "x", JourneySpec(transactions_count="two", present_txn_disposition="unrecognized",
                                          present_txn_delivery="standard_delivery"), source="project")
    asks = [l["text"] for l in c["lines"] if l.get("state") == "PRESENT_TXN"]
    assert len(asks) == 2 and "replace my card" in asks[0] and "replace" not in asks[1]
    c = PG._case(B, "y", "y", JourneySpec(authenticate_customer="failed_2x"), source="project")
    assert [l["text"].count("8421") for l in c["lines"][:2]] == [1, 1]
    from gmsjourney.s01_packs.project import Project
    proj = Project.load(domain.current().project_path)
    for f in proj.factors.values():                 # the customer never asks to secure the card
        for lv in f.levels.values():
            for op in lv.customer:
                assert "SECURE_CARD" not in str(op), (f.name, op)


def test_what_was_that_is_read_as_unclear_not_as_a_question():
    from gmsjourney.s06_bots.llm_bot import LLMReader
    _bench()
    r = LLMReader.__new__(LLMReader)
    r.g = domain.current().graph
    assert r._not_heard("Hmm, sorry, what was that?") and r._not_heard("Could you say that again?")
    assert not r._not_heard("How much does expedited delivery cost?")


def test_the_speech_playground_is_copied_into_a_run_and_its_cards_keep_what_it_shows(tmp_path=None):
    import json
    import tempfile
    from gmsjourney.s07_checks.scoring import response_cards
    from gmsjourney.s09_reports.cases import write_speech_playground
    out = Path(tmp_path or tempfile.mkdtemp())
    page = write_speech_playground(out).read_text()
    assert "dataset/case_results.jsonl" in page and "webkitdirectory" in page and "Said vs truth" in page
    _, B = _bench()
    conv = B.speech.conversation(None)
    t = "Thanks, I've noted charge T1 as recognized. Is there anything else I can help with?"
    rep = conv.check([{"act": "CONFIRM_RECORDED", "text": t}], t, [], [])
    card = response_cards([{**rep.as_dict(), "turn": 1}], [])
    assert card and "heard" in card[0] and "core_ok" in card[0]
    assert all("sentence" in c for c in card[0]["claims"]) and json.dumps(card)


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
    # the journey's cases no longer ask for clarifications (no unclear_answer factor); the graph still clarifies
    from gmsjourney.s04_cases.journey import JourneySpec, UserSimulator
    from gmsjourney.s04_cases.meaning import SemanticFrame as F
    from gmsjourney.s06_bots.turns import split_turns
    _, B = _bench()
    md = _scenario().base.metadata
    for n, want in ((1, ["ASK_CLARIFY", "ASK_MORE_HELP"]), (2, ["SUMMARIZE", "TRANSFER"])):
        sim, left = UserSimulator(JourneySpec(), md["fixture"]), [n]

        def respond(act, sim=sim, left=left):
            if act.get("act") == "ASK_MORE_HELP" and left[0] > 0:
                left[0] -= 1
                return F.make("UNCLEAR")
            return sim.respond(act)
        _, turns = split_turns(B.oracle.run(md["spec"], md["fixture"], responder=respond).events)
        unclear = [t for t in turns if str(t["frame"]).startswith("UNCLEAR")]
        assert len(unclear) == n and [a["act"] for a in unclear[-1]["acts"]][-len(want):] == want, (n, unclear[-1]["acts"])
    g = domain.current().graph
    assert "UNCLEAR" not in (g.states["ASK_MORE_HELP"].get("on") or {})


def test_the_journey_asks_for_no_clarification():
    from gmsjourney.s01_packs.project import Project
    PG, B = _bench()
    assert "unclear_answer" not in Project.load(domain.current().project_path).factors
    for c in PG.cases("project", None)["cases"]:
        md = _scenario(c["id"]).base.metadata
        acts = [a.get("act") for e in B.oracle.run(md["spec"], md["fixture"]).events if e.get("role") == "bot"
                for a in e.get("acts") or []]
        assert "ASK_CLARIFY" not in acts, c["id"]


def test_incorrect_action_off_the_journey_and_side_steps_are_told_apart():
    from gmsjourney.s06_bots.turns import compare_turn
    rec = lambda d: [{"role": "tool", "tool": "record_charge_response",                       # noqa: E731
                      "args": {"transaction_id": "T1", "disposition": d}}]
    said = [{"act": "CONFIRM_RECORDED", "txn": "T1"}, {"act": "ASK_MORE_HELP"}]
    assert compare_turn(rec("UNRECOGNIZED"), said, rec("RECOGNIZED"), said, SIDE)["status"] == "wrong_action"
    assert compare_turn([], [{"act": "TRANSFER", "reason": "ambiguity_persists"}], [],
                        [{"act": "TRANSFER", "reason": "customer_request"}], SIDE)["status"] == "wrong_action"
    assert compare_turn(rec("UNRECOGNIZED"), said, [], [{"act": "END"}], SIDE)["status"] == "differs"
    assert compare_turn([], [{"act": "ASK_CLARIFY"}, {"act": "ASK_MORE_HELP"}], [],
                        [{"act": "ANSWER_KB"}, {"act": "ASK_MORE_HELP"}], SIDE)["status"] == "side"


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
