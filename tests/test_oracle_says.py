"""An agent that gives only tool calls and words is graded against the oracle's SAYS held in its words
(s06_bots/oracle_says.py): the same verdicts as from its own act labels, without them."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_turn_grading import _bench, _scenario  # noqa: E402


class Unlabelled:
    """Any bot with its act labels taken away: only its tool calls (made on the backend) and its words remain."""

    def __init__(self, inner):
        self.inner = inner

    def _strip(self, t):
        from gmsjourney.s06_bots.reference import BotTurn
        return BotTurn(t.text, [], t.ended, None)

    def start(self):
        return self._strip(self.inner.start())

    def respond(self, text):
        return self._strip(self.inner.respond(text))


def _runner(fac, acts_from="oracle"):
    from gmsjourney.s05_rewording.proof import GraphInvarianceValidator
    from gmsjourney.s06_bots.harness import JourneyRunner
    PG, B = _bench()
    val = GraphInvarianceValidator(B.store, B.oracle, B.reader, B.realizer)
    return JourneyRunner(store=B.store, realizer=B.realizer, reader=B.reader, bot_factory=fac, customer="scripted",
                         paths=val, acts_from=acts_from)


def _misreads():
    from fake_sut_model import Misreads
    from gmsjourney.llm import CachedLLM, LLMSettings
    from gmsjourney.s06_bots.llm_bot import make_structured_factory
    PG, B = _bench()
    r = CachedLLM(LLMSettings(model="fake-Misreads", temperature=0, max_tokens=200, json_mode=True), model=Misreads(), tag="sut")
    return make_structured_factory(B.store, reader_llm=r, wording_llm=None)


def _grade(runner, cid):
    from gmsjourney.s07_checks.scoring import score
    PG, B = _bench()
    scn = _scenario(cid)
    res = runner.run(scn)
    row = score(scn, res, B.oracle.limits, B.oracle.store.workflow)
    return res, row, (row["correct"], row.get("c_speech"), [s["status"] for s in res.extras["turn_status"]],
                      bool(res.extras["stopped"]))


def test_every_act_of_the_pack_can_be_heard_from_words():
    PG, B = _bench()
    c = B.speech.c
    deaf = [a for a, s in c.acts.items() if not s.cues and not [f for f in s.says if f.required and c.claim_field(f)]]
    assert deaf == ["ANSWER_KB"], deaf            # a policy answer is heard by its figure about what was asked


def test_a_bot_without_labels_is_graded_as_with_them():
    from gmsjourney.s06_bots.reference import make_graph_factory
    PG, B = _bench()
    ref = make_graph_factory(B.store)
    mis = _misreads()
    for fac in (ref, mis):
        for cid in ("default", "present_txn_disposition=unrecognized", "present_txn_delivery=expedited_delivery",
                    "human_request=at_provide_context", "authenticate_customer=failed_2x"):
            labelled = _grade(_runner(fac, "bot"), cid)[2]
            unlabelled = _grade(_runner(lambda tools, f=fac: Unlabelled(f(tools))), cid)[2]
            assert labelled == unlabelled, (cid, labelled, unlabelled)


def test_the_oracle_says_are_checked_in_the_words():
    res, row, _ = _grade(_runner(_misreads()), "present_txn_disposition=unrecognized")
    ev = [e for e in res.extras["trace"].events if e.get("role") == "bot"][2]
    says = {r["act"]: r for r in ev["says"]["says"]}
    assert says["CONFIRM_RECORDED"]["status"] == "contradicted"
    assert says["CONFIRM_RECORDED"]["detail"] == "disposition: the oracle says UNRECOGNIZED, the words say RECOGNIZED"
    assert says["SUMMARIZE"]["status"] == "contradicted" and says["ASK_MORE_HELP"]["status"] == "held"
    assert says["ASK_MORE_HELP"]["how"] == "the cue “anything else”"
    # graded as said: the words' values, not the oracle's
    assert ev["acts"][0] == {"act": "CONFIRM_RECORDED", "txn": "T1", "disposition": "RECOGNIZED"}
    assert ev["bot_acts"][0]["act"] == "CONFIRM_RECORDED"          # the agent's own labels kept, never graded
    assert row["c_speech"] == 0 and row["correct"] == 0


def test_words_the_oracle_does_not_expect_are_acts_of_the_agents_own():
    res, _, _ = _grade(_runner(_misreads()), "present_txn_delivery=expedited_delivery")
    ev = [e for e in res.extras["trace"].events if e.get("role") == "bot"][2]
    st = {r["act"]: r["status"] for r in ev["says"]["says"]}
    assert st == {"CONFIRM_RECORDED": "contradicted", "DISCLOSE_FEE": "not_heard", "CONFIRM_ACTION": "not_heard"}
    assert [x["act"] for x in ev["says"]["extra"]] == ["SUMMARIZE", "ASK_MORE_HELP"]
    assert [a["act"] for a in ev["acts"]] == ["CONFIRM_RECORDED", "SUMMARIZE", "ASK_MORE_HELP"]


def test_a_retry_asking_for_the_digits_again_is_one_act():
    from gmsjourney.s06_bots.oracle_says import hold
    from gmsjourney.s07_checks.speech_reading import SessionFacts
    PG, B = _bench()
    h = hold(B.speech, [{"act": "ASK_AUTH_RETRY"}],
             "That didn't match. Please tell me the last four digits of your card again.", SessionFacts())
    assert [r["status"] for r in h["says"]] == ["held"] and h["extra"] == [] and h["acts"] == [{"act": "ASK_AUTH_RETRY"}]


def test_a_policy_answer_with_an_upper_bound_is_correct():
    from gmsjourney.s03_knowledge.questions import judge_kb_answer, kb_frames
    from gmsjourney.s01_packs.loading import current
    PG, B = _bench()
    f = next(x for x in kb_frames(current().graph) if x.get("attr") == "business_days_max"
             and x.get("entity") == "standard_delivery")
    assert judge_kb_answer(B.store, f, "Standard delivery takes at most 7 business days.")["correct"]
    assert judge_kb_answer(B.store, f, "Standard delivery takes 3 to 7 business days.")["correct"]
    assert not judge_kb_answer(B.store, f, "Standard delivery takes at most 5 business days.")["correct"]


def test_the_case_page_shows_the_oracles_says_and_every_claim_card():
    from gmsjourney.s09_reports.cases import _case, turn_by_turn
    from gmsjourney.s07_checks.scoring import score
    PG, B = _bench()
    runner = _runner(_misreads())
    scn = _scenario("present_txn_disposition=unrecognized")
    df, summ = turn_by_turn(runner, B.oracle, scn, validator=runner.paths)
    row = score(scn, runner.run(scn), B.oracle.limits, B.oracle.store.workflow)
    page = _case(row, summ, df.to_dict(orient="records"))
    for want in ("Actual (agent): its calls and its words", "The oracle's SAYS, checked in these words",
                 "✗ other value", "✓ held", "Claim cards", "says RECOGNIZED, the truth is UNRECOGNIZED",
                 "what the customer said", "successful write"):
        assert want in page, want
    agent = page.split("Actual (agent): its calls and its words")[2]
    assert '<span class="sk">says</span>' not in agent.split("Expected (oracle)")[0]   # no SAYS of the agent's own


def _facts():
    from gmsjourney.s07_checks.speech_reading import SessionFacts
    log = [{"tool": "authenticate_customer", "args": {"credential": "4821"}, "result": {"status": "AUTHENTICATED"}},
           {"tool": "get_review_transactions", "args": {},
            "result": {"status": "TRANSACTIONS", "transactions": [
                {"transaction_id": "T1", "merchant": "Amazon", "amount": "45.67"},
                {"transaction_id": "T2", "merchant": "Starbucks", "amount": "6.25"}]}},
           {"tool": "record_charge_response", "args": {"transaction_id": "T1", "disposition": "RECOGNIZED"},
            "result": {"status": "SUCCESS", "transaction_id": "T1", "disposition": "RECOGNIZED"}},
           {"tool": "record_charge_response", "args": {"transaction_id": "T2", "disposition": "UNRECOGNIZED"},
            "result": {"status": "SUCCESS", "transaction_id": "T2", "disposition": "UNRECOGNIZED"}}]
    return SessionFacts.from_run(log, [], None, {})


def test_the_agent_names_a_charge_by_its_details_and_they_are_checked():
    from gmsjourney.s06_bots.reference import _say, items_of
    PG, B = _bench()
    f = _facts()
    items = items_of({"get_review_transactions": {"transactions": list(f.items.values())}})
    assert _say({"act": "PRESENT_TXN", "txn": "T1"}, items) == "Let's look at the Amazon charge for $45.67. Do you recognize it?"
    assert _say({"act": "SUMMARIZE", "recorded": {"T1": "RECOGNIZED", "T2": "UNRECOGNIZED"}}, items) == \
        "To summarize, I've noted the Amazon charge for $45.67 as recognized and the Starbucks charge for $6.25 as unrecognized."
    v = B.speech
    got = lambda text, label=None: [(c.field, c.value, c.about, v.verify_claim(c, f, label).status)     # noqa: E731
                                    for c in v.reader.read(text, f).claims]
    assert got("Let's look at the Amazon charge for $45.67.") == [("item", "T1", "T1", "supported"),
                                                                    ("amount", 45.67, "T1", "supported")]
    assert ("amount", 54.67, "T1", "contradicted") in got("Let's look at the Amazon charge for $54.67.")
    # a summary that swaps two charges' answers: each is held to what was written for that charge
    rec = {"act": "SUMMARIZE", "recorded": {"T1": "RECOGNIZED", "T2": "UNRECOGNIZED"}}
    swapped = got("To summarize, I've noted the Amazon charge as unrecognized and the Starbucks charge as recognized.", rec)
    assert ("disposition", "UNRECOGNIZED", "T1", "contradicted") in swapped
    assert ("disposition", "RECOGNIZED", "T2", "contradicted") in swapped
    # a fee is still a fee when the sentence names the delivery option and no charge
    assert got("Expedited delivery has a fee of $15.")[-1][0] == "fee_usd"
