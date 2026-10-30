"""What the bot says: the reader, the session facts, the verifier and the controls (gmsjourney/speech).

The first group runs on the policy graph alone (exact figures and rule stances read straight from
the graph, no knowlytix needed): fast, and it pins the behaviour of each piece. The second group
runs the campaign's own knowlytix store with the reference bot.

    python tests/test_speech.py            # all
    pytest -q tests/test_speech.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gmsjourney.knowledge.policy_kg import PolicyKG  # noqa: E402
from gmsjourney.ontology import SemanticFrame as F  # noqa: E402
from gmsjourney.speech import SessionFacts, SpeechContract, UtteranceVerifier  # noqa: E402
from gmsjourney.speech.reader import words_to_digits  # noqa: E402


# ------------------------------------------------------------------ a store made of the graph alone
class GraphOnlyStore:
    """Exact figures and rule stances straight from the policy graph; no geometry (score_triple raises,
    so the verifier reports no distance)."""

    def __init__(self, kg):
        self.kg = kg

    def lookup_enm(self, category, key):
        if category == "limit":
            return self.kg.limit(key)
        rel = {"delivery_fee_usd": "has_fee_usd", "delivery_days_min": "has_business_days_min",
               "delivery_days_max": "has_business_days_max"}[category]
        return self.kg.tails(key, rel)[0]

    def score_triple(self, h, r, t):
        raise RuntimeError("no geometry")

    def tau(self, r):
        raise RuntimeError("no geometry")

    def stance_check(self, s, p, o, polarity):
        st = next(t["qualifiers"]["stance"] for t in self.kg.triples
                  if (t["head"], t["relation"], str(t["tail"])) == (s, p, o) and "stance" in t.get("qualifiers", {}))
        claimed = "must" if polarity > 0 else "must_not"
        return {"verdict": "supported" if claimed == ("must" if st == "may" else st) else "contradicted", "energy": None}


@lru_cache(maxsize=1)
def graph_env():
    kg = PolicyKG.load()
    contract = SpeechContract.from_graph(kg)
    return kg, contract, UtteranceVerifier(GraphOnlyStore(kg), contract)


TXNS = [{"transaction_id": "T1", "merchant": "Amazon", "amount": "45.67", "date": "2026-03-13", "age_days": 3},
        {"transaction_id": "T2", "merchant": "Starbucks", "amount": "6.25", "date": "2026-03-08", "age_days": 8}]


def log_with(*extra):
    """A tool log: authenticated, two charges returned, then ``extra`` calls."""
    return [{"tool": "authenticate_customer", "args": {"credential": "4821"}, "result": {"status": "AUTHENTICATED"}, "ok": True},
            {"tool": "get_review_transactions", "args": {}, "result": {"status": "TRANSACTIONS", "transactions": TXNS},
             "ok": True}, *extra]


RECORD_T1 = {"tool": "record_charge_response", "args": {"transaction_id": "T1", "disposition": "UNRECOGNIZED"},
             "result": {"status": "SUCCESS", "transaction_id": "T1", "disposition": "UNRECOGNIZED"}, "ok": True}
OPTIONS = {"tool": "get_card_action_options", "args": {}, "result": {"status": "OPTIONS",
                                                                    "options": ["SECURE_CARD", "REPLACE_CARD"]}, "ok": True}
CALLER = [F.make("PROVIDE_CREDENTIAL", credential="4821"),
          F.make("INFORM_DISPOSITION", disposition="UNRECOGNIZED", card_action="REPLACE_CARD", delivery="expedited_delivery")]


def checker(pending=("PRESENT_TXN",), last_item="T1", presented=()):
    _, _, v = graph_env()
    c = v.conversation("2026-03-16")
    c.reports.append(None)                          # past the opening turn
    c.pending, c.last_item, c.presented = set(pending), last_item, set(presented)
    return c


def statuses(report):
    return {(x.claim.field, str(x.claim.value)): x.status for x in report.verdicts}


# ------------------------------------------------------------------ graph-only tests
def test_graph_declares_what_every_act_says():
    kg, c, _ = graph_env()
    assert kg.validate() == []
    assert c.acts["PRESENT_TXN"].required == ("merchant", "amount", "date")
    assert c.acts["DISCLOSE_FEE"].source_of("fee") == "policy.delivery_fee_usd"
    assert c.acts["SUMMARIZE"].source_of("recorded") == "writes.record_charge_response"
    assert "not recognized" in c.phrases["UNRECOGNIZED"] and "not mine" in c.phrases["UNRECOGNIZED"]
    assert c.item_fields["date"] == "date"
    assert "R27" in c.stances and c.stances["R27"].stance == "must_not"
    assert all(c.acts[s].says or c.acts[s].cues or s == "ANSWER_KB" for s in kg.states)
    print("ok test_graph_declares_what_every_act_says")


def test_reader_reads_claims_without_labels():
    _, c, v = graph_env()
    facts = SessionFacts.from_run(log_with(RECORD_T1), CALLER, "2026-03-16", c.success_values)
    r = v.reader.read("Thanks, I've noted that you don't recognize the Amazon charge of $45.67.", facts)
    got = {(x.field, str(x.value), x.about, x.mode) for x in r.claims}
    assert ("disposition", "UNRECOGNIZED", "T1", "recorded") in got, got
    assert ("amount", "45.67", "T1", "states") in got and ("merchant", "Amazon", "T1", "states") in got
    assert not v.reader.read("Do you recognize that charge?", facts).of("disposition")      # a question asserts nothing
    r = v.reader.read("I see a charge at Starbucks for $6.25 on March 8. Do you recognize it?", facts)
    assert [(x.field, x.value, x.about) for x in r.of("date")] == [("date", "2026-03-08", "T2")]
    assert "PRESENT_TXN" in r.heard
    r = v.reader.read("Express delivery costs fifteen dollars and arrives in two business days.", facts)
    assert [(x.value, x.about) for x in r.of("fee")] == [(15, "expedited_delivery")]
    assert [x.value for x in r.of("delivery_days")] == [(2, 2)]
    r = v.reader.read("I recorded the Amazon charge as not recognized and the Starbucks charge as recognized.", facts)
    assert {(x.about, x.value) for x in r.of("disposition")} == {("T1", "UNRECOGNIZED"), ("T2", "RECOGNIZED")}
    r = v.reader.read("An unrecognized charge is not, by itself, proof of fraud.", facts)
    assert [(x.about, x.value) for x in r.of("stance")] == [("R27", -1)]
    assert words_to_digits("twenty-five dollars, forty five") == "25 dollars, 45"
    print("ok test_reader_reads_claims_without_labels")


def test_session_facts_count_only_true_successes():
    _, c, _ = graph_env()
    echo_wrong = {"tool": "record_charge_response", "args": {"transaction_id": "T2", "disposition": "RECOGNIZED"},
                  "result": {"status": "SUCCESS", "transaction_id": "T1", "disposition": "RECOGNIZED"}, "ok": True}
    failed = {"tool": "submit_card_action", "args": {"action": "REPLACE_CARD"}, "result": {"error": "x"}, "ok": False}
    f = SessionFacts.from_run(log_with(RECORD_T1, echo_wrong, failed), CALLER, "2026-03-16", c.success_values)
    assert [w["args"]["transaction_id"] for w in f.writes["record_charge_response"]] == ["T1"]
    assert f.write_for("record_charge_response", "T2") is None                  # echo mismatch is not a success
    assert f.write_for("submit_card_action", None) is None
    assert f.resolve("item.date", "T2").value == "2026-03-08"
    assert f.caller["delivery"] == "expedited_delivery"
    print("ok test_session_facts_count_only_true_successes")


def test_the_worked_example_is_caught():
    """Labels right, tools right, words wrong: the customer hears the opposite answer and the wrong fee."""
    acts = [{"act": "CONFIRM_RECORDED", "txn": "T1", "disposition": "UNRECOGNIZED",
             "text": "Thanks, I've noted that you recognize the Amazon charge of $45.67."},
            {"act": "DISCLOSE_FEE", "fee": 15, "delivery": "expedited_delivery",
             "text": "Express delivery costs $25 and arrives in 2 business days."},
            {"act": "CONFIRM_ACTION", "action": "REPLACE_CARD", "delivery": "expedited_delivery",
             "text": "Shall I send you a replacement card with express delivery?"}]
    rep = checker(presented={"T1"}).check(acts, " ".join(a["text"] for a in acts), log_with(RECORD_T1, OPTIONS), CALLER)
    st = statuses(rep)
    assert st[("disposition", "RECOGNIZED")] == "contradicted"
    assert st[("fee", "25")] == "contradicted"
    assert st[("amount", "45.67")] == "supported" and st[("delivery_days", "(2, 2)")] == "supported"
    assert st[("action", "REPLACE_CARD")] == "supported"                       # offered, and it was returned
    kinds = {(f.kind, f.act) for f in rep.findings}
    assert ("text_contradicts_label", "CONFIRM_RECORDED") in kinds and ("text_contradicts_label", "DISCLOSE_FEE") in kinds
    assert not rep.ok
    good = [dict(acts[0], text="Thanks, I've noted that you don't recognize the Amazon charge."),
            dict(acts[1], text="Expedited delivery has a $15 fee and arrives in 2 business days."), acts[2]]
    rep = checker(presented={"T1"}).check(good, " ".join(a["text"] for a in good), log_with(RECORD_T1, OPTIONS), CALLER)
    assert rep.ok and rep.flow_ok, [f.as_dict() for f in rep.findings] + [v.as_dict() for v in rep.verdicts if v.failed]
    print("ok test_the_worked_example_is_caught")


def test_required_information_and_reasks():
    log = log_with()
    no_date = [{"act": "PRESENT_TXN", "txn": "T1", "text": "I see a charge at Amazon for $45.67. Do you recognize it?"}]
    rep = checker(pending=("ASK_AUTH",), last_item=None).check(no_date, no_date[0]["text"], log, CALLER[:1])
    assert [(f.kind, f.act, f.rule) for f in rep.findings] == [("missing", "PRESENT_TXN", "pdf.2B")]
    reask = [{"act": "ASK_CLARIFY", "txn": "T1", "text": "Sorry, I didn't catch that."},
             {"act": "PRESENT_TXN", "txn": "T1", "text": "Do you recognize that charge?"}]
    rep = checker(presented={"T1"}).check(reask, "x", log, CALLER[:1] + [F.make("UNCLEAR")])
    assert rep.ok and rep.flow_ok, [f.as_dict() for f in rep.findings]
    summary = [{"act": "SUMMARIZE", "recorded": {"T1": "UNRECOGNIZED"}, "text": "Here's a summary of what I recorded."}]
    rep = checker(pending=("PRESENT_TXN",)).check(summary, "x", log_with(RECORD_T1), CALLER)
    assert [(f.kind, f.rule) for f in rep.findings if f.kind == "missing"] == [("missing", "pdf.2F")]
    print("ok test_required_information_and_reasks")


def test_claims_need_a_success_and_flow_follows_the_graph():
    done = [{"act": "ACTION_DONE", "action": "REPLACE_CARD", "text": "That's done. Your card has been replaced."}]
    rep = checker(pending=("CONFIRM_ACTION",)).check(done, "x", log_with(RECORD_T1, OPTIONS), CALLER + [F.make("CONFIRM")])
    assert statuses(rep)[("action", "REPLACE_CARD")] == "unsupported"         # no submit succeeded (R17)
    rec = [{"act": "CONFIRM_RECORDED", "txn": "T2", "text": "Thanks, I've recorded that the Starbucks charge is yours."}]
    rep = checker().check(rec, "x", log_with(RECORD_T1), CALLER)
    assert statuses(rep)[("disposition", "RECOGNIZED")] == "unsupported"      # nothing recorded for T2 (R13)
    jump = [{"act": "PRESENT_TXN", "txn": "T1", "text": "I see a charge at Amazon for $45.67 on March 13."},
            {"act": "ACTION_DONE", "action": "SECURE_CARD", "text": "Your card has been secured."}]
    rep = checker(pending=("ASK_AUTH",), last_item=None).check(jump, "x", log_with(), CALLER[:1])
    assert ("not_allowed_here", "ACTION_DONE") in {(f.kind, f.act) for f in rep.findings}
    # answering a policy question nobody asked is out of flow; asking to clarify never is (the bot cannot
    # know what was meant, and the policy prefers asking to guessing)
    recite = [{"act": "ANSWER_KB", "text": "Expedited delivery has a $15 fee."},
              {"act": "PRESENT_TXN", "txn": "T1", "text": "Do you recognize that charge?"}]
    rep = checker(presented={"T1"}).check(recite, "x", log_with(), CALLER[:1] + [F.make("UNCLEAR")])
    assert [(f.kind, f.act) for f in rep.findings] == [("not_allowed_here", "ANSWER_KB")]
    clarify = [{"act": "ASK_CLARIFY", "text": "Sorry, I didn't catch that."},
               {"act": "ASK_AUTH", "text": "Please tell me the last four digits of your card."}]
    rep = checker(pending=("ASK_AUTH",), last_item=None).check(clarify, "x", [], CALLER[:1])
    assert rep.flow_ok, [f.as_dict() for f in rep.findings]
    fraud = [{"act": "ANSWER_KB", "text": "Yes, an unrecognized charge means fraud."}]
    rep = checker().check(fraud, "x", log_with(), CALLER[:1] + [F.make("ASK_KB", intent="fraud")])
    assert statuses(rep)[("stance", "1")] == "contradicted"                   # R27 (must_not)
    print("ok test_claims_need_a_success_and_flow_follows_the_graph")


def test_a_bot_without_labels_is_read_from_its_words():
    text = "Thanks, I've noted that you recognize the Amazon charge. Expedited delivery costs $25."
    rep = checker(presented={"T1"}).check([], text, log_with(RECORD_T1, OPTIONS), CALLER)
    assert {"CONFIRM_RECORDED", "DISCLOSE_FEE"} <= set(rep.heard)
    assert statuses(rep)[("disposition", "RECOGNIZED")] == "contradicted" and statuses(rep)[("fee", "25")] == "contradicted"
    print("ok test_a_bot_without_labels_is_read_from_its_words")


# ------------------------------------------------------------------ with the knowlytix store
@lru_cache(maxsize=1)
def store_env():
    from _store import kx_store
    from gmsjourney.bases.journey import JourneyBaseGenerator, to_item
    from gmsjourney.core import Scenario
    from gmsjourney.enrich.reader import FrameReader
    from gmsjourney.enrich.realizers import Realizer
    from gmsjourney.oracle import PolicyOracle
    store = kx_store()
    oracle = PolicyOracle(store)
    gen = JourneyBaseGenerator(oracle, seed=3)
    rows, rep = gen.covering()
    specs = rows + gen.closure(rows, store, rep)
    scns = [Scenario(f"S{i}~base", to_item(sp, oracle, f"S{i}"), {}, "base") for i, sp in enumerate(specs)]
    return store, oracle, scns, {"store": store, "realizer": Realizer(), "reader": FrameReader(store)}


def test_reference_bot_words_are_grounded_and_word_defects_are_caught():
    from gmsjourney.evaluation.runner import run
    from gmsjourney.sut.adapter import JourneyRunner, make_reference_factory
    store, oracle, scns, kw = store_env()
    rows = run(scns, JourneyRunner(bot_factory=make_reference_factory(store), **kw), oracle.limits, store.workflow, show=False)
    # every word the reference bot says on its base cases is grounded, except where the policy-answer grader
    # independently finds its answer wrong
    bad = [r for r in rows if not r["c_speech"]]
    assert all(r.get("kb_correct") == 0 for r in bad), [(r["sid"], r.get("speech_problems")) for r in bad]
    assert sum(r["speech_claims"] for r in rows) > 500
    for m in ("says_wrong_fee", "misstates_recorded_answer", "names_wrong_amount", "summary_omits"):
        mrows = run(scns, JourneyRunner(bot_factory=make_reference_factory(store, frozenset({m})), **kw),
                    oracle.limits, store.workflow, show=False)
        assert all(r["outcome_correct"] == ref["outcome_correct"] for r, ref in zip(mrows, rows))   # words only
        assert any(ref["correct"] and not r["correct"] for r, ref in zip(mrows, rows)), m        # and caught
    print("ok test_reference_bot_words_are_grounded_and_word_defects_are_caught")


def test_word_only_controls_are_all_flagged():
    from gmsjourney.speech.controls import run_controls
    from gmsjourney.sut.adapter import JourneyRunner, make_reference_factory
    store, _, scns, kw = store_env()
    res = run_controls(lambda f: JourneyRunner(bot_factory=f, **kw), make_reference_factory(store), scns)
    applied = {k: v for k, v in res["controls"].items() if v["applied"]}
    assert len(applied) >= 9, res["controls"].keys()
    assert all(v["detection_rate"] == 1.0 for v in applied.values()), \
        {k: (v["detected"], v["applied"], v["missed"][:1]) for k, v in applied.items()}
    print(f"ok test_word_only_controls_are_all_flagged ({res['overall']['detected']}/{res['overall']['applied']})")


def test_policy_figures_carry_the_gms_reading():
    store, *_ = store_env()
    from gmsjourney.speech import SpeechContract, UtteranceVerifier
    v = UtteranceVerifier(store, SpeechContract.from_graph(store.kg))
    facts = SessionFacts.from_run(log_with(), CALLER, "2026-03-16", v.c.success_values)
    right = v.reader.read("Expedited delivery has a $15 fee.", facts).of("fee")[0]
    wrong = v.reader.read("Expedited delivery has a $25 fee.", facts).of("fee")[0]
    a, b = v._corroborated(v.verify_claim(right, facts)), v._corroborated(v.verify_claim(wrong, facts))
    assert a.status == "supported" and a.geometry["inside_cap"] and a.geometry["knowlytix"] == "confirmed"
    assert b.status == "contradicted" and not b.geometry["inside_cap"] and b.geometry["knowlytix"] == "not found"
    # an answer whose subject is in the customer's question is graded, but knowlytix (one sentence, no
    # dialog) is not asked to read it
    q = F.make("ASK_KB", intent="fee", entity="standard_delivery")
    short = v.reader.read("No, it's $0.", facts, question=q).of("fee")[0]
    c = v._corroborated(v.verify_claim(short, facts))
    assert c.status == "supported" and "knowlytix" not in (c.geometry or {})
    print("ok test_policy_figures_carry_the_gms_reading")


def test_word_defects_are_flagged_live_and_sealed_in_the_audit():
    """A defect in the words alone reaches the runtime monitor's hash-chained audit as it happens."""
    from gmsjourney.governance.monitor import GovernanceConfig
    from gmsjourney.sut.adapter import JourneyRunner, make_reference_factory
    store, _, scns, kw = store_env()
    g = GovernanceConfig(theta=0.0, drift_budget=1e9)          # only the speech layers matter here
    runner = JourneyRunner(bot_factory=make_reference_factory(store, frozenset({"misstates_recorded_answer"})),
                           governance=g, **kw)
    res = runner.run(scns[0])
    gov = res.extras["governance"]
    sealed = [e for e in gov["events"] if e["gate"] == "speech_grounding"]
    assert sealed and sealed[0]["decision"] == "deny" and sealed[0]["rule"] == "claim:disposition", sealed[:1]
    assert gov["audit"]["verified"]
    clean = JourneyRunner(bot_factory=make_reference_factory(store), governance=g, **kw).run(scns[0])
    assert not [e for e in clean.extras["governance"]["events"] if e["gate"] == "speech_grounding"]
    print("ok test_word_defects_are_flagged_live_and_sealed_in_the_audit")


if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (not only or any(o in name for o in only)):
            fn()
