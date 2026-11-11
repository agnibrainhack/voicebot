"""Amounts and policy in what the bot says, on every journey (s07_checks/speech_policy.py, the anchored policy-answer grade).

For each pack (the PDF-built charge review and both compact graphs), with nothing written for the pack:

* a policy answer is graded on the figure tied to what was asked, in any word order;
* a second figure that names nothing is checked against every member, not pinned to the asked one;
* any fact besides the one asked is ``volunteered`` and fails the turn (a policy answer gives exactly the fact asked);
* a policy the graph does not state, a wrong limit in other words, a promise of an action the graph has no tool
  for, and an internal (or never-volunteer) figure each fail the turn; small talk does not;
* the LLM judge must cite a statement it was given, or its verdict is ignored.

    python tests/test_policy_speech.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gmsjourney.s01_packs import loading as domain  # noqa: E402
from gmsjourney.s04_cases.meaning import SemanticFrame as F  # noqa: E402

PACKS = ("charge_review", "charge_review_graph", "supplementary_card")


@lru_cache(maxsize=None)
def env(pack: str):
    from gmsjourney.s07_checks.speech_reading import SpeechContract
    from gmsjourney.s07_checks.speech_checks import UtteranceVerifier
    dom = domain.set_current(pack)
    if dom.graph is None:
        from test_speech import GraphOnlyStore
        store = GraphOnlyStore(dom.kg)
        attr, members = "fee", ["standard_delivery", "expedited_delivery"]
        fee = lambda m: store.lookup_enm("delivery_fee_usd", m)      # noqa: E731
    else:
        from gmsjourney.s02_graph.compile import compile_graph
        store = compile_graph(dom.graph_path).store
        attr = next(a for a in dom.graph.numeric_attributes() if a.endswith("_usd"))
        members = list(dom.graph.numeric_attributes()[attr]["members"])
        fee = lambda m: store.lookup_enm(attr, m)                     # noqa: E731
    v = UtteranceVerifier(store, SpeechContract.from_graph(store.kg))
    return dom, store, v, attr, members, fee


def _word(dom, m: str) -> str:
    if dom.graph is not None:
        return (dom.graph.aliases(m) or [m.lower().replace("_", " ")])[0]
    return m.replace("_", " ")


def _money(x) -> str:
    return "free" if float(x) == 0 else f"${int(x) if float(x).is_integer() else x}"


def turn(pack: str, text: str, question=None):
    from gmsjourney.s07_checks.speech_reading import SessionFacts
    dom, store, v, *_ = env(pack)
    domain.set_current(pack)
    return v.check_turn([{"act": "ANSWER_KB", "text": text}], text, SessionFacts.from_run([], [], None, {}), turn=1,
                        question=question, pending_prompts=set(), customer_act="ASK_KB" if question else None)


def kinds(rep) -> list[str]:
    return [f.kind for f in rep.findings if f.kind != "not_allowed_here"]


def policy(rep) -> list[tuple[str, str]]:
    return [(x.claim.field, x.status) for x in rep.verdicts if x.claim.field in ("policy", "promise", "policy_figure")]


# ------------------------------------------------------------------ amounts
def test_answers_are_graded_on_what_was_asked_in_any_order():
    from gmsjourney.s03_knowledge.questions import judge_kb_answer
    for pack in PACKS:
        dom, store, v, attr, (a, b, *_), fee = env(pack)
        domain.set_current(pack)
        q = F.make("ASK_KB", intent="value", attr=attr, entity=a)
        wa, wb = _word(dom, a), _word(dom, b)
        say = lambda w, m: f"the {w} is {_money(fee(m))}"                       # noqa: E731
        cases = {f"{say(wa, a).capitalize()}.": True,
                 f"{say(wa, a).capitalize()}, and {say(wb, b)}.": True,
                 f"{say(wb, b).capitalize()}, and {say(wa, a)}.": True,          # the other one named first
                 f"{say(wb, b).capitalize()}.": False}                           # did not answer what was asked
        for text, want in cases.items():
            got = judge_kb_answer(store, q, text)
            assert got["correct"] is want, (pack, text, got)
            rep = turn(pack, text, q)
            if want:                                   # the asked fee: ok alone, a failure with another one too
                assert rep.ok is (wb not in text.lower()), (pack, text, [x.as_dict() for x in rep.verdicts], kinds(rep))
            assert ("volunteered" in kinds(rep)) is (wb in text.lower()), (pack, text, kinds(rep))
    print("ok test_answers_are_graded_on_what_was_asked_in_any_order")


def test_a_second_figure_naming_nothing_is_checked_against_every_member():
    for pack in PACKS:
        dom, store, v, attr, (a, b, *_), fee = env(pack)
        q = F.make("ASK_KB", intent="value", attr=attr, entity=a)
        text = f"The {_word(dom, a)} is {_money(fee(a))}. There's also a {_money(fee(b))} option."
        rep = turn(pack, text, q)                       # true, but not what was asked: volunteered
        assert not rep.ok and "volunteered" in kinds(rep), (pack, [x.as_dict() for x in rep.verdicts], kinds(rep))
        assert turn(pack, f"The {_word(dom, a)} is {_money(fee(a))}. That's {_money(fee(a))} in total.", q).ok, pack
        wrong = f"The {_word(dom, a)} is {_money(fee(a))}. There's also a $977 option."
        assert not turn(pack, wrong, q).ok, pack
    print("ok test_a_second_figure_naming_nothing_is_checked_against_every_member")


# ------------------------------------------------------------------ policy
def test_policy_statements_are_checked_against_the_graph():
    for pack in PACKS:
        dom, store, v, attr, (a, *_), fee = env(pack)
        q = F.make("ASK_KB", intent="value", attr=attr, entity=a)
        lead = f"The {_word(dom, a)} is {_money(fee(a))}. "
        corpus = v.policy.corpus
        lim = next(f for f in corpus.figures if f.attr == "limit" and not f.internal and f.unit)
        bad = {
            "invented": "Our policy is to always refund the full amount on request.",
            "promise": "I'll also waive any fees on your account this year.",
            "wrong limit": f"Just so you know, the limit is {int(float(lim.value)) * 3 + 7} {lim.unit.replace('_', ' ')}.",
        }
        for name, extra in bad.items():
            rep = turn(pack, lead + extra, q)
            assert not rep.ok, (pack, name, [x.as_dict() for x in rep.verdicts])
        for extra in ("Let me check that for you.", "Thanks for waiting, I'll be happy to help.",
                      "Let me transfer you to a representative."):
            rep = turn(pack, lead + extra, q)
            assert rep.ok, (pack, extra, policy(rep), kinds(rep))
        rep = turn(pack, lead + "Our policy is to always refund the full amount on request.", q)
        assert ("policy", "unsupported") in policy(rep), (pack, policy(rep))
        rep = turn(pack, lead + "I'll also waive any fees on your account this year.", q)
        assert ("promise", "unsupported") in policy(rep), (pack, policy(rep))
    print("ok test_policy_statements_are_checked_against_the_graph")


def test_internal_or_never_volunteered_figures_are_disclosures():
    from gmsjourney.s07_checks.speech_policy import settings
    for pack in PACKS:
        dom, store, v, attr, (a, *_), fee = env(pack)
        domain.set_current(pack)
        never = set(settings()["never_volunteer"])
        f = next((f for f in v.policy.corpus.figures if f.attr == "limit" and (f.internal or f.subject in never) and f.unit),
                 None)
        assert f is not None, f"{pack}: no internal limit and no never_volunteer limit to test"
        rep = turn(pack, f"I can only go up to {int(float(f.value))} {f.unit.replace('_', ' ')} with you.")
        assert "disclosed" in kinds(rep) and not rep.ok, (pack, kinds(rep), policy(rep))
    print("ok test_internal_or_never_volunteered_figures_are_disclosures")


def test_a_true_rule_in_other_words_restates_it_and_its_denial_is_not_flagged():
    for pack in ("charge_review", "charge_review_graph"):
        rep = turn(pack, "Not recognizing a purchase doesn't automatically make it fraudulent.")
        assert rep.ok, (pack, policy(rep))
        rep = turn(pack, "An unrecognized charge means fraud.")
        assert not rep.ok, (pack, policy(rep))
    print("ok test_a_true_rule_in_other_words_restates_it_and_its_denial_is_not_flagged")


# ------------------------------------------------------------------ the LLM judge
class _FakeLLM:
    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def invoke_json(self, system, user, salt=""):
        self.seen.append(user)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer(user) if callable(self.answer) else self.answer


def test_the_llm_judge_must_cite_a_statement_it_was_given():
    from gmsjourney.s07_checks.speech_policy import ClauseVerdict, LLMPolicyJudge
    dom, store, v, *_ = env("charge_review_graph")
    corpus = v.policy.corpus
    clause = "We always refund the full amount on request."
    graph_v = ClauseVerdict(clause, clause, "statement", "unsupported", why="graph")
    first = lambda user: user.split("\n")[1].split(":")[0]                       # noqa: E731
    ok = LLMPolicyJudge(llm=_FakeLLM(lambda u: {"verdict": "restates", "statement": first(u), "why": "same"}))
    assert ok.review(graph_v, corpus).status == "supported"
    made_up = LLMPolicyJudge(llm=_FakeLLM({"verdict": "restates", "statement": "K999", "why": "?"}))
    assert made_up.review(graph_v, corpus).status == "unsupported"              # no such statement: ignored
    down = LLMPolicyJudge(llm=_FakeLLM(RuntimeError("gateway unreachable")))
    assert down.review(graph_v, corpus).status == "unsupported" and down.errors
    print("ok test_the_llm_judge_must_cite_a_statement_it_was_given")


def test_the_project_checker_reads_speech_policy():
    from gmsjourney.s01_packs.project import _check_speech_policy
    domain.set_current("supplementary_card")
    assert not _check_speech_policy({"advisory": ["unread"], "never_volunteer": ["max_turns"], "judge": "llm"})
    bad = _check_speech_policy({"advisory": ["typo"], "never_volunteer": ["no_such_limit"], "judge": "maybe", "x": 1})
    assert len(bad) == 4, bad
    print("ok test_the_project_checker_reads_speech_policy")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
