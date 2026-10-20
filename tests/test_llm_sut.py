"""The LLM-driven bot (gmsjourney/sut/llm_bot.py) with a scripted fake model: no gateway, no key, no network.

The fake plays a minimal happy-path bot (authenticate, review the first charge, record it, summarise, end), so
the checks are about the plumbing: tool calls reach the harness backend in order, acts are read as the trace
expects, a defective bot is graded as such (two charges: the fake reviews only one), malformed replies are
retried once and then surface as an UNKNOWN turn, and a re-run of the same conversations is served from cache.

Run with `pytest -q tests/` or `python tests/test_llm_sut.py`.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsjourney.bases.journey import to_item  # noqa: E402
from gmsjourney.core import Scenario  # noqa: E402
from gmsjourney.enrich.llm.client import CachedLLM, LLMSettings  # noqa: E402
from gmsjourney.enrich.reader import FrameReader  # noqa: E402
from gmsjourney.enrich.realizers import Realizer  # noqa: E402
from gmsjourney.evaluation.runner import run  # noqa: E402
from gmsjourney.journey import JourneySpec  # noqa: E402
from gmsjourney.knowledge.geode import build_store  # noqa: E402
from gmsjourney.oracle import PolicyOracle  # noqa: E402
from gmsjourney.sut.adapter import JourneyRunner  # noqa: E402
from gmsjourney.sut.llm_bot import LLMBot, make_llm_factory  # noqa: E402
from gmsjourney.sut.tools import ToolEnvironment  # noqa: E402

POLICY_KG = Path(__file__).resolve().parents[1] / "gmsjourney" / "policy" / "charge_review_kg.json"


@lru_cache(maxsize=1)
def env():
    store = build_store(POLICY_KG).store
    return store, PolicyOracle(store)


def _r(obj):
    return SimpleNamespace(content=json.dumps(obj), usage_metadata={})


class HappyPathModel:
    """invoke(messages) -> reply, reading the JSON conversation the bot sends."""

    def __init__(self):
        self.calls = 0

    def invoke(self, msgs):
        self.calls += 1
        conv = json.loads(msgs[-1].content)["conversation"]
        last = conv[-1]
        say = lambda acts, end=False: _r({"say": acts, "end": end})
        if "event" in last:
            return say([{"act": "OPENING", "text": "Before I review account-specific charges, I need to authenticate you."},
                        {"act": "ASK_AUTH", "text": "Please tell me the last four digits of your card."}])
        if "tool" in last:
            n, r = last["tool"], last["result"]
            if n == "authenticate_customer":
                return _r({"tool": "get_review_transactions", "args": {}})
            if n == "get_review_transactions":
                t = r["transactions"][0]
                return say([{"act": "PRESENT_TXN", "txn": t["transaction_id"],
                             "text": f"I see a charge at {t['merchant']} for ${t['amount']}. Do you recognize it?"}])
            if n == "record_charge_response":
                return say([{"act": "CONFIRM_RECORDED", "txn": r["transaction_id"], "disposition": r["disposition"],
                             "text": "Thanks, I've recorded that."},
                            {"act": "SUMMARIZE", "recorded": {r["transaction_id"]: r["disposition"]}, "actions": [],
                             "text": "Here's a summary."},
                            {"act": "ASK_MORE_HELP", "text": "Anything else?"}])
        c = last.get("customer", "").lower()
        prev = [x for x in conv if "bot" in x][-1]["bot"][-1]
        if prev["act"] == "ASK_AUTH":
            return _r({"tool": "authenticate_customer", "args": {"credential": "".join(re.findall(r"\d", c))[-4:]}})
        if prev["act"] == "PRESENT_TXN":
            return _r({"tool": "record_charge_response",
                       "args": {"transaction_id": prev["txn"], "disposition": "RECOGNIZED"}})
        return say([{"act": "END", "text": "Thank you, goodbye."}], True)


def _llm(model):
    return CachedLLM(LLMSettings(temperature=0.0, max_tokens=900, cache_dir=tempfile.mkdtemp()), model=model, tag="sut")


def _scenarios(oracle):
    return [Scenario(f"S{i}~base", to_item(sp, oracle, f"S{i}"), {}, "base")
            for i, sp in enumerate((JourneySpec(n_txn=1), JourneySpec(n_txn=2)))]


def test_llm_bot_runs_through_the_harness_and_is_graded():
    store, oracle = env()
    model = HappyPathModel()
    factory = make_llm_factory(store, llm=_llm(model))
    assert "OPERATING PROCEDURE" in factory.system_prompt and "transfer_to_human" in factory.system_prompt
    runner = JourneyRunner(store=store, realizer=Realizer(), reader=FrameReader(store), bot_factory=factory)
    scns = _scenarios(oracle)
    rows = run(scns, runner, oracle.limits, store.workflow, None)
    one, two = rows
    assert one["correct"] == 1 and one["tool_sequence"] == \
        "authenticate_customer>get_review_transactions>record_charge_response"
    assert two["correct"] == 0                       # reviews one of two charges: graded as a defect
    st = factory.stats.as_dict()
    assert st["conversations"] == 2 and st["invalid_replies"] == 0 and st["unknown_turns"] == 0
    n = model.calls
    run(scns, runner, oracle.limits, store.workflow, None)
    assert model.calls == n                          # same conversations again: all from the replay cache


def test_malformed_replies_are_retried_once_then_surface():
    class Broken:
        def __init__(self):
            self.calls = 0

        def invoke(self, msgs):
            self.calls += 1
            return SimpleNamespace(content="I would like to help you!", usage_metadata={})
    store, _o = env()
    model = Broken()
    factory = make_llm_factory(store, llm=_llm(model))
    bot = factory(ToolEnvironment({"credential": "4821", "transactions": []}))
    turn = bot.start()
    assert turn.acts[0]["act"] == "UNKNOWN" and not turn.ended
    assert factory.stats.invalid_replies == 2 and factory.stats.unknown_turns == 1 and model.calls == 2


def test_tool_errors_and_bad_arguments_reach_the_model_not_the_harness():
    class ToolThenSay:
        def __init__(self):
            self.seen = []

        def invoke(self, msgs):
            conv = json.loads(msgs[-1].content)["conversation"]
            if "tool" in conv[-1]:
                self.seen.append(conv[-1]["result"])
                if len(self.seen) == 1:
                    return _r({"tool": "authenticate_customer", "args": {"credential": 4821, "bogus": 1}})
                return _r({"say": [{"act": "ASK_AUTH", "text": "Digits?"}], "end": False})
            return _r({"tool": "no_such_tool", "args": {}})
    store, _o = env()
    model = ToolThenSay()
    factory = make_llm_factory(store, llm=_llm(model))
    tools = ToolEnvironment({"credential": "4821", "auth_results": ["AUTHENTICATED"], "transactions": []})
    bot = factory(tools)
    bot.start()
    assert "error" in model.seen[0]                  # unknown tool: an error result, nothing called
    assert model.seen[1] == {"status": "AUTHENTICATED"}   # int credential coerced, unknown arg dropped
    assert [c["tool"] for c in tools.log] == ["authenticate_customer"]
    assert isinstance(bot, LLMBot)



def test_sample_bases_keeps_the_category_mix_and_all_by_default():
    from gmsjourney.pipeline import sample_bases
    B = [SimpleNamespace(qid=f"{c}{i}", base=c) for c, n in (("a", 6), ("b", 3), ("c", 1)) for i in range(n)]
    assert sample_bases(B, None) == B and sample_bases(B, 99) == B
    pick = sample_bases(B, 6)
    assert len(pick) == 6 and {b.base for b in pick} == {"a", "b", "c"}
    assert sum(b.base == "a" for b in pick) >= sum(b.base == "b" for b in pick)


def test_failure_reasons_read_the_graded_row():
    from gmsjourney.report import failure_reasons, _strip
    row = {"why": {"expected": {"end": "TRANSFERRED", "transfer_reason": "record_failed", "recorded": [], "action": None},
                   "actual": {"end": "COMPLETED", "transfer_reason": None, "recorded": [["T1", "RECOGNIZED"]],
                              "action": None},
                   "stages": {"record": {"expected": "T1=RECOGNIZED:FAILED", "actual": "T1=RECOGNIZED:SUCCESS"}}},
           "violations": ["claim_without_success:T1"],
           "kb_detail": [{"frame": "ASK_KB(intent=fee)", "answer": "It's $0.", "correct": False, "detail": "truth=15"}]}
    text = " | ".join(_strip(r) for r in failure_reasons(row))
    for want in ("expected TRANSFERRED (record_failed)", "COMPLETED", "T1=RECOGNIZED", "Stage record",
                 "claimed a response", "truth=15"):
        assert want in text, (want, text)


def test_case_comparison_page_from_campaign_files():
    import csv
    from gmsjourney.case_compare import write_case_comparison
    d = Path(tempfile.mkdtemp())
    (d / "dataset").mkdir()
    sig_e = {"end": "TRANSFERRED", "transfer_reason": "record_failed", "recorded": [], "action": None}
    sig_a = {"end": "ENDED", "transfer_reason": None, "recorded": [["T1", "RECOGNIZED"]], "action": None}
    rows = [{"sid": "S1~base", "seed": "S1", "base": "journey", "arm": "base", "correct": 0, "workflow_adherent": 0,
             "c_record": 0, "violations": ["claim_without_success:T1"], "severity": "critical",
             "signature": {"expected": sig_e, "actual": sig_a},
             "why": {"expected": sig_e, "actual": sig_a, "stages": {"record": {"expected": "a", "actual": "b"}}}},
            {"sid": "S2~base", "seed": "S2", "base": "journey", "arm": "base", "correct": 1, "workflow_adherent": 1,
             "c_record": 1, "violations": [], "signature": {"expected": sig_a, "actual": sig_a}}]
    (d / "dataset" / "case_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with open(d / "dataset" / "turn_by_turn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["case_id", "turn", "state", "customer_text", "customer_card", "expected_tools",
                               "actual_tools", "expected_bot", "actual_bot_acts", "actual_bot_text", "turn_match",
                               "after_divergence", "arm"])
        w.writeheader()
        w.writerow({"case_id": "S1~base", "turn": 0, "state": "PRESENT_TXN", "customer_text": "yes that's mine",
                    "expected_tools": "record_charge_response(T1) -> FAILED", "actual_tools": "record_charge_response(T1) -> FAILED",
                    "expected_bot": "TRANSFER(record_failed)", "actual_bot_acts": "CONFIRM_RECORDED(T1)",
                    "actual_bot_text": "Recorded!", "turn_match": False, "after_divergence": False, "arm": "base"})
    page = write_case_comparison(d).read_text()
    for want in ("How a case is judged", "outcome signature", "Expected (oracle)", "Actual (bot)", "✗ differs",
                 "TRANSFERRED", "record_failed", "T1 = RECOGNIZED", "Why it failed", "claimed a response",
                 "data-ok=\"1\"", "All cases", "Recorded!"):
        assert want in page, want


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
