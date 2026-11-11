"""Campaign plumbing around the bot under test: sampling the base cases, reading failure reasons from the graded
rows, and the case comparison page from a campaign's files. (The LLM bot itself: tests/test_structured_bot.py.)

Run with `pytest -q tests/` or `python tests/test_llm_sut.py`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _store import kx_store  # noqa: E402
from gmsjourney.s02_graph.flow import GraphOracle as PolicyOracle  # noqa: E402

POLICY_KG = Path(__file__).resolve().parents[1] / "gmsjourney" / "policy" / "charge_review" / "charge_review_kg.json"


@lru_cache(maxsize=1)
def env():
    store = kx_store()
    return store, PolicyOracle(store)


def test_sample_bases_keeps_the_category_mix_and_all_by_default():
    from gmsjourney.s08_campaign.pipeline import sample_bases
    B = [SimpleNamespace(qid=f"{c}{i}", base=c) for c, n in (("a", 6), ("b", 3), ("c", 1)) for i in range(n)]
    assert sample_bases(B, None) == B and sample_bases(B, 99) == B
    pick = sample_bases(B, 6)
    assert len(pick) == 6 and {b.base for b in pick} == {"a", "b", "c"}
    assert sum(b.base == "a" for b in pick) >= sum(b.base == "b" for b in pick)


def test_failure_reasons_read_the_graded_row():
    from gmsjourney.s09_reports.report import failure_reasons, _strip
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
    from gmsjourney.s09_reports.cases import write_case_comparison
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
