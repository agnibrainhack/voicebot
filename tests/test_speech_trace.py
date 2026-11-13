"""The speech check's step-by-step trace (s07_checks/speech_trace.py), shown by the speech check page.

Every claim keeps the rules tried to find its truth (``steps``) and the comparison made (``compare``); every turn keeps
what the checker knew (the customer's meaning cards, the confirmed writes), each sentence as read (cues, mode, every
reader's matches, kept and dropped), and the faithfulness, completeness and flow checks it made, passing ones too.
The trace only reports: claims and verdicts are unchanged by it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_turn_grading import _bench, _runner, _scenario  # noqa: E402


def _misread_turn():
    from fake_sut_model import Misreads
    from gmsjourney.llm import CachedLLM, LLMSettings
    from gmsjourney.s06_bots.llm_bot import make_structured_factory
    from gmsjourney.s07_checks.scoring import score
    PG, B = _bench()
    r = CachedLLM(LLMSettings(model="fake-Misreads", temperature=0, max_tokens=200, json_mode=True), model=Misreads(), tag="sut")
    fac = make_structured_factory(B.store, reader_llm=r, wording_llm=None)
    scn = _scenario("present_txn_disposition=unrecognized")
    res = _runner(fac).run(scn)
    row = score(scn, res, B.oracle.limits, B.oracle.store.workflow)
    turn = next(t for t in res.extras["speech"]["turns"] if "CONFIRM_RECORDED" in t["acts"])
    return turn, row


def test_each_claim_keeps_the_rules_tried_and_the_comparison():
    turn, _ = _misread_turn()
    said_back, summary = [c for c in turn["claims"] if c["field"] == "disposition"]
    assert [(c["field"], c["status"]) for c in turn["claims"] if c["field"] != "disposition"] == \
        [("item", "supported"), ("amount", "supported")] * 2      # the charge, named by its details, and its amount
    assert said_back["status"] == "contradicted" and summary["status"] == "supported"
    rules = [s["rule"] for s in said_back["steps"]]
    assert said_back["steps"][-1] == {"rule": "what the customer said", "hit": True,
                                      "detail": "the customer's meaning card gave disposition = UNRECOGNIZED (facts.caller.disposition)"}
    assert not any(s["hit"] for s in said_back["steps"][:-1]) and "a summary of what was written" in rules
    assert said_back["compare"] == {"op": "equals, ignoring case", "said": "recognized", "truth": "unrecognized", "result": False}
    assert summary["steps"][-1]["rule"] == "a summary of what was written" and summary["steps"][-1]["hit"]
    assert summary["compare"]["op"] == "is one of" and summary["compare"]["result"] is True
    assert summary["basis"] == "successful write"            # where the check read it, not the field's default


def test_the_turn_keeps_what_was_known_and_how_each_sentence_was_read():
    turn, _ = _misread_turn()
    tr = turn["trace"]
    assert tr["split"] == "per_act" and [g["act"] for g in tr["segments"]] == ["CONFIRM_RECORDED", "SUMMARIZE", "ASK_MORE_HELP"]
    f = tr["facts"]
    assert f["caller"]["disposition"] == "UNRECOGNIZED" and f["customer"][-1]["slots"] == {"disposition": "UNRECOGNIZED"}
    assert f["writes"] == [{"tool": "record_charge_response", "args": {"transaction_id": "T1", "disposition": "RECOGNIZED"},
                            "status": "SUCCESS"}]
    s0 = tr["segments"][0]["sentences"][0]
    assert s0["low"] == "thanks, i've noted the amazon charge for $45.67 as recognized." and s0["mode"] == "states" \
        and s0["cues"] == {}
    hit = next(x for x in s0["scans"] if x.get("value_set") == "disposition")["hits"][0]
    assert (hit["words"], hit["value"], hit["start"], hit["end"], hit["kept"]) == ("recognized", "RECOGNIZED", 51, 61, True)
    assert "free" in s0["quiet"] and tr["readers"]["disposition phrases"] > 10
    assert tr["segments"][0]["claims"] == [0, 1, 2] and tr["segments"][1]["claims"] == [3, 4, 5] \
        and tr["segments"][2]["claims"] == []
    ask = tr["segments"][2]["sentences"][0]
    assert ask["mode"] == "offered" and ask["cues"]["offer"]["words"] == "i can" and ask["cues"]["question_or_condition"]
    # the checks of the turn as a whole, passing ones too
    assert tr["segments"][0]["faithfulness"][0]["ok"] and any(c["check"].endswith("says the disposition") and c["ok"]
                                                              for c in tr["segments"][0]["completeness"])
    assert all(c["ok"] for c in tr["flow"]) and tr["flow_acts"] == ["CONFIRM_RECORDED", "SUMMARIZE", "ASK_MORE_HELP"]


def test_the_case_row_keeps_every_turns_grade_and_the_cards_keep_the_trace():
    _, row = _misread_turn()
    grades = row["speech_turn_grades"]
    assert [g["turn"] for g in grades] == list(range(len(grades))) and not all(g["core_ok"] for g in grades)
    assert row["c_speech"] == 0 and all("trace" in c for c in row["speech_turns"])


def test_longer_phrases_take_the_words_first_and_the_trace_shows_what_was_dropped():
    from gmsjourney.s07_checks.speech_reading import SessionFacts
    from gmsjourney.s07_checks.speech_trace import explain_sentence
    PG, B = _bench()
    reader = B.speech.reader
    s = explain_sentence(reader, "No, that's not mine.", SessionFacts())
    hits = next(x for x in s["scans"] if x.get("value_set") == "disposition")["hits"]
    kept = [h for h in hits if h["kept"]]
    dropped = [h for h in hits if not h["kept"]]
    assert [h["value"] for h in kept] == ["UNRECOGNIZED"]
    assert dropped and all(h["blocked_by"] == kept[0]["words"] for h in dropped)
    assert {h["words"] for h in dropped} == {"not mine", "mine"}          # "mine" alone would mean RECOGNIZED
    assert [c.value for c in reader.read("No, that's not mine.", SessionFacts()).claims] == ["UNRECOGNIZED"]


def test_limits_with_a_two_word_unit_are_read():
    from gmsjourney.s07_checks.speech_reading import SessionFacts
    PG, B = _bench()
    r = B.speech.reader.read("The limit is 30 calendar days.", SessionFacts())
    assert [(c.field, c.value, c.about) for c in r.claims] == [("limit", 30, "lookback_days")]
    assert B.speech.verify_claim(r.claims[0], SessionFacts()).status == "supported"


def test_the_page_has_every_step():
    page = (Path(__file__).resolve().parents[1] / "gmsjourney" / "s09_reports" / "speech_playground.html").read_text()
    for step in ("What the check knows", "Pieces and sentences", "Word by word", "How it is put", "Readers and matches",
                 "Claim cards", "Where the truth is", "Said vs truth", "Acts: labelled vs heard", "Required wording",
                 "Dialog flow", "This response", "The case"):
        assert f'["{step}",' in page
