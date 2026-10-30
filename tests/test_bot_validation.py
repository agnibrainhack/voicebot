"""The bot validation page (gmsjourney/bot_validation.py) and the response cards it reads
(evaluation/runner.response_cards), on small hand-made rows: no store, no campaign.

    python tests/test_bot_validation.py
    pytest -q tests/test_bot_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsjourney import bot_validation as bv  # noqa: E402
from gmsjourney.evaluation.runner import response_cards  # noqa: E402

R = {"design": {"factors": ["noise", "disfluency"], "levels": {"noise": ["none", "asr"], "disfluency": ["none", "fillers"]},
                "level_provenance": [{"factor": "noise", "origin": "official"},
                                     {"factor": "disfluency", "origin": "custom"}]},
     "diagnosis": {"outcome_attribution": [{"factor": "noise", "p_value_adj": 0.001, "significant_adj": True},
                                           {"factor": "disfluency", "p_value_adj": 0.4, "significant_adj": False}]},
     "results": {"bot": "test bot"}}

FEE_OK = {"field": "fee", "value": 15, "about": "expedited_delivery", "mode": "states", "words": "$15",
          "status": "supported", "basis": "GMS exact register", "truth": 15, "source": "register", "why": "",
          "geometry": {"distance": 0.01, "cap": 0.2, "inside_cap": True, "knowlytix": "confirmed"}}
FEE_BAD = {**FEE_OK, "value": 0, "words": "$0", "status": "contradicted", "why": "expedited_delivery costs $15",
           "geometry": {"distance": 0.5, "cap": 0.2, "inside_cap": False}}
AMOUNT = {"field": "amount", "value": "45.67", "about": "T1", "mode": "states", "words": "$45.67",
          "status": "supported", "basis": "tool result", "truth": "45.67", "source": "tool", "why": ""}
DAYS = {"field": "delivery_days", "value": [2, 2], "about": "expedited_delivery", "mode": "states",
        "words": "2 business days", "status": "supported", "basis": "GMS exact register", "truth": [2, 2],
        "source": "register", "why": "", "geometry": {"distance": 0.02, "cap": 0.2, "inside_cap": True,
                                                      "knowlytix": "confirmed"}}


def card(turn, text, claims, findings=(), ok=True):
    return {"turn": turn, "where": f"the reply to customer turn {turn - 1}", "customer": "how much is it?",
            "text": text, "acts": ["ANSWER_KB"], "ok": ok, "flow_ok": True, "claims": list(claims),
            "findings": list(findings)}


def row(sid, arm, outcome, speech, noise=None, disfluency=None, cards=(), base="recovery"):
    r = {"sid": sid, "arm": arm, "base": base, "outcome_correct": int(outcome), "c_speech": int(speech),
         "correct": int(outcome and speech), "workflow_adherent": 1, "c_record": int(outcome), "speech_turns": list(cards)}
    if arm != "base":
        r.update({"f_noise": noise, "f_disfluency": disfluency})
    return r


ROWS = [
    row("J1~base", "base", True, True, cards=[card(1, "Expedited delivery has a $15 fee and arrives in 2 business days.",
                                                    [FEE_OK, DAYS])]),
    row("J2~base", "base", True, False, cards=[card(1, "Expedited delivery costs $0.", [FEE_BAD], ok=False)]),
    row("J1~0001", "template", True, True, "none", "fillers", [card(1, "I see a charge at Amazon for $45.67.", [AMOUNT])]),
    row("J1~0002", "template", False, True, "asr", "none"),
    row("J2~0003", "template", True, False, "asr", "fillers",
        [card(2, "Expedited delivery costs $0.", [FEE_BAD], ok=False)]),
    row("J2~0004~llm", "llm", False, False, "asr", "none",
        [card(2, "Here's a summary.", [], [{"kind": "missing", "act": "SUMMARIZE", "detail": "no record named",
                                             "rule": "pdf.2F"}], ok=False)]),
]


def test_three_rates_and_the_two_by_two():
    rt = bv.rates(ROWS)
    assert rt["n"] == 6 and abs(rt["did_right"] - 4 / 6) < 1e-9 and abs(rt["said_right"] - 3 / 6) < 1e-9
    assert abs(rt["correct"] - 2 / 6) < 1e-9
    assert bv.two_by_two(ROWS) == {"both": 2, "did_not_said": 2, "said_not_did": 1, "neither": 1}
    print("ok test_three_rates_and_the_two_by_two")


def test_performance_by_what_the_case_contained():
    perf = bv.factor_performance(R, ROWS)
    assert [x["level"] for x in perf["arm"]] == ["base", "template", "llm"]
    assert [f["factor"] for f in perf["doe"]] == ["noise"] and [f["factor"] for f in perf["enrichment"]] == ["disfluency"]
    noise = perf["doe"][0]
    assert noise["significant"] and noise["baseline"] == "none"
    assert [(x["level"], x["n"]) for x in noise["levels"]] == [("none", 1), ("asr", 3)]   # enriched runs only
    asr = noise["levels"][1]
    assert abs(asr["did_right"] - 1 / 3) < 1e-9 and abs(asr["said_right"] - 1 / 3) < 1e-9 and asr["correct"] == 0
    print("ok test_performance_by_what_the_case_contained")


def test_truth_basis_and_gms_readings():
    bb = {b["basis"]: b for b in bv.basis_breakdown(ROWS)}
    assert bb["GMS exact register"]["claims"] == 4 and bb["GMS exact register"]["contradicted"] == 2
    assert bb["tool result"]["supported"] == 1
    g = {k["kind"]: k for k in bv.gms_figures(ROWS)["kinds"]}
    fees = g["Fees"]
    assert fees["claims"] == 3 and fees["supported"] == 1 and fees["contradicted"] == 2
    assert fees["inside"] == 1 and fees["outside"] == 2 and fees["agree"] == 3     # geometry agrees every time
    assert fees["kx_confirmed"] == 1 and fees["kx_not_asked"] == 2
    ex = bv.gms_figures(ROWS)["examples"]
    assert len(ex) == 1 and ex[0]["words"] == "$0" and ex[0]["geometry"]["inside_cap"] is False   # one per wording
    print("ok test_truth_basis_and_gms_readings")


def test_failing_cards_are_grouped_and_span_the_arms():
    groups = bv.failing_cards(ROWS, per_group=2)
    top = groups[0]
    assert (top["kind"], top["what"], top["responses"], top["cases"]) == ("contradicted", "fee", 2, 2)
    assert {r["arm"] for r, _ in top["cards"]} == {"base", "template"}
    assert ("missing", "SUMMARIZE") in {(g["kind"], g["what"]) for g in groups}
    ex = bv.example_card(ROWS)
    assert ex is not None and ex[0]["sid"] == "J1~base"
    print("ok test_failing_cards_are_grouped_and_span_the_arms")


def test_the_page_renders_and_escapes():
    hostile = card(1, "<script>x</script>", [], [{"kind": "label_not_said", "act": "<b>ACT</b>", "detail": "<i>d</i>",
                                                   "rule": None}], ok=False)
    rows = ROWS + [row("J9~base", "base", True, False, cards=[hostile])]   # its own problem group, so it is shown
    html = bv.render_bot_validation_html(R, rows)
    for part in ("Bot validation", "Performance by what the test case contained", "DOE factors (official catalogs)",
                 "Enrichment factors (added for this bot)", "The semantic check in detail", "Checked against the GMS",
                 "Bot response cards", "noise = asr", "enrichment: LLM", "outside"):
        assert part in html, part
    assert "<script>x</script>" not in html and "&lt;script&gt;x" in html and "<i>d</i>" not in html
    s = bv.validation_summary(R, rows)
    assert s["by_arm"]["base"]["n"] == 3 and s["significant_factors"][0]["factor"] == "noise"
    print("ok test_the_page_renders_and_escapes")


def test_response_cards_keep_what_the_customer_said():
    turns = [{"turn": 0, "text": "Hello.", "acts": ["OPENING"], "ok": True, "flow_ok": True, "claims": [],
              "findings": []},
             {"turn": 1, "text": "It costs $15.", "acts": ["ANSWER_KB"], "ok": True, "flow_ok": True,
              "claims": [FEE_OK], "findings": []}]
    tx = [{"role": "bot", "text": "Hello."}, {"role": "user", "text": "how much is expedited?"},
          {"role": "bot", "text": "It costs $15."}]
    cards = response_cards(turns, tx)
    assert len(cards) == 1 and cards[0]["customer"] == "how much is expedited?"
    assert cards[0]["where"] == "the reply to customer turn 0" and cards[0]["claims"][0]["basis"] == "GMS exact register"
    print("ok test_response_cards_keep_what_the_customer_said")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
