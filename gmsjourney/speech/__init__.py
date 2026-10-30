"""Validating what the bot actually says, sentence by sentence, against the policy graph and the GMS.

Why
---
A bot hands the harness two things every turn: act labels (``CONFIRM_RECORDED(T1, UNRECOGNIZED)``)
and the words the customer hears. Outcome and trajectory grading read the labels and the tool log.
The words were never re-read, so a bot that labels everything correctly but *tells* the customer
something else ("I've noted that you recognize the Amazon charge", "express delivery is $25") passed.
This package reads the words on their own and checks every fact they state.

The flow for one bot turn
-------------------------
1. ``SessionFacts`` (session.py) holds what is true in this conversation so far: the transactions
   the tool returned, every write that truly succeeded, the card options offered, and what the
   customer said (as the runner's reader understood it).
2. ``UtteranceReader`` (reader.py) reads the text, ignoring the labels, into ``Claim``s: one fact
   each ("amount = 45.67 about T1", "fee = 25 about expedited_delivery", "disposition recorded =
   RECOGNIZED about T1"), plus the acts its words express.
3. ``UtteranceVerifier`` (verify.py) checks each claim where its truth lives:
     account facts       -> SessionFacts (what the tools returned / what was written)
     policy facts        -> the GMS: exact register (lookup_enm), with the trained geometry's
                            distance (score_triple vs the relation's cap) reported alongside
     rule stances        -> the GMS stance check (u-space tension, calibrated cuts)
     the customer        -> what the customer said
   and checks the turn as a whole: does the text say what the labels claim (faithfulness), does
   each act say what the policy requires it to say (completeness), and is each act allowed at
   this point of the dialog-state graph (flow).
4. The result, a ``TurnReport``, lists every claim with its verdict and every finding.

What it reads from the policy graph (policy/charge_review_kg.json, kind "speech")
----------------------------------------------------------------------------------
* ``ACT says FIELD``  (from = where the truth lives, required)   e.g. PRESENT_TXN says date, from item.date
* ``ACT spoken_cue W`` words that mark an act with no content     e.g. ASK_MORE_HELP "anything else"
* ``VALUE spoken_as W`` how the bot words a value                 e.g. UNRECOGNIZED "isn't yours"
* ``TOOL returns_field F``                                        e.g. get_review_transactions date
* ``RULE asserted_by / denied_by W``                              e.g. R27 "means fraud" / "not proof of fraud"
plus the graph's value sets, synonyms, opening statement and dialog-state transitions.

Usage
-----
    from gmsjourney.speech import SpeechContract, SessionFacts, UtteranceVerifier

    verifier = UtteranceVerifier(store, SpeechContract.from_graph(store.kg))
    facts = SessionFacts.from_run(tool_log, customer_frames, session_date="2026-03-16")
    report = verifier.check_turn(acts, text, facts, previous_acts=prev, customer_act="INFORM_DISPOSITION")
    report.ok              # True when every claim is supported and there is no finding
    report.as_dict()       # every claim, its verdict and truth, every finding (for reports)

The campaign runs this on every bot turn of every scenario (sut/adapter.py) and grades the run on
it (evaluation/runner.py: ``c_speech`` and the ``speech_*`` columns).
"""
from __future__ import annotations

from .contract import ActContract, FieldSpec, SpeechContract
from .reader import Claim, ReadUtterance, UtteranceReader
from .session import SessionFacts, Truth
from .verify import ClaimVerdict, Finding, TurnReport, UtteranceVerifier

__all__ = ["ActContract", "FieldSpec", "SpeechContract", "Claim", "ReadUtterance", "UtteranceReader",
           "SessionFacts", "Truth", "ClaimVerdict", "Finding", "TurnReport", "UtteranceVerifier"]
