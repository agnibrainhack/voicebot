"""A stand-in for the gateway model, for tests and demos of the LLM bot without network access.

    GMSJOURNEY_SUT_FAKE_MODEL=fake_sut_model:HappyPath   (with tests/ on PYTHONPATH)

The LLM bot (gmsjourney/s06_bots/llm_bot.py) asks its model two things, and this answers both from the JSON it is sent:

* reading a customer turn (a request carrying "menu"): the reference bot's keyword reader on the active journey's
  graph gives the intent and slots; thanks and small talk are "social"; a policy figure stated with "right?" is a
  "claim"; anything the keyword reader cannot place is "unclear";
* wording the bot's acts (a request carrying "lines"): the template lines unchanged, with a short acknowledgement as
  the lead after a social turn.

``calls``, ``readings`` and ``wordings`` count what it was asked.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace


def _r(obj):
    return SimpleNamespace(content=json.dumps(obj), usage_metadata={})


SOCIAL = re.compile(r"^\W*(thanks|thank you|cheers|sorry|hello|hi|hey|good (morning|afternoon|evening)|lovely weather|"
                    r"how are you|hope you'?re|my (dog|kid|kids|son|daughter)|one (sec|second|moment)|hold on)\b", re.I)


class HappyPath:
    def __init__(self):
        self.calls = self.readings = self.wordings = 0

    def invoke(self, msgs):
        self.calls += 1
        payload = json.loads(msgs[-1].content)
        if "menu" in payload:
            self.readings += 1
            return _r(self.read(payload))
        self.wordings += 1
        lead = "Thanks for that." if payload.get("customer_turn") == "social" else ""
        return _r({"lead": lead, "lines": [x["template"] for x in payload["lines"]]})

    def read(self, p):
        from gmsjourney.s01_packs.loading import current
        from gmsjourney.s06_bots.reference import KeywordReader
        g = current().graph
        text, menu = p.get("customer") or "", p["menu"]
        state = p.get("state") or ""
        f = KeywordReader(g).read(text, state)
        q = (g.glob.get("question") or {}).get("act")
        if f.act == q:
            about = (f"value:{f.get('attr')}:{f.get('entity')}" if f.get("intent") == "value" else
                     f"limit:{f.get('key')}" if f.get("intent") == "limit" else "other")
            if re.search(r"\b(right|correct|isn't it|is that so)\s*\?", text, re.I) and re.search(r"\d", text):
                return {"kind": "claim", "about": about, "claimed": text}
            return {"kind": "question", "about": about}
        if f.act in menu["intents"]:
            return {"kind": "actionable", "intent": f.act, "slots": dict(f.slots)}
        if SOCIAL.search(text):
            return {"kind": "social"}
        return {"kind": "unclear"}


class Misreads(HappyPath):
    """A reading model with a defect: every answer about a charge is read as RECOGNIZED (fake_sut_model:Misreads)."""

    def read(self, p):
        out = super().read(p)
        if out.get("kind") == "actionable" and "disposition" in (out.get("slots") or {}):
            out["slots"]["disposition"] = "RECOGNIZED"
        return out
