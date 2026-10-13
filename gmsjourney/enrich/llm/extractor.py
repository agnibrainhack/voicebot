"""A second, independent reader for LLM-written utterances (dual read).

The rule reader (enrich/reader.py) is the primary instrument of the proof. For the LLM
arm a second reader can be required to agree: an LLM call that sees only the utterance
and the agent's question - never the base meaning, the base wording or the factor
levels - and must fill the closed meaning-card schema of the ontology. Because it is
blind to the intended meaning, agreement between the two readers is evidence about the
text, not about the prompt that produced it.

It can only make the gate stricter: with ``dual_read="gate"`` a turn is accepted when
the rule reader AND this reader both recover the base card; with ``"report"`` the
agreement is measured and reported but does not gate.
"""
from __future__ import annotations

import re
from typing import Any

from ...ontology import SemanticFrame as F
from .client import CachedLLM, LLMUnavailable
from .prompts import build_extract_prompt, extraction_schema


def _coerce(slot: str, v: Any) -> Any:
    if v is None or v == "":
        return None
    if slot == "credential":
        d = re.sub(r"\D", "", str(v))
        return d or None
    if slot == "value":
        try:
            return str(int(float(str(v).replace("$", ""))))
        except ValueError:
            return None
    if slot == "days":
        try:
            return int(float(str(v)))
        except ValueError:
            return None
    if slot == "options":
        vals = v if isinstance(v, (list, tuple)) else [v]
        out = []
        for x in vals:
            try:
                out.append(int(float(str(x).replace("$", ""))))
            except ValueError:
                return None
        return tuple(out)
    if slot == "name":
        s = re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_")
        return s if s.endswith("_delivery") else f"{s}_delivery"
    return str(v)


# The slots each policy-question intent carries (the meaning cards the bases are built from). The reader
# may volunteer more ("subject=REPLACE_CARD" on a window check because the caller mentioned their card);
# those are not part of the question's meaning and are dropped, so they cannot fail the comparison.
KB_INTENT_SLOTS = {
    "value": ("attr", "entity"), "limit": ("key",), "plausibility": ("attr", "entity", "value"),
    "mcq": ("attr", "entity", "options"), "window_check": ("days",), "unknown_option": ("attr", "name"),
    "unstated_fact": ("attr", "subject"),
}
KB_DEFAULT_ATTR = {"plausibility": "fee", "mcq": "fee", "unstated_fact": "fee"}   # the only reading these have


class LLMFrameExtractor:
    def __init__(self, llm: CachedLLM, store):
        from ...knowledge.policy_kg import ACT_SCHEMA
        self.llm, self.store = llm, store
        self.schema = extraction_schema(store)
        self.acts = ACT_SCHEMA
        self.calls = 0
        self.failures = 0

    def read(self, text: str, state: str | None, ctx: dict[str, Any] | None = None,
             strict: bool = False) -> F | None:
        """The meaning card the independent reader recovers, or None (unavailable / unparsable).

        strict=True re-raises LLMUnavailable instead of returning None, so a *network* failure is
        never mistaken for "the reader recovered a different meaning"."""
        system, user = build_extract_prompt(text, state, ctx, self.schema)
        self.calls += 1
        try:
            obj = self.llm.invoke_json(system, user, salt="extract")
        except LLMUnavailable:
            self.failures += 1
            if strict:
                raise
            return None
        if not obj or obj.get("act") not in self.acts:
            self.failures += 1
            return None
        act = obj["act"]
        raw = obj.get("slots") or {}
        if not isinstance(raw, dict):
            raw = {}
        slots = {}
        for k, v in raw.items():
            if k not in self.acts[act]:
                continue
            c = _coerce(k, v)
            if c is not None:
                slots[k] = c
        if act == "ASK_KB":
            intent = slots.get("intent")
            keep = ("intent",) + KB_INTENT_SLOTS.get(intent, ())
            slots = {k: v for k, v in slots.items() if k in keep}
            if intent in KB_DEFAULT_ATTR:
                slots.setdefault("attr", KB_DEFAULT_ATTR[intent])
        # a value outside the slot's value set is not a reading of that slot
        for k in list(slots):
            vs = (self.schema.get(act, {}).get(k) or {}).get("values")
            if isinstance(vs, list) and vs and isinstance(slots[k], str) and slots[k] not in vs:
                del slots[k]
        return F.make(act, **slots)
