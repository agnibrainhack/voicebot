"""The domain lexicon: how customers say things and how their words are read back.

One source (``domain.yaml`` -> ``lexicon``) for the realizer, the frame reader, the noise
operators and the negative controls, so the generator and the instrument of the
invariance proof cannot drift apart. ``Lexicon.current()`` is the active domain's lexicon.
"""
from __future__ import annotations

import re
from functools import cached_property
from typing import Any


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def low(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s.startswith("I ") and not s.startswith("I'") else s


def fill(template: str, **fields: Any) -> str:
    """str.format with a capitalised twin of every string field ({phrase} -> {Phrase})."""
    data = dict(fields)
    for k, v in fields.items():
        if isinstance(v, str) and k[:1].islower():
            data.setdefault(cap(k), cap(v))
    return template.format(**data)


def cue_regex(cues: list[str]) -> str:
    """A cue list as one word-bounded alternation, the form the reader matches."""
    return r"\b(" + "|".join(cues) + r")\b"


class Lexicon:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg or {}

    _cache: dict[int, "Lexicon"] = {}

    @classmethod
    def current(cls) -> "Lexicon":
        from ..domain import current
        dom = current()
        key = id(dom)
        if key not in cls._cache:
            cls._cache[key] = cls(dom.section("lexicon"))
        return cls._cache[key]

    # ------------------------------------------------------------------ sections
    def __getitem__(self, key: str) -> Any:
        return self.cfg[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.cfg.get(key, default)

    @property
    def values(self) -> dict[str, dict]:
        return self.cfg.get("values", {})

    @property
    def acts(self) -> dict[str, dict]:
        return self.cfg.get("acts", {})

    @property
    def slots(self) -> dict[str, dict]:
        return self.cfg.get("slots", {})

    # ------------------------------------------------------------------ saying
    def phrase(self, value: str, level: str, **fields: Any) -> str:
        return fill(self.values[value]["say"][level], **fields)

    # ------------------------------------------------------------------ reading
    @cached_property
    def value_patterns(self) -> dict[str, str]:
        return {v: cue_regex(spec["cues"]) for v, spec in self.values.items() if spec.get("cues")}

    @cached_property
    def act_patterns(self) -> dict[str, str]:
        return {a: cue_regex(spec["cues"]) for a, spec in self.acts.items() if spec.get("cues")}

    def slot_of(self, value: str) -> str | None:
        for slot, spec in self.slots.items():
            if value in spec.get("order", []):
                return slot
        return None

    @cached_property
    def speech_confusions(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in (self.cfg.get("speech", {}).get("confusions") or {}).items()}

    @cached_property
    def speech_repairs(self) -> list[tuple[str, str]]:
        return [(p, r) for p, r in (self.cfg.get("speech", {}).get("repairs") or [])]

    @cached_property
    def protect(self) -> set[str]:
        return set(self.cfg.get("protect", []))

    @cached_property
    def vocabulary(self) -> list[str]:
        v = self.cfg.get("vocabulary", "")
        return v.split() if isinstance(v, str) else list(v)

    @cached_property
    def kb(self) -> dict[str, Any]:
        return self.cfg.get("kb_questions", {})

    @cached_property
    def option_name_regex(self) -> re.Pattern:
        nouns = "|".join(self.kb.get("option_nouns", {}).values())
        return re.compile(r"\b([a-z]+(?:[ -][a-z]+)?) (?:" + nouns + r")\b")
