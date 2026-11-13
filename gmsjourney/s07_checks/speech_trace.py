"""How the speech check read one sentence, step by step: for the speech check page (speech_playground.html).

Input   the reader the verifier uses (``GenericReader`` for a compiled pack, ``UtteranceReader`` otherwise), one
        sentence, and the session's facts.
Output  ``explain_sentence(...)`` -> a JSON-ready dict:
          text        the sentence as said
          low         the words the reader works on: plain punctuation, number words as digits, lower case
          numbers     the number words turned into digits ([{words, digits}])
          cues        the cue words found, per list (question_or_condition, offer, completion, recording), with
                      where they are in ``low``
          mode        how the sentence is put: offered | done | states (the claims' mode)
          scans       one entry per reader that found something, in the order they run:
                        value sets  {reader, kind: "phrases", value_set, tried, hits: [{words, value, start, end,
                                    kept, blocked_by}]}: every phrase that matched, longest first, and the ones
                                    dropped because a longer phrase already took those words
                        patterns    {reader, kind: "pattern", field, pattern, hits: [{words, start, end, note}]}
          quiet       the names of the readers that ran and found nothing (``readers`` gives how many phrases each tried)

It re-runs the readers' own patterns on the same words, so it shows what the check saw; it decides nothing (the
claims and verdicts come from the reader and the verifier, unchanged).
"""
from __future__ import annotations

import re
from typing import Any

from . import speech_reading as _r
from .speech_reading import BUSINESS_DAYS, MONEY, _NUMBER_WORDS, plain, words_to_digits


def _hit(m: re.Match, **extra) -> dict[str, Any]:
    return {"words": m.group(0), "start": m.start(), "end": m.end(), **extra}


def _cue(rx: re.Pattern, s: str) -> dict | None:
    m = rx.search(s)
    return _hit(m) if m else None


def explain_sentence(reader, sent: str, facts=None) -> dict[str, Any]:
    """What the reader saw in ``sent`` (see the module docstring)."""
    from .speech_checks import AT_LEAST, UP_TO, GenericReader
    low = words_to_digits(sent).lower()
    generic = isinstance(reader, GenericReader)
    q = _r.QUESTION_OR_CONDITION.search(sent.strip())
    cues = {"question_or_condition": ({"words": q.group(0).strip() or "?", "start": None, "end": None,
                                       "why": "ends with a question mark" if q.group(0).strip().startswith("?")
                                       else "opens with a condition"} if q else None),
            "offer": _cue(_r.OFFER, low), "completion": _cue(_r.COMPLETION, low)}
    if not generic:
        cues["recording"] = _cue(_r.RECORDING_VERBS, low)
    mode = "offered" if cues["question_or_condition"] or cues["offer"] else "done" if cues["completion"] else "states"
    numbers = [{"words": m.group(0), "digits": words_to_digits(m.group(0))} for m in _NUMBER_WORDS.finditer(plain(sent))]

    scans: list[dict[str, Any]] = []
    sets = list(reader.m.value_sets) if generic else list(reader._value_phrases)
    if facts is not None:                              # the charges the tools returned, by merchant (or id)
        hits = []
        for tid, it in (getattr(facts, "items", None) or {}).items():
            for name in dict.fromkeys([str(it.get("merchant") or "").lower()] + ([str(tid).lower()] if generic else [])):
                if name:
                    hits += [_hit(m, value=tid) for m in re.finditer(r"(?<![\w$])" + re.escape(name) + r"(?![\w])", low)]
        scans.append({"reader": "charges the tools returned", "kind": "pattern", "field": "item" if generic else "merchant",
                      "pattern": "the merchant names (or ids) of the charges the tools returned; money near one is its amount",
                      "hits": sorted(hits, key=lambda x: x["start"])})
    for vs in sets:
        phrases = reader._value_phrases.get(vs, [])
        taken: list[tuple[int, int, str]] = []
        hits = []
        for phrase, value in phrases:
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
                by = next((w for s, e, w in taken if m.start() < e and s < m.end()), None)
                hits.append(_hit(m, value=value, phrase=phrase, kept=by is None, blocked_by=by))
                if by is None:
                    taken.append((m.start(), m.end(), m.group(0)))
        if not generic and vs == "disposition" and cues["question_or_condition"]:
            for x in hits:
                x["note"] = "not read: the sentence asks, it does not assert"
        scans.append({"reader": f"{vs} phrases", "kind": "phrases", "value_set": vs, "tried": len(phrases),
                      "longest": phrases[0][0] if phrases else None, "hits": sorted(hits, key=lambda x: x["start"])})

    days = [(m.start(), m.end()) for m in BUSINESS_DAYS.finditer(low)]
    money_field = (reader.m.money[0] if reader.m.money else None) if generic else "fee or amount"
    if money_field:
        scans.append({"reader": "money", "kind": "pattern", "field": money_field, "pattern": MONEY.pattern,
                      "hits": [_hit(m, value=m.group(1) or m.group(2), note=_charge_note(scans, m.start()))
                               for m in MONEY.finditer(low)]})
    fm = _r.FREE.search(low)
    need = (reader.m.numeric[money_field]["set"] if generic and money_field else None) if generic else "delivery"
    named = any(x["kept"] for s in scans if s.get("value_set") == need for x in s["hits"])
    scans.append({"reader": "free", "kind": "pattern", "field": money_field or "fee", "pattern": _r.FREE.pattern,
                  "hits": [_hit(fm, value=0, kept=named, note=None if named else
                                f"not read: the sentence names no {need or 'priced option'}")] if fm else []})
    if not generic:
        hits = [_hit(m) for rx in (_r.DATE_MD, _r.DATE_DM, _r.DATE_ISO) for m in rx.finditer(low)]
        scans.append({"reader": "dates", "kind": "pattern", "field": "date", "pattern": "month day / day month / yyyy-mm-dd",
                      "hits": hits})
        scans.append({"reader": "days ago", "kind": "pattern", "field": "age_days", "pattern": _r.AGE.pattern,
                      "hits": [_hit(m, value=int(m.group(1))) for m in _r.AGE.finditer(low)]})
    bd = []
    for m in BUSINESS_DAYS.finditer(low):
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        before = low[max(0, m.start() - 24):m.start()]
        bound = "at most (the maximum)" if UP_TO.search(before) and lo == hi else \
            "at least (the minimum)" if AT_LEAST.search(before) and lo == hi else None
        bd.append(_hit(m, value=[lo, hi], note=f"read as {bound}" if bound and generic else
                       "a range" if lo != hi else None))
    scans.append({"reader": "business days", "kind": "pattern",
                  "field": ", ".join(reader.m.days) if generic else "delivery_days", "pattern": BUSINESS_DAYS.pattern,
                  "hits": bd})
    for key, regs in reader._limit_res.items():
        hits = []
        for rx in regs:
            for m in rx.finditer(low):
                inside = generic and any(s <= m.start() < e for s, e in days)
                hits.append(_hit(m, value=int(m.group(1)), kept=not inside,
                                 note="not read: part of the business days figure" if inside else None))
        scans.append({"reader": f"limit: {key}", "kind": "pattern", "field": "limit", "about": key,
                      "pattern": " | ".join(rx.pattern for rx in regs), "hits": hits})
    if not generic:
        m = _r.RECORDED_COUNT.search(low)
        scans.append({"reader": "recorded count", "kind": "pattern", "field": "recorded_count",
                      "pattern": _r.RECORDED_COUNT.pattern, "hits": [_hit(m, value=int(m.group(1)))] if m else []})
    quiet = [x["reader"] for x in scans if not x["hits"]]
    cues = {k: v for k, v in cues.items() if v}
    return {"text": sent, "low": low, **({"numbers": numbers} if numbers else {}), "cues": cues, "mode": mode,
            "scans": [x for x in scans if x["hits"]], "quiet": quiet, **({} if generic else {"legacy": True})}


def _charge_note(scans: list[dict], pos: int) -> str | None:
    """Money said near a named charge is read as that charge's amount (s07_checks/speech_checks.GenericReader)."""
    named = next((x["hits"] for x in scans if x.get("field") == "item"), [])
    if not named:
        return None
    near = min(named, key=lambda h: abs(h["start"] - pos))
    return f"read as the amount of {near['value']} (named nearest), unless a priced option is named nearer"


def readers(reader) -> dict[str, int]:
    """How many phrases each value-set reader tries (the same for every sentence: kept once per turn)."""
    return {f"{vs} phrases": len(ps) for vs, ps in reader._value_phrases.items()}


def facts_snapshot(facts, frames: list[Any]) -> dict[str, Any]:
    """What the checker knew when the agent spoke: the customer's meaning cards, the tools' results and writes."""
    def short(v: Any) -> Any:
        s = repr(v)
        return v if len(s) <= 160 else s[:157] + "..."
    out = {"customer": [{"turn": i, "card": str(fr), "act": getattr(fr, "act", None),
                         "slots": {k: v for k, v in getattr(fr, "slots", ()) if v is not None}}
                        for i, fr in enumerate(frames)],
           "caller": dict(facts.caller),
           "writes": [{"tool": t, "args": dict(w.get("args") or {}), "status": (w.get("result") or {}).get("status")}
                      for t, ws in facts.writes.items() for w in ws],
           "failed_writes": [{"tool": t, "args": dict(w.get("args") or {}),
                              "status": (w.get("result") or {}).get("status") if isinstance(w.get("result"), dict) else None}
                             for t, ws in facts.failed_writes.items() for w in ws],
           "results": {t: {k: short(v) for k, v in r.items()} for t, r in facts.results.items()},
           "items": {tid: {k: v for k, v in it.items() if k != "transaction_id"} for tid, it in facts.items.items()}}
    if facts.options is not None:
        out["options"] = list(facts.options)
    return out
