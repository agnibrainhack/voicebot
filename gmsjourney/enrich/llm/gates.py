"""Per-turn gates for LLM-written customer utterances.

The template arm is trusted to *construct* text from a meaning card; the LLM arm is
not, so every LLM utterance passes the gates below before the invariance proof (the
same L1-L4 the template arm passes) is even attempted. Gates on the raw LLM text:

G0 form       one spoken line: non-empty, bounded length, no speaker labels, no
              stage directions, no internal names (tool names, rule ids, act names,
              'JSON', 'as an AI') - a leak means the model saw or invented internals;
G1 facts      no figure, amount or merchant that the base wording, the scenario's
              template wording or the meaning card does not already carry (no
              hallucinated content, no answering its own question), and every
              'keep verbatim' token present;

and on the final utterance (after the deterministic surface-noise layer):

G2 meaning    (in validate.py) the rule reader's card in the dialog state equals the
              base card and is admissible there; optionally a second, independent
              LLM reader that never saw the base must agree;
G3 one clear  the utterance carries no more competing decision cues than the
   decision   requested presentation allows (a self-correction may carry two; nothing
              else may) - an utterance a careful agent could read two ways is not the
              same test even if our reader happens to pick the right reading.

``factor_realisation`` is a manipulation check, not a gate: it reports whether each
requested factor level is visible in the text, so attribution can use the levels that
actually reached the bot (an unrealised factor cannot cause a failure).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ...journey import MERCHANTS
from ...ontology import SemanticFrame as F
from ..realizers import BASE_LEVELS
from .prompts import READER_LEXICON, protected_tokens

MAX_CHARS = 600
MAX_WORDS = 90
_LEAK = re.compile(
    r"\b(authenticate_customer|get_review_transactions|record_charge_response|get_card_action_options|"
    r"submit_card_action|transfer_to_human|inform_disposition|provide_credential|request_card_action|ask_kb|"
    r"meaning card|json|utterance|as an ai|language model)\b", re.I)
_RULE_ID = re.compile(r"\bR\d{1,2}\b")
_STATUS = re.compile(r"\b(SUCCESS|AUTHENTICATED|FAILED|UNAVAILABLE)\b")
_SPEAKER = re.compile(r"(^|\n)\s*(customer|agent|bot|user|assistant|caller)\s*:", re.I)
_STAGE = re.compile(r"[\[\(]\s*(laughs|sighs|pause|whispers|coughs|clears throat)[^\]\)]*[\]\)]", re.I)
_FILLER = re.compile(r"\b(um+|uh+|erm|hmm+|you know|like,|i mean)\b", re.I)
_REPAIR = re.compile(r"(—|–|--|\.\.\.|\bwait\b|\bactually\b|\bi mean\b|\bsorry\b|\bno,? no\b)", re.I)
_URGENT = re.compile(r"\b(hurry|urgent|urgently|quickly|asap|right away|worried|panick|scared|please,? fast)\b", re.I)
_SKEPTIC = re.compile(r"\b(doubt|skeptic|sceptic|not sure (this|you) (will|can) help|won't help|waste of time|"
                      r"really\?|seriously)\b", re.I)
_CASUAL = re.compile(r"\b(hey|hi|yeah|yep|yup|gonna|wanna|kinda|cool|ok so|okay so)\b", re.I)
_INDIRECT = re.compile(r"\b(i suppose|i guess|i think|let me (check|think|see)|thinking (about|it over)|i was wondering|"
                       r"if i remember|as far as i)\b", re.I)
_DOWNPLAY = re.compile(r"\b(doesn'?t (really )?matter|just for the record|probably nothing|small thing|no big deal|"
                       r"just (a|one) (quick|small))\b", re.I)


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (t or "").lower())


def _content(t: str) -> set[str]:
    stop = {"the", "a", "an", "i", "you", "it", "is", "to", "and", "of", "for", "my", "me", "that", "this", "on",
            "in", "be", "so", "do", "was", "are", "with", "your", "just", "please", "can", "will", "what", "it's"}
    return {w for w in _words(t) if w not in stop and len(w) > 2}


def _recall(base: str, text: str) -> float:
    b = _content(base)
    return len(b & _content(text)) / len(b) if b else 1.0


@dataclass
class GateReport:
    ok: bool
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons)}


class LLMOutputGates:
    def __init__(self, store, reader):
        self.store, self.reader = store, reader
        self.tools = {t for t, ty in store.entity_types.items() if ty == "tool"}
        self.merchants = {m.lower() for m, _ in MERCHANTS}

    # ------------------------------------------------------------------ helpers
    def figures(self, text: str) -> set[str]:
        """Numbers a careful listener would hear, normalised like the reader does (spoken -> digits)."""
        t = self.reader.normalize(text or "").replace("<dot>", ".")
        out = set()
        for m in re.finditer(r"\d+(?:\.\d+)?", t):
            v = m.group(0)
            out.add(str(int(float(v))) if float(v).is_integer() else v)
            if "." in v:                                   # $45.67 also licenses 45 and 67 spoken separately
                a, b = v.split(".")
                out.update({str(int(a)), str(int(b))})
        return out

    # ------------------------------------------------------------------ G0
    def form(self, text: str) -> GateReport:
        r = []
        t = (text or "").strip()
        if not t:
            r.append("empty utterance")
        if len(t) > MAX_CHARS or len(_words(t)) > MAX_WORDS:
            r.append(f"too long ({len(_words(t))} words; max {MAX_WORDS})")
        if _SPEAKER.search(t):
            r.append("contains a speaker label")
        if "\n" in t.strip():
            r.append("more than one line")
        if _STAGE.search(t):
            r.append("contains a stage direction")
        if _LEAK.search(t) or any(tool in t for tool in self.tools):
            r.append("mentions an internal name")
        if _RULE_ID.search(t) or _STATUS.search(t):
            r.append("mentions a rule id or a tool status")
        return GateReport(not r, r)

    # ------------------------------------------------------------------ G1
    def facts(self, text: str, frame: F, ctx: dict[str, Any] | None, levels: dict[str, str],
              licensed_texts: list[str]) -> GateReport:
        r = []
        allowed = set()
        for s in licensed_texts:
            allowed |= self.figures(s)
        for k in ("credential", "value", "days"):
            if frame.get(k) is not None:
                allowed |= self.figures(str(frame.get(k)))
        for o in frame.get("options") or ():
            allowed |= self.figures(str(o))
        if ctx and ctx.get("amount"):
            allowed |= self.figures(f"${ctx['amount']}")
        new = sorted(self.figures(text) - allowed, key=lambda x: (len(x), x))
        if new:
            r.append(f"introduces figures not in the base meaning: {', '.join(new)}")
        low = (text or "").lower()
        own = str((ctx or {}).get("merchant") or "").lower()
        foreign = sorted(m for m in self.merchants if m != own and re.search(rf"\b{re.escape(m)}\b", low))
        if foreign:
            r.append(f"mentions another merchant: {', '.join(foreign)}")
        nt = self.reader.normalize(text or "")
        for tok in protected_tokens(frame, levels, ctx):
            if self.reader.normalize(tok) not in nt and tok.lower() not in low:
                r.append(f"drops the value '{tok}'")
        return GateReport(not r, r)

    # ------------------------------------------------------------------ G3
    def one_clear_decision(self, text: str, frame: F, levels: dict[str, str]) -> GateReport:
        cues = self.reader.cue_counts(text)
        lv = {**BASE_LEVELS, **(levels or {})}
        r = []
        if frame.act == "INFORM_DISPOSITION":
            allowed = 2 if (frame.get("corrected_from") or lv["self_correction"] != "none") else 1
            if cues["disposition"] > allowed:
                r.append(f"{cues['disposition']} competing disposition cues (allowed {allowed})")
        elif cues["disposition"] and frame.act in ("CONFIRM", "DENY", "REQUEST_CARD_ACTION", "END", "REQUEST_HUMAN",
                                                   "PROVIDE_CREDENTIAL"):
            r.append("carries a disposition cue the meaning does not have")
        if frame.get("delivery") or frame.get("entity"):
            if cues["delivery"] > 1 and lv["self_correction"] == "none":
                r.append(f"{cues['delivery']} delivery cues (one option is meant)")
        return GateReport(not r, r)

    # ------------------------------------------------------------------ manipulation check
    def factor_realisation(self, text: str, base_text: str, frame: F, levels: dict[str, str],
                           ctx: dict[str, Any] | None, asides: dict[str, str] | None = None) -> dict[str, dict]:
        """For every non-base level: detected True/False, or None when the factor cannot apply to this turn."""
        lv = {**BASE_LEVELS, **(levels or {})}
        asides = asides or {}
        t = text or ""
        low = t.lower()
        nb = len(_words(base_text)) or 1
        nw = len(_words(t))
        out: dict[str, dict] = {}

        def put(k, det, ev=""):
            out[k] = {"level": lv[k], "detected": det, "evidence": ev}

        for k, v in lv.items():
            if BASE_LEVELS.get(k) == v:
                continue
            if k == "noise":
                put(k, True, "applied deterministically after the LLM")
            elif k == "clarity":
                if frame.act in ("END", "UNCLEAR", "DENY") or lv["style"] == "terse":
                    put(k, None, "not applicable to this act/style")
                else:
                    rx = _INDIRECT if v == "indirect" else _DOWNPLAY
                    put(k, bool(rx.search(t)), "lead-in/aside marker")
            elif k == "entity_aliasing":
                vals = [frame.get(s) for s in ("disposition", "delivery", "card_action", "entity", "subject")]
                vals = [x for x in vals if x in READER_LEXICON]
                if not vals:
                    put(k, None, "no aliasable value in this turn")
                else:
                    canon = [READER_LEXICON[x][0].lower() for x in vals]
                    put(k, not any(c in low for c in canon), "official name avoided")
            elif k == "specificity":
                if frame.act == "ASK_KB":
                    put(k, bool(re.search(r"replacement|new card", low)), "replacement card named")
                elif (ctx or {}).get("merchant") and frame.act in ("INFORM_DISPOSITION", "PROVIDE_CREDENTIAL"):
                    put(k, str(ctx["merchant"]).lower() in low or str(ctx.get("amount")) in t
                        or "calling about" in low, "merchant/amount named")
                else:
                    put(k, None, "nothing specific to name")
            elif k == "numeric_format":
                has_num = bool(self.figures(base_text))
                put(k, (not re.search(r"\d", t)) if has_num else None, "numbers in words")
            elif k == "disfluency":
                put(k, bool(_FILLER.search(t)) if v == "fillers" else bool(_REPAIR.search(t)), "fillers/restart")
            elif k == "self_correction":
                if frame.act not in ("PROVIDE_CREDENTIAL", "INFORM_DISPOSITION", "CONFIRM") or frame.get("corrected_from"):
                    put(k, None, "not applicable to this act")
                else:
                    put(k, bool(_REPAIR.search(t)), "repair marker")
            elif k == "paraphrase_depth":
                rc = _recall(base_text, t)
                det = rc >= 0.75 if v == "none" else (0.3 <= rc < 0.95 if v == "light" else rc < 0.65)
                put(k, det, f"content-word recall vs base {rc:.2f}")
            elif k == "question_format":
                if frame.act not in ("ASK_KB", "REQUEST_HUMAN") or lv["style"] == "terse":
                    put(k, None, "not a question")
                elif v == "declarative":
                    put(k, "?" not in t, "no question mark")
                else:
                    put(k, "?" not in t and bool(re.search(r"\b(tell|give|let me know|explain|transfer|connect|put me)\b", low)),
                        "request form")
            elif k == "style":
                if v == "terse":
                    put(k, nw <= 6, f"{nw} words")
                elif v == "casual":
                    put(k, bool(_CASUAL.search(t)) or "'" in t, "casual markers")
                else:
                    put(k, None, "")
            elif k == "emotional_frame":
                rx = _URGENT if v == "urgent" else _SKEPTIC
                a = asides.get(k, "")
                put(k, bool(rx.search(t)) or (bool(a) and _recall(a, t) >= 0.4), "tone/aside")
            elif k == "length":
                put(k, nw >= (1.3 if v == "medium" else 1.8) * nb, f"{nw} vs base {nb} words")
            else:                                                     # history, instruction_conflict, anchoring, pressure
                a = asides.get(k, "")
                if not a:
                    put(k, None, "no aside for this act")
                else:
                    put(k, _recall(a, t) >= 0.4, f"aside content recall {_recall(a, t):.2f}")
        return out
