"""Reading a bot utterance into claims, without looking at the bot's act labels.

Input   the text the customer hears, the session facts (to recognise the merchants that were
        returned), the speech contract (how values and acts are worded, from the policy graph)
        and, optionally, the transaction the act is about (used only when a sentence states an
        account fact without naming the merchant: "It was $45.67, 3 days ago.").
Output  a ``ReadUtterance``: the sentences, the ``Claim``s they make, and the acts the words
        express (``heard``).

A claim is one fact the words state, e.g. for "Thanks, I've noted that you don't recognize the
$45.67 Amazon charge." the reader returns

    Claim(field="amount",      value="45.67",        about="T1", mode="states",   words="$45.67")
    Claim(field="merchant",    value="Amazon",       about="T1", mode="states",   words="Amazon")
    Claim(field="disposition", value="UNRECOGNIZED", about="T1", mode="recorded", words="don't recognize")

``mode`` says how the fact is put: "states" (asserted), "recorded" (said to have been recorded),
"done" (an action said to be completed), "offered" (an action proposed in a question). A question
or an "if ..." sentence asserts no disposition and no completion ("Do you recognize it?" is not a
claim that the customer recognized it), but the figures it carries are still claims ("Shall I send
it by expedited delivery for $25?" states a $25 fee).

How each fact is found (all case-insensitive; numbers spoken as words are read as digits):
  merchant        a returned transaction's merchant name
  amount          a dollar figure that is not about a fee         -> the nearest named merchant's transaction
  date            "March 13", "13 March", "2026-03-13"            -> the nearest named merchant's transaction
  age_days        "3 days ago"                                    -> the nearest named merchant's transaction
  fee             a dollar figure (or "free") in a sentence about a fee or a delivery option -> that option
  delivery_days   "2 business days", "5-7 business days"          -> the nearest delivery option named
  limit           the graph's limit wordings ("up to {n} charges") -> that limit
  disposition / action / delivery   the graph's wordings of each value (bot phrasings + customer synonyms)
  stance          the graph's asserting / denying wordings of a rule's fact (denials read first)
  opening         the required opening statement, word for word
  recorded_count  "I recorded 2 responses"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .contract import SpeechContract
from .session import SessionFacts

# ------------------------------------------------------------------ plain-English signals (not policy)
QUESTION_OR_CONDITION = re.compile(r"\?\s*$|^\s*(if|when|unless|in case)\b", re.I)
RECORDING_VERBS = re.compile(r"\b(recorded|i've noted|i have noted|noted that|noted it|noted the|saved|logged|"
                             r"marked|captured|i've put|i have put)\b", re.I)
COMPLETION = re.compile(r"\b(has been|have been|is done|that's done|it's done|all done|done\b|completed|is complete|"
                        r"i've (?:secured|locked|replaced|frozen|blocked|ordered|sent)|"
                        r"i have (?:secured|locked|replaced|frozen|blocked|ordered|sent)|is now)\b", re.I)
OFFER = re.compile(r"\b(shall i|should i|would you like|do you want|can i|may i|i can|want me to)\b", re.I)
FEE_WORDS = re.compile(r"\b(fee|fees|cost|costs|price|pay|charge for|charges? you)\b", re.I)
FREE = re.compile(r"\b(free|no fee|no charge|at no cost)\b", re.I)
MONEY = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)|\b(\d+(?:\.\d{1,2})?)\s*dollars?\b", re.I)
BUSINESS_DAYS = re.compile(r"\b(\d+)(?:\s*(?:-|–|to)\s*(\d+))?\s+business\s+days?\b", re.I)
AGE = re.compile(r"\b(\d+)\s+days?\s+ago\b", re.I)
RECORDED_COUNT = re.compile(r"\brecorded\s+(\d+)\s+(?:responses?|answers?)\b", re.I)
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                          "nov", "dec"])}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_MD = re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
DATE_DM = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}(?:,?\s+(\d{{4}}))?", re.I)
DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_NUMBER_WORDS = re.compile(r"\b(" + "|".join(_TENS) + r")(?:[ -](" + "|".join(list(_UNITS)[1:10]) + r"))?\b|\b("
                           + "|".join(_UNITS) + r")\b", re.I)


def words_to_digits(text: str) -> str:
    """'fifteen dollars' -> '15 dollars', 'twenty-five' -> '25' (0-99; enough for spoken figures)."""
    def sub(m: re.Match) -> str:                       # one number word (or tens-units pair) -> digits
        if m.group(1):
            return str(_TENS[m.group(1).lower()] + (_UNITS[m.group(2).lower()] if m.group(2) else 0))
        return str(_UNITS[m.group(3).lower()])
    return _NUMBER_WORDS.sub(sub, text)


# ------------------------------------------------------------------ the output
@dataclass(frozen=True)
class Claim:
    """One fact an utterance states.

    field     what is claimed: merchant, amount, date, age_days, disposition, action, delivery, fee,
              delivery_days, limit, stance, opening, recorded_count
    value     the value the words give (a delivery_days value is a (low, high) pair; a stance value is
              +1 for asserting the rule's fact and -1 for denying it)
    words     the exact words that state it
    sentence  the sentence it is in
    about     what it is about: a transaction id, a delivery option, a limit key or a rule id (None
              when the sentence does not say)
    mode      states | recorded | done | offered
    """
    field: str
    value: Any
    words: str
    sentence: str
    about: str | None = None
    mode: str = "states"

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view (the sentence is left out; it is on the turn)."""
        return {"field": self.field, "value": list(self.value) if isinstance(self.value, tuple) else self.value,
                "about": self.about, "mode": self.mode, "words": self.words}


@dataclass
class ReadUtterance:
    """What the reader understood from one utterance.

    text       the utterance as given
    sentences  the utterance cut into sentences (claims never cross a sentence)
    claims     every fact the words state, in sentence order
    heard      the acts the words express (used when the bot gives no labels, and to check a label was said)
    """
    text: str
    sentences: list[str]
    claims: list[Claim] = field(default_factory=list)
    heard: list[str] = field(default_factory=list)       # acts the words express, in order of first evidence

    def of(self, field_name: str) -> list[Claim]:
        """The claims about one field: ``read.of("fee")``."""
        return [c for c in self.claims if c.field == field_name]


# ------------------------------------------------------------------ the reader
@dataclass
class _Mention:
    """Where a known thing (a returned charge, a delivery option, a disposition, ...) is named in a sentence."""
    kind: str           # item | option | money | ...
    value: Any
    start: int
    end: int
    words: str


class UtteranceReader:
    """Reads utterances with the wordings the speech contract (the policy graph) declares."""

    def __init__(self, contract: SpeechContract):
        self.c = contract
        # value phrasings, longest first, so "not recognized" wins over "recognized"
        self._value_phrases: dict[str, list[tuple[str, str]]] = {}
        for vs, vals in contract.value_sets.items():
            pairs = [(p.lower(), v) for v in vals for p in contract.phrases.get(v, [])]
            self._value_phrases[vs] = sorted(set(pairs), key=lambda x: -len(x[0]))
        self._limit_res = {k: [re.compile(r"\b" + re.escape(p.lower()).replace(r"\{n\}", r"(\d+)") + r"\b")
                               for p in ps] for k, ps in contract.limit_phrases.items()}
        self._opening = _squash(contract.opening)

    # ---------------------------------------------------------------- public
    def read(self, text: str, facts: SessionFacts, about_item: str | None = None,
             question: Any = None) -> ReadUtterance:
        """Read ``text`` into claims.

        about_item  the transaction a sentence is about when it names no merchant (the act's transaction)
        question    the customer's policy question when the text answers one (an ASK_KB frame): its
                    figures are then policy figures about what was asked (a fee of the option asked
                    about), and "N days ago" restates the question, not an account fact
        """
        out = ReadUtterance(text=text or "", sentences=_sentences(text or ""))
        if self._opening and self._opening in _squash(text or ""):
            out.claims.append(Claim("opening", True, self.c.opening, text))
        year = int(facts.session_date[:4]) if facts.session_date else date.today().year
        for sent in out.sentences:
            out.claims += self._read_sentence(sent, facts, about_item, year, question)
        out.heard = self._heard(out)
        return out

    # ---------------------------------------------------------------- one sentence
    def _read_sentence(self, sent: str, facts: SessionFacts, about_item: str | None, year: int,
                       question: Any = None) -> list[Claim]:
        """Every claim in one sentence. Facts attach to the nearest thing named in the sentence (a merchant
        for account facts, a delivery option for policy figures), else to ``about_item`` / the question."""
        low = words_to_digits(sent).lower()
        asks = bool(QUESTION_OR_CONDITION.search(sent.strip()))
        claims: list[Claim] = []
        items = self._items_named(low, facts)                      # merchants of returned transactions
        options = self._values_named(low, "delivery")              # delivery options named
        asked_option = question.get("entity") if question is not None else None
        account_talk = question is None or bool(items)             # an answer speaks of the account only by name
        near_item = lambda pos: _nearest(items, pos) or about_item  # noqa: E731
        near_option = lambda pos: _nearest(options, pos) or asked_option  # noqa: E731

        for m in items:
            claims.append(Claim("merchant", facts.items[m.value]["merchant"], m.words, sent, m.value))

        # a dollar figure is a fee when the sentence is about a delivery option, or speaks of a fee and
        # names no charge, or answers a policy question; otherwise it is the amount of the charge it is next to
        about_fee = bool(options) or (bool(FEE_WORDS.search(low)) and not items) or not account_talk
        for m in MONEY.finditer(low):
            v = m.group(1) or m.group(2)
            if about_fee:
                claims.append(Claim("fee", _num(v), m.group(0), sent, near_option(m.start())))
            else:
                claims.append(Claim("amount", v, m.group(0), sent, near_item(m.start())))
        if options and FREE.search(low):
            fm = FREE.search(low)
            claims.append(Claim("fee", 0, fm.group(0), sent, near_option(fm.start())))

        if account_talk:
            for m, iso in self._dates(low, year):
                claims.append(Claim("date", iso, m.group(0), sent, near_item(m.start())))
            for m in AGE.finditer(low):
                claims.append(Claim("age_days", int(m.group(1)), m.group(0), sent, near_item(m.start())))
        for m in BUSINESS_DAYS.finditer(low):
            lo = int(m.group(1))
            hi = int(m.group(2) or m.group(1))
            claims.append(Claim("delivery_days", (lo, hi), m.group(0), sent, near_option(m.start())))
        for key, regs in self._limit_res.items():
            for rx in regs:
                for m in rx.finditer(low):
                    claims.append(Claim("limit", int(m.group(1)), m.group(0), sent, key))
        m = RECORDED_COUNT.search(low)
        if m:
            claims.append(Claim("recorded_count", int(m.group(1)), m.group(0), sent))

        # dispositions and actions: what the words assert, never what a question asks
        if not asks:
            mode = "recorded" if RECORDING_VERBS.search(low) else "states"
            for m in self._values_named(low, "disposition"):
                claims.append(Claim("disposition", m.value, m.words, sent, near_item(m.start), mode))
        act_mode = ("offered" if asks or OFFER.search(low) else "done" if COMPLETION.search(low) else "states")
        for m in self._values_named(low, "card_action"):
            claims.append(Claim("action", m.value, m.words, sent, None, act_mode))
        for m in options:
            claims.append(Claim("delivery", m.value, m.words, sent, None, act_mode))

        for rid, sp in self.c.stances.items():
            hit = next((p for p in sp.denied if p.lower() in low), None)
            if hit:
                claims.append(Claim("stance", -1, hit, sent, rid))
                continue
            hit = next((p for p in sp.asserted if p.lower() in low), None)
            if hit and not asks:
                claims.append(Claim("stance", +1, hit, sent, rid))
        return _dedupe(claims)

    # ---------------------------------------------------------------- mentions
    @staticmethod
    def _items_named(low: str, facts: SessionFacts) -> list[_Mention]:
        """Mentions of the returned transactions' merchants, in sentence order."""
        out = []
        for tid, it in facts.items.items():
            name = str(it.get("merchant") or "").lower()
            if not name:
                continue
            for m in re.finditer(r"\b" + re.escape(name) + r"\b", low):
                out.append(_Mention("item", tid, m.start(), m.end(), m.group(0)))
        return sorted(out, key=lambda x: x.start)

    def _values_named(self, low: str, value_set: str) -> list[_Mention]:
        """Non-overlapping mentions of a value set's values, longest wording first."""
        taken: list[tuple[int, int]] = []
        out = []
        for phrase, value in self._value_phrases.get(value_set, []):
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                out.append(_Mention(value_set, value, m.start(), m.end(), m.group(0)))
        return sorted(out, key=lambda x: x.start)

    @staticmethod
    def _dates(low: str, year: int):
        """(match, "YYYY-MM-DD") for every calendar date in the sentence; a date with no year takes ``year``."""
        for m in DATE_MD.finditer(low):
            iso = _iso(int(m.group(3) or year), MONTHS[m.group(1)[:3]], int(m.group(2)))
            if iso:
                yield m, iso
        for m in DATE_DM.finditer(low):
            iso = _iso(int(m.group(3) or year), MONTHS[m.group(2)[:3]], int(m.group(1)))
            if iso:
                yield m, iso
        for m in DATE_ISO.finditer(low):
            iso = _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if iso:
                yield m, iso

    # ---------------------------------------------------------------- acts the words express
    def _heard(self, r: ReadUtterance) -> list[str]:
        """The acts the words express. An act is heard when
          * one of its cue words is said ("anything else" -> ASK_MORE_HELP), or
          * every fact it requires from the tools or the policy is stated (a merchant, an amount and a
            date -> PRESENT_TXN; a fee -> DISCLOSE_FEE; the opening statement -> OPENING), or
          * the way something is put says it: an action said to be done -> ACTION_DONE, offered in a
            question -> CONFIRM_ACTION, an answer said to be recorded -> CONFIRM_RECORDED."""
        low = words_to_digits(r.text).lower()
        said = {c.field for c in r.claims}
        modes = {f"{c.field}:{c.mode}" for c in r.claims}
        heard = []
        for act, spec in self.c.acts.items():
            facts = [f.field for f in spec.says if f.required and f.source.split(".")[0] in ("item", "policy", "text")]
            if any(q.lower() in low for q in spec.cues) or (facts and all(f in said for f in facts)):
                heard.append(act)
        for mode, act in (("action:done", "ACTION_DONE"), ("action:offered", "CONFIRM_ACTION"),
                          ("disposition:recorded", "CONFIRM_RECORDED")):
            if mode in modes and act in self.c.acts and act not in heard:
                heard.append(act)
        return heard


# ------------------------------------------------------------------ helpers
def _sentences(text: str) -> list[str]:
    """Split on . ? ! followed by a capital, a digit, a $ or a quote ("$45.67" and "U.S." stay whole)."""
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z$\d\"'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _squash(text: str) -> str:
    """Lower case, punctuation as single spaces: for word-for-word comparisons that ignore punctuation."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _num(v: str) -> float | int:
    """ "15" -> 15, "6.25" -> 6.25."""
    f = float(v)
    return int(f) if f.is_integer() else f


def _iso(y: int, mo: int, d: int) -> str | None:
    """An ISO date, or None for an impossible one ("February 30")."""
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def _nearest(mentions: list[_Mention], pos: int) -> str | None:
    """The mention closest to a position in the sentence (a tie goes to the one before it)."""
    if not mentions:
        return None
    best = min(mentions, key=lambda m: (min(abs(pos - m.end), abs(pos - m.start)), m.start > pos))
    return best.value


def _dedupe(claims: list[Claim]) -> list[Claim]:
    """Drop exact repeats (the same words read twice as the same claim), keeping the first."""
    seen, out = set(), []
    for c in claims:
        k = (c.field, str(c.value), c.about, c.mode, c.words)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out
