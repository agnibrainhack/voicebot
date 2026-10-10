"""Surface-noise operators shared by calibration and the realizers.

All operators are deterministic given a ``random.Random`` instance, and all can
be told which tokens are protected (slot values whose identity must survive).
"""
from __future__ import annotations

import random
import re

# Voice-channel confusions a speech recogniser commonly makes.
ASR_HOMOPHONES = {
    "four": "for", "two": "to", "eight": "ate", "one": "won",
    "card": "cart", "charge": "charged", "recognize": "recognise", "replace": "re place",
    "expedited": "expedite it", "secure": "see cure", "transaction": "trans action",
    "fee": "fee's", "know": "no", "no": "know", "yes": "yeah", "mine": "mind",
    "right": "write", "there": "their", "human": "humane", "agent": "a gent",
}

KEYBOARD_NEIGHBOURS = {
    "a": "s", "s": "a", "e": "r", "r": "e", "i": "o", "o": "i", "n": "m", "m": "n",
    "t": "y", "y": "t", "c": "v", "d": "f", "g": "h", "l": "k", "p": "o", "u": "i",
}


def _letters(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def typo(text: str, rng: random.Random, rate: float = 0.12, protect: set[str] | None = None) -> str:
    """Keyboard-style typos (swap, drop, neighbour key) inside words of 5+ letters.

    Protected words (compared letters-only, so "wasn't" protects "wasnt") and
    any token containing a digit are never touched."""
    protect = {_letters(p) for p in (protect or set())}
    words = text.split(" ")
    eligible = [i for i, w in enumerate(words)
                if len(_letters(w)) >= 5 and _letters(w) not in protect and not any(ch.isdigit() for ch in w)]

    def perturb(w: str) -> str:
        chars = list(w)
        idx = [i for i, ch in enumerate(chars) if ch.isalpha()]
        i = rng.choice(idx[1:-1]) if len(idx) > 2 else idx[0]
        op = rng.choice(["swap", "drop", "sub"])
        if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop":
            chars.pop(i)
        else:
            chars[i] = KEYBOARD_NEIGHBOURS.get(chars[i].lower(), chars[i])
        return "".join(chars)

    hit = False
    for i in eligible:
        if rng.random() < max(rate, 0.0) * 3:
            words[i] = perturb(words[i])
            hit = True
    if not hit and eligible:                 # guarantee one perturbation, on an eligible word
        i = max(eligible, key=lambda k: len(_letters(words[k])))
        words[i] = perturb(words[i])
    return " ".join(words)


def asr_noise(text: str, rng: random.Random, rate: float = 0.35, protect: set[str] | None = None) -> str:
    """Speech-recognition confusions: homophones and lowercasing.

    Sentence punctuation is kept, as modern recognisers restore it."""
    protect = {p.lower() for p in (protect or set())}
    toks = re.findall(r"[A-Za-z']+|\d+(?:\.\d+)?|[^\sA-Za-z\d]", text)
    out: list[str] = []
    hit = False
    for t in toks:
        low = t.lower()
        if low in ASR_HOMOPHONES and low not in protect and rng.random() < rate:
            out.append(ASR_HOMOPHONES[low])
            hit = True
        else:
            out.append(low)
    if not hit:                              # force one confusion when any is available
        for i, t in enumerate(out):
            if t in ASR_HOMOPHONES and t not in protect:
                out[i] = ASR_HOMOPHONES[t]
                break
    s = " ".join(out)
    s = re.sub(r"\s+([.,?!;:])", r"\1", s)
    return re.sub(r"\$\s+", "$", re.sub(r"\s+", " ", s)).strip()


FILLERS = ["um", "uh", "like", "you know", "hmm", "so", "okay so"]


def disfluency(text: str, rng: random.Random, kind: str = "fillers") -> str:
    """Voice disfluencies: fillers between words, or a restarted phrase."""
    words = text.split()
    if not words:
        return text
    if kind == "restart":
        k = min(3, len(words))
        return " ".join(words[:k]) + " — sorry, " + " ".join(words)
    out = []
    for i, w in enumerate(words):
        if i in (0, len(words) // 2) or rng.random() < 0.12:
            out.append(rng.choice(FILLERS) + ",")
        out.append(w)
    return " ".join(out)


DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
                13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
                50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def number_to_words(n: int) -> str:
    if n in NUMBER_WORDS:
        return NUMBER_WORDS[n]
    if n < 100:
        return f"{NUMBER_WORDS[n // 10 * 10]} {NUMBER_WORDS[n % 10]}"
    if n < 1000:
        rest = n % 100
        return f"{NUMBER_WORDS[n // 100]} hundred" + (f" {number_to_words(rest)}" if rest else "")
    return str(n)


def spell_digits(code: str) -> str:
    return " ".join(DIGIT_WORDS[int(c)] for c in code if c.isdigit())
