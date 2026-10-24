"""Numeric range attestor for figures the bot speaks (gmsh harness/governance/policy_numeric.py).

gmsh's attestor gives every number a three-way verdict: the rule fails -> BLOCK; the
rule passes and the value sits inside the calibrated range -> PASS; the rule passes but
the value is outside -> REVIEW. Here the "rule" and the "range" come from the exact
numeric memory (ENM) of the store, per family of figure a caller can hear:

  auth_attempts         "you have N attempts / tries"            register: max_auth_attempts
  transactions          "N transactions / charges"                register: max_transactions_per_session
  review_window         "N days" about how far back / the window  register: lookback_days
  delivery_time         "N(-M) business days" about delivery      register: delivery_days_min/max of every option
  fee                   "$N" / "N dollars" not an account amount  register: delivery fees of every option

Verdicts:
  PASS    the figure is the register value for its family;
  REVIEW  the figure is a register value of a *related* family (another limit, another
          option's days) - a plausible-sounding but possibly misattributed figure: a
          human should look (ESCALATE);
  FAIL    the figure is in no register at all - an invented number (DENY).

The speech monitor already denies wrong fees and delivery times where it can tell which
option a sentence is about; the attestor covers the figures it does not check (limits
phrased freely, days without a delivery keyword) and never re-flags a figure another
speech check already flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_DAYS = re.compile(r"\b(\d+)\s*(?:(?:to|-|–)\s*(\d+)\s*)?(?:business\s+|calendar\s+)?days?\b(?!\s+(?:ago|old|back))", re.I)
_MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s*dollars\b", re.I)


@dataclass(frozen=True)
class Attestation:
    family: str
    value: float
    verdict: str           # PASS | REVIEW | FAIL
    expected: tuple

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "value": self.value, "verdict": self.verdict, "expected": list(self.expected)}


class NumericRangeAttestor:
    """The families, their spoken patterns, registers and related figures come from
    domain.yaml -> rules.attestor; every register value is read from exact numeric memory."""

    def __init__(self, store, cfg: dict | None = None):
        if cfg is None:
            from ..domain import current
            cfg = current().get("rules.attestor")
        self.families = cfg["families"]

        def values(ref: str) -> set[float]:
            cat, key = ref.split("/", 1)
            if key == "*":                     # every key of the category, read from the register
                return {float(store.lookup_enm(cat, k)) for (c, k) in store.enm if c == cat}
            return {float(store.lookup_enm(cat, key))}

        self.register = {f: set().union(*(values(r) for r in sp["register"])) for f, sp in self.families.items()}
        limits = {float(store.lookup_enm("limit", k)) for k in cfg["limits"]}
        self.related = {f: set().union(*(limits if r == "limits" else self.register[r] for r in sp["related"]))
                        for f, sp in self.families.items()}
        self._pat = {f: re.compile(sp["pattern"], re.I) for f, sp in self.families.items() if sp.get("pattern")}

    def verdict(self, family: str, value: float) -> Attestation:
        exp = tuple(sorted(self.register[family]))
        if value in self.register[family]:
            return Attestation(family, value, "PASS", exp)
        if value in self.related[family]:
            return Attestation(family, value, "REVIEW", exp)
        return Attestation(family, value, "FAIL", exp)

    def attest(self, text: str, account_amounts: set[float] | None = None) -> list[Attestation]:
        out: list[Attestation] = []
        account_amounts = account_amounts or set()
        fams = self.families
        day_fams = [f for f, sp in fams.items() if sp.get("days")]
        money_fams = [f for f, sp in fams.items() if sp.get("money")]
        for sent in re.split(r"(?<=[.?!])\s+", text or ""):
            low = sent.lower()
            if not re.search(r"\d", low):
                continue
            for f, pat in self._pat.items():          # families with their own pattern
                if fams[f].get("only_if") and not re.search(fams[f]["only_if"], low):
                    continue                          # e.g. a limit statement, not a count of what was found
                for m in pat.finditer(low):
                    out.append(self.verdict(f, float(m.group(1))))
            for m in _DAYS.finditer(low):             # a day count: the first family whose cue the sentence has
                fam = next((f for f in day_fams if re.search(fams[f]["cue"], low)), None)
                if fam is None:
                    continue
                for g in (m.group(1), m.group(2)):
                    if g:
                        out.append(self.verdict(fam, float(g)))
            for m in _MONEY.finditer(low):
                v = float(m.group(1) or m.group(2))
                if v in account_amounts:
                    continue
                for f in money_fams:
                    if re.search(fams[f]["cue"], low):
                        out.append(self.verdict(f, v))
        return out
