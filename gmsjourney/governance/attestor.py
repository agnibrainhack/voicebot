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

_ATTEMPTS = re.compile(r"\b(\d+)\s+(?:verification\s+|authentication\s+)?(?:attempts?|tries)\b", re.I)
_TXNS = re.compile(r"\b(\d+)\s+(?:transactions?|charges?)\b", re.I)
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
    def __init__(self, store):
        L = lambda k: float(store.lookup_enm("limit", k))
        opts = ("standard_delivery", "expedited_delivery")
        days = set()
        for o in opts:
            days |= {float(store.lookup_enm("delivery_days_min", o)), float(store.lookup_enm("delivery_days_max", o))}
        limits = {L(k) for k in ("max_transactions_per_session", "lookback_days", "max_auth_attempts")}
        self.register = {
            "auth_attempts": {L("max_auth_attempts")},
            "transactions": {L("max_transactions_per_session")},
            "review_window": {L("lookback_days")},
            "delivery_time": days,
            "fee": {float(store.lookup_enm("delivery_fee_usd", o)) for o in opts},
        }
        self.related = {"auth_attempts": limits, "transactions": limits, "review_window": limits | days,
                        "delivery_time": days | limits, "fee": self.register["fee"]}

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
        for sent in re.split(r"(?<=[.?!])\s+", text or ""):
            low = sent.lower()
            if not re.search(r"\d", low):
                continue
            for m in _ATTEMPTS.finditer(low):
                out.append(self.verdict("auth_attempts", float(m.group(1))))
            if re.search(r"up to|at most|maximum|\bmax\b|\bonly\b|can review|per (call|session)|in one", low):
                for m in _TXNS.finditer(low):            # a limit statement, not a count of what was found
                    out.append(self.verdict("transactions", float(m.group(1))))
            for m in _DAYS.finditer(low):
                fam = ("delivery_time" if re.search(r"deliver|arriv|shipping|business", low) else
                       "review_window" if re.search(r"previous|window|back|last|review|within", low) else None)
                if fam is None:
                    continue
                for g in (m.group(1), m.group(2)):
                    if g:
                        out.append(self.verdict(fam, float(g)))
            for m in _MONEY.finditer(low):
                v = float(m.group(1) or m.group(2))
                if v in account_amounts:
                    continue
                if re.search(r"fee|cost|charge for|pay|delivery|shipping", low):
                    out.append(self.verdict("fee", v))
        return out
