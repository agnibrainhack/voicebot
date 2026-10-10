"""Core records shared across the pipeline (names follow the book's API, App. A)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QAItem:
    """One base case: content + ground truth (+ per-component truth)."""
    qid: str
    query: str                       # canonical (unenriched) rendering, for reading
    answer: Any                      # outcome signature computed by the oracle
    answer_type: str = "decision"
    base: str | None = None          # base category (Table 5.1)
    components: dict[str, Any] = field(default_factory=dict)   # expected_<stage>
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """A base case under one assignment of presentation factor levels."""
    sid: str
    base: QAItem
    factor_levels: dict[str, str]
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SUTResult:
    answer: Any = None
    components: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"               # ok | escalated | failed
    escalation_trigger: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
