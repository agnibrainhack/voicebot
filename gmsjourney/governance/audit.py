"""Tamper-evident audit trail: a SHA-256 hash chain (book Ch. 12; forgeloop.audit re-implemented).

Every gate decision is appended as an event whose hash covers the previous hash,
so editing, dropping or reordering any past event breaks ``verify()``. Events use a
logical sequence number instead of wall-clock time, so a replayed run produces the
same chain (deterministic, diffable audit logs).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

GENESIS = "0" * 64


@dataclass(frozen=True)
class AuditEvent:
    run_id: str
    seq: int
    kind: str                       # tool | speech | input | plan
    state_hash: str
    proposed: dict[str, Any]
    decisions: list[dict[str, Any]]
    outcome: str                    # allowed | denied | escalated | executed | blocked | flagged


@dataclass(frozen=True)
class SealedEvent:
    event: AuditEvent
    prev_hash: str
    event_hash: str


def _digest(event: AuditEvent, prev_hash: str) -> str:
    d = asdict(event)
    d["prev_hash"] = prev_hash
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def verify_chain(sealed: list[SealedEvent]) -> bool:
    prev = GENESIS
    for s in sealed:
        if s.prev_hash != prev or _digest(s.event, prev) != s.event_hash:
            return False
        prev = s.event_hash
    return True


class AuditLogger:
    def __init__(self) -> None:
        self._events: list[SealedEvent] = []
        self._head = GENESIS

    def log(self, event: AuditEvent) -> SealedEvent:
        h = _digest(event, self._head)
        s = SealedEvent(event, self._head, h)
        self._events.append(s)
        self._head = h
        return s

    @property
    def events(self) -> list[SealedEvent]:
        return list(self._events)

    def head(self) -> str:
        return self._head

    def verify(self) -> bool:
        return verify_chain(self._events)


def state_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
