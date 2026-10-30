"""Session facts: what is true in one conversation so far.

The policy store holds what is true for every customer (fees, limits, rules). What is true for
*this* customer - which charges were returned, what was recorded, which card actions were offered,
what the customer said - lives only in the conversation. ``SessionFacts`` collects it from the
mock backend's tool log and the customer's turns (as the runner's reader understood them), so a
sentence like "the $45.67 Amazon charge, recorded as not recognized" can be checked.

Input   the tool log (``[{tool, args, result, ok}]`` in call order), the customer's semantic
        frames in turn order, the session date and the success status of each tool (from the
        policy graph).
Output  ``SessionFacts`` with:
          items       transaction_id -> the fields the tool returned (merchant, amount, date, ...)
          writes      tool -> every write that truly succeeded (status = success value AND every
                      argument the tool echoes back equals what was sent: rules R12/R13/R17/R33)
          options     the card actions get_card_action_options offered (None if never asked)
          caller      slot -> the latest value the customer stated (disposition, card_action, ...)
        and ``resolve(binding, about)`` -> ``Truth``: the true value of one binding (see
        knowledge/policy_kg.py BINDING_KINDS), with where it was read.

    facts = SessionFacts.from_run(tr.tool_log, tr.user_frames, "2026-03-16", contract.success_values)
    facts.resolve("item.amount", about="T1")                         # Truth(value='45.67', source='tool result: T1.amount')
    facts.resolve("write.record_charge_response.disposition", about="T1")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WRITE_KEY = {"record_charge_response": "transaction_id", "submit_card_action": "action"}   # what a write is about


@dataclass(frozen=True)
class Truth:
    """The true value for a claim and where it was read. ``known`` is False when the session has
    nothing to compare against (e.g. a 'recorded' claim before any record succeeded)."""
    value: Any
    source: str
    known: bool = True


NOTHING = Truth(None, "", known=False)


@dataclass
class SessionFacts:
    """What is true for this customer, so far in this conversation (see the module docstring)."""
    session_date: str | None = None
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    writes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    failed_writes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    options: list[str] | None = None
    caller: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_run(cls, tool_log: list[dict[str, Any]], customer_frames: list[Any], session_date: str | None = None,
                 success_values: dict[str, str] | None = None) -> "SessionFacts":
        """Collect the session's facts from the tool log and the customer's frames.

        tool_log         ``[{tool, args, result, ok}]`` in call order (the mock backend's ``env.log``)
        customer_frames  the customer's semantic frames so far, in turn order (later answers win)
        session_date     the fixture's "today" (dates are read against it), or None
        success_values   tool -> the status that means it worked (``contract.success_values``); "SUCCESS" if absent
        """
        success_values = success_values or {}
        f = cls(session_date=session_date)
        for c in tool_log:
            tool, res = c.get("tool"), c.get("result") or {}
            if not c.get("ok", True):
                f.failed_writes.setdefault(tool, []).append(c)
                continue
            if tool == "get_review_transactions":
                for t in res.get("transactions") or []:
                    f.items[str(t.get("transaction_id"))] = dict(t)
            elif tool == "get_card_action_options":
                f.options = list(res.get("options") or [])
            elif tool in WRITE_KEY:
                ok = res.get("status") == success_values.get(tool, "SUCCESS")
                echoed = all(res.get(k) == v for k, v in (c.get("args") or {}).items() if k in res and v is not None)
                (f.writes if ok and echoed else f.failed_writes).setdefault(tool, []).append(c)
        for fr in customer_frames:
            for slot, value in getattr(fr, "slots", ()):
                if value is not None:
                    f.caller[slot] = value
        return f

    # ------------------------------------------------------------------ lookups
    def item_by_merchant(self, name: str) -> str | None:
        """The transaction id whose returned merchant is ``name`` (case-insensitive), or None."""
        n = name.strip().lower()
        return next((tid for tid, it in self.items.items() if str(it.get("merchant", "")).lower() == n), None)

    def write_for(self, tool: str, about: str | None) -> dict[str, Any] | None:
        """The latest successful write of ``tool`` about ``about`` (a transaction id / an action), or the
        latest successful write when ``about`` is None."""
        key = WRITE_KEY.get(tool)
        rows = [w for w in self.writes.get(tool, []) if about is None or str((w.get("args") or {}).get(key)) == str(about)]
        return rows[-1] if rows else None

    def resolve(self, binding: str, about: str | None = None) -> Truth:
        """The true value of a session binding (item.*, write.*, writes.*, caller.*). Policy bindings are
        resolved by the verifier against the GMS, not here."""
        kind, _, rest = binding.partition(".")
        if kind == "item":
            it = self.items.get(str(about)) if about is not None else None
            if it is None or rest not in it:
                return NOTHING
            return Truth(it[rest], f"tool result get_review_transactions: {about}.{rest}")
        if kind == "write":
            tool, _, arg = rest.partition(".")
            w = self.write_for(tool, about if tool == "record_charge_response" else None)
            if w is None:
                return NOTHING
            return Truth((w.get("args") or {}).get(arg), f"successful {tool}({_args(w)})")
        if kind == "writes":
            rows = self.writes.get(rest, [])
            return Truth([dict(w.get("args") or {}) for w in rows], f"every successful {rest}", known=True)
        if kind == "caller":
            if rest in self.caller:
                return Truth(self.caller[rest], f"the customer said {rest}={self.caller[rest]}")
            return NOTHING
        return NOTHING

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view (for reports and debugging)."""
        return {"session_date": self.session_date, "items": self.items,
                "writes": {k: [_args(w) for w in v] for k, v in self.writes.items()},
                "options": self.options, "caller": self.caller}


def _args(w: dict[str, Any]) -> str:
    """A write's arguments as "transaction_id=T1, disposition=UNRECOGNIZED"."""
    return ", ".join(f"{k}={v}" for k, v in (w.get("args") or {}).items() if v is not None)
