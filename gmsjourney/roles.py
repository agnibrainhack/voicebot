"""Tools and values that core code refers to by role, resolved in the active pack's graph on every access.

A tool's role is stated in the graph (``tool has_role auth | read | write | handoff``); charge review's graph
also orders its servicing tools (``precedes``). From these:

    GATE          the auth tool (role auth; else the tool account servicing ``requires``)
    TRANSFER      the handoff tool                                 REASON  its argument
    ACCOUNT_TOOLS / WRITE_TOOLS   read + write tools / write tools
    ITEMS         the read tool that returns items with an id      KEY     that id field
    RECORD        the write that takes the item id                 ANSWER  its other argument
    ACTION        another write, not tied to an item               ACT / OPTION  its arguments
    OPTIONS       the read just before ACTION in ``precedes`` order
    DELIVERY      the values of the delivery value set (when the pack has one)
    FEE_OPTION / FREE_OPTION  a value that needs a fee confirmation / one that does not
    OK            the success status shared by the writes

A role the pack does not have reads as None (supplementary card has no ITEMS, RECORD or OPTIONS). Use
``roles.RECORD`` (attribute access), not ``from roles import RECORD``: the active pack can change.
"""
from __future__ import annotations

from .names import N, graph_names


def _arg(tool, i):
    D = graph_names()
    args = D.tools[tool].args if tool in D.tools else ()
    return args[i] if tool and len(args) > i else None


def _first(xs):
    return xs[0] if xs else None


_ROLES = {
    "D": lambda D: D,
    "GATE": lambda D: D.gate_tool,
    "ITEMS": lambda D: D.item_tool,
    "RECORD": lambda D: D.record_tool,
    "OPTIONS": lambda D: D.options_tool,
    "ACTION": lambda D: D.action_tool,
    "TRANSFER": lambda D: D.transfer_tool,
    "KEY": lambda D: D.item_key,
    "ANSWER": lambda D: _arg(D.record_tool, 1),
    "ACT": lambda D: _arg(D.action_tool, 0),
    "OPTION": lambda D: _arg(D.action_tool, 1),
    "REASON": lambda D: _arg(D.transfer_tool, 0),
    "ACCOUNT_TOOLS": lambda D: D.account_tools,
    "WRITE_TOOLS": lambda D: D.write_tools,
    "DELIVERY": lambda D: D.value_sets.get("delivery", ()),
    "FEE_OPTION": lambda D: _first(D.fee_values),
    "FREE_OPTION": lambda D: next((o for o in D.fee_free_values), None),
    "OK": lambda D: next((D.tools[t].success for t in sorted(D.write_tools)), None),
}


def __getattr__(name: str):
    if name in _ROLES:
        return _ROLES[name](graph_names())
    raise AttributeError(name)


__all__ = ["N", *_ROLES]
