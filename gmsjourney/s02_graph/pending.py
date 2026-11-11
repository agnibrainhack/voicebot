"""The acts a bot has decided but not yet said, each with the place it was decided among the tool calls.

The graph's flow oracle and its interpreter both queue the acts of a turn and say them together at the next question
(one ``{"role": "bot", "acts": [...]}`` event), so a trace lists a turn's tool calls first and its acts after them.
``Pending`` remembers, for every act, how many tool calls the trace held when the act was decided; the flushed event
carries those numbers as ``"at"``, and a turn can then be shown in the order it happened ("calls record, says
CONFIRM_RECORDED, calls get_card_action_options, says PRESENT_OPTIONS") by ``s09_reports.cases.segment``.

    p = Pending(lambda: n_tool_calls)        # count() is read when an act is appended
    p.append(act); p.pop(); p.clear()        # as a list; ``p.at`` follows
    p.event()                                -> {"role": "bot", "acts": [...], "at": [...]}
"""
from __future__ import annotations

from typing import Any, Callable


class Pending(list):
    """A list of acts that records, for each, the tool-call count when it was appended (see the module docstring)."""

    def __init__(self, count: Callable[[], int]):
        super().__init__()
        self.at: list[int] = []
        self._count = count

    def append(self, act: Any) -> None:            # noqa: D102
        super().append(act)
        self.at.append(self._count())

    def pop(self, i: int = -1) -> Any:              # noqa: D102
        self.at.pop(i)
        return super().pop(i)

    def clear(self) -> None:                        # noqa: D102
        super().clear()
        self.at.clear()

    def event(self) -> dict[str, Any]:
        """The bot event these acts are said in."""
        return {"role": "bot", "acts": list(self), "at": list(self.at)}


def tool_count(events: list[dict[str, Any]]) -> int:
    """How many tool calls a trace's events hold."""
    return sum(1 for e in events if e.get("role") == "tool")
