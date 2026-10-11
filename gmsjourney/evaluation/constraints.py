"""Per-turn constraint budgets (gmsh ConstraintOracle style, applied to a voice bot).

A voice bot can be right and still unusable: an answer that arrives after three
seconds or runs to eighty words is a failed turn on a phone line. These budgets are
checked on every bot turn of every run and reported next to correctness - never folded
into ``correct`` (so pass rates stay comparable with earlier campaigns) - and feed the
severity model as LOW-severity failure codes.

* ``max_bot_latency_ms``     worst single bot turn (wall clock around start()/respond())
* ``p95_bot_latency_ms``     95th percentile over the run's bot turns
* ``max_bot_words``          longest bot utterance, in words
* ``max_tool_calls``         absolute tool-call budget per run (None = off)
* ``tool_call_slack``        tool calls allowed above the oracle's count for the same case
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ConstraintBudget:
    max_bot_latency_ms: float | None = 2500.0
    p95_bot_latency_ms: float | None = 1500.0
    max_bot_words: int | None = 60
    max_tool_calls: int | None = None
    tool_call_slack: int | None = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return float(s[k])


def check_constraints(bot_turns: list[dict[str, Any]], tool_calls: int, oracle_tool_calls: int | None,
                      budget: ConstraintBudget | None) -> dict[str, Any]:
    lat = [float(t.get("latency_ms", 0.0)) for t in bot_turns]
    words = [int(t.get("words", 0)) for t in bot_turns]
    m = {"bot_latency_ms_max": round(max(lat), 2) if lat else 0.0, "bot_latency_ms_p95": round(_p95(lat), 2),
         "bot_words_max": max(words) if words else 0, "bot_turns": len(bot_turns)}
    v: list[str] = []
    if budget is not None:
        b = budget
        if b.max_bot_latency_ms is not None and m["bot_latency_ms_max"] > b.max_bot_latency_ms:
            v.append(f"latency_max:{m['bot_latency_ms_max']:.0f}ms>{b.max_bot_latency_ms:.0f}ms")
        if b.p95_bot_latency_ms is not None and m["bot_latency_ms_p95"] > b.p95_bot_latency_ms:
            v.append(f"latency_p95:{m['bot_latency_ms_p95']:.0f}ms>{b.p95_bot_latency_ms:.0f}ms")
        if b.max_bot_words is not None and m["bot_words_max"] > b.max_bot_words:
            v.append(f"words:{m['bot_words_max']}>{b.max_bot_words}")
        if b.max_tool_calls is not None and tool_calls > b.max_tool_calls:
            v.append(f"tool_calls:{tool_calls}>{b.max_tool_calls}")
        if b.tool_call_slack is not None and oracle_tool_calls is not None and \
                tool_calls > oracle_tool_calls + b.tool_call_slack:
            v.append(f"tool_calls_over_oracle:{tool_calls}>{oracle_tool_calls}+{b.tool_call_slack}")
    return {**m, "within_budget": int(not v), "budget_violations": v}
