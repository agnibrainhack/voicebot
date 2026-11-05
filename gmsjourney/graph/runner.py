"""Driving any bot through a journey mined from a compact policy graph (book Ch. 9.3).

The same per-turn loop as sut/adapter.py: the simulator answers the bot's act with the next
meaning card of the script, the card is realised under the scenario's factor levels (pinned seed
from the proof), read back by the frame reader (runtime invariance monitor) and sent to the bot;
tool calls are observed at the mock backend. What the bot *says* is read from its acts: success
claims (states that ``claims`` a tool), outcome acts (states marked ``outcome``) and policy
answers, so the trace is scored exactly as the oracle's.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..core import Scenario, SUTResult
from ..kb_truth import judge_kb_answer
from ..ontology import Ontology
from ..sut.adapter import JourneyRunner, MAX_USER_TURNS, _turn_stats
from ..sut.tools import FaultProfile, ToolEnvironment, ToolGateway
from .journeys import GraphSimulator
from .trace import GraphTrace


def _graph():
    from ..domain import current
    return current().graph


def item_index(fixture: dict[str, Any]) -> dict[str, dict]:
    """item id -> item, for every list of items a scripted tool returns (context for realisation)."""
    g = _graph()
    out = {}
    for t, sp in g.tools.items():
        for fld, rs in (sp.get("returns") or {}).items():
            if not isinstance(rs, dict) or "items" not in rs:
                continue
            for res in fixture.get(f"tool:{t}") or []:
                for it in (res.get(fld) or []) if isinstance(res, dict) else []:
                    out[str(it.get(rs["id"]))] = it
    return out


@dataclass
class GraphRunner(JourneyRunner):
    def __post_init__(self) -> None:
        self.ont = Ontology.from_store(self.store)

    def run(self, scn: Scenario, faults: list[FaultProfile] | None = None,
            oracle_path: list | None = None) -> SUTResult:
        g = _graph()
        item = scn.base
        spec, fx = item.metadata["spec"], item.metadata["fixture"]
        env = ToolEnvironment(json.loads(json.dumps(fx)))
        tools = ToolGateway(env, list(faults)) if faults else env
        monitor = None
        if self.governance is not None:
            from ..governance.monitor import GovernedTools
            from .governance import GraphRuntimeMonitor
            monitor = GraphRuntimeMonitor(self.store, self.reader, self.governance, run_id=scn.sid)
            tools = GovernedTools(tools, monitor)
        bot = self.bot_factory(tools)
        sim = GraphSimulator(spec, fx)
        items = item_index(fx)
        overrides = scn.metadata.get("seed_overrides", {})
        realizer = (self.arm_realizers or {}).get(scn.metadata.get("arm", "template"), self.realizer)
        bot_stats: list[dict[str, Any]] = []
        tr = GraphTrace()
        transcript: list[dict[str, Any]] = []
        violations: list[dict[str, Any]] = []
        current = None
        last_kb = None
        t0 = time.perf_counter()
        turn = bot.start()
        bot_stats.append(_turn_stats(turn.text, t0))
        for i in range(MAX_USER_TURNS + 1):
            acts = turn.acts if (self.use_structured_acts and turn.acts) else [{"act": "UNKNOWN"}]
            tr.events.append({"role": "bot", "acts": acts, "text": turn.text})
            transcript.append({"role": "bot", "text": turn.text, "acts": [a.get("act") for a in acts]})
            if monitor is not None:
                monitor.on_bot(turn.text, turn.acts if self.use_structured_acts else None)
            self._observe(tr, acts, last_kb, turn.text)
            for a in acts:
                for v in a.values():
                    if isinstance(v, str) and v in items:
                        current = items[v]
            if turn.ended or i == MAX_USER_TURNS:
                break
            act = dict(acts[-1])
            frame = sim.respond(act)
            if frame is None:
                break
            st = sim.last_state
            ctx = {**(current or {}), "state": st}
            pinned = oracle_path is not None and i < len(oracle_path) and oracle_path[i][0] == frame
            att = overrides.get(i, 0) if pinned else 0
            seed = f"{scn.sid}|{i}" + (f"#{att}" if att else "")
            text = realizer.realize(frame, scn.factor_levels, seed, ctx)
            read = self.reader.read(text, st)
            if not self.ont.equivalent(frame, read, st):
                violations.append({"turn": i, "state": st, "frame": str(frame), "text": text, "read": str(read)})
            tr.turns += 1
            tr.user_frames.append(frame)
            tr.events.append({"role": "user", "frame": frame, "text": text, "state": st})
            transcript.append({"role": "user", "text": text, "frame": str(frame), "state": st})
            last_kb = frame if frame.act == (g.glob.get("question") or {}).get("act") else None
            if monitor is not None:
                monitor.on_user(text, st)
            n0 = len(env.log)
            t0 = time.perf_counter()
            turn = bot.respond(text)
            bot_stats.append(_turn_stats(turn.text, t0))
            for c in env.log[n0:]:
                tr.events.append({"role": "tool", "tool": c["tool"], "args": c["args"], "result": c.get("result"),
                                  "ok": c.get("ok", True)})
        tr.tool_log = list(env.log)
        tr.ended = bool(turn.ended) and not tr.transferred
        comps = tr.components()
        status = "escalated" if tr.transferred else ("ok" if tr.ended else "failed")
        return SUTResult(
            answer=tr.signature(), components=comps,
            trajectory=[{"tool": c["tool"], "ok": c.get("ok", True)} for c in tr.tool_log],
            status=status, escalation_trigger=tr.transfer_reason,
            extras={"trace": tr, "transcript": transcript, "invariance_violations": violations,
                    "bot_turn_stats": bot_stats, "arm": scn.metadata.get("arm", "template"),
                    "injected": list(getattr(tools, "injected", [])),
                    "governance": monitor.finish() if monitor is not None else None})

    def _observe(self, tr: GraphTrace, acts: list[dict], last_kb, text: str) -> None:
        g = _graph()
        q = g.glob.get("question") or {}
        for a in acts:
            k = a.get("act")
            s = g.states.get(k) or {}
            if s.get("outcome"):
                tr.told.append(dict(a))
            if s.get("claims"):
                tr.claims.append((s["claims"], dict(a)))
            if k == q.get("answer") and last_kb is not None:
                ans = a.get("text") or text
                verdict = judge_kb_answer(self.store, last_kb, ans)
                tr.kb.append({"frame": str(last_kb), "answer": ans, **verdict})
