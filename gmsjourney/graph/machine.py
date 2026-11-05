"""The oracle for a compact policy graph: an interpreter over its state machine.

``GraphOracle.run(spec)`` plays the journey a base case prescribes against the same mock
backend and simulated customer the bot will face, following the graph exactly:

  say state    emit the state's act (fields resolved from captured slots, the current loop
               item, tool results, attributes and exact limits) and follow ``go``
  ask state    emit the act and take the customer's meaning card; the global behaviours apply
               in every ask state (a request for a person -> transfer, a policy question ->
               answer then re-ask, an unclear answer -> one clarification then transfer, the
               turn limit -> transfer, or the end at the wrap-up); otherwise the act's slots are
               captured and its target followed
  call state   call the tool with the resolved arguments and follow the branch for its status
               (UNAVAILABLE when the call raises, ECHO_MISMATCH when a success echoes another
               value than the one requested)
  each state   the next in-scope item of a returned list, up to the cap; ``@next`` comes back
  transfer     say what was truly recorded (if anything), TRANSFER with the reason, call the
               handoff tool, stop
  end          emit the end act, stop

The walk is split in ``advance`` (from a state to the next prompt, emitting the bot's acts) and
``react`` (the customer's answer in that prompt), so the same interpreter also drives the
reference bot one turn at a time (graph/bot.py). The trace (graph/trace.py) is the ground truth;
like the v1 oracle the interpreter consumes meaning cards only, never text.
"""
from __future__ import annotations

from typing import Any

from ..ontology import SemanticFrame
from ..sut.tools import ToolEnvironment, ToolError
from .model import ECHO_MISMATCH, NEXT, UNAVAILABLE, PolicyGraph
from .trace import GraphTrace


class _Stop(Exception):
    pass


class NeedChoice(Exception):
    """Raised in explore mode when the run needs a decision the script does not make yet."""

    def __init__(self, kind: str, key: str, index: int):
        self.kind, self.key, self.index = kind, key, index
        self.slots: dict[str, Any] = {}


def _graph() -> PolicyGraph:
    from ..domain import current
    return current().graph


class GraphOracle:
    def __init__(self, store):
        from ..oracle import Limits
        self.store = store
        self.g = _graph()
        self.limits = Limits.from_store(store)

    def L(self, name: str) -> Any:
        return getattr(self.limits, name)

    # ------------------------------------------------------------------ run
    def run(self, spec, fixture: dict[str, Any] | None = None, frames_override: list[SemanticFrame] | None = None,
            gateway_factory=None, explore: bool = False) -> GraphTrace:
        from .journeys import GraphSimulator, build_fixture
        fixture = fixture if fixture is not None else build_fixture(spec)
        env = ToolEnvironment(fixture, explore=explore)
        tools = gateway_factory(env) if gateway_factory else env
        self.start_session(tools, env)
        self._sim = GraphSimulator(spec, fixture, explore=explore)
        self._replay = list(frames_override) if frames_override is not None else None
        state = self.g.start
        try:
            while True:
                st = self.advance(state)
                f = self._ask(st)
                nxt = self.react(st, f)
                state = st if nxt is None else nxt
        except _Stop:
            pass
        self._tr.tool_log = list(env.log)
        return self._tr

    def start_session(self, tools, env) -> None:
        self._tr = GraphTrace()
        self._tools, self._env = tools, env
        self._slots: dict[str, Any] = {}
        self._results: dict[str, dict] = {}
        self._pending: list[dict[str, Any]] = []
        self._loops: dict[str, dict[str, Any]] = {}
        self._counters: dict[str, int] = {}
        self._unclear: dict[str, int] = {}
        self._item: dict[str, Any] | None = None

    # ------------------------------------------------------------------ plumbing
    def _emit(self) -> None:
        if self._pending:
            self._tr.events.append({"role": "bot", "acts": list(self._pending)})
            self._pending = []

    def _ask(self, state: str) -> SemanticFrame:
        self._emit()
        tr = self._tr
        if self._replay is not None:
            frame = self._replay.pop(0) if self._replay else None
        else:
            try:
                frame = self._sim.respond(tr.events[-1]["acts"][-1] if tr.events else {"act": state})
            except NeedChoice as nc:              # explore mode: the miner branches here
                nc.slots = dict(self._slots)
                raise
        if frame is None:
            tr.notes.append("customer silent: the conversation ends")
            tr.ended = True
            raise _Stop()
        self.record_user(state, frame)
        return frame

    def record_user(self, state: str, frame: SemanticFrame) -> None:
        tr = self._tr
        tr.turns += 1
        tr.user_frames.append(frame)
        ev = {"role": "user", "frame": frame, "state": state}
        if self._item is not None:
            ev["item"] = dict(self._item)
        tr.events.append(ev)

    def _call(self, tool: str, args: dict[str, Any]) -> tuple[str, dict]:
        """(status, result) of one call: UNAVAILABLE when it raises, ECHO_MISMATCH when a success
        echoes another value than requested."""
        sp = self.g.tools[tool]
        try:
            res = self._tools.call(tool, **args)
        except NeedChoice as nc:
            nc.slots = dict(self._slots)
            raise
        except ToolError as e:
            self._tr.events.append({"role": "tool", "tool": tool, "args": args, "result": {"error": str(e)}, "ok": False})
            return UNAVAILABLE, {}
        self._tr.events.append({"role": "tool", "tool": tool, "args": args, "result": res, "ok": True})
        status = res.get("status")
        echo = sp.get("echo")
        if status == sp["ok"] and echo and res.get(echo) != args.get(echo):
            status = ECHO_MISMATCH
        self._results[tool] = res
        return status, res

    def _calls(self, tool: str) -> list[dict]:
        return [c for c in self._env.log if c["tool"] == tool]

    def _writes(self, tool: str) -> list[dict]:
        sp = self.g.tools[tool]
        out = []
        for c in self._calls(tool):
            r = c.get("result") or {}
            if c.get("ok") and r.get("status") == sp["ok"] and (not sp.get("echo") or
                                                                r.get(sp["echo"]) == c["args"].get(sp["echo"])):
                out.append(c)
        return out

    # ------------------------------------------------------------------ values
    def resolve(self, ref: Any) -> Any:
        """A field value: $opening, $<limit>, $<tool>.<field>, $<slot>, $<slot>.<attr>,
        $<item>.<field>, {"$writes": tool, "key": arg, "value": arg}; numbers from exact memory."""
        if isinstance(ref, dict) and "$writes" in ref:
            return {c["args"].get(ref["key"]): c["args"].get(ref["value"]) for c in self._writes(ref["$writes"])}
        if not isinstance(ref, str) or not ref.startswith("$"):
            return ref
        body = ref[1:]
        if body == "opening":
            return self.store.required_language.get("opening", self.g.opening)
        if body in self.g.limits:
            return self.L(body)
        head, _, tail = body.partition(".")
        if head in self.g.tools:
            return (self._results.get(head) or {}).get(tail)
        val = self._slots.get(head)
        if not tail:
            return val
        if isinstance(val, dict):
            return val.get(tail)
        return self.attribute(val, tail)

    def attribute(self, value: Any, attr: str) -> Any:
        if value is None:
            return None
        num = self.g.numeric_attributes().get(attr)
        if num is not None and value in num["members"]:
            return self.store.lookup_enm(attr, value)
        return self.g.attr(value, attr)

    def holds(self, cond: dict[str, Any] | list) -> bool:
        if isinstance(cond, list):                        # several conditions, all must hold
            return all(self.holds(c) for c in cond)
        for k, want in cond.items():
            if k.startswith("#"):
                if len(self._calls(k[1:])) not in want:
                    return False
                continue
            slot, _, attr = k.partition(".")
            v = self._slots.get(slot)
            if attr:
                v = self.attribute(v, attr)
            if isinstance(want, dict):
                if "not" in want and v in want["not"]:
                    return False
                if "in" in want and v not in (self.resolve(want["in"]) or []):
                    return False
            elif v not in want:
                return False
        return True

    def _target(self, target: Any, state: str) -> str:
        """Follow a target to a plain state name, '!reason' or '@next'."""
        if isinstance(target, str):
            return target
        if isinstance(target, list):
            for alt in target:
                if isinstance(alt, dict) and "then" in alt:
                    if self.holds(alt.get("if", {})):
                        return self._target(alt["then"], state)
                else:
                    return self._target(alt, state)
            raise ValueError(f"{state}: no alternative holds")
        if isinstance(target, dict) and "below" in target:
            n = len(self._calls(self.g.states[state]["call"]))
            return self._target(target["then"] if n < self.L(target["below"]) else target["else"], state)
        if isinstance(target, dict) and "count" in target:
            name = target["count"]
            self._counters[name] = self._counters.get(name, 0) + 1
            ok = self._counters[name] <= self.L(target["upto"])
            return self._target(target["then"] if ok else target["else"], state)
        raise ValueError(f"bad target {target!r}")

    def act_of(self, state: str) -> dict[str, Any]:
        s = self.g.states[state]
        act = {"act": state}
        for fld, ref in (s.get("say") or {}).items():
            act[fld] = self.resolve(ref)
        return act

    def _note(self, state: str, act: dict[str, Any]) -> None:
        s = self.g.states[state]
        if s.get("outcome"):
            self._tr.told.append(dict(act))
        if s.get("claims"):
            self._tr.claims.append((s["claims"], dict(act)))

    # ------------------------------------------------------------------ the walk
    def advance(self, state: str) -> str:
        """From ``state`` to the next prompt: emit the bot's acts on the way, return the ask state
        (its act is pending). Raises _Stop at the end or after a transfer."""
        g, tr = self.g, self._tr
        while True:
            if state.startswith("!"):
                self._transfer(state[1:])
            s = g.states[state]
            tr.visited.append(state)
            tr.rules_fired.update(s.get("rules", []))
            kind = g.kind(state)
            if kind == "end":
                self._pending.append({"act": state})
                tr.ended = True
                self._emit()
                raise _Stop()
            if kind == "say":
                act = self.act_of(state)
                self._pending.append(act)
                self._note(state, act)
                state = self._target(s["go"], state)
                if state == NEXT:
                    state = g.loop_of[tr.visited[-1]]
                continue
            if kind == "call":
                args = {a: self.resolve(ref) for a, ref in (s.get("args") or {}).items()}
                status, _ = self._call(s["call"], args)
                state = self._target(s["on"][status], state)
                if state == NEXT:
                    state = g.loop_of[tr.visited[-1]]
                continue
            if kind == "each":
                state = self._each(state)
                continue
            act = self.act_of(state)
            self._turn_guard(state)
            self._pending.append(act)
            self._note(state, act)
            return state

    def _each(self, state: str) -> str:
        s = self.g.states[state]
        lp = self._loops.get(state)
        if lp is None:
            items = list(self.resolve(s["each"]) or [])
            for fld, lim in (s.get("where") or {}).items():
                if "max" in lim:
                    items = [x for x in items if x.get(fld) is not None and x[fld] <= self.L(lim["max"])]
                if "min" in lim:
                    items = [x for x in items if x.get(fld) is not None and x[fld] >= self.L(lim["min"])]
            lp = self._loops[state] = {"items": items, "i": -1}
            if not items:
                return s["empty"]
        lp["i"] += 1
        cap = self.L(s["cap"]) if s.get("cap") else len(lp["items"])
        if lp["i"] >= min(cap, len(lp["items"])):
            self._item = None
            if len(lp["items"]) > cap and s.get("over_cap"):
                return s["over_cap"]
            return s["done"]
        item = lp["items"][lp["i"]]
        self._slots[s.get("as", "item")] = item
        self._item = item
        self._counters = {}
        self._unclear = {}
        return s["do"]

    def react(self, state: str, f: SemanticFrame) -> str | None:
        """The customer's answer in a prompt: the next state, '!reason', or None to re-ask."""
        g, tr = self.g, self._tr
        G = g.glob
        s = g.states[state]
        if G.get("human") and f.act == G["human"]["act"]:
            return G["human"]["then"]
        if G.get("question") and f.act == G["question"]["act"]:
            tr.rules_fired.add(G["question"]["rule"])
            self._pending.append(self._answer_kb(f))
            return None
        if f.act in s["on"]:
            update = g.updates(f.act)
            for sl in g.slot_specs(f.act):
                v = f.get(sl)
                if v is not None or not update:
                    self._slots[sl] = v
            self._unclear.pop(state, None)
            nxt = self._target(s["on"][f.act], state)
            return g.loop_of.get(state, state) if nxt == NEXT else nxt
        u = G.get("unclear")
        if not u:
            return None
        self._unclear[state] = self._unclear.get(state, 0) + 1
        tr.rules_fired.add(u["rule"])
        if self._unclear[state] > self.L(u["limit"]):
            return u["else"]
        self._pending.append({"act": u["clarify"], "state": state})
        return None

    def _turn_guard(self, state: str) -> None:
        tl = self.g.glob.get("turn_limit")
        if not tl or self._tr.turns < self.L(tl["limit"]):
            return
        self._tr.rules_fired.add(tl["rule"])
        wrap = tl.get("wrap_up")
        targets = [t for t in (self.g.states[state].get("on") or {}).values() if isinstance(t, str)]
        if wrap and wrap in targets:
            self._pending.append({"act": wrap})
            self._tr.ended = True
            self._emit()
            raise _Stop()
        self._transfer(tl["else"][1:])

    def _answer_kb(self, frame: SemanticFrame) -> dict[str, Any]:
        from ..kb_truth import kb_answer
        truth = kb_answer(self.store, frame)
        self._tr.kb.append({"frame": frame, "answer": truth.answer, "correct": True})
        return {"act": self.g.glob["question"]["answer"], "intent": frame.get("intent"), "answer": truth.answer}

    def _transfer(self, reason: str) -> None:
        g, tr = self.g, self._tr
        T = g.transfers
        tr.rules_fired.add(T["reasons"][reason])
        sm = T.get("summary")
        if sm:
            rec = self.resolve(sm["value"])
            if rec:
                tr.rules_fired.add(sm["rule"])
                self._pending.append({"act": sm["act"], sm["field"]: rec})
        self._pending.append({"act": g.transfer_state, "reason": reason})
        self._emit()
        try:
            self._call(T["tool"], {T.get("arg", "reason"): reason})
        except ToolError:
            pass
        raise _Stop()
