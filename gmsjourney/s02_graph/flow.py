"""The oracle: what the bot should do, computed by walking the policy graph's flow.

The graph's transitions carry small expressions (``call('record_charge_response') == 'SUCCESS'``,
``turns() > limit('max_turns')``) written in a vetted subset of Python: names, literals, comparisons and calls
to a fixed function table, nothing else. ``FlowSpec`` reads them from the graph and checks every name they use;
``GraphOracle`` walks them against the mock bank and the simulated customer and records each step, rule fired
and tool called. Like a referee with the rule book: given what the customer said and what the bank returned,
there is exactly one correct next move.

Parts of this file:
  1. The expression language (``parse``, ``evaluate``), the flow (``FlowSpec``, ``Transition``) and the walker
     (``GraphOracle``, ``Context``).
  2. ``PolicyOracle``: the name the rest of the code uses for the walker.
"""
from __future__ import annotations

import ast
import copy
import operator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable


# =====================================================================================================================
# PART 1 - The journey, executed from the policy graph.
#
# ``GraphOracle`` walks the graph's dialog-state machine. Nothing about the domain is written here: the
# states, the transitions out of each state, when each one applies, what it does, which rules it exercises,
# what each bot act carries and which arguments each tool takes all come from the graph
# (``s02_graph/triples.py`` emits them as triples):
#
#     (S, leads_to, T)      qualifiers  on      the customer act that triggers it, ``auto`` for none,
#                                               ``else`` for any act no other transition of S takes,
#                                               ``any`` for a check made before S prompts (the turn limit)
#                                       match   expression: does this transition apply to this frame at all
#                                       pre     statements run before ``guard`` (bind what the guard needs)
#                                       guard   expression: take this transition?
#                                       do      statements run when it is taken
#                                       say     acts said on the way, before T's own act
#                                       rules   rules exercised when it is taken
#                                       flush / quiet / discard / retract   turn-boundary controls (see ``_take``)
#     (A, carries, field)   qualifiers  from    expression for the field's value in act A (optional: omit if None)
#     (tool, has_arg, arg)  qualifiers  from    expression for the argument
#
# Expressions are a small, safe subset of Python (no attribute access, no comprehensions, only the functions
# in ``Context.FUNCTIONS``), evaluated against a ``Context``: the customer's frame, named counters and
# variables, the review queue, the tool log of this run, the exact numbers of the store and the graph.
#
# The walk:
#   1. Enter a state: say any ``say`` acts, then the state's act (its ``carries`` fields filled in).
#   2. A state that prompts the customer: first try its ``any`` transitions (the turn limit); otherwise end
#      the bot's turn, take the customer's next frame (the simulator, or a replayed frame list), and take the
#      first transition on that act whose ``match``, ``pre`` and ``guard`` pass (else the ``else`` ones).
#   3. A state that does not prompt: take the first ``auto`` transition whose guard passes.
#   4. A state with no way out ends the walk.
#
# Tool calls happen when an expression first asks for a tool's result in a step (``call('tool')``): the
# arguments come from ``has_arg``; a tool error reads as the status ``UNAVAILABLE``. So the order of the
# guards is the order of the calls, as in the procedure.
# =====================================================================================================================

from ..s04_cases.meaning import SemanticFrame
from ..s06_bots.tools import ToolEnvironment, ToolError
from ..s06_bots.tools import Trace, new_trace


class FlowError(RuntimeError):
    """The graph's flow specification cannot be executed as written."""


# ----------------------------------------------------------------------------- the expression language
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub}
_NAMES = {"True": True, "False": False, "None": None}


def parse(expr: str, functions: frozenset | None = None) -> ast.AST:
    """Parse and vet one expression (cached). ``functions``: the names it may call (default: the flow's)."""
    return _parse(expr, frozenset(Context.FUNCTIONS) if functions is None else functions)


@lru_cache(maxsize=4096)
def _parse(expr: str, functions: frozenset) -> ast.AST:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise FlowError(f"cannot parse {expr!r}: {e}") from None
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub,
                               ast.BinOp, ast.Add, ast.Sub, ast.Compare, ast.Call, ast.Name, ast.Load,
                               ast.Constant, ast.Tuple, ast.List, ast.IfExp, *tuple(_CMP)))
        if not ok:
            raise FlowError(f"{type(node).__name__} is not allowed in flow expressions: {expr!r}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.keywords):
            raise FlowError(f"only plain calls of named functions are allowed: {expr!r}")
        if isinstance(node, ast.Name) and node.id not in _NAMES and node.id not in functions:
            raise FlowError(f"unknown name {node.id!r} in {expr!r}")
    return tree


@lru_cache(maxsize=8192)
def _code(expr: str, functions: frozenset):
    """The vetted expression compiled once (the vetting above admits no attribute access, subscripts,
    comprehensions or unknown names, so evaluating the compiled form can only call the context's functions)."""
    return compile(_parse(expr, functions), "<flow expression>", "eval")


def evaluate(expr: str | None, ctx: Any) -> Any:
    """Evaluate an expression against a context whose ``f_<name>`` methods are the callable functions."""
    if expr is None or expr == "":
        return True
    fns = ctx.__dict__.get("_fns") if hasattr(ctx, "__dict__") else None
    if fns is None:
        fns = frozenset(ctx.FUNCTIONS)
        ns = {n: getattr(ctx, "f_" + n) for n in fns}
        try:
            ctx.__dict__["_fns"], ctx.__dict__["_ns"] = fns, ns
        except (AttributeError, TypeError):
            pass
    else:
        ns = ctx.__dict__["_ns"]
    return eval(_code(expr, fns), {"__builtins__": {}}, ns)          # noqa: S307 - vetted subset, see _parse


def _ev(n: ast.AST, ctx: "Context") -> Any:
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.Name):
        return _NAMES[n.id]
    if isinstance(n, (ast.Tuple, ast.List)):
        return tuple(_ev(e, ctx) for e in n.elts)
    if isinstance(n, ast.BoolOp):
        if isinstance(n.op, ast.And):
            v = True
            for e in n.values:
                v = _ev(e, ctx)
                if not v:
                    return v
            return v
        v = False
        for e in n.values:
            v = _ev(e, ctx)
            if v:
                return v
        return v
    if isinstance(n, ast.UnaryOp):
        v = _ev(n.operand, ctx)
        return (not v) if isinstance(n.op, ast.Not) else -v
    if isinstance(n, ast.BinOp):
        return _BIN[type(n.op)](_ev(n.left, ctx), _ev(n.right, ctx))
    if isinstance(n, ast.Compare):
        left = _ev(n.left, ctx)
        for op, comp in zip(n.ops, n.comparators):
            right = _ev(comp, ctx)
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(n, ast.IfExp):
        return _ev(n.body, ctx) if _ev(n.test, ctx) else _ev(n.orelse, ctx)
    if isinstance(n, ast.Call):
        return getattr(ctx, "f_" + n.func.id)(*[_ev(a, ctx) for a in n.args])
    raise FlowError(f"cannot evaluate {ast.dump(n)}")


# ----------------------------------------------------------------------------- the flow specification
@dataclass(frozen=True)
class Transition:
    """One way out of a state: on which customer act, under which guard, doing what, saying what, exercising
    which rules; plus the turn-boundary controls.
    """
    src: str
    dst: str
    on: str
    when: str = ""                      # the procedure's words (evidence), not executed
    match: str | None = None
    pre: tuple[str, ...] = ()
    guard: str | None = None
    do: tuple[str, ...] = ()
    say: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()
    flush: bool = False                 # end the bot's turn before taking it
    quiet: bool = False                 # do not say the target's act
    discard: bool = False               # drop what the bot was about to say (the turn never happened)
    retract: bool = False               # take back the source state's own act (pre-empted before it was said)


@dataclass
class FlowSpec:
    """The executable flow read from the graph: states, prompt states, transitions out of each state, what each
    act carries, what each tool is called with; every name an expression uses is checked to exist.
    """
    states: list[str]
    prompts: set[str]
    out: dict[str, list[Transition]]
    carries: dict[str, list[tuple[str, str, bool]]]          # act -> [(field, expression, optional)]
    args: dict[str, list[tuple[str, str]]]                   # tool -> [(arg, expression)]
    item_key: str | None                                     # id field of the reviewed items
    initial: str
    texts: dict[str, str]                                    # head -> has_text (opening statement, ...)
    kg: Any = None

    @classmethod
    def from_kg(cls, kg) -> "FlowSpec":
        states = [t["head"] for t in kg.where("is_a", tail="dialog_state")]
        prompts = set(kg.prompt_states)
        out: dict[str, list[Transition]] = {s: [] for s in states}
        incoming: set[str] = set()
        for t in kg.where("leads_to"):
            q = t.get("qualifiers") or {}
            tr = Transition(t["head"], t["tail"], q.get("on", "auto"), q.get("when", ""), q.get("match"),
                            tuple(q.get("pre", ())), q.get("guard"), tuple(q.get("do", ())), tuple(q.get("say", ())),
                            tuple(q.get("rules", ())), bool(q.get("flush")), bool(q.get("quiet")),
                            bool(q.get("discard")), bool(q.get("retract")))
            out.setdefault(tr.src, []).append(tr)
            incoming.add(tr.dst)
        carries: dict[str, list[tuple[str, str, bool]]] = {}
        for t in kg.where("carries"):
            q = t.get("qualifiers") or {}
            carries.setdefault(t["head"], []).append((t["tail"], q["from"], bool(q.get("optional"))))
        args: dict[str, list[tuple[str, str]]] = {}
        for t in kg.where("has_arg"):
            args.setdefault(t["head"], []).append((t["tail"], (t.get("qualifiers") or {})["from"]))
        id_fields = [t["tail"] for t in kg.where("returns_field") if (t.get("qualifiers") or {}).get("type") == "id"]
        marked = [t["head"] for t in kg.where("starts_dialog") if str(t["tail"]) == "yes"]
        initial = marked[0] if marked else next((s for s in states if s not in incoming), states[0] if states else "")
        texts = {t["head"]: t["tail"] for t in kg.where("has_text")}
        spec = cls(states, prompts, out, carries, args, id_fields[0] if id_fields else None, initial, texts, kg)
        spec.validate()
        return spec

    def validate(self) -> None:
        """Every expression parses and names only known functions; every target is a state."""
        for s, trs in self.out.items():
            for tr in trs:
                if tr.dst not in self.out and tr.dst not in self.states:
                    raise FlowError(f"{s} leads_to unknown state {tr.dst}")
                for e in (tr.match, tr.guard, *tr.pre, *tr.do, *tr.say):
                    if e:
                        parse(e)
        for _, fields in self.carries.items():
            for _, e, _ in fields:
                parse(e)
        for _, a in self.args.items():
            for _, e in a:
                parse(e)


# ----------------------------------------------------------------------------- the run context
class _Stop(Exception):
    pass


@dataclass
class Context:
    """Everything an expression can see or change during one walk."""
    spec: FlowSpec
    store: Any
    limits: Any
    tools: Any
    trace: Trace
    frame: SemanticFrame | None = None
    vars: dict[str, Any] = field(default_factory=dict)
    queue: list[dict[str, Any]] = field(default_factory=list)
    pos: int = -1
    truncated: bool = False
    step_calls: dict[str, dict[str, Any]] = field(default_factory=dict)   # tool -> its call in this step
    kb_cache: dict[str, Any] = field(default_factory=dict)
    path: list[tuple[str, str, str]] = field(default_factory=list)         # transitions taken, in order
    loops: dict[str, dict[str, Any]] = field(default_factory=dict)        # each-state -> its items and position
    item: Any = None                                                      # the loop item being handled

    FUNCTIONS = {
        "act", "slot", "get", "set", "n", "inc", "limit", "enm", "tail", "call", "field", "echoed", "count",
        "select", "size", "has_value", "load", "next_item", "item", "pending", "overflow", "recorded",
        "claim_recorded", "claim_action", "fire", "kb_answer", "kb_log", "end", "turns", "text", "options",
        "member", "loop_step", "where", "at", "writes",
    }

    # -- the customer's frame
    def f_act(self) -> str | None:
        return self.frame.act if self.frame is not None else None

    def f_slot(self, name: str) -> Any:
        return self.frame.get(name) if self.frame is not None else None

    # -- variables and counters
    def f_get(self, name: str) -> Any:
        return self.vars.get(name)

    def f_set(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def f_n(self, name: str) -> int:
        return int(self.vars.get(name) or 0)

    def f_inc(self, name: str) -> int:
        self.vars[name] = self.f_n(name) + 1
        return self.vars[name]

    # -- the store and the graph
    def f_limit(self, key: str) -> Any:
        return getattr(self.limits, key)

    def f_enm(self, category: str, key: str) -> Any:
        return self.store.lookup_enm(category, key)

    def f_tail(self, head: Any, relation: str) -> Any:
        if head is None:
            return None
        v = self.spec.kg.tails(str(head), relation)
        return v[0] if v else None

    def f_member(self, value: Any, value_set: str) -> bool:
        return value in self.spec.kg.value_set(value_set)

    def f_text(self, head: str) -> str:
        return self.spec.texts.get(head, "")

    # -- tools
    def f_call(self, tool: str) -> str:
        """The status of ``tool`` in this step, calling it the first time it is asked for."""
        if tool not in self.step_calls:
            args = {a: evaluate(e, self) for a, e in self.spec.args.get(tool, [])}
            try:
                res = self.tools.call(tool, **args)
                self.trace.events.append({"role": "tool", "tool": tool, "args": args, "result": res})
                self.step_calls[tool] = {"args": args, "result": res, "ok": True}
            except ToolError as e:
                self.trace.events.append({"role": "tool", "tool": tool, "args": args, "result": {"error": str(e)},
                                          "ok": False})
                self.step_calls[tool] = {"args": args, "result": {"status": "UNAVAILABLE"}, "ok": False}
        return (self.step_calls[tool]["result"] or {}).get("status")

    def _last(self, tool: str) -> dict[str, Any] | None:
        if tool in self.step_calls:
            return self.step_calls[tool]
        for e in reversed(self.trace.events):
            if e.get("role") == "tool" and e.get("tool") == tool:
                return {"args": e.get("args"), "result": e.get("result"), "ok": e.get("ok", True)}
        return None

    def f_field(self, tool: str, name: str) -> Any:
        c = self._last(tool)
        return ((c or {}).get("result") or {}).get(name)

    def f_options(self, tool: str, name: str) -> tuple:
        return tuple(self.f_field(tool, name) or ())

    def f_echoed(self, tool: str, *names: str) -> bool:
        c = self._last(tool) or {}
        res, args = c.get("result") or {}, c.get("args") or {}
        return all(res.get(k) == args.get(k) for k in names)

    def f_count(self, tool: str, status: str | None = None) -> int:
        n = 0
        for e in self.trace.events:
            if e.get("role") == "tool" and e.get("tool") == tool:
                if status is None or (e.get("result") or {}).get("status") == status:
                    n += 1
        return n

    # -- the reviewed items
    def f_select(self, tool: str, list_field: str, attr: str, max_value: Any) -> tuple:
        return tuple(x for x in (self.f_field(tool, list_field) or []) if x.get(attr) is not None and x[attr] <= max_value)

    def f_size(self, xs: Any) -> int:
        return len(xs or ())

    def f_has_value(self, xs: Any, attr: str, value: Any) -> bool:
        return any(x.get(attr) == value for x in (xs or ()))

    def f_load(self, items: Any, cap: int) -> None:
        items = list(items or ())
        self.truncated = len(items) > cap
        self.queue, self.pos = items[:cap], -1

    def f_next_item(self) -> Any:
        self.pos += 1
        iid = self.f_item()
        self.trace.presented.append(iid)
        for k in ("unsure", "unclear"):
            self.vars.pop(k, None)
        return iid

    def f_item(self) -> Any:
        if 0 <= self.pos < len(self.queue) and self.spec.item_key:
            return self.queue[self.pos].get(self.spec.item_key)
        return None

    def f_pending(self) -> bool:
        return self.pos < len(self.queue) - 1

    def f_overflow(self) -> bool:
        return self.truncated

    # -- loops over a returned list (a compiled pack's each-states)
    def f_where(self, xs: Any, attr: str, low: Any = None, high: Any = None) -> tuple:
        """The items of ``xs`` whose ``attr`` lies within [low, high] (either bound may be None)."""
        return tuple(x for x in (xs or ()) if x.get(attr) is not None and (low is None or x[attr] >= low)
                     and (high is None or x[attr] <= high))

    def f_at(self, obj: Any, key: str) -> Any:
        """A field of a loop item (or of any record)."""
        return obj.get(key) if isinstance(obj, dict) else None

    def f_loop_step(self, name: str, items: Any, cap: Any, var: str) -> str:
        """Advance the loop ``name``: the first call loads ``items``; each call moves to the next one, binds it to
        ``var`` and resets the per-item counters. Returns item | empty | over (more items than the cap) | done."""
        lp = self.loops.get(name)
        if lp is None:
            lp = self.loops[name] = {"items": list(items or ()), "i": -1}
            if not lp["items"]:
                return "empty"
        lp["i"] += 1
        n = len(lp["items"])
        cap = n if cap is None else cap
        if lp["i"] >= min(cap, n):
            self.item = None
            return "over" if n > cap else "done"
        self.item = lp["items"][lp["i"]]
        self.vars[var] = self.item
        for k in [k for k in self.vars if k.startswith(("count:", "unclear:"))]:
            del self.vars[k]
        return "item"

    def f_writes(self, tool: str, key: str, value: str) -> dict[str, Any]:
        """key -> value over the successful calls of a write tool (success status, echo matching)."""
        kg = self.spec.kg
        ok = (kg.tails(tool, "success_value") or [None])[0]
        echo = (kg.tails(tool, "echoes") or [None])[0]
        out: dict[str, Any] = {}
        for e in self.trace.events:
            if e.get("role") != "tool" or e.get("tool") != tool or e.get("ok") is False:
                continue
            r, a = e.get("result") or {}, e.get("args") or {}
            if r.get("status") == ok and (not echo or r.get(echo) == a.get(echo)):
                out[a.get(key)] = a.get(value)
        return out

    # -- what the bot may claim
    def f_recorded(self) -> dict[str, str]:
        return dict(self.trace.claims_recorded)

    def f_claim_recorded(self) -> None:
        self.trace.claims_recorded[self.f_item()] = self.vars.get("disposition")

    def f_claim_action(self) -> None:
        self.trace.claims_actions.append((self.vars.get("action"), self.vars.get("delivery")))

    # -- bookkeeping
    def f_fire(self, *rules: str) -> None:
        self.trace.rules_fired.update(rules)

    def f_kb_answer(self) -> Any:
        from ..s03_knowledge.questions import kb_answer
        key = self.frame.canonical() if self.frame is not None else ""
        if key not in self.kb_cache:
            self.kb_cache[key] = kb_answer(self.store, self.frame).answer
        return self.kb_cache[key]

    def f_kb_log(self) -> None:
        self.trace.kb.append({"frame": self.frame, "answer": self.f_kb_answer(), "correct": True})

    def f_end(self) -> None:
        self.trace.ended = True

    def f_turns(self) -> int:
        return self.trace.turns


# ----------------------------------------------------------------------------- the walker
class GraphOracle:
    """Expected behaviour, computed by walking the policy graph (drop-in for ``PolicyOracle``)."""

    def __init__(self, store, limits=None, spec: FlowSpec | None = None):
        self.store = store
        self.limits = limits or Limits.from_store(store)
        self.spec = spec or FlowSpec.from_kg(store.kg)
        self.opening = self.spec.texts.get("opening_statement", "")

    def run(self, spec, fixture: dict[str, Any] | None = None, frames_override: list[SemanticFrame] | None = None,
            gateway_factory: Callable | None = None, responder: Callable | None = None) -> Trace:
        """Play one journey. ``responder(last_act) -> frame | None`` answers each prompt; by default the user
        simulator for ``spec``; ``frames_override`` replays a fixed list of frames instead."""
        from ..s04_cases.journey import UserSimulator, build_fixture
        fixture = fixture or build_fixture(spec)
        env = ToolEnvironment(fixture)
        tools = gateway_factory(env) if gateway_factory else env
        if frames_override is not None:
            replay = list(frames_override)
            respond = lambda act: replay.pop(0) if replay else None          # noqa: E731
        elif responder is not None:
            respond = responder
        else:
            respond = UserSimulator(spec, fixture, getattr(self.limits, "lookback_days", None)).respond
        tr = new_trace()
        ctx = Context(self.spec, self.store, self.limits, tools, tr)
        try:
            self._walk(ctx, respond)
        except _Stop:
            pass
        tr.tool_log = list(env.log)
        tr.flow_path = list(ctx.path)                # which graph transitions this journey took
        if hasattr(tr, "finish"):
            tr.finish()
        return tr

    # ------------------------------------------------------------------ internals
    def _act(self, name: str, ctx: Context) -> dict[str, Any]:
        a: dict[str, Any] = {"act": name}
        for fld, expr, optional in self.spec.carries.get(name, []):
            v = evaluate(expr, ctx)
            if v is None and optional:
                continue
            a[fld] = copy.deepcopy(v)
        return a

    def _flush(self, ctx: Context, pending: list[dict[str, Any]]) -> None:
        if pending:
            ctx.trace.events.append({"role": "bot", "acts": list(pending)})
            pending.clear()

    def _choose(self, state: str, on: str | None, ctx: Context) -> Transition | None:
        trs = self.spec.out.get(state, [])
        groups = [[t for t in trs if t.on == on]] if on is not None else []
        if on not in (None, "auto", "any"):
            groups.append([t for t in trs if t.on == "else"])
        for group in groups:
            for t in group:
                if t.match and not evaluate(t.match, ctx):
                    continue
                for s in t.pre:
                    evaluate(s, ctx)
                if evaluate(t.guard, ctx):
                    return t
        return None

    def _take(self, t: Transition, ctx: Context, pending: list[dict[str, Any]]) -> str:
        ctx.path.append((t.src, t.dst, t.on))
        if t.discard:
            pending.clear()
        if t.retract and pending and pending[-1].get("act") == t.src:
            pending.pop()
        if t.flush:
            self._flush(ctx, pending)
        for s in t.do:
            evaluate(s, ctx)
        ctx.trace.rules_fired.update(t.rules)
        for s in t.say:
            name = evaluate(s, ctx)
            if name:
                pending.append(self._act(name, ctx))
        if not t.quiet:
            pending.append(self._act(t.dst, ctx))
        return t.dst

    def _walk(self, ctx: Context, respond: Callable) -> None:
        pending: list[dict[str, Any]] = []
        state = self.spec.initial
        pending.append(self._act(state, ctx))
        while True:
            ctx.step_calls = {}
            if not self.spec.out.get(state):                       # no way out: the walk ends here
                self._flush(ctx, pending)
                raise _Stop()
            if state in self.spec.prompts:
                t = self._choose(state, "any", ctx)
                if t is None:
                    self._flush(ctx, pending)
                    last = ctx.trace.events[-1]["acts"][-1] if ctx.trace.events and ctx.trace.events[-1].get("role") == "bot" else {}
                    frame = respond(dict(last))
                    if frame is None:
                        ctx.trace.notes.append("customer_silent")
                        ctx.trace.ended = True
                        raise _Stop()
                    ctx.trace.turns += 1
                    ctx.trace.user_frames.append(frame)
                    ev = {"role": "user", "frame": frame}
                    if hasattr(ctx.trace, "visited"):    # a compiled pack: the prompt this turn answers (and the item)
                        ev["state"] = state
                        if ctx.item is not None:
                            ev["item"] = dict(ctx.item)
                    ctx.trace.events.append(ev)
                    ctx.frame = frame
                    ctx.step_calls = {}
                    if frame.act == "CHITCHAT" and not any(x.on == "CHITCHAT" for x in self.spec.out.get(state, [])):
                        pending.append(self._act(state, ctx))   # small talk is no answer: ask again, no clarification
                        continue
                    t = self._choose(state, frame.act, ctx)
                    if t is None:
                        raise FlowError(f"no transition out of {state} on {frame.act}")
            else:
                t = self._choose(state, "auto", ctx)
                if t is None:
                    raise FlowError(f"no automatic transition out of {state} applies")
            state = self._take(t, ctx, pending)


# =====================================================================================================================
# PART 2 - The policy oracle: expected behaviour, computed rather than annotated (Ch. 12.4).
#
# The oracle is an executable reading of the policy graph: ``PolicyOracle`` is ``flow.GraphOracle``, which
# walks the graph's dialog states (``leads_to`` with its executable qualifiers), reads every number from the
# exact numeric register (lookup_enm) and plays the journey against the user simulator on the same backend
# fixture the bot will face. The resulting ``Trace`` is the ground truth: outcome signature, per-stage
# decisions, which rules fired and which workflow edges were traversed.
#
# The oracle consumes semantic frames only. It never reads text, so its answer depends on what the customer
# *meant*, which is exactly what the invariance proof (s05_rewording/proof.py) shows enrichment leaves unchanged.
# =====================================================================================================================

class Limits:
    """Every limit the policy graph declares (``(key, is_a, limit)``), read exactly from the store's ENM.

    Attributes are the graph's limit keys (``limits.max_turns``, ``limits.lookback_days`` ...); an attribute
    may be overridden on a copy (the turn-limit generator plays journeys with ``max_turns`` lifted)."""

    def __init__(self, values: dict[str, Any]):
        self.__dict__.update(values)

    @classmethod
    def from_store(cls, store) -> "Limits":
        kg = getattr(store, "kg", None)
        keys = [t["head"] for t in kg.where("is_a", tail="limit")] if kg is not None else \
            [k for (c, k) in store.enm if c == "limit"]
        return cls({k: store.lookup_enm("limit", k) for k in keys})

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __repr__(self) -> str:
        return f"Limits({self.__dict__})"


PolicyOracle = GraphOracle  # noqa: E402,F401  (flow imports Limits from here, lazily)
