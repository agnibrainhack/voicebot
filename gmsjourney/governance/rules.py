"""Runtime rule checks as generic templates (domain.yaml -> rules).

Every policy rule the runtime monitor enforces is one of a small, closed set of check
templates; a domain pack only supplies parameters (which tool, which argument, which limit,
which rule id). Limits and figures are read from exact numeric memory when the checks are
compiled, never from the config.

Call checks (before every tool call, first non-allow decides):
  no_call_after_handoff  nothing after the handoff tool
  requires_auth          read / write tools only after the auth tool succeeded
  max_calls              a tool is called at most <limit> times
  item_scope             the item a tool writes was returned by the read tool, is inside the
                         window and within the per-session cap
  value_from_caller      the value written is the caller's latest clear answer for that slot
  terminal_value         once <value> is written the next action must be the handoff
  offered_only           the value chosen was among the options the read tool returned
  disclose_then_confirm  a figure was spoken and a confirmation followed before the write
  handoff_when_requested the caller asked for a person: the next action should be the handoff
  turn_limit             no further work past the turn limit
Completion checks (end of a conversation without a handoff):
  all_items_written, terminal_value_handed_off
"""
from __future__ import annotations

from typing import Any, Callable

from .gates import GateDecision, GateResult, GateState, ToolCall

A, D, E = GateDecision.ALLOW, GateDecision.DENY, GateDecision.ESCALATE
CallCheck = Callable[[ToolCall, GateState], GateResult]


def _domain():
    from ..domain import current
    return current()


def _tools() -> dict[str, dict]:
    return _domain().section("tools")


def _role_tool(role: str) -> str | None:
    return next((t for t, sp in _tools().items() if sp.get("role") == role), None)


def _ok_status(tool: str) -> str:
    sp = _tools()[tool]
    if sp.get("check"):
        return sp["check"]["success"]
    return (sp.get("status") or {}).get("ok") or (sp.get("status") or {}).get("some")


def figure(store, name: str) -> Any:
    """A limit or named figure from exact numeric memory."""
    dom = _domain()
    named = dom.get(f"figures.named.{name}")
    if named:
        return store.lookup_enm(named["enm_category"], named["key"])
    return store.lookup_enm(dom.get("figures.limits.enm_category", "limit"), name)


def _items(st: GateState, source: str, ok_only: bool = False) -> list[dict]:
    c = st.last(source)
    if not c or (ok_only and not c.get("ok")):
        return []
    return ((c.get("result") or {}).get(_tools()[source]["key"], []))


# ---------------------------------------------------------------------- call checks
def compile_call_checks(store, specs: list[dict] | None = None) -> list[CallCheck]:
    specs = specs if specs is not None else (_domain().get("rules.calls") or [])
    handoff = _role_tool("handoff")
    auth = _role_tool("auth")
    from ..sut.tools import account_tools
    acct = account_tools()

    def authed(st: GateState) -> bool:
        ok = _ok_status(auth)
        return any((c.get("result") or {}).get("status") == ok for c in st.calls(auth))

    out: list[CallCheck] = []
    for sp in specs:
        kind = sp["check"]
        rule = sp.get("rule")

        if kind == "no_call_after_handoff":
            def chk(a, st, rule=rule):
                if st.calls(handoff):
                    return GateResult(D, "policy", f"tool call after {handoff}", rule=rule)
                return GateResult(A, "policy")

        elif kind == "requires_auth":
            def chk(a, st, rule=rule):
                if a.tool_name in acct and not authed(st):
                    return GateResult(D, "policy", f"{a.tool_name} before {_ok_status(auth)}", rule=rule)
                return GateResult(A, "policy")

        elif kind == "max_calls":
            def chk(a, st, sp=sp, rule=rule):
                L = figure(store, sp["limit"])
                if a.tool_name == sp["tool"] and len(st.calls(sp["tool"])) >= L:
                    return GateResult(D, "policy", f"attempt {len(st.calls(sp['tool'])) + 1} > {sp['limit']}={L}",
                                      rule=rule)
                return GateResult(A, "policy")

        elif kind == "item_scope":
            def chk(a, st, sp=sp, rule=rule):
                if a.tool_name != sp["tool"]:
                    return GateResult(A, "policy")
                arg, win, cap = sp["arg"], sp.get("window"), sp.get("cap")
                tid = a.arguments.get(arg)
                got = {t[arg]: t for t in _items(st, sp["source"])} if st.last(sp["source"]) else {}
                if tid not in got:
                    return GateResult(D, "policy", f"{tid} was not returned by {sp['source']}", rule=rule)
                if win:
                    L = figure(store, win["limit"])
                    if got[tid].get(win["field"], 0) > L:
                        return GateResult(D, "policy", f"{tid} is {got[tid][win['field']]} days old > "
                                          f"{win['limit']}={L}", rule=win["rule"])
                if cap:
                    L = figure(store, cap["limit"])
                    done = {c["args"].get(arg) for c in st.calls(sp["tool"])}
                    if tid not in done and len(done) >= L:
                        return GateResult(D, "policy", f"{cap['noun']} {len(done) + 1} > {cap['limit']}={L}",
                                          rule=cap["rule"])
                return GateResult(A, "policy")

        elif kind == "value_from_caller":
            def chk(a, st, sp=sp):
                if a.tool_name != sp["tool"]:
                    return GateResult(A, "policy")
                said = None
                for f in reversed(st.user_frames):
                    if getattr(f, "act", None) == sp["act"]:
                        said = f.get(sp["slot"])
                        break
                want = a.arguments.get(sp["arg"])
                if said is None:
                    return GateResult(D, "policy", f"records {want} but the caller gave no clear answer",
                                      rule=sp["rule_missing"])
                if said != want:
                    return GateResult(D, "policy", f"records {want}; the caller's latest clear answer is {said}",
                                      rule=sp["rule_mismatch"])
                return GateResult(A, "policy")

        elif kind == "terminal_value":
            def chk(a, st, sp=sp, rule=rule):
                c = st.last(sp["tool"])
                if c and c["args"].get(sp["arg"]) == sp["value"] and a.tool_name != handoff:
                    return GateResult(D, "policy", f"{sp['value']} was recorded; the next action must be a transfer",
                                      rule=rule)
                return GateResult(A, "policy")

        elif kind == "offered_only":
            def chk(a, st, sp=sp, rule=rule):
                if a.tool_name == sp["tool"]:
                    c = st.last(sp["source"])
                    opts = ((c or {}).get("result") or {}).get(sp["key"], []) if c and c.get("ok") else []
                    if a.arguments.get(sp["arg"]) not in opts:
                        return GateResult(D, "policy", f"{a.arguments.get(sp['arg'])} not among returned options {opts}",
                                          rule=rule)
                return GateResult(A, "policy")

        elif kind == "disclose_then_confirm":
            def chk(a, st, sp=sp, rule=rule):
                if a.tool_name != sp["tool"]:
                    return GateResult(A, "policy")
                since = st.options_at if st.options_at is not None else 0
                confirmed = [i for i, f in enumerate(st.user_frames)
                             if i >= since and getattr(f, "act", None) == sp["confirm_act"]]
                if all(a.arguments.get(k) == v for k, v in sp["when"].items()):
                    if st.fee_disclosed_at is None or not any(i >= st.fee_disclosed_at for i in confirmed):
                        what = " ".join(str(v).replace("_", " ") for v in sp["when"].values())
                        return GateResult(D, "policy", f"{what} without disclosing the ${figure(store, sp['figure'])} "
                                          "fee and a confirmation after it", rule=rule)
                if not confirmed:
                    return GateResult(E, "policy", "submit without a customer confirmation", rule=sp["confirm_rule"])
                return GateResult(A, "policy")

        elif kind == "handoff_when_requested":
            def chk(a, st, rule=rule):
                if st.human_requested and a.tool_name != handoff:
                    return GateResult(E, "policy", "customer asked for a human; next action should be a transfer",
                                      rule=rule)
                return GateResult(A, "policy")

        elif kind == "turn_limit":
            def chk(a, st, sp=sp, rule=rule):
                L = figure(store, sp["limit"])
                if st.user_turns > L and a.tool_name != handoff:
                    return GateResult(E, "policy", f"turn {st.user_turns} > {sp['limit']}={L}", rule=rule)
                return GateResult(A, "policy")

        elif kind.startswith("graph_"):                  # a compact policy graph's checks (graph/model.py)
            chk = _graph_check(kind, sp, rule)

        else:
            raise ValueError(f"rules.calls: unknown check {kind!r}")
        chk.__name__ = kind
        out.append(chk)
    return out


def _graph_check(kind: str, sp: dict, rule: str | None) -> CallCheck:
    """Call checks of a compact policy graph's machine-checkable rules (the same rules the trace's
    admissibility scores): requires_status, value_allowed, requires_slot, confirm_before,
    disclose_before. The monitor sees the caller's frames and the bot's acts, never the oracle."""
    from ..domain import current
    g = current().graph
    k = kind[len("graph_"):]
    tool = sp[k]

    def chk(a, st, sp=sp, rule=rule):
        if a.tool_name != tool:
            return GateResult(A, "policy")
        if k == "requires_status":
            c = st.last(sp["needs"])
            if not c or (c.get("result") or {}).get("status") != sp["status"]:
                return GateResult(D, "policy", f"{tool} without {sp['needs']} {sp['status']}", rule=rule)
        elif k == "value_allowed":
            v = a.arguments.get(sp["arg"])
            if g.attr(v, sp["attr"]) not in sp["values"]:
                return GateResult(D, "policy", f"{tool} for {v}: {sp['attr']} is {g.attr(v, sp['attr'])}", rule=rule)
        elif k == "requires_slot":
            said = [f.get(sp["slot"]) for f in st.user_frames if getattr(f, "get", None) and f.get(sp["slot"]) is not None]
            if not said or said[-1] not in sp["values"]:
                return GateResult(D, "policy", f"{tool} without the caller saying {sp['slot']} in {sp['values']}", rule=rule)
        elif k == "confirm_before":
            last = st.user_frames[-1] if st.user_frames else None
            if getattr(last, "act", None) != sp["act"]:
                return GateResult(E, "policy", f"{tool} without a caller {sp['act']} just before", rule=rule)
        elif k == "disclose_before":
            v = a.arguments.get(sp["arg"])
            if g.attr(v, sp["attr"]) in sp["values"]:
                acts = getattr(st, "bot_acts", [])
                said = [i for i, x in acts if x.get("act") == sp["act"] and x.get(sp["arg"], v) == v]
                conf = [i for i, f in enumerate(st.user_frames) if said and i >= said[-1] and getattr(f, "act", None) == "CONFIRM"]
                if not said or not conf:
                    return GateResult(D, "policy", f"{tool} for {v} without {sp['act']} and a confirmation after it",
                                      rule=rule)
        return GateResult(A, "policy")
    return chk


# ---------------------------------------------------------------------- completion checks
def completion_results(store, st: GateState, specs: list[dict] | None = None) -> list[GateResult]:
    specs = specs if specs is not None else (_domain().get("rules.completion") or [])
    if st.calls(_role_tool("handoff")):
        return []
    out: list[GateResult] = []
    for sp in specs:
        kind = sp["check"]
        ok = _ok_status(sp["tool"])
        if kind == "all_items_written":
            arg, win = sp["arg"], sp["window"]
            look, cap = figure(store, win["limit"]), figure(store, sp["cap"])
            due = [t[arg] for t in _items(st, sp["source"], ok_only=True) if t.get(win["field"], 0) <= look][:cap]
            done = {x["args"].get(arg) for x in st.calls(sp["tool"]) if (x.get("result") or {}).get("status") == ok}
            missing = [t for t in due if t not in done]
            if missing:
                out.append(GateResult(E, "completeness", f"conversation ended with reviewable "
                                      f"transactions never recorded: {missing}", rule=sp["rule"]))
        elif kind == "terminal_value_handed_off":
            item_arg = next((c["arg"] for c in specs if c["check"] == "all_items_written"), "transaction_id")
            hits = [x["args"].get(item_arg) for x in st.calls(sp["tool"])
                    if x["args"].get(sp["arg"]) == sp["value"] and (x.get("result") or {}).get("status") == ok]
            if hits:
                out.append(GateResult(E, "completeness", f"{sp['value']} recorded for {hits} but the "
                                      "conversation ended without a transfer", rule=sp["rule"]))
        else:
            raise ValueError(f"rules.completion: unknown check {kind!r}")
    return out
