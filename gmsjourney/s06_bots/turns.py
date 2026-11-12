"""One customer turn of a run compared with the same turn of the graph's own run: the trajectory and the side steps.

A turn is what the bot does between two customer lines: the tool calls it makes and the acts it says. Two kinds of
act do not move the conversation along the policy graph:

* side steps: the answer to a policy question (the graph's ``question.answer``, ANSWER_KB) and a clarification
  (the graph's ``unclear.clarify``, ASK_CLARIFY). A bot may clarify where the graph did not, or answer a question
  in other words; the conversation is still where the graph is;
* acknowledgements the bot may add or leave out (``OPTIONAL_ACTS``), and the same question asked again right after a
  side step (``REASK_AFTER``).

Everything else (the tool calls with their key arguments, and the other acts in order) is the **trajectory**.

    compare_turn(exp_tools, exp_acts, act_tools, act_acts) -> {"status", "tools_ok", "acts_ok", "core_ok"}
        status  "same"     the turn is what the graph did, side steps included
                "side"     an immaterial deviation: the trajectory is the graph's, only side steps differ (flagged,
                           the run goes on)
                "differs"  a material deviation: the trajectory is not the graph's
    asked_again(act_tools, act_acts, state) -> True when the bot made no call and is still asking the question the
        customer just answered (the customer then says it again; see harness.JourneyRunner)

Analogy: a driver following a route. Asking the passenger to repeat an address, or answering a question about the
fare, is not a wrong turn (an immaterial deviation); taking the wrong exit is (a material deviation). One wrong exit
can be put right at the next junction; two in a row, or stopping the car, and the trip is off the route.
"""
from __future__ import annotations

from typing import Any

from .tools import key_args

# Acknowledgements the bot may add or omit without changing what it does; the record they acknowledge is checked on
# the tool call itself.
OPTIONAL_ACTS = {"OPENING", "CONFIRM_RECORDED"}
REASK_AFTER = {"ASK_CLARIFY", "PROVIDE_CONTEXT", "ANSWER_KB"}


def side_acts() -> set[str]:
    """The acts that answer a policy question or ask for a clarification: the active graph's own names for them
    (``globals.question.answer``, ``globals.unclear.clarify``), else the older packs' ANSWER_KB and ASK_CLARIFY."""
    from ..s01_packs.loading import current
    g = current().graph
    if g is None:
        return {"ANSWER_KB", "ASK_CLARIFY"}
    out = set()
    q, u = g.glob.get("question") or {}, g.glob.get("unclear") or {}
    if q.get("answer"):
        out.add(q["answer"])
    if u.get("clarify"):
        out.add(u["clarify"])
    return out


def normalize_acts(acts: list[dict]) -> list[tuple]:
    """The decision content of a bot reply: (act, txn, reason) in order, without optional acknowledgements, without
    a re-presentation of the same charge right after a clarification / context / policy answer (on both sides alike,
    so a clarification in place of a policy answer stays a side step), and without immediate repeats."""
    out: list[tuple] = []
    prev = None
    for a in acts:
        name = a.get("act")
        if name in OPTIONAL_ACTS:
            continue
        if name == "PRESENT_TXN" and prev is not None and prev.get("act") in REASK_AFTER \
                and (prev.get("txn") in (None, a.get("txn"))):
            prev = a
            continue
        # a policy answer is about the question, not the charge on screen; its correctness is graded separately
        # (kb_correct: the figure must equal the exact register)
        key = (name, None if name == "ANSWER_KB" else a.get("txn"), a.get("reason"))
        if not out or out[-1] != key:
            out.append(key)
        prev = a
    return out


def tool_key(e: dict) -> tuple:
    """A tool call as its name and key arguments (what decides whether two calls are the same call)."""
    args = e.get("args") or {}
    return (e["tool"], tuple(sorted((k, str(v)) for k, v in args.items() if k in key_args().get(e["tool"], ()))))


def compare_turn(exp_tools: list[dict], exp_acts: list[dict], act_tools: list[dict], act_acts: list[dict],
                 side: set[str] | None = None) -> dict[str, Any]:
    """How a bot turn compares with the graph's turn (see the module docstring)."""
    side = side_acts() if side is None else side
    en, an = normalize_acts(exp_acts), normalize_acts(act_acts)
    core = lambda xs: [x for x in xs if x[0] not in side]          # noqa: E731
    tools_ok = [tool_key(x) for x in exp_tools] == [tool_key(x) for x in act_tools]
    acts_ok, core_ok = en == an, core(en) == core(an)
    status = "same" if tools_ok and acts_ok else "side" if tools_ok and core_ok else "differs"
    return {"status": status, "tools_ok": tools_ok, "acts_ok": acts_ok, "core_ok": core_ok}


def asked_again(act_tools: list[dict], act_acts: list[dict], state: str | None) -> bool:
    """The bot made no call and its last act asks the question the customer just answered (a re-ask)."""
    if act_tools or not act_acts or not state:
        return False
    return act_acts[-1].get("act") == state


def split_turns(events: list[dict]) -> tuple[dict, list[dict]]:
    """A run's events cut into the opening and one entry per customer line, each with its tool calls, its acts and
    its ``steps``: calls and acts in the order they happened (an act's ``at`` from s02_graph/pending.py places it
    among the calls; without one it comes after them).

        -> (opening, turns); every entry {"tools": [...], "acts": [...], "steps": [("call", event) | ("say", act)],
            "bot_text": [...]}, and a turn also {"frame", "text", "line", "repeat"} from its customer event."""
    def new(user: dict | None) -> dict:
        d = {"tools": [], "acts": [], "steps": [], "bot_text": [], "_keys": []}
        if user is not None:
            d.update({"frame": user.get("frame"), "text": user.get("text"), "line": user.get("line"),
                      "repeat": bool(user.get("repeat"))})
        return d
    opening, turns, cur, n = new(None), [], None, 0
    for ev in events:
        role = ev.get("role")
        if role == "user":
            cur = new(ev)
            turns.append(cur)
            continue
        tgt = cur if cur is not None else opening
        if role == "tool":
            n += 1
            tgt["tools"].append(ev)
            tgt["_keys"].append(((n, 0, len(tgt["_keys"])), ("call", ev)))
        elif role == "bot":
            acts = list(ev.get("acts") or [])
            at = ev.get("at") or [None] * len(acts)
            for a, k in zip(acts, at):
                tgt["acts"].append(a)
                tgt["_keys"].append(((n if k is None else min(k, n), 1, len(tgt["_keys"])), ("say", a)))
            if ev.get("text"):
                tgt["bot_text"].append(ev["text"])
    for d in [opening] + turns:
        d["steps"] = [s for _, s in sorted(d.pop("_keys"), key=lambda x: x[0])]
    return opening, turns
