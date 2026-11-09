"""Every name the policy graph defines, and the tools core code finds by role.

Code never spells a state, act, tool or value as a bare string (``"PRESENT_TXN"``). It asks the graph's name
registry (``N.state.PRESENT_TXN``), which fails loudly, with a suggestion, when a name does not exist; and it
finds tools by what they do (``roles.RECORD``: the tool that records an answer) rather than by their names. So a
renamed concept breaks where it is used, instead of silently matching nothing.

Parts of this file:
  1. ``GraphNames`` (``graph_names()``): the nomenclature read once from the active graph: states, prompts,
     acts and their slots, value sets, tools with their statuses, rules, triggers, limits, stages.
  2. Roles (module attributes resolved on every access): GATE, ITEMS, RECORD, OPTIONS, ACTION, TRANSFER, KEY,
     FEE_OPTION, ACCOUNT_TOOLS ... looked up in the active pack's graph.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any


# =====================================================================================================================
# PART 1 - The domain's nomenclature, read once from the policy graph and used everywhere else.
#
# Every name the harness works with (dialog states, the customer's acts and their slots, value sets and
# their values, tools and their statuses, limits, rules, escalation triggers, journey stages) is defined
# by the graph. ``GraphNames`` collects them; ``Names`` lets code refer to them as attributes that fail loudly
# when the graph does not define them:
#
#     D = GraphNames.from_kg(store.kg)
#     D.state.PRESENT_TXN          -> "PRESENT_TXN"
#     D.tool.record_charge_response
#     D.value.UNRECOGNIZED
#     D.state.PRESENT_TX           -> AttributeError: PRESENT_TX is not a dialog state in the policy graph
#                                     (did you mean PRESENT_TXN?)
#
# so a renamed or removed concept breaks at the line that uses it instead of silently matching nothing.
# =====================================================================================================================

class Names:
    """Attribute access to a set of names defined by the graph."""

    def __init__(self, kind: str, names):
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_names", tuple(dict.fromkeys(str(n) for n in names)))

    def __getattr__(self, name: str) -> str:
        if name.startswith("__"):
            raise AttributeError(name)
        if name in self._names:
            return name
        close = difflib.get_close_matches(name, self._names, n=1)
        hint = f" (did you mean {close[0]}?)" if close else ""
        raise AttributeError(f"{name} is not a {self._kind} in the policy graph{hint}")

    def __setattr__(self, name, value):
        raise AttributeError("graph names are read-only")

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"Names({self._kind}: {', '.join(self._names)})"


@dataclass(frozen=True)
class Tool:
    """One graph tool as the code sees it: stage, success and failure statuses, what happens on failure, arguments
    and returned fields.
    """
    name: str
    stage: str | None
    success: str | None
    failures: tuple[str, ...]
    on_failure: str | None
    args: tuple[str, ...]
    returns: dict[str, str]                  # field -> type


@dataclass(frozen=True)
class Rule:
    """One graph rule: id, stance, text, source and the fact it states."""
    rid: str
    stance: str | None
    text: str
    source: str
    fact: tuple[str, str, str] | None        # (subject, predicate, object)


@dataclass(frozen=True)
class Trigger:
    """One graph trigger: a condition (with the rule it serves) checked in the listed states."""
    name: str
    condition: str
    rule: str | None
    states: tuple[str, ...]


@dataclass
class GraphNames:
    """Every name the policy graph defines (states, acts, slots, value sets, aliases, tools, rules, limits,
    triggers ...), read once from the graph; the registry behind ``N`` and the roles.
    """
    title: str
    states: tuple[str, ...]
    prompt_states: tuple[str, ...]
    stage_of: dict[str, str]
    customer_acts: tuple[str, ...]
    slots: dict[str, dict[str, dict[str, Any]]]          # act -> slot -> {values_from, required}
    accepts: dict[str, tuple[str, ...]]                   # prompt state -> customer acts admissible there
    yes_no: dict[str, tuple[str | None, str | None]]      # prompt state -> (bare yes, bare no)
    value_sets: dict[str, tuple[str, ...]]
    aliases: dict[str, tuple[str, ...]]                    # canonical value -> customer synonyms
    tools: dict[str, Tool]
    limits: dict[str, Any]                                 # key -> exact value (graph has_value)
    rules: dict[str, Rule]
    triggers: dict[str, Trigger]
    stages: tuple[str, ...]
    transitions: list[dict[str, Any]] = field(default_factory=list)
    tool_roles: dict[str, str] = field(default_factory=dict)            # tool -> auth | read | write | handoff
    fee_free_values: tuple[str, ...] = ()                  # values whose choice needs no fee confirmation
    gate_tool: str | None = None                           # the tool account servicing requires
    chain: tuple[str, ...] = ()                            # servicing tools in `precedes` order after the gate
    state_tools: dict[str, tuple[str, ...]] = field(default_factory=dict)   # state -> tools it may call
    fee_values: tuple[str, ...] = ()                       # values whose choice needs a fee confirmation
    spoken: dict[str, tuple[str, ...]] = field(default_factory=dict)   # value -> how it is worded (spoken_as + aliases)
    echo_of: dict[str, str] = field(default_factory=dict)              # write tool -> the field its success echoes

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_kg(cls, kg) -> "GraphNames":
        def tails(h, r):
            return kg.tails(h, r)

        def first(h, r):
            v = tails(h, r)
            return v[0] if v else None

        states = tuple(t["head"] for t in kg.where("is_a", tail="dialog_state"))
        prompts = tuple(s for s in states if first(s, "prompts_customer") == "yes")
        acts = tuple(t["head"] for t in kg.where("is_a", tail="dialog_act"))
        value_sets: dict[str, list[str]] = {}
        for t in kg.where("member_of"):
            value_sets.setdefault(t["tail"], []).append(t["head"])
        tools = {}
        for t in kg.where("is_a", tail="tool"):
            n = t["head"]
            tools[n] = Tool(n, first(n, "has_stage"), first(n, "success_value"), tuple(tails(n, "failure_value")),
                            first(n, "on_failure"), tuple(tails(n, "has_arg")),
                            {r["tail"]: (r.get("qualifiers") or {}).get("type", "text")
                             for r in kg.where("returns_field", head=n)})
        facts = {}
        for t in kg.triples:
            q = t.get("qualifiers") or {}
            if t["kind"] == "policy" and q.get("rule") and "stance" in q:
                facts[q["rule"]] = (t["head"], t["relation"], str(t["tail"]))
        rules = {}
        for t in kg.where("is_a", tail="rule"):
            r = t["head"]
            rules[r] = Rule(r, first(r, "has_stance"), first(r, "has_text") or "", t.get("source", ""), facts.get(r))
        triggers = {}
        for t in kg.where("is_a", tail="escalation_trigger"):
            g = t["head"]
            triggers[g] = Trigger(g, first(g, "has_condition") or "", first(g, "justified_by"),
                                  tuple(tails(g, "fires_in_state")))
        stage_of = {s: first(s, "in_stage") for s in states}
        stages = tuple(dict.fromkeys(v for v in stage_of.values() if v))
        roles = {t["head"]: str(t["tail"]) for t in kg.where("has_role") if t["head"] in tools}
        gate = next((t["tail"] for t in kg.where("requires") if t["tail"] in tools), None) or \
            next((n for n, r in roles.items() if r == "auth"), None)
        chain, cur = [], gate
        while cur is not None:
            nxt = next((x for x in tails(cur, "precedes") if x in tools and x not in chain), None)
            if nxt:
                chain.append(nxt)
            cur = nxt
        return cls(
            title=kg.data.get("title", ""), states=states, prompt_states=prompts, stage_of=stage_of,
            customer_acts=acts,
            slots={a: {r["tail"]: dict(r.get("qualifiers") or {}) for r in kg.where("has_slot", head=a)} for a in acts},
            accepts={s: tuple(tails(s, "accepts_act")) for s in prompts},
            yes_no={s: (first(s, "interprets_yes_as"), first(s, "interprets_no_as")) for s in prompts},
            value_sets={k: tuple(v) for k, v in value_sets.items()},
            aliases={v: tuple(tails(v, "has_alias")) for vals in value_sets.values() for v in vals},
            tools=tools,
            limits={t["head"]: first(t["head"], "has_value") for t in kg.where("is_a", tail="limit")},
            rules=rules, triggers=triggers, stages=stages,
            transitions=[dict(src=t["head"], dst=t["tail"], **(t.get("qualifiers") or {})) for t in kg.where("leads_to")],
            gate_tool=gate, chain=tuple(chain),
            state_tools={s: tuple(tails(s, "calls_tool")) for s in states},
            fee_values=tuple(t["head"] for t in kg.where("requires_fee_confirmation") if str(t["tail"]) == "yes"),
            fee_free_values=tuple(t["head"] for t in kg.where("requires_fee_confirmation") if str(t["tail"]) == "no"),
            tool_roles=roles,
            echo_of={t["head"]: str(t["tail"]) for t in kg.where("echoes")},
            spoken={v: tuple(tails(v, "spoken_as")) + tuple(tails(v, "has_alias"))
                    for vals in value_sets.values() for v in vals},
        )

    # ------------------------------------------------------------------ names
    @property
    def state(self) -> Names:
        return Names("dialog state", self.states)

    @property
    def act(self) -> Names:
        return Names("customer act", self.customer_acts)

    @property
    def tool(self) -> Names:
        return Names("tool", self.tools)

    @property
    def value(self) -> Names:
        return Names("value", [v for vals in self.value_sets.values() for v in vals])

    @property
    def status(self) -> Names:
        """Every status a tool can return (its success and failure values)."""
        return Names("tool status", [x for n in self.tools for x in self.tool_outcomes(n)])

    @property
    def value_set(self) -> Names:
        return Names("value set", self.value_sets)

    @property
    def limit(self) -> Names:
        return Names("limit", self.limits)

    @property
    def rule(self) -> Names:
        return Names("rule", self.rules)

    @property
    def trigger(self) -> Names:
        return Names("escalation trigger", self.triggers)

    @property
    def stage(self) -> Names:
        return Names("stage", self.stages)

    def all_names(self) -> dict[str, tuple[str, ...]]:
        """Every graph-defined name, by kind (used to check that code does not spell names of its own)."""
        return {"dialog state": self.states, "customer act": self.customer_acts, "tool": tuple(self.tools),
                "value": tuple(self.value), "tool status": tuple(self.status), "value set": tuple(self.value_sets), "limit": tuple(self.limits),
                "rule": tuple(self.rules), "escalation trigger": tuple(self.triggers), "stage": self.stages}

    # ------------------------------------------------------------------ derived views
    def tool_outcomes(self, tool: str) -> tuple[str, ...]:
        t = self.tools[tool]
        return tuple(x for x in (t.success, *t.failures) if x)

    def options_for(self, state: str) -> list[dict[str, Any]]:
        """Every meaning card a customer may give in a prompt state: act, and for each slot its values."""
        out = []
        for a in self.accepts.get(state, ()):
            slots = {s: list(self.value_sets.get(q.get("values_from"), ())) for s, q in self.slots.get(a, {}).items()}
            out.append({"act": a, "slots": slots,
                        "required": [s for s, q in self.slots.get(a, {}).items() if q.get("required")]})
        return out

    def out_of(self, state: str) -> list[dict[str, Any]]:
        return [t for t in self.transitions if t["src"] == state]

    # ------------------------------------------------------------------ tool roles (derived, not named)
    # Code that needs "the record tool" asks for its role; the graph decides which tool plays it.
    @property
    def question_state(self) -> str | None:
        """The state a policy question is asked in when nothing else is pending: the first prompt the journey
        reaches once the gate passes (charge review: the item being presented)."""
        gate = self.tools.get(self.gate_tool) if self.gate_tool else None
        gate_states = {s for s in self.states if self.gate_tool in self.state_tools.get(s, ())}
        ok = gate.success if gate is not None else None
        frontier = [t["dst"] for t in self.transitions if t["src"] in gate_states and t["dst"] not in gate_states
                    and ok and ok in str(t.get("guard") or t.get("when") or "")]
        seen = set()
        while frontier:
            s = frontier.pop(0)
            if s in seen:
                continue
            seen.add(s)
            if s in self.prompt_states:
                return s
            frontier += [t["dst"] for t in self.transitions if t["src"] == s and t["dst"] not in seen]
        return next((s for s in self.prompt_states if s not in gate_states), None)

    @property
    def account_tools(self) -> frozenset[str]:
        """Tools that touch the account: the read and write tools (graph has_role), else everything after the
        authentication gate in `precedes` order."""
        if self.tool_roles:
            return frozenset(t for t, r in self.tool_roles.items() if r in ("read", "write"))
        return frozenset(self.chain)

    @property
    def write_tools(self) -> frozenset[str]:
        """Tools that change the account (role write), else the account tools that take arguments."""
        if self.tool_roles:
            return frozenset(t for t, r in self.tool_roles.items() if r == "write")
        return frozenset(t for t in self.chain if self.tools[t].args)

    @property
    def item_tool(self) -> str | None:
        """The tool that returns the items under review (it declares an id field)."""
        return next((n for n, t in self.tools.items() if "id" in t.returns.values()), None)

    @property
    def item_key(self) -> str | None:
        t = self.item_tool
        return next((f for f, ty in self.tools[t].returns.items() if ty == "id"), None) if t else None

    @property
    def record_tool(self) -> str | None:
        """The write that takes the item id: it records the customer's answer about one item."""
        return next((t for t in self.chain if t in self.write_tools and self.item_key in self.tools[t].args), None)

    @property
    def action_tool(self) -> str | None:
        """The other write: an account action not tied to one item."""
        return next((t for t in self.chain if t in self.write_tools and t != self.record_tool), None)

    @property
    def options_tool(self) -> str | None:
        """The read that comes just before the action in `precedes` order (it lists the eligible actions)."""
        c = list(self.chain)
        i = c.index(self.action_tool) if self.action_tool in c else -1
        return c[i - 1] if i > 0 and c[i - 1] not in self.write_tools else None

    @property
    def transfer_tool(self) -> str | None:
        """The handoff tool (role handoff), else the tool of the transfer stage."""
        h = next((t for t, r in self.tool_roles.items() if r == "handoff"), None)
        if h or "transfer" not in self.stages:
            return h
        return next((n for n, t in self.tools.items() if t.stage == "transfer"), None)

    def arg(self, tool: str, i: int) -> str:
        """The i-th argument the graph declares for a tool (graph order)."""
        return self.tools[tool].args[i]

    def ok(self, tool: str) -> str:
        """A tool's success status."""
        return self.tools[tool].success

    def summary(self) -> dict[str, Any]:
        return {"states": len(self.states), "prompt_states": len(self.prompt_states),
                "customer_acts": len(self.customer_acts), "value_sets": len(self.value_sets),
                "tools": len(self.tools), "limits": len(self.limits), "rules": len(self.rules),
                "triggers": len(self.triggers), "transitions": len(self.transitions)}


# ----------------------------------------------------------------------------- the registry in use
_CACHE: dict = {}


def graph_names() -> "GraphNames":
    """The names of the active domain pack's graph, read once per pack."""
    from .loading import current
    dom = current()
    key = (str(dom.dir), id(dom))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = GraphNames.from_kg(dom.kg)
    return _CACHE[key]


class _Registry:
    """``N.state.PRESENT_TXN``, ``N.tool.record_charge_response``, ``N.value.SUCCESS`` ... for the project's graph.
    Core code spells graph names through this, so a name the graph does not define fails where it is used."""

    def __getattr__(self, kind: str) -> Names:
        if kind.startswith("__"):
            raise AttributeError(kind)
        return getattr(graph_names(), kind)


N = _Registry()


def _reset() -> None:
    _CACHE.clear()


from .loading import on_switch as _on_switch  # noqa: E402  (domain imports nothing from here)

_on_switch(_reset)


# =====================================================================================================================
# PART 2 - Tools and values that core code refers to by role, resolved in the active pack's graph on every access.
#
# A tool's role is stated in the graph (``tool has_role auth | read | write | handoff``); charge review's graph
# also orders its servicing tools (``precedes``). From these:
#
#     GATE          the auth tool (role auth; else the tool account servicing ``requires``)
#     TRANSFER      the handoff tool                                 REASON  its argument
#     ACCOUNT_TOOLS / WRITE_TOOLS   read + write tools / write tools
#     ITEMS         the read tool that returns items with an id      KEY     that id field
#     RECORD        the write that takes the item id                 ANSWER  its other argument
#     ACTION        another write, not tied to an item               ACT / OPTION  its arguments
#     OPTIONS       the read just before ACTION in ``precedes`` order
#     DELIVERY      the values of the delivery value set (when the pack has one)
#     FEE_OPTION / FREE_OPTION  a value that needs a fee confirmation / one that does not
#     OK            the success status shared by the writes
#
# A role the pack does not have reads as None (supplementary card has no ITEMS, RECORD or OPTIONS). Use
# ``roles.RECORD`` (attribute access), not ``from roles import RECORD``: the active pack can change.
# =====================================================================================================================



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
