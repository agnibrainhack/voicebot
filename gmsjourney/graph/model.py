"""The compact policy graph (schema ``gmsjourney.policy_graph/2``).

One JSON file states the policy: limits, value sets with customer synonyms and attributes,
customer acts, tools with their result statuses, rules (optionally machine-checkable), the
transfer reasons, the always-available behaviours (a request for a person, a policy question,
an unclear answer, the turn limit) and the dialog as a state machine whose branches are
machine-readable:

  ask state     the bot asks; ``on`` maps each customer act to a target
  call state    the bot calls a tool; ``on`` maps each result status to a target
                (UNAVAILABLE = the call raised; ECHO_MISMATCH = success echoing another value)
  say state     the bot says something and moves on (``go``, any target)
  each state    a loop over the items a tool returned (``each: $tool.field``, ``as`` names the
                current item): items outside ``where`` are skipped, at most ``cap`` are visited
                (``over_cap`` after them when more remain), ``do`` starts the body, ``@next`` in
                the body moves on, ``empty`` / ``done`` when there is none / no more
  end state     the conversation ends

A target is a state name, ``!<reason>`` (transfer for that reason), a list of guarded
alternatives ``[{"if": {...}, "then": t}, ..., default]`` or a counter
``{"below": <limit>, "then": t, "else": t2}`` (calls of the state's tool so far against a
limit), or a named counter ``{"count": <name>, "upto": <limit>, "then": t, "else": t2}``
(incremented each time, reset per loop item). A guard ``{"slot": [values]}`` tests a captured
slot (``null`` = not given), ``{"slot.attr": [values]}`` an attribute of the captured value,
``{"slot": {"not": [values]}}`` the opposite, ``{"slot": {"in": "$tool.field"}}`` membership in
a returned list and ``{"#tool": [n]}`` how many times a tool was called. Field values ``$slot``, ``$slot.attr``, ``$tool.field``,
``$<limit>`` and ``$opening`` are resolved at run time; numbers always from exact memory.

Everything a campaign needs is mined from this file (``derive_config``); ``domain.yaml`` adds
only the customer wording (lexicon) and the enrichment catalogs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

SCHEMA_V2 = "gmsjourney.policy_graph/2"
UNAVAILABLE = "UNAVAILABLE"
ECHO_MISMATCH = "ECHO_MISMATCH"
STANCES = ("must", "must_not", "may")
CHECK_KINDS = ("requires_auth", "max_calls", "requires_status", "say_result", "value_allowed", "requires_slot",
               "disclose_before", "confirm_before", "claim_needs_success", "handoff_when_requested", "turn_limit",
               "no_call_after_handoff", "offered_only", "max_distinct", "item_window", "value_from_caller",
               "terminal_value", "forbidden_stance")
NEXT = "@next"                      # in a loop body: on to the next item
KB_ACT = "ASK_KB"
KB_SLOTS = {"intent": ("kb_intent", True), "attr": ("kb_attribute", False), "entity": (None, False),
            "key": ("limit", False), "name": ("option_name", False)}


def is_v2(path: str | Path) -> bool:
    p = Path(path)
    if p.suffix.lower() != ".json" or not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as fh:
            head = fh.read(400)
        return SCHEMA_V2 in head or json.loads(p.read_text(encoding="utf-8")).get("schema") == SCHEMA_V2
    except (OSError, ValueError):
        return False


def attr_relation(attr: str) -> str:
    """The graph relation an attribute is stored under (requires_x / is_x keep their name)."""
    return attr if attr.startswith(("requires_", "is_")) else f"has_{attr}"


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ============================================================================ the graph
class PolicyGraph:
    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self.data = data
        self.path = path
        self.values: dict[str, dict[str, dict[str, Any]]] = {}
        for vs, members in (data.get("values") or {}).items():
            norm = {}
            for m, spec in members.items():
                norm[m] = {"aliases": list(spec)} if isinstance(spec, list) else {"aliases": [], **spec}
            self.values[vs] = norm
        self.rules: dict[str, dict[str, Any]] = {}
        for rid, r in (data.get("rules") or {}).items():
            if isinstance(r, list):
                self.rules[rid] = {"stance": r[0], "text": r[1], "check": (r[2] if len(r) > 2 else None)}
            else:
                self.rules[rid] = {"stance": r["stance"], "text": r["text"], "check": r.get("check")}
        self.sha256 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "PolicyGraph":
        p = Path(path)
        return cls(json.loads(p.read_text(encoding="utf-8")), p)

    # --------------------------------------------------------------- plain sections
    id = property(lambda s: s.data["id"])
    title = property(lambda s: s.data.get("title", s.data["id"]))
    root = property(lambda s: s.data.get("root", f"{s.data['id']}_policy"))
    opening = property(lambda s: s.data.get("opening", ""))
    limits = property(lambda s: s.data.get("limits") or {})
    tools = property(lambda s: s.data.get("tools") or {})
    states = property(lambda s: s.data.get("states") or {})
    transfers = property(lambda s: s.data.get("transfers") or {})
    glob = property(lambda s: s.data.get("global") or {})
    knowledge = property(lambda s: list(s.data.get("knowledge") or []))

    def limit_value(self, key: str) -> int | float:
        return self.limits[key]["value"]

    @property
    def start(self) -> str:
        return next(iter(self.states))

    # --------------------------------------------------------------- vocabulary
    def member_set(self, value: str) -> str | None:
        for vs, members in self.values.items():
            if value in members:
                return vs
        return None

    def members(self, vs: str) -> list[str]:
        return list(self.values.get(vs, {}))

    def attr(self, value: str, attr: str) -> Any:
        vs = self.member_set(value)
        return self.values[vs][value].get(attr) if vs else None

    def aliases(self, value: str) -> list[str]:
        vs = self.member_set(value)
        return list(self.values[vs][value]["aliases"]) if vs else []

    def attributes(self, vs: str) -> list[str]:
        out: list[str] = []
        for spec in self.values.get(vs, {}).values():
            out += [k for k in spec if k != "aliases" and k not in out]
        return out

    def numeric_attributes(self) -> dict[str, dict[str, Any]]:
        """attr -> {"set": value set, "members": {value: number}} for every numeric attribute."""
        out: dict[str, dict[str, Any]] = {}
        for vs, members in self.values.items():
            for m, spec in members.items():
                for k, v in spec.items():
                    if k != "aliases" and _is_number(v):
                        out.setdefault(k, {"set": vs, "members": {}})["members"][m] = v
        return out

    @property
    def global_acts(self) -> dict[str, dict]:
        g = self.glob
        out = {}
        if g.get("human"):
            out[g["human"]["act"]] = {}
        if g.get("question"):
            out[g["question"]["act"]] = {}
        if g.get("unclear"):
            out[g["unclear"]["act"]] = {}
        return out

    def slot_specs(self, act: str) -> dict[str, dict[str, Any]]:
        """slot -> {"set", "optional", "with", ...} for a customer act."""
        raw = (self.data.get("acts") or {}).get(act) or {}
        out = {}
        for sl, sp in raw.items():
            if sl.startswith("_"):
                continue
            out[sl] = {"set": sp, "optional": False} if isinstance(sp, str) else {"optional": False, **sp}
        return out

    def updates(self, act: str) -> bool:
        """Does capturing this act keep earlier values of slots it does not give (an update)?"""
        return bool(((self.data.get("acts") or {}).get(act) or {}).get("_update"))

    @property
    def acts(self) -> dict[str, dict[str, str]]:
        """Every customer act and its slots (slot -> value set), the global ones included."""
        out = {a: {sl: sp["set"] for sl, sp in self.slot_specs(a).items()} for a in (self.data.get("acts") or {})}
        for a in self.global_acts:
            out.setdefault(a, {})
        out.setdefault("CHITCHAT", {})
        return out

    def slot_set(self, slot: str) -> str | None:
        for a, slots in self.acts.items():
            if slot in slots:
                return slots[slot]
        return None

    # --------------------------------------------------------------- states
    def kind(self, state: str) -> str:
        s = self.states[state]
        if s.get("end"):
            return "end"
        if "each" in s:
            return "each"
        if "call" in s:
            return "call"
        if "on" in s:
            return "ask"
        return "say"

    @property
    def prompt_states(self) -> list[str]:
        return [s for s in self.states if self.kind(s) == "ask"]

    @property
    def transfer_state(self) -> str:
        return self.transfers.get("state", "TRANSFER")

    @property
    def end_state(self) -> str:
        return next((s for s in self.states if self.states[s].get("end")), "END")

    def stage(self, state: str) -> str | None:
        if state == self.transfer_state:
            return self.tools[self.transfers["tool"]].get("stage", "handoff")
        return (self.states.get(state) or {}).get("stage")

    @property
    def stages(self) -> list[str]:
        out: list[str] = []
        for s in self.states:
            st = self.stage(s)
            if st and st not in out:
                out.append(st)
        h = self.stage(self.transfer_state)
        if h not in out:
            out.append(h)
        return out

    def accepted_acts(self, state: str) -> list[str]:
        s = self.states.get(state) or {}
        if self.kind(state) != "ask":
            return []
        return list(s["on"]) + [a for a in self.global_acts if a not in s["on"]]

    def interprets(self, state: str) -> tuple[str | None, str | None]:
        s = self.states.get(state) or {}
        if self.kind(state) != "ask":
            return None, None
        acc = set(s["on"])
        y = s.get("yes", "CONFIRM" if "CONFIRM" in acc else "UNCLEAR")
        n = s.get("no", "DENY" if "DENY" in acc else "UNCLEAR")
        return y, n

    # --------------------------------------------------------------- loops
    @property
    def loop_of(self) -> dict[str, str]:
        """state -> the each-state whose body it belongs to."""
        if "_loop_of" in self.__dict__:
            return self.__dict__["_loop_of"]
        out: dict[str, str] = {}
        for name, s in self.states.items():
            if self.kind(name) != "each":
                continue
            todo = [s["do"]]
            while todo:
                x = todo.pop()
                if x.startswith(("!", "@")) or x in out or x == name or x not in self.states:
                    continue
                out[x] = name
                todo += [t for t, _ in self._raw_next(x)]
        self.__dict__["_loop_of"] = out
        return out

    def _raw_next(self, state: str) -> list[tuple[str, dict]]:
        s = self.states[state]
        k = self.kind(state)
        if k == "say":
            return list(self.alternatives(s["go"]))
        if k in ("ask", "call"):
            return [x for tgt in s["on"].values() for x in self.alternatives(tgt)]
        if k == "each":
            return [(s[f], {}) for f in ("do", "empty", "over_cap", "done") if s.get(f)]
        return []

    # --------------------------------------------------------------- targets
    def resolve(self, target: Any, src: str | None = None) -> str:
        """The state a plain target names (a transfer target -> the transfer state, @next -> its loop)."""
        if isinstance(target, str) and target.startswith("!"):
            return self.transfer_state
        if target == NEXT:
            return self.loop_of.get(src, src)
        return target

    def alternatives(self, target: Any) -> Iterator[tuple[str, dict]]:
        """Every (plain target, condition) a target can lead to."""
        if isinstance(target, str):
            yield target, {}
        elif isinstance(target, list):
            for alt in target:
                if isinstance(alt, dict) and "then" in alt:
                    cond = alt.get("if", {})
                    if isinstance(cond, list):                 # all of several conditions (keys may repeat)
                        merged: dict[str, Any] = {}
                        for i, c0 in enumerate(cond):
                            for k, v in c0.items():
                                merged[k if k not in merged else f"{k}#{i}"] = v
                        cond = merged
                    for t, c in self.alternatives(alt["then"]):
                        yield t, {**cond, **c}
                else:
                    yield from self.alternatives(alt)
        elif isinstance(target, dict) and "below" in target:
            for t, c in self.alternatives(target["then"]):
                yield t, {"below": target["below"], **c}
            for t, c in self.alternatives(target["else"]):
                yield t, {"at_least": target["below"], **c}
        elif isinstance(target, dict) and "count" in target:
            for t, c in self.alternatives(target["then"]):
                yield t, {"count_upto": target["upto"], **c}
            for t, c in self.alternatives(target["else"]):
                yield t, {"count_over": target["upto"], **c}
        else:
            raise ValueError(f"bad target {target!r}")

    def edges(self) -> list[dict[str, Any]]:
        """Every transition of the machine: src, dst, on (act | status | auto | any), when, rules,
        reason (for a transfer)."""
        out = []
        for name, s in self.states.items():
            k = self.kind(name)
            if k == "say":
                for t, cond in self.alternatives(s["go"]):
                    e = {"src": name, "dst": self.resolve(t, name), "on": "auto",
                         "when": json.dumps(cond, sort_keys=True, default=str) if cond else "always",
                         "rules": s.get("rules", [])}
                    if t.startswith("!"):
                        e["reason"] = t[1:]
                        e["rules"] = e["rules"] + [self.transfers["reasons"][t[1:]]]
                    out.append(e)
            elif k == "each":
                for f, when in (("do", "next item"), ("empty", "no item in scope"), ("over_cap", f"more than {s.get('cap')}"),
                                ("done", "no item left")):
                    if s.get(f):
                        t = s[f]
                        e = {"src": name, "dst": self.resolve(t, name), "on": "auto", "when": when, "rules": s.get("rules", [])}
                        if t.startswith("!"):
                            e["reason"] = t[1:]
                        out.append(e)
            elif k in ("ask", "call"):
                for key, target in s["on"].items():
                    for t, cond in self.alternatives(target):
                        e = {"src": name, "dst": self.resolve(t, name), "on": key if k == "ask" else f"{s['call']}:{key}",
                             "when": json.dumps(cond, sort_keys=True, default=str) if cond else "always",
                             "rules": s.get("rules", [])}
                        if t.startswith("!"):
                            e["reason"] = t[1:]
                            e["rules"] = e["rules"] + [self.transfers["reasons"][t[1:]]]
                        out.append(e)
        g = self.glob
        for st in self.prompt_states:
            if g.get("human"):
                out.append({"src": st, "dst": self.transfer_state, "on": g["human"]["act"], "when": "customer asks for a human",
                            "reason": g["human"]["then"][1:], "rules": [self.transfers["reasons"][g["human"]["then"][1:]]]})
            if g.get("question"):
                out.append({"src": st, "dst": st, "on": g["question"]["act"], "when": "policy question, then re-ask",
                            "rules": [g["question"]["rule"]]})
            if g.get("unclear"):
                u = g["unclear"]
                out.append({"src": st, "dst": st, "on": u["act"], "when": f"unclear, below {u['limit']}",
                            "rules": [u["rule"]]})
                out.append({"src": st, "dst": self.transfer_state, "on": u["act"], "when": f"still unclear after {u['limit']}",
                            "reason": u["else"][1:], "rules": [u["rule"], self.transfers["reasons"][u["else"][1:]]]})
            if g.get("turn_limit"):
                tl = g["turn_limit"]
                out.append({"src": st, "dst": self.transfer_state, "on": "any", "when": f"{tl['limit']} reached",
                            "reason": tl["else"][1:], "rules": [tl["rule"]]})
        if self.transfers:
            out.append({"src": self.transfer_state, "dst": self.transfers.get("then", self.end_state), "on": "auto",
                        "when": f"{self.transfers['tool']} called", "rules": []})
        return out

    def reasons_in(self, state: str) -> list[str]:
        """Transfer reasons that can fire in a state."""
        return sorted({e["reason"] for e in self.edges() if e["src"] == state and e.get("reason")})

    def tools_called_from(self, state: str) -> list[str]:
        """Tools the bot may call next from a state (the call states it reaches without a prompt)."""
        seen, out, todo = set(), [], [state]
        while todo:
            s = todo.pop()
            for e in self.edges():
                if e["src"] != s or e["dst"] in seen:
                    continue
                d = e["dst"]
                seen.add(d)
                if d == self.transfer_state:
                    t = self.transfers["tool"]
                    if t not in out:
                        out.append(t)
                elif self.kind(d) == "call":
                    if self.states[d]["call"] not in out:
                        out.append(self.states[d]["call"])
                elif self.kind(d) in ("say", "each"):
                    todo.append(d)
        return out

    # --------------------------------------------------------------- workflow (tool level)
    def workflow(self) -> list[dict[str, str]]:
        """Tool-level workflow: start -> tool -> ... -> end, one edge per (from, to) pair the
        machine allows, with the condition that takes it."""
        start, end = "start", "end"
        pairs: dict[tuple[str, str], list[str]] = {}

        def node(state: str) -> str | None:
            if state == self.transfer_state:
                return self.transfers["tool"]
            if self.kind(state) == "call":
                return self.states[state]["call"]
            if self.kind(state) == "end":
                return end
            return None

        def nexts(state: str, via: str, seen: frozenset) -> Iterator[tuple[str, str]]:
            for e in self.edges():
                if e["src"] != state:
                    continue
                d, cond = e["dst"], (e["on"] if e["on"] != "auto" else via)
                n = node(d)
                if n is not None:
                    yield n, cond
                elif d not in seen:
                    yield from nexts(d, cond, seen | {d})

        sources = [(start, self.start)] + [(self.states[s]["call"], s) for s in self.states if self.kind(s) == "call"]
        sources.append((self.transfers["tool"], self.transfer_state))
        for n0, st in sources:
            for n1, cond in nexts(st, "always", frozenset({st})):
                pairs.setdefault((n0, n1), [])
                if cond not in pairs[(n0, n1)]:
                    pairs[(n0, n1)].append(cond)
        out = []
        for i, ((a, b), conds) in enumerate(sorted(pairs.items(), key=lambda x: (x[0][0] != "start", x[0])), start=1):
            out.append({"wid": f"W{i}", "src": a, "dst": b, "condition": " | ".join(conds)})
        return out

    # --------------------------------------------------------------- checks
    def rule_checks(self) -> list[tuple[str, str, dict]]:
        """(rule id, check kind, parameters) for every machine-checkable rule."""
        out = []
        for rid, r in self.rules.items():
            cs = r.get("check")
            for c in (cs if isinstance(cs, list) else [cs] if cs else []):
                kind = next(k for k in c if k in CHECK_KINDS)
                out.append((rid, kind, c))
        return out

    def validate(self) -> list[str]:
        errs: list[str] = []
        d = self.data
        if d.get("schema") != SCHEMA_V2:
            errs.append(f"schema {d.get('schema')!r} != {SCHEMA_V2}")
        for k in ("id", "limits", "tools", "rules", "states"):
            if not d.get(k):
                errs.append(f"{k} missing")
        if errs:
            return errs
        states, tools, rules = set(self.states), self.tools, self.rules
        acts = self.acts
        for lk, lv in self.limits.items():
            if not _is_number(lv.get("value")):
                errs.append(f"limits.{lk}: value must be a number")
        for rid, r in rules.items():
            if r["stance"] not in STANCES:
                errs.append(f"rules.{rid}: stance {r['stance']!r} not in {STANCES}")
            cs = r.get("check")
            for c in (cs if isinstance(cs, list) else [cs] if cs else []):
                if not any(k in c for k in CHECK_KINDS):
                    errs.append(f"rules.{rid}: check has none of {CHECK_KINDS}")
        for a, slots in acts.items():
            for sl, vs in slots.items():
                if vs not in self.values and vs not in ("credential_code", "text"):
                    errs.append(f"acts.{a}.{sl}: value set {vs!r} not declared")
        for tname, t in tools.items():
            if t.get("role") not in ("auth", "read", "write", "handoff"):
                errs.append(f"tools.{tname}: role must be auth / read / write / handoff")
            if "ok" not in t:
                errs.append(f"tools.{tname}: no ok status")
        tr = self.transfers
        if tr:
            if tr.get("tool") not in tools:
                errs.append(f"transfers.tool {tr.get('tool')!r} is not a tool")
            for reason, rid in tr.get("reasons", {}).items():
                if rid not in rules:
                    errs.append(f"transfers.reasons.{reason}: {rid} is not a rule")
        reasons = set(tr.get("reasons", {}))

        def chk_target(where: str, target: Any, cond_slots: set[str]):
            try:
                alts = list(self.alternatives(target))
            except ValueError as e:
                errs.append(f"{where}: {e}")
                return
            for t, cond in alts:
                if t.startswith("!"):
                    if t[1:] not in reasons:
                        errs.append(f"{where}: transfer reason {t[1:]!r} not in transfers.reasons")
                elif t == NEXT:
                    if where.split(".")[1] not in self.loop_of:
                        errs.append(f"{where}: {NEXT} outside a loop body")
                elif t not in states:
                    errs.append(f"{where}: {t!r} is not a state")
                for k in cond:
                    if k in ("below", "at_least", "count_upto", "count_over"):
                        if cond[k] not in self.limits:
                            errs.append(f"{where}: {cond[k]!r} is not a limit")
                        continue
                    if k.startswith("#"):
                        if k[1:] not in tools:
                            errs.append(f"{where}: {k[1:]!r} is not a tool")
                        continue
                    v = cond[k]
                    if isinstance(v, dict) and "in" in v:
                        errs.extend(self._check_ref(f"{where}.{k}.in", v["in"], cond_slots))
                    slot, _, attr = k.split("#")[0].partition(".")
                    if slot not in cond_slots:
                        errs.append(f"{where}: condition on {slot!r}, which no act captures")
                    vs = self.slot_set(slot)
                    if attr and vs and attr not in self.attributes(vs):
                        errs.append(f"{where}: {vs} has no attribute {attr!r}")

        all_slots = {sl for slots in acts.values() for sl in slots}
        all_slots |= {s["as"] for s in self.states.values() if "each" in s and s.get("as")}
        for name, s in self.states.items():
            k = self.kind(name)
            for r in s.get("rules", []):
                if r not in rules:
                    errs.append(f"states.{name}: rule {r} not declared")
            if s.get("stage") is None:
                errs.append(f"states.{name}: no stage")
            if k == "say":
                chk_target(f"states.{name}.go", s.get("go"), all_slots)
            if k == "each":
                errs += self._check_ref(f"states.{name}.each", s["each"], all_slots)
                for f in ("do", "empty", "over_cap", "done"):
                    if s.get(f) is not None:
                        chk_target(f"states.{name}.{f}", s[f], all_slots)
                for lim in [s.get("cap")] + [w.get("max") for w in (s.get("where") or {}).values()]:
                    if lim is not None and lim not in self.limits:
                        errs.append(f"states.{name}: {lim!r} is not a limit")
            if k == "ask":
                for a, target in s["on"].items():
                    if a not in acts:
                        errs.append(f"states.{name}.on: {a!r} is not a customer act")
                    chk_target(f"states.{name}.on.{a}", target, all_slots)
            if k == "call":
                t = s["call"]
                if t not in tools:
                    errs.append(f"states.{name}: tool {t!r} not declared")
                    continue
                statuses = {tools[t]["ok"], *tools[t].get("fail", []), UNAVAILABLE}
                if tools[t].get("echo"):
                    statuses.add(ECHO_MISMATCH)
                for st in statuses - set(s["on"]):
                    errs.append(f"states.{name}: no branch for {t} status {st}")
                for st, target in s["on"].items():
                    if st not in statuses:
                        errs.append(f"states.{name}.on: {st!r} is not a status of {t}")
                    chk_target(f"states.{name}.on.{st}", target, all_slots)
                for arg in s.get("args", {}):
                    if arg not in (tools[t].get("args") or {}):
                        errs.append(f"states.{name}.args: {arg!r} is not an argument of {t}")
            for fld, ref in (s.get("say") or {}).items():
                errs += self._check_ref(f"states.{name}.say.{fld}", ref, all_slots)
            if s.get("claims") and s["claims"] not in tools:
                errs.append(f"states.{name}.claims: {s['claims']!r} is not a tool")
        # reachability
        reach, todo = {self.start}, [self.start]
        edges = self.edges()
        while todo:
            x = todo.pop()
            for e in edges:
                if e["src"] == x and e["dst"] not in reach:
                    reach.add(e["dst"])
                    todo.append(e["dst"])
        unreached = states - reach
        if unreached:
            errs.append(f"unreachable states: {sorted(unreached)}")
        if self.end_state not in states:
            errs.append("no end state")
        # every rule is applied somewhere (a state, a transfer reason, a global behaviour or a check)
        used = {r for s in self.states.values() for r in s.get("rules", [])} | set(tr.get("reasons", {}).values())
        used |= {g.get("rule") for g in self.glob.values() if g.get("rule")}
        used |= {rid for rid, _, _ in self.rule_checks()}
        for rid in set(rules) - used:
            errs.append(f"rule {rid} is applied nowhere (no state, transfer, global behaviour or check)")
        for rid, kind, c in self.rule_checks():
            for key in ("max_calls", "requires_status", "needs", "value_allowed", "requires_slot", "disclose_before",
                        "confirm_before", "tool"):
                if key in c and isinstance(c[key], str) and c[key] not in tools:
                    errs.append(f"rules.{rid}.{key}: {c[key]!r} is not a tool")
            if kind == "claim_needs_success" and c["claim_needs_success"] not in states:
                errs.append(f"rules.{rid}.claim_needs_success: {c['claim_needs_success']!r} is not a state")
            for key in ("offered_only", "max_distinct", "item_window", "value_from_caller", "terminal_value", "source"):
                if key in c and isinstance(c[key], str) and c[key] not in tools:
                    errs.append(f"rules.{rid}.{key}: {c[key]!r} is not a tool")
            for key in ("limit", "turn_limit"):
                if key in c and isinstance(c[key], str) and c[key] not in self.limits:
                    errs.append(f"rules.{rid}.{key}: {c[key]!r} is not a limit")
            if kind == "say_result" and c["say_result"] not in states:
                errs.append(f"rules.{rid}.say_result: {c['say_result']!r} is not a state")
            if kind in ("disclose_before", "confirm_before") and c.get("act") not in states | set(acts):
                errs.append(f"rules.{rid}.act: {c.get('act')!r} is neither a state nor an act")
        g = self.glob
        for key in ("human", "question", "unclear"):
            if g.get(key) and g[key].get("act") is None:
                errs.append(f"global.{key}: no act")
        for key in ("unclear", "turn_limit"):
            if g.get(key) and g[key].get("limit") not in self.limits:
                errs.append(f"global.{key}.limit: {g[key].get('limit')!r} is not a limit")
        return errs

    def _check_ref(self, where: str, ref: Any, slots: set[str]) -> list[str]:
        if isinstance(ref, dict) and "$writes" in ref:
            t = ref["$writes"]
            if t not in self.tools:
                return [f"{where}: {t!r} is not a tool"]
            args = self.tools[t].get("args") or {}
            return [f"{where}: {t} has no argument {ref[k]!r}" for k in ("key", "value") if ref.get(k) not in args]
        if not isinstance(ref, str) or not ref.startswith("$"):
            return []
        body = ref[1:]
        if body == "opening" or body in self.limits:
            return []
        head, _, tail = body.partition(".")
        if head in self.tools:
            ret = self.tools[head].get("returns") or {}
            return [] if tail in ret or tail == "status" else [f"{where}: {head} returns no {tail!r}"]
        loops = {s["as"]: s for s in self.states.values() if "each" in s and s.get("as")}
        if head in loops:
            return []
        if head in slots:
            vs = self.slot_set(head)
            if tail and vs and tail not in self.attributes(vs):
                return [f"{where}: {vs} has no attribute {tail!r}"]
            return []
        return [f"{where}: {ref!r} names no slot, tool result, limit or $opening"]


# ============================================================================ triples
def to_triples(g: PolicyGraph) -> list[dict[str, Any]]:
    """The graph as head / relation / tail triples (the PolicyKG shape every reader uses)."""
    out: list[dict[str, Any]] = []

    def add(h, r, t, kind, source, *, evidence="", qualifiers=None, gms=True, state=None):
        tr = {"id": f"G{len(out) + 1:04d}", "head": h, "relation": r, "tail": t,
              "tail_type": "literal" if not isinstance(t, str) or t in ("yes", "no") else "entity",
              "kind": kind, "source": source, "evidence": evidence, "gms": gms}
        if qualifiers:
            tr["qualifiers"] = qualifiers
        if state:
            tr["state"] = state
        out.append(tr)

    root = g.root
    add(root, "is_a", "policy", "policy", "policy")
    for k, lv in g.limits.items():
        add(k, "is_a", "limit", "policy", f"limits.{k}", evidence=lv.get("text", ""))
        add(k, "has_value", lv["value"], "policy", f"limits.{k}", evidence=lv.get("text", ""))
        add(k, "has_unit", lv.get("unit", ""), "policy", f"limits.{k}", gms=False)
        add(k, "has_text", lv.get("text", ""), "policy", f"limits.{k}", gms=False)
        add(root, f"has_{k}", lv["value"], "policy", f"limits.{k}", evidence=lv.get("text", ""))
    for vs, members in g.values.items():
        add(vs, "is_a", "value_set", "domain", f"values.{vs}")
        for m, spec in members.items():
            src = f"values.{vs}.{m}"
            add(m, "member_of", vs, "domain", src)
            for a in spec["aliases"]:
                add(m, "has_alias", a, "lexicon", src)
            for k, v in spec.items():
                if k != "aliases":
                    add(m, attr_relation(k), v, "domain", src)
    for a, slots in g.acts.items():
        add(a, "is_a", "dialog_act", "dialog", f"acts.{a}")
        specs = g.slot_specs(a)
        for sl, vs in slots.items():
            req = not (specs.get(sl) or {}).get("optional")
            add(a, "has_slot", sl, "dialog", f"acts.{a}", qualifiers={"values_from": vs, "required": req})
    q = g.glob.get("question")
    if q:
        for sl, (vs, req) in KB_SLOTS.items():
            add(q["act"], "has_slot", sl, "dialog", "global.question", qualifiers={"values_from": vs, "required": req})
    for t, sp in g.tools.items():
        src = f"tools.{t}"
        add(t, "is_a", "tool", "tool", src, evidence=sp.get("text", ""))
        add(t, "has_stage", sp.get("stage", ""), "tool", src)
        add(t, "has_role", sp["role"], "tool", src)
        add(t, "success_value", sp["ok"], "tool", src)
        for f in list(sp.get("fail", [])) + [UNAVAILABLE]:
            add(t, "failure_value", f, "tool", src)
        if sp.get("text"):
            add(t, "has_text", sp["text"], "tool", src, gms=False)
    for rid, r in g.rules.items():
        src = f"rules.{rid}"
        add(rid, "is_a", "rule", "policy", src, evidence=r["text"])
        add(rid, "has_stance", r["stance"], "policy", src)
        add(rid, "has_text", r["text"], "policy", src, gms=False)
        subj, pred, obj = rule_reification(g, rid)
        add(rid, "constrains", subj, "policy", src)
        add(subj, pred, obj, "policy", src, qualifiers={"rule": rid, "stance": r["stance"]}, evidence=r["text"])
    for name, s in g.states.items():
        src = f"states.{name}"
        add(name, "is_a", "dialog_state", "dialog", src)
        add(name, "in_stage", g.stage(name), "dialog", src, state=name)
        add(name, "prompts_customer", "yes" if g.kind(name) == "ask" else "no", "dialog", src, state=name)
        for a in g.accepted_acts(name):
            add(name, "accepts_act", a, "dialog", src, state=name)
        y, n = g.interprets(name)
        if y:
            add(name, "interprets_yes_as", y, "dialog", src, state=name)
            add(name, "interprets_no_as", n, "dialog", src, state=name)
        for r in s.get("rules", []):
            add(name, "applies_rule", r, "dialog", src, state=name)
        for t in g.tools_called_from(name) if g.kind(name) != "end" else []:
            add(name, "calls_tool", t, "dialog", src, state=name)
        lims = set()
        for _, tgt in (s.get("on") or {}).items():
            for _, cond in g.alternatives(tgt):
                lims |= {cond[k] for k in ("below", "at_least") if k in cond}
        for lim in sorted(lims):
            add(name, "limited_by", lim, "dialog", src, state=name)
    ts = g.transfer_state
    if g.transfers:
        add(ts, "is_a", "dialog_state", "dialog", "transfers")
        add(ts, "in_stage", g.stage(ts), "dialog", "transfers", state=ts)
        add(ts, "prompts_customer", "no", "dialog", "transfers", state=ts)
        add(ts, "calls_tool", g.transfers["tool"], "dialog", "transfers", state=ts)
    g_unclear = g.glob.get("unclear")
    for st in g.prompt_states:
        for lim in ([g_unclear["limit"]] if g_unclear else []) + ([g.glob["turn_limit"]["limit"]] if g.glob.get("turn_limit") else []):
            add(st, "limited_by", lim, "dialog", "global", state=st)
    for e in g.edges():
        add(e["src"], "leads_to", e["dst"], "trajectory", f"states.{e['src']}" if e["src"] in g.states else "transfers",
            qualifiers={"on": e["on"], "when": e["when"], "rules": list(e["rules"])}, state=e["src"])
    for reason, rid in (g.transfers.get("reasons") or {}).items():
        add(reason, "is_a", "escalation_trigger", "policy", "transfers")
        add(reason, "justified_by", rid, "policy", "transfers")
        add(g.transfers["tool"], "transfer_when", reason, "policy", "transfers", qualifiers={"rule": rid, "stance": "must"})
        for st in g.states:
            if reason in g.reasons_in(st):
                add(reason, "fires_in_state", st, "dialog", "transfers")
    for w in g.workflow():
        add(w["src"], "enables", w["dst"], "workflow", "workflow", qualifiers={"edge": w["wid"], "condition": w["condition"]})
    for h, r, tl in g.data.get("facts") or []:
        add(h, r, tl, "domain", "facts")
    for i, sent in enumerate(g.knowledge, start=1):
        add("knowledge_base", "states", sent, "knowledge", f"knowledge.{i}", gms=False)
    if g.opening:
        add("opening_statement", "has_text", g.opening, "policy", "opening", gms=False)
    return out


def rule_reification(g: PolicyGraph, rid: str) -> tuple[str, str, str]:
    """(subject, predicate, object) a rule states: from its check when it has one, else from the
    states that apply it."""
    r = g.rules[rid]
    c = r.get("check") or {}
    kind = next((k for k in c if k in CHECK_KINDS), None)
    if kind:
        obj = c[kind] if isinstance(c[kind], str) else (c.get("tool") or c.get("act") or g.root)
        return "bot", kind, obj
    states = [s for s, sp in g.states.items() if rid in sp.get("rules", [])]
    reasons = [k for k, v in (g.transfers.get("reasons") or {}).items() if v == rid]
    if reasons:
        return "bot", "transfers_on", reasons[0]
    if states:
        return "bot", "applies_in", states[0]
    return "bot", "follows", g.root


# ============================================================================ the graph as PolicyKG
def graph_kg(g: PolicyGraph):
    """A PolicyKG over the expanded triples, so every reader of the graph works unchanged."""
    from ..knowledge.policy_kg import PolicyKG

    class GraphKG(PolicyKG):
        engine = "graph"

        def validate(self) -> list[str]:
            return g.validate()

        def entity_types(self) -> dict[str, str]:
            out = {}
            for t in self.where("is_a"):
                out.setdefault(t["head"], t["tail"])
            for t in self.where("member_of"):
                out[t["head"]] = t["tail"]
            for t in self.where("in_stage"):
                out.setdefault(t["tail"], "stage")
            for t in self.where("has_slot"):
                out.setdefault(t["tail"], "slot")
            return out

        def consistency(self, store) -> dict[str, Any]:
            missing = [tr for tr in self.hrt(gms_only=True) if tr not in store.triples]
            enm = []
            for k in g.limits:
                got = store.lookup_enm("limit", k)
                enm.append({"key": k, "kg": g.limit_value(k), "store": got, "agree": float(got) == float(g.limit_value(k))})
            for attr, spec in g.numeric_attributes().items():
                for m, v in spec["members"].items():
                    got = store.lookup_enm(attr, m)
                    enm.append({"key": f"{attr}/{m}", "kg": v, "store": got, "agree": float(got) == float(v)})
            wf = {(w.wid, w.src, w.dst) for w in store.workflow}
            kgwf = {(t["qualifiers"]["edge"], t["head"], t["tail"]) for t in self.where("enables")}
            rules = set(store.rules) == set(g.rules)
            return {"missing_in_store": missing, "enm": enm, "enm_agree": all(e["agree"] for e in enm),
                    "workflow_agree": wf == kgwf, "rules_agree": rules,
                    "all_agree": not missing and all(e["agree"] for e in enm) and wf == kgwf and rules}

        def to_markdown(self) -> str:
            from .compile import render_policy_markdown
            return render_policy_markdown(g)

    data = {"schema": SCHEMA_V2, "title": g.title,
            "source": {"document": (g.data.get("source") or {}).get("document", g.path.name if g.path else g.id),
                       "sections": {}},
            "relations": RELATIONS, "triples": to_triples(g), "sha256": g.sha256}
    kg = GraphKG(data, g.path)
    kg.graph = g
    return kg


RELATIONS = {
    "is_a": "entity -> class", "has_value": "limit -> exact value", "has_unit": "limit -> unit",
    "has_text": "entity -> the policy's wording", "member_of": "value -> value set", "has_alias": "value -> customer synonym",
    "has_slot": "act -> slot (qualifiers: values_from, required)", "has_stage": "tool -> journey stage",
    "has_role": "tool -> auth / read / write / handoff", "success_value": "tool -> success status",
    "failure_value": "tool -> failure status", "has_stance": "rule -> must / must_not / may", "constrains": "rule -> subject",
    "in_stage": "dialog state -> stage", "prompts_customer": "dialog state -> yes/no",
    "accepts_act": "dialog state -> admissible customer act", "interprets_yes_as": "dialog state -> meaning of a bare yes",
    "interprets_no_as": "dialog state -> meaning of a bare no", "applies_rule": "dialog state -> rule",
    "calls_tool": "dialog state -> tool the bot may call next", "limited_by": "dialog state -> limit",
    "leads_to": "dialog state -> next state (qualifiers: on, when, rules)", "justified_by": "transfer reason -> rule",
    "fires_in_state": "transfer reason -> state", "transfer_when": "handoff tool -> reason",
    "enables": "tool -> next tool (qualifiers: edge, condition)", "states": "knowledge base -> sentence",
}


# ============================================================================ derived config
def derive_config(g: PolicyGraph, graph_file: str) -> dict[str, Any]:
    """The domain sections a campaign reads, mined from the graph. domain.yaml is laid on top."""
    optional_sets = {spec["set"] for a in (g.data.get("acts") or {}) for spec in g.slot_specs(a).values()
                     if spec.get("optional")}
    required_sets = {spec["set"] for a in (g.data.get("acts") or {}) for spec in g.slot_specs(a).values()
                     if not spec.get("optional")}
    tools = {}
    for t, sp in g.tools.items():
        # an argument filled from a slot the customer may leave out is optional (None) in the backend contract
        args = {a: (None if vs in optional_sets and vs not in required_sets else "") for a, vs in (sp.get("args") or {}).items()}
        statuses = [sp["ok"], *sp.get("fail", [])]
        spec = {"role": sp["role"], "behaviour": "script", "args": args, "key_args": list(args),
                "results_field": f"tool:{t}", "default_results": [{"status": sp["ok"]}],
                "arg_values": {a: vs for a, vs in (sp.get("args") or {}).items() if vs in g.values},
                "status": {"ok": sp["ok"]}, "statuses": statuses, "unavailable": UNAVAILABLE,
                "stage": sp.get("stage"),
                "api": {"args": {a: f"{vs.replace('_', ' ')} value" for a, vs in (sp.get("args") or {}).items()},
                        "returns": json.dumps({"status": " | ".join(statuses), **{k: v for k, v in (sp.get("returns") or {}).items()}})
                        + "; raises an error when the service is unavailable",
                        "description": sp.get("text", "")}}
        lists = [k for k, v in (sp.get("returns") or {}).items() if isinstance(v, dict)]
        if lists:
            spec["key"] = lists[0]
        if sp.get("echo"):
            spec["echo"] = sp["echo"]
            spec["plausible_wrong"] = {"flip": {"field": sp["echo"], "from_arg": sp["echo"],
                                                "map": _flip_map(g, (sp.get("args") or {}).get(sp["echo"])),
                                                "default": None}}
        if sp.get("check"):
            spec["check"] = {"arg": sp["check"]["arg"], "fixture": sp["check"]["fixture"], "success": sp["ok"],
                             "else": (sp.get("fail") or ["FAILED"])[0]}
        if sp["role"] == "auth":
            spec["plausible_wrong"] = {"set": {"status": sp["ok"]}}
        tools[t] = spec

    num = g.numeric_attributes()
    figures = {"limits": {"class": "limit", "enm_category": "limit", "kx": {"type": "limits", "key": "{key}"},
                          "attribute": "limit", "attribute_label": "has_value"},
               "relations": {attr_relation(a): {"enm_category": a, "unit": _unit(a), "kind": _kind(a), "attribute": a,
                                                "kx": {"type": spec["set"], "key": "{key}/" + a}}
                             for a, spec in num.items()},
               "named": {}}

    rules_calls, speech = derive_rule_checks(g)
    from .trace import scored_stages
    stages = scored_stages(g)
    cfg = {
        "domain": {"id": g.id, "title": g.title, "graph": graph_file, "root": g.root, "engine": "graph",
                   "channel": "voice", "language": "en"},
        "tools": tools,
        "figures": figures,
        "fixtures": {"credential": "7316"},
        "rules": {"calls": rules_calls, "completion": [], "speech": speech,
                  "tracking": {"handoff_request_act": (g.glob.get("human") or {}).get("act", "REQUEST_HUMAN")}},
        "flow": {"engine": "graph", "stages": stages, "stage_names": {s: s for s in stages},
                 "workflow": {"start": "start", "summary": "end", "end": "end"}},
    }
    return cfg


def derive_rule_checks(g: PolicyGraph) -> tuple[list[dict], list[dict]]:
    """Runtime call checks (governance/rules.py templates) and speech checks from the graph's
    machine-checkable rules."""
    calls, speech = [], []
    scope: dict[tuple, dict] = {}
    for rid, kind, c in g.rule_checks():
        if kind == "item_window":
            s = scope.setdefault((c["item_window"], c["arg"]), {"check": "item_scope", "tool": c["item_window"],
                                                                "arg": c["arg"], "rule": rid})
            s.update(source=c["source"], window={"field": c["age"], "limit": c["limit"], "rule": rid})
        elif kind == "max_distinct":
            s = scope.setdefault((c["max_distinct"], c["arg"]), {"check": "item_scope", "tool": c["max_distinct"],
                                                                 "arg": c["arg"], "rule": rid})
            s["cap"] = {"limit": c["limit"], "rule": rid, "noun": c["arg"].replace("_id", "")}
    calls += [s for s in scope.values() if "source" in s]
    for rid, kind, c in g.rule_checks():
        if kind == "no_call_after_handoff":
            calls.append({"check": "no_call_after_handoff", "rule": rid})
        elif kind == "requires_auth":
            calls.append({"check": "requires_auth", "rule": rid})
        elif kind == "max_calls":
            calls.append({"check": "max_calls", "tool": c["max_calls"], "limit": c["limit"], "rule": rid})
        elif kind == "handoff_when_requested":
            calls.append({"check": "handoff_when_requested", "rule": rid})
        elif kind == "turn_limit":
            calls.append({"check": "turn_limit", "limit": c["turn_limit"], "rule": rid})
        elif kind in ("requires_status", "value_allowed", "requires_slot", "confirm_before", "disclose_before"):
            calls.append({"check": "graph_" + kind, "rule": rid, **{k: v for k, v in c.items()}})
        elif kind == "offered_only":
            calls.append({"check": "offered_only", "tool": c["offered_only"], "arg": c["arg"], "source": c["source"],
                          "key": c["field"], "rule": rid})
        elif kind == "terminal_value":
            calls.append({"check": "terminal_value", "tool": c["terminal_value"], "arg": c["arg"], "value": c["value"],
                          "rule": rid})
        elif kind == "value_from_caller":
            calls.append({"check": "value_from_caller", "tool": c["value_from_caller"], "arg": c["arg"], "act": c["act"],
                          "slot": c["slot"], "rule_missing": rid, "rule_mismatch": rid})
        elif kind == "forbidden_stance":
            speech.append({"check": "forbidden_stance", "pattern": c["pattern"], "stance": c["forbidden_stance"],
                           "message": "asserts a stance the policy contradicts", "rule": rid})
        elif kind == "claim_needs_success":
            t = c["tool"]
            speech.append({"check": "claim_after_success", "act": c["claim_needs_success"], "tool": t,
                           "echo": g.tools[t].get("echo"), "pattern": r"\b(submitted|has been sent|is on its way|application is in)\b",
                           "message": "claims the application was submitted without a matching success", "rule": rid})
    return calls, speech


def _flip_map(g: PolicyGraph, vs: str | None) -> dict[str, str]:
    m = g.members(vs) if vs else []
    return {a: b for a, b in zip(m, m[1:] + m[:1])} if len(m) > 1 else {}


def _unit(attr: str) -> str:
    if attr.endswith("_usd"):
        return "usd"
    if "days" in attr:
        return "business_days" if "business" in attr else "days"
    return ""


def _kind(attr: str) -> str:
    if attr.endswith("_usd"):
        return "money"
    if "days" in attr:
        return "days"
    return "count"


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def dump_compact(data: dict[str, Any]) -> str:
    """A compact policy graph in its reading layout: one line per limit, value, act, tool, rule and state."""
    j = lambda x: json.dumps(x, ensure_ascii=False)
    out = ["{"]
    keys = list(data)
    for i, k in enumerate(keys):
        v = data[k]
        end = "," if i < len(keys) - 1 else ""
        if k == "values" and isinstance(v, dict):
            out.append(f' {j(k)}: {{')
            sets = list(v)
            for si, vs in enumerate(sets):
                out.append(f'  {j(vs)}: {{')
                mem = list(v[vs])
                out += [f'   {j(m)}: {j(v[vs][m])}' + ("," if mi < len(mem) - 1 else "") for mi, m in enumerate(mem)]
                out.append("  }" + ("," if si < len(sets) - 1 else ""))
            out.append(" }" + end)
        elif isinstance(v, dict) and k in ("limits", "acts", "tools", "rules", "states", "global", "transfers"):
            out.append(f' {j(k)}: {{')
            ks = list(v)
            out += [f'  {j(x)}: {j(v[x])}' + ("," if xi < len(ks) - 1 else "") for xi, x in enumerate(ks)]
            out.append(" }" + end)
        elif isinstance(v, list) and k in ("knowledge", "facts"):
            out.append(f' {j(k)}: [')
            out += [f'  {j(x)}' + ("," if xi < len(v) - 1 else "") for xi, x in enumerate(v)]
            out.append(" ]" + end)
        else:
            out.append(f' {j(k)}: {j(v)}{end}')
    out.append("}")
    return "\n".join(out) + "\n"
