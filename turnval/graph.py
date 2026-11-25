"""Block 1 - the graph layer: the AOP flow (what the bot must do each turn) and the HRT graph (what it must say).

Inputs (turnval/data/, kept out of git):
  aop_flow.json   sections[] -> nodes (START / USE / IF / WAIT UNTIL / INSTRUCTION|SAY (bot speech) / CUSTOMER_INPUT /
                  GOTO / SWITCH / END ...) + edges (flow / condition / customer / wait / resume)
  hrt_graph.json  aops[] -> global_policies[] + sections[] -> golden_boxes[] -> branches[] -> MUST_CONVEY /
                  MUST_NOT_CONVEY / METADATA_CHECK / LOCAL_GUARDRAIL groups of atomic HRTs

Turn model (checked against a real call): every customer turn re-enters the AOP from its first section (User
Identification: Routing, which follows metadata.checkpoint; Charge Verification: Fraud Charges Routing). The walker
follows tools and IF branches on their outputs until it reaches bot speech - collecting every speaking box up to the
point where the customer can answer (one assembled response, e.g. the transaction readout) - or a hand-off (SWITCH,
escalation, end_call). Speech that was already given and answered (its WAIT UNTIL now holds) is skipped.

``resolve`` turns the speaking boxes of a turn into the HRTs that apply: each box's applicable branches (applies_when
evaluated against the turn's context) plus the AOP's global policies whose condition holds.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

DATA = Path(__file__).parent / "data"
FIRST_SECTION = {"User Identification": "Routing", "Charge Verification": "Fraud Charges Routing"}
AOP_KEYS = {"User Identification": "UI", "Charge Verification": "CV"}


# ----------------------------------------------------------------------------------------------------------- loading
def load_json(path: str | Path) -> Any:
    """Load JSON, repairing the two exporter faults seen in hrt_graph.json: a condition split into two strings that
    lost a closing quote (``== \\"X\\","@y`` ) and a value whose opening quote was dropped (``== \\X\\""``)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r'== \\([A-Za-z_][A-Za-z_\-]*)\\""', r'== \\"\1\\""', text)
        text = re.sub(r'\\",("@|"NOT )', r'\\"",\1', text)
        return json.loads(text)


def clause(box_or_node_id: str) -> str:
    """'graph:24/UI_CLAUSE_108' -> 'UI_CLAUSE_108'."""
    return box_or_node_id.split("/")[-1]


# ----------------------------------------------------------------------------------------------------------- the flow
class Flow:
    def __init__(self, path: str | Path = DATA / "aop_flow.json"):
        self.sections: dict[tuple[str, str], tuple[dict, dict]] = {}
        self.order: dict[str, list[str]] = {}
        for s in load_json(path):
            nodes = {n["id"]: n for n in s["nodes"]}
            out: dict[str, list] = {}
            for e in s["edges"]:
                out.setdefault(e["from"], []).append(e)
            self.sections[(s["aop"], s["section"])] = (nodes, out)
            self.order.setdefault(s["aop"], []).append(s["section"])

    def speaking_nodes(self, aop: str | None = None) -> list[tuple[str, str, dict]]:
        return [(a, s, n) for (a, s), (nodes, _) in self.sections.items() if aop in (None, a)
                for n in nodes.values() if n.get("bot_speech") and n["op"] in SPEECH_OPS]

    def decisions(self, aop: str | None = None) -> dict[tuple[str, str], set[str]]:
        """Every (tool, field) the flow branches on, with the values it compares against."""
        out: dict[tuple[str, str], set[str]] = {}
        for (a, _s), (nodes, _) in self.sections.items():
            if aop not in (None, a):
                continue
            for n in nodes.values():
                if n["op"] not in ("IF", "WAIT UNTIL"):
                    continue
                for m in re.finditer(r'@([A-Za-z_0-9]+)(?:\.([A-Za-z_0-9]+))?\s*(?:==|!=)\s*\\?"([^"\\]+)', n["text"]):
                    out.setdefault((m.group(1), m.group(2) or ""), set()).add(m.group(3))
        return out


class TurnInput(Protocol):
    """What one customer turn supplies to the walker."""
    def tool(self, name: str, fld: str | None) -> Any: ...       # a tool output (lazy: the scenario decides it)
    def meta(self, key: str) -> Any: ...                        # a metadata override for this turn (or None)
    def semantic(self, condition: str) -> bool | None: ...      # a customer-driven condition (None: does not hold)


@dataclass
class DictTurn:
    """A turn given as plain dicts (tests, replays of real logs)."""
    tools: dict[str, Any] = field(default_factory=dict)
    semantics: dict[str, bool] = field(default_factory=dict)
    metas: dict[str, Any] = field(default_factory=dict)

    def tool(self, name, fld):
        v = self.tools.get(name, {})
        return v if fld is None else (v.get(fld) if isinstance(v, dict) else None)

    def meta(self, key):
        return self.metas.get(key)

    def semantic(self, condition):
        return self.semantics.get(condition)


@dataclass
class Step:
    """What the bot must do on a turn."""
    aop: str
    sections: list[str]
    speech: list[str]               # speaking clause ids, in order (one assembled response); [] for a silent hand-off
    outcome: str                    # SPEAK | SPEAK then <hand-off> | <hand-off> | WAIT
    path: list[str]                 # the nodes walked ("~" marks speech skipped as already given)
    tools: list[str]                # tools used on the way
    handoff: str | None = None      # SWITCH / escalation / end_call text when the turn leaves the AOP

    @property
    def state(self) -> str:
        """The turn's state id: its speaking boxes joined (a complex state is one id), else the hand-off."""
        return "+".join(self.speech) if self.speech else f"HANDOFF:{self.handoff or self.outcome}"


def _expr(cond: str) -> str:
    c = cond.replace("\\", "")
    c = re.sub(r"\bNOT\s*\(", "not (", c)
    c = re.sub(r"@([A-Za-z_0-9]+)\.([A-Za-z_0-9]+)", r'T("\1","\2")', c)
    c = re.sub(r"@([A-Za-z_0-9]+)", r'T("\1",None)', c)
    c = re.sub(r"metadata\.([A-Za-z_0-9.]+)", r'M("\1")', c)
    c = re.sub(r"\btrue\b", "True", c)
    c = re.sub(r"\bfalse\b", "False", c)
    return c


SPEECH_OPS = ("INSTRUCTION", "SAY", "LOOP_SAY")
_HANDOFF = re.compile(r"^(SWITCH|USE (escalate|end_call|language_switch))")


class Walker:
    """One AOP's per-turn walk. ``metadata`` persists across turns (set_metadata nodes write to it)."""

    def __init__(self, flow: Flow, aop: str, metadata: dict[str, Any] | None = None):
        self.flow, self.aop = flow, aop
        self.md: dict[str, Any] = metadata if metadata is not None else {}
        self.unresolved: set[str] = set()

    # -- conditions
    def holds(self, cond: str, turn: TurnInput) -> bool:
        sem = turn.semantic(cond)
        if sem is not None:
            return bool(sem)
        def T(tool, fld):
            return turn.tool(tool, fld)
        def M(k):
            v = turn.meta(k)
            return self.md.get(k) if v is None else v
        try:
            return bool(eval(_expr(cond), {"__builtins__": {}}, {"T": T, "M": M, "None": None, "True": True,
                                                                  "False": False}))
        except SyntaxError:                  # natural language the turn did not assert: it does not hold
            self.unresolved.add(cond)
            return False

    def _step(self, nodes, out, cur, turn):
        n, edges = nodes[cur], out.get(cur, [])
        if n["op"] in ("IF", "WAIT UNTIL"):
            ce = [e for e in edges if e["kind"] == "condition"]
            pos = [e for e in ce if not e["condition"].startswith("NOT (")]
            neg = [e for e in ce if e["condition"].startswith("NOT (")]
            if pos and self.holds(pos[0]["condition"], turn):
                return pos[0]["to"]
            return neg[0]["to"] if neg else None
        nxt = [e for e in edges if e["kind"] in ("flow", "condition")] or \
              [e for e in edges if e["kind"] in ("resume", "handoff")]          # loop-overlay gates: the source path
        return nxt[0]["to"] if nxt else None

    def _set(self, n) -> bool:
        t = re.match(r"USE set_(is_[A-Za-z0-9]+_recollected)_true\s*$", n["text"])
        if t:                                            # set_is_4ssn_recollected_true and its kin
            self.md[t.group(1)] = True
            return True
        m = re.match(r'USE set_metadata\("([^"]+)",\s*"([^"]*)"\)', n["text"])
        if m:
            v: Any = m.group(2)
            self.md[m.group(1)] = {"True": True, "False": False}.get(v, v)
            return True
        return False

    def _block(self, nodes, out, cur, turn):
        """From a speaking node: every speaking node up to where the customer can answer."""
        speech, path, tools = [], [], []
        for _ in range(300):
            n = nodes[cur]
            path.append(cur)
            if n.get("bot_speech") and n["op"] in SPEECH_OPS:
                speech.append(n)
            cust = [e for e in out.get(cur, []) if e["kind"] == "customer"]
            if cust:
                after = [e2["to"] for e2 in out.get(cust[0]["to"], [])]
                return speech, path, tools, (after[0] if after else None), cur
            if n.get("terminal") or n["op"] in ("GOTO", "SWITCH", "END", "FINISH"):
                return speech, path, tools, None, cur
            if n["op"] == "USE" and not self._set(n):
                tools.append(n["text"][4:])
            nxt = self._step(nodes, out, cur, turn)
            if nxt is None:
                return speech, path, tools, None, cur
            cur = nxt
        raise RuntimeError("speech block does not end")

    def run(self, turn: TurnInput, section: str | None = None, _depth: int = 0) -> Step:
        section = section or FIRST_SECTION[self.aop]
        nodes, out = self.flow.sections[(self.aop, section)]
        cur, path, tools = "START", [], []
        for _ in range(800):
            n = nodes[cur]
            if n.get("bot_speech") and n["op"] in SPEECH_OPS:
                speech, bpath, btools, after, stop = self._block(nodes, out, cur, turn)
                w = nodes.get(after) if after else None
                if w is not None and w["op"] == "WAIT UNTIL":
                    pos = [e for e in out.get(w["id"], []) if e["kind"] == "condition"]
                    if pos and self.holds(pos[0]["condition"], turn):        # asked before and answered now
                        path += [f"{section}/{x}~" for x in bpath]
                        cur = pos[0]["to"]
                        continue
                path += [f"{section}/{x}" for x in bpath]
                tools += btools
                last = nodes[stop]
                if last["op"] == "GOTO" and not speech:
                    continue
                hand = last["text"] if (_HANDOFF.match(last["text"]) or last["op"] == "SWITCH") else None
                return Step(self.aop, [section], [x["id"] for x in speech],
                            "SPEAK" + (f" then {hand}" if hand else ""), path, tools, hand)
            path.append(f"{section}/{cur}")
            if n["op"] == "USE":
                if not self._set(n):
                    tools.append(n["text"][4:])
                if n.get("terminal"):
                    return Step(self.aop, [section], [], n["text"], path, tools, n["text"])
            if n["op"] == "GOTO":
                if _depth > 12:
                    raise RuntimeError("GOTO loop")
                st = self.run(turn, n["destination"]["section"], _depth + 1)
                return Step(st.aop, [section] + st.sections, st.speech, st.outcome, path + st.path,
                            tools + st.tools, st.handoff)
            if n["op"] in ("SWITCH", "END", "FINISH", "REVIEW"):
                hand = n["text"] if n["op"] == "SWITCH" else None
                return Step(self.aop, [section], [], n["text"], path, tools, hand)
            nxt = self._step(nodes, out, cur, turn)
            if nxt is None:
                return Step(self.aop, [section], [], "WAIT" if n["op"] == "WAIT UNTIL" else n["op"], path, tools)
            cur = nxt
        raise RuntimeError("walk does not end")


# ------------------------------------------------------------------------------------------------------- HRT graph
@dataclass
class Atom:
    """One HRT the response is judged on."""
    ref: str                        # hrt_ref (trace id)
    box: str                        # clause id, or the global policy id
    branch: str
    kind: str                       # convey | forbid | metadata | guard_must | guard_must_not | global_must | global_must_not
    head: str
    head_meaning: str
    tail: str
    description: str
    wording: str = "meaning"
    field: str | None = None
    check: str | None = None
    group: str = "AND"              # AND | OR | ANY: how this atom combines with its siblings
    group_id: str = ""
    in_order: bool = False
    response_conditional: bool = False   # applies only if the response does the thing (e.g. "references the account number")

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "", False)}


@dataclass
class Context:
    """What the branch conditions are judged against on a turn."""
    metadata: dict[str, Any] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)               # customer/conversation facts (see CONDITION_RULES)
    txn: dict[str, Any] = field(default_factory=dict)          # current transaction
    card: dict[str, Any] = field(default_factory=dict)         # current card
    txn_index: int = 1
    card_txn_count: int = 1
    txn_day: str = "older"                                     # today | yesterday | older


RESPONSE_CONDITIONAL = "RESPONSE_CONDITIONAL"
# Natural-language branch conditions -> how the context decides them (first match wins; order matters).
CONDITION_RULES: list[tuple[str, Callable[[Context], Any]]] = [
    (r"^always$", lambda c: True),
    (r"transactionMode is present", lambda c: bool(c.txn.get("transactionMode"))),
    (r"merchantAddress is present", lambda c: bool(c.txn.get("merchantAddress"))),
    (r"merchantAddress is absent", lambda c: not c.txn.get("merchantAddress")),
    (r'transactionStatus == "approved"', lambda c: str(c.txn.get("transactionStatus", "")).lower() == "approved"),
    (r"latest message is a request such as", lambda c: "next_txn_request" in c.flags),
    (r"equals the date returned by @get_today_date", lambda c: c.txn_day == "today"),
    (r"day before today", lambda c: c.txn_day == "yesterday"),
    (r"response references the account number|response addresses the user by name", lambda c: RESPONSE_CONDITIONAL),
    (r"current_transaction_index == 1 and the card has one", lambda c: c.txn_index == 1 and c.card_txn_count == 1),
    (r"current_transaction_index == 1 and the card has more than one",
     lambda c: c.txn_index == 1 and c.card_txn_count > 1),
    (r"current_transaction_index > 1", lambda c: c.txn_index > 1),
    (r'role == "BASIC" and isBasic == true', lambda c: c.card.get("role") == "BASIC" and c.card.get("isBasic") is True),
    (r'role != "BASIC" and isBasic == false', lambda c: c.card.get("role") != "BASIC" and c.card.get("isBasic") is False),
    (r'role == "BASIC" and isBasic == false', lambda c: c.card.get("role") == "BASIC" and c.card.get("isBasic") is False),
    (r"has not already been thanked", lambda c: "already_thanked" not in c.flags),
    (r"already thanked", lambda c: "already_thanked" in c.flags),
    (r"doesn't have their card on hand$", lambda c: "no_card_on_hand" in c.flags),
    (r"doesn't know the last 5 digits", lambda c: bool({"no_card_on_hand", "doesnt_know_digits"} & c.flags)),
    (r"asks a clarifying question", lambda c: "clarifying_question" in c.flags),
    (r"still looking for the information", lambda c: "hold_on" in c.flags),
    (r"unrelated to the verification question", lambda c: bool({"off_topic", "unrelated"} & c.flags)),
    (r"asks for a human|speak to a human", lambda c: "asks_human" in c.flags),
    (r"asks for Spanish|in Spanish", lambda c: "spanish" in c.flags),
    (r"inquiry unrelated to identification", lambda c: "off_topic" in c.flags),
    (r"new unrelated concern|not interested in continuing|do not wish to continue",
     lambda c: bool({"new_concern", "unwilling"} & c.flags)),
    (r"No grounded information", lambda c: "ungrounded_question" in c.flags),
    (r"asks a question during the journey", lambda c: bool({"asks_question", "ungrounded_question"} & c.flags)),
    (r"procedural question", lambda c: "procedural_question" in c.flags),
    (r"asks a question OR restates", lambda c: bool({"asks_question", "restates_reason"} & c.flags)),
    (r"Identification and verification are complete", lambda c: "idv_complete" in c.flags),
    (r"Travel Disambiguation", lambda c: False),
    (r"offers an alternative identification", lambda c: "alternative_id" in c.flags),
    (r"Previous identifier attempt failed", lambda c: "identifier_switch_after_fail" in c.flags),
    (r"After a successful match", lambda c: "identifier_switch_after_fail" not in c.flags),
    (r"first request for identifying information", lambda c: "first_id_request" in c.flags),
    (r"later request for identifying information", lambda c: "later_id_request" in c.flags),
]


def decide(condition: str, ctx: Context) -> Any:
    for rx, fn in CONDITION_RULES:
        if re.search(rx, condition, re.I):
            return fn(ctx)
    # a metadata expression
    try:
        return bool(eval(_expr(condition), {"__builtins__": {}},
                         {"T": lambda t, f: None, "M": lambda k: ctx.metadata.get(k), "None": None,
                          "True": True, "False": False}))
    except Exception:
        return None                          # unknown: reported, treated as not applying


class HRTGraph:
    def __init__(self, path: str | Path = DATA / "hrt_graph.json"):
        g = load_json(path)
        self.raw = g
        self.boxes: dict[str, dict] = {}
        self.globals: dict[str, list[dict]] = {}
        self.complex: dict[str, dict] = {}
        for a in g["aops"]:
            self.globals[a["aop"]] = a.get("global_policies", [])
            for s in a["sections"]:
                for cs in s.get("complex_states", []):
                    self.complex[cs["complex_state_id"]] = cs
                for b in s["golden_boxes"]:
                    self.boxes.setdefault(clause(b["box_id"]), b)      # source boxes first; loop overlays repeat them
        self.unknown_conditions: set[str] = set()

    # -- atoms
    def _atoms(self, grp: Any, box: str, branch: str, kind: str, out: list[Atom], gid: str, op: str = "AND",
               in_order: bool = False) -> None:
        if not isinstance(grp, dict) or not grp:
            return
        in_order = in_order or bool(grp.get("in_order"))
        for k in ("AND", "OR", "ANY"):
            for i, x in enumerate(grp.get(k, [])):
                if any(q in x for q in ("AND", "OR", "ANY")):
                    self._atoms(x, box, branch, kind, out, f"{gid}.{k}{i}", k, in_order)
                    continue
                pol = x.get("polarity")
                kk = kind
                if kind == "global":
                    kk = {"MUST": "global_must", "MUST_NOT": "global_must_not", "METADATA": "metadata"}.get(pol, "global_must")
                out.append(Atom(x.get("hrt_ref", ""), box, branch, kk, x.get("head", ""), x.get("head_meaning", ""),
                                x.get("tail", ""), x.get("description", ""), x.get("wording", "meaning"),
                                x.get("field"), x.get("check"), k if k != "AND" else op, f"{gid}.{k}", in_order))

    def _branch_atoms(self, br: dict, box: str, is_global: bool) -> list[Atom]:
        out: list[Atom] = []
        b = br["branch"]
        if is_global:
            for key in ("MUST_CONVEY", "MUST_NOT_CONVEY", "METADATA_CHECK"):
                self._atoms(br.get(key, {}), box, b, "global", out, key)
            return out
        self._atoms(br.get("MUST_CONVEY", {}), box, b, "convey", out, "C")
        self._atoms(br.get("MUST_NOT_CONVEY", {}), box, b, "forbid", out, "N")
        self._atoms(br.get("METADATA_CHECK", {}), box, b, "metadata", out, "M")
        lg = br.get("LOCAL_GUARDRAIL", {}) or {}
        self._atoms(lg.get("MUST", {}), box, b, "guard_must", out, "GM")
        self._atoms(lg.get("MUST_NOT", {}), box, b, "guard_must_not", out, "GN")
        return out

    def _applies(self, aw: Any, ctx: Context) -> Any:
        cond = aw if isinstance(aw, str) else aw.get("condition", "")
        r = decide(cond, ctx)
        if r is None:
            self.unknown_conditions.add(cond)
            return False
        return r

    def resolve(self, aop: str, speech: list[str], ctx: Context) -> dict[str, Any]:
        """The HRTs that apply to a turn whose correct response is made of the boxes ``speech``."""
        atoms: list[Atom] = []
        branches: list[str] = []
        for cid in speech:
            box = self.boxes.get(cid)
            if not box:
                continue
            for br in box["branches"]:
                a = self._applies(br["applies_when"], ctx)
                if a is False:
                    continue
                branches.append(f"{cid}/{br['branch']}")
                got = self._branch_atoms(br, cid, False)
                if a == RESPONSE_CONDITIONAL:
                    for x in got:
                        x.response_conditional = True
                atoms += got
        policies: list[str] = []
        for gp in self.globals.get(aop, []):
            only = gp.get("only_when") or []
            if any(decide(c, ctx) is False for c in only):
                continue
            for br in gp.get("branches", []):
                a = self._applies(br["applies_when"], ctx)
                if a is False:
                    continue
                policies.append(f"{gp['policy_id'].split(':')[-1]}/{br['branch']}")
                got = self._branch_atoms(br, gp["policy_id"].split(":")[-1], True)
                if a == RESPONSE_CONDITIONAL:
                    for x in got:
                        x.response_conditional = True
                atoms += got
        return {"boxes": list(speech), "branches": branches, "policies": policies, "atoms": atoms}

    def box_text(self, cid: str) -> str:
        b = self.boxes.get(cid)
        return b["original_text"] if b else ""

    def box_type(self, cid: str) -> str:
        b = self.boxes.get(cid)
        return b["box_type"] if b else ""
