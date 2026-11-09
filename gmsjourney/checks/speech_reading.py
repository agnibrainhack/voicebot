"""Reading what the bot said: the contract, the session's facts and the utterance reader.

A bot labels its turns (``CONFIRM_RECORDED(T1, UNRECOGNIZED)``) and speaks words. The labels were always graded;
this reads the words on their own, like a second listener on the line.

Parts of this file:
  1. ``SpeechContract``: what each act must say and where each spoken value's truth lives, from the graph.
  2. ``SessionFacts``: what is true in this conversation so far (returned items, true writes, offered options,
     what the customer said).
  3. ``UtteranceReader``: sentence -> claims (field, value, about what, mode: statement, recorded, completed,
     offered), without looking at the labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any


# =====================================================================================================================
# PART 1 - The speech contract: everything the validator needs from the policy graph, in one object.
#
# Input   a ``PolicyKG`` (graph/triples.py), usually ``store.kg``.
# Output  a ``SpeechContract``: per bot act what it says and where each value's truth lives, the
#         words that mark an act, how the bot words each value, the rule stances it can be heard
#         asserting or denying, the fields a returned transaction has, the opening statement and
#         the dialog-state transitions.
#
# Nothing here is specific to a bot; it is read from the graph's "speech" triples (written in
# graph/triples.py: BOT_SPEECH, TOOL_RETURNS, BOT_PHRASES, STANCE_PHRASES) and its existing
# vocabulary and state layers.
#
#     contract = SpeechContract.from_graph(store.kg)
#     contract.acts["PRESENT_TXN"].required          # ('merchant', 'amount', 'date')
#     contract.acts["PRESENT_TXN"].source_of("date")  # 'item.date'
#     contract.phrases["UNRECOGNIZED"]               # ['not recognized', "isn't yours", ..., 'not mine']
# =====================================================================================================================

from ..packs.names import N, graph_names


def also_allowed() -> dict[str, set[str]]:
    """Customer act (or "any") -> bot acts the speech-flow check accepts after it in any prompt state, beyond what
    the graph's transitions allow (project config: speech.also_allowed)."""
    from ..packs.project import project
    return {k: set(v) for k, v in ((project().raw.get("speech") or {}).get("also_allowed") or {}).items()}


@dataclass(frozen=True)
class FieldSpec:
    """One piece of information an act tells the customer.

    field     its name ("merchant", "fee", "recorded", ...)
    source    where its true value comes from: item.<f>, write.<tool>.<arg>, writes.<tool>,
              caller.<slot>, policy.<category>, policy.limit:<key> or text.opening
    required  the act must say it whenever there is something to say
    """
    field: str
    source: str
    required: bool = False


@dataclass(frozen=True)
class ActContract:
    """What one bot act (dialog state) says and the words that mark it."""
    act: str
    says: tuple[FieldSpec, ...] = ()
    cues: tuple[str, ...] = ()
    source: str | None = None          # the PDF section that says what the act must say (e.g. pdf.2B)

    @property
    def fields(self) -> tuple[str, ...]:
        """Every field the act may say, in the graph's order: ("merchant", "amount", "date")."""
        return tuple(f.field for f in self.says)

    @property
    def required(self) -> tuple[str, ...]:
        """The fields the act must say (a missing one is a ``missing`` finding)."""
        return tuple(f.field for f in self.says if f.required)

    def source_of(self, field_name: str) -> str | None:
        """Where the truth of ``field_name`` lives for this act ("item.date"), or None if the act does not say it."""
        return next((f.source for f in self.says if f.field == field_name), None)


@dataclass(frozen=True)
class StanceSpec:
    """A rule whose fact the bot can be heard asserting or denying (e.g. R27, 'not proof of fraud').

    fact      (subject, predicate, object) the rule states; stance its stance (must / must_not)
    asserted  words that assert the fact; denied: words that deny it (read first)
    """
    rule: str
    fact: tuple[str, str, str]
    stance: str
    asserted: tuple[str, ...]
    denied: tuple[str, ...]


@dataclass
class SpeechContract:
    """Everything the reader and the verifier need to know about speech, read once from the policy graph.

    Build it with ``SpeechContract.from_graph(kg)``; it is read-only afterwards. The reader uses ``phrases``,
    ``limit_phrases``, ``stances`` and the cues in ``acts`` to understand a sentence; the verifier uses the
    sources in ``acts``, ``item_fields``, ``success_values`` and ``transitions`` to grade it."""
    acts: dict[str, ActContract]
    phrases: dict[str, list[str]]                 # value -> how it is worded (bot phrasings + customer synonyms)
    value_sets: dict[str, list[str]]              # disposition / card_action / delivery -> values
    stances: dict[str, StanceSpec]
    item_fields: dict[str, str]                   # returned transaction field -> type (money, date, days, ...)
    opening: str
    limits: list[str]
    limit_phrases: dict[str, list[str]]           # limit -> how it is worded around its number ("up to {n} charges")
    transitions: list[tuple[str, str, str]]       # (from state, to state, on: customer act | auto | any)
    prompt_states: set[str]
    success_values: dict[str, str] = field(default_factory=dict)     # tool -> the status that means it worked
    global_acts: dict[str, set[str]] = field(default_factory=dict)   # customer act -> states it may lead to anywhere

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_graph(cls, kg) -> "SpeechContract":
        """Read the contract from a PolicyKG. Raises if the graph has no speech layer."""
        if not kg.where("says"):
            raise ValueError("the policy graph has no speech layer (build it with graph/triples.py)")
        acts = {}
        for st in kg.states:
            says = tuple(FieldSpec(r["field"], r["from"], bool(r.get("required"))) for r in kg.says(st))
            src = next((t["source"] for t in kg.where("says", head=st)), None)
            acts[st] = ActContract(st, says, tuple(kg.spoken_cues(st)), src)
        g = getattr(kg, "graph", None)              # a pack compiled from a compact graph: generic fields
        if g is None:
            value_sets = {vs: kg.value_set(vs) for vs in (N.value_set.disposition, N.value_set.card_action,
                                                          N.value_set.delivery)}
        else:
            from .speech_checks import PackModel
            from ..packs.loading import current
            model = PackModel(g, current().section("lexicon"))
            value_sets = {vs: tuple(g.members(vs)) for vs in model.value_sets}
        phrases = {v: kg.spoken_as(v) for vals in value_sets.values() for v in vals}
        limits = [t["head"] for t in kg.where("is_a", tail="limit")]
        limit_phrases = {k: [p for p in kg.tails(k, "spoken_as") if "{n}" in p] for k in limits}
        stances = {}
        for t in kg.where("asserted_by"):
            rid = t["head"]
            if rid in stances:
                continue
            fact = kg.rule_fact(rid)
            stance = (kg.tails(rid, "has_stance") or ["must"])[0]
            asserted, denied = kg.stance_phrases(rid)
            if fact:
                stances[rid] = StanceSpec(rid, fact, stance, tuple(asserted), tuple(denied))
        dialog = set(kg.states)
        transitions = []
        for t in kg.transitions():
            # a compiled transition may say acts on the way (a policy answer, a clarification, then the prompt again):
            # each said act sits between the source and the target
            said, maybe = [], []
            for x in (t.get("say") or []):              # 'ACT', or 'ACT' if <condition> else None (said only sometimes)
                mt = re.match(r"""^['"]([A-Za-z_][A-Za-z0-9_]*)['"](\s+if\s+.*)?$""", str(x).strip())
                if mt and mt.group(1) in dialog and mt.group(1) != t["dst"]:
                    (maybe if mt.group(2) else said).append(mt.group(1))
            for extra in ([[]] + [[m] for m in maybe]) if maybe else [[]]:
                hops = [t["src"], *extra, *said, t["dst"]]
                transitions += [(a, b, t.get("on", "auto") if i == 0 else "auto") for i, (a, b) in enumerate(zip(hops, hops[1:]))]
        if any(d not in dialog for _, d, _ in transitions):     # quiet tool steps: what the bot says next lies past them
            out: dict[str, list[tuple[str, str]]] = {}
            for s0, d0, o0 in transitions:
                out.setdefault(s0, []).append((d0, o0))
            flat = []
            for s0, d0, o0 in transitions:
                if s0 not in dialog:
                    continue
                todo, seen = [d0], set()
                while todo:
                    x = todo.pop()
                    if x in seen:
                        continue
                    seen.add(x)
                    if x in dialog:
                        flat.append((s0, x, o0))
                    else:
                        todo += [d for d, _ in out.get(x, [])]
            transitions = list(dict.fromkeys(flat))
        # what may happen in ANY prompt state: an act every prompt state handles leads, anywhere, to the union
        # of where it leads (pdf.2F, pdf.3, R28: a person on request, a policy answer, the turn limit's summary
        # and transfer). The project adds the grading leniencies the graph does not state (speech.also_allowed).
        prompt = set(kg.prompt_states)
        handled: dict[str, dict[str, set[str]]] = {}
        for src, dst, on in transitions:
            if src in prompt and on not in ("auto", "else"):
                handled.setdefault(on, {}).setdefault(src, set()).add(dst)
        global_acts = {on: set().union(*m.values()) for on, m in handled.items() if set(m) == prompt}
        for on, extra in also_allowed().items():
            global_acts.setdefault(on, set()).update(extra)
        opening = (kg.tails("opening_statement", "has_text") or [""])[0]
        if g is not None:
            phrases = {v: list(ps) for v, ps in model.phrases.items()}
        c = cls(acts=acts, phrases=phrases, value_sets=value_sets, stances=stances,
                   item_fields=kg.returned_fields(graph_names().item_tool) if graph_names().item_tool else {}, opening=opening,
                   limits=limits, limit_phrases={k: v for k, v in limit_phrases.items() if v}, transitions=transitions,
                   prompt_states=set(kg.prompt_states), global_acts=global_acts,
                   success_values={t["head"]: str(t["tail"]) for t in kg.where("success_value")})
        c.model = model if g is not None else None
        return c

    model: Any = None                              # generic.PackModel for a pack compiled from a compact graph

    def claim_field(self, fs) -> str | None:
        """The claim field a required ``says`` field is read as (the field name itself for charge review)."""
        if self.model is None:
            return fs.field
        from .speech_checks import claim_field
        return claim_field(self.model, fs.source)

    # ------------------------------------------------------------------ lookups
    def value_set_of(self, value: str) -> str | None:
        """The value set a value belongs to: "REPLACE_CARD" -> "card_action"; None for an unknown value."""
        return next((vs for vs, vals in self.value_sets.items() if value in vals), None)

    def next_states(self, state: str, on: str | None = None) -> set[str]:
        """States the graph lets the bot move to from ``state`` (on a customer act, or automatically)."""
        return {dst for src, dst, o in self.transitions if src == state and
                (on is None or o in (on, "any") or (o == "else" and on not in ("auto",)))}

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view of the contract (for reports and debugging)."""
        return {"acts": {a: {"says": [vars(f) for f in c.says], "cues": list(c.cues)} for a, c in self.acts.items()},
                "phrases": self.phrases, "stances": {r: vars(s) for r, s in self.stances.items()},
                "item_fields": self.item_fields, "opening": self.opening}


# =====================================================================================================================
# PART 2 - Session facts: what is true in one conversation so far.
#
# The policy store holds what is true for every customer (fees, limits, rules). What is true for
# *this* customer - which charges were returned, what was recorded, which card actions were offered,
# what the customer said - lives only in the conversation. ``SessionFacts`` collects it from the
# mock backend's tool log and the customer's turns (as the runner's reader understood them), so a
# sentence like "the $45.67 Amazon charge, recorded as not recognized" can be checked.
#
# Input   the tool log (``[{tool, args, result, ok}]`` in call order), the customer's semantic
#         frames in turn order, the session date and the success status of each tool (from the
#         policy graph).
# Output  ``SessionFacts`` with:
#           items       transaction_id -> the fields the tool returned (merchant, amount, date, ...)
#           writes      tool -> every write that truly succeeded (status = success value AND every
#                       argument the tool echoes back equals what was sent: rules R12/R13/R17/R33)
#           options     the card actions get_card_action_options offered (None if never asked)
#           caller      slot -> the latest value the customer stated (disposition, card_action, ...)
#         and ``resolve(binding, about)`` -> ``Truth``: the true value of one binding (see
#         graph/triples.py BINDING_KINDS), with where it was read.
#
#     facts = SessionFacts.from_run(tr.tool_log, tr.user_frames, "2026-03-16", contract.success_values)
#     facts.resolve("item.amount", about="T1")                         # Truth(value='45.67', source='tool result: T1.amount')
#     facts.resolve("write.record_charge_response.disposition", about="T1")
# =====================================================================================================================

from ..packs import names as roles

def write_key(tool: str) -> str | None:
    """What a write is about: the field it echoes (graph ``echoes``), else for the per-item write its item id and
    for the other write its action argument."""
    from ..packs.names import graph_names
    kg = graph_names()
    echo = kg.echo_of.get(tool)
    if echo:
        return echo
    if tool == roles.RECORD:
        return roles.KEY
    if tool == roles.ACTION:
        return roles.ACT
    return None


@dataclass(frozen=True)
class Truth:
    """The true value for a claim and where it was read. ``known`` is False when the session has
    nothing to compare against (e.g. a 'recorded' claim before any record succeeded)."""
    value: Any
    source: str
    known: bool = True


NOTHING = Truth(None, "", known=False)


@dataclass
class SessionFacts:
    """What is true for this customer, so far in this conversation (see the module docstring)."""
    session_date: str | None = None
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    writes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    failed_writes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    options: list[str] | None = None
    caller: dict[str, Any] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)     # tool -> its latest result

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_run(cls, tool_log: list[dict[str, Any]], customer_frames: list[Any], session_date: str | None = None,
                 success_values: dict[str, str] | None = None) -> "SessionFacts":
        """Collect the session's facts from the tool log and the customer's frames.

        tool_log         ``[{tool, args, result, ok}]`` in call order (the mock backend's ``env.log``)
        customer_frames  the customer's semantic frames so far, in turn order (later answers win)
        session_date     the fixture's "today" (dates are read against it), or None
        success_values   tool -> the status that means it worked (``contract.success_values``); "SUCCESS" if absent
        """
        success_values = success_values or {}
        f = cls(session_date=session_date)
        for c in tool_log:
            tool, res = c.get("tool"), c.get("result") or {}
            if c.get("ok", True):
                f.results[tool] = dict(res)
            if not c.get("ok", True):
                f.failed_writes.setdefault(tool, []).append(c)
                continue
            if tool is not None and tool == roles.ITEMS:
                for t in res.get("transactions") or []:
                    f.items[str(t.get("transaction_id"))] = dict(t)
            elif tool is not None and tool == roles.OPTIONS:
                f.options = list(res.get("options") or [])
            elif tool in roles.WRITE_TOOLS:
                ok = res.get("status") == success_values.get(tool, roles.OK)
                echoed = all(res.get(k) == v for k, v in (c.get("args") or {}).items() if k in res and v is not None)
                (f.writes if ok and echoed else f.failed_writes).setdefault(tool, []).append(c)
        for fr in customer_frames:
            for slot, value in getattr(fr, "slots", ()):
                if value is not None:
                    f.caller[slot] = value
        return f

    # ------------------------------------------------------------------ lookups
    def item_by_merchant(self, name: str) -> str | None:
        """The transaction id whose returned merchant is ``name`` (case-insensitive), or None."""
        n = name.strip().lower()
        return next((tid for tid, it in self.items.items() if str(it.get("merchant", "")).lower() == n), None)

    def write_for(self, tool: str, about: str | None) -> dict[str, Any] | None:
        """The latest successful write of ``tool`` about ``about`` (a transaction id / an action), or the
        latest successful write when ``about`` is None."""
        key = write_key(tool)
        rows = [w for w in self.writes.get(tool, []) if about is None or str((w.get("args") or {}).get(key)) == str(about)]
        return rows[-1] if rows else None

    def resolve(self, binding: str, about: str | None = None) -> Truth:
        """The true value of a session binding (item.*, write.*, writes.*, caller.*). Policy bindings are
        resolved by the verifier against the GMS, not here."""
        kind, _, rest = binding.partition(".")
        if kind == "item":
            it = self.items.get(str(about)) if about is not None else None
            if it is None or rest not in it:
                return NOTHING
            return Truth(it[rest], f"tool result get_review_transactions: {about}.{rest}")
        if kind == "write":
            tool, _, arg = rest.partition(".")
            w = self.write_for(tool, about if tool == roles.RECORD else None)
            if w is None:
                return NOTHING
            return Truth((w.get("args") or {}).get(arg), f"successful {tool}({_args(w)})")
        if kind == "writes":
            rows = self.writes.get(rest, [])
            return Truth([dict(w.get("args") or {}) for w in rows], f"every successful {rest}", known=True)
        if kind == "caller":
            if rest in self.caller:
                return Truth(self.caller[rest], f"the customer said {rest}={self.caller[rest]}")
            return NOTHING
        return NOTHING

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view (for reports and debugging)."""
        return {"session_date": self.session_date, "items": self.items,
                "writes": {k: [_args(w) for w in v] for k, v in self.writes.items()},
                "options": self.options, "caller": self.caller}


def _args(w: dict[str, Any]) -> str:
    """A write's arguments as "transaction_id=T1, disposition=UNRECOGNIZED"."""
    return ", ".join(f"{k}={v}" for k, v in (w.get("args") or {}).items() if v is not None)


# =====================================================================================================================
# PART 3 - Reading a bot utterance into claims, without looking at the bot's act labels.
#
# Input   the text the customer hears, the session facts (to recognise the merchants that were
#         returned), the speech contract (how values and acts are worded, from the policy graph)
#         and, optionally, the transaction the act is about (used only when a sentence states an
#         account fact without naming the merchant: "It was $45.67, 3 days ago.").
# Output  a ``ReadUtterance``: the sentences, the ``Claim``s they make, and the acts the words
#         express (``heard``).
#
# A claim is one fact the words state, e.g. for "Thanks, I've noted that you don't recognize the
# $45.67 Amazon charge." the reader returns
#
#     Claim(field="amount",      value="45.67",        about="T1", mode="states",   words="$45.67")
#     Claim(field="merchant",    value="Amazon",       about="T1", mode="states",   words="Amazon")
#     Claim(field="disposition", value="UNRECOGNIZED", about="T1", mode="recorded", words="don't recognize")
#
# ``mode`` says how the fact is put: "states" (asserted), "recorded" (said to have been recorded),
# "done" (an action said to be completed), "offered" (an action proposed in a question). A question
# or an "if ..." sentence asserts no disposition and no completion ("Do you recognize it?" is not a
# claim that the customer recognized it), but the figures it carries are still claims ("Shall I send
# it by expedited delivery for $25?" states a $25 fee).
#
# How each fact is found (all case-insensitive; numbers spoken as words are read as digits):
#   merchant        a returned transaction's merchant name
#   amount          a dollar figure that is not about a fee         -> the nearest named merchant's transaction
#   date            "March 13", "13 March", "2026-03-13"            -> the nearest named merchant's transaction
#   age_days        "3 days ago"                                    -> the nearest named merchant's transaction
#   fee             a dollar figure (or "free") in a sentence about a fee or a delivery option -> that option
#   delivery_days   "2 business days", "5-7 business days"          -> the nearest delivery option named
#   limit           the graph's limit wordings ("up to {n} charges") -> that limit
#   disposition / action / delivery   the graph's wordings of each value (bot phrasings + customer synonyms)
#   stance          the graph's asserting / denying wordings of a rule's fact (denials read first)
#   opening         the required opening statement, word for word
#   recorded_count  "I recorded 2 responses"
# =====================================================================================================================


# ------------------------------------------------------------------ plain-English signals (not policy)
SIGNALS = ("condition_openers", "recording", "completion", "offer", "fee", "free")


def signal_words() -> dict[str, list[str]]:
    """The project's cue lists (project.yaml, speech.signals)."""
    from ..packs.project import project
    return dict((project().raw.get("speech") or {}).get("signals") or {})


def _words(name: str) -> str:
    words = signal_words().get(name)
    if not words:
        raise ValueError(f"the project has no speech.signals.{name} cue list (project.yaml; "
                         "python -m gmsjourney.packs.project check)")
    return "|".join(words)


def _compile_signals() -> None:
    """Compile the cue patterns from the active project (again on a pack switch)."""
    global QUESTION_OR_CONDITION, RECORDING_VERBS, COMPLETION, OFFER, FEE_WORDS, FREE
    QUESTION_OR_CONDITION = re.compile(rf"\?\s*$|^\s*({_words('condition_openers')})\b", re.I)
    RECORDING_VERBS = re.compile(rf"\b({_words('recording')})\b", re.I)
    COMPLETION = re.compile(rf"\b({_words('completion')})\b", re.I)
    OFFER = re.compile(rf"\b({_words('offer')})\b", re.I)
    FEE_WORDS = re.compile(rf"\b({_words('fee')})\b", re.I)
    FREE = re.compile(rf"\b({_words('free')})\b", re.I)


_compile_signals()
MONEY = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)|\b(\d+(?:\.\d{1,2})?)\s*dollars?\b", re.I)
BUSINESS_DAYS = re.compile(r"\b(\d+)(?:\s*(?:-|–|to)\s*(\d+))?\s+business\s+days?\b", re.I)
AGE = re.compile(r"\b(\d+)\s+days?\s+ago\b", re.I)
RECORDED_COUNT = re.compile(r"\brecorded\s+(\d+)\s+(?:responses?|answers?)\b", re.I)
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                          "nov", "dec"])}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_MD = re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
DATE_DM = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}(?:,?\s+(\d{{4}}))?", re.I)
DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_NUMBER_WORDS = re.compile(r"\b(" + "|".join(_TENS) + r")(?:[ -](" + "|".join(list(_UNITS)[1:10]) + r"))?\b|\b("
                           + "|".join(_UNITS) + r")\b", re.I)


def words_to_digits(text: str) -> str:
    """'fifteen dollars' -> '15 dollars', 'twenty-five' -> '25' (0-99; enough for spoken figures)."""
    def sub(m: re.Match) -> str:                       # one number word (or tens-units pair) -> digits
        if m.group(1):
            return str(_TENS[m.group(1).lower()] + (_UNITS[m.group(2).lower()] if m.group(2) else 0))
        return str(_UNITS[m.group(3).lower()])
    return _NUMBER_WORDS.sub(sub, text)


# A figure that names nothing, said after the asked-about thing already got its figure ("Standard delivery is free.
# There's also a $15 option."): it is about one of the members, not necessarily the asked one, so it is checked
# against all of them (``about == ANY``).
ANY = "*"


# ------------------------------------------------------------------ the output
@dataclass(frozen=True)
class Claim:
    """One fact an utterance states.

    field     what is claimed: merchant, amount, date, age_days, disposition, action, delivery, fee,
              delivery_days, limit, stance, opening, recorded_count
    value     the value the words give (a delivery_days value is a (low, high) pair; a stance value is
              +1 for asserting the rule's fact and -1 for denying it)
    words     the exact words that state it
    sentence  the sentence it is in
    about     what it is about: a transaction id, a delivery option, a limit key or a rule id (None
              when the sentence does not say)
    mode      states | recorded | done | offered
    """
    field: str
    value: Any
    words: str
    sentence: str
    about: str | None = None
    mode: str = "states"

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view (the sentence is left out; it is on the turn)."""
        return {"field": self.field, "value": list(self.value) if isinstance(self.value, tuple) else self.value,
                "about": self.about, "mode": self.mode, "words": self.words}


@dataclass
class ReadUtterance:
    """What the reader understood from one utterance.

    text       the utterance as given
    sentences  the utterance cut into sentences (claims never cross a sentence)
    claims     every fact the words state, in sentence order
    heard      the acts the words express (used when the bot gives no labels, and to check a label was said)
    """
    text: str
    sentences: list[str]
    claims: list[Claim] = field(default_factory=list)
    heard: list[str] = field(default_factory=list)       # acts the words express, in order of first evidence

    def of(self, field_name: str) -> list[Claim]:
        """The claims about one field: ``read.of("fee")``."""
        return [c for c in self.claims if c.field == field_name]


# ------------------------------------------------------------------ the reader
@dataclass
class _Mention:
    """Where a known thing (a returned charge, a delivery option, a disposition, ...) is named in a sentence."""
    kind: str           # item | option | money | ...
    value: Any
    start: int
    end: int
    words: str


class UtteranceReader:
    """Reads utterances with the wordings the speech contract (the policy graph) declares."""

    def __init__(self, contract: SpeechContract):
        self.c = contract
        # value phrasings, longest first, so "not recognized" wins over "recognized"
        self._value_phrases: dict[str, list[tuple[str, str]]] = {}
        for vs, vals in contract.value_sets.items():
            pairs = [(p.lower(), v) for v in vals for p in contract.phrases.get(v, [])]
            self._value_phrases[vs] = sorted(set(pairs), key=lambda x: -len(x[0]))
        self._limit_res = {k: [re.compile(r"\b" + re.escape(p.lower()).replace(r"\{n\}", r"(\d+)") + r"\b")
                               for p in ps] for k, ps in contract.limit_phrases.items()}
        self._opening = _squash(contract.opening)

    # ---------------------------------------------------------------- public
    def read(self, text: str, facts: SessionFacts, about_item: str | None = None,
             question: Any = None) -> ReadUtterance:
        """Read ``text`` into claims.

        about_item  the transaction a sentence is about when it names no merchant (the act's transaction)
        question    the customer's policy question when the text answers one (an ASK_KB frame): its
                    figures are then policy figures about what was asked (a fee of the option asked
                    about), and "N days ago" restates the question, not an account fact
        """
        out = ReadUtterance(text=text or "", sentences=_sentences(text or ""))
        if self._opening and self._opening in _squash(text or ""):
            out.claims.append(Claim("opening", True, self.c.opening, text))
        year = int(facts.session_date[:4]) if facts.session_date else date.today().year
        seen: set[tuple[str, Any]] = set()                     # (field, about) of the figures said so far
        for sent in out.sentences:
            new = self._read_sentence(sent, facts, about_item, year, question, seen)
            seen.update((c.field, c.about) for c in new)
            out.claims += new
        out.heard = self._heard(out)
        return out

    # ---------------------------------------------------------------- one sentence
    def _read_sentence(self, sent: str, facts: SessionFacts, about_item: str | None, year: int,
                       question: Any = None, seen: set | None = None) -> list[Claim]:
        """Every claim in one sentence. Facts attach to the nearest thing named in the sentence (a merchant
        for account facts, a delivery option for policy figures), else to ``about_item`` / the question."""
        low = words_to_digits(sent).lower()
        asks = bool(QUESTION_OR_CONDITION.search(sent.strip()))
        claims: list[Claim] = []
        items = self._items_named(low, facts)                      # merchants of returned transactions
        options = self._values_named(low, "delivery")              # delivery options named
        asked_option = question.get("entity") if question is not None else None
        account_talk = question is None or bool(items)             # an answer speaks of the account only by name
        near_item = lambda pos: _nearest(items, pos) or about_item  # noqa: E731
        seen = seen if seen is not None else set()

        def near_option(pos: int, fld: str) -> str | None:
            """The option named nearest; else the asked one, unless it already has this figure (then any option)."""
            return _nearest(options, pos) or fallback(asked_option, fld, seen)

        for m in items:
            claims.append(Claim("merchant", facts.items[m.value]["merchant"], m.words, sent, m.value))

        # a dollar figure is a fee when the sentence is about a delivery option, or speaks of a fee and
        # names no charge, or answers a policy question; otherwise it is the amount of the charge it is next to
        about_fee = bool(options) or (bool(FEE_WORDS.search(low)) and not items) or not account_talk
        for m in MONEY.finditer(low):
            v = m.group(1) or m.group(2)
            if about_fee:
                claims.append(Claim("fee", _num(v), m.group(0), sent, near_option(m.start(), "fee")))
            else:
                claims.append(Claim("amount", v, m.group(0), sent, near_item(m.start())))
        if options and FREE.search(low):
            fm = FREE.search(low)
            claims.append(Claim("fee", 0, fm.group(0), sent, near_option(fm.start(), "fee")))

        if account_talk:
            for m, iso in self._dates(low, year):
                claims.append(Claim("date", iso, m.group(0), sent, near_item(m.start())))
            for m in AGE.finditer(low):
                claims.append(Claim("age_days", int(m.group(1)), m.group(0), sent, near_item(m.start())))
        for m in BUSINESS_DAYS.finditer(low):
            lo = int(m.group(1))
            hi = int(m.group(2) or m.group(1))
            claims.append(Claim("delivery_days", (lo, hi), m.group(0), sent, near_option(m.start(), "delivery_days")))
        for key, regs in self._limit_res.items():
            for rx in regs:
                for m in rx.finditer(low):
                    claims.append(Claim("limit", int(m.group(1)), m.group(0), sent, key))
        m = RECORDED_COUNT.search(low)
        if m:
            claims.append(Claim("recorded_count", int(m.group(1)), m.group(0), sent))

        # dispositions and actions: what the words assert, never what a question asks
        if not asks:
            mode = "recorded" if RECORDING_VERBS.search(low) else "states"
            for m in self._values_named(low, "disposition"):
                claims.append(Claim("disposition", m.value, m.words, sent, near_item(m.start), mode))
        act_mode = ("offered" if asks or OFFER.search(low) else "done" if COMPLETION.search(low) else "states")
        for m in self._values_named(low, "card_action"):
            claims.append(Claim("action", m.value, m.words, sent, None, act_mode))
        for m in options:
            claims.append(Claim("delivery", m.value, m.words, sent, None, act_mode))

        for rid, sp in self.c.stances.items():
            hit = next((p for p in sp.denied if p.lower() in low), None)
            if hit:
                claims.append(Claim("stance", -1, hit, sent, rid))
                continue
            hit = next((p for p in sp.asserted if p.lower() in low), None)
            if hit and not asks:
                claims.append(Claim("stance", +1, hit, sent, rid))
        return _dedupe(claims)

    # ---------------------------------------------------------------- mentions
    @staticmethod
    def _items_named(low: str, facts: SessionFacts) -> list[_Mention]:
        """Mentions of the returned transactions' merchants, in sentence order."""
        out = []
        for tid, it in facts.items.items():
            name = str(it.get("merchant") or "").lower()
            if not name:
                continue
            for m in re.finditer(r"\b" + re.escape(name) + r"\b", low):
                out.append(_Mention("item", tid, m.start(), m.end(), m.group(0)))
        return sorted(out, key=lambda x: x.start)

    def _values_named(self, low: str, value_set: str) -> list[_Mention]:
        """Non-overlapping mentions of a value set's values, longest wording first."""
        taken: list[tuple[int, int]] = []
        out = []
        for phrase, value in self._value_phrases.get(value_set, []):
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                out.append(_Mention(value_set, value, m.start(), m.end(), m.group(0)))
        return sorted(out, key=lambda x: x.start)

    @staticmethod
    def _dates(low: str, year: int):
        """(match, "YYYY-MM-DD") for every calendar date in the sentence; a date with no year takes ``year``."""
        for m in DATE_MD.finditer(low):
            iso = _iso(int(m.group(3) or year), MONTHS[m.group(1)[:3]], int(m.group(2)))
            if iso:
                yield m, iso
        for m in DATE_DM.finditer(low):
            iso = _iso(int(m.group(3) or year), MONTHS[m.group(2)[:3]], int(m.group(1)))
            if iso:
                yield m, iso
        for m in DATE_ISO.finditer(low):
            iso = _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if iso:
                yield m, iso

    # ---------------------------------------------------------------- acts the words express
    def _heard(self, r: ReadUtterance) -> list[str]:
        """The acts the words express. An act is heard when
          * one of its cue words is said ("anything else" -> ASK_MORE_HELP), or
          * every fact it requires from the tools or the policy is stated (a merchant, an amount and a
            date -> PRESENT_TXN; a fee -> DISCLOSE_FEE; the opening statement -> OPENING), or
          * the way something is put says it: an action said to be done -> ACTION_DONE, offered in a
            question -> CONFIRM_ACTION, an answer said to be recorded -> CONFIRM_RECORDED."""
        low = words_to_digits(r.text).lower()
        said = {c.field for c in r.claims}
        modes = {f"{c.field}:{c.mode}" for c in r.claims}
        heard = []
        for act, spec in self.c.acts.items():
            facts = [f.field for f in spec.says if f.required and f.source.split(".")[0] in ("item", "policy", "text")]
            if any(q.lower() in low for q in spec.cues) or (facts and all(f in said for f in facts)):
                heard.append(act)
        for mode, act in (("action:done", N.state.ACTION_DONE), ("action:offered", N.state.CONFIRM_ACTION),
                          ("disposition:recorded", N.state.CONFIRM_RECORDED)):
            if mode in modes and act in self.c.acts and act not in heard:
                heard.append(act)
        return heard


# ------------------------------------------------------------------ helpers
def _sentences(text: str) -> list[str]:
    """Split on . ? ! followed by a capital, a digit, a $ or a quote ("$45.67" and "U.S." stay whole)."""
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z$\d\"'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _squash(text: str) -> str:
    """Lower case, punctuation as single spaces: for word-for-word comparisons that ignore punctuation."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _num(v: str) -> float | int:
    """ "15" -> 15, "6.25" -> 6.25."""
    f = float(v)
    return int(f) if f.is_integer() else f


def _iso(y: int, mo: int, d: int) -> str | None:
    """An ISO date, or None for an impossible one ("February 30")."""
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def fallback(asked: str | None, fld: str, seen: set) -> str | None:
    """What a figure that names nothing is about: the asked-about thing the first time, ``ANY`` after that."""
    if asked is None:
        return None
    return ANY if (fld, asked) in seen else asked


def _nearest(mentions: list[_Mention], pos: int) -> str | None:
    """The mention closest to a position in the sentence (a tie goes to the one before it)."""
    if not mentions:
        return None
    best = min(mentions, key=lambda m: (min(abs(pos - m.end), abs(pos - m.start)), m.start > pos))
    return best.value


def _dedupe(claims: list[Claim]) -> list[Claim]:
    """Drop exact repeats (the same words read twice as the same claim), keeping the first."""
    seen, out = set(), []
    for c in claims:
        k = (c.field, str(c.value), c.about, c.mode, c.words)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


from ..packs.loading import on_switch as _on_switch  # noqa: E402

_on_switch(_compile_signals)
