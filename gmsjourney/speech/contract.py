"""The speech contract: everything the validator needs from the policy graph, in one object.

Input   a ``PolicyKG`` (knowledge/policy_kg.py), usually ``store.kg``.
Output  a ``SpeechContract``: per bot act what it says and where each value's truth lives, the
        words that mark an act, how the bot words each value, the rule stances it can be heard
        asserting or denying, the fields a returned transaction has, the opening statement and
        the dialog-state transitions.

Nothing here is specific to a bot; it is read from the graph's "speech" triples (written in
knowledge/policy_kg.py: BOT_SPEECH, TOOL_RETURNS, BOT_PHRASES, STANCE_PHRASES) and its existing
vocabulary and state layers.

    contract = SpeechContract.from_graph(store.kg)
    contract.acts["PRESENT_TXN"].required          # ('merchant', 'amount', 'date')
    contract.acts["PRESENT_TXN"].source_of("date")  # 'item.date'
    contract.phrases["UNRECOGNIZED"]               # ['not recognized', "isn't yours", ..., 'not mine']
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
            raise ValueError("the policy graph has no speech layer (build it with knowledge/policy_kg.py)")
        acts = {}
        for st in kg.states:
            says = tuple(FieldSpec(r["field"], r["from"], bool(r.get("required"))) for r in kg.says(st))
            src = next((t["source"] for t in kg.where("says", head=st)), None)
            acts[st] = ActContract(st, says, tuple(kg.spoken_cues(st)), src)
        value_sets = {vs: kg.value_set(vs) for vs in ("disposition", "card_action", "delivery")}
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
        transitions = [(t["src"], t["dst"], t.get("on", "auto")) for t in kg.transitions()]
        # what the policy allows from ANY prompt state (pdf.2F, pdf.3, R28): a person on request, a
        # policy answer, and the turn limit's summary / transfer / end. A clarification is allowed after
        # anything: the bot cannot know what the customer meant, and asking is what the policy prefers
        # to guessing (R9, R21). An unneeded clarification costs a turn; the outcome and the turn-by-turn
        # comparison grade that, not the flow of what was said.
        global_acts = {"REQUEST_HUMAN": {"TRANSFER", "SUMMARIZE"}, "ASK_KB": {"ANSWER_KB"},
                       "UNCLEAR": {"TRANSFER", "SUMMARIZE"}, "any": {"ASK_CLARIFY", "TRANSFER", "SUMMARIZE", "END"}}
        opening = (kg.tails("opening_statement", "has_text") or [""])[0]
        return cls(acts=acts, phrases=phrases, value_sets=value_sets, stances=stances,
                   item_fields=kg.returned_fields("get_review_transactions"), opening=opening,
                   limits=limits, limit_phrases={k: v for k, v in limit_phrases.items() if v}, transitions=transitions,
                   prompt_states=set(kg.prompt_states), global_acts=global_acts,
                   success_values={t["head"]: str(t["tail"]) for t in kg.where("success_value")})

    # ------------------------------------------------------------------ lookups
    def value_set_of(self, value: str) -> str | None:
        """The value set a value belongs to: "REPLACE_CARD" -> "card_action"; None for an unknown value."""
        return next((vs for vs, vals in self.value_sets.items() if value in vals), None)

    def next_states(self, state: str, on: str | None = None) -> set[str]:
        """States the graph lets the bot move to from ``state`` (on a customer act, or automatically)."""
        return {dst for src, dst, o in self.transitions if src == state and (on is None or o in (on, "any"))}

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view of the contract (for reports and debugging)."""
        return {"acts": {a: {"says": [vars(f) for f in c.says], "cues": list(c.cues)} for a, c in self.acts.items()},
                "phrases": self.phrases, "stances": {r: vars(s) for r, s in self.stances.items()},
                "item_fields": self.item_fields, "opening": self.opening}
