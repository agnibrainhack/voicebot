"""The LLM rewording arm: a language model writes customer turns, under gates.

The template realizer cannot produce the open-ended variety of real callers; an LLM can, but it is an untrusted
writer (it may drop a value, add a figure, hedge a decision). So each LLM-written turn passes per-turn gates
before the shared proof accepts it.

Parts of this file:
  1. Prompts: the rephraser (meaning in plain words, base wording, factor instructions; never the expected
     outcome), the independent reader and the adversary.
  2. ``LLMRealizer``: meaning card + factor levels -> utterance written by the model.
  3. ``LLMFrameExtractor``: a second, independent reader for the dual read.
  4. ``LLMOutputGates``: G0 form, G1 facts, G3 one clear decision (G2 meaning is the shared reader's).
  5. CSV exports for the arm (one row per turn, controls audit, arm comparison).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =====================================================================================================================
# PART 1 - Prompts for the LLM enrichment arm (rephraser, independent reader, adversary).
#
# What the rephraser is allowed to see, by design ("blind generation"):
#
# * the bot's question for the dialog state the turn answers (generic, from ``STATE_QUESTIONS``);
# * what the customer means, in plain words (``describe_meaning`` of the base meaning card);
# * the customer's base wording (the template realisation at all-base levels);
# * the presentation instructions for the scenario's factor levels (``factor_instructions``);
# * the tokens it must keep verbatim and the names the rule reader understands.
#
# What it never sees: the expected bot behaviour, the outcome signature, the oracle trace,
# tool names, rule ids, or any policy figure that is not already in the customer's own
# words. A model that never sees the answer cannot leak it or bend the question toward it.
#
# The independent reader (``EXTRACT_*``) sees only the utterance and the bot's question,
# never the base meaning, and must fill a closed meaning-card schema. The adversary
# (``ADVERSARY_*``) is asked to change exactly one meaning element; its rewrites are
# negative controls for the LLM arm's gate stack.
# =====================================================================================================================

from ..s04_cases.meaning import SemanticFrame as F
from .wording import number_to_words, spell_digits
from .wording import BASE_LEVELS, Realizer

# The bot's question in each customer-facing state (generic: the rephraser must not
# depend on one bot's phrasing). {merchant}/{amount} are filled from the fixture.
STATE_QUESTIONS = {
    "ASK_AUTH": "Before I review account-specific charges, I need to verify you. What are the last four digits of your card?",
    "ASK_AUTH_RETRY": "That didn't match. Please try the last four digits of your card again.",
    "PRESENT_TXN": "I see a charge of ${amount} at {merchant}. Do you recognize that charge?",
    "PROVIDE_CONTEXT": "It was ${amount} at {merchant}. Do you recognize it?",
    "ASK_CLARIFY": "Sorry, I didn't catch that. Do you recognize the {merchant} charge?",
    "ANSWER_KB": "(The agent has just answered a question.) Now, about the {merchant} charge: do you recognize it?",
    "CONFIRM_ACTION": "Shall I go ahead with that card request?",
    "DISCLOSE_FEE": "Expedited delivery has a fee. Shall I go ahead?",
    "ASK_MORE_HELP": "Is there anything else I can help with?",
    "EXPLAIN_NO_TXNS": "There are no charges available for review. Is there anything else I can help with?",
}
DEFAULT_QUESTION = "The agent is waiting for your reply."

# Words the rule reader binds (s05_rewording/wording.py patterns). The LLM may only name
# values with these words; anything else the reader would not read is a false reject.
READER_LEXICON = {
    "standard_delivery": ["standard delivery", "regular shipping", "normal delivery"],
    "expedited_delivery": ["expedited delivery", "express shipping", "rush delivery"],
    "SECURE_CARD": ["secure my card", "lock my card", "freeze the card", "block my card"],
    "REPLACE_CARD": ["replace my card", "send me a new card", "get me a fresh card", "reissue the card"],
    "RECOGNIZED": ["I recognize it", "it's mine", "that was me", "I made that purchase", "that's mine"],
    "UNRECOGNIZED": ["I don't recognize it", "it's not mine", "that wasn't me", "I never made that", "I didn't buy that"],
    "UNSURE": ["I'm not sure", "I can't remember", "I'm not certain", "no idea"],
    "REQUEST_HUMAN": ["a human", "a real person", "a representative", "a live agent"],
    "END": ["that's all", "nothing else", "all set", "that's everything"],
}
LABEL = {"standard_delivery": "standard delivery", "expedited_delivery": "expedited delivery",
         "SECURE_CARD": "secure the card", "REPLACE_CARD": "replace the card",
         "RECOGNIZED": "'I recognize it'", "UNRECOGNIZED": "'I don't recognize it'", "UNSURE": "'I'm not sure'",
         "max_transactions_per_session": "how many transactions can be reviewed in one call",
         "lookback_days": "how far back (how many days) charges can be reviewed",
         "max_auth_attempts": "how many verification attempts are allowed"}


def state_question(state: str | None, ctx: dict[str, Any] | None) -> str:
    """What the agent just asked in a state, as a sentence (the context the LLM rewrites the customer's line in)."""
    from . import llm_pack as pack
    dom = pack.active()
    if dom is not None:
        return pack.state_question(dom, state, ctx)
    q = STATE_QUESTIONS.get(state or "", DEFAULT_QUESTION)
    ctx = ctx or {}
    return q.replace("{merchant}", str(ctx.get("merchant") or "this")).replace("{amount}", str(ctx.get("amount") or "…"))


# ----------------------------------------------------------------------------- meaning
def _num(v: Any, spoken: bool, money: bool = False) -> str:
    n = int(float(v))
    if spoken:
        return f"{number_to_words(n)} dollars" if money else number_to_words(n)
    return f"${n}" if money else str(n)


def describe_meaning(f: F, spoken: bool = False) -> str:
    """The meaning card in plain words (no ground truth: what the customer *says*, not what is right)."""
    from . import llm_pack as pack
    dom = pack.active()
    if dom is not None:
        d = pack.describe_meaning(dom, f, spoken)
        if d is not None:
            return d
    a = f.act
    if a == "PROVIDE_CREDENTIAL":
        c = f.get("credential")
        return f"gives the last four digits of their card: {c}" + (f" (spoken: {spell_digits(c)})" if spoken else "")
    if a == "INFORM_DISPOSITION":
        d = f.get("disposition")
        s = {"RECOGNIZED": "says the charge being discussed IS theirs (they recognize it)",
             "UNRECOGNIZED": "says the charge being discussed is NOT theirs (they do not recognize it)",
             "UNSURE": "says they are NOT SURE whether the charge is theirs (no yes or no)"}[d]
        if f.get("corrected_from"):
            first = {"RECOGNIZED": "that it is theirs", "UNRECOGNIZED": "that it is not theirs",
                     "UNSURE": "that they are unsure"}[f.get("corrected_from")]
            s += (f"; they FIRST say {first}, then correct themselves in the same breath with an explicit "
                  f"repair ('wait, actually ...'), so the FINAL answer is the one above")
        if f.get("card_action"):
            s += f"; they ALSO ask you to {LABEL[f.get('card_action')]}"
            if f.get("delivery"):
                s += f" with {LABEL[f.get('delivery')]}"
            s += " - BOTH parts must be said: the answer about the charge AND the card request"
        return s
    if a == "REQUEST_CARD_ACTION":
        s = f"asks you to {LABEL[f.get('card_action')]}"
        if f.get("delivery"):
            s += f" with {LABEL[f.get('delivery')]} instead"
        return s
    if a == "CONFIRM":
        return "clearly agrees: yes, go ahead"
    if a == "DENY":
        return "clearly declines: no, do not do it"
    if a == "REQUEST_HUMAN":
        return "asks to speak to a human representative"
    if a == "END":
        return "says they need nothing else and the call can end"
    if a == "UNCLEAR":
        return "did not catch what the agent said and asks them to repeat it (no decision at all)"
    if a == "CHITCHAT":
        return "makes an off-topic remark with no decision"
    if a == "ASK_KB":
        return "asks a question: " + _describe_kb(f, spoken)
    return a


def _describe_kb(f: F, spoken: bool) -> str:
    i = f.get("intent")
    E = LABEL.get(f.get("entity"), "")
    if i == "value":
        return f"how much {E} costs" if f.get("attr") == "fee" else f"how long {E} takes (business days)"
    if i == "limit":
        return LABEL[f.get("key")]
    if i == "plausibility":
        v = int(f.get("value"))
        return f"whether {E} is free (a yes/no check)" if v == 0 else \
            f"whether {E} costs {_num(v, spoken, money=True)} (a yes/no check of exactly that figure)"
    if i == "mcq":
        opts = ", ".join(_num(o, spoken, money=True) for o in f.get("options"))
        return f"which of these fees applies to {E}, keeping the options in this order: {opts}"
    return {
        "faster": "which delivery option is faster",
        "cheaper": "which delivery option is cheaper",
        "fee_difference": "how much more expedited delivery costs than standard delivery",
        "expedited_total": "the fee, the delivery time AND whether a confirmation is needed for expedited delivery, all in one question",
        "count_options": "how many delivery options there are",
        "needs_confirmation": "which delivery option requires them to confirm a fee",
        "faster_option_fee": "what the fastest delivery option costs (without naming the option)",
        "refund": "whether they will get a refund for a charge they don't recognize",
        "fraud": "whether an unrecognized charge means it is fraud",
        "window_check": f"whether a charge from {_num(f.get('days') or 0, spoken)} days ago can still be reviewed",
        "skip_auth": "whether they can skip the verification",
        "procedure": "what the steps of this review are",
        "unknown_option": (f"{'how much' if f.get('attr') == 'fee' else 'how long'} an option called "
                           f"'{str(f.get('name') or '').replace('_', ' ')}' "
                           f"{'costs' if f.get('attr') == 'fee' else 'takes'} - use exactly that option name"),
        "unstated_fact": f"whether there is a fee to {LABEL.get(f.get('subject'), 'do that')}",
    }.get(i, str(i))


def _lexicon() -> tuple[dict[str, list[str]], dict[str, str]]:
    """(reader lexicon, labels) of the active pack."""
    from . import llm_pack as pack
    dom = pack.active()
    if dom is None:
        return READER_LEXICON, LABEL
    lex = pack.reader_lexicon(dom)
    return lex, {v: pack.label(dom, v) for v in lex}


def _slot_values(f: F) -> list:
    from . import llm_pack as pack
    if pack.active() is not None:
        return pack.slot_values(f)
    return [f.get(k) for k in ("disposition", "delivery", "card_action", "entity", "subject")]


def protected_tokens(f: F, levels: dict[str, str], ctx: dict[str, Any] | None) -> list[str]:
    """Strings the utterance must contain verbatim (decision-carrying values and names)."""
    spoken = (levels or {}).get("numeric_format") == "spoken"
    out: list[str] = []
    if f.act == "PROVIDE_CREDENTIAL":
        c = f.get("credential")
        out.append(spell_digits(c) if spoken else c)
    if f.act == "ASK_KB":
        i = f.get("intent")
        if i == "plausibility" and int(f.get("value")) != 0:
            out.append(_num(f.get("value"), spoken, money=True))
        if i == "mcq":
            out += [_num(o, spoken, money=True) for o in f.get("options")]
        if i == "window_check":
            out.append(_num(f.get("days"), spoken))
        if i == "unknown_option":
            out.append(str(f.get("name")).replace("_delivery", "").replace("_", " "))
    return out


# ----------------------------------------------------------------------------- factor instructions
_FACTOR_TEXT = {
    ("clarity", "indirect"): "Make the decision clear but say it indirectly: a short lead-in such as 'let me "
                             "think' or a supporting remark, while the decision itself stays unambiguous.",
    ("clarity", "downplayed"): "State the decision clearly but frame it as a throwaway aside, as if it hardly matters "
                               "('it's probably nothing, but ...').",
    ("numeric_format", "spoken"): "Say every number in words, never digits (card digits one by one, e.g. 'four eight "
                                  "two one'; money as words, e.g. 'fifteen dollars').",
    ("disfluency", "fillers"): "Add natural spoken fillers (um, uh, you know, like) in two or three places, never "
                               "inside a value or a name.",
    ("disfluency", "restart"): "Start the sentence, break off with a dash, say 'sorry' and restart it.",
    ("self_correction", "false_start"): "Begin with a brief false start that you immediately repair ('No - sorry, "
                                        "yes, ...'); the LAST clear answer must be the intended meaning.",
    ("paraphrase_depth", "none"): "Keep the base wording almost word for word.",
    ("paraphrase_depth", "light"): "Rephrase lightly: change a few words, keep the sentence structure.",
    ("paraphrase_depth", "deep"): "Rephrase thoroughly: different structure and vocabulary, exactly the same meaning.",
    ("question_format", "declarative"): "Phrase the question as a statement ('I want to know ...'), with no question mark.",
    ("question_format", "imperative"): "Phrase the question as a request ('Tell me ...'), with no question mark.",
    ("style", "casual"): "Casual register: contractions, 'hey', 'yeah'.",
    ("style", "terse"): "Be extremely brief: one to five words, no full sentence.",
    ("emotional_frame", "urgent"): "Sound urgent and worried.",
    ("emotional_frame", "skeptical"): "Sound sceptical that this call will help.",
}
WRAPPER_FACTORS = ("history", "instruction_conflict", "answer_anchoring", "confidence_pressure",
                   "emotional_frame", "length")
_KB_ONLY = {"question_format"}
_FALSE_START_ACTS = {"PROVIDE_CREDENTIAL", "INFORM_DISPOSITION", "CONFIRM"}


def _aside(f: F, factor: str, level: str, ctx: dict[str, Any] | None) -> str:
    """The template arm's aside for one factor level, as content the LLM paraphrases."""
    R = Realizer()
    lv = {**BASE_LEVELS, factor: level}
    core = R._core(f, lv, ctx or {})
    wrapped = R._wrap(f, core, lv, random.Random(0))
    if core in wrapped:
        pre, post = wrapped.split(core, 1)
        return " ".join(x.strip() for x in (pre, post) if x.strip())
    return ""


def factor_instructions(f: F, levels: dict[str, str], ctx: dict[str, Any] | None,
                        aliases_for: dict[str, list[str]] | None = None, asides: str = "paraphrase") -> list[str]:
    """One instruction per non-base factor level that applies to this turn.

    asides="verbatim": the wrapper factors (history, instruction conflict, anchoring, pressure,
    emotion, length) are NOT given to the model - the code adds the template arm's exact aside
    sentences around the model's line afterwards, so a paraphrased aside can never smuggle in a
    competing decision cue (the commonest cause of rejection). The model writes the decision."""
    out: list[str] = []
    lv = {**BASE_LEVELS, **(levels or {})}
    for k, v in lv.items():
        if BASE_LEVELS.get(k) == v or k == "noise":          # noise is applied deterministically after the LLM
            continue
        if k in _KB_ONLY and f.act not in ("ASK_KB", "REQUEST_HUMAN"):
            continue
        if k == "self_correction" and (f.act not in _FALSE_START_ACTS or f.get("corrected_from")):
            continue
        if k == "entity_aliasing":
            names = []
            lex, lab = _lexicon()
            for val in _slot_values(f):
                if val in lex:
                    names.append(f"{lab.get(val, val)} -> one of {lex[val][1:]}")
            if names:
                out.append("Do not use the official names; use an everyday alternative instead: " + "; ".join(names) + ".")
            continue
        if k == "specificity" and v == "specific":
            if f.act == "ASK_KB":
                out.append("Make clear the question is about a replacement card.")
            elif (ctx or {}).get("merchant") and f.act in ("INFORM_DISPOSITION", "PROVIDE_CREDENTIAL"):
                out.append(f"Name the charge concretely: the ${ctx['amount']} {ctx['merchant']} charge "
                           f"(or, for card digits, 'the card I'm calling about').")
            continue
        if k in WRAPPER_FACTORS:
            if asides == "verbatim":
                continue                                  # added verbatim by the realizer
            a = _aside(f, k, v, ctx)
            tone = _FACTOR_TEXT.get((k, v))
            if a:
                out.append(f"Include, in your own words, this aside: \"{a}\"" + (f" {tone}" if tone else ""))
            elif tone:
                out.append(tone)
            continue
        t = _FACTOR_TEXT.get((k, v))
        if t:
            out.append(t)
    return out


# ----------------------------------------------------------------------------- rephraser
REPHRASE_SYSTEM = """You write one line that a bank customer says on a phone call with an automated voice agent.
You receive: the agent's question, what the customer MEANS, the customer's plain base wording, and presentation
instructions. Write exactly one customer utterance that:
1. means exactly what the meaning says - the same decision and the same values; nothing added, nothing dropped,
   no second request, no answer to your own question;
2. follows every presentation instruction;
3. contains every 'keep verbatim' token exactly as given;
4. contains no number, amount, date or merchant that is not already in the base wording or the instructions;
5. names options, actions and decisions only with the words listed as understood for them, and names
   nothing else (no other option, no other action, no greeting or thanks unless an instruction asks);
6. is plain spoken English, as a real caller would say it on the phone: no quotation marks, no speaker
   labels, no stage directions, no emojis, no explanation of what you changed.
If the presentation instructions pull against the meaning, the meaning wins.
Reply with JSON only: {"utterance": "..."}"""


def build_rephrase_prompt(f: F, levels: dict[str, str], ctx: dict[str, Any] | None, state: str | None,
                          base_text: str, feedback: str | None = None,
                          asides: str = "paraphrase") -> tuple[str, str]:
    """The system and user prompts asking the LLM to reword one customer line: the meaning to keep, the wording
    factors to apply, the tokens to keep as written and feedback from a failed attempt.
    """
    spoken = (levels or {}).get("numeric_format") == "spoken"
    instr = factor_instructions(f, levels, ctx, asides=asides) or ["None: keep it plain, short and natural."]
    if asides == "verbatim":
        instr = instr + ["Write only the customer's answer itself - no small talk, no story, no remarks about other "
                         "people or other calls (those are added separately)."]
    keep = protected_tokens(f, levels, ctx)
    # only the names for what this turn means: offering every name in the lexicon invites the model
    # to mention options, actions or decisions the customer is not talking about (a competing cue)
    lex, lab = _lexicon()
    vals = _slot_values(f)
    if f.act in lex:
        vals.append(f.act)
    understood = {lab.get(v, v).strip("'"): lex[v] for v in vals if v in lex}
    user = {
        "agent_question": state_question(state, ctx),
        "customer_means": describe_meaning(f, spoken),
        "base_wording": base_text,
        "presentation_instructions": instr,
        "keep_verbatim": keep,
        "understood_names": understood,
    }
    if feedback:
        user["previous_attempt_rejected_because"] = feedback
    return REPHRASE_SYSTEM, json.dumps(user, ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------- independent reader
def extraction_schema(store) -> dict[str, Any]:
    """The closed meaning-card schema the independent reader must fill (from the ontology)."""
    from . import llm_pack as pack
    dom = pack.active()
    if dom is not None:
        return pack.extraction_schema(dom)
    from ..s02_graph.triples import ACT_SCHEMA, EXTRA_VALUE_SETS
    vs = dict(store.value_sets)
    vs.update(EXTRA_VALUE_SETS)
    if not vs.get("limit"):
        # the limit keys a policy question can ask about; without them the schema showed the bare set name
        # "limit" and the reader wrote key="limit" (a frequent false rejection)
        kg = getattr(store, "kg", None)
        keys = sorted(getattr(kg, "limits", {}) or {}) if kg is not None and isinstance(getattr(kg, "limits", None), dict) else []
        vs["limit"] = keys or ["max_transactions_per_session", "lookback_days", "max_auth_attempts"]
    acts = {}
    for act, slots in ACT_SCHEMA.items():
        acts[act] = {s: {"values": vs.get(v) if vs.get(v) else v, "required": req} for s, (v, req) in slots.items()}
    return acts


EXTRACT_SYSTEM = """You read one customer utterance from a bank phone call and report what the customer MEANS as a
meaning card. Use only the acts, slots and values in the schema. Rules:
- Report the customer's own final decision. Ignore what they say other people told them.
- If they correct themselves ('no - wait, actually yes'), the final answer counts; put the earlier one in corrected_from.
- A bare yes is CONFIRM and a bare no is DENY (the agent's question decides what they mean).
- Fillers, small talk, emotions and asides carry no meaning.
- For a question (ASK_KB) choose the intent that matches; numbers as digits; an option the schema does not list goes
  in slot "name" as snake_case ending in _delivery (intent unknown_option).
- For ASK_KB fill ONLY the slots listed for the chosen intent; a card or option the caller merely mentions in
  passing is not part of the question.
- If no decision can be recovered, use UNCLEAR.
Reply with JSON only: {"act": "...", "slots": {...}}"""


def build_extract_prompt(text: str, state: str | None, ctx: dict[str, Any] | None, schema: dict[str, Any]) -> tuple[str, str]:
    """The prompts asking an independent LLM to read a customer line back into a meaning card (the dual read)."""
    intents = {
        "value": "fee or delivery time of a named option (attr fee|days, entity)",
        "limit": "a session limit (key)", "plausibility": "is <option> <price>? (entity, value as a string number; 0 = free)",
        "mcq": "which of several listed fees (entity, options in the order said)",
        "faster": "which option is faster", "cheaper": "which option is cheaper",
        "fee_difference": "how much more expedited costs than standard",
        "expedited_total": "fee + time + confirmation of expedited in one question", "count_options": "how many options",
        "needs_confirmation": "which option needs a fee confirmation", "faster_option_fee": "the price of the fastest option",
        "refund": "refund for an unrecognized charge", "fraud": "does unrecognized mean fraud",
        "window_check": "can a charge from N days ago be reviewed (days)", "skip_auth": "can verification be skipped",
        "procedure": "the steps of the review", "unknown_option": "fee/time of an option the schema does not list (attr, name)",
        "unstated_fact": "is there a fee to secure the card (attr fee, subject)",
    }
    intents = {k: v + (f" | slots: {', '.join(KB_INTENT_SLOTS[k])}" if k in KB_INTENT_SLOTS else " | no other slots")
               for k, v in intents.items()}
    from . import llm_pack as pack
    if pack.active() is not None:
        intents = pack.kb_intents_text(pack.active())
    user = {"agent_question": state_question(state, ctx), "utterance": text, "schema": schema, "kb_intents": intents}
    return EXTRACT_SYSTEM, json.dumps(user, ensure_ascii=False, indent=1, default=str)


# ----------------------------------------------------------------------------- adversary
ADVERSARY_SYSTEM = """You help test a meaning checker. Rewrite the customer utterance so that its MEANING changes to
the target meaning, while keeping the style, fillers, asides and length as close to the original as possible.
Change only what is needed for the new meaning. Reply with JSON only: {"utterance": "..."}"""


def build_adversary_prompt(text: str, target: F, state: str | None, ctx: dict[str, Any] | None,
                           spoken: bool = False) -> tuple[str, str]:
    """The prompts asking an LLM to argue that a line could mean something other than its target meaning."""
    user = {"agent_question": state_question(state, ctx), "original_utterance": text,
            "target_meaning": describe_meaning(target, spoken)}
    return ADVERSARY_SYSTEM, json.dumps(user, ensure_ascii=False, indent=1)


# =====================================================================================================================
# PART 2 - LLM realizer: meaning card + factor levels -> customer utterance, written by the LLM (Microsoft gateway).
#
# Same interface as the template ``Realizer`` (``realize(frame, levels, seed, ctx)``), so the
# journey runner drives either arm unchanged. Three layers, mirroring the template arm:
#
# 1. core + asides  written by the LLM from the meaning card, the base wording and the
#                   factor instructions (prompts.py); it never sees the expected outcome;
# 2. surface noise  typo / speech-recognition noise applied *deterministically* afterwards
#                   with the same operators and protected tokens as the template arm (an
#                   LLM is a poor and unrepeatable source of realistic typos);
# 3. checks         G0 form and G1 facts on the raw LLM text (gates.py).
#
# ``generate`` is what the invariance validator calls, attempt by attempt, with feedback
# on the gate that failed. Accepted utterances are *pinned* (``pin``): at run time the
# runner gets exactly the text that was proved, keyed by the scenario seed. A turn the
# proof never saw (the bot left the oracle's path) is generated live and must pass the
# same gates and the reader check in its state; otherwise it falls back to the template
# arm and is counted, so a run never silently uses an unchecked utterance.
# =====================================================================================================================

from ..s04_cases.meaning import Ontology
from .wording import asr_noise, typo
from .wording import Lexicon
from ..llm import CachedLLM, LLMUnavailable, parse_json_object


def _clean(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
    t = re.sub(r"\s+", " ", t)
    if len(t) >= 2 and t[0] in "\"'“‘" and t[-1] in "\"'”’":
        t = t[1:-1].strip()
    return t


@dataclass
class LLMTurn:
    """One LLM-written turn: its seed, raw and final text, the template and base texts, the parsed reply, the facts
    it claims, and whether it fell back to the template.
    """
    seed: str
    raw: str = ""
    final: str = ""
    base_text: str = ""
    template_text: str = ""
    json_reply: bool = True
    form: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    error: str | None = None
    fallback: bool = False

    @property
    def text_ok(self) -> bool:
        return not self.error and self.form.get("ok", False) and self.facts.get("ok", False)

    def reasons(self) -> list[str]:
        return ([self.error] if self.error else []) + self.form.get("reasons", []) + self.facts.get("reasons", [])


class LLMRealizer:
    """Rewords customer turns with the LLM: each line is generated, passed through the output gates and read back;
    a failed line is retried, and falls back to the template wording when it keeps failing.
    """
    def __init__(self, llm: CachedLLM, store, reader, template: Realizer | None = None,
                 fallback: str = "template", runtime_attempts: int = 2, asides: str = "verbatim"):
        self.llm, self.store, self.reader = llm, store, reader
        self.asides_mode = asides                 # verbatim: template asides added around the LLM line
        self.template = template or Realizer()
        self.gates = LLMOutputGates(store, reader)
        self.ont = Ontology.from_store(store)
        self.fallback = fallback
        self.runtime_attempts = runtime_attempts
        self.bank: dict[str, str] = {}
        self.stats = {"generated": 0, "pinned_hits": 0, "runtime_generated": 0, "runtime_accepted": 0,
                      "runtime_fallbacks": 0, "non_json_replies": 0}

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _key(seed: str, frame: F, levels: dict[str, str], state: str | None) -> str:
        lv = {k: v for k, v in sorted((levels or {}).items())}
        return hashlib.sha256(f"{seed}|{frame.canonical()}|{state}|{lv}".encode()).hexdigest()[:24]

    def pin(self, seed: str, frame: F, levels: dict[str, str], state: str | None, text: str) -> None:
        self.bank[self._key(seed, frame, levels, state)] = text

    # ------------------------------------------------------------------ generation
    def asides(self, frame: F, levels: dict[str, str], ctx: dict[str, Any] | None) -> dict[str, str]:
        lv = {**BASE_LEVELS, **(levels or {})}
        return {k: _aside(frame, k, lv[k], ctx) for k in WRAPPER_FACTORS if lv[k] != BASE_LEVELS[k]}

    def prompt(self, frame: F, levels: dict[str, str], seed: str, ctx: dict[str, Any] | None,
               state: str | None, feedback: str | None = None) -> tuple[str, str, str, str]:
        """(system, user, base_text, template_text) exactly as ``generate`` sends them - so a prefetch
        can issue the same calls ahead of time and ``generate`` then finds them in the cache."""
        ctx = {k: v for k, v in (ctx or {}).items() if k in ("merchant", "amount")}
        base_text = self.template.realize(frame, {}, "base", ctx)
        tmpl_text = self.template.realize(frame, levels, seed, ctx)
        system, user = build_rephrase_prompt(frame, levels, ctx, state, base_text, feedback, asides=self.asides_mode)
        return system, user, base_text, tmpl_text

    def generate(self, frame: F, levels: dict[str, str], seed: str, ctx: dict[str, Any] | None,
                 state: str | None, feedback: str | None = None) -> LLMTurn:
        system, user, base_text, tmpl_text = self.prompt(frame, levels, seed, ctx, state, feedback)
        ctx = {k: v for k, v in (ctx or {}).items() if k in ("merchant", "amount")}
        turn = LLMTurn(seed, base_text=base_text, template_text=tmpl_text)
        try:
            reply = self.llm.invoke(system, user, salt=seed)
        except LLMUnavailable as e:
            turn.error = f"llm unavailable: {e}"
            return turn
        self.stats["generated"] += 1
        obj = parse_json_object(reply)
        if obj is None or not isinstance(obj.get("utterance"), str):
            turn.json_reply = False
            self.stats["non_json_replies"] += 1
            raw = _clean(reply)
        else:
            raw = _clean(obj["utterance"])
        if self.asides_mode == "verbatim" and raw:
            lv = {**BASE_LEVELS, **(levels or {})}  # the template arm's exact asides around the LLM's line
            raw = self.template._wrap(frame, raw, lv, random.Random(f"{seed}|wrap"))
        turn.raw = raw
        turn.form = self.gates.form(raw).as_dict()
        turn.facts = self.gates.facts(raw, frame, ctx, levels, [base_text, tmpl_text]).as_dict()
        turn.final = self._surface(raw, frame, levels, seed, ctx)
        return turn

    def _surface(self, text: str, frame: F, levels: dict[str, str], seed: str, ctx: dict[str, Any] | None) -> str:
        n = {**BASE_LEVELS, **(levels or {})}["noise"]
        if n == "none" or not text:
            return text
        protect = set(Lexicon.current().protect)
        for tok in protected_tokens(frame, levels, ctx):
            protect.update(re.findall(r"[a-z']+", tok.lower()))
        rng = random.Random(f"{seed}|noise")
        return typo(text, rng, protect=protect) if n == "typo" else asr_noise(text, rng, protect=protect)

    # ------------------------------------------------------------------ runner interface
    def realize(self, frame: F, levels: dict[str, str], seed: str, ctx: dict[str, Any] | None = None) -> str:
        ctx = dict(ctx or {})
        state = ctx.get("state")
        k = self._key(seed, frame, levels, state)
        if k in self.bank:
            self.stats["pinned_hits"] += 1
            return self.bank[k]
        # a turn the proof never saw: generate live, under the same gates and the reader check
        feedback = None
        for a in range(self.runtime_attempts):
            s = f"{seed}|rt{a}"
            t = self.generate(frame, levels, s, ctx, state, feedback)
            if t.error:
                break
            self.stats["runtime_generated"] += 1
            g3 = self.gates.one_clear_decision(t.final, frame, levels)
            same = self.ont.equivalent(frame, self.reader.read(t.final, state), state)
            if t.text_ok and g3.ok and same:
                self.stats["runtime_accepted"] += 1
                self.bank[k] = t.final
                return t.final
            feedback = "; ".join(t.reasons() + g3.reasons + ([] if same else ["the meaning changed"]))
        if self.fallback != "template":
            raise LLMUnavailable(f"no accepted LLM utterance for {seed}")
        self.stats["runtime_fallbacks"] += 1
        return self.template.realize(frame, levels, seed, {k2: v for k2, v in ctx.items() if k2 in ("merchant", "amount")})


# =====================================================================================================================
# PART 3 - A second, independent reader for LLM-written utterances (dual read).
#
# The rule reader (s05_rewording/wording.py) is the primary instrument of the proof. For the LLM
# arm a second reader can be required to agree: an LLM call that sees only the utterance
# and the agent's question - never the base meaning, the base wording or the factor
# levels - and must fill the closed meaning-card schema of the ontology. Because it is
# blind to the intended meaning, agreement between the two readers is evidence about the
# text, not about the prompt that produced it.
#
# It can only make the gate stricter: with ``dual_read="gate"`` a turn is accepted when
# the rule reader AND this reader both recover the base card; with ``"report"`` the
# agreement is measured and reported but does not gate.
# =====================================================================================================================



def _coerce(slot: str, v: Any) -> Any:
    if v is None or v == "":
        return None
    if slot == "credential":
        d = re.sub(r"\D", "", str(v))
        return d or None
    if slot == "value":
        try:
            return str(int(float(str(v).replace("$", ""))))
        except ValueError:
            return None
    if slot == "days":
        try:
            return int(float(str(v)))
        except ValueError:
            return None
    if slot == "options":
        vals = v if isinstance(v, (list, tuple)) else [v]
        out = []
        for x in vals:
            try:
                out.append(int(float(str(x).replace("$", ""))))
            except ValueError:
                return None
        return tuple(out)
    if slot == "name":
        from . import llm_pack as pack
        s = re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_")
        if pack.active() is not None:
            return s
        return s if s.endswith("_delivery") else f"{s}_delivery"
    return str(v)


# The slots each policy-question intent carries (the meaning cards the bases are built from). The reader
# may volunteer more ("subject=REPLACE_CARD" on a window check because the caller mentioned their card);
# those are not part of the question's meaning and are dropped, so they cannot fail the comparison.
KB_INTENT_SLOTS = {
    "value": ("attr", "entity"), "limit": ("key",), "plausibility": ("attr", "entity", "value"),
    "mcq": ("attr", "entity", "options"), "window_check": ("days",), "unknown_option": ("attr", "name"),
    "unstated_fact": ("attr", "subject"),
}
KB_DEFAULT_ATTR = {"plausibility": "fee", "mcq": "fee", "unstated_fact": "fee"}   # the only reading these have


class LLMFrameExtractor:
    """The dual read: an LLM reads a customer line back into a meaning card, constrained to the pack's acts and slots.
    """
    def __init__(self, llm: CachedLLM, store):
        from ..s02_graph.triples import ACT_SCHEMA
        from . import llm_pack as pack
        self.llm, self.store = llm, store
        self.schema = extraction_schema(store)
        self.acts = pack.act_schema(pack.active()) if pack.active() is not None else ACT_SCHEMA
        self.calls = 0
        self.failures = 0

    def read(self, text: str, state: str | None, ctx: dict[str, Any] | None = None,
             strict: bool = False) -> F | None:
        """The meaning card the independent reader recovers, or None (unavailable / unparsable).

        strict=True re-raises LLMUnavailable instead of returning None, so a *network* failure is
        never mistaken for "the reader recovered a different meaning"."""
        system, user = build_extract_prompt(text, state, ctx, self.schema)
        self.calls += 1
        try:
            obj = self.llm.invoke_json(system, user, salt="extract")
        except LLMUnavailable:
            self.failures += 1
            if strict:
                raise
            return None
        if not obj or obj.get("act") not in self.acts:
            self.failures += 1
            return None
        act = obj["act"]
        raw = obj.get("slots") or {}
        if not isinstance(raw, dict):
            raw = {}
        slots = {}
        for k, v in raw.items():
            if k not in self.acts[act]:
                continue
            c = _coerce(k, v)
            if c is not None:
                slots[k] = c
        if act == "ASK_KB":
            intent = slots.get("intent")
            keep = ("intent",) + KB_INTENT_SLOTS.get(intent, ())
            slots = {k: v for k, v in slots.items() if k in keep}
            if intent in KB_DEFAULT_ATTR:
                slots.setdefault("attr", KB_DEFAULT_ATTR[intent])
        # a value outside the slot's value set is not a reading of that slot
        for k in list(slots):
            vs = (self.schema.get(act, {}).get(k) or {}).get("values")
            if isinstance(vs, list) and vs and isinstance(slots[k], str) and slots[k] not in vs:
                del slots[k]
        return F.make(act, **slots)


# =====================================================================================================================
# PART 4 - Per-turn gates for LLM-written customer utterances.
#
# The template arm is trusted to *construct* text from a meaning card; the LLM arm is
# not, so every LLM utterance passes the gates below before the invariance proof (the
# same L1-L4 the template arm passes) is even attempted. Gates on the raw LLM text:
#
# G0 form       one spoken line: non-empty, bounded length, no speaker labels, no
#               stage directions, no internal names (tool names, rule ids, act names,
#               'JSON', 'as an AI') - a leak means the model saw or invented internals;
# G1 facts      no figure, amount or merchant that the base wording, the scenario's
#               template wording or the meaning card does not already carry (no
#               hallucinated content, no answering its own question), and every
#               'keep verbatim' token present;
#
# and on the final utterance (after the deterministic surface-noise layer):
#
# G2 meaning    (in validate.py) the rule reader's card in the dialog state equals the
#               base card and is admissible there; optionally a second, independent
#               LLM reader that never saw the base must agree;
# G3 one clear  the utterance carries no more competing decision cues than the
#    decision   requested presentation allows (a self-correction may carry two; nothing
#               else may) - an utterance a careful agent could read two ways is not the
#               same test even if our reader happens to pick the right reading.
#
# ``factor_realisation`` is a manipulation check, not a gate: it reports whether each
# requested factor level is visible in the text, so attribution can use the levels that
# actually reached the bot (an unrealised factor cannot cause a failure).
# =====================================================================================================================

from ..s04_cases.journey import MERCHANTS

MAX_CHARS = 600
MAX_WORDS = 90
_LEAK = re.compile(
    r"\b(authenticate_customer|get_review_transactions|record_charge_response|get_card_action_options|"
    r"submit_card_action|transfer_to_human|inform_disposition|provide_credential|request_card_action|ask_kb|"
    r"meaning card|json|utterance|as an ai|language model)\b", re.I)
_RULE_ID = re.compile(r"\bR\d{1,2}\b")
_STATUS = re.compile(r"\b(SUCCESS|AUTHENTICATED|FAILED|UNAVAILABLE)\b")
_SPEAKER = re.compile(r"(^|\n)\s*(customer|agent|bot|user|assistant|caller)\s*:", re.I)
_STAGE = re.compile(r"[\[\(]\s*(laughs|sighs|pause|whispers|coughs|clears throat)[^\]\)]*[\]\)]", re.I)
_FILLER = re.compile(r"\b(um+|uh+|erm|hmm+|you know|like,|i mean)\b", re.I)
_REPAIR = re.compile(r"(—|–|--|\.\.\.|\bwait\b|\bactually\b|\bi mean\b|\bsorry\b|\bno,? no\b)", re.I)
_URGENT = re.compile(r"\b(hurry|urgent|urgently|quickly|asap|right away|worried|panick|scared|please,? fast)\b", re.I)
_SKEPTIC = re.compile(r"\b(doubt|skeptic|sceptic|not sure (this|you) (will|can) help|won't help|waste of time|"
                      r"really\?|seriously)\b", re.I)
_CASUAL = re.compile(r"\b(hey|hi|yeah|yep|yup|gonna|wanna|kinda|cool|ok so|okay so)\b", re.I)
_INDIRECT = re.compile(r"\b(i suppose|i guess|i think|let me (check|think|see)|thinking (about|it over)|i was wondering|"
                       r"if i remember|as far as i)\b", re.I)
_DOWNPLAY = re.compile(r"\b(doesn'?t (really )?matter|just for the record|probably nothing|small thing|no big deal|"
                       r"just (a|one) (quick|small))\b", re.I)


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (t or "").lower())


def _content(t: str) -> set[str]:
    stop = {"the", "a", "an", "i", "you", "it", "is", "to", "and", "of", "for", "my", "me", "that", "this", "on",
            "in", "be", "so", "do", "was", "are", "with", "your", "just", "please", "can", "will", "what", "it's"}
    return {w for w in _words(t) if w not in stop and len(w) > 2}


def _recall(base: str, text: str) -> float:
    b = _content(base)
    return len(b & _content(text)) / len(b) if b else 1.0


@dataclass
class GateReport:
    """Whether a line passed the output gates, and the reasons it did not."""
    ok: bool
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons)}


class LLMOutputGates:
    """The checks every LLM line must pass: no new figures, no tool or merchant names it was not given, the protected
    tokens kept, nothing the agent says, and the rule reader reads back the target meaning.
    """
    def __init__(self, store, reader):
        self.store, self.reader = store, reader
        self.tools = {t for t, ty in store.entity_types.items() if ty == "tool"}
        self.merchants = {m.lower() for m, _ in MERCHANTS}
        from . import llm_pack as pack
        self.pack = pack.active()

    # ------------------------------------------------------------------ helpers
    def figures(self, text: str) -> set[str]:
        """Numbers a careful listener would hear, normalised like the reader does (spoken -> digits)."""
        t = self.reader.normalize(text or "").replace("<dot>", ".")
        out = set()
        for m in re.finditer(r"\d+(?:\.\d+)?", t):
            v = m.group(0)
            out.add(str(int(float(v))) if float(v).is_integer() else v)
            if "." in v:                                   # $45.67 also licenses 45 and 67 spoken separately
                a, b = v.split(".")
                out.update({str(int(a)), str(int(b))})
        return out

    # ------------------------------------------------------------------ G0
    def form(self, text: str) -> GateReport:
        r = []
        t = (text or "").strip()
        if not t:
            r.append("empty utterance")
        if len(t) > MAX_CHARS or len(_words(t)) > MAX_WORDS:
            r.append(f"too long ({len(_words(t))} words; max {MAX_WORDS})")
        if _SPEAKER.search(t):
            r.append("contains a speaker label")
        if "\n" in t.strip():
            r.append("more than one line")
        if _STAGE.search(t):
            r.append("contains a stage direction")
        if _LEAK.search(t) or any(tool in t for tool in self.tools):
            r.append("mentions an internal name")
        if _RULE_ID.search(t) or _STATUS.search(t):
            r.append("mentions a rule id or a tool status")
        return GateReport(not r, r)

    # ------------------------------------------------------------------ G1
    def facts(self, text: str, frame: F, ctx: dict[str, Any] | None, levels: dict[str, str],
              licensed_texts: list[str]) -> GateReport:
        r = []
        allowed = set()
        for s in licensed_texts:
            allowed |= self.figures(s)
        for k in ("credential", "value", "days"):
            if frame.get(k) is not None:
                allowed |= self.figures(str(frame.get(k)))
        for o in frame.get("options") or ():
            allowed |= self.figures(str(o))
        if ctx and ctx.get("amount"):
            allowed |= self.figures(f"${ctx['amount']}")
        new = sorted(self.figures(text) - allowed, key=lambda x: (len(x), x))
        if new:
            r.append(f"introduces figures not in the base meaning: {', '.join(new)}")
        low = (text or "").lower()
        own = str((ctx or {}).get("merchant") or "").lower()
        foreign = sorted(m for m in self.merchants if m != own and re.search(rf"\b{re.escape(m)}\b", low))
        if foreign:
            r.append(f"mentions another merchant: {', '.join(foreign)}")
        nt = self.reader.normalize(text or "")
        for tok in protected_tokens(frame, levels, ctx):
            if self.reader.normalize(tok) not in nt and tok.lower() not in low:
                r.append(f"drops the value '{tok}'")
        return GateReport(not r, r)

    # ------------------------------------------------------------------ G3
    def one_clear_decision(self, text: str, frame: F, levels: dict[str, str]) -> GateReport:
        if self.pack is not None:
            from . import llm_pack as pack
            r = pack.gates_one_decision(self.reader, text, frame, levels)
            return GateReport(not r, r)
        cues = self.reader.cue_counts(text)
        lv = {**BASE_LEVELS, **(levels or {})}
        r = []
        if frame.act == "INFORM_DISPOSITION":
            allowed = 2 if (frame.get("corrected_from") or lv["self_correction"] != "none") else 1
            if cues["disposition_values"] > allowed:          # different answers, not the same one said twice
                r.append(f"{cues['disposition_values']} competing disposition cues (allowed {allowed})")
        elif cues["disposition"] and frame.act in ("CONFIRM", "DENY", "REQUEST_CARD_ACTION", "END", "REQUEST_HUMAN",
                                                   "PROVIDE_CREDENTIAL"):
            r.append("carries a disposition cue the meaning does not have")
        if frame.get("delivery") or frame.get("entity"):
            if cues["delivery_values"] > 1 and lv["self_correction"] == "none":
                r.append(f"{cues['delivery_values']} delivery cues (one option is meant)")
        return GateReport(not r, r)

    # ------------------------------------------------------------------ manipulation check
    def factor_realisation(self, text: str, base_text: str, frame: F, levels: dict[str, str],
                           ctx: dict[str, Any] | None, asides: dict[str, str] | None = None) -> dict[str, dict]:
        """For every non-base level: detected True/False, or None when the factor cannot apply to this turn."""
        lv = {**BASE_LEVELS, **(levels or {})}
        asides = asides or {}
        t = text or ""
        low = t.lower()
        nb = len(_words(base_text)) or 1
        nw = len(_words(t))
        out: dict[str, dict] = {}

        def put(k, det, ev=""):
            out[k] = {"level": lv[k], "detected": det, "evidence": ev}

        for k, v in lv.items():
            if BASE_LEVELS.get(k) == v:
                continue
            if k == "noise":
                put(k, True, "applied deterministically after the LLM")
            elif k == "clarity":
                if frame.act in ("END", "UNCLEAR", "DENY") or lv["style"] == "terse":
                    put(k, None, "not applicable to this act/style")
                else:
                    rx = _INDIRECT if v == "indirect" else _DOWNPLAY
                    put(k, bool(rx.search(t)), "lead-in/aside marker")
            elif k == "entity_aliasing":
                lex, _ = _lexicon()
                vals = [x for x in _slot_values(frame) if x in lex]
                if not vals:
                    put(k, None, "no aliasable value in this turn")
                else:
                    canon = [lex[x][0].lower() for x in vals]
                    put(k, not any(c in low for c in canon), "official name avoided")
            elif k == "specificity":
                if frame.act == "ASK_KB":
                    put(k, bool(re.search(r"replacement|new card", low)), "replacement card named")
                elif (ctx or {}).get("merchant") and frame.act in ("INFORM_DISPOSITION", "PROVIDE_CREDENTIAL"):
                    put(k, str(ctx["merchant"]).lower() in low or str(ctx.get("amount")) in t
                        or "calling about" in low, "merchant/amount named")
                else:
                    put(k, None, "nothing specific to name")
            elif k == "numeric_format":
                has_num = bool(self.figures(base_text))
                put(k, (not re.search(r"\d", t)) if has_num else None, "numbers in words")
            elif k == "disfluency":
                put(k, bool(_FILLER.search(t)) if v == "fillers" else bool(_REPAIR.search(t)), "fillers/restart")
            elif k == "self_correction":
                if frame.act not in ("PROVIDE_CREDENTIAL", "INFORM_DISPOSITION", "CONFIRM") or frame.get("corrected_from"):
                    put(k, None, "not applicable to this act")
                else:
                    put(k, bool(_REPAIR.search(t)), "repair marker")
            elif k == "paraphrase_depth":
                rc = _recall(base_text, t)
                det = rc >= 0.75 if v == "none" else (0.3 <= rc < 0.95 if v == "light" else rc < 0.65)
                put(k, det, f"content-word recall vs base {rc:.2f}")
            elif k == "question_format":
                if frame.act not in ("ASK_KB", "REQUEST_HUMAN") or lv["style"] == "terse":
                    put(k, None, "not a question")
                elif v == "declarative":
                    put(k, "?" not in t, "no question mark")
                else:
                    put(k, "?" not in t and bool(re.search(r"\b(tell|give|let me know|explain|transfer|connect|put me)\b", low)),
                        "request form")
            elif k == "style":
                if v == "terse":
                    put(k, nw <= 6, f"{nw} words")
                elif v == "casual":
                    put(k, bool(_CASUAL.search(t)) or "'" in t, "casual markers")
                else:
                    put(k, None, "")
            elif k == "emotional_frame":
                rx = _URGENT if v == "urgent" else _SKEPTIC
                a = asides.get(k, "")
                put(k, bool(rx.search(t)) or (bool(a) and _recall(a, t) >= 0.4), "tone/aside")
            elif k == "length":
                put(k, nw >= (1.3 if v == "medium" else 1.8) * nb, f"{nw} vs base {nb} words")
            else:                                                     # history, instruction_conflict, anchoring, pressure
                a = asides.get(k, "")
                if not a:
                    put(k, None, "no aside for this act")
                else:
                    put(k, _recall(a, t) >= 0.4, f"aside content recall {_recall(a, t):.2f}")
        return out


# =====================================================================================================================
# PART 5 - CSV exports for the LLM enrichment arm (one row per customer turn, controls audit, arm comparison).
# =====================================================================================================================

def _w(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return 0
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})
    return len(rows)


def write_llm_exports(out_dir: str | Path, proofs, controls: dict[str, Any] | None,
                      paired: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write the LLM arm's export files (per-turn records, gate and control results, paired comparisons) into a
    run folder and return their summary.
    """
    out = Path(out_dir)
    turns = []
    for p in proofs:
        for t in p.turns:
            fr = t.get("factor_realisation") or {}
            turns.append({
                "sid": p.sid, "base": p.qid, "scenario_accepted": p.accepted, "turn": t["turn"], "state": t["state"],
                "meaning_card": t["frame"], "base_text": t.get("base_text"), "template_text": t.get("template_text"),
                "llm_raw": t.get("llm_raw"), "llm_final": t.get("enriched_text"),
                "rule_reader_card": t.get("read_back"), "independent_reader_card": t.get("dual_read"),
                "turn_accepted": t["equivalent"], **{f"gate_{k}": v for k, v in (t.get("gates") or {}).items()},
                "reasons": t.get("reasons"), "attempts": len(t.get("attempts", [])),
                # llm | template_fallback (the LLM failed every attempt; the proved template turn is used)
                "turn_source": t.get("turn_source", "llm"),
                "llm_rejected_reasons": t.get("llm_rejected_reasons"),
                # every attempt: what the model wrote and why it was rejected (or the transport error)
                "attempt_log": [{"attempt": a.get("attempt"), "text": a.get("final") or a.get("raw"),
                                 "rule_read": a.get("read"), "independent_read": a.get("dual_read"),
                                 "failed": [k for k in ("G0_form", "G1_facts", "G3_one_decision", "G2_reader",
                                                        "G2_dual_read") if a.get(k) is False],
                                 "reasons": a.get("reasons"), "error": a.get("error")}
                                for a in t.get("attempts", [])],
                "surface_cosine_to_base": round(t.get("surface_cosine", 0.0), 4),
                "surface_cosine_to_template": round(t.get("template_cosine", 0.0), 4) if "template_cosine" in t else None,
                "factors_requested": {k: v["level"] for k, v in fr.items()},
                "factors_realised": sorted(k for k, v in fr.items() if v["detected"]),
                "factors_not_realised": sorted(k for k, v in fr.items() if v["detected"] is False),
                "levels": p.levels, "realised_levels": p.extra.get("realised_levels"),
            })
    res = {"llm_turns": _w(out / "llm_enrichment_turns.csv", turns)}
    if controls:
        res["llm_controls_audit"] = _w(out / "llm_negative_controls_audit.csv", controls.get("_audit", []))
    if paired:
        res["arm_pairs"] = _w(out / "arm_pairs.csv", paired)
    return res
