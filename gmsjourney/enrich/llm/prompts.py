"""Prompts for the LLM enrichment arm (rephraser, independent reader, adversary).

What the rephraser is allowed to see, by design ("blind generation"):

* the bot's question for the dialog state the turn answers (generic, from ``STATE_QUESTIONS``);
* what the customer means, in plain words (``describe_meaning`` of the base meaning card);
* the customer's base wording (the template realisation at all-base levels);
* the presentation instructions for the scenario's factor levels (``factor_instructions``);
* the tokens it must keep verbatim and the names the rule reader understands.

What it never sees: the expected bot behaviour, the outcome signature, the oracle trace,
tool names, rule ids, or any policy figure that is not already in the customer's own
words. A model that never sees the answer cannot leak it or bend the question toward it.

The independent reader (``EXTRACT_*``) sees only the utterance and the bot's question,
never the base meaning, and must fill a closed meaning-card schema. The adversary
(``ADVERSARY_*``) is asked to change exactly one meaning element; its rewrites are
negative controls for the LLM arm's gate stack.
"""
from __future__ import annotations

import json
import random
from typing import Any

from ...ontology import SemanticFrame as F
from ..noise import number_to_words, spell_digits
from ..realizers import BASE_LEVELS, Realizer

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

# Words the rule reader binds (enrich/reader.py patterns). The LLM may only name
# values with these words; anything else the reader would not read is a false reject.
READER_LEXICON = {
    "standard_delivery": ["standard delivery", "regular shipping", "normal delivery"],
    "expedited_delivery": ["expedited delivery", "express shipping", "rush delivery", "overnight shipping"],
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
            for slot in ("disposition", "delivery", "card_action", "entity", "subject"):
                val = f.get(slot)
                if val in READER_LEXICON:
                    names.append(f"{LABEL.get(val, val)} -> one of {READER_LEXICON[val][1:]}")
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
    spoken = (levels or {}).get("numeric_format") == "spoken"
    instr = factor_instructions(f, levels, ctx, asides=asides) or ["None: keep it plain, short and natural."]
    if asides == "verbatim":
        instr = instr + ["Write only the customer's answer itself - no small talk, no story, no remarks about other "
                         "people or other calls (those are added separately)."]
    keep = protected_tokens(f, levels, ctx)
    # only the names for what this turn means: offering every name in the lexicon invites the model
    # to mention options, actions or decisions the customer is not talking about (a competing cue)
    vals = [f.get(k) for k in ("disposition", "delivery", "card_action", "entity", "subject")]
    if f.act in READER_LEXICON:
        vals.append(f.act)
    understood = {LABEL.get(v, v).strip("'"): READER_LEXICON[v] for v in vals if v in READER_LEXICON}
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
    from ...knowledge.policy_kg import ACT_SCHEMA, EXTRA_VALUE_SETS
    vs = dict(store.value_sets)
    vs.update(EXTRA_VALUE_SETS)
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
- If no decision can be recovered, use UNCLEAR.
Reply with JSON only: {"act": "...", "slots": {...}}"""


def build_extract_prompt(text: str, state: str | None, ctx: dict[str, Any] | None, schema: dict[str, Any]) -> tuple[str, str]:
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
    user = {"agent_question": state_question(state, ctx), "utterance": text, "schema": schema, "kb_intents": intents}
    return EXTRACT_SYSTEM, json.dumps(user, ensure_ascii=False, indent=1, default=str)


# ----------------------------------------------------------------------------- adversary
ADVERSARY_SYSTEM = """You help test a meaning checker. Rewrite the customer utterance so that its MEANING changes to
the target meaning, while keeping the style, fillers, asides and length as close to the original as possible.
Change only what is needed for the new meaning. Reply with JSON only: {"utterance": "..."}"""


def build_adversary_prompt(text: str, target: F, state: str | None, ctx: dict[str, Any] | None,
                           spoken: bool = False) -> tuple[str, str]:
    user = {"agent_question": state_question(state, ctx), "original_utterance": text,
            "target_meaning": describe_meaning(target, spoken)}
    return ADVERSARY_SYSTEM, json.dumps(user, ensure_ascii=False, indent=1)
