"""Block 5 - the words: customer turns, correct and erroneous bot responses, step summaries.

Two renderers with one interface:
  LLMRenderer       gpt-oss on vLLM or the Amex gateway, through gmsjourney.llm (cache, pacer, circuit breaker).
                    Every call asks for a JSON object; a reply that is not usable is retried with a new salt.
                    Optional quality gate (``verify: true``): an LLM judge checks a correct response against its
                    HRTs and an erroneous one for its intended defect; failures are regenerated (``max_tries``).
  TemplateRenderer  deterministic text from the HRTs (dry runs, tests, CI) - no model needed.

The prompts carry only what the speaker may know: the customer sees the conversation and their own goal; the bot
writer sees the AOP box text, the HRTs that apply, the data values (metadata / transaction), the conversation and
the customer's latest words.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .scenario import HANDOFF_STATE, OPENING

PROMPT_VERSION = "turnval-text/1"


# ---------------------------------------------------------------------------------------------------- facts
def facts_for(gold_turn, profile: dict[str, Any]) -> dict[str, Any]:
    """The values the bot may speak on this turn."""
    ctx = gold_turn.context
    card = ctx.get("card") or {}
    md = ctx.get("metadata") or {}
    f = {"first_name": profile["first_name"], "preferred_first_name": profile["preferred_first_name"],
         "last5AccountNumber": card.get("last5AccountNumber") or profile.get("cm5"),
         "questionText": md.get("questionText") or profile.get("question"), "today": profile.get("today")}
    if ctx.get("txn"):
        f.update({"current_transaction_data": ctx["txn"], "card": card, "current_transaction_index": ctx.get("txn_index"),
                  "card_index": ctx.get("card_index"),
                  "num_transactions": len(next((c["transactions"] for c in profile.get("cards", [])
                                                if c.get("last5AccountNumber") == card.get("last5AccountNumber")), [1]))})
    return f


def fill(text: str, facts: dict[str, Any]) -> str:
    def rep(m):
        key = m.group(1).split(".")[-1]
        v = facts.get(key)
        return json.dumps(v) if isinstance(v, (dict, list)) else str(v if v is not None else "")
    return re.sub(r"\{\{\s*@?(?:metadata\.)?([A-Za-z_0-9.@]+)\s*\}\}", rep, text)


def _atom_lines(hrts: dict[str, Any], facts: dict[str, Any]) -> str:
    rows = []
    for a in hrts.get("atoms", []):
        k = a.get("kind")
        tag = {"convey": "MUST SAY", "global_must": "MUST", "forbid": "MUST NOT SAY", "global_must_not": "MUST NOT",
               "metadata": "VALUE CHECK", "guard_must": "HOW (must)", "guard_must_not": "HOW (must not)"}.get(k, k)
        grp = f" ({a['group']} group)" if a.get("group") in ("OR", "ANY") else ""
        cond = " (only if the response does this)" if a.get("response_conditional") else ""
        order = " [in order]" if a.get("in_order") else ""
        rows.append(f"- {tag}{grp}{order}{cond}: [{a.get('head')}] {fill(a.get('description') or a.get('tail', ''), facts)}"
                    + (f"  (field {a['field']})" if a.get("field") else ""))
    return "\n".join(rows)


def history_text(history: list[tuple[str, str]], last: int = 12) -> str:
    return "\n".join(f"{'Agent' if who == 'bot' else 'Customer'}: {text}" for who, text in history[-last:])


# ---------------------------------------------------------------------------------------------------- prompts
CUSTOMER_SYSTEM = """You write the next thing a caller says on an American Express Account Protection phone call with an AI agent.
The text is what speech recognition produced, so write it as spoken words. Stay in character: you are the cardmember.
Follow the INTENT exactly (what you provide, refuse, ask or correct). Never mention being simulated.
Answer with a JSON object: {"utterance": "<what the caller says>"}"""

BOT_SYSTEM = """You write the next spoken response of the American Express Account Protection AI voice agent.
Follow the AOP step and its requirements exactly: say every MUST SAY item (verbatim items word for word, others in
meaning), never say a MUST NOT item, speak data values exactly as given, respect HOW rules (pronunciation, order,
brevity). Natural, concise, spoken English; no markdown, no lists, no stage directions.
Answer with a JSON object: {"response": "<the agent's words>"}"""

ERROR_SYSTEM = """You write a realistic but FLAWED next response of the American Express Account Protection AI voice agent,
to train a validator. Write what the agent says when it makes exactly the DEFECT described - fluent and confident,
never hinting at the mistake. Apart from the defect, keep the response natural for the conversation.
Answer with a JSON object: {"response": "<the agent's words>"}"""

SUMMARY_SYSTEM = """You summarise a customer-service call for a turn validator. Write the step-level state of the call
before the customer's latest turn: which steps happened (questions asked, what the caller provided - digits masked as
'<digits>' - and the outcome), what the agent asked last and is waiting for, and anything the caller raised.
At most 120 words, plain sentences, past tense, no speculation.
Answer with a JSON object: {"summary": "..."}"""

JUDGE_SYSTEM = """You check a voice agent's response against requirements. For each numbered requirement answer
"ok" (satisfied / not violated), "violated" or "n/a" (does not apply to this response).
Answer with a JSON object: {"checks": {"1": "ok", "2": "violated", ...}, "defect_present": true|false}"""

STYLE_HINTS = {
    "terse": "Very short: a few words.", "plain": "One normal sentence.", "chatty": "Talkative: two or three sentences.",
    "canonical": "Plain standard wording.", "colloquial": "Casual everyday wording.",
    "hesitant": "Hesitant: fillers (uh, um), restarts.", "formal": "Formal and polite.",
    "clean": "", "transcript_artifacts": "Transcription artifacts: a repeated word, missing punctuation, numbers as words.",
    "calm": "", "anxious": "Sounds worried about fraud.", "frustrated": "Sounds annoyed.",
    "digits": "Say numbers as digits.", "words": "Say numbers as words (one two three).",
    "grouped": "Say numbers in groups (twelve thirty-four).", "mixed": "Mix digits and number words.",
}

INTERJECTION_HINTS = {
    "off_topic": "Instead of answering, ask about something unrelated (rewards points, a different card, pay over time).",
    "new_concern": "Raise a different concern (e.g. a lost card, an address change) instead of answering.",
    "asks_human": "Ask to speak to a real person / customer service / an agent.",
    "multi_demand": "Answer the question AND also ask for something unrelated in the same turn.",
    "clarifying_question": "Ask a clarifying question about the request (where to find it, why it is needed) instead of answering.",
    "hold_on": "Say you need a moment to find the information (hold on, give me a minute).",
    "alternative_id": "Offer a different piece of identification instead of the one asked for.",
    "procedural_question": "Ask how this step works (e.g. can I type it on the keypad) instead of answering.",
    "asks_question": "Ask a question about the charge being reviewed (where it was made, why it was flagged) instead of answering.",
    "ungrounded_question": "Ask something the agent cannot know (how the fraudster got the card number, whether the merchant is legit).",
    "next_txn_request": "Do not answer about this charge; ask to move on ('what's the next one?', 'continue').",
}


def customer_intent_text(cust: dict[str, Any], profile: dict[str, Any]) -> str:
    """The customer's goal for the turn, from the simulator's intent."""
    addr, out = cust.get("addressed"), cust.get("outcome")
    intent = cust.get("intent") or {}
    if cust.get("react_to_error"):
        base = ("React naturally to the agent's last message (it may have skipped a step, repeated itself or said "
                "something odd); you still want to get through the fraud alert.")
    elif addr == "opening":
        base = "Say yes, you are calling about a fraud alert."
    elif addr == "card_confirm":
        base = f"{'Confirm' if out else 'Say no to'} that you are calling about the card ending in {profile['cm5']}."
    elif addr in ("set_4ssn_metadata.match", "collected_4SSN"):
        base = _id_text("the last four digits of your social security number", profile["ssn4"], out)
    elif addr == "set_fullssn_metadata.match":
        base = _id_text("your full nine-digit social security number", profile["ssn9"], out)
    elif addr in ("set_cm5_metadata.match", "collected_5CM"):
        base = ("Say you don't have your card with you right now / are looking for it." if out == "SEARCHING"
                else _id_text("the last five digits of your card", profile["cm5"], out))
    elif addr == "set_cm15_metadata.match":
        base = _id_text("your fifteen-digit card number", profile["cm15"], out)
    elif addr == "collected_zipcode":
        base = _id_text("your five-digit ZIP code", profile["zip"], out)
    elif addr == "check_fullssn_or_cm15.match":
        base = {"FULL_SSN": "Choose to give your full social security number.",
                "CM_15": "Choose to give your fifteen-digit card number.",
                "NO_PREFERENCE_SHARED": "Answer vaguely without picking either option.",
                "UNKNOWN": "Say you have neither available."}.get(out, "Answer the question.")
    elif addr == "check_OTP_channel.match":
        base = {"SMS": "Ask for the code by text message.", "EMAIL": "Ask for the code by email.",
                "NO_PREFERENCE_SHARED": "Answer without saying text or email.",
                "UNKNOWN": "Say you can't receive texts or emails."}.get(out, "Answer the question.")
    elif addr == "check_verification.match":
        o = out[0] if isinstance(out, tuple) else out
        sub = out[1] if isinstance(out, tuple) else None
        q = profile.get("question_type")
        true_ans = {"date_of_birth": profile["dob"], "zip_code": profile["zip"], "street_address": profile["street"],
                    "phone_last4": profile["phone4"]}.get(q, profile["dob"])
        if o in ("VALIDATE",) and sub == "VERIFICATION_PASSED":
            base = f"Answer the verification question correctly (if asked for a one-time passcode, give {profile['otp']}; else {true_ans})."
        elif o == "VALIDATE" or o in ("repeatQuestion", "nextQuestion"):
            base = "Answer the verification question, but with a slightly wrong value."
        elif o == "INPUT_USER_ERROR":
            base = "Answer in an invalid format (incomplete or garbled)."
        elif o == "nextQuestion-OTP":
            base = f"Answer the verification question ({true_ans})."
        else:
            base = "Do not answer the question yet."
    elif addr == "txn_answer":
        act = intent.get("act")
        if act == "unsure":
            base = "Say you are not sure whether you made this charge."
        elif act == "non_answer":
            base = "Say something that does not answer whether you made the charge."
        elif act == "talks_about_unknown_transaction":
            base = "Talk about a different charge than the one read out, one the agent does not have."
        else:
            base = ("Say yes, you made this charge." if intent.get("attempted") else "Say no, you did not make this charge.")
    else:
        base = "Answer the agent's question."
    inter = cust.get("interjection", "none")
    if inter and inter != "none":
        base = (base + " " if inter == "multi_demand" else "") + INTERJECTION_HINTS.get(inter, "")
    corr = cust.get("correction", "none")
    if corr == "self_correct_in_turn":
        base += " Start with a slightly wrong value or answer, then correct yourself in the same turn."
    elif corr == "reverse_previous":
        base += " Also take back your previous answer (e.g. 'actually, wait - it was me / it is that card')."
    return base


def _id_text(what: str, value: str, outcome: Any) -> str:
    if outcome in ("INPUT_ERROR",):
        return f"Give {what}, but with the wrong number of digits (e.g. drop one)."
    if outcome in ("ATTEMPT_ERROR", "PROFILE_NO_MATCH", "PROFILE_NOT_IDENTIFIED"):
        return f"Give {what}, but a wrong value."
    if outcome in (None, "NO_USER_RESPONSE"):
        return f"Do not give {what} yet."
    return f"Give {what}: {value}."


# ---------------------------------------------------------------------------------------------------- renderers
@dataclass
class TurnText:
    customer: str
    bot: str
    summary: str


class TemplateRenderer:
    """Deterministic text: enough to run and test the whole pipeline without a model."""
    name = "template"

    def customer(self, cust, profile, history, style) -> str:
        t = customer_intent_text(cust, profile)
        m = re.search(r":\s*([0-9A-Za-z ,]+)\.$", t)
        if m and "Give" in t:
            return m.group(1)
        return t.replace("Say ", "").replace("Answer ", "").strip()

    def bot(self, spec, facts, history, customer, box_texts) -> str:
        if spec.state == HANDOFF_STATE and spec.kind == "gold":
            return "Okay, let me connect you with a specialist who can help."
        parts = []
        for a in spec.hrts.get("atoms", []):
            if a.get("kind") in ("convey",) and not a.get("response_conditional"):
                parts.append(fill(a.get("description") or a.get("tail", ""), facts))
        text = " ".join(dict.fromkeys(parts)) or "Okay."
        if spec.kind == "error":
            if spec.operator == "drop":
                for v in spec.instruction.split("; "):
                    head = re.sub(r"^Leave out: ", "", v).split("] ")[-1]
                    text = text.replace(fill(head, facts), "")
            else:
                text = f"{text} [{spec.operator}: {spec.instruction[:80]}]"
        return re.sub(r"\s+", " ", text).strip()

    def summary(self, history) -> str:
        return " | ".join(f"{'A' if w == 'bot' else 'C'}: {t[:60]}" for w, t in history[-6:])

    def check(self, *a, **k) -> dict[str, Any]:
        return {"accepted": True}


class LLMRenderer:
    """Text from gpt-oss (vLLM) or the Amex gateway via gmsjourney.llm."""
    name = "llm"

    def __init__(self, cfg: dict[str, Any]):
        from gmsjourney.llm import CachedLLM, LLMSettings
        def mk(tag, temp, max_tokens):
            s = LLMSettings(backend=cfg.get("backend", ""), model=cfg.get("model", ""), vllm_url=cfg.get("vllm_url", ""),
                            local_path=cfg.get("local_path", ""), temperature=temp, max_tokens=max_tokens,
                            json_mode=True, cache_dir=cfg.get("cache_dir"), role=cfg.get("role", "enrichment"),
                            mode=cfg.get("mode", "live"))
            return CachedLLM(settings=s, tag=tag)
        mt = cfg.get("max_tokens", {}) or {}            # gpt-oss reasons before it answers: leave room for both
        self.cust = mk("tv_customer", float(cfg.get("customer_temperature", 0.9)), int(mt.get("customer", 800)))
        self.botm = mk("tv_bot", float(cfg.get("bot_temperature", 0.6)), int(mt.get("bot", 1600)))
        self.errm = mk("tv_error", float(cfg.get("error_temperature", 0.8)), int(mt.get("bot", 1600)))
        self.summ = mk("tv_summary", 0.2, int(mt.get("summary", 1000)))
        self.judge = mk("tv_judge", 0.0, int(mt.get("judge", 1600)))
        effort = cfg.get("reasoning_effort", "low")
        self.sys_suffix = f"\nReasoning: {effort}" if effort else ""
        self.max_tries = int(cfg.get("max_tries", 3))
        self.verify = bool(cfg.get("verify", False))

    def _ask(self, llm, system, user, key, salt) -> str:
        last = ""
        for i in range(self.max_tries):
            obj = llm.invoke_json(system + self.sys_suffix, user, salt=f"{salt}#{i}")
            v = (obj or {}).get(key)
            if isinstance(v, str) and v.strip() and "{{" not in v:
                return v.strip()
            last = str(obj)[:200]
        raise ValueError(f"no usable '{key}' after {self.max_tries} tries: {last}")

    def customer(self, cust, profile, history, style) -> str:
        user = (f"CONVERSATION SO FAR:\n{history_text(history)}\n\nINTENT for your next turn: "
                f"{customer_intent_text(cust, profile)}\n\nSTYLE: "
                + " ".join(STYLE_HINTS.get(s, "") for s in style if s) + f"\nYour first name: {profile['first_name']}.")
        return self._ask(self.cust, CUSTOMER_SYSTEM, user, "utterance", f"c|{len(history)}|{json.dumps(cust, default=str)}")

    def bot(self, spec, facts, history, customer, box_texts) -> str:
        if spec.state == HANDOFF_STATE:
            step = "Hand-off: the caller is being transferred / the call is ending. " + spec.hrts.get("handoff", "")
        else:
            step = "\n".join(f"[{b}] {fill(t, facts)}" for b, t in box_texts)
        user = (f"AOP STEP(S):\n{step}\n\nREQUIREMENTS:\n{_atom_lines(spec.hrts, facts)}\n\nDATA VALUES:\n"
                f"{json.dumps(facts, default=str)}\n\nCONVERSATION SO FAR:\n{history_text(history)}\n"
                f"Customer: {customer}\n")
        if spec.kind == "error":
            user += f"\nDEFECT to make: {spec.instruction}\n"
            return self._ask(self.errm, ERROR_SYSTEM, user, "response", f"e|{spec.operator}|{len(history)}")
        return self._ask(self.botm, BOT_SYSTEM, user, "response", f"b|{spec.state}|{len(history)}")

    def summary(self, history) -> str:
        if not history:
            return ""
        return self._ask(self.summ, SUMMARY_SYSTEM, f"CALL SO FAR:\n{history_text(history, last=60)}", "summary",
                         f"s|{len(history)}")

    def check(self, spec, facts, response: str) -> dict[str, Any]:
        """Quality gate: a correct response satisfies its HRTs; an erroneous one shows its defect."""
        if not self.verify:
            return {"accepted": True}
        atoms = [a for a in spec.hrts.get("atoms", []) if not a.get("response_conditional")]
        reqs = "\n".join(f"{i + 1}. {line[2:]}" for i, line in enumerate(_atom_lines({"atoms": atoms}, facts).splitlines()))
        user = f"REQUIREMENTS:\n{reqs}\n\nRESPONSE:\n{response}\n"
        if spec.kind == "error":
            user += f"\nINTENDED DEFECT: {spec.instruction}\n"
        obj = self.judge.invoke_json(JUDGE_SYSTEM + self.sys_suffix, user, salt="judge") or {}
        checks = obj.get("checks") or {}
        if spec.kind == "gold":
            bad = [k for k, v in checks.items() if str(v).lower() == "violated"]
            return {"accepted": not bad, "violated": bad}
        return {"accepted": bool(obj.get("defect_present")), "checks": checks}


def make_renderer(cfg: dict[str, Any]):
    return LLMRenderer(cfg) if cfg.get("renderer", "template") == "llm" else TemplateRenderer()


def mask_digits(text: str) -> str:
    """Customer identity digits as the bot logs show them."""
    return re.sub(r"(?<!\w)(\d[\d\s-]{2,}\d)(?!\w)", "[REDACTED]", text)


__all__ = ["OPENING", "TemplateRenderer", "LLMRenderer", "make_renderer", "facts_for", "fill", "mask_digits",
           "customer_intent_text"]
