"""A reference implementation of the charge-review voice bot (the stand-in SUT).

This is deliberately *not* the oracle. It is written the way a production bot
typically is: a keyword NLU feeding a dialogue state machine, with its own
configuration file (not the knowledge store) and a retrieve-then-template
knowledge-base answerer. Its policy logic follows the procedure, so it passes
clean base cases; it fails where real bots fail - lexicon gaps, typos and
speech-recognition confusions it cannot repair, aggregation questions its
retriever cannot answer.

``MUTATIONS`` are switchable defects used for mutation testing: each one is a
realistic bug (an off-by-one limit, claiming a write that failed, failing
silently on a tool error, ...). A good suite must kill all of them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .tools import ToolError

MUTATIONS = {
    "limit_off_by_one": "Reviews up to 5 transactions instead of 4.",
    "window_off_by_one": "Excludes a transaction exactly 30 days old.",
    "present_all_returned": "Presents every returned transaction, ignoring the 30-day window.",
    "auth_three_tries": "Allows a third authentication attempt.",
    "claim_before_success": "Claims a response was recorded when the write failed, and continues.",
    "silent_tool_failure": "On a card-tool error, carries on as if nothing happened.",
    "no_echo_check": "Trusts a SUCCESS whose echoed transaction/action does not match.",
    "unsure_no_transfer": "Records UNSURE and keeps going instead of transferring.",
    "card_erases_review": "After a card action, skips the remaining transactions.",
    "expedited_no_disclosure": "Submits expedited delivery without disclosing the $15 fee.",
    "first_answer_wins": "Ignores corrections and keeps the first answer.",
    "no_turn_limit": "Never enforces the 20-turn limit.",
    "ignore_human_mid_review": "Treats a request for a human during review as noise.",
    "guess_on_ambiguity": "Assumes RECOGNIZED when an answer is unclear.",
    "fraud_claim": "Tells the customer an unrecognized charge means fraud.",
    "kb_first_amount": "Answers fee questions with the first dollar amount in the knowledge base.",
    "kb_fabricate_unknown": "Answers fee/time questions about options or facts the policy does not state with the expedited figures.",
}

# words that can precede "delivery"/"shipping" in a question without naming an option
_NON_NAMES = set("""the a an does do is for of your my this that which whichever any pick choose use get with by via
what whats how much long if i about on go want take cost costs fee price many business days option options would
should and or to it its please tell me know wondering was were be have has can could will standard regular normal
expedited express rush overnight fast faster fastest quick quicker quickest cheap cheaper cheapest free""".split())

DEFAULT_CONFIG = {"max_transactions": 4, "lookback_days": 30, "max_auth_attempts": 2,
                  "max_clarifications": 1, "max_unsure": 1, "max_turns": 20, "expedited_fee": 15,
                  "opening": "Before I review account-specific charges, I need to authenticate you."}


@dataclass
class BotTurn:
    text: str
    acts: list[dict[str, Any]]
    ended: bool = False


# ------------------------------------------------------------------ NLU
class KeywordNLU:
    """A typical intent/slot classifier: lowercase keyword lists, no repair."""
    DIGITS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
              "six": "6", "seven": "7", "eight": "8", "nine": "9"}

    def __init__(self, first_answer_wins: bool = False):
        self.first_answer_wins = first_answer_wins

    def parse(self, text: str) -> dict[str, Any]:
        t = text.lower()
        words = re.findall(r"[a-z']+|\d+", t)
        spoken = "".join(self.DIGITS.get(w, " ") if w in self.DIGITS else " " for w in words)
        out: dict[str, Any] = {"text": t}
        if re.search(r"\b(speak|talk|transfer|connect|put me through|get me|want|need)\b.{0,25}"
                     r"\b(human|agent|representative|person|someone)\b|\bhuman, please\b|\breal person\b|\bactual agent\b", t):
            out["intent"] = "human"
            return out
        m = re.findall(r"\b\d{4}\b", t) or re.findall(r"\d{4}", spoken.replace(" ", "|"))
        if m:
            out.update(intent="credential", credential=m[-1])
            return out
        sentences = [x.strip() for x in re.split(r"(?<=[.?!])\s+", t) if x.strip()]
        if any(x.endswith("?") or re.match(r"^(how|what|which|is|does|can|do|will|am|tell me|i want to know|"
                                             r"i was wondering|walk me through)\b", x) for x in sentences):
            out["intent"] = "question"
        segs = re.split(r"\b(?:wait|actually|i mean)\b", t)
        if self.first_answer_wins:
            segs = segs[:1] + segs[1:]
            order = segs
        else:
            order = list(reversed(segs))
        for seg in order:
            d = self._disposition(seg)
            if d:
                out["disposition"] = d
                break
        if re.search(r"\b(replace|new card|another card|fresh card)\b", t):
            out["card_action"] = "REPLACE_CARD"
        elif re.search(r"\b(lock|freeze|secure|block)\b", t):
            out["card_action"] = "SECURE_CARD"
        if re.search(r"\b(expedited|express|rush|fast)\b", t):
            out["delivery"] = "expedited_delivery"
        elif re.search(r"\b(standard|regular|normal)\b", t):
            out["delivery"] = "standard_delivery"
        if re.search(r"\b(that's all|nothing else|no thanks|bye|all set)\b", t):
            out["end"] = True
        if re.search(r"\b(yes|yeah|sure|go ahead|okay|ok|please do|proceed)\b", t):
            out["yes"] = True
        if re.search(r"\b(no|don't|cancel|nope)\b", t):
            out["no"] = True
        return out

    @staticmethod
    def _disposition(seg: str) -> str | None:
        if re.search(r"\b(not sure|unsure|don't know|not certain|can't remember)\b", seg):
            return "UNSURE"
        if re.search(r"\b(don't recognize|not mine|isn't mine|wasn't me|never|unauthorized|didn't make)\b", seg):
            return "UNRECOGNIZED"
        if re.search(r"\b(recognize|mine|that's me|was me|legit|i made|i remember)\b", seg):
            return "RECOGNIZED"
        return None


# ------------------------------------------------------------------ KB answerer
class KBAnswerer:
    def __init__(self, sentences: list[str], cfg: dict, mutations: set[str]):
        self.sents = sentences
        self.cfg = cfg
        self.mut = mutations

    def _best(self, q: str) -> str:
        qw = set(re.findall(r"[a-z]+", q.lower())) - {"the", "a", "is", "for", "i", "you", "do", "does", "how", "what", "can"}
        scored = sorted(self.sents, key=lambda s: -len(qw & set(re.findall(r"[a-z]+", s.lower()))))
        return scored[0] if scored and qw & set(re.findall(r"[a-z]+", scored[0].lower())) else ""

    _NUMW = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "one": 1, "two": 2, "three": 3,
             "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "fifteen": 15, "twenty five": 25}

    def _words_to_digits(self, t: str) -> str:
        t = re.sub(r"\b(twenty|thirty|forty|fifty) (one|two|three|four|five|six|seven|eight|nine)\b",
                   lambda m: str(self._NUMW[m.group(1)] + self._NUMW[m.group(2)]), t)
        return re.sub(r"\b(" + "|".join(sorted(self._NUMW, key=len, reverse=True)) + r")\b(?= (days?|dollars))",
                      lambda m: str(self._NUMW[m.group(1)]), t)

    def answer(self, q: str) -> str:
        t = self._words_to_digits(q.lower())
        ent = "expedited" if re.search(r"\b(expedited|express|rush)\b", t) else \
              "standard" if re.search(r"\b(standard|regular|normal)\b", t) else None
        if "fraud" in t:
            return ("Yes, an unrecognized charge means fraud." if "fraud_claim" in self.mut else
                    "An unrecognized transaction is a concern you report; it is not, by itself, proof of fraud.")
        if re.search(r"\b(refund|money back)\b", t):
            return "I don't have information on refunds; a representative can help with that."
        if re.search(r"\bskip\b|without going through", t):
            return "I need to verify your identity first; I can't skip that step."
        if re.search(r"\b(steps|process|how this works)\b", t):
            return ("First I verify you, then I review your recent transactions one at a time, "
                    "record each response, and handle any card request before summarizing.")
        m = re.search(r"\b(\d+) days? (ago|old|back)\b", t)
        if m:
            n = int(m.group(1))
            ok = n <= self.cfg["lookback_days"]
            return (f"Yes, a charge from {n} days ago is within the {self.cfg['lookback_days']}-day window." if ok else
                    f"No, I can only review charges from the last {self.cfg['lookback_days']} days.")
        if re.search(r"\b(how far back|how old|window)\b", t):
            return f"I can review transactions from the previous {self.cfg['lookback_days']} calendar days."
        if re.search(r"\b(attempts|tries|try to verify)\b", t):
            return f"You have {self.cfg['max_auth_attempts']} verification attempts."
        if re.search(r"(how many|most).{0,30}(transactions|charges)", t):
            return f"I can review up to {self.cfg['max_transactions']} transactions in one session."
        fee_q = re.search(r"\b(how much|cost|fee|price|pay)\b", t)
        days_q = re.search(r"\b(how long|days|arrive|take)\b", t)
        if (fee_q or days_q) and not ent:
            unknown = None
            if fee_q and re.search(r"\b(secure|lock|freeze|block)\b.{0,12}\bcard\b", t):
                unknown = "a fee for securing your card"
            m = re.search(r"\b([a-z]+(?:[ -][a-z]+)?) (?:delivery|shipping|courier)\b", t)
            if unknown is None and m:
                import difflib
                rest = [w for w in re.split(r"[ -]", m.group(1))
                        if w not in _NON_NAMES and not difflib.get_close_matches(w, _NON_NAMES, n=1, cutoff=0.8)]
                if rest:
                    unknown = f"{' '.join(rest)} delivery"
            if unknown is not None:
                if "kb_fabricate_unknown" in self.mut:
                    return "That costs $15." if fee_q else "That takes 1 to 2 business days."
                return f"I don't have information on {unknown}; a representative can help with that."
        claimed = re.findall(r"\$(\d+)", t)
        if ent and claimed and re.match(r"^(is|does|am i|is it true)", t.strip()) and len(claimed) == 1:
            fee = 15 if ent == "expedited" else 0
            return "Yes, that's right." if int(claimed[0]) == fee else f"No, it's ${fee}."
        if ent and re.search(r"\b(free)\b", t) and re.match(r"^(is|am i|is it true)", t.strip()):
            return "Yes, standard delivery is free." if ent == "standard" else "No, expedited costs $15."
        if ent and re.search(r"\b(how much|cost|fee|price|pay)\b", t):
            if "kb_first_amount" in self.mut:
                amt = re.search(r"\$\d+", " ".join(self.sents))
                return f"It's {amt.group(0)}." if amt else "I'm not sure."
            s = self._best(q + " fee")
            amt = re.search(r"\$\d+", s)
            return f"{ent.capitalize()} delivery costs {amt.group(0)}." if amt else (s or "I'm not sure.")
        if ent and re.search(r"\b(how long|days|arrive|take)\b", t):
            s = self._best(q + " arrives business days")
            rng = re.search(r"(\d+)\s*-\s*(\d+) business days", s) or re.search(r"(\d+) business days", s)
            return f"{ent.capitalize()} delivery takes {rng.group(0)}." if rng else (s or "I'm not sure.")
        return self._best(q) or ""


# ------------------------------------------------------------------ bot
@dataclass
class ReferenceBot:
    tools: Any
    kb_sentences: list[str]
    config: dict = field(default_factory=lambda: dict(DEFAULT_CONFIG))
    mutations: frozenset = frozenset()

    def __post_init__(self) -> None:
        c = dict(self.config)
        m = set(self.mutations)
        if "limit_off_by_one" in m:
            c["max_transactions"] += 1
        if "auth_three_tries" in m:
            c["max_auth_attempts"] += 1
        self.cfg = c
        self.nlu = KeywordNLU(first_answer_wins="first_answer_wins" in m)
        self.kb = KBAnswerer(self.kb_sentences, c, m)
        self.m = m
        self.state = "AUTH"
        self.turns = 0
        self.auth_attempts = self.clar = self.unsure = 0
        self.queue: list[dict] = []
        self.more = False
        self.cur: dict | None = None
        self.card_req: dict = {}
        self.card_done = False
        self.pending_card: dict = {}
        self.recorded: dict[str, str] = {}
        self.actions: list = []
        self.ended = False

    # -------------------------------------------------------- helpers
    def _say(self, acts: list[dict], ended: bool = False) -> BotTurn:
        self.ended = self.ended or ended
        return BotTurn(" ".join(a.get("text", a["act"]) for a in acts), acts, self.ended)

    def _call(self, tool: str, **args):
        return self.tools.call(tool, **args)

    def _transfer(self, reason: str, prefix: list[dict] | None = None) -> BotTurn:
        acts = list(prefix or [])
        if self.recorded:
            acts.append({"act": "SUMMARIZE", "recorded": dict(self.recorded), "actions": list(self.actions),
                         "text": f"So far I recorded {len(self.recorded)} response(s)."})
        acts.append({"act": "TRANSFER", "reason": reason, "text": "Let me transfer you to a representative."})
        try:
            self._call("transfer_to_human", reason=reason)
        except ToolError:
            pass
        self.state = "DONE"
        return self._say(acts, ended=True)

    def _present(self, prefix: list[dict]) -> BotTurn:
        t = self.cur
        self.state = "REVIEW"
        self.clar = self.unsure = 0
        return self._say(prefix + [{"act": "PRESENT_TXN", "txn": t["transaction_id"],
                                    "text": f"I see a charge at {t['merchant']} for ${t['amount']}. Do you recognize it?"}])

    def _next(self, prefix: list[dict]) -> BotTurn:
        if self.queue and not ("card_erases_review" in self.m and self.card_done):
            self.cur = self.queue.pop(0)
            return self._present(prefix)
        if self.more:
            return self._transfer("transaction_limit", prefix + [{"act": "EXPLAIN_LIMIT",
                                  "text": f"I can review up to {self.cfg['max_transactions']} charges here."}])
        self.state = "MORE_HELP"
        return self._say(prefix + [{"act": "SUMMARIZE", "recorded": dict(self.recorded), "actions": list(self.actions),
                                    "text": "Here's a summary of what I recorded."},
                                   {"act": "ASK_MORE_HELP", "text": "Is there anything else I can help with?"}])

    # -------------------------------------------------------- protocol
    def start(self) -> BotTurn:
        return self._say([{"act": "OPENING", "text": self.cfg["opening"]},
                          {"act": "ASK_AUTH", "text": "Please tell me the last four digits of your card."}])

    def respond(self, text: str) -> BotTurn:
        if self.ended:
            return self._say([{"act": "END", "text": "Goodbye."}], ended=True)
        self.turns += 1
        turn = self._route(self.nlu.parse(text))
        if not turn.ended and self.turns >= self.cfg["max_turns"] and "no_turn_limit" not in self.m:
            if self.state == "MORE_HELP":
                return self._say(turn.acts[:-1] + [{"act": "END", "text": "Thank you, goodbye."}], ended=True)
            return self._transfer("turn_limit")
        return turn

    def _route(self, u: dict) -> BotTurn:
        if self.state == "AUTH":
            return self._auth(u)
        if self.state == "REVIEW":
            return self._review(u)
        if self.state == "CARD_CONFIRM":
            return self._card_confirm(u)
        if self.state == "MORE_HELP":
            return self._more_help(u)
        return self._say([{"act": "END"}], ended=True)

    def _unclear(self, reask: list[dict]) -> BotTurn:
        self.clar += 1
        if self.clar > self.cfg["max_clarifications"]:
            return self._transfer("ambiguity_persists")
        return self._say([{"act": "ASK_CLARIFY", "txn": (self.cur or {}).get("transaction_id"),
                           "text": "Sorry, I didn't catch that."}] + reask)

    def _kb(self, u: dict, reask: list[dict]) -> BotTurn:
        ans = self.kb.answer(u["text"])
        if not ans:                                  # nothing retrieved: treat as unclear, clarify once
            return self._unclear(reask)
        return self._say([{"act": "ANSWER_KB", "txn": (self.cur or {}).get("transaction_id"), "text": ans}] + reask)

    def _auth(self, u: dict) -> BotTurn:
        if u.get("intent") == "human":
            return self._transfer("customer_request")
        if u.get("intent") == "credential":
            try:
                res = self._call("authenticate_customer", credential=u["credential"])
            except ToolError:
                return self._transfer("auth_unavailable")
            if res.get("status") == "AUTHENTICATED":
                return self._retrieve()
            self.auth_attempts += 1
            if self.auth_attempts >= self.cfg["max_auth_attempts"]:
                return self._transfer("auth_failed")
            return self._say([{"act": "ASK_AUTH_RETRY", "text": "That didn't match. Please try again."}])
        if u.get("intent") == "question":
            return self._kb(u, [{"act": "ASK_AUTH", "text": "Now, the last four digits of your card?"}])
        return self._unclear([{"act": "ASK_AUTH", "text": "Please tell me the last four digits of your card."}])

    def _retrieve(self) -> BotTurn:
        try:
            res = self._call("get_review_transactions")
        except ToolError:
            return self._transfer("retrieve_unavailable")
        txns = res.get("transactions", [])
        L = self.cfg["lookback_days"]
        if "present_all_returned" in self.m:
            within = txns
        elif "window_off_by_one" in self.m:
            within = [t for t in txns if t["age_days"] < L]
        else:
            within = [t for t in txns if t["age_days"] <= L]
        if not within:
            self.state = "MORE_HELP"
            return self._say([{"act": "EXPLAIN_NO_TXNS", "text": "There are no charges available for review."},
                              {"act": "ASK_MORE_HELP", "text": "Is there anything else I can help with?"}])
        self.queue = within[: self.cfg["max_transactions"]]
        self.more = len(within) > self.cfg["max_transactions"]
        self.cur = self.queue.pop(0)
        return self._present([])

    def _review(self, u: dict) -> BotTurn:
        tid = self.cur["transaction_id"]
        reprompt = [{"act": "PRESENT_TXN", "txn": tid, "text": "Do you recognize that charge?"}]
        if u.get("intent") == "human" and "ignore_human_mid_review" not in self.m:
            return self._transfer("customer_request")
        if u.get("intent") == "question" and not u.get("disposition"):
            return self._kb(u, reprompt)
        d = u.get("disposition")
        if d is None and "guess_on_ambiguity" in self.m:
            d = "RECOGNIZED"
        if d in ("RECOGNIZED", "UNRECOGNIZED"):
            if u.get("card_action"):
                self.card_req = {"card_action": u["card_action"], "delivery": u.get("delivery")}
            return self._record(tid, d)
        if d == "UNSURE":
            self.unsure += 1
            if self.unsure > self.cfg["max_unsure"]:
                return self._record(tid, "UNSURE")
            t = self.cur
            return self._say([{"act": "PROVIDE_CONTEXT", "txn": tid,
                               "text": f"It was ${t['amount']} at {t['merchant']}, {t['age_days']} days ago. Do you recognize it?"}])
        return self._unclear(reprompt)

    def _record(self, tid: str, d: str) -> BotTurn:
        try:
            res = self._call("record_charge_response", transaction_id=tid, disposition=d)
        except ToolError:
            return self._transfer("record_failed")
        echo_ok = "no_echo_check" in self.m or res.get("transaction_id") == tid
        if res.get("status") != "SUCCESS" or not echo_ok:
            if "claim_before_success" in self.m:
                self.recorded[tid] = d
                return self._next([{"act": "CONFIRM_RECORDED", "txn": tid, "disposition": d, "text": "Got it, recorded."}])
            return self._transfer("record_failed")
        self.recorded[tid] = d
        prefix = [{"act": "CONFIRM_RECORDED", "txn": tid, "disposition": d, "text": "Thanks, I've recorded that."}]
        if d == "UNSURE":
            if "unsure_no_transfer" in self.m:
                return self._next(prefix)
            return self._transfer("unsure_twice", prefix)
        if d == "UNRECOGNIZED" and self.card_req and not self.card_done:
            return self._card_options(prefix)
        return self._next(prefix)

    def _card_options(self, prefix: list[dict]) -> BotTurn:
        self.card_done = True
        try:
            res = self._call("get_card_action_options")
        except ToolError:
            if "silent_tool_failure" in self.m:
                return self._next(prefix)
            return self._transfer("card_options_unavailable", prefix)
        if self.card_req["card_action"] not in res.get("options", []):
            return self._next(prefix + [{"act": "INFORM_NO_CARD_ACTION", "text": "That card action isn't available."}])
        self.pending_card = dict(self.card_req)
        if "expedited_no_disclosure" in self.m and self.pending_card.get("delivery") == "expedited_delivery":
            return self._submit(prefix)
        return self._confirm_card(prefix)

    def _confirm_card(self, prefix: list[dict]) -> BotTurn:
        a, dl = self.pending_card["card_action"], self.pending_card.get("delivery")
        acts = list(prefix)
        if a == "REPLACE_CARD" and dl == "expedited_delivery":
            acts.append({"act": "DISCLOSE_FEE", "fee": self.cfg["expedited_fee"], "delivery": dl,
                         "text": f"Expedited delivery has a ${self.cfg['expedited_fee']} fee and arrives in 2 business days."})
        acts.append({"act": "CONFIRM_ACTION", "action": a, "delivery": dl, "text": "Shall I go ahead?"})
        self.state = "CARD_CONFIRM"
        self.clar = 0
        return self._say(acts)

    def _card_confirm(self, u: dict) -> BotTurn:
        if u.get("intent") == "human":
            return self._transfer("customer_request")
        if u.get("card_action") or u.get("delivery"):
            if u.get("delivery"):
                self.pending_card["delivery"] = u["delivery"]
            if u.get("card_action"):
                self.pending_card["card_action"] = u["card_action"]
            return self._confirm_card([])
        if u.get("yes") and not u.get("no"):
            return self._submit([])
        if u.get("no"):
            return self._next([{"act": "ACK_DECLINED", "text": "Okay, I won't do that."}])
        return self._unclear([{"act": "CONFIRM_ACTION", "action": self.pending_card["card_action"],
                               "delivery": self.pending_card.get("delivery"), "text": "Shall I go ahead?"}])

    def _submit(self, prefix: list[dict]) -> BotTurn:
        a, dl = self.pending_card["card_action"], self.pending_card.get("delivery")
        try:
            res = self._call("submit_card_action", action=a, delivery=dl)
        except ToolError:
            if "silent_tool_failure" in self.m:
                self.actions.append((a, dl))
                return self._next(prefix + [{"act": "ACTION_DONE", "action": a, "text": "Done."}])
            return self._transfer("submit_failed", prefix)
        echo_ok = "no_echo_check" in self.m or res.get("action") == a
        if res.get("status") != "SUCCESS" or not echo_ok:
            return self._transfer("submit_failed", prefix)
        self.actions.append((a, dl))
        return self._next(prefix + [{"act": "ACTION_DONE", "action": a, "delivery": dl, "text": "That's done."}])

    def _more_help(self, u: dict) -> BotTurn:
        if u.get("intent") == "human":
            return self._transfer("customer_request")
        if u.get("intent") == "question" and not u.get("end"):
            return self._kb(u, [{"act": "ASK_MORE_HELP", "text": "Anything else?"}])
        return self._say([{"act": "END", "text": "Thank you for calling. Goodbye."}], ended=True)
