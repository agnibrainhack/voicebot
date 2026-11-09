"""The reference bots: well-behaved bots under test that follow the policy.

A reference bot is a stand-in for your bot, built to follow the policy graph and to understand customers by
keywords only. It validates the test suite itself (a correct bot should pass the original cases) and, with one
deliberate defect planted at a time, shows that the suite catches real mistakes (mutation analysis).

Parts of this file:
  1. ``ReferenceBot`` for the charge-review pack (keyword NLU, a knowledge-base answerer, the dialog).
  2. ``GraphBot`` for any compact-graph pack (the graph's interpreter with the pack's wording) and ``mutants``:
     every defect the graph makes possible (a failure status treated as success, a skipped confirmation, a limit
     plus one ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# =====================================================================================================================
# PART 1 - A reference implementation of the charge-review voice bot (the stand-in SUT).
#
# This is deliberately *not* the oracle. It is written the way a production bot
# typically is: a keyword NLU feeding a dialogue state machine, with its own
# configuration file (not the knowledge store) and a retrieve-then-template
# knowledge-base answerer. Its policy logic follows the procedure, so it passes
# clean base cases; it fails where real bots fail - lexicon gaps, typos and
# speech-recognition confusions it cannot repair, aggregation questions its
# retriever cannot answer.
#
# ``MUTATIONS`` are switchable defects used for mutation testing: each one is a
# realistic bug (an off-by-one limit, claiming a write that failed, failing
# silently on a tool error, ...). A good suite must kill all of them.
# =====================================================================================================================

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
    # defects in the WORDS only: the act labels and every tool call stay correct, so only the check of
    # what the bot says (checks/speech_*.py) can catch them
    "says_wrong_fee": "Labels the $15 expedited fee correctly but tells the customer $25.",
    "misstates_recorded_answer": "Records the right answer but tells the customer it recorded the opposite.",
    "names_wrong_amount": "Presents the right charge but reads out an amount one dollar off.",
    "summary_omits": "Summarizes only the first recorded response.",
}

# words that can precede "delivery"/"shipping" in a question without naming an option
_NON_NAMES = set("""the a an does do is for of your my this that which whichever any pick choose use get with by via
what whats how much long if i about on go want take cost costs fee price many business days option options would
should and or to it its please tell me know wondering was were be have has can could will standard regular normal
expedited express rush overnight fast faster fastest quick quicker quickest cheap cheaper cheapest free""".split())

DEFAULT_CONFIG = {"max_transactions": 4, "lookback_days": 30, "max_auth_attempts": 2,
                  "max_clarifications": 1, "max_unsure": 1, "max_turns": 20, "expedited_fee": 15, "expedited_days": 2,
                  "opening": "Before I review account-specific charges, I need to authenticate you."}


@dataclass
class BotTurn:
    """What a bot returns for one turn: the words, the dialog acts (labels with fields) and whether it ended."""
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
    """The reference bot's policy-question answerer: picks the knowledge-base sentence that shares most words
    with the question (deliberately narrow, so the suite has something to find).
    """
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


# ------------------------------------------------------------------ wording
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]
SPOKEN_DISPOSITION = {"RECOGNIZED": "recognized", "UNRECOGNIZED": "not recognized", "UNSURE": "unsure"}
SPOKEN_ACTION = {"SECURE_CARD": ("secure your card", "secured"), "REPLACE_CARD": ("replace your card", "replaced")}
SPOKEN_DELIVERY = {"standard_delivery": "standard delivery", "expedited_delivery": "expedited delivery"}


def spoken_date(iso: str | None) -> str:
    """'2026-03-13' -> 'March 13' (how the bot reads a charge's date out)."""
    if not iso:
        return ""
    y, m, d = (int(x) for x in iso.split("-"))
    return f"{MONTH_NAMES[m - 1]} {d}"


# ------------------------------------------------------------------ bot
@dataclass
class ReferenceBot:
    """The charge-review reference bot: keyword understanding, the procedure as code, planted defects on request."""
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
        self.merchants: dict[str, str] = {}      # transaction_id -> merchant, for the summary
        self.ended = False

    # -------------------------------------------------------- helpers
    def _say(self, acts: list[dict], ended: bool = False) -> BotTurn:
        self.ended = self.ended or ended
        return BotTurn(" ".join(a.get("text", a["act"]) for a in acts), acts, self.ended)

    def _call(self, tool: str, **args):
        return self.tools.call(tool, **args)

    def _summary_text(self, lead: str) -> str:
        """Every response recorded and every action completed, in words (pdf.2F, R19)."""
        items = list(self.recorded.items())
        if "summary_omits" in self.m:
            items = items[:1]
        said = [f"the {self.merchants.get(t, t)} charge as {SPOKEN_DISPOSITION.get(d, d.lower())}" for t, d in items]
        text = f"{lead} I recorded " + (", ".join(said[:-1]) + " and " + said[-1] if len(said) > 1 else said[0]) + "."
        for a, dl in self.actions:
            text += f" Your card has been {SPOKEN_ACTION[a][1]}" + (f", with {SPOKEN_DELIVERY[dl]}" if dl else "") + "."
        return text

    def _transfer(self, reason: str, prefix: list[dict] | None = None) -> BotTurn:
        acts = list(prefix or [])
        if self.recorded:
            acts.append({"act": "SUMMARIZE", "recorded": dict(self.recorded), "actions": list(self.actions),
                         "text": self._summary_text("So far,")})
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
        amount = f"{float(t['amount']) + 1:.2f}" if "names_wrong_amount" in self.m else t["amount"]
        on = f" on {spoken_date(t.get('date'))}" if t.get("date") else ""
        return self._say(prefix + [{"act": "PRESENT_TXN", "txn": t["transaction_id"],
                                    "text": f"I see a charge at {t['merchant']} for ${amount}{on}. Do you recognize it?"}])

    def _next(self, prefix: list[dict]) -> BotTurn:
        if self.queue and not ("card_erases_review" in self.m and self.card_done):
            self.cur = self.queue.pop(0)
            return self._present(prefix)
        if self.more:
            return self._transfer("transaction_limit", prefix + [{"act": "EXPLAIN_LIMIT",
                                  "text": f"I can review up to {self.cfg['max_transactions']} charges here."}])
        self.state = "MORE_HELP"
        return self._say(prefix + [{"act": "SUMMARIZE", "recorded": dict(self.recorded), "actions": list(self.actions),
                                    "text": self._summary_text("Here's a summary:")},
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
        self.merchants = {t["transaction_id"]: t["merchant"] for t in txns}
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
        said = "Thanks, I've recorded that."
        if "misstates_recorded_answer" in self.m:
            opposite = {"RECOGNIZED": "you don't recognize it", "UNRECOGNIZED": "you recognize it"}.get(d, "you're unsure")
            said = f"Thanks, I've recorded that {opposite}."
        prefix = [{"act": "CONFIRM_RECORDED", "txn": tid, "disposition": d, "text": said}]
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
            fee = self.cfg["expedited_fee"] + (10 if "says_wrong_fee" in self.m else 0)
            acts.append({"act": "DISCLOSE_FEE", "fee": self.cfg["expedited_fee"], "delivery": dl,
                         "text": f"Expedited delivery has a ${fee} fee and arrives in "
                                 f"{self.cfg['expedited_days']} business days."})
        acts.append({"act": "CONFIRM_ACTION", "action": a, "delivery": dl, "text": self._confirm_text(a, dl)})
        self.state = "CARD_CONFIRM"
        self.clar = 0
        return self._say(acts)

    @staticmethod
    def _confirm_text(a: str, dl: str | None) -> str:
        """'Shall I replace your card with expedited delivery?' - the customer hears what they confirm (R16)."""
        return f"Shall I {SPOKEN_ACTION[a][0]}" + (f" with {SPOKEN_DELIVERY[dl]}" if dl and a == "REPLACE_CARD" else "") + "?"

    @staticmethod
    def _done_text(a: str, dl: str | None) -> str:
        return f"That's done. Your card has been {SPOKEN_ACTION[a][1]}" + \
            (f", with {SPOKEN_DELIVERY[dl]}" if dl and a == "REPLACE_CARD" else "") + "."

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
        a, dl = self.pending_card["card_action"], self.pending_card.get("delivery")
        return self._unclear([{"act": "CONFIRM_ACTION", "action": a, "delivery": dl, "text": self._confirm_text(a, dl)}])

    def _submit(self, prefix: list[dict]) -> BotTurn:
        a, dl = self.pending_card["card_action"], self.pending_card.get("delivery")
        try:
            res = self._call("submit_card_action", action=a, delivery=dl)
        except ToolError:
            if "silent_tool_failure" in self.m:
                self.actions.append((a, dl))
                return self._next(prefix + [{"act": "ACTION_DONE", "action": a, "text": self._done_text(a, dl)}])
            return self._transfer("submit_failed", prefix)
        echo_ok = "no_echo_check" in self.m or res.get("action") == a
        if res.get("status") != "SUCCESS" or not echo_ok:
            return self._transfer("submit_failed", prefix)
        self.actions.append((a, dl))
        return self._next(prefix + [{"act": "ACTION_DONE", "action": a, "delivery": dl, "text": self._done_text(a, dl)}])

    def _more_help(self, u: dict) -> BotTurn:
        if u.get("intent") == "human":
            return self._transfer("customer_request")
        if u.get("intent") == "question" and not u.get("end"):
            return self._kb(u, [{"act": "ASK_MORE_HELP", "text": "Anything else?"}])
        return self._say([{"act": "END", "text": "Thank you for calling. Goodbye."}], ended=True)


# =====================================================================================================================
# PART 2 - A reference bot for a compact policy graph, and the defects (mutants) mined from the graph.
#
# ``GraphBot`` follows the graph with the same interpreter as the oracle (graph/interpreter.py), one
# turn at a time, against the live tools it is given. It understands customers with its own
# ``KeywordReader``: the graph's canonical names and customer synonyms only, bare yes / no words
# and a 4-digit code. That reader is deliberately simpler than the frame reader that proves the
# invariance (no typo or speech repair, no wording from the lexicon), so on enriched wording the
# reference bot fails where a plain keyword bot would.
#
# Mutants are derived from the graph, one per thing a bot could get wrong:
#
#   status_as_ok:<tool>        treat a failure status as success (skip auth, ignore a decline)
#   ignore_echo:<tool>         accept a success that echoes another value
#   skip_say:<state>           never say this (a disclosure, a limit explanation, a reason)
#   skip_ask:<state>           act without asking (submit without a confirmation)
#   ignore_guard:<state>       always take the default branch of a guarded choice
#   limit_plus_one:<limit>     an off-by-one on a limit
#   wrong_field:<state>.<f>    say another value than the one the policy gives
#   no_human                   ignore a request for a person
#   no_turn_limit              ignore the turn limit
#   clarify_forever            never give up on an unclear answer
# =====================================================================================================================

from ..cases.meaning import SemanticFrame as F
from ..graph.interpreter import GraphOracle, _Stop
from ..graph.compact import ECHO_MISMATCH, NEXT, UNAVAILABLE

_YES = r"\b(yes|yeah|yep|sure|go ahead|please do|correct|ok|okay|proceed)\b"
_NO = r"\b(no|nope|don't|dont|cancel|not now)\b"
_END = r"\b(that's all|thats all|nothing else|no thanks|all set|that's everything|im done|i'm done)\b"
_HUMAN = r"\b(human|real person|representative|an agent|live agent|transfer me|speak to someone|a person)\b"


class KeywordReader:
    """The reference bot's own understanding: the domain's value and act cues (the pack's lexicon,
    as a bot built for this domain would know them) plus the graph's synonyms, bare yes / no and
    a 4-digit code - but none of the proof reader's normalisation: no typo or speech-recognition
    repair, no number words, no reported speech, no correction markers."""

    def __init__(self, g):
        self.g = g
        from ..rewording.wording import Lexicon
        lx = Lexicon.current()
        self.cues: dict[str, list[tuple[str, str]]] = {}
        for vs, members in g.values.items():
            rows = []
            for m, spec in members.items():
                for a in [m.lower().replace("_", " ")] + [a.lower() for a in spec["aliases"]]:
                    rows.append((r"\b" + re.escape(a.replace("'", "")) + r"\b", m))
                for c in (lx.values.get(m) or {}).get("cues", []):
                    rows.append((r"\b(" + c + r")\b", m))
            self.cues[vs] = rows
        self.act_cues = {a: r"\b(" + "|".join(sp["cues"]) + r")\b" for a, sp in lx.acts.items() if sp.get("cues")}

    def find(self, vs: str, text: str) -> str | None:
        """The member named last in the text."""
        found = sorted(((mm.start(), mm.end(), m) for pat, m in self.cues.get(vs, []) for mm in re.finditer(pat, text)),
                       key=lambda x: (x[0], -(x[1] - x[0])))
        kept, end = [], -1
        for s, e, m in found:                      # earliest-longest, non-overlapping
            if s >= end:
                kept.append(m)
                end = e
        return kept[-1] if kept else None

    def read(self, text: str, state: str) -> F:
        g = self.g
        t = (text or "").lower().replace("’", "'").replace("'", "")
        human = (g.glob.get("human") or {}).get("act")
        if human in self.act_cues and re.search(self.act_cues[human], t):
            return F.make(human)
        G = g.glob
        if G.get("human") and re.search(_HUMAN, t):
            return F.make(G["human"]["act"])
        s = g.states.get(state) or {}
        on = s.get("on") or {}
        if G.get("question") and "?" in t:
            q = self._question(t)
            if q is not None:
                return q
        for act in on:
            specs = g.slot_specs(act)
            if not specs:
                continue
            kw = {}
            for sl, sp in specs.items():
                if sp["set"] == "credential_code":
                    m = re.findall(r"\b(\d{4})\b", t)
                    if m:
                        kw[sl] = m[-1]
                elif not sp.get("correction_of"):
                    v = self.find(sp["set"], t)
                    if v is not None:
                        kw[sl] = v
            req = [sl for sl, sp in specs.items() if not sp.get("optional")]
            if kw and all(r in kw for r in req):
                return F.make(act, **kw)
        if "END" in on and (re.search(_END, t) or ("END" in self.act_cues and re.search(self.act_cues["END"], t))):
            return F.make("END")
        y, n = g.interprets(state)
        if re.search(_YES, t) and not re.search(_NO, t):
            return self._as(y or "CONFIRM")
        if re.search(_NO, t):
            return self._as(n or "DENY")
        return F.make(G["unclear"]["act"] if G.get("unclear") else "UNCLEAR")

    def _as(self, target: str) -> F:
        vs = self.g.member_set(target)
        if vs:
            for a in self.g.acts:
                for sl, s in self.g.acts[a].items():
                    if s == vs:
                        return F.make(a, **{sl: target})
        return F.make(target)

    def _question(self, t: str) -> F | None:
        g = self.g
        from ..rewording.wording import Lexicon
        kbq = Lexicon.current().kb
        if kbq.get("attributes") or kbq.get("limits"):          # the domain's question wording
            has = lambda cues, s: any(re.search(r"\b(" + c + r")\b", s) for c in cues)
            num = g.numeric_attributes()
            for attr, spec in (kbq.get("attributes") or {}).items():
                vs = (num.get(attr) or {}).get("set")
                ent = self.find(vs, t) if vs else None
                if ent is None:
                    continue
                rest = t
                for pat, m in self.cues.get(vs, []):
                    if m == ent:
                        rest = re.sub(pat, " ", rest)
                if has(spec["cues"], rest):
                    return F.make("ASK_KB", intent="value", attr=attr, entity=ent)
            for key, spec in (kbq.get("limits") or {}).items():
                if has(spec["cues"], t):
                    return F.make("ASK_KB", intent="limit", key=key)
            return None
        num = g.numeric_attributes()
        for attr, spec in num.items():
            ent = self.find(spec["set"], t)
            if ent is None:
                continue
            money = attr.endswith("_usd")
            if money and re.search(r"\b(fee|cost|how much|price|pay)\b", t):
                return F.make("ASK_KB", intent="value", attr=attr, entity=ent)
            if not money and re.search(r"\b(how long|days|arrive|when)\b", t):
                return F.make("ASK_KB", intent="value", attr=attr, entity=ent)
        for k, lv in g.limits.items():
            if lv.get("internal"):
                continue
            words = [w for w in k.split("_") if len(w) > 3 and w not in ("max", "min", "per")]
            if words and any(re.search(r"\b" + w.rstrip("s"), t) for w in words):
                return F.make("ASK_KB", intent="limit", key=k)
        return None


def mutants(g) -> list[str]:
    """Every defect the graph makes possible."""
    out = []
    for name, s in g.states.items():
        k = g.kind(name)
        if k == "call":
            t = s["call"]
            if g.tools[t].get("fail") and g.tools[t]["role"] != "handoff":
                out.append(f"status_as_ok:{t}")
            if g.tools[t].get("echo"):
                out.append(f"ignore_echo:{t}")
        elif k == "say" and (s.get("rules") or s.get("outcome")) and name not in (g.start,):
            out.append(f"skip_say:{name}")
        elif k == "ask":
            if "CONFIRM" in s["on"]:
                out.append(f"skip_ask:{name}")
            if any(isinstance(tg, list) for tg in s["on"].values()):
                out.append(f"ignore_guard:{name}")
    for name, s in g.states.items():
        if g.kind(name) == "say" and isinstance(s.get("go"), list):
            out.append(f"ignore_guard:{name}")
    lims = set()
    for e in g.edges():
        try:
            import json
            c = json.loads(e["when"]) if e["when"].startswith("{") else {}
        except ValueError:
            c = {}
        lims |= {c[x] for x in ("below", "at_least", "count_upto", "count_over") if x in c}
    for s in g.states.values():
        if s.get("cap"):
            lims.add(s["cap"])
        for w in (s.get("where") or {}).values():
            if w.get("max"):
                lims.add(w["max"])
    out += [f"limit_plus_one:{x}" for x in sorted(lims)]
    for rid, kind, c in g.rule_checks():
        if kind == "say_result":
            out.append(f"wrong_field:{c['say_result']}.{c['field']}")
    if g.glob.get("human"):
        out.append("no_human")
    if g.glob.get("turn_limit"):
        out.append("no_turn_limit")
    if g.glob.get("unclear"):
        out.append("clarify_forever")
    return list(dict.fromkeys(out))


class GraphBot(GraphOracle):
    """The reference bot: start() / respond(text) -> BotTurn, tools observed at the mock backend."""

    def __init__(self, tools, store, mutations: frozenset = frozenset()):
        super().__init__(store)
        self.m = set(mutations)
        self.reader = KeywordReader(self.g)
        env = tools if hasattr(tools, "log") else getattr(tools, "env", tools)
        self.start_session(tools, env)
        self._state: str | None = None
        self.ended = False

    # -------------------------------------------------------- mutation hooks
    def L(self, name: str) -> Any:
        v = super().L(name)
        if f"limit_plus_one:{name}" in self.m:
            return v + 1
        if "clarify_forever" in self.m and self.g.glob.get("unclear", {}).get("limit") == name:
            return 10 ** 6
        if "no_turn_limit" in self.m and self.g.glob.get("turn_limit", {}).get("limit") == name:
            return 10 ** 6
        return v

    def _call(self, tool: str, args: dict[str, Any]) -> tuple[str, dict]:
        status, res = super()._call(tool, args)
        sp = self.g.tools[tool]
        if f"status_as_ok:{tool}" in self.m and status in sp.get("fail", []):
            status = sp["ok"]
        if f"ignore_echo:{tool}" in self.m and status == ECHO_MISMATCH:
            status = sp["ok"]
        return status, res

    def _target(self, target: Any, state: str) -> str:
        if f"ignore_guard:{state}" in self.m and isinstance(target, list):
            target = target[-1]
        return super()._target(target, state)

    def act_of(self, state: str) -> dict[str, Any]:
        act = super().act_of(state)
        for k in list(act):
            if f"wrong_field:{state}.{k}" in self.m:
                vs = self.g.member_set(act[k]) if isinstance(act[k], str) else None
                others = [m for m in (self.g.members(vs) if vs else []) if m != act[k]]
                act[k] = others[0] if others else None
        return act

    def advance(self, state: str) -> str:
        g = self.g
        while True:
            if state.startswith("!"):
                return super().advance(state)
            k = g.kind(state)
            s = g.states[state]
            if k == "say" and f"skip_say:{state}" in self.m:
                self._tr.visited.append(state)
                state = self._target(s["go"], state)
                if state == NEXT:
                    state = g.loop_of[self._tr.visited[-1]]
                continue
            if k == "ask" and f"skip_ask:{state}" in self.m:
                nxt = self._target(s["on"][next(iter(s["on"]))], state)
                state = g.loop_of.get(state, state) if nxt == NEXT else nxt
                continue
            # run the oracle's advance one step at a time so the hooks above see every state
            return self._advance_one(state)

    def _advance_one(self, state: str) -> str:
        g = self.g
        k = g.kind(state)
        if k in ("ask", "end") or state.startswith("!"):
            return super().advance(state)
        # a say / call / each state: do it, then continue with the hooks
        s = g.states[state]
        tr = self._tr
        tr.visited.append(state)
        tr.rules_fired.update(s.get("rules", []))
        if k == "say":
            act = self.act_of(state)
            self._pending.append(act)
            self._note(state, act)
            nxt = self._target(s["go"], state)
        elif k == "call":
            args = {a: self.resolve(ref) for a, ref in (s.get("args") or {}).items()}
            status, _ = self._call(s["call"], args)
            nxt = self._target(s["on"].get(status, s["on"].get(UNAVAILABLE)), state)
        else:
            nxt = self._each(state)
        if nxt == NEXT:
            nxt = g.loop_of[state]
        return self.advance(nxt)

    def react(self, state: str, f: F) -> str | None:
        G = self.g.glob
        if "no_human" in self.m and G.get("human") and f.act == G["human"]["act"]:
            f = F.make(G["unclear"]["act"]) if G.get("unclear") else f
        return super().react(state, f)

    def _answer_kb(self, frame: F) -> dict[str, Any]:
        """Policy answers from the bot's own copy of the graph (not the store)."""
        g = self.g
        intent = frame.get("intent")
        if intent == "value":
            v = g.attr(frame.get("entity"), frame.get("attr"))
            what = frame.get("attr").replace("_usd", "").replace("_", " ")
            ent = frame.get("entity").replace("_", " ").lower()
            text = f"The {what} for {ent} is " + (f"${v}." if frame.get("attr").endswith("_usd") else f"{v}.")
        elif intent == "limit":
            v = g.limit_value(frame.get("key"))
            text = f"The limit is {v}."
        else:
            v, text = None, "I don't have information on that; a representative can help."
        self._tr.kb.append({"frame": frame, "answer": v, "correct": True})
        return {"act": g.glob["question"]["answer"], "intent": intent, "answer": v, "text": text}

    # -------------------------------------------------------- the bot interface
    def _turn(self, ended: bool) -> BotTurn:
        acts = [{**a, "text": _say(a)} for e in self._tr.events[self._mark:] if e["role"] == "bot" for a in e["acts"]]
        self.ended = self.ended or ended
        return BotTurn(" ".join(a["text"] for a in acts), acts, self.ended)

    def start(self) -> BotTurn:
        self._mark = len(self._tr.events)
        try:
            self._state = self.advance(self.g.start)
            self._emit()
            return self._turn(False)
        except _Stop:
            return self._turn(True)

    def respond(self, text: str) -> BotTurn:
        if self.ended or self._state is None:
            return BotTurn("", [], True)
        self._mark = len(self._tr.events)
        st = self._state
        f = self.reader.read(text, st)
        self.record_user(st, f)
        try:
            nxt = self.react(st, f)
            self._state = self.advance(st if nxt is None else nxt)
            self._emit()
            return self._turn(False)
        except _Stop:
            return self._turn(True)


def _say(a: dict[str, Any]) -> str:
    """The act in words: its text, else the pack's template (domain.yaml -> bot.says) with each field worded,
    else its fields in plain words."""
    if a.get("text"):
        return str(a["text"])
    from ..packs.loading import current
    dom = current()
    tpl = (dom.get("bot.says") or {}).get(a["act"])
    words = {k: _word(dom, k, v) for k, v in a.items() if k != "act"}
    if tpl:
        class _Fill(dict):
            def __missing__(self, k):
                return ""
        # [ ... {field} ... ] is said only when every field inside it has a value ("[ with {delivery}]")
        tpl = re.sub(r"\[([^\[\]]*)\]", lambda mt: mt.group(1) if all(words.get(f) not in (None, "")
                     for f in re.findall(r"{(\w+)}", mt.group(1))) else "", tpl)
        out = " ".join(tpl.format_map(_Fill(words)).split())
        return out[:1].upper() + out[1:]
    fields = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in words.items() if v not in (None, ""))
    return f"{a['act'].replace('_', ' ').capitalize()}{': ' + fields if fields else ''}."


def _word(dom, field: str, v: Any) -> str:
    """One field value as the bot says it: a value in its lexicon wording, money as $N, else as is."""
    if v is None:
        return ""
    if isinstance(v, dict):                          # a summary: key -> value ("T1 as recognized and T2 as ...")
        return " and ".join(f"{k} as {_word(dom, field, x)}" for k, x in v.items()) or "nothing"
    if isinstance(v, str):
        bv = (dom.get("bot.values") or {}).get(v)    # the bot's own word for a value
        if bv:
            return str(bv)
        say = ((dom.get("lexicon.values") or {}).get(v) or {}).get("say") or {}
        if say.get("canonical"):
            return say["canonical"]
        g = dom.graph
        if g is not None and g.member_set(v):
            al = g.aliases(v)
            return al[0] if al else v.lower().replace("_", " ")
        return v
    if isinstance(v, (int, float)) and ("fee" in field or field.endswith("_usd")):
        return f"${v:g}"
    return f"{v:g}" if isinstance(v, float) else str(v)


def make_graph_factory(store, mutations: frozenset = frozenset()):
    """A bot factory for the graph reference bot of the active compact-graph pack."""
    def factory(tools):
        return GraphBot(tools, store, frozenset(mutations))
    return factory
