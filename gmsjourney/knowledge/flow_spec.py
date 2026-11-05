"""The dialog flow of the charge-review procedure, written as data for the policy graph.

Each row becomes one ``(from, leads_to, to)`` triple; its qualifiers make the transition executable
(``gmsjourney/flow.py`` walks them; that module documents the expression language). Next to the
executable part every row keeps the procedure's own words (``when``), the rules it exercises and the
PDF section and sentence that state it.

Conventions used below:

* ``on`` is the customer act that triggers a transition, ``auto`` when the bot moves on by itself,
  ``else`` for any act the state's other transitions do not take, ``any`` for a check made before the
  state prompts the customer (the turn limit, R28).
* Every transfer goes through ``_transfer``: when some responses are already recorded the bot first
  summarizes them (R19), then transfers; the reason is the escalation trigger's name.
* ``rules`` are the rules a journey exercises by taking the transition (they drive the coverage closure).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- reusable expressions
TURN_LIMIT = "turns() >= limit('max_turns')"
RETRIEVED = "field('get_review_transactions', 'transactions')"
REVIEWABLE = "select('get_review_transactions', 'transactions', 'age_days', limit('lookback_days'))"
CLEAR = "slot('disposition') in ('RECOGNIZED', 'UNRECOGNIZED')"
UNSURE_AGAIN = "slot('disposition') == 'UNSURE' and n('unsure') + 1 > limit('max_unsure_retries')"
RECORD_DOWN = "call('record_charge_response') == 'UNAVAILABLE'"
RECORD_OK = "call('record_charge_response') == 'SUCCESS' and echoed('record_charge_response', 'transaction_id')"
R33_IF_ECHO = "fire('R33') if not echoed('record_charge_response', 'transaction_id') else None"
R27_IF_FRAUD = "fire('R27') if slot('intent') == 'fraud' else None"
CARD_WANTED = "get('disposition') == 'UNRECOGNIZED' and get('action') and not get('card_done')"
FEE_TO_CONFIRM = "tail(get('delivery'), 'requires_fee_confirmation') == 'yes'"
CAPTURE = ("set('disposition', slot('disposition'))", "set('action', slot('card_action'))",
           "set('delivery', slot('delivery'))", "fire('R21') if slot('corrected_from') else None")
WINDOW_EDGE = (f"fire('R30') if size({RETRIEVED}) != size({REVIEWABLE}) "
               f"or has_value({RETRIEVED}, 'age_days', limit('lookback_days')) else None")


@dataclass
class Row:
    src: str
    dst: str
    on: str
    when: str
    source: str
    evidence: str
    rules: tuple[str, ...] = ()
    edges: tuple[str, ...] = ()
    match: str | None = None
    pre: tuple[str, ...] = ()
    guard: str | None = None
    do: tuple[str, ...] = ()
    say: tuple[str, ...] = ()
    flush: bool = False
    quiet: bool = False
    discard: bool = False

    def qualifiers(self) -> dict:
        q = {"on": self.on, "when": self.when, "rules": list(self.rules), "edges": list(self.edges)}
        for k in ("match", "guard"):
            if getattr(self, k):
                q[k] = getattr(self, k)
        for k in ("pre", "do", "say"):
            if getattr(self, k):
                q[k] = list(getattr(self, k))
        for k in ("flush", "quiet", "discard"):
            if getattr(self, k):
                q[k] = True
        return q


@dataclass
class Act:
    """What a bot act carries in the trace: field -> expression (optional: left out when None)."""
    fields: list[tuple[str, str, bool]] = field(default_factory=list)


HUMAN = ("pdf.2F", "If the customer requests a human at any point, transfer the interaction.")
KB = ("pdf.7", "General policy questions may trigger a knowledge-base lookup.")
TURNS = ("program", "at max 20 turns")
CLARIFY = ("pdf.3", "ask one short clarification")
STILL_UNCLEAR = ("pdf.3", "If the required information or decision is still unclear after that one retry, "
                          "transfer to a human representative.")
TOOL_FAILS = ("pdf.7", "If a required tool fails, transfer to a human.")


def _transfer(src: str, on: str, reason: str, when: str, source: str, evidence: str, rules=(), edges=(),
              match=None, pre=(), guard=None, do=(), flush=False, discard=False, recordable=True) -> list[Row]:
    """A transfer for escalation trigger ``reason``; with responses recorded, summarize first (R19)."""
    do = (f"set('reason', '{reason}')", *do)
    rows = []
    if recordable:
        g = f"({guard}) and recorded()" if guard else "recorded()"
        rows.append(Row(src, "SUMMARIZE", on, f"{when}; responses recorded: summarize, then transfer",
                        source, evidence, (*rules, "R19"), edges, match, pre, g, do, (), flush, False, discard))
    rows.append(Row(src, "TRANSFER", on, when, source, evidence, rules, edges, match, pre, guard, do, (), flush,
                    False, discard))
    return rows


def _kb(src: str, resume: str) -> Row:
    return Row(src, "ANSWER_KB", "ASK_KB", "general policy question", *KB, ("R25",),
               do=(f"set('resume', '{resume}')", "kb_log()", R27_IF_FRAUD))


def _auth_rows(src: str) -> list[Row]:
    s, rows = src, []
    rows += _transfer(s, "any", "turn_limit", "max_turns reached before the journey completes", *TURNS, ("R28",),
                      guard=TURN_LIMIT, discard=True, recordable=False)
    rows += _transfer(s, "REQUEST_HUMAN", "customer_request", "customer asks for a human", *HUMAN, ("R20",),
                      ("W21",) if s == "ASK_AUTH" else (), recordable=False)
    rows.append(_kb(s, "ASK_AUTH"))
    rows += _transfer(s, "PROVIDE_CREDENTIAL", "auth_unavailable", "authenticate_customer UNAVAILABLE", "pdf.2A",
                      "If authentication fails or the tool is unavailable, transfer to a human", ("R5",), ("W4",),
                      guard="call('authenticate_customer') == 'UNAVAILABLE'", recordable=False)
    rows += _transfer(s, "PROVIDE_CREDENTIAL", "retrieve_unavailable", "AUTHENTICATED, get_review_transactions "
                      "UNAVAILABLE", *TOOL_FAILS, ("R3", "R14"), ("W3", "W7"),
                      guard="call('authenticate_customer') == 'AUTHENTICATED' "
                            "and call('get_review_transactions') == 'UNAVAILABLE'", recordable=False)
    rows.append(Row(s, "PRESENT_TXN", "PROVIDE_CREDENTIAL", "AUTHENTICATED and transactions returned", "pdf.2B",
                    "Present one transaction at a time using merchant, amount and date.",
                    ("R3", "R7", "R24", "R6"), ("W3",),
                    guard=f"call('authenticate_customer') == 'AUTHENTICATED' and size({REVIEWABLE}) > 0",
                    do=(f"load({REVIEWABLE}, limit('max_transactions_per_session'))", WINDOW_EDGE, "next_item()")))
    rows.append(Row(s, "EXPLAIN_NO_TXNS", "PROVIDE_CREDENTIAL", "AUTHENTICATED and no transactions returned",
                    "pdf.2B", "If the tool returns no transactions", ("R3", "R7", "R24", "R8"), ("W3", "W6"),
                    guard="call('authenticate_customer') == 'AUTHENTICATED'", do=(WINDOW_EDGE,)))
    rows += _transfer(s, "PROVIDE_CREDENTIAL", "auth_failed", "authenticate_customer FAILED at max_auth_attempts",
                      "pdf.7", "On failure, transfer to human.", ("R4",), ("W4",),
                      guard="count('authenticate_customer', 'FAILED') >= limit('max_auth_attempts')", recordable=False)
    rows.append(Row(s, "ASK_AUTH_RETRY", "PROVIDE_CREDENTIAL",
                    "authenticate_customer FAILED and attempts below max_auth_attempts", "program",
                    "at most 2 times incorrect authentication", (), ("W2",)))
    rows += _transfer(s, "else", "ambiguity_persists", "still unclear after one retry", *STILL_UNCLEAR, ("R23",),
                      match="n('auth_unclear') + 1 > limit('max_clarification_retries')", do=("inc('auth_unclear')",),
                      recordable=False)
    rows.append(Row(s, "ASK_AUTH", "else", "first unclear answer: one short clarification", *CLARIFY, ("R22",),
                    do=("inc('auth_unclear')",), say=("'ASK_CLARIFY'",)))
    return rows


def _capture_rows(src: str) -> list[Row]:
    s, rows = src, []
    rows += _transfer(s, "any", "turn_limit", "max_turns reached before the journey completes", *TURNS, ("R28",),
                      guard=TURN_LIMIT, discard=True)
    rows.append(_kb(s, "PRESENT_TXN"))
    rows += _transfer(s, "REQUEST_HUMAN", "customer_request", "customer asks for a human", *HUMAN, ("R20",))
    # a clear answer: record it, advance only after SUCCESS that echoes the request
    rows += _transfer(s, "INFORM_DISPOSITION", "record_failed", "clear answer; record_charge_response UNAVAILABLE",
                      *TOOL_FAILS, ("R9", "R14"), ("W11",), match=CLEAR, pre=CAPTURE, guard=RECORD_DOWN)
    rows.append(Row(s, "CONFIRM_RECORDED", "INFORM_DISPOSITION",
                    "clear RECOGNIZED/UNRECOGNIZED and record_charge_response SUCCESS", "pdf.2C",
                    "A clear answer maps to RECOGNIZED or UNRECOGNIZED.", ("R9", "R12"), ("W5",),
                    match=CLEAR, pre=CAPTURE, guard=RECORD_OK, do=("claim_recorded()",)))
    rows += _transfer(s, "INFORM_DISPOSITION", "record_failed", "record_charge_response not SUCCESS", "pdf.4",
                      "Advance only after SUCCESS; otherwise transfer.", ("R9", "R12", "R13", "R14"), ("W11",),
                      match=CLEAR, pre=CAPTURE, do=(R33_IF_ECHO,))
    # unsure a second time: record UNSURE, then transfer (R11, via CONFIRM_RECORDED)
    unsure = ("set('disposition', 'UNSURE')", "set('action', None)", "set('delivery', None)")
    rows += _transfer(s, "INFORM_DISPOSITION", "record_failed", "still UNSURE; record_charge_response UNAVAILABLE",
                      *TOOL_FAILS, ("R10", "R14"), ("W11",), match=UNSURE_AGAIN, pre=unsure, guard=RECORD_DOWN)
    rows.append(Row(s, "CONFIRM_RECORDED", "INFORM_DISPOSITION", "still UNSURE: record UNSURE (then transfer)",
                    "pdf.2C", "If still unsure, record UNSURE and transfer.", ("R10", "R12"), ("W5",),
                    match=UNSURE_AGAIN, pre=unsure, guard=RECORD_OK, do=("claim_recorded()",)))
    rows += _transfer(s, "INFORM_DISPOSITION", "record_failed", "still UNSURE; record_charge_response not SUCCESS",
                      "pdf.4", "Advance only after SUCCESS; otherwise transfer.", ("R10", "R12", "R13", "R14"),
                      ("W11",), match=UNSURE_AGAIN, pre=unsure, do=(R33_IF_ECHO,))
    rows.append(Row(s, "PROVIDE_CONTEXT", "INFORM_DISPOSITION", "disposition UNSURE (first time)", "pdf.2C",
                    "If the customer is unsure, provide available transaction context and ask once more.", ("R10",),
                    match="slot('disposition') == 'UNSURE'", do=("inc('unsure')",)))
    rows += _transfer(s, "else", "ambiguity_persists", "still unclear after one retry", *STILL_UNCLEAR,
                      ("R22", "R23"), match="n('unclear') + 1 > limit('max_clarification_retries')",
                      do=("inc('unclear')",))
    rows.append(Row(s, "ASK_CLARIFY", "else", "answer unclear (first time)", *CLARIFY, ("R22",),
                    do=("inc('unclear')",)))
    return rows


def _advance_rows(src: str, source: str, evidence: str) -> list[Row]:
    """After a transaction is settled: the next one, the session limit, or the summary."""
    return [
        Row(src, "PRESENT_TXN", "auto", "next transaction pending", "pdf.2D",
            "Call record_charge_response for the current transaction before advancing.", ("R6",), ("W8",),
            guard="pending()", do=("next_item()",)),
        Row(src, "EXPLAIN_LIMIT", "auto", "more reviewable transactions than max_transactions_per_session", "pdf.5",
            "The bot may review up to 4 recent transactions in one self-service session.", ("R31",),
            guard="overflow()"),
        Row(src, "SUMMARIZE", "auto", "no transaction pending", source, evidence, ("R19",), ("W10",)),
    ]


def _card_reask(src: str, on: str, when: str, source: str, evidence: str, rules_fee=(), pre=(), do=(),
                match=None) -> list[Row]:
    """Ask for confirmation of the card action again: the fee first when the delivery needs it (R26)."""
    return [Row(src, "DISCLOSE_FEE", on, f"{when}; expedited delivery: disclose the fee first", "pdf.5",
                "Before selecting expedited delivery, the bot must tell the customer the $15 fee", rules_fee,
                match=match, pre=pre, guard=FEE_TO_CONFIRM, do=do),
            Row(src, "CONFIRM_ACTION", on, when, source, evidence, (), match=match, pre=pre, do=do)]


def rows() -> list[Row]:
    R: list[Row] = []
    R.append(Row("OPENING", "ASK_AUTH", "auto", "always", "pdf.7",
                 "Authenticate customer. On failure, transfer to human. On success, continue.", ("R1", "R2"), ("W1",)))
    R += _auth_rows("ASK_AUTH")
    R += _auth_rows("ASK_AUTH_RETRY")
    for s in ("PRESENT_TXN", "ASK_CLARIFY", "PROVIDE_CONTEXT"):
        R += _capture_rows(s)
    # a policy question answered: ask again what was pending
    for resume in ("ASK_AUTH", "PRESENT_TXN"):
        R.append(Row("ANSWER_KB", resume, "auto", f"re-ask the pending {resume} prompt", "pdf.7",
                     "Present one transaction and ask whether the customer recognizes it." if resume == "PRESENT_TXN"
                     else "Call authenticate_customer.", ("R6",) if resume == "PRESENT_TXN" else (),
                     guard=f"get('resume') == '{resume}'"))
    R += _card_reask("ANSWER_KB", "auto", "re-ask the pending card-action confirmation", "pdf.2E",
                     "confirm the requested action", ("R26",), match="get('resume') == 'CONFIRM_ACTION'")
    R.append(Row("ANSWER_KB", "ASK_MORE_HELP", "auto", "re-ask whether the customer needs other help", "pdf.2B",
                 "ask whether the customer needs other help", (), guard="get('resume') == 'ASK_MORE_HELP'"))
    # recorded: transfer if still unsure; card action if asked for; then advance
    R += _transfer("CONFIRM_RECORDED", "auto", "unsure_twice", "UNSURE recorded: transfer", "pdf.2C",
                   "If still unsure, record UNSURE and transfer.", ("R11",), ("W11",),
                   guard="get('disposition') == 'UNSURE'", discard=True)
    R += _transfer("CONFIRM_RECORDED", "auto", "card_options_unavailable", "get_card_action_options UNAVAILABLE",
                   *TOOL_FAILS, ("R32",), ("W20",),
                   guard=f"{CARD_WANTED} and call('get_card_action_options') == 'UNAVAILABLE'",
                   do=("set('card_done', True)",), discard=True)
    R.append(Row("CONFIRM_RECORDED", "INFORM_NO_CARD_ACTION", "auto", "the requested card action is not returned",
                 "pdf.4", "Offer only returned options.", ("R15",), ("W13", "W14"),
                 guard=f"{CARD_WANTED} and get('action') not in options('get_card_action_options', 'options')",
                 do=("set('card_done', True)",)))
    R += _card_reask("CONFIRM_RECORDED", "auto", "UNRECOGNIZED with card intent and an eligible option", "pdf.2E",
                     "If an eligible action is returned, confirm the requested action", ("R15", "R26"),
                     match=CARD_WANTED, do=("set('card_done', True)",))
    R[-1].rules = ("R15",)
    R[-1].edges = R[-2].edges = ("W9",)
    R += _advance_rows("CONFIRM_RECORDED", "pdf.2F", "After the available charges are reviewed, summarize the responses")
    R.append(Row("DISCLOSE_FEE", "CONFIRM_ACTION", "auto", "fee disclosed; ask for clear confirmation", "pdf.5",
                 "obtain a clear confirmation"))
    # the card action
    c = "CONFIRM_ACTION"
    R += _transfer(c, "any", "turn_limit", "max_turns reached before the journey completes", *TURNS, ("R28",),
                   guard=TURN_LIMIT, discard=True)
    R += _transfer(c, "CONFIRM", "submit_failed", "submit_card_action UNAVAILABLE", *TOOL_FAILS, ("R16", "R29"),
                   ("W17",), guard="call('submit_card_action') == 'UNAVAILABLE'")
    R.append(Row(c, "ACTION_DONE", "CONFIRM", "submit_card_action SUCCESS", "pdf.2E",
                 "call submit_card_action, and describe it as completed only after SUCCESS", ("R16", "R17"), ("W12",),
                 guard="call('submit_card_action') == 'SUCCESS' and echoed('submit_card_action', 'action')",
                 do=("claim_action()", "fire('R18') if pending() else None")))
    R += _transfer(c, "CONFIRM", "submit_failed", "submit_card_action not SUCCESS", "pdf.7",
                   "Do not claim an action completed unless its API returns SUCCESS.", ("R16", "R17", "R29"), ("W17",))
    switch = ("set('action', slot('card_action') or get('action'))", "set('delivery', slot('delivery') or get('delivery'))")
    R += _card_reask(c, "REQUEST_CARD_ACTION", "customer switches the card action or delivery option", "pdf.2E",
                     "confirm the requested action", ("R26",), pre=switch)
    R.append(Row(c, "ACK_DECLINED", "DENY", "customer declines the action", "pdf.2E",
                 "This second intent must not erase the charge-review intent."))
    R += _transfer(c, "REQUEST_HUMAN", "customer_request", "customer asks for a human", *HUMAN, ("R20",))
    R.append(_kb(c, c))
    R += _transfer(c, "else", "ambiguity_persists", "still unclear after one retry", *STILL_UNCLEAR, ("R23",),
                   match="n('card_unclear') + 1 > limit('max_clarification_retries')", do=("inc('card_unclear')",))
    R += _card_reask(c, "else", "unclear reply: ask for confirmation again", *CLARIFY, ("R26",),
                     do=("inc('card_unclear')",))
    for s, ev in (("ACTION_DONE", "summarize the responses and any account action that actually completed"),
                  ("ACK_DECLINED", "After the available charges are reviewed, summarize the responses"),
                  ("INFORM_NO_CARD_ACTION", "After the available charges are reviewed, summarize the responses")):
        R += _advance_rows(s, "pdf.2F", ev)
    R += _transfer("EXPLAIN_LIMIT", "auto", "transaction_limit", "limit explained; summarize, then transfer the rest",
                   "pdf.5", "The bot may review up to 4 recent transactions in one self-service session.", ("R31",),
                   flush=True)
    R.append(Row("EXPLAIN_NO_TXNS", "ASK_MORE_HELP", "auto", "no charges to review", "pdf.2B",
                 "ask whether the customer needs other help", (), ("W6",)))
    R.append(Row("SUMMARIZE", "TRANSFER", "auto", "a transfer is pending", *HUMAN, (), guard="get('reason')"))
    R.append(Row("SUMMARIZE", "END", "auto", "the turn limit was reached after the journey completed", *TURNS, (),
                 guard="get('ending')", do=("end()",)))
    R.append(Row("SUMMARIZE", "ASK_MORE_HELP", "auto", "journey complete", "pdf.2B",
                 "ask whether the customer needs other help", (), ("W18",)))
    m = "ASK_MORE_HELP"
    R.append(Row(m, "SUMMARIZE", "any", "max_turns reached after the journey completed: summarize and end", *TURNS,
                 guard=TURN_LIMIT, do=("set('ending', True)",), discard=True))
    R.append(_kb(m, m))
    R += _transfer(m, "REQUEST_HUMAN", "customer_request", "customer asks for a human", *HUMAN, ("R20",))
    R.append(Row(m, "END", "END", "customer needs nothing else", "pdf.7", "then end.", (), ("W18",), do=("end()",)))
    R.append(Row(m, "END", "else", "anything else ends the call", "pdf.7", "then end.", (), ("W18",), do=("end()",)))
    R.append(Row("TRANSFER", "END", "auto", "transfer_to_human (the call leaves self-service)", "pdf.4",
                 "End automated servicing.", (), ("W19",), do=("call('transfer_to_human')",), flush=True, quiet=True))
    return R


# what each bot act carries in the trace: act -> [(field, expression, optional)]
CARRIES: dict[str, list[tuple[str, str, bool]]] = {
    "OPENING": [("text", "text('opening_statement')", False)],
    "PRESENT_TXN": [("txn", "item()", False)],
    "PROVIDE_CONTEXT": [("txn", "item()", False)],
    "ASK_CLARIFY": [("txn", "item()", True)],
    "ANSWER_KB": [("intent", "slot('intent')", False), ("answer", "kb_answer()", False)],
    "CONFIRM_RECORDED": [("txn", "item()", False), ("disposition", "get('disposition')", False)],
    "DISCLOSE_FEE": [("fee", "enm('delivery_fee_usd', get('delivery'))", False), ("delivery", "get('delivery')", False)],
    "CONFIRM_ACTION": [("action", "get('action')", False), ("delivery", "get('delivery')", False)],
    "ACTION_DONE": [("action", "get('action')", False), ("delivery", "get('delivery')", False)],
    "SUMMARIZE": [("recorded", "recorded()", False)],
    "TRANSFER": [("reason", "get('reason')", False)],
}

# the arguments each tool is called with
TOOL_ARGS: dict[str, list[tuple[str, str]]] = {
    "authenticate_customer": [("credential", "slot('credential')")],
    "record_charge_response": [("transaction_id", "item()"), ("disposition", "get('disposition')")],
    "submit_card_action": [("action", "get('action')"), ("delivery", "get('delivery')")],
    "transfer_to_human": [("reason", "get('reason')")],
}
