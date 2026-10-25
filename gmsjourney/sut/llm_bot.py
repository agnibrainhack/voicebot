"""An LLM-driven charge-review voice bot: a second system under test, next to the keyword/state-machine
``ReferenceBot``.

The bot is a tool-using agent on the same Microsoft model gateway and replay cache as the LLM enrichment arm
(``enrich/llm/client.py``, ``CachedLLM`` with its own cache tag ``sut``). It is given what a production team
would give an LLM bot - and nothing the oracle computes:

* the operating procedure (``policy/charge_review/charge_review_policy.md``: limits, delivery options, tools, workflow,
  rules, knowledge-base sentences) as its instructions;
* the backend tools' API (names, arguments, results, and the transfer reason codes the
  ``transfer_to_human`` API takes);
* the dialog-act inventory its turns must be labelled with (the act names of the policy graph, the same
  inventory the reference bot uses), so the harness can read what it claims - which is graded against
  what its tool calls actually did.

Each step the model answers with ONE JSON object: ``{"tool": name, "args": {...}}`` (the call is made on
the harness's mock backend, so faults, governance gates and the trace see it) or ``{"say": [acts...],
"end": bool}`` (the turn the customer hears). Up to ``max_steps`` tool calls per customer turn; a reply
that is not valid JSON is sent back once with the reason; a second failure produces a visible
``UNKNOWN`` turn (a defect of the bot, graded as such - never repaired by the harness).

Replies are cached by content hash (temperature 0 by default), so a campaign re-runs byte for byte with
``mode="replay"`` and the harness's repeated runs of the same conversation (turn-by-turn export) cost
nothing. ``LLMUnavailable`` (no credential, gateway down, replay miss) is raised, never hidden.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .reference_bot import BotTurn
from .tools import ToolError

SUT_PROMPT_VERSION = "llm-sut/1"

# backend API as the bot team would document it: the domain pack's tool entries (domain.yaml -> tools.*.api)
def _tool_api() -> dict[str, dict[str, Any]]:
    from .tools import tool_specs
    return {t: sp["api"] for t, sp in tool_specs().items() if sp.get("api")}


TRANSFER_REASONS = {
    "customer_request": "the customer asked for a person",
    "auth_failed": "authentication failed the maximum number of times",
    "auth_unavailable": "authenticate_customer was unavailable",
    "retrieve_unavailable": "get_review_transactions was unavailable",
    "record_failed": "record_charge_response failed or did not confirm the charge",
    "unsure_twice": "the customer stayed unsure after the additional context",
    "ambiguity_persists": "the answer stayed unclear after one clarification",
    "transaction_limit": "more charges remain than one session may review",
    "card_options_unavailable": "get_card_action_options was unavailable",
    "submit_failed": "submit_card_action failed or did not confirm the action",
    "turn_limit": "the conversation reached its turn limit",
}

# the dialog-act inventory (names = the policy graph's dialog states) and the fields each act carries
ACTS = {
    "OPENING": ("the required opening statement", []),
    "ASK_AUTH": ("ask for the last four digits of the card", []),
    "ASK_AUTH_RETRY": ("the digits did not match; ask again", []),
    "PRESENT_TXN": ("read out ONE returned charge (merchant, amount) and ask whether the customer recognizes it",
                    ["txn"]),
    "ANSWER_KB": ("answer a general policy question from the procedure", ["txn"]),
    "ASK_CLARIFY": ("ask the customer to clarify an answer you could not understand", ["txn"]),
    "PROVIDE_CONTEXT": ("give more detail about the charge under review and ask once more", ["txn"]),
    "CONFIRM_RECORDED": ("confirm that a response was recorded", ["txn", "disposition"]),
    "DISCLOSE_FEE": ("state the fee of a delivery option before the customer confirms", ["fee", "delivery"]),
    "CONFIRM_ACTION": ("ask the customer to confirm a card action", ["action", "delivery"]),
    "ACTION_DONE": ("tell the customer a card action is complete", ["action", "delivery"]),
    "ACK_DECLINED": ("acknowledge that the customer declined the card action", []),
    "INFORM_NO_CARD_ACTION": ("tell the customer the requested card action is not available", []),
    "EXPLAIN_LIMIT": ("explain a session limit", []),
    "EXPLAIN_NO_TXNS": ("explain there are no charges available for review", []),
    "SUMMARIZE": ("summarize the recorded responses and completed card actions", ["recorded", "actions"]),
    "ASK_MORE_HELP": ("ask whether the customer needs anything else", []),
    "TRANSFER": ("hand the customer over to a human representative", ["reason"]),
    "END": ("close the call", []),
}
FIELD_HELP = {
    "txn": "transaction_id of the charge concerned (e.g. \"T1\")",
    "disposition": "RECOGNIZED | UNRECOGNIZED | UNSURE (as recorded)",
    "fee": "the fee in dollars, a number",
    "delivery": "standard_delivery | expedited_delivery | null",
    "action": "SECURE_CARD | REPLACE_CARD",
    "recorded": "object {transaction_id: disposition} of every response actually recorded",
    "actions": "list of [action, delivery] pairs actually completed",
    "reason": "the transfer reason code",
}


def default_policy_text() -> str:
    from ..domain import current
    return current().transcription_path.read_text(encoding="utf-8")


def build_system_prompt(policy_text: str, act_names: list[str] | None = None) -> str:
    acts = {k: v for k, v in ACTS.items() if act_names is None or k in act_names}
    tools = "\n".join(f"- {n}({', '.join(f'{a}: {d}' for a, d in s['args'].items())}) -> {s['returns']}"
                      for n, s in _tool_api().items())
    reasons = "\n".join(f"- {k}: {v}" for k, v in TRANSFER_REASONS.items())
    act_lines = "\n".join(f"- {k}: {d}" + (f"  [fields: {', '.join(f)}]" if f else "") for k, (d, f) in acts.items())
    fields = "\n".join(f"- {k}: {v}" for k, v in FIELD_HELP.items())
    return f"""[{SUT_PROMPT_VERSION}] You are the charge-review voice bot of a card issuer, speaking with a customer on
the phone. Follow the operating procedure below exactly; it is your only source of policy, limits and
knowledge-base answers. Account facts (identity, charges, results of recording or card actions) come only
from tool results - never invent them, and never claim something was recorded or completed unless the tool
returned SUCCESS for exactly that charge or action.

=== OPERATING PROCEDURE ===
{policy_text}
=== END OF PROCEDURE ===

BACKEND TOOLS (call them yourself; their results are added to the conversation):
{tools}

Transfer reason codes for transfer_to_human:
{reasons}

DIALOG ACTS. Every turn you speak is a list of acts, in the order you say them; each act is an object
{{"act": NAME, "text": "the words you say", ...fields}}. The customer answers the LAST act of your turn, so
end a turn with the act you are waiting on (e.g. PRESENT_TXN, ASK_AUTH, CONFIRM_ACTION, ASK_MORE_HELP).
{act_lines}
Field formats:
{fields}

HOW TO REPLY. You work in steps. At each step reply with exactly ONE JSON object and nothing else:
  {{"tool": "<tool name>", "args": {{...}}}}      to call one backend tool (you will then see its result), or
  {{"say": [<act>, ...], "end": false}}           to speak to the customer and wait for their answer, or
  {{"say": [<act>, ...], "end": true}}            to speak and finish the call (after TRANSFER or END).
Call the tool before you speak about its result. To hand over, call transfer_to_human first, then say TRANSFER.
The input you receive is the conversation so far as JSON: customer turns, your earlier turns and every tool
call with its result, plus how many customer turns have happened."""


@dataclass
class LLMBotStats:
    conversations: int = 0
    turns: int = 0
    model_steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    bad_tool_requests: int = 0
    invalid_replies: int = 0
    unknown_turns: int = 0
    step_limit_hits: int = 0
    unknown_acts: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class LLMBot:
    """start() / respond(text) -> BotTurn, like ReferenceBot; tool calls go to `tools` (the harness backend)."""
    tools: Any
    llm: Any                                  # enrich.llm.client.CachedLLM (or anything with invoke(system, user, salt))
    system: str
    max_steps: int = 6
    stats: LLMBotStats = field(default_factory=LLMBotStats)
    act_names: frozenset = frozenset(ACTS)

    def __post_init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.customer_turns = 0
        self.ended = False
        self.stats.conversations += 1

    # ------------------------------------------------------------------ protocol
    def start(self) -> BotTurn:
        self.history.append({"event": "call connected"})
        return self._turn()

    def respond(self, text: str) -> BotTurn:
        if self.ended:
            return BotTurn("Goodbye.", [{"act": "END", "text": "Goodbye."}], True)
        self.customer_turns += 1
        self.history.append({"customer": text})
        return self._turn()

    # ------------------------------------------------------------------ one customer turn
    def _ask(self, feedback: str | None) -> str:
        user = {"customer_turns_so_far": self.customer_turns, "conversation": self.history}
        if feedback:
            user["problem_with_your_last_reply"] = feedback
        self.stats.model_steps += 1
        return self.llm.invoke(self.system, json.dumps(user, ensure_ascii=False), salt="sut")

    def _turn(self) -> BotTurn:
        self.stats.turns += 1
        steps, invalid, feedback = 0, 0, None
        for _ in range(self.max_steps + 4):
            obj = _parse(self._ask(feedback))
            feedback = None
            if obj is None or not ({"tool", "say"} & set(obj)):
                invalid += 1
                self.stats.invalid_replies += 1
                if invalid >= 2:
                    break
                feedback = 'Reply with exactly one JSON object: {"tool": ..., "args": {...}} or {"say": [...], "end": ...}.'
                continue
            if "tool" in obj and "say" not in obj:
                if steps >= self.max_steps:
                    self.stats.step_limit_hits += 1
                    feedback = f"You have made {steps} tool calls this turn; speak to the customer now."
                    continue
                steps += 1
                self._tool(str(obj.get("tool") or ""), obj.get("args") or {})
                continue
            acts = self._acts(obj.get("say"))
            if not acts:
                invalid += 1
                self.stats.invalid_replies += 1
                if invalid >= 2:
                    break
                feedback = '"say" must be a non-empty list of act objects {"act": NAME, "text": ...}.'
                continue
            end = bool(obj.get("end")) or any(a["act"] in ("TRANSFER", "END") for a in acts)
            self.history.append({"bot": [dict(a) for a in acts], "end": end})
            self.ended = end
            return BotTurn(" ".join(a.get("text", "") for a in acts).strip(), acts, end)
        self.stats.unknown_turns += 1             # the model never produced a usable turn: a visible bot defect
        acts = [{"act": "UNKNOWN", "text": "Sorry, I'm having trouble right now."}]
        self.history.append({"bot": acts, "end": False})
        return BotTurn(acts[0]["text"], acts, False)

    def _tool(self, name: str, args: Any) -> None:
        spec = _tool_api().get(name)
        if spec is None or not isinstance(args, dict):
            self.stats.bad_tool_requests += 1
            self.history.append({"tool": name, "args": args, "result": {"error": "unknown tool or bad arguments"}})
            return
        clean = {k: (None if v in ("", "null") else v) for k, v in args.items() if k in spec["args"]}
        clean = {k: v for k, v in clean.items() if v is not None or k == "delivery"}
        if name == "authenticate_customer" and "credential" in clean:
            clean["credential"] = str(clean["credential"])
        self.stats.tool_calls += 1
        try:
            res = self.tools.call(name, **clean)
        except ToolError as e:
            self.stats.tool_errors += 1
            res = {"error": str(e) or "service unavailable"}
        except TypeError as e:                    # arguments the backend does not accept
            self.stats.bad_tool_requests += 1
            res = {"error": f"bad arguments: {e}"}
        self.history.append({"tool": name, "args": clean, "result": res})

    def _acts(self, say: Any) -> list[dict[str, Any]]:
        if isinstance(say, dict):
            say = [say]
        if not isinstance(say, list):
            return []
        out = []
        for a in say:
            if isinstance(a, str):
                a = {"act": "UNKNOWN", "text": a}
            if not isinstance(a, dict) or not a.get("act"):
                continue
            name = str(a["act"]).strip().upper()
            if name not in self.act_names:
                self.stats.unknown_acts += 1
            act = {"act": name, "text": str(a.get("text") or "")}
            for f in ACTS.get(name, ("", []))[1]:
                if a.get(f) is not None:
                    v = a[f]
                    if f == "actions" and isinstance(v, list):
                        v = [list(x) if isinstance(x, (list, tuple)) else [x, None] for x in v]
                    act[f] = v
            out.append(act)
        return out


def _parse(text: str) -> dict[str, Any] | None:
    from ..enrich.llm.client import parse_json_object
    return parse_json_object(text or "")


def make_llm_factory(store=None, settings=None, llm=None, policy_text: str | None = None, max_steps: int = 6):
    """A bot_factory for CampaignConfig / JourneyRunner. One CachedLLM (tag "sut") is shared by every
    conversation, so the cache and the call statistics cover the whole campaign: ``factory.llm.report()``,
    ``factory.stats.as_dict()``. settings default: the gateway's default deployment at temperature 0."""
    from ..enrich.llm.client import CachedLLM, LLMSettings
    if llm is None:
        s = settings or LLMSettings(temperature=0.0, max_tokens=900)
        llm = CachedLLM(s, tag="sut")
    kg = getattr(store, "kg", None) if store is not None else None
    names = [s for s in getattr(kg, "states", []) or []] or list(ACTS)
    system = build_system_prompt(policy_text or default_policy_text(), [n for n in ACTS if n in names])
    stats = LLMBotStats()

    def factory(tools):
        return LLMBot(tools, llm, system, max_steps, stats, frozenset(n for n in ACTS if n in names))
    factory.llm, factory.stats, factory.system_prompt = llm, stats, system
    return factory
