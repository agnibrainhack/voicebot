"""An LLM bot under test: a language model given the procedure, the tools and the dialog acts.

Parts of this file:
  1. ``LocalChat``: a local model (Llama 3.1 8B Instruct or any chat model vLLM serves), in this process or through
     an OpenAI-compatible vLLM server; ``status`` says whether it can be used.
  2. ``LLMBot`` and ``make_llm_factory``: the system prompt (the tester's instructions, attached references, then
     the tools, transfer reasons, acts and reply format the harness needs), one JSON step per model call (a tool
     call or the acts it says), and the critique layer: a second call reviews every proposed step and approves it
     or returns a corrected one.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# =====================================================================================================================
# PART 1 - A local model behind the LLM bot under test: Llama 3.1 8B Instruct (or any chat model vLLM serves) loaded with vLLM
#
# from a folder on this machine. For campaign runs only: the console's Playground keeps to the gateway, so a chat
# never loads an 8B model into the console's process.
#
#     GMSJOURNEY_SUT_LOCAL_MODEL=/models/Meta-Llama-3.1-8B-Instruct     the model folder (set it in the notebook)
#
# Optional knobs (environment): ``GMSJOURNEY_VLLM_GPU_UTIL`` (fraction of GPU memory vLLM may take, default 0.80;
# lower it when the knowledge store shares the GPU), ``GMSJOURNEY_VLLM_MAX_LEN`` (context length, default 16384: the
# bot's prompt is about 6k tokens and the conversation grows from there), ``GMSJOURNEY_VLLM_TP`` (tensor-parallel
# GPUs, default 1), ``GMSJOURNEY_VLLM_DTYPE`` (default auto), ``GMSJOURNEY_VLLM_URL`` (an OpenAI-compatible vLLM server,
# e.g. http://127.0.0.1:8000/v1, used instead of loading the model in-process; the folder is then the served model's
# name or path).
#
# ``LocalChat.invoke(messages) -> reply`` has the shape ``CachedLLM`` expects, so the bot, its replay cache (keyed by
# ``local-<folder name>``) and its statistics work as with the gateway. Generation is greedy at temperature 0, so a
# re-run of the same conversation gives the same reply.
# =====================================================================================================================

ENV = "GMSJOURNEY_SUT_LOCAL_MODEL"


def local_model_path() -> str:
    """The local model folder from GMSJOURNEY_SUT_LOCAL_MODEL (empty when not set)."""
    return os.environ.get(ENV, "").strip()


def status() -> dict[str, Any]:
    """What the console shows about the local model, without loading it."""
    import importlib.util
    path = local_model_path()
    url = os.environ.get("GMSJOURNEY_VLLM_URL", "").strip()
    p = Path(path).expanduser() if path else None
    folder_ok = bool(p and p.is_dir() and (p / "config.json").exists())
    vllm = importlib.util.find_spec("vllm") is not None
    if not path:
        why = f"set {ENV} to the model folder in the notebook (before console.show()) to enable it"
    elif url:
        why = f"served by the vLLM server at {url}"
    elif not folder_ok:
        why = f"{path} is not a model folder (it needs config.json and the weights)"
    elif not vllm:
        why = "vLLM is not installed in this kernel's Python (pip install vllm)"
    else:
        why = "ready: loaded with vLLM when a campaign starts"
    return {"env": ENV, "path": path, "name": Path(path).name if path else None, "url": url or None,
            "folder_ok": folder_ok, "vllm_installed": vllm, "usable": bool(path) and (bool(url) or (folder_ok and vllm)),
            "detail": why}


def _messages(msgs: list[Any]) -> list[dict[str, str]]:
    from ..llm import _role, _text
    return [{"role": _role(m), "content": _text(m)} for m in msgs]


class LocalChat:
    """A chat model on vLLM: in-process from a folder, or through an OpenAI-compatible vLLM server."""

    def __init__(self, path: str | None = None, temperature: float = 0.0, max_tokens: int = 900, engine: Any = None):
        self.path = path or local_model_path()
        if not self.path:
            raise RuntimeError(f"no local model: set {ENV} to the model folder")
        self.temperature, self.max_tokens = float(temperature), int(max_tokens)
        self.url = os.environ.get("GMSJOURNEY_VLLM_URL", "").strip().rstrip("/")
        self._engine = engine                       # tests pass a stand-in with .chat(messages, params)
        self.calls, self.seconds = 0, 0.0

    @property
    def name(self) -> str:
        return "local-" + Path(self.path).name

    def _llm(self):
        if self._engine is None:
            p = Path(self.path).expanduser()
            if not (p.is_dir() and (p / "config.json").exists()):
                raise RuntimeError(f"{self.path} is not a model folder (config.json missing)")
            try:
                from vllm import LLM
            except ImportError as e:
                raise RuntimeError("vLLM is not installed (pip install vllm), or set GMSJOURNEY_VLLM_URL to a vLLM server") from e
            self._engine = LLM(model=str(p), tensor_parallel_size=int(os.environ.get("GMSJOURNEY_VLLM_TP", "1")),
                               gpu_memory_utilization=float(os.environ.get("GMSJOURNEY_VLLM_GPU_UTIL", "0.80")),
                               max_model_len=int(os.environ.get("GMSJOURNEY_VLLM_MAX_LEN", "16384")),
                               dtype=os.environ.get("GMSJOURNEY_VLLM_DTYPE", "auto"), seed=0)
        return self._engine

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        msgs = _messages(messages)
        t0 = time.time()
        if self.url:
            text, usage = self._server(msgs)
        else:
            try:
                from vllm import SamplingParams
                params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, seed=0)
            except ImportError:
                params = {"temperature": self.temperature, "max_tokens": self.max_tokens}
            eng = self._llm()
            try:
                out = eng.chat(msgs, params, use_tqdm=False)
            except TypeError:                           # an engine without the progress-bar argument
                out = eng.chat(msgs, params)
            o = out[0]
            text = o.outputs[0].text
            usage = {"input_tokens": len(getattr(o, "prompt_token_ids", None) or []),
                     "output_tokens": len(getattr(o.outputs[0], "token_ids", None) or [])}
        self.calls += 1
        self.seconds += time.time() - t0
        return SimpleNamespace(content=text, usage_metadata=usage)

    def _server(self, msgs: list[dict[str, str]]) -> tuple[str, dict]:
        body = json.dumps({"model": self.path, "messages": msgs, "temperature": self.temperature,
                           "max_tokens": self.max_tokens, "seed": 0}).encode()
        req = urllib.request.Request(f"{self.url}/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.loads(r.read())
        u = j.get("usage") or {}
        return j["choices"][0]["message"]["content"], {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}


# =====================================================================================================================
# PART 2 - An LLM-driven charge-review voice bot: a second system under test, next to the keyword/state-machine
#
# ``ReferenceBot``.
#
# The bot is a tool-using agent on the same Microsoft model gateway and replay cache as the LLM enrichment arm
# (``llm.py``, ``CachedLLM`` with its own cache tag ``sut``). It is given what a production team
# would give an LLM bot - and nothing the oracle computes:
#
# * the operating procedure (``policy/charge_review/charge_review_policy.md``: limits, delivery options, tools, workflow,
#   rules, knowledge-base sentences) as its instructions;
# * the backend tools' API (names, arguments, results, and the transfer reason codes the
#   ``transfer_to_human`` API takes);
# * the dialog-act inventory its turns must be labelled with (the act names of the policy graph, the same
#   inventory the reference bot uses), so the harness can read what it claims - which is graded against
#   what its tool calls actually did.
#
# Each step the model answers with ONE JSON object: ``{"tool": name, "args": {...}}`` (the call is made on
# the harness's mock backend, so faults, governance gates and the trace see it) or ``{"say": [acts...],
# "end": bool}`` (the turn the customer hears). Up to ``max_steps`` tool calls per customer turn; a reply
# that is not valid JSON is sent back once with the reason; a second failure produces a visible
# ``UNKNOWN`` turn (a defect of the bot, graded as such - never repaired by the harness).
#
# Replies are cached by content hash (temperature 0 by default), so a campaign re-runs byte for byte with
# ``mode="replay"`` and the harness's repeated runs of the same conversation (turn-by-turn export) cost
# nothing. ``LLMUnavailable`` (no credential, gateway down, replay miss) is raised, never hidden.
# =====================================================================================================================

from .reference import BotTurn
from .tools import ToolError

SUT_PROMPT_VERSION = "llm-sut/2"

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
    "PRESENT_TXN": ("read out ONE returned charge (merchant, amount, date) and ask whether the customer recognizes it",
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
    """The operating procedure the bot is given: the pack's policy transcription when it has one, else the policy
    graph rendered as text (what a team without a separate SOP would hand an LLM bot)."""
    from ..packs.loading import current
    dom = current()
    p = dom.transcription_path
    if p is not None and Path(p).exists():
        return Path(p).read_text(encoding="utf-8")
    kg = dom.kg
    if hasattr(kg, "to_markdown"):
        return kg.to_markdown()
    return json.dumps(json.loads(Path(dom.graph_path).read_text()), indent=1)


def journey_title() -> str:
    """The active journey's title, for the LLM bot's prompt."""
    from ..packs.loading import current
    dom = current()
    g = dom.graph
    return (getattr(g, "title", None) if g is not None else None) or dom.get("domain.title") or dom.name.replace("_", " ")


def act_inventory(store=None) -> dict[str, tuple[str, list[str]]]:
    """name -> (what the act is for, the fields it carries): the dialog states of the active pack's graph. A state
    the charge-review inventory above describes keeps that description; any other is described from the graph
    (whether it waits for the customer) and the pack's own wording of it (domain.yaml bot.says)."""
    from ..packs.loading import current
    dom = current()
    kg = getattr(store, "kg", None) or dom.kg
    states = [t["head"] for t in kg.where("is_a", tail="dialog_state")]
    if not states:
        return dict(ACTS)
    carried: dict[str, list[str]] = {}
    for t in kg.where("carries"):
        if t["tail"] not in ("text", "state"):
            carried.setdefault(t["head"], []).append(t["tail"])
    prompts = set(getattr(kg, "prompt_states", None) or [])
    says = dom.get("bot.says") or {}
    out = {}
    for s in states:
        known = ACTS.get(s)
        fields = list(dict.fromkeys(carried.get(s, []) + (known[1] if known else [])))
        if known:
            desc = known[0]
        else:
            desc = ("ask the customer and wait for the answer" if s in prompts else "tell the customer") + \
                   f" ({s.lower().replace('_', ' ')})"
            if says.get(s) and "{text}" not in str(says[s]):
                desc += f'; for example: "{says[s]}"'
        out[s] = (desc, fields)
    return out


def transfer_reasons(store=None) -> dict[str, str]:
    """The reason codes the transfer tool takes: the graph's (``<transfer tool> transfer_when <reason>``) when it
    names them, else the charge-review codes above."""
    from ..packs.loading import current
    dom = current()
    kg = getattr(store, "kg", None) or dom.kg
    api = _tool_api()
    tools = {t["head"] for t in kg.where("transfer_when")} & set(api)
    if tools:
        tool = sorted(tools)[0]
        def say(code: str) -> str:              # the description above only where every tool it names is this pack's
            d = TRANSFER_REASONS.get(code)
            named = re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", d or "")
            return d if d and all(n in api for n in named) else code.replace("_", " ")
        return {t["tail"]: say(t["tail"]) for t in kg.where("transfer_when", head=tool)}
    return dict(TRANSFER_REASONS)


def _allowed(name: str, tool: str | None = None) -> str:
    """' (one of: A | B)' when the graph names the values an argument or act field takes, else ''."""
    from ..packs.loading import current
    g = current().graph
    if g is None:
        return ""
    vs = None
    if tool is not None:
        vs = ((g.tools.get(tool) or {}).get("args") or {}).get(name)
        vs = vs if vs in g.values else None
    vs = vs or g.slot_set(name)
    members = g.members(vs) if vs and vs in g.values else []
    return f" (one of: {' | '.join(members)})" if members and len(members) <= 12 else ""


def suggested_instructions(policy_text: str | None = None) -> str:
    """The part of the bot's prompt a tester may rewrite: who the bot is, how it must behave, and the operating
    procedure. The console shows it as the suggested prompt."""
    return f"""You are the voice bot for "{journey_title()}", speaking with a customer on the phone. Follow the
operating procedure below exactly; it is your only source of policy, limits and knowledge-base answers. Account
facts (identity, items, results of recording or actions) come only from tool results - never invent them, and
never claim something was recorded or completed unless the tool returned success for exactly that item or action.

=== OPERATING PROCEDURE ===
{policy_text if policy_text is not None else default_policy_text()}
=== END OF PROCEDURE ==="""


def references_text(references: list[dict] | None) -> str:
    """Files the tester attached, as reference material after the instructions."""
    parts = []
    for r in references or []:
        txt = str(r.get("text") or "").strip()
        if txt:
            parts.append(f"=== REFERENCE MATERIAL: {r.get('name') or 'attachment'} ===\n{txt}\n=== END OF {r.get('name') or 'attachment'} ===")
    if not parts:
        return ""
    return ("Reference material supplied with these instructions. Use it to answer and to decide, unless it "
            "contradicts the operating procedure above, which takes precedence.\n\n" + "\n\n".join(parts))


def protocol_text(acts: dict[str, tuple[str, list[str]]] | None = None, reasons: dict[str, str] | None = None) -> str:
    """The fixed part of the prompt: the tools, the transfer reasons, the dialog acts and the JSON reply format.
    The harness needs it to run the bot's tool calls and to read what it says, so it is not editable."""
    acts = acts if acts is not None else act_inventory()
    reasons = reasons if reasons is not None else transfer_reasons()
    tools = "\n".join(f"- {n}({', '.join(f'{a}: {d}' + _allowed(a, n) for a, d in s['args'].items())}) -> {s['returns']}"
                      for n, s in _tool_api().items())
    reason_lines = "\n".join(f"- {k}: {v}" for k, v in reasons.items())
    act_lines = "\n".join(f"- {k}: {d}" + (f"  [fields: {', '.join(f)}]" if f else "") for k, (d, f) in acts.items())
    used = {f for _, fs in acts.values() for f in fs}
    fields = "\n".join(f"- {k}: {FIELD_HELP.get(k, 'the value as the policy names it')}" + _allowed(k)
                       for k in sorted(used))
    return f"""[{SUT_PROMPT_VERSION}] BACKEND TOOLS (call them yourself; their results are added to the conversation):
{tools}

Transfer reason codes for the transfer tool:
{reason_lines}

DIALOG ACTS. Every turn you speak is a list of acts, in the order you say them; each act is an object
{{"act": NAME, "text": "the words you say", ...fields}}. The customer answers the LAST act of your turn, so
end a turn with the act you are waiting on (a question to the customer).
{act_lines}
Field formats:
{fields}

HOW TO REPLY. You work in steps. At each step reply with exactly ONE JSON object and nothing else:
  {{"tool": "<tool name>", "args": {{...}}}}      to call one backend tool (you will then see its result), or
  {{"say": [<act>, ...], "end": false}}           to speak to the customer and wait for their answer, or
  {{"say": [<act>, ...], "end": true}}            to speak and finish the call (after TRANSFER or END).
Call the tool before you speak about its result. To hand over, call the transfer tool first, then say TRANSFER.
The input you receive is the conversation so far as JSON: customer turns, your earlier turns and every tool
call with its result, plus how many customer turns have happened."""


def build_system_prompt(policy_text: str | None = None, act_names: list[str] | None = None,
                        instructions: str | None = None, references: list[dict] | None = None,
                        acts: dict | None = None, reasons: dict | None = None) -> str:
    """instructions (default: the suggested ones on ``policy_text``) + attached references + the fixed protocol."""
    acts = acts if acts is not None else act_inventory()
    if act_names is not None:
        acts = {k: v for k, v in acts.items() if k in act_names}
    head = instructions if instructions and instructions.strip() else suggested_instructions(policy_text)
    ref = references_text(references)
    return "\n\n".join(x for x in (head.strip(), ref, protocol_text(acts, reasons)) if x)


@dataclass
class LLMBotStats:
    """Counters over a whole campaign of LLM-bot conversations (model steps, tool calls and errors, invalid
    replies, the critique layer's reviews and corrections); the corrections themselves are kept too.
    """
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
    critiques: int = 0                         # steps the critique layer reviewed
    critique_approved: int = 0
    critique_revised: int = 0                  # steps it replaced with a corrected one
    critique_unusable: int = 0                 # a review that could not be read, or a correction that was not a valid step

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def note_revision(self, entry: dict[str, Any], keep: int = 200) -> None:
        log = self.__dict__.setdefault("_revisions", [])
        if len(log) < keep:
            log.append(entry)

    def revisions(self) -> list[dict[str, Any]]:
        return list(self.__dict__.get("_revisions", []))


CRITIC_VERSION = "critic/1"

CRITIC_INSTRUCTIONS = """You are the quality reviewer for a customer-service voice bot. Before each step the bot proposes is carried
out, you check it against the bot's own instructions (below, between the markers) and the conversation so far.

The bot's step is either a tool call {"tool": NAME, "args": {...}} or a reply {"say": [acts...], "end": true|false}.

Check a TOOL CALL for: the right tool at this point of the procedure; its preconditions met (for example the caller is
verified before any account tool); every argument taken exactly from what the customer said or a tool returned (no
invented, guessed or reformatted ids, codes, amounts or choices); no repeat of a call that already succeeded; limits
and attempt counts respected; no tool call after a transfer.

Check a REPLY for: acts the procedure allows at this point, with the act names and fields of the reply format; every
fact (ids, amounts, names, statuses, figures, fees, dates) exactly as the tool results and the policy state them; no
claim that something was done unless a tool result says it succeeded; required confirmations and disclosures made
before acting; a transfer whenever a rule requires one, and no early transfer or ending when the procedure continues;
one question at a time; nothing promised that the procedure does not allow.

Be conservative: approve unless the step clearly breaks the instructions or the facts. When it does, give the corrected
step - the step the bot should take instead, in exactly the bot's reply format (a tool call or a reply, whichever is
right now; it may be a different kind of step).

Reply with exactly one JSON object and nothing else:
  {"verdict": "approve"}
  {"verdict": "revise", "reason": "<one sentence>", "step": <the corrected step>}"""


def critic_system(bot_system: str) -> str:
    """The critique layer's system prompt: the reviewer's instructions, then the bot's own instructions."""
    return (CRITIC_INSTRUCTIONS + "\n\n===== THE BOT'S INSTRUCTIONS =====\n" + bot_system
            + "\n===== END OF THE BOT'S INSTRUCTIONS =====")


@dataclass
class LLMBot:
    """start() / respond(text) -> BotTurn, like ReferenceBot; tool calls go to `tools` (the harness backend)."""
    tools: Any
    llm: Any                                  # enrich.llm.client.CachedLLM (or anything with invoke(system, user, salt))
    system: str
    max_steps: int = 6
    stats: LLMBotStats = field(default_factory=LLMBotStats)
    act_names: frozenset = frozenset(ACTS)
    act_fields: dict | None = None            # act -> the fields it carries (default: the charge-review inventory)
    critic: Any = None                        # an LLM (same interface as llm) that reviews every step; None = off
    critic_system: str = ""

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

    # ------------------------------------------------------------------ critique layer
    def _valid(self, obj: Any) -> bool:
        if not isinstance(obj, dict) or not ({"tool", "say"} & set(obj)):
            return False
        if "tool" in obj and "say" not in obj:
            return bool(obj.get("tool")) and isinstance(obj.get("args", {}), dict)
        return bool(self._acts(obj.get("say"), count=False))

    def _review(self, obj: dict[str, Any]) -> dict[str, Any]:
        """The step to carry out: ``obj``, or the critic's correction of it. One review per step (a correction is not
        reviewed again), so the layer costs one extra call per step."""
        if self.critic is None:
            return obj
        self.stats.critiques += 1
        user = {"customer_turns_so_far": self.customer_turns, "conversation": self.history, "proposed_step": obj}
        try:
            out = _parse(self.critic.invoke(self.critic_system, json.dumps(user, ensure_ascii=False), salt="critique"))
        except Exception:                         # noqa: BLE001 - a reviewer outage must not stop the bot
            out = None
        verdict = str((out or {}).get("verdict") or "").lower()
        if verdict == "approve":
            self.stats.critique_approved += 1
            return obj
        step = (out or {}).get("step")
        if verdict == "revise" and self._valid(step):
            self.stats.critique_revised += 1
            self.stats.note_revision({"turn": self.customer_turns, "before": obj, "after": step,
                                      "reason": str((out or {}).get("reason") or "")[:300]})
            return step
        self.stats.critique_unusable += 1
        return obj

    def _turn(self) -> BotTurn:
        self.stats.turns += 1
        steps, invalid, feedback = 0, 0, None
        for _ in range(self.max_steps + 4):
            obj = _parse(self._ask(feedback))
            feedback = None
            if self._valid(obj):
                obj = self._review(obj)
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
        for k, v in list(clean.items()):           # codes and ids are strings to the backend (a 4-digit code read as a number)
            if isinstance(v, int) and not isinstance(v, bool) and spec["args"].get(k, "").lower().find("number") < 0:
                clean[k] = str(v)
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

    def _acts(self, say: Any, count: bool = True) -> list[dict[str, Any]]:
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
            if name not in self.act_names and count:
                self.stats.unknown_acts += 1
            act = {"act": name, "text": str(a.get("text") or "")}
            fields = (self.act_fields or {}).get(name) if self.act_fields else ACTS.get(name, ("", []))[1]
            for f in fields or []:
                if a.get(f) is not None:
                    v = a[f]
                    if f == "actions" and isinstance(v, list):
                        v = [list(x) if isinstance(x, (list, tuple)) else [x, None] for x in v]
                    act[f] = v
            out.append(act)
        return out


def _parse(text: str) -> dict[str, Any] | None:
    from ..llm import parse_json_object
    return parse_json_object(text or "")


def fake_model():
    """A stand-in chat model named by ``GMSJOURNEY_SUT_FAKE_MODEL=module:Class`` (tests and demos without the
    gateway), else None."""
    spec = os.environ.get("GMSJOURNEY_SUT_FAKE_MODEL", "").strip()
    if not spec:
        return None
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)()


def make_llm_factory(store=None, settings=None, llm=None, policy_text: str | None = None, max_steps: int = 6,
                     instructions: str | None = None, references: list[dict] | None = None, critique: bool = False,
                     critic_llm=None):
    """A bot_factory for CampaignConfig / JourneyRunner. One CachedLLM (tag "sut") is shared by every
    conversation, so the cache and the call statistics cover the whole campaign: ``factory.llm.report()``,
    ``factory.stats.as_dict()``. settings default: the gateway's default deployment at temperature 0.
    ``instructions`` replaces the suggested instructions (who the bot is and its procedure); ``references`` are
    attached files ({name, text}) added after them; the tools, acts and reply format are always appended.

    ``critique`` adds the critique layer: a second call (``critic_llm``, default the same model) reviews every
    proposed step - each tool call and each reply - against the same instructions and approves it or returns a
    corrected step, which the bot carries out instead (``LLMBot._review``)."""
    from ..llm import CachedLLM, LLMSettings
    if llm is None:
        s = settings or LLMSettings(temperature=0.0, max_tokens=900)
        fake = fake_model()
        if fake is not None:
            s.model = f"fake-{type(fake).__name__}"
        llm = CachedLLM(s, model=fake, tag="sut") if fake is not None else CachedLLM(s, tag="sut")
    acts = act_inventory(store)
    reasons = transfer_reasons(store)
    system = build_system_prompt(policy_text, None, instructions, references, acts, reasons)
    stats = LLMBotStats()
    names = frozenset(acts)

    critic = (critic_llm or llm) if critique else None
    csys = critic_system(system) if critique else ""

    def factory(tools):
        return LLMBot(tools, llm, system, max_steps, stats, names, {k: f for k, (_, f) in acts.items()},
                      critic=critic, critic_system=csys)
    factory.llm, factory.stats, factory.system_prompt = llm, stats, system
    factory.critique, factory.critic_llm, factory.critic_prompt = bool(critique), critic, csys
    return factory
