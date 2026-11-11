"""The LLM bot under test: a language model reads the customer and words the bot; the journey's policy graph decides.

Parts of this file:
  1. ``LocalChat``: a local model (gpt-oss, Llama or any chat model vLLM serves), in this process or through an
     OpenAI-compatible vLLM server; ``status`` says whether it can be used.
  2. ``StructuredBot`` and ``make_structured_factory``: each customer turn is read by the LLM into a kind (actionable,
     question, claim, social, unclear), an intent and slots chosen from what the current state of the graph accepts;
     the graph's state machine calls the allowed tools and decides the acts; the LLM words those acts, each line
     checked against its facts (the template is said when a check fails).
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


def status(path: str | None = None, url: str | None = None) -> dict[str, Any]:
    """What the console shows about a local model, without loading it. ``path`` (the model folder) and ``url`` (an
    OpenAI-compatible vLLM server) default to GMSJOURNEY_SUT_LOCAL_MODEL and GMSJOURNEY_VLLM_URL."""
    import importlib.util
    path = (path or "").strip() or local_model_path()
    url = (url or "").strip() or os.environ.get("GMSJOURNEY_VLLM_URL", "").strip()
    p = Path(path).expanduser() if path else None
    folder_ok = bool(p and p.is_dir() and (p / "config.json").exists())
    vllm = importlib.util.find_spec("vllm") is not None
    if not path:
        why = "type the model folder (or set " + ENV + " before starting the console)"
    elif url:
        why = f"served by the vLLM server at {url}"
    elif p is not None and not p.exists():
        why = f"{path} does not exist on this machine"
    elif not folder_ok:
        why = f"{path} is not a model folder (it needs config.json and the weights)"
    elif not vllm:
        why = "vLLM is not installed in this kernel's Python (pip install vllm)"
    else:
        why = "ready: loaded with vLLM when a test run starts"
    return {"env": ENV, "path": path, "name": Path(path).name if path else None, "url": url or None,
            "folder_ok": folder_ok, "vllm_installed": vllm, "usable": bool(path) and (bool(url) or (folder_ok and vllm)),
            "detail": why}


def _messages(msgs: list[Any]) -> list[dict[str, str]]:
    from ..llm import _role, _text
    return [{"role": _role(m), "content": _text(m)} for m in msgs]


def is_reasoning_model(name: str | None) -> bool:
    """A model that thinks before it answers (gpt-oss), by its folder or served name."""
    n = (name or "").lower()
    return "gpt-oss" in n or "gpt_oss" in n


_ENGINES: dict[str, Any] = {}                       # one vLLM engine per model folder, shared by every client
_ENGINE_LOCK = __import__("threading").Lock()        # vLLM's in-process engine is not safe to call from two threads


class LocalChat:
    """A chat model on vLLM: in-process from a folder, or through an OpenAI-compatible vLLM server. Every client of
    the same folder shares one engine (the bot under test, its critique, the rewording and the judges load the
    weights once), and in-process calls are serialised."""

    def __init__(self, path: str | None = None, temperature: float = 0.0, max_tokens: int = 900, engine: Any = None,
                 url: str | None = None):
        self.path = (path or "").strip() or local_model_path()
        if not self.path:
            raise RuntimeError(f"no local model: give the model folder (or set {ENV})")
        self.temperature, self.max_tokens = float(temperature), int(max_tokens)
        self.url = ((url or "").strip() or os.environ.get("GMSJOURNEY_VLLM_URL", "").strip()).rstrip("/")
        self._engine = engine                       # tests pass a stand-in with .chat(messages, params)
        self.calls, self.seconds = 0, 0.0

    @property
    def name(self) -> str:
        return "local-" + Path(self.path).name

    def _llm(self):
        if self._engine is None:
            key = str(Path(self.path).expanduser().resolve())
            with _ENGINE_LOCK:
                if key not in _ENGINES:
                    _ENGINES[key] = self._load()
                self._engine = _ENGINES[key]
        return self._engine

    def _load(self):
        p = Path(self.path).expanduser()
        if not (p.is_dir() and (p / "config.json").exists()):
            raise RuntimeError(f"{self.path} is not a model folder (config.json missing)")
        try:
            from vllm import LLM
        except ImportError as e:
            raise RuntimeError("vLLM is not installed (pip install vllm), or set GMSJOURNEY_VLLM_URL to a vLLM server") from e
        return LLM(model=str(p), tensor_parallel_size=int(os.environ.get("GMSJOURNEY_VLLM_TP", "1")),
                   gpu_memory_utilization=float(os.environ.get("GMSJOURNEY_VLLM_GPU_UTIL", "0.80")),
                   max_model_len=int(os.environ.get("GMSJOURNEY_VLLM_MAX_LEN", "16384")),
                   dtype=os.environ.get("GMSJOURNEY_VLLM_DTYPE", "auto"), seed=0)

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        msgs = _messages(messages)
        t0 = time.time()
        if self.url:
            text, usage = self._server(msgs)
        else:
            try:
                from vllm import SamplingParams
                params = SamplingParams(temperature=self.temperature, max_tokens=self._budget(), seed=0)
            except ImportError:
                params = {"temperature": self.temperature, "max_tokens": self._budget()}
            eng = self._llm()
            with _ENGINE_LOCK:
                try:
                    out = eng.chat(msgs, params, use_tqdm=False)
                except TypeError:                       # an engine without the progress-bar argument
                    out = eng.chat(msgs, params)
            o = out[0]
            text = o.outputs[0].text
            if is_reasoning_model(self.path) and "assistantfinal" in text:   # harmony output: keep the final answer
                text = text.rsplit("assistantfinal", 1)[1].strip()
            usage = {"input_tokens": len(getattr(o, "prompt_token_ids", None) or []),
                     "output_tokens": len(getattr(o.outputs[0], "token_ids", None) or [])}
        self.calls += 1
        self.seconds += time.time() - t0
        return SimpleNamespace(content=text, usage_metadata=usage)

    def _budget(self) -> int:
        """Output tokens: the answer's, plus a thinking budget for a reasoning model (see ``_body``)."""
        if is_reasoning_model(self.path) or os.environ.get("GMSJOURNEY_VLLM_REASONING_EFFORT", "").strip():
            return self.max_tokens + int(os.environ.get("GMSJOURNEY_VLLM_REASONING_TOKENS", "1024"))
        return self.max_tokens

    def _body(self, msgs: list[dict[str, str]]) -> dict[str, Any]:
        """The chat request. A reasoning model (gpt-oss) thinks before it answers and its thinking spends output
        tokens: it is asked for ``reasoning_effort`` (GMSJOURNEY_VLLM_REASONING_EFFORT, default "low") and given
        a thinking budget on top of the answer's tokens (GMSJOURNEY_VLLM_REASONING_TOKENS, default 1024), so a
        300-token judge verdict is not cut off by its own reasoning. GMSJOURNEY_VLLM_EXTRA_BODY (JSON) adds any
        other field the server takes."""
        body: dict[str, Any] = {"model": self.path, "messages": msgs, "temperature": self.temperature,
                                "max_tokens": self.max_tokens, "seed": 0}
        effort = os.environ.get("GMSJOURNEY_VLLM_REASONING_EFFORT", "").strip()
        if effort or is_reasoning_model(self.path):
            body["reasoning_effort"] = effort or "low"
            body["max_tokens"] = self._budget()
        extra = os.environ.get("GMSJOURNEY_VLLM_EXTRA_BODY", "").strip()
        if extra:
            body.update(json.loads(extra))
        return body

    _DROPPED: dict[str, set] = {}                     # per server: optional fields it turned out to reject

    def _server(self, msgs: list[dict[str, str]]) -> tuple[str, dict]:
        import urllib.error
        dropped = LocalChat._DROPPED.setdefault(self.url, set())
        while True:
            body = {k: v for k, v in self._body(msgs).items() if k not in dropped}
            req = urllib.request.Request(f"{self.url}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    j = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:800]
                # an optional field the server does not take (an older vLLM, another model): sent once more without
                # it, and not sent to this server again; anything else is the server's own error, shown as it is
                opt = [k for k in ("reasoning_effort", "seed") if k in body and k not in dropped
                       and (e.code in (400, 422) or k in detail)]
                if opt:
                    dropped.add(opt[0])
                    continue
                raise RuntimeError(f"the vLLM server at {self.url} answered HTTP {e.code}: {detail or e.reason}"
                                   + (f" (sent without: {', '.join(sorted(dropped))})" if dropped else "")) from None
        u = j.get("usage") or {}
        m = j["choices"][0]["message"]
        # a reasoning model's answer is "content" (its thinking, "reasoning_content", is not the answer); when the
        # budget ran out mid-thought there is no answer: an empty reply, which the caller treats as unreadable
        return m.get("content") or "", {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}


# =====================================================================================================================
# PART 2 - The LLM bot under test, structured: the model reads the customer and words the bot; the policy graph decides
#
#     customer speech
#       -> read it (LLM, a small prompt): kind, intent and slots, chosen from what the current state accepts  LLMReader
#       -> the graph's state machine (deterministic): react, the allowed tool, its result, the next state      GraphBot
#       -> the acts the graph requires there (deterministic)
#       -> word them (LLM, checked against the acts' facts; the template when a check fails or no LLM)         Wording
#       -> customer
#
# How a customer turn is read (``kind``):
#   actionable  an answer the state accepts, a request for a person, or the end: its intent and slots. Every value must
#               be one of the graph's members (an alias is mapped to its member) and a code must appear in the words
#   question    a policy question about a figure or a limit the graph states: answered from the graph, with exactly the
#               fact asked (about anything else: "the policy states nothing")
#   claim       the customer states a policy fact ("the fee is waived, right?"): never agreed to as said; answered like
#               a question, from the graph ("Actually, ..." when the customer's figure is not the graph's)
#   social      small talk, thanks, apologies: no state change and no policy; the pending question is asked again (after
#               a short acknowledgement when the wording LLM is on)
#   unclear     nothing the state accepts: the graph's clarification
# A reading that names an intent, a slot or a value the state does not accept is not used: the turn is read as unclear
# and the rejection is counted (``StructuredStats``).
#
# The policy never goes into a prompt: each customer turn costs one reading call (the state's menu, the bot's last
# words, the customer's words: well under 1k tokens) and at most one wording call (the acts, their facts and the
# template lines). A wording that adds or drops a fact, or adds a figure, an option, a promise or a policy word, is
# not used: the template line is said instead.
# =====================================================================================================================

from ..s04_cases.meaning import SemanticFrame
from .reference import BotTurn, GraphBot, KeywordReader, _say, _word, turn_order

STRUCTURED_VERSION = "structured-bot/1"
KINDS = ("actionable", "question", "claim", "social", "unclear")
SOCIAL_ACT = "CHITCHAT"

READ_SYSTEM = """You read one customer turn in a customer-service phone call and say what it means, as JSON only.

You get the bot's last words, the customer's words and a MENU of what the conversation can take at this point.
Choose only from the menu: never invent an intent, a slot, a value or a fact.

kind:
  "actionable"  the customer answers or acts: "intent" is one of the menu's intents and "slots" gives its slots,
                each value exactly one of that slot's listed values (a code exactly as the customer said it, in digits)
  "question"    the customer asks about a policy fact: "about" is one of the menu's facts, or "other"
  "claim"       the customer states a policy fact as true ("the fee is waived, right?", "you said it was free"):
                "about" as for a question, and "claimed" is what they stated, in their words
  "social"      small talk, thanks, an apology or a remark with no answer, question or claim in it
  "unclear"     anything else, or an answer you cannot place on the menu
A turn with social talk and an answer is "actionable": the answer counts. When the customer changes their mind within
the turn, give the final answer. A plain yes or a plain no means what the menu's "yes" and "no" say.

Reply with one JSON object, leaving out the keys the kind does not use:
{"kind": "...", "intent": "...", "slots": {}, "about": "...", "claimed": "..."}"""

WORD_SYSTEM = """You word what a customer-service voice bot says next on a phone call.

You get LINES: each is one thing the bot must say, with its facts and a plain template line. Return every line in
natural spoken words, in the same order, one output line for each input line.
- Say every fact of a line exactly (names, amounts, dates, numbers, options) and add no other fact, figure, option,
  promise or policy. Do not answer or comment on anything that is not in the lines.
- A question stays a question about the same thing.
- "lead" (optional): one short sentence before the first line that acknowledges something social the customer said
  (thanks, small talk, an apology). No facts or promises in it. Leave it empty otherwise.
- Follow the VOICE if one is given (tone only: it never changes what is said).
Reply with JSON only: {"lead": "...", "lines": ["...", "..."]}"""


def journey_title() -> str:
    """The active journey's title."""
    from ..s01_packs.loading import current
    dom = current()
    g = dom.graph
    return (getattr(g, "title", None) if g is not None else None) or dom.get("domain.title") or dom.name.replace("_", " ")


@dataclass
class StructuredStats:
    """Counters over every conversation of a factory (a whole test run, or one Playground chat)."""
    turns: int = 0
    readings: int = 0
    kinds: dict = field(default_factory=dict)          # kind -> customer turns read as it
    rejected: int = 0                                  # readings not used: read as unclear instead
    worded: int = 0                                    # lines worded by the LLM and said
    templated: int = 0                                 # lines said from the template (no wording LLM, or a check failed)
    wording_rejected: int = 0                          # wordings not used because a check failed
    wording_errors: int = 0                            # wording calls that failed (the template was said)
    leads: int = 0                                     # acknowledgements said before the first line
    notes: list = field(default_factory=list)          # the latest rejections, with why (the last 50)

    def __post_init__(self) -> None:
        import threading
        self._lock = threading.Lock()

    def add(self, key: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, key, getattr(self, key) + n)

    def kind(self, k: str) -> None:
        with self._lock:
            self.kinds[k] = self.kinds.get(k, 0) + 1

    def note(self, what: str, why: str, detail: Any = None) -> None:
        with self._lock:
            self.notes.append({"what": what, "why": why, "detail": detail})
            del self.notes[:-50]

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("turns", "readings", "kinds", "rejected", "worded", "templated",
                                              "wording_rejected", "wording_errors", "leads")} | {"notes": list(self.notes)}

    def revisions(self) -> list:
        return []                                      # no critique layer: nothing is revised after the fact


def _num_set(text: str) -> list[str]:
    """The distinct numbers a line says, as digits (number words read, thousands separators dropped), sorted."""
    from ..s07_checks.speech_reading import words_to_digits
    t = words_to_digits(text or "").replace(",", "")
    return sorted({str(float(x)).rstrip("0").rstrip(".") for x in re.findall(r"\d+(?:\.\d+)?", t)})


def _low(text: str) -> str:
    return " ".join(str(text or "").lower().replace("’", "'").split())


class LLMReader:
    """Reads a customer turn with the LLM into a meaning card the graph's state machine takes (see PART 2)."""

    def __init__(self, g, llm, stats: StructuredStats):
        from ..s01_packs.loading import current
        self.g, self.llm, self.stats = g, llm, stats
        self.kw = KeywordReader(g)                     # only to map a value's alias to its member
        self.dom = current()
        self.bot_said = ""
        self.last: dict[str, Any] = {}

    # ------------------------------------------------------------------ the menu
    def _value(self, m: str) -> str:
        say = ((self.dom.get("lexicon.values") or {}).get(m) or {}).get("say") or {}
        hint = str(say.get("canonical") or "").replace("{obj}", "it").strip()
        al = self.g.aliases(m)
        return hint or (al[0] if al else m.lower().replace("_", " "))

    def facts(self) -> dict[str, str]:
        """fact id -> what it is, for every figure and every public limit the graph states."""
        g, out = self.g, {}
        for attr, spec in g.numeric_attributes().items():
            for m in g.members(spec["set"]):
                if g.attr(m, attr) is not None:
                    name = (g.aliases(m) or [m.lower().replace("_", " ")])[0]
                    out[f"value:{attr}:{m}"] = f"the {attr.replace('_usd', '').replace('_', ' ')} of {name}"
        for k, lv in g.limits.items():
            if not lv.get("internal"):
                out[f"limit:{k}"] = k.replace("_", " ")
        return out

    def menu(self, state: str) -> dict[str, Any]:
        from ..s04_cases.meaning import USER_ACTS
        g = self.g
        s = g.states.get(state) or {}
        on = s.get("on") or {}
        intents: dict[str, Any] = {}
        for act in on:
            if act == "END":
                intents[act] = {"means": "the customer has nothing more and ends the call"}
                continue
            slots = {}
            for sl, sp in g.slot_specs(act).items():
                vs = sp["set"]
                d: dict[str, Any] = {"required": not sp.get("optional")}
                if vs in g.values:
                    d["values"] = {m: self._value(m) for m in g.members(vs)}
                else:
                    d["code"] = "the digits the customer says"
                if sp.get("correction_of"):
                    d["only_when"] = f"the customer takes back an earlier {sp['correction_of']} in this turn: that earlier value"
                if sp.get("with"):
                    d["only_with"] = sp["with"]
                slots[sl] = d
            intents[act] = {"means": USER_ACTS.get(act) or act.lower().replace("_", " "), "slots": slots}
        G = g.glob
        if G.get("human"):
            intents[G["human"]["act"]] = {"means": "the customer asks for a human representative"}
        out: dict[str, Any] = {"intents": intents}
        y, n = g.interprets(state)
        for word, t in (("yes", y), ("no", n)):
            if t and t != "UNCLEAR":
                out[word] = self._yes_no(t)
        if G.get("question"):
            out["facts"] = self.facts()
        return out

    def _yes_no(self, target: str) -> str:
        f = self.kw._as(target)
        return f"intent {f.act}" + (f" with {', '.join(f'{k}={v}' for k, v in f.slots)}" if f.slots else "")

    # ------------------------------------------------------------------ reading
    def read(self, text: str, state: str):
        from ..llm import parse_json_object
        menu = self.menu(state)
        user = json.dumps({"state": state, "bot_said": self.bot_said, "customer": text, "menu": menu}, ensure_ascii=False)
        self.stats.add("readings")
        raw = self.llm.invoke(READ_SYSTEM, user, salt=f"read|{STRUCTURED_VERSION}|{state}")
        obj = parse_json_object(raw)
        f, kind, why = self.frame(obj, text, state, menu)
        self.last = {"kind": kind, "why": why, "read": obj, "claimed": (obj or {}).get("claimed") if kind == "claim" else None}
        self.stats.kind(kind)
        if why:
            self.stats.add("rejected")
            self.stats.note("reading", why, {"state": state, "customer": text, "read": obj})
        return f

    def frame(self, obj: Any, text: str, state: str, menu: dict) -> tuple[Any, str, str | None]:
        """(meaning card, kind, why the reading was not used or None)."""
        F = SemanticFrame
        G = self.g.glob
        unclear = F.make(G["unclear"]["act"] if G.get("unclear") else "UNCLEAR")
        if not isinstance(obj, dict):
            return unclear, "unclear", "the reading was not a JSON object"
        kind = str(obj.get("kind") or "").strip().lower()
        if kind == "social":
            return F.make(SOCIAL_ACT), "social", None
        if kind in ("question", "claim"):
            if not G.get("question"):
                return F.make(SOCIAL_ACT), kind, None          # a journey without policy questions: ask again
            about = str(obj.get("about") or "other").strip()
            q = G["question"]["act"]
            if about in (menu.get("facts") or {}):
                parts = about.split(":")
                if parts[0] == "value":
                    return F.make(q, intent="value", attr=parts[1], entity=parts[2]), kind, None
                return F.make(q, intent="limit", key=parts[1]), kind, None
            return F.make(q, intent="unstated_fact"), kind, None if about == "other" else f"no such fact: {about}"
        if kind != "actionable":
            return unclear, "unclear", None
        intent = str(obj.get("intent") or "").strip()
        intents = menu["intents"]
        if intent.lower() in ("yes", "no") and intent.lower() in menu:   # a plain yes / no: what the state says it means
            y, n = self.g.interprets(state)
            f = self.kw._as(y if intent.lower() == "yes" else n)
            if f.act in intents:
                return f, "actionable", None
        if intent not in intents:
            target = intent.upper()                       # a value named as the intent ("RECOGNIZED"): what it means here
            if self.g.member_set(target):
                f = self.kw._as(target)
                if f.act in intents:
                    return f, "actionable", None
            return unclear, "unclear", f"intent {intent!r} is not one this state takes"
        if not intents[intent].get("slots"):
            return F.make(intent), "actionable", None
        given = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
        kw = {}
        for sl, spec in intents[intent]["slots"].items():
            v = given.get(sl)
            if v in (None, "", []):
                continue
            if "code" in spec:
                code = re.sub(r"\D", "", str(v))
                from ..s07_checks.speech_reading import words_to_digits
                said = re.sub(r"\D", "", words_to_digits(text or ""))
                pat = (((self.dom.get("lexicon.acts") or {}).get(intent) or {}).get("code") or {}).get("pattern")
                if not code or code not in said:
                    return unclear, "unclear", f"the code {v!r} is not in the customer's words"
                if pat and not re.search(pat, code):
                    return unclear, "unclear", f"the code {v!r} is not a valid {sl}"
                kw[sl] = code
                continue
            m = self._member(spec["values"], str(v))
            if m is None:
                return unclear, "unclear", f"{sl}={v!r} is not one of {sorted(spec['values'])}"
            kw[sl] = m
        if any(spec.get("required") and sl not in kw for sl, spec in intents[intent]["slots"].items()):
            return unclear, "unclear", f"{intent} without its required slots"
        return F.make(intent, **kw), "actionable", None

    def _member(self, values: dict[str, str], v: str) -> str | None:
        for m in values:
            if v.strip().lower() == m.lower() or v.strip().lower() == m.lower().replace("_", " "):
                return m
        vs = self.g.member_set(next(iter(values)))
        found = self.kw.find(vs, " " + v.lower().replace("'", "") + " ") if vs else None
        return found if found in values else None


class Wording:
    """Words a bot turn's acts with the LLM, checked line by line against the template (see PART 2)."""

    def __init__(self, llm, stats: StructuredStats, voice: str = ""):
        from ..s01_packs.loading import current
        from ..s07_checks.speech_policy import markers
        self.llm, self.stats, self.voice = llm, stats, (voice or "").strip()
        self.dom = current()
        g = self.dom.graph
        forms = set()
        for vs, members in g.values.items():
            for m in members:
                forms.update(_low(x) for x in [m.replace("_", " ")] + list(g.aliases(m)) + [self._word("", m)] if x)
        self.value_forms = sorted((f for f in forms if len(f) > 2), key=len, reverse=True)
        mk = markers()
        self.risky = [_low(x) for x in mk["policy"] + mk["commitment"]]

    def _word(self, k: str, v: Any) -> str:
        return _word(self.dom, k, v)

    def facts(self, a: dict[str, Any]) -> dict[str, Any]:
        return {k: self._word(k, v) for k, v in a.items() if k not in ("act", "text", "state") and v not in (None, "")}

    def check(self, line: str, template: str, a: dict[str, Any]) -> str | None:
        """Why a worded line cannot be said in place of its template, or None."""
        from ..s07_checks.speech_reading import words_to_digits
        L, T = _low(words_to_digits(line)), _low(words_to_digits(template))
        if not L:
            return "empty"
        if _num_set(line) != _num_set(template):
            return f"its figures {_num_set(line)} are not the template's {_num_set(template)}"
        for k, w in self.facts(a).items():
            w = _low(w)
            if w and w in T and w not in L:
                return f"it leaves out {k} ({w})"
        has = lambda f, x: re.search(r"\b" + re.escape(f) + r"\b", x) is not None     # noqa: E731
        named = [f for f in self.value_forms if has(f, T)]
        for f in self.value_forms:
            if has(f, L) and not has(f, T) and not any(f in n or n in f for n in named):
                return f"it adds {f!r}"
        for f in named:                                  # an option the template names: said in one of its wordings
            if not has(f, L) and not any(has(o, L) for o in self._forms_of(f)):
                return f"it leaves out {f!r}"
        for r in self.risky:
            if re.search(r"\b" + re.escape(r) + r"\b", L) and not re.search(r"\b" + re.escape(r) + r"\b", T):
                return f"it adds {r!r} (a promise or a policy word)"
        return None

    def _forms_of(self, form: str) -> list[str]:
        """Every wording of the value a form belongs to (its name, aliases and the bot's word for it)."""
        g = self.dom.graph
        for members in g.values.values():
            for m in members:
                fs = {_low(x) for x in [m.replace("_", " ")] + list(g.aliases(m)) + [self._word("", m)] if x}
                if form in fs:
                    return sorted(fs)
        return [form]

    def lead_ok(self, lead: str) -> str | None:
        L = _low(lead)
        if not L:
            return "empty"
        if _num_set(lead) or "?" in L or len(L.split()) > 25:
            return "a figure, a question or too long"
        for f in self.value_forms + self.risky:
            if re.search(r"\b" + re.escape(f) + r"\b", L):
                return f"it says {f!r}"
        return None

    def word(self, acts: list[dict], lines: list[str], heard: dict[str, Any]) -> list[str]:
        """The lines to say: each worded line that passes its check, else its template."""
        from ..llm import LLMUnavailable, parse_json_object
        user = json.dumps({"customer_said": heard.get("text") or "", "customer_turn": heard.get("kind") or "",
                           "voice": self.voice,
                           "lines": [{"act": a.get("act"), "facts": self.facts(a), "template": t} for a, t in zip(acts, lines)]},
                          ensure_ascii=False)
        try:
            obj = parse_json_object(self.llm.invoke(WORD_SYSTEM, user, salt=f"word|{STRUCTURED_VERSION}"))
        except LLMUnavailable as e:
            self.stats.add("wording_errors")
            self.stats.add("templated", len(lines))
            self.stats.note("wording", f"the wording call failed: {str(e)[:160]}")
            return list(lines)
        got = (obj or {}).get("lines") if isinstance(obj, dict) else None
        if not isinstance(got, list) or len(got) != len(lines):
            self.stats.add("wording_rejected", len(lines))
            self.stats.add("templated", len(lines))
            self.stats.note("wording", "not one line for each act", obj)
            return list(lines)
        out = []
        for a, t, w in zip(acts, lines, got):
            why = self.check(str(w), t, a)
            if why:
                self.stats.add("wording_rejected")
                self.stats.add("templated")
                self.stats.note("wording", why, {"act": a.get("act"), "template": t, "worded": w})
                out.append(t)
            else:
                self.stats.add("worded")
                out.append(" ".join(str(w).split()))
        lead = str((obj or {}).get("lead") or "").strip()
        if lead and heard.get("kind") in ("social", "claim", "actionable", "question") and out:
            why = self.lead_ok(lead)
            if why is None:
                self.stats.add("leads")
                out[0] = f"{lead} {out[0]}"
            else:
                self.stats.note("lead", why, lead)
        return out


class StructuredBot(GraphBot):
    """The LLM bot under test: start() / respond(text) -> BotTurn. The graph's own interpreter (``GraphBot``, with no
    planted defects) decides; ``LLMReader`` reads each customer turn and ``Wording`` words each bot turn (see PART 2)."""

    def __init__(self, tools, store, reader_llm, wording_llm=None, stats: StructuredStats | None = None, voice: str = ""):
        super().__init__(tools, store)
        self.stats = stats or StructuredStats()
        self.reader = LLMReader(self.g, reader_llm, self.stats)
        self.wording = Wording(wording_llm, self.stats, voice) if wording_llm is not None else None
        self._heard: dict[str, Any] = {}
        self._said = ""

    def respond(self, text: str) -> BotTurn:
        self._heard = {"text": text}
        self.reader.bot_said = self._said
        return super().respond(text)

    def react(self, state: str, f):
        self._heard["kind"] = self.reader.last.get("kind")
        return super().react(state, f)

    def _answer_kb(self, frame) -> dict[str, Any]:
        """The graph's answer, exactly the fact asked; to a claim, said as agreeing only when the figures match."""
        out = super()._answer_kb(frame)
        if self.reader.last.get("kind") == "claim":
            truth, said = _num_set(str(out.get("answer"))), _num_set(str(self.reader.last.get("claimed") or ""))
            t = out["text"]
            if out.get("answer") is None or frame.get("intent") not in ("value", "limit"):
                out["text"] = "I can't confirm that. " + t
            elif said and said == truth:
                out["text"] = "Yes, that's right. " + t
            else:
                out["text"] = "Actually, " + t[:1].lower() + t[1:]
        return out

    def _turn(self, ended: bool) -> BotTurn:
        decided, before = turn_order(self._tr.events, self._mark)
        acts = [dict(a) for a in decided]
        lines = [_say(a) for a in acts]
        if self.wording is not None and acts:
            lines = self.wording.word(acts, lines, self._heard)
        else:
            self.stats.add("templated", len(lines))
        acts = [{**a, "text": t} for a, t in zip(acts, lines)]
        self.ended = self.ended or ended
        self.stats.add("turns")
        self._said = " ".join(lines)
        return BotTurn(self._said, acts, self.ended, before)


def make_structured_factory(store=None, reader_llm=None, wording_llm=None, voice: str = "", settings=None):
    """A bot_factory for CampaignConfig / JourneyRunner: the structured LLM bot on the active journey's graph.

    ``reader_llm``: the CachedLLM that reads customer turns (default: the gateway's default deployment, or the stand-in
    model of GMSJOURNEY_SUT_FAKE_MODEL); ``wording_llm``: the one that words the bot's acts (None: the templates).
    ``store`` may be set later (``factory.store``): a test run builds the bot before its knowledge store, and the
    pipeline hands the store to a factory whose ``store`` is None. One ``StructuredStats`` covers every conversation:
    ``factory.stats.as_dict()``; ``factory.llm.report()`` / ``factory.wording_llm.report()`` the calls."""
    from ..llm import CachedLLM, LLMSettings
    from ..s01_packs.loading import current
    if current().graph is None:
        raise RuntimeError("the LLM bot is built on the journey's policy graph; this journey has no compact graph")
    if reader_llm is None:
        s = settings or LLMSettings(temperature=0.0, max_tokens=200, json_mode=True)
        fake = fake_model()
        reader_llm = CachedLLM(s, model=fake, tag="sut") if fake is not None else CachedLLM(s, tag="sut")
    stats = StructuredStats()

    def factory(tools):
        if factory.store is None:
            raise RuntimeError("the LLM bot needs the journey's knowledge store (factory.store)")
        return StructuredBot(tools, factory.store, factory.llm, factory.wording_llm, stats, voice)

    factory.store, factory.llm, factory.wording_llm, factory.stats = store, reader_llm, wording_llm, stats
    factory.voice, factory.critique = voice, False
    factory.system_prompt = (f"# {STRUCTURED_VERSION}\n\n## Reading each customer turn (system prompt)\n\n{READ_SYSTEM}\n\n"
                             "The user message is JSON: the bot's last words, the customer's words and the menu of the "
                             "current state (its intents with their slots and values, what a plain yes / no means, and "
                             "the policy facts a question can be about).\n\n"
                             f"## Wording the bot's acts (system prompt)\n\n{WORD_SYSTEM}\n")
    return factory


REF_KEYS = ("transaction_id", "txn", "id", "item_id")


def _ref(v: dict) -> Any:
    """A field the model wrote as an object that names an item (``"txn": {"transaction_id": "T1", "merchant": ...}``)
    where a reference was asked for: the reference (its id). Any other object (``"recorded": {"T1": ...}``) is kept
    as it is."""
    for k in REF_KEYS:
        x = v.get(k)
        if isinstance(x, (str, int, float)) and x != "":
            return x
    return v


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
