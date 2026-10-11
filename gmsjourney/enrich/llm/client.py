"""The LLM behind the second enrichment arm: DeepSeek through LangChain, with a replay cache.

``make_chat_model`` builds a LangChain chat model for DeepSeek:

* ``langchain_deepseek.ChatDeepSeek`` when that integration is installed, else
* ``langchain_openai.ChatOpenAI`` pointed at DeepSeek's OpenAI-compatible endpoint.

The API key is read from the environment variable named in ``LLMSettings.api_key_env``
(``DEEPSEEK_API_KEY`` by default). It is passed to the client object only; it is never
logged, written to the cache or put in any report.

``CachedLLM`` puts a content-addressed cache in front of the model. Every response is
stored under the SHA-256 of (prompt version, model, temperature, messages, salt), so a
campaign can be re-run byte-for-byte without network access (``mode="replay"``): a miss
in replay mode raises ``LLMUnavailable`` instead of calling out. The cache holds model
outputs only - prompts are hashed, not stored - so it can be shared with a reviewer.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROMPT_VERSION = "llm-arm/1"


class LLMUnavailable(RuntimeError):
    """No usable LLM: integration not installed, no API key, or a replay-cache miss."""


@dataclass
class LLMSettings:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    temperature: float = 0.7              # paraphrase diversity; reproducibility comes from the cache
    max_tokens: int = 220
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    timeout: float = 60.0
    max_retries: int = 2                  # transport retries inside LangChain
    mode: str = "live"                    # live (call on a cache miss) | replay (cache only)
    cache_dir: str | None = None          # default: <gmsjourney cache>/llm/<model>

    def public(self) -> dict[str, Any]:
        """Settings safe to put in a report (no key, no key material)."""
        d = asdict(self)
        d["api_key_present"] = bool(os.environ.get(self.api_key_env))
        return d


def make_chat_model(s: LLMSettings):
    """A LangChain chat model for DeepSeek, or LLMUnavailable with the reason."""
    key = os.environ.get(s.api_key_env)
    if not key:
        raise LLMUnavailable(f"environment variable {s.api_key_env} is not set")
    try:
        from langchain_deepseek import ChatDeepSeek                     # type: ignore
        return ChatDeepSeek(model=s.model, temperature=s.temperature, max_tokens=s.max_tokens,
                            api_key=key, timeout=s.timeout, max_retries=s.max_retries)
    except ImportError:
        pass
    try:
        from langchain_openai import ChatOpenAI                          # type: ignore
    except ImportError as e:
        raise LLMUnavailable("install the LLM extra: pip install 'gmsjourney[llm]' "
                             "(langchain-core + langchain-deepseek or langchain-openai)") from e
    return ChatOpenAI(model=s.model, temperature=s.temperature, max_tokens=s.max_tokens,
                      api_key=key, base_url=s.base_url, timeout=s.timeout, max_retries=s.max_retries)


def _messages(system: str, user: str):
    try:
        from langchain_core.messages import HumanMessage, SystemMessage    # type: ignore
    except ImportError as e:
        raise LLMUnavailable("langchain-core is not installed") from e
    return [SystemMessage(content=system), HumanMessage(content=user)]


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    last_error: str | None = None


@dataclass
class CachedLLM:
    """``invoke(system, user, salt) -> str`` through LangChain, behind a replay cache.

    ``model`` may be any LangChain chat model (tests pass a fake one); when it is None
    the DeepSeek model is built lazily on the first cache miss."""
    settings: LLMSettings = field(default_factory=LLMSettings)
    model: Any = None
    stats: LLMStats = field(default_factory=LLMStats)
    tag: str = "default"                   # separates caches of different roles (rephrase/extract/adversary)

    def __post_init__(self) -> None:
        from ...backend import CACHE
        base = Path(self.settings.cache_dir) if self.settings.cache_dir else CACHE / "llm"
        self.dir = base / self.settings.model.replace("/", "_") / self.tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self._model_error: str | None = None

    # ------------------------------------------------------------------ cache
    def key(self, system: str, user: str, salt: str = "") -> str:
        s = self.settings
        payload = json.dumps([PROMPT_VERSION, s.provider, s.model, round(float(s.temperature), 4),
                              s.max_tokens, system, user, salt], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def cached(self, system: str, user: str, salt: str = "") -> str | None:
        p = self._path(self.key(system, user, salt))
        if p.exists():
            try:
                return json.loads(p.read_text())["text"]
            except (OSError, ValueError, KeyError):
                return None
        return None

    @property
    def available(self) -> bool:
        """True when a miss can be served (live mode and a model can be built)."""
        if self.settings.mode != "live":
            return False
        if self.model is not None:
            return True
        if self._model_error is not None:
            return False
        try:
            self.model = make_chat_model(self.settings)
            return True
        except LLMUnavailable as e:
            self._model_error = str(e)
            return False

    def unavailable_reason(self) -> str | None:
        if self.settings.mode != "live":
            return "replay mode: cache only"
        return None if self.available else self._model_error

    # ------------------------------------------------------------------ call
    def invoke(self, system: str, user: str, salt: str = "") -> str:
        self.stats.calls += 1
        key = self.key(system, user, salt)
        p = self._path(key)
        if p.exists():
            try:
                self.stats.cache_hits += 1
                return json.loads(p.read_text())["text"]
            except (OSError, ValueError, KeyError):
                pass
        self.stats.cache_misses += 1
        if not self.available:
            raise LLMUnavailable(self.unavailable_reason() or "LLM unavailable")
        t0 = time.time()
        try:
            out = self.model.invoke(_messages(system, user))
        except Exception as e:                       # noqa: BLE001 - transport / provider errors
            self.stats.errors += 1
            self.stats.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            raise LLMUnavailable(self.stats.last_error) from e
        self.stats.seconds += time.time() - t0
        text = out.content if hasattr(out, "content") else str(out)
        if isinstance(text, list):                   # content blocks
            text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
        usage = getattr(out, "usage_metadata", None) or {}
        self.stats.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.stats.output_tokens += int(usage.get("output_tokens", 0) or 0)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"text": text, "model": self.settings.model, "prompt_version": PROMPT_VERSION,
                                 "salt": salt, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False))
        return text

    def invoke_json(self, system: str, user: str, salt: str = "") -> dict[str, Any] | None:
        """invoke() and parse the first JSON object in the reply (None when there is none)."""
        return parse_json_object(self.invoke(system, user, salt))

    def report(self) -> dict[str, Any]:
        return {"tag": self.tag, **asdict(self.stats), "cache_dir": str(self.dir),
                "unavailable_reason": None if self.settings.mode == "live" and self.model is not None
                else self._model_error}


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first balanced {...} in a reply, parsed; tolerant of code fences and prose around it."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    start = t.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except ValueError:
                        break
        start = t.find("{", start + 1)
    return None
