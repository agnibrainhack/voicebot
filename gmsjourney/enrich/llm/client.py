"""The LLM behind the second enrichment arm, with a replay cache. Two transports:

* ``aexp`` (default) — an internal, OpenAI-compatible model gateway. The bearer token is a
  short-lived ``gcloud`` credential (``LLMSettings.token_cmd``), refreshed on a 401, and the
  deployment is named by ``model`` (``gpt-41`` for gpt-4.1). Needs only ``requests`` — no
  LangChain integration. The token is never stored, logged or put in a report; set
  ``AEXP_ACCESS_TOKEN`` to bypass the CLI, and ``AEXP_USE_CASE_ID`` for the routing header.
* ``deepseek`` — the public DeepSeek API through LangChain (``ChatDeepSeek`` when that
  integration is installed, else ``ChatOpenAI`` on DeepSeek's OpenAI-compatible endpoint).
  The key comes from ``LLMSettings.api_key_env`` (``DEEPSEEK_API_KEY``) and, like the token,
  is passed to the client object only.

``LLMSettings.provider`` chooses the transport at run time, and
``LLMSettings.fallback_provider`` names the one to use when the first cannot be built
(no ``gcloud``, no key, integration missing) or fails during a run — the switch is recorded
in the report, and because the provider is part of the cache key, text from one transport is
never filed under another's cache entry.

``CachedLLM`` puts a content-addressed cache in front of the model. Every response is
stored under the SHA-256 of (prompt version, provider, model, temperature, messages, salt),
so a campaign can be re-run byte-for-byte without network access (``mode="replay"``): a miss
in replay mode raises ``LLMUnavailable`` instead of calling out. The cache holds model
outputs only - prompts are hashed, not stored - so it can be shared with a reviewer.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROMPT_VERSION = "llm-arm/1"

# One entry per transport: the model used when none is named, and the endpoint. The gateway
# entry is an OpenAI-compatible deployment behind a corporate API gateway, authenticated with
# a short-lived bearer token from a CLI (see LLMSettings.token_cmd).
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "aexp": {"model": "gpt-41", "base_url": "https://gaiapis.aexp.com/gcpapi/v1/microsoft"},
    "deepseek": {"model": "deepseek-chat", "base_url": "https://api.deepseek.com"},
}


class LLMUnavailable(RuntimeError):
    """No usable LLM: integration not installed, no credential, or a replay-cache miss."""


@dataclass
class LLMSettings:
    provider: str = "aexp"                # aexp (gateway, gcloud token) | deepseek
    fallback_provider: str | None = "deepseek"   # used when the chosen one cannot be built or fails
    model: str = ""                       # empty: the provider's own default (PROVIDER_DEFAULTS)
    temperature: float = 0.7              # paraphrase diversity; reproducibility comes from the cache
    max_tokens: int = 220
    api_key_env: str = "DEEPSEEK_API_KEY"  # deepseek only
    base_url: str = ""                    # empty: the provider's own default
    use_case_id: str = "825"              # aexp only: the x-aip-usecase-id header
    token_cmd: str = "gcloud auth print-access-token"   # aexp only: how the bearer token is obtained
    access_token_env: str = "AEXP_ACCESS_TOKEN"         # aexp only: override for the CLI (tests, CI)
    # aexp only: send temperature/max_tokens as well (standard OpenAI fields). Set
    # AEXP_SEND_SAMPLING_PARAMS=0 if the deployment rejects anything beyond model + messages.
    send_sampling_params: bool = True
    timeout: float = 60.0
    max_retries: int = 2                  # transport retries inside the client
    mode: str = "live"                    # live (call on a cache miss) | replay (cache only)
    cache_dir: str | None = None          # default: <gmsjourney cache>/llm/<model>
    resolved_provider: str | None = None  # what was actually built (set by make_chat_model)

    def __post_init__(self) -> None:
        d = PROVIDER_DEFAULTS.get(self.provider, {})
        self.model = self.model or d.get("model", "")
        self.base_url = (self.base_url or os.environ.get("GMSJOURNEY_LLM_BASE_URL", "")
                         or d.get("base_url", ""))
        self.use_case_id = os.environ.get("AEXP_USE_CASE_ID", self.use_case_id)
        if os.environ.get("AEXP_SEND_SAMPLING_PARAMS", "").strip() in ("0", "false", "no"):
            self.send_sampling_params = False

    @property
    def effective_provider(self) -> str:
        """The provider a response actually comes from: the resolved one when a fallback was
        taken, otherwise the requested one. It is part of the cache key, so a fallback can
        never write one provider's text under another provider's cache entry."""
        return self.resolved_provider or self.provider

    def public(self) -> dict[str, Any]:
        """Settings safe to put in a report (no key, no token, no token material)."""
        d = asdict(self)
        d["effective_provider"] = self.effective_provider
        d["api_key_present"] = bool(os.environ.get(self.api_key_env))
        d["access_token_present"] = bool(os.environ.get(self.access_token_env))
        return d


def _token(s: LLMSettings) -> str:
    """The bearer token for the gateway: from the environment when set, else the CLI."""
    tok = os.environ.get(s.access_token_env)
    if tok:
        return tok.strip()
    cmd = shlex.split(s.token_cmd)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise LLMUnavailable(f"{' '.join(cmd)} could not provide an access token: {e}") from e
    tok = (out.stdout or "").strip()
    if not tok:
        raise LLMUnavailable(f"{' '.join(cmd)} returned no token")
    return tok


_ROLE = {"system": "system", "human": "user", "ai": "assistant"}


def _role(m: Any) -> str:
    return _ROLE.get(getattr(m, "type", "") or "", "user")


def _text(m: Any) -> str:
    c = getattr(m, "content", m)
    if isinstance(c, list):                    # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)


class GatewayChat:
    """A chat model for an OpenAI-compatible gateway behind a token-authenticated proxy.

    Only ``invoke(messages) -> reply`` is implemented, which is all ``CachedLLM`` needs, so the
    gateway transport works without any LangChain integration — ``requests`` is the only
    dependency. The token is fetched once and refreshed on a 401, so a campaign that outlives
    the hour-long credential does not fail mid-run.
    """

    def __init__(self, s: LLMSettings):
        self.s = s
        self._token = _token(s)

    def _url(self) -> str:
        return f"{self.s.base_url.rstrip('/')}/models/{self.s.model}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "x-aip-usecase-id": self.s.use_case_id,
                "Content-Type": "application/json"}

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        import requests                                    # the gateway path needs only requests
        payload = {"model": self.s.model,
                   "messages": [{"role": _role(m), "content": _text(m)} for m in messages]}
        if self.s.send_sampling_params:                # AEXP_SEND_SAMPLING_PARAMS=0 drops these two,
            payload["temperature"] = self.s.temperature       # leaving exactly model + messages
            payload["max_tokens"] = self.s.max_tokens
        last_error = "no attempt made"
        for attempt in range(self.s.max_retries + 1):
            try:
                r = requests.post(self._url(), headers=self._headers(), json=payload, timeout=self.s.timeout)
            except Exception as e:                         # noqa: BLE001 - transport errors
                last_error = f"{type(e).__name__}: {str(e)[:160]}"
                continue
            if r.status_code == 401 and attempt == 0:      # the credential expired: refresh once
                self._token = _token(self.s)
                continue
            if r.status_code != 200:
                raise LLMUnavailable(f"gateway HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            return SimpleNamespace(content=((choice.get("message") or {}).get("content") or ""),
                                   usage_metadata={"input_tokens": usage.get("prompt_tokens", 0),
                                                   "output_tokens": usage.get("completion_tokens", 0)})
        raise LLMUnavailable(f"gateway unreachable after {self.s.max_retries + 1} attempts: {last_error}")


def _deepseek_model(s: LLMSettings):
    """The DeepSeek transport through LangChain (ChatDeepSeek, else the OpenAI-compatible client)."""
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


def _build(provider: str, s: LLMSettings):
    """Build one provider's client and point the settings at what was built, so the model name,
    the endpoint, the cache key and the report all describe the transport actually in use."""
    alt = replace(s, provider=provider, model="", base_url="", resolved_provider=None)
    model = GatewayChat(alt) if provider == "aexp" else (
        _deepseek_model(alt) if provider == "deepseek" else
        (_ for _ in ()).throw(LLMUnavailable(f"unknown provider {provider!r}")))
    s.model, s.base_url = alt.model, alt.base_url
    s.resolved_provider = provider
    return model


class FailoverChat:
    """The primary client, with the fallback provider built on first failure.

    A gateway outage in the middle of a campaign would otherwise lose the whole arm; instead the
    fallback is built once, the settings are re-pointed at it (so later cache keys change with
    it), and the switch is recorded for the report.
    """

    def __init__(self, primary, settings: LLMSettings, fallback: str | None):
        self._primary, self._s, self._fallback = primary, settings, fallback
        self._alt = None
        self.switched_to: str | None = None
        self.reason: str | None = None

    def invoke(self, messages: list[Any]):
        try:
            return self._primary.invoke(messages)
        except Exception as e:                             # noqa: BLE001 - any provider failure
            if not self._fallback or self._fallback == self._s.effective_provider:
                raise
            if self._alt is None:
                self._alt = _build(self._fallback, self._s)
                self.switched_to = self._fallback
                self.reason = f"{type(e).__name__}: {str(e)[:160]}"
            return self._alt.invoke(messages)


def make_chat_model(s: LLMSettings):
    """The chat model for the configured provider, else LLMUnavailable naming every reason tried.

    Construction tries the requested provider first (for the gateway that includes obtaining a
    token, which is where ``gcloud`` absence shows up) and then ``fallback_provider``. A provider
    that builds but then fails at call time is handled by ``FailoverChat``.
    """
    order = [s.provider] + ([s.fallback_provider]
                            if s.fallback_provider and s.fallback_provider != s.provider else [])
    problems: list[str] = []
    for provider in order:
        try:
            model = _build(provider, s)
        except LLMUnavailable as e:
            problems.append(f"{provider}: {e}")
            continue
        if provider != s.provider:                         # the fallback was built directly
            return model
        return FailoverChat(model, s, s.fallback_provider) if s.fallback_provider else model
    raise LLMUnavailable("no usable provider — " + "; ".join(problems))


def _messages(system: str, user: str, provider: str | None = None):
    """Message objects for one exchange.

    The gateway transport is deliberately LangChain-free end to end: for any provider other than
    ``deepseek`` this returns plain ``role``/``content`` messages, so ``gateway`` calls never import
    or construct a LangChain object. ``deepseek`` is itself a LangChain client, so that path uses
    LangChain messages when langchain-core is installed and plain ones when it is not.
    """
    plain = [SimpleNamespace(type="system", content=system), SimpleNamespace(type="human", content=user)]
    if provider is not None and provider != "deepseek":
        return plain
    try:
        from langchain_core.messages import HumanMessage, SystemMessage    # type: ignore
    except ImportError:
        return plain
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
    provider: str | None = None            # the transport responses actually came from
    switched_to: str | None = None         # set when the fallback provider took over mid-run
    failover_reason: str | None = None


@dataclass
class CachedLLM:
    """``invoke(system, user, salt) -> str`` through a chat model, behind a replay cache.

    ``model`` may be any chat model with ``invoke(messages) -> reply`` (the tests pass a fake
    one); when it is None the configured provider is built lazily on the first cache miss."""
    settings: LLMSettings = field(default_factory=LLMSettings)
    model: Any = None
    stats: LLMStats = field(default_factory=LLMStats)
    tag: str = "default"                   # separates caches of different roles (rephrase/extract/adversary)

    def __post_init__(self) -> None:
        if not self.settings.model:        # resolve the provider's default now, so keys and the
            self.settings.model = PROVIDER_DEFAULTS.get(              # cache directory are stable
                self.settings.provider, {}).get("model", "default")
        self._model_error: str | None = None

    # ------------------------------------------------------------------ cache
    @property
    def dir(self) -> Path:
        """The cache directory *for the transport in use*, computed on each access rather than
        fixed at construction: a run that falls back to another provider must read and write that
        provider's generations, not the requested one's."""
        from ...backend import CACHE
        base = Path(self.settings.cache_dir) if self.settings.cache_dir else CACHE / "llm"
        return base / (self.settings.model or "default").replace("/", "_") / self.tag

    def key(self, system: str, user: str, salt: str = "") -> str:
        """Content-addressed on everything that changes the answer — including the provider that
        actually served it, so a fallback never collides with the provider it replaced."""
        s = self.settings
        payload = json.dumps([PROMPT_VERSION, s.effective_provider, s.model, round(float(s.temperature), 4),
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
        # Resolving the provider (above) may have switched it — a fallback re-points the model and
        # the endpoint. Re-ask for the key and the path so the entry is filed under the transport
        # that actually answered, and so an existing generation under that transport is used rather
        # than thrown away.
        key2 = self.key(system, user, salt)
        if key2 != key:
            self.stats.cache_misses -= 1
            p = self._path(key2)
            if p.exists():
                try:
                    self.stats.cache_hits += 1
                    return json.loads(p.read_text())["text"]
                except (OSError, ValueError, KeyError):
                    pass
            self.stats.cache_misses += 1
        t0 = time.time()
        try:
            out = self.model.invoke(_messages(system, user, provider=self.settings.effective_provider))
        except Exception as e:                       # noqa: BLE001 - transport / provider errors
            self.stats.errors += 1
            self.stats.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            raise LLMUnavailable(self.stats.last_error) from e
        self.stats.seconds += time.time() - t0
        switched = getattr(self.model, "switched_to", None)     # the fallback took over on this call
        if switched and self.stats.switched_to is None:
            self.stats.switched_to = switched
            self.stats.failover_reason = getattr(self.model, "reason", None)
        self.stats.provider = self.settings.effective_provider
        text = out.content if hasattr(out, "content") else str(out)
        if isinstance(text, list):                   # content blocks
            text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
        usage = getattr(out, "usage_metadata", None) or {}
        self.stats.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.stats.output_tokens += int(usage.get("output_tokens", 0) or 0)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"text": text, "model": self.settings.model,
                                 "provider": self.settings.effective_provider,
                                 "prompt_version": PROMPT_VERSION, "salt": salt,
                                 "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False))
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
