"""The LLM behind the second enrichment arm: the internal Microsoft (Azure OpenAI) model gateway,
behind a replay cache. There is one transport and no fallback.

``GatewayChat`` posts OpenAI-style chat completions to the gateway with ``requests``:

* the bearer token is a short-lived ``gcloud`` credential (``LLMSettings.token_cmd``), refreshed on a
  401/403; ``AEXP_ACCESS_TOKEN`` bypasses the CLI; the token is never stored, logged or reported;
* ``x-aip-usecase-id`` comes from ``use_case_id`` (``AEXP_USE_CASE_ID`` overrides);
* the deployment is ``model`` (``gpt-41`` for gpt-4.1, ``gpt-55`` for 5.5);
* the endpoint is ``GMSJOURNEY_LLM_URL`` when set (the full chat-completions URL, ``{model}`` is filled
  in), else ``{base_url}/models/{model}/chat/completions``;
* TLS goes through the corporate CA bundle (``GMSJOURNEY_CA_BUNDLE``, else ``REQUESTS_CA_BUNDLE``, else
  ``/etc/ssl/certs/ca-certificates.crt``);
* if the deployment rejects ``temperature``/``max_tokens`` or the system role (HTTP 400) the client adapts
  once and records it; 429 and 5xx back off and retry.

``CachedLLM`` puts a content-addressed cache in front of it. Every response is stored under the SHA-256 of
(prompt version, provider, model, temperature, messages, salt), so a campaign re-runs byte for byte without
network access (``mode="replay"``); a miss in replay mode raises ``LLMUnavailable``. The cache holds model
outputs only - prompts are hashed, not stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROMPT_VERSION = "llm-arm/1"
PROVIDER = "aexp"                          # the only transport (kept in the cache key)
DEFAULT_MODEL = "gpt-41"
DEFAULT_BASE_URL = "https://gaiapis.aexp.com/gcpapi/v1/microsoft"


class LLMUnavailable(RuntimeError):
    """No usable LLM: no credential, the gateway refused or is unreachable, or a replay-cache miss."""


@dataclass
class LLMSettings:
    model: str = DEFAULT_MODEL            # gateway deployment: gpt-41 (gpt-4.1), gpt-55 (5.5), ...
    temperature: float = 0.7              # paraphrase diversity; reproducibility comes from the cache
    max_tokens: int = 220
    base_url: str = DEFAULT_BASE_URL
    # the full chat-completions URL ("{model}" is filled in); env GMSJOURNEY_LLM_URL. Empty: from base_url
    url: str = ""
    use_case_id: str = "825"              # the x-aip-usecase-id header; env AEXP_USE_CASE_ID
    token_cmd: str = "gcloud auth print-access-token"   # how the bearer token is obtained
    access_token_env: str = "AEXP_ACCESS_TOKEN"         # override for the CLI (tests, CI)
    # CA bundle for the corporate TLS proxy; env GMSJOURNEY_CA_BUNDLE, then REQUESTS_CA_BUNDLE
    ca_bundle: str = "/etc/ssl/certs/ca-certificates.crt"
    # send temperature/max_tokens (standard OpenAI fields); AEXP_SEND_SAMPLING_PARAMS=0 sends model +
    # messages only. A deployment that rejects them is also detected and adapted to automatically.
    send_sampling_params: bool = True
    timeout: float = 60.0                 # read timeout per request (seconds)
    connect_timeout: float = 10.0         # an unreachable endpoint fails in seconds, not minutes
    max_retries: int = 2
    mode: str = "live"                    # live (call on a cache miss) | replay (cache only)
    cache_dir: str | None = None          # default: <gmsjourney cache>/llm/<model>
    # print every live gateway response as it arrives (role, seed, seconds, reply text); cache hits are
    # silent. Env GMSJOURNEY_LLM_LOG=1. log_max_chars truncates long replies (0 = print in full).
    log_responses: bool = False
    log_max_chars: int = 400
    # circuit breaker: after this many failed calls in a row (no success in between) the client stops
    # calling and every further miss fails at once with the reason - instead of hours of timeouts
    max_consecutive_failures: int = 8

    def __post_init__(self) -> None:
        self.model = self.model or DEFAULT_MODEL
        self.base_url = os.environ.get("GMSJOURNEY_LLM_BASE_URL", "") or self.base_url or DEFAULT_BASE_URL
        self.url = os.environ.get("GMSJOURNEY_LLM_URL", "") or self.url
        self.use_case_id = os.environ.get("AEXP_USE_CASE_ID", self.use_case_id)
        self.ca_bundle = (os.environ.get("GMSJOURNEY_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE")
                          or self.ca_bundle)
        if os.environ.get("AEXP_SEND_SAMPLING_PARAMS", "").strip() in ("0", "false", "no"):
            self.send_sampling_params = False
        if os.environ.get("GMSJOURNEY_LLM_LOG", "").strip() in ("1", "true", "yes"):
            self.log_responses = True

    # kept so cache keys and reports stay comparable with earlier runs
    provider = PROVIDER
    effective_provider = PROVIDER

    def public(self) -> dict[str, Any]:
        """Settings safe to put in a report (no token, no token material)."""
        d = asdict(self)
        d["provider"] = PROVIDER
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

    # what the deployment turned out to reject, learnt from HTTP 400s and kept for the rest of the run
    _DROP_ON_400 = (("temperature", "temperature"), ("max_tokens", "max_tokens"), ("system", "system"))

    def __init__(self, s: LLMSettings):
        self.s = s
        self._token = _token(s)
        self.send_sampling = s.send_sampling_params
        self.token_field = "max_tokens"         # reasoning deployments (gpt-5.x) want max_completion_tokens
        self.system_role = True                 # some gateways reject the system role: folded into user
        self.adaptations: list[str] = []

    def _url(self) -> str:
        if self.s.url:
            u = self.s.url.replace("{model}", self.s.model)
            return u if u.startswith("http") else "https://" + u
        return f"{self.s.base_url.rstrip('/')}/models/{self.s.model}/chat/completions"

    def _verify(self):
        cb = self.s.ca_bundle
        return cb if cb and Path(cb).exists() else True

    def _payload(self, messages: list[Any]) -> dict[str, Any]:
        msgs = [{"role": _role(m), "content": _text(m)} for m in messages]
        if not self.system_role and msgs and msgs[0]["role"] == "system":
            sys_text = msgs.pop(0)["content"]
            if msgs and msgs[0]["role"] == "user":
                msgs[0] = {"role": "user", "content": f"{sys_text}\n\n{msgs[0]['content']}"}
            else:
                msgs.insert(0, {"role": "user", "content": sys_text})
        payload: dict[str, Any] = {"model": self.s.model, "messages": msgs}
        if self.send_sampling:
            payload["temperature"] = self.s.temperature
            payload[self.token_field] = self.s.max_tokens
        return payload

    def _adapt(self, body: str) -> bool:
        """After an HTTP 400, drop or rename what the deployment rejected; False when nothing applies."""
        b = body.lower()
        if self.send_sampling and self.token_field == "max_tokens" and "max_completion_tokens" in b:
            self.token_field = "max_completion_tokens"
            self.adaptations.append("max_tokens -> max_completion_tokens")
            return True
        if self.send_sampling and ("temperature" in b or "max_tokens" in b or "max_completion_tokens" in b
                                   or "unsupported" in b or "unrecognized" in b or "extra" in b):
            self.send_sampling = False
            self.adaptations.append("sampling parameters dropped (model + messages only)")
            return True
        if self.system_role and "system" in b and "role" in b:
            self.system_role = False
            self.adaptations.append("system message folded into the user message")
            return True
        return False

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "x-aip-usecase-id": self.s.use_case_id,
                "Content-Type": "application/json"}

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        import requests                                    # the gateway path needs only requests
        last_error = "no attempt made"
        refreshed, adapted, attempt = False, 0, 0
        while attempt <= self.s.max_retries:
            try:
                r = requests.post(self._url(), headers=self._headers(), json=self._payload(messages),
                                  timeout=(min(self.s.connect_timeout, self.s.timeout), self.s.timeout),
                                  verify=self._verify())
            except Exception as e:                         # noqa: BLE001 - transport errors (incl. TLS)
                last_error = f"{type(e).__name__}: {str(e)[:160]}"
                attempt += 1
                time.sleep(min(8.0, 1.5 ** attempt))
                continue
            if r.status_code in (401, 403) and not refreshed:   # the credential expired: refresh once
                self._token = _token(self.s)
                refreshed = True
                continue
            if r.status_code == 400 and adapted < 3 and self._adapt(r.text):
                adapted += 1                               # retried with what the deployment accepts
                continue
            if r.status_code == 429 or r.status_code >= 500:   # throttled / transient: back off
                last_error = f"HTTP {r.status_code}: {r.text[:160]}"
                attempt += 1
                wait = r.headers.get("retry-after")
                time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else min(20.0, 2.0 ** attempt))
                continue
            if r.status_code != 200:
                raise LLMUnavailable(f"gateway HTTP {r.status_code} at {self._url()}: {r.text[:200]}")
            try:
                data = r.json()
            except ValueError as e:
                raise LLMUnavailable(f"gateway returned non-JSON: {r.text[:200]}") from e
            choice = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            return SimpleNamespace(content=((choice.get("message") or {}).get("content") or ""),
                                   usage_metadata={"input_tokens": usage.get("prompt_tokens", 0),
                                                   "output_tokens": usage.get("completion_tokens", 0)})
        raise LLMUnavailable(f"gateway unreachable after {self.s.max_retries + 1} attempts: {last_error}")


def make_chat_model(s: LLMSettings) -> "GatewayChat":
    """The gateway client (obtains the token now), or LLMUnavailable with the reason."""
    return GatewayChat(s)


def _messages(system: str, user: str, provider: str | None = None):
    """Plain role/content messages for one exchange (no LangChain anywhere)."""
    return [SimpleNamespace(type="system", content=system), SimpleNamespace(type="human", content=user)]


_PRINT_LOCK = __import__("threading").Lock()


def _log_line(msg: str) -> None:
    """One line, safe from several threads and above any tqdm progress bar."""
    with _PRINT_LOCK:
        try:
            from tqdm.auto import tqdm
            tqdm.write(msg)
        except ImportError:
            print(msg, flush=True)


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
    provider: str = PROVIDER


@dataclass
class CachedLLM:
    """``invoke(system, user, salt) -> str`` through a chat model, behind a replay cache.

    ``model`` may be any chat model with ``invoke(messages) -> reply`` (the tests pass a fake
    one); when it is None the gateway client is built lazily on the first cache miss."""
    settings: LLMSettings = field(default_factory=LLMSettings)
    model: Any = None
    stats: LLMStats = field(default_factory=LLMStats)
    tag: str = "default"                   # separates caches of different roles (rephrase/extract/adversary)

    def __post_init__(self) -> None:
        self.settings.model = self.settings.model or DEFAULT_MODEL
        self._model_error: str | None = None
        # optional context manager entered around the network call only; the parallel proof uses it
        # to let other scenarios run their (non-network) checks while this call waits on the gateway
        self.io_release: Any = None
        self._fails_in_row = 0
        self._tripped: str | None = None      # set by the circuit breaker

    # ------------------------------------------------------------------ cache
    @property
    def dir(self) -> Path:
        """The cache directory for this model and role."""
        from ...backend import CACHE
        base = Path(self.settings.cache_dir) if self.settings.cache_dir else CACHE / "llm"
        return base / (self.settings.model or "default").replace("/", "_") / self.tag

    def key(self, system: str, user: str, salt: str = "") -> str:
        """Content-addressed on everything that changes the answer."""
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
        if self._tripped:
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
        if self._tripped:
            return self._tripped
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
        if self._tripped:                            # the breaker is open: fail at once, no network
            raise LLMUnavailable(self._tripped)
        self.stats.cache_misses += 1
        if not self.available:
            raise LLMUnavailable(self.unavailable_reason() or "LLM unavailable")
        t0 = time.time()
        try:
            msgs = _messages(system, user, provider=self.settings.effective_provider)
            if self.io_release is not None:
                with self.io_release():
                    out = self.model.invoke(msgs)
            else:
                out = self.model.invoke(msgs)
        except Exception as e:                       # noqa: BLE001 - transport / gateway errors
            self.stats.errors += 1
            self.stats.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            self._fails_in_row += 1
            if self.settings.log_responses or self.stats.errors <= 3:      # failures always show at first
                _log_line(f"[LLM {self.tag} FAILED · {salt} · {time.time() - t0:.1f}s] {self.stats.last_error}")
            lim = self.settings.max_consecutive_failures
            if lim and self._fails_in_row >= lim and not self._tripped:
                self._tripped = (f"stopped calling the gateway after {self._fails_in_row} failed calls in a row; "
                                 f"last error: {self.stats.last_error}")
                _log_line(f"[LLM {self.tag}] {self._tripped}")
            raise LLMUnavailable(self.stats.last_error) from e
        self.stats.seconds += time.time() - t0
        self._fails_in_row = 0
        text = out.content if hasattr(out, "content") else str(out)
        if isinstance(text, list):                   # content blocks
            text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
        if self.settings.log_responses:
            n = self.settings.log_max_chars
            shown = text if not n or len(text) <= n else text[:n] + f" ...[+{len(text) - n} chars]"
            _log_line(f"[LLM {self.tag} #{self.stats.cache_misses} · {salt} · {time.time() - t0:.1f}s] "
                      + shown.replace("\n", " "))
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
        m = self.model
        gw = {"endpoint": m._url(), "adaptations": list(m.adaptations)} if isinstance(m, GatewayChat) else {}
        return {"tag": self.tag, **asdict(self.stats), **gw, "cache_dir": str(self.dir),
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
