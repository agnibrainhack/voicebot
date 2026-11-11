"""The LLM behind the test tools (the enrichment arm, the dual read, the adversary; the LLM judges, the policy judge,
the case narration): the internal Microsoft (Azure OpenAI) model gateway, or a local model on vLLM, behind a replay cache.

Which one is ``LLMSettings.backend``: ``"gateway"`` (default) or ``"local"`` (a model folder loaded in this process
with vLLM, or an OpenAI-compatible vLLM server). It is chosen per **role**, separately:

* ``enrichment``: the LLM arm that rewords the customer's turns (rephrase, dual read, adversary);
* ``testing``: the two LLM judges, the policy judge of the speech check and the case narration.

``use_backend(role, ...)`` sets a role's choice for this process and the processes it starts (environment
``GMSJOURNEY_LLM_<ROLE>_BACKEND`` / ``_MODEL`` / ``_LOCAL_MODEL`` / ``_VLLM_URL``; without a role, the shared
``GMSJOURNEY_LLM_BACKEND`` ... that both roles fall back to). The console's two settings and a campaign's
``llm_backend`` / ``judges_backend`` do this. The bot under test chooses its own model separately.

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
  once and records it;
* every gateway call in the process goes through one **pacer** (``_Pacer``): at most
  ``GMSJOURNEY_LLM_MAX_INFLIGHT`` calls at once (default 4). When the gateway throttles (HTTP 429) or strains (5xx,
  a read timeout), every client pauses for its ``Retry-After`` (else an exponential, jittered wait) and the number
  of calls at once is halved; each success raises it again, one step at a time, up to the maximum. So several
  conversations at once, the critique and the judges share the gateway's capacity instead of racing for it;
* a throttled or straining call is retried patiently, for up to ``retry_seconds`` (``GMSJOURNEY_LLM_RETRY_SECONDS``,
  default 300) before it fails with ``LLMThrottled``; an unreachable endpoint still fails after ``max_retries``.

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
from . import progress as _progress
from typing import Any


PROMPT_VERSION = "llm-arm/1"
PROVIDER = "aexp"                          # the only transport (kept in the cache key)
DEFAULT_MODEL = "gpt-41"
DEFAULT_BASE_URL = "https://gaiapis.aexp.com/gcpapi/v1/microsoft"


ROLES = ("enrichment", "testing")


def _env(role: str | None, key: str) -> str:
    """A role's setting from the environment, else the shared one."""
    if role:
        v = os.environ.get(f"GMSJOURNEY_LLM_{role.upper()}_{key}", "").strip()
        if v:
            return v
    return os.environ.get(f"GMSJOURNEY_LLM_{key}", "").strip()


def backend_env(role: str | None = None) -> dict[str, str]:
    """A role's LLM as the environment sets it: backend ("gateway" | "local"), gateway model, local model folder, vLLM
    URL (each falling back to the shared setting, then to the bot's local-model variables)."""
    b = _env(role, "BACKEND").lower()
    return {"role": role or "", "backend": "local" if b == "local" else "gateway", "model": _env(role, "MODEL"),
            "local_path": _env(role, "LOCAL_MODEL") or os.environ.get("GMSJOURNEY_SUT_LOCAL_MODEL", "").strip(),
            "vllm_url": _env(role, "VLLM_URL") or os.environ.get("GMSJOURNEY_VLLM_URL", "").strip()}


def use_backend(backend: str | None, local_path: str | None = None, vllm_url: str | None = None,
                role: str | None = None, model: str | None = None) -> dict[str, str]:
    """Point a role's LLM (``role``: enrichment | testing; None: both, through the shared setting) at ``backend`` from
    now on, in this process and the processes it starts: "gateway" (with an optional deployment ``model``) or
    "local" with a model folder and/or a vLLM server URL. None leaves the environment as it is."""
    pre = f"GMSJOURNEY_LLM_{role.upper()}_" if role else "GMSJOURNEY_LLM_"
    if backend:
        os.environ[pre + "BACKEND"] = "local" if backend == "local" else "gateway"
        for k, v in (("LOCAL_MODEL", local_path), ("VLLM_URL", vllm_url), ("MODEL", model)):
            if v is not None:
                if str(v).strip():
                    os.environ[pre + k] = str(v).strip()
                else:
                    os.environ.pop(pre + k, None)
    return backend_env(role)


class LLMUnavailable(RuntimeError):
    """No usable LLM: no credential, the gateway refused or is unreachable, or a replay-cache miss."""


class LLMThrottled(LLMUnavailable):
    """The gateway kept throttling or straining (429, 5xx, read timeouts) for longer than the retry window: it is
    there but busy, so this does not open the circuit breaker."""


class _Pacer:
    """One deployment's share of the gateway: a **token budget** and a cap on calls in flight.

    Azure OpenAI admits a request against the deployment's tokens-per-minute quota by its prompt tokens **plus its
    max_tokens**, so a bot step (a 5-6k-token prompt) costs ten times a rewording call. The pacer keeps a bucket of
    tokens that refills at the quota's rate and lets a call through only when the bucket holds its estimated cost:
    calls are spaced so the gateway admits them, instead of being sent at once, refused (429) and retried. The quota
    is ``GMSJOURNEY_LLM_TPM`` when set, else learnt from the deployment's own ``x-ratelimit-*`` headers (sent with
    every Azure OpenAI reply), else unknown (no budget: only the cap in flight).

    The cap in flight is halved on a 429/5xx and raised one step per success (AIMD), and a 429 makes every client
    wait for its ``Retry-After``."""

    def __init__(self, cap: int, tpm: float | None = None):
        import threading
        self.cap = max(1, cap)
        self.limit = float(self.cap)
        self.inflight = 0
        self.pause_until = 0.0
        self.cv = threading.Condition()
        self.throttled = 0                   # how often the gateway pushed back (reported)
        self.tpm = tpm                       # tokens per minute (None: not known)
        self.tpm_source = "GMSJOURNEY_LLM_TPM" if tpm else ""
        self.tokens = tpm or 0.0
        self.last = time.time()
        self.rpm: float | None = None
        self.headers: dict[str, str] = {}    # the last rate-limit headers seen

    def _refill(self) -> None:
        now = time.time()
        if self.tpm:
            self.tokens = min(self.tpm, self.tokens + (now - self.last) * self.tpm / 60.0)
        self.last = now

    def acquire(self, cost: float = 0.0) -> None:
        with self.cv:
            while True:
                self._refill()
                wait = self.pause_until - time.time()
                if wait > 0:
                    self.cv.wait(min(wait, 5.0))
                    continue
                if self.inflight >= max(1, int(self.limit)):
                    self.cv.wait(1.0)
                    continue
                if self.tpm and self.tokens < min(cost, self.tpm):   # wait for the budget to cover this call
                    self.cv.wait(min(5.0, (min(cost, self.tpm) - self.tokens) * 60.0 / self.tpm + 0.05))
                    continue
                self.tokens -= cost
                self.inflight += 1
                return

    def release(self, pushed_back: bool = False, pause: float = 0.0) -> None:
        with self.cv:
            self.inflight = max(0, self.inflight - 1)
            if pushed_back:
                self.throttled += 1
                self.limit = max(1.0, self.limit / 2)
                self.pause_until = max(self.pause_until, time.time() + pause)
            else:
                self.limit = min(float(self.cap), self.limit + 1.0 / max(1.0, self.limit))
            self.cv.notify_all()

    def observe(self, headers) -> None:
        """Learn the quota from a reply's x-ratelimit-* headers and line the budget up with what is left."""
        def num(k):
            try:
                return float(headers.get(k))
            except (TypeError, ValueError):
                return None
        lim, left = num("x-ratelimit-limit-tokens"), num("x-ratelimit-remaining-tokens")
        rpm = num("x-ratelimit-limit-requests")
        seen = {k: headers.get(k) for k in ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
                                            "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                                            "retry-after-ms", "retry-after") if headers.get(k) is not None}
        with self.cv:
            if seen:
                self.headers = seen
            if lim and lim > 0 and self.tpm_source != "GMSJOURNEY_LLM_TPM":
                if not self.tpm:
                    _log_line(f"[LLM] the deployment's quota: {int(lim)} tokens a minute"
                              + (f", {int(rpm)} requests" if rpm else "") + " (from its rate-limit headers)")
                    self.tokens = lim
                self.tpm, self.tpm_source = lim, "x-ratelimit-limit-tokens"
            if rpm:
                self.rpm = rpm
            if left is not None and self.tpm:
                self.tokens = min(self.tokens, left)

    def quota(self) -> dict[str, Any]:
        return {"tokens_per_minute": self.tpm, "source": self.tpm_source or None, "requests_per_minute": self.rpm,
                "calls_at_once": int(self.limit), "throttled": self.throttled, "headers": dict(self.headers)}


def estimate_tokens(messages: list[dict[str, Any]], max_tokens: int) -> float:
    """What the gateway counts against the quota when it admits a call: the prompt (about 3.5 characters a token
    for English and JSON) plus the reply's max_tokens."""
    return sum(len(str(m.get("content") or "")) for m in messages) / 3.5 + 8 * len(messages) + max_tokens


_PACERS: dict[str, _Pacer] = {}


def _pacer(url: str) -> _Pacer:
    """The pacer of one deployment (its URL names the model: the quota is per deployment), shared by every client
    calling it: the bot under test, its critique, the Playground, the rewording and the judges on the same model."""
    with _SHARED_LOCK:
        if url not in _PACERS:
            tpm = os.environ.get("GMSJOURNEY_LLM_TPM", "").strip()
            _PACERS[url] = _Pacer(int(os.environ.get("GMSJOURNEY_LLM_MAX_INFLIGHT", "4") or 4),
                                  float(tpm) if tpm else None)
        return _PACERS[url]


def gateway_quota() -> dict[str, Any]:
    """What every deployment's pacer knows: quota, calls at once, how often the gateway pushed back."""
    with _SHARED_LOCK:
        return {u: p.quota() for u, p in _PACERS.items()}


@dataclass
class LLMSettings:
    """Where and how the LLM is called: model, temperature, token limit, gateway URL, use-case header, how the
    bearer token is obtained, CA bundle, cache folder and timeouts (each also settable from the environment).
    """
    model: str = ""                       # gateway deployment: gpt-41 (gpt-4.1), gpt-55 (5.5), ...; empty: the role's, else gpt-41
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
    timeout: float = 180.0                # read timeout per request (seconds): a reasoning deployment on a long prompt
    connect_timeout: float = 10.0         # an unreachable endpoint fails in seconds, not minutes
    max_retries: int = 2                  # retries of an endpoint that cannot be reached
    # how long a throttled (429) or straining (5xx, read timeout) call keeps being retried before it fails;
    # env GMSJOURNEY_LLM_RETRY_SECONDS
    retry_seconds: float = 300.0
    # ask for a JSON object reply (response_format json_object): the bot and its critique answer in JSON, and a
    # reply that is not JSON costs a whole second call; dropped automatically where a deployment rejects it
    json_mode: bool = False
    mode: str = "live"                    # live (call on a cache miss) | replay (cache only)
    cache_dir: str | None = None          # default: <gmsjourney cache>/llm/<model>
    # print every live gateway response as it arrives (role, seed, seconds, reply text); cache hits are
    # silent. Env GMSJOURNEY_LLM_LOG=1. log_max_chars truncates long replies (0 = print in full).
    log_responses: bool = False
    log_max_chars: int = 400
    # circuit breaker: after this many failed calls in a row (no success in between) the client stops
    # calling and every further miss fails at once with the reason - instead of hours of timeouts
    max_consecutive_failures: int = 8
    # where the model runs: "gateway" (the Microsoft gateway) or "local" (vLLM: local_path in this process, or the
    # server at vllm_url). Empty: the environment's choice (backend_env), else the gateway.
    backend: str = ""
    local_path: str = ""
    vllm_url: str = ""
    role: str = ""                        # enrichment | testing: whose setting an empty backend/model takes

    def __post_init__(self) -> None:
        env = backend_env(self.role or None)
        self.backend = (self.backend or env["backend"]).strip().lower()
        if self.backend == "local":
            self.local_path = self.local_path or env["local_path"]
            self.vllm_url = self.vllm_url or env["vllm_url"]
            if not str(self.model or "").startswith("local-"):    # the cache keys on the local model's name
                self.model = "local-" + (Path(self.local_path).name if self.local_path else "vllm-server")
        else:
            self.backend = "gateway"
            self.model = self.model or env["model"]
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
        d["where"] = ("local model " + (self.vllm_url or self.local_path)) if self.backend == "local" else "Microsoft gateway"
        d["access_token_present"] = bool(os.environ.get(self.access_token_env))
        return d


_TOKENS: dict[str, tuple[float, str]] = {}          # token command -> (when fetched, token): one per process
_TOKEN_TTL = 40 * 60.0                                # gcloud tokens live an hour; refreshed well before
_SHARED_LOCK = __import__("threading").Lock()


def _token(s: LLMSettings, fresh: bool = False) -> str:
    """The bearer token for the gateway: from the environment when set, else the CLI. Fetched once per process and
    shared by every client (the bot, its critique, the judges, the rewording ...); ``fresh`` fetches a new one (after
    a 401/403)."""
    tok = os.environ.get(s.access_token_env)
    if tok:
        return tok.strip()
    with _SHARED_LOCK:
        hit = _TOKENS.get(s.token_cmd)
        if hit and not fresh and time.time() - hit[0] < _TOKEN_TTL:
            return hit[1]
        tok = _fetch_token(s)
        _TOKENS[s.token_cmd] = (time.time(), tok)
        return tok


_SESSIONS = __import__("threading").local()          # one HTTP session (kept-alive connections) per thread


def _session():
    """This thread's requests session: connections to the gateway are kept alive and reused, so a call does not pay
    a new TCP and TLS handshake through the proxy."""
    ses = getattr(_SESSIONS, "s", None)
    if ses is None:
        import requests
        ses = requests.Session()
        ses.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=16))
        _SESSIONS.s = ses
    return ses


# what a deployment turned out to reject (learnt from an HTTP 400), shared by every client of the same endpoint so
# each one does not spend a failed call learning it again
_ADAPTED: dict[str, dict[str, Any]] = {}


def _fetch_token(s: LLMSettings) -> str:
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
        self.json_mode = True                   # response_format json_object, when the settings ask for it
        self.last_headers: dict[str, str] = {}
        self.adaptations: list[str] = []
        self._learn()

    def _learn(self) -> None:
        """Take what another client of this endpoint already learnt the deployment rejects."""
        learnt = _ADAPTED.get(self._url())
        if learnt and learnt["adaptations"] != self.adaptations:
            self.send_sampling = learnt["send_sampling"] and self.send_sampling
            self.token_field, self.system_role = learnt["token_field"], learnt["system_role"]
            self.json_mode = learnt.get("json_mode", True) and self.json_mode
            self.adaptations = list(learnt["adaptations"])

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
        if self.s.json_mode and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _adapt(self, body: str) -> bool:
        """After an HTTP 400, drop or rename what the deployment rejected; False when nothing applies."""
        b = body.lower()
        if self.s.json_mode and self.json_mode and ("response_format" in b or "json_object" in b):
            self.json_mode = False
            self.adaptations.append("JSON mode (response_format) dropped")
            return True
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
        import random
        import requests
        last_error = "no attempt made"
        self._learn()
        pacer = _pacer(self._url())
        cost = estimate_tokens(self._payload(messages)["messages"], self.s.max_tokens)
        window = float(os.environ.get("GMSJOURNEY_LLM_RETRY_SECONDS", "") or self.s.retry_seconds)
        deadline = None                                    # set at the first send: queueing for the quota is not giving up
        refreshed, adapted, unreachable, busy = False, 0, 0, 0
        while True:
            pacer.acquire(cost)
            if deadline is None:
                deadline = time.time() + window
            pushed_back, pause = False, 0.0
            try:
                r = _session().post(self._url(), headers=self._headers(), json=self._payload(messages),
                                    timeout=(min(self.s.connect_timeout, self.s.timeout), self.s.timeout),
                                    verify=self._verify())
            except (requests.exceptions.ReadTimeout, requests.exceptions.ChunkedEncodingError) as e:
                r, busy = None, busy + 1                   # it is there but slow: straining, retried patiently
                last_error = f"{type(e).__name__}: {str(e)[:160]}"
                pushed_back, pause = True, min(30.0, 2.0 ** busy) * (0.5 + random.random())
            except Exception as e:                         # noqa: BLE001 - cannot reach it (DNS, TLS, proxy, refused)
                r, unreachable = None, unreachable + 1
                last_error = f"{type(e).__name__}: {str(e)[:160]}"
            else:
                hdrs = getattr(r, "headers", None) or {}
                pacer.observe(hdrs)
                self.last_headers = dict(getattr(pacer, "headers", {}) or {})
                if r.status_code == 429 or r.status_code >= 500:   # throttled / straining: everyone backs off
                    busy += 1
                    last_error = f"HTTP {r.status_code}: {r.text[:160]}"
                    ms, wait = hdrs.get("retry-after-ms"), hdrs.get("retry-after")
                    if ms and ms.replace(".", "", 1).isdigit():
                        pause = float(ms) / 1000.0
                    elif wait and wait.replace(".", "", 1).isdigit():
                        pause = float(wait)
                    else:
                        pause = min(30.0, 2.0 ** busy) * (0.5 + random.random())
                    pushed_back = True
            finally:
                pacer.release(pushed_back, pause)
            if r is None:
                if pushed_back:
                    if time.time() + pause > deadline:
                        raise LLMThrottled(f"the gateway stayed busy for {int(window)} s ({busy} tries): {last_error}")
                    if busy == 1 or busy % 5 == 0:
                        _log_line(f"[LLM] gateway slow ({last_error[:60]}); retrying, {int(pacer.limit)} calls at once")
                    continue
                if unreachable > self.s.max_retries:
                    raise LLMUnavailable(f"gateway unreachable after {unreachable} attempts: {last_error}")
                time.sleep(min(8.0, 1.5 ** unreachable))
                continue
            if r.status_code in (401, 403) and not refreshed:   # the credential expired: refresh once (for everyone)
                self._token = _token(self.s, fresh=self._token == _TOKENS.get(self.s.token_cmd, (0, ""))[1])
                refreshed = True
                continue
            if r.status_code == 400 and adapted < 3 and self._adapt(r.text):
                adapted += 1                               # retried with what the deployment accepts
                _ADAPTED[self._url()] = {"send_sampling": self.send_sampling, "token_field": self.token_field,
                                         "system_role": self.system_role, "json_mode": self.json_mode,
                                         "adaptations": list(self.adaptations)}
                continue
            if pushed_back:
                if time.time() + pause > deadline:
                    raise LLMThrottled(f"the gateway kept answering {r.status_code} for {int(window)} s "
                                       f"({busy} tries): {last_error}")
                if busy == 1 or busy % 5 == 0:
                    _log_line(f"[LLM] gateway throttling ({last_error[:60]}): every client waits {pause:.0f} s, "
                              f"{int(pacer.limit)} calls at once")
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


def make_chat_model(s: LLMSettings):
    """The client for ``s.backend``: the gateway client (obtains the token now) or the local vLLM model (loaded on the
    first call, shared by every client of the same folder), or LLMUnavailable with the reason."""
    if s.backend == "local":
        from .s06_bots.llm_bot import LocalChat, status
        st = status(s.local_path, s.vllm_url)
        if not st["usable"]:
            raise LLMUnavailable(f"the local model cannot be used: {st['detail']}")
        return LocalChat(s.local_path or None, temperature=s.temperature, max_tokens=s.max_tokens,
                         url=s.vllm_url or None)
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
    """Counts of one client's calls: calls, cache hits and misses, errors, tokens in and out, seconds, last error."""
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
        from .s03_knowledge.build import CACHE
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
                text = json.loads(p.read_text())["text"]
                self.stats.cache_hits += 1
                _progress.llm_call(self.tag, "cached")
                return text
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
            if isinstance(e, LLMThrottled):          # busy, not down: the pacer slows everyone; no breaker
                _progress.llm_call(self.tag, "error", time.time() - t0)
                _log_line(f"[LLM {self.tag} gave up after the retry window · {salt}] {self.stats.last_error}")
                raise
            with _SHARED_LOCK:
                self._fails_in_row += 1
            _progress.llm_call(self.tag, "error", time.time() - t0)
            if self.settings.log_responses or self.stats.errors <= 3:      # failures always show at first
                _log_line(f"[LLM {self.tag} FAILED · {salt} · {time.time() - t0:.1f}s] {self.stats.last_error}")
            lim = self.settings.max_consecutive_failures
            if lim and self._fails_in_row >= lim and not self._tripped:     # several down-type failures in a row
                self._tripped = (f"stopped calling the {'local model' if self.settings.backend == 'local' else 'gateway'} after {self._fails_in_row} failed calls in a row; "
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
        _progress.llm_call(self.tag, "live", time.time() - t0,
                           int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0))
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
