"""gpt-oss (or any chat model vLLM serves) for turnval's text generation: through an OpenAI-compatible vLLM server
(``url``, e.g. http://gpu-host:8000/v1, with ``path`` = the served model name), or a model folder loaded in this
process with vLLM. ``status`` says whether it can be used. (Taken from the earlier case-generation project.)

``LocalChat.invoke(messages) -> reply`` has the shape ``llm.CachedLLM`` expects. A reasoning model (gpt-oss) is sent
``reasoning_effort`` (``TURNVAL_VLLM_REASONING_EFFORT``, default "low") and a thinking budget on top of the answer's
tokens (``TURNVAL_VLLM_REASONING_TOKENS``, default 1024). In-process knobs: ``TURNVAL_VLLM_GPU_UTIL`` (default 0.80),
``TURNVAL_VLLM_MAX_LEN`` (16384), ``TURNVAL_VLLM_TP`` (tensor-parallel GPUs, 1), ``TURNVAL_VLLM_DTYPE`` (auto).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ENV = "TURNVAL_LOCAL_MODEL"


def local_model_path() -> str:
    """The local model folder from TURNVAL_LOCAL_MODEL (empty when not set)."""
    return os.environ.get(ENV, "").strip()


def status(path: str | None = None, url: str | None = None) -> dict[str, Any]:
    """What the console shows about a local model, without loading it. ``path`` (the model folder) and ``url`` (an
    OpenAI-compatible vLLM server) default to TURNVAL_LOCAL_MODEL and TURNVAL_VLLM_URL."""
    import importlib.util
    path = (path or "").strip() or local_model_path()
    url = (url or "").strip() or os.environ.get("TURNVAL_VLLM_URL", "").strip()
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
    from .llm import _role, _text
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
        self.url = ((url or "").strip() or os.environ.get("TURNVAL_VLLM_URL", "").strip()).rstrip("/")
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
            raise RuntimeError("vLLM is not installed (pip install vllm), or set TURNVAL_VLLM_URL to a vLLM server") from e
        return LLM(model=str(p), tensor_parallel_size=int(os.environ.get("TURNVAL_VLLM_TP", "1")),
                   gpu_memory_utilization=float(os.environ.get("TURNVAL_VLLM_GPU_UTIL", "0.80")),
                   max_model_len=int(os.environ.get("TURNVAL_VLLM_MAX_LEN", "16384")),
                   dtype=os.environ.get("TURNVAL_VLLM_DTYPE", "auto"), seed=0)

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
        if is_reasoning_model(self.path) or os.environ.get("TURNVAL_VLLM_REASONING_EFFORT", "").strip():
            return self.max_tokens + int(os.environ.get("TURNVAL_VLLM_REASONING_TOKENS", "1024"))
        return self.max_tokens

    def _body(self, msgs: list[dict[str, str]]) -> dict[str, Any]:
        """The chat request. A reasoning model (gpt-oss) thinks before it answers and its thinking spends output
        tokens: it is asked for ``reasoning_effort`` (TURNVAL_VLLM_REASONING_EFFORT, default "low") and given
        a thinking budget on top of the answer's tokens (TURNVAL_VLLM_REASONING_TOKENS, default 1024), so a
        300-token judge verdict is not cut off by its own reasoning. TURNVAL_VLLM_EXTRA_BODY (JSON) adds any
        other field the server takes."""
        body: dict[str, Any] = {"model": self.path, "messages": msgs, "temperature": self.temperature,
                                "max_tokens": self.max_tokens, "seed": 0}
        effort = os.environ.get("TURNVAL_VLLM_REASONING_EFFORT", "").strip()
        if effort or is_reasoning_model(self.path):
            body["reasoning_effort"] = effort or "low"
            body["max_tokens"] = self._budget()
        extra = os.environ.get("TURNVAL_VLLM_EXTRA_BODY", "").strip()
        if extra:
            body.update(json.loads(extra))
        return body

    _DROPPED: dict[str, set] = {}                     # per server: optional fields it turned out to reject

    def _server(self, msgs: list[dict[str, str]]) -> tuple[str, dict]:
        import urllib.error
        import urllib.request
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


