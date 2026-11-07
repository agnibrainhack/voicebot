"""A local model behind the LLM bot under test: Llama 3.1 8B Instruct (or any chat model vLLM serves) loaded with vLLM
from a folder on this machine. For campaign runs only: the console's Playground keeps to the gateway, so a chat
never loads an 8B model into the console's process.

    GMSJOURNEY_SUT_LOCAL_MODEL=/models/Meta-Llama-3.1-8B-Instruct     the model folder (set it in the notebook)

Optional knobs (environment): ``GMSJOURNEY_VLLM_GPU_UTIL`` (fraction of GPU memory vLLM may take, default 0.80;
lower it when the knowledge store shares the GPU), ``GMSJOURNEY_VLLM_MAX_LEN`` (context length, default 16384: the
bot's prompt is about 6k tokens and the conversation grows from there), ``GMSJOURNEY_VLLM_TP`` (tensor-parallel
GPUs, default 1), ``GMSJOURNEY_VLLM_DTYPE`` (default auto), ``GMSJOURNEY_VLLM_URL`` (an OpenAI-compatible vLLM server,
e.g. http://127.0.0.1:8000/v1, used instead of loading the model in-process; the folder is then the served model's
name or path).

``LocalChat.invoke(messages) -> reply`` has the shape ``CachedLLM`` expects, so the bot, its replay cache (keyed by
``local-<folder name>``) and its statistics work as with the gateway. Generation is greedy at temperature 0, so a
re-run of the same conversation gives the same reply.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ENV = "GMSJOURNEY_SUT_LOCAL_MODEL"


def local_model_path() -> str:
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
        why = f"set {ENV} to the model folder in the notebook (before app.show()) to enable it"
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
    from ..enrich.llm.client import _role, _text
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
