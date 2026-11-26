"""A stand-in for a vLLM server (OpenAI-compatible /v1/models and /v1/chat/completions), for smoke runs of the
notebook and the tests without a GPU. It answers every prompt kind turnval sends with a well-formed JSON object.

    from turnval.mock_vllm import start
    url = start()          # http://127.0.0.1:<port>/v1, served in a background thread
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "openai/gpt-oss-20b"


def _answer(system: str, user: str) -> dict:
    if "caller says" in system:
        m = re.search(r"INTENT for your next turn: (.*?)\n", user)
        return {"utterance": (m.group(1) if m else "yes").replace("Say ", "").replace("Give ", "")[:200]}
    if "summarise" in system:
        n = user.count("Agent:")
        return {"summary": f"The agent has spoken {n} times; the call is in progress."}
    if "check a voice agent" in system:
        k = len(re.findall(r"^\d+\.", user, re.M))
        return {"checks": {str(i + 1): "ok" for i in range(k)}, "defect_present": "INTENDED DEFECT" in user}
    req = re.findall(r"- MUST SAY[^:]*: \[[^\]]*\] (.*)", user)
    text = " ".join(r.split("  (field")[0] for r in req[:4]) or "Okay, let me help you with that."
    if "DEFECT to make" in user:
        text += " (flawed)"
    return {"response": text}


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):                         # quiet
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs = body.get("messages", [])
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        content = json.dumps(_answer(system, user))
        self._send({"id": "mock", "object": "chat.completion", "model": body.get("model", MODEL),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(content) // 4}})


def start(port: int = 0) -> str:
    srv = ThreadingHTTPServer(("127.0.0.1", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/v1"
