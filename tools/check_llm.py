"""One tiny call through the Microsoft model gateway, to answer "will it work *here*?".

    python tools/check_llm.py
    python tools/check_llm.py --model gpt-41 --usecase 825
    python tools/check_llm.py --url "https://gaiapis.aexp.com/gcpapi/<path>/completions"

A single call exercises everything a campaign needs: the gcloud credential, the endpoint, the
headers, the CA bundle, the payload shape and the reply parsing. It prints the endpoint, the
latency, the reply and any adaptation the client made (e.g. sampling parameters dropped) - or
the exact error. It writes nothing to disk; the token is never printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsjourney.enrich.llm import LLMSettings                               # noqa: E402
from gmsjourney.enrich.llm.client import LLMUnavailable, _messages, make_chat_model   # noqa: E402

PROMPT = 'Reply with exactly this JSON and nothing else: {"ok": true}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="", help="gateway deployment (default gpt-41)")
    ap.add_argument("--usecase", default=None, help="the x-aip-usecase-id header (default 825)")
    ap.add_argument("--url", default=None, help="full chat-completions URL ({model} is filled in)")
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()
    s = LLMSettings(model=a.model, timeout=a.timeout)
    if a.usecase:
        s.use_case_id = a.usecase
    if a.url:
        s.url = a.url
    print(f"model     : {s.model}")
    print(f"use case  : {s.use_case_id}")
    print(f"CA bundle : {s.ca_bundle}  (exists: {Path(s.ca_bundle).exists() if s.ca_bundle else False})")
    print("credential: $" + s.access_token_env + " or `" + s.token_cmd + "`")
    try:
        model = make_chat_model(s)
    except LLMUnavailable as e:
        print(f"\nFAILED - no token:\n  {e}")
        return 1
    print(f"endpoint  : {model._url()}")
    t0 = time.time()
    try:
        out = model.invoke(_messages("You are a terse assistant.", PROMPT))
    except Exception as e:                                     # noqa: BLE001 - report anything
        print(f"\nFAILED - the gateway call raised:\n  {type(e).__name__}: {str(e)[:400]}")
        return 1
    print(f"\nOK in {time.time() - t0:.2f}s")
    print(f"  reply      : {out.content[:160]!r}")
    print(f"  tokens     : {getattr(out, 'usage_metadata', None)}")
    print(f"  adaptations: {model.adaptations or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
