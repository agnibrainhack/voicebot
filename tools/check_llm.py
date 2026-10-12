"""One tiny call through the configured LLM transport, to answer "will it work *here*?".

    python tools/check_llm.py                        # the default: gateway, then DeepSeek fallback
    python tools/check_llm.py --provider aexp        # the gateway only — no fallback to hide behind
    python tools/check_llm.py --provider deepseek    # DeepSeek only
    python tools/check_llm.py --provider aexp --model gpt-41 --usecase 825

Run it once in the environment you are shipping to. A single call exercises everything a campaign
needs: obtaining the credential, the endpoint, the headers, the payload shape and the reply
parsing. It prints which transport answered, the model, the latency and the reply — or the exact
reason each transport could not be used. It makes exactly one call and writes nothing to disk; the
credential is never printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsjourney.enrich.llm import LLMSettings                      # noqa: E402
from gmsjourney.enrich.llm.client import (LLMUnavailable, _messages,   # noqa: E402
                                          make_chat_model, PROVIDER_DEFAULTS)

PROMPT = 'Reply with exactly this JSON and nothing else: {"ok": true}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider", default="auto", choices=["auto", "aexp", "deepseek"],
                    help="auto (default): the gateway, falling back to DeepSeek")
    ap.add_argument("--model", default="", help="default: the provider's own model")
    ap.add_argument("--usecase", default=None, help="the gateway's x-aip-usecase-id header")
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()

    provider, fallback = ("aexp", "deepseek") if a.provider == "auto" else (a.provider, None)
    s = LLMSettings(provider=provider, fallback_provider=fallback, model=a.model, timeout=a.timeout)
    if a.usecase:
        s.use_case_id = a.usecase

    print(f"transport : {provider}" + (f" → {fallback} if it cannot be built or fails" if fallback else
                                       " (no fallback: this checks it honestly)"))
    print(f"model     : {s.model}  (endpoint {s.base_url})")
    print(f"credential: " + ("access token from $" + s.access_token_env + " or `" + s.token_cmd + "`"
                             if provider == "aexp" else "$" + s.api_key_env))

    try:
        model = make_chat_model(s)
    except LLMUnavailable as e:
        print(f"\nFAILED — no usable transport:\n  {e}")
        return 1
    print(f"resolved  : {s.effective_provider} · {s.model}"
          + (f"   (fell back from {provider})" if s.effective_provider != provider else ""))

    t0 = time.time()
    try:
        out = model.invoke(_messages("You are a terse assistant.", PROMPT,
                                     provider=s.effective_provider))
    except Exception as e:                                     # noqa: BLE001 - report anything
        print(f"\nFAILED — the {s.effective_provider} call raised:\n  {type(e).__name__}: {str(e)[:400]}")
        return 1
    dt = time.time() - t0
    text = out.content if hasattr(out, "content") else str(out)
    print(f"\nOK — answered by {s.effective_provider} in {dt:.2f}s")
    print(f"  reply : {text[:160]!r}")
    usage = getattr(out, "usage_metadata", None)
    if usage:
        print(f"  tokens: {usage}")
    print("\nThis one call exercised: credential, endpoint, headers, payload shape, reply parsing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
