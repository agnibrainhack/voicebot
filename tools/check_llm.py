"""One tiny call through the Microsoft model gateway, to answer "will it work *here*?".

    python tools/check_llm.py
    python tools/check_llm.py --model gpt-41 --usecase 825
    python tools/check_llm.py --url "https://gaiapis.aexp.com/gcpapi/<path>/completions"

A single call exercises everything a campaign needs: the gcloud credential, the endpoint, the
headers, the CA bundle, the payload shape and the reply parsing. It prints the endpoint, the
latency, the reply and any adaptation the client made (e.g. sampling parameters dropped) - or
the exact error. It writes nothing to disk; the token is never printed.

    python tools/check_llm.py --bot charge_review_graph

sends one step of the LLM bot under test instead: its real system prompt (5-6k tokens), JSON mode and its
max_tokens, and prints what the deployment's rate-limit headers say: its tokens-per-minute quota, what one bot
step costs against it, and so how many bot steps (and customer turns, with the critique) fit in a minute. A bot
that "gets no response" while the small rewording calls work is almost always this quota.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsjourney.llm import LLMSettings  # noqa: E402
from gmsjourney.llm import LLMUnavailable, _messages, make_chat_model  # noqa: E402

PROMPT = 'Reply with exactly this JSON and nothing else: {"ok": true}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="", help="gateway deployment (default gpt-41)")
    ap.add_argument("--usecase", default=None, help="the x-aip-usecase-id header (default 825)")
    ap.add_argument("--url", default=None, help="full chat-completions URL ({model} is filled in)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--bot", default=None, metavar="PACK",
                    help="send one step of the LLM bot under test of this journey (its real prompt) and report the quota")
    a = ap.parse_args()
    s = LLMSettings(model=a.model, timeout=max(a.timeout, 180.0) if a.bot else a.timeout, backend="gateway",
                    temperature=0.0, max_tokens=500 if a.bot else 220, json_mode=bool(a.bot))
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
    system, user = "You are a terse assistant.", PROMPT
    if a.bot:
        from gmsjourney.s01_packs import loading as domain
        from gmsjourney.s10_console import bots as PG
        domain.set_current(a.bot)
        fac, _name, _ = PG.make_factory({"kind": "llm", "model": s.model, "critique": False})
        system = fac.system_prompt
        user = json.dumps({"customer_turns_so_far": 1, "conversation": [
            {"customer": "Hi, I got a text about some charges on my card."}]})
    from gmsjourney.llm import estimate_tokens
    cost = estimate_tokens([{"content": system}, {"content": user}], s.max_tokens)
    if a.bot:
        print(f"bot step  : prompt {len(system) + len(user)} characters, about {int(cost)} tokens counted "
              f"against the quota (prompt + max_tokens {s.max_tokens})")
    t0 = time.time()
    try:
        out = model.invoke(_messages(system, user))
    except Exception as e:                                     # noqa: BLE001 - report anything
        print(f"\nFAILED - the gateway call raised:\n  {type(e).__name__}: {str(e)[:400]}")
        return 1
    print(f"\nOK in {time.time() - t0:.2f}s")
    print(f"  reply      : {out.content[:160]!r}")
    print(f"  tokens     : {getattr(out, 'usage_metadata', None)}")
    print(f"  adaptations: {model.adaptations or 'none'}")
    h = getattr(model, "last_headers", {}) or {}
    print(f"  rate limits: {h or 'the gateway sent no x-ratelimit headers'}")
    try:
        tpm = float(h.get("x-ratelimit-limit-tokens"))
    except (TypeError, ValueError):
        tpm = None
    if tpm:
        steps = tpm / cost
        print(f"\nquota     : {int(tpm)} tokens a minute -> about {steps:.1f} calls of this size a minute")
        if a.bot:
            print(f"            a customer turn is about 1.5 bot steps, and as many critique calls: about "
                  f"{steps / 3:.1f} turns a minute with the critique, {steps / 1.5:.1f} without")
            print("            set GMSJOURNEY_LLM_TPM to this quota to pace the calls from the first one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
