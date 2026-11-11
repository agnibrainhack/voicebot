"""One tiny call through the Microsoft model gateway, to answer "will it work *here*?".

    python tools/check_llm.py
    python tools/check_llm.py --model gpt-41 --usecase 825
    python tools/check_llm.py --url "https://gaiapis.aexp.com/gcpapi/<path>/completions"

A single call exercises everything a campaign needs: the gcloud credential, the endpoint, the
headers, the CA bundle, the payload shape and the reply parsing. It prints the endpoint, the
latency, the reply and any adaptation the client made (e.g. sampling parameters dropped) - or
the exact error. It writes nothing to disk; the token is never printed.

    python tools/check_llm.py --bot charge_review_graph

sends what the LLM bot under test sends for one customer turn instead: its real reading call (the reading prompt
and the menu of the journey's first question) and its real wording call (the acts of that question), with JSON mode
and their max_tokens, and prints what the deployment's rate-limit headers say: its tokens-per-minute quota, what a
customer turn costs against it, and so how many customer turns fit in a minute.
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
    s = LLMSettings(model=a.model, timeout=max(a.timeout, 60.0) if a.bot else a.timeout, backend="gateway",
                    temperature=0.0, max_tokens=200 if a.bot else 220, json_mode=bool(a.bot))
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
    calls = [("call", "You are a terse assistant.", PROMPT, s.max_tokens)]
    if a.bot:
        calls = bot_calls(a.bot)
    from gmsjourney.llm import estimate_tokens
    total = 0.0
    for what, system, user, mt in calls:
        s.max_tokens = mt
        cost = estimate_tokens([{"content": system}, {"content": user}], mt)
        total += cost
        if a.bot:
            print(f"\n{what}: prompt {len(system) + len(user)} characters, about {int(cost)} tokens counted against the "
                  f"quota (prompt + max_tokens {mt})")
        t0 = time.time()
        try:
            out = model.invoke(_messages(system, user))
        except Exception as e:                                 # noqa: BLE001 - report anything
            print(f"\nFAILED - the gateway call raised:\n  {type(e).__name__}: {str(e)[:400]}")
            return 1
        print(f"OK in {time.time() - t0:.2f}s")
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
        print(f"\nquota     : {int(tpm)} tokens a minute -> about {tpm / total:.1f} "
              + ("customer turns (one reading and one wording call each) a minute" if a.bot else "calls of this size a minute"))
        if a.bot:
            print("            set GMSJOURNEY_LLM_TPM to this quota to pace the calls from the first one")
    return 0


def bot_calls(pack: str) -> list[tuple[str, str, str, int]]:
    """The LLM bot's two calls for one customer turn at the journey's first question: (what, system, user, max_tokens)."""
    from gmsjourney.s01_packs import loading as domain
    from gmsjourney.s06_bots.llm_bot import READ_SYSTEM, WORD_SYSTEM, LLMReader, StructuredStats, Wording
    from gmsjourney.s06_bots.reference import _say
    domain.set_current(pack)
    g = domain.current().graph
    if g is None:
        raise SystemExit(f"{pack} has no policy graph: the LLM bot is built on one")
    first = next(st for st in g.states if g.kind(st) == "ask")
    acts = [{"act": first}]
    lines = [_say(x) for x in acts]
    read = json.dumps({"state": first, "bot_said": " ".join(lines), "customer": "Sure, one moment, let me check.",
                       "menu": LLMReader(g, None, StructuredStats()).menu(first)}, ensure_ascii=False)
    w = Wording(None, StructuredStats())
    word = json.dumps({"customer_said": "", "customer_turn": "", "voice": "",
                       "lines": [{"act": x["act"], "facts": w.facts(x), "template": t} for x, t in zip(acts, lines)]})
    return [("reading call", READ_SYSTEM, read, 200), ("wording call", WORD_SYSTEM, word, 400)]


if __name__ == "__main__":
    raise SystemExit(main())
