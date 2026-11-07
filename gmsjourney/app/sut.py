"""The bot under test, as the console configures it: the reference bot of the journey, or an LLM bot on the model
gateway with the tester's prompt and attached files.

A console job carries ``sut``::

    {"kind": "reference"}                                    the pack's reference bot (the default)
    {"kind": "llm", "provider": "gateway" | "local", "model": "gpt-41", "mode": "live" | "replay", "temperature": 0.0,
     "max_steps": 6, "instructions": "...", "references": [{"name": "faq.pdf", "text": "..."}]}

``provider`` "local" runs Llama (or any chat model) with vLLM from the folder in GMSJOURNEY_SUT_LOCAL_MODEL
(sut/local_llm.py); campaign runs only, the Playground refuses it.

``instructions`` is the editable part of the prompt (who the bot is, how it behaves, the operating procedure); the
references are added after it as reference material; the tools, dialog acts and reply format the harness needs are
always appended (``sut.llm_bot.protocol_text``) and are not editable.
"""
from __future__ import annotations

import time
from typing import Any


def suggested(store=None) -> dict[str, Any]:
    """The suggested prompt for the active journey, and the fixed part the harness adds."""
    from ..domain import current
    from ..sut.llm_bot import SUT_PROMPT_VERSION, act_inventory, journey_title, protocol_text, suggested_instructions, transfer_reasons
    dom = current()
    src = dom.transcription_path
    return {"pack": dom.name, "title": journey_title(), "instructions": suggested_instructions(),
            "protocol": protocol_text(act_inventory(store), transfer_reasons(store)), "version": SUT_PROMPT_VERSION,
            "policy_source": (f"the policy transcription ({src.name})" if src is not None and src.exists()
                              else f"the policy graph ({dom.graph_path.name}) as text")}


def settings_for(sut: dict[str, Any]):
    from ..enrich.llm.client import LLMSettings
    return LLMSettings(model=str(sut.get("model") or ""), temperature=float(sut.get("temperature") or 0.0),
                       max_tokens=int(sut.get("max_tokens") or 900), mode=str(sut.get("mode") or "live"))


def make_factory(sut: dict[str, Any] | None, store=None):
    """(bot_factory or None for the reference bot, display name, the LLM factory or None)."""
    sut = sut or {}
    if sut.get("kind") != "llm":
        return None, None, None
    from ..sut.llm_bot import fake_model, make_llm_factory
    s = settings_for(sut)
    llm, where = None, "gateway"
    if sut.get("provider") == "local":              # Llama on vLLM, on this machine (campaign runs only)
        from ..enrich.llm.client import CachedLLM
        from ..sut.local_llm import LocalChat, status
        st = status()
        if not st["usable"]:
            raise RuntimeError(f"the local model cannot be used: {st['detail']}")
        chat = LocalChat(temperature=s.temperature, max_tokens=s.max_tokens)
        s.model = chat.name
        llm, where = CachedLLM(s, model=chat, tag="sut"), "vLLM" + (" server" if st["url"] else ", this machine")
    fac = make_llm_factory(store, s, llm=llm, instructions=sut.get("instructions") or None,
                           references=[r for r in (sut.get("references") or []) if r.get("text")],
                           max_steps=int(sut.get("max_steps") or 6))
    name = f"LLM bot ({fac.llm.settings.model}" + (f", {where}" if llm is not None else "") + \
        (", replay" if s.mode == "replay" else "") + (", stand-in model" if llm is None and fake_model() is not None else "") + ")"
    return fac, name, fac


def check_gateway(model: str = "", timeout: float = 30.0) -> dict[str, Any]:
    """One tiny call through the gateway (tools/check_llm.py in one function): ok, endpoint, seconds, the reply or
    the exact error. A configured stand-in model (GMSJOURNEY_SUT_FAKE_MODEL) is reported instead of calling out."""
    from ..sut.llm_bot import fake_model
    if fake_model() is not None:
        return {"ok": True, "fake": True, "detail": "a stand-in model is configured (GMSJOURNEY_SUT_FAKE_MODEL); "
                                                    "the gateway is not called"}
    from ..enrich.llm.client import LLMSettings, LLMUnavailable, _messages, make_chat_model
    s = LLMSettings(model=model, timeout=timeout)
    try:
        chat = make_chat_model(s)
    except LLMUnavailable as e:
        return {"ok": False, "model": s.model, "detail": f"no credential: {e}"}
    except Exception as e:                         # noqa: BLE001
        return {"ok": False, "model": s.model, "detail": f"{type(e).__name__}: {e}"}
    url = chat._url() if hasattr(chat, "_url") else None
    t0 = time.time()
    try:
        out = chat.invoke(_messages("You are a terse assistant.", 'Reply with exactly this JSON and nothing else: {"ok": true}'))
    except Exception as e:                         # noqa: BLE001 - report anything
        return {"ok": False, "model": s.model, "endpoint": url, "seconds": round(time.time() - t0, 2),
                "detail": f"{type(e).__name__}: {str(e)[:400]}"}
    return {"ok": True, "model": s.model, "endpoint": url, "seconds": round(time.time() - t0, 2),
            "detail": f"replied {str(getattr(out, 'content', ''))[:80]!r}"}
