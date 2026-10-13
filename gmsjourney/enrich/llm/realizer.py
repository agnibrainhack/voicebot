"""LLM realizer: meaning card + factor levels -> customer utterance, written by the LLM (Microsoft gateway).

Same interface as the template ``Realizer`` (``realize(frame, levels, seed, ctx)``), so the
journey runner drives either arm unchanged. Three layers, mirroring the template arm:

1. core + asides  written by the LLM from the meaning card, the base wording and the
                  factor instructions (prompts.py); it never sees the expected outcome;
2. surface noise  typo / speech-recognition noise applied *deterministically* afterwards
                  with the same operators and protected tokens as the template arm (an
                  LLM is a poor and unrepeatable source of realistic typos);
3. checks         G0 form and G1 facts on the raw LLM text (gates.py).

``generate`` is what the invariance validator calls, attempt by attempt, with feedback
on the gate that failed. Accepted utterances are *pinned* (``pin``): at run time the
runner gets exactly the text that was proved, keyed by the scenario seed. A turn the
proof never saw (the bot left the oracle's path) is generated live and must pass the
same gates and the reader check in its state; otherwise it falls back to the template
arm and is counted, so a run never silently uses an unchecked utterance.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any

from ...ontology import Ontology, SemanticFrame as F
from ..noise import asr_noise, typo
from ..realizers import BASE_LEVELS, PROTECT, Realizer
from .client import CachedLLM, LLMUnavailable, parse_json_object
from .gates import LLMOutputGates
from .prompts import WRAPPER_FACTORS, _aside, build_rephrase_prompt, protected_tokens


def _clean(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
    t = re.sub(r"\s+", " ", t)
    if len(t) >= 2 and t[0] in "\"'“‘" and t[-1] in "\"'”’":
        t = t[1:-1].strip()
    return t


@dataclass
class LLMTurn:
    seed: str
    raw: str = ""
    final: str = ""
    base_text: str = ""
    template_text: str = ""
    json_reply: bool = True
    form: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    error: str | None = None
    fallback: bool = False

    @property
    def text_ok(self) -> bool:
        return not self.error and self.form.get("ok", False) and self.facts.get("ok", False)

    def reasons(self) -> list[str]:
        return ([self.error] if self.error else []) + self.form.get("reasons", []) + self.facts.get("reasons", [])


class LLMRealizer:
    def __init__(self, llm: CachedLLM, store, reader, template: Realizer | None = None,
                 fallback: str = "template", runtime_attempts: int = 2):
        self.llm, self.store, self.reader = llm, store, reader
        self.template = template or Realizer()
        self.gates = LLMOutputGates(store, reader)
        self.ont = Ontology.from_store(store)
        self.fallback = fallback
        self.runtime_attempts = runtime_attempts
        self.bank: dict[str, str] = {}
        self.stats = {"generated": 0, "pinned_hits": 0, "runtime_generated": 0, "runtime_accepted": 0,
                      "runtime_fallbacks": 0, "non_json_replies": 0}

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _key(seed: str, frame: F, levels: dict[str, str], state: str | None) -> str:
        lv = {k: v for k, v in sorted((levels or {}).items())}
        return hashlib.sha256(f"{seed}|{frame.canonical()}|{state}|{lv}".encode()).hexdigest()[:24]

    def pin(self, seed: str, frame: F, levels: dict[str, str], state: str | None, text: str) -> None:
        self.bank[self._key(seed, frame, levels, state)] = text

    # ------------------------------------------------------------------ generation
    def asides(self, frame: F, levels: dict[str, str], ctx: dict[str, Any] | None) -> dict[str, str]:
        lv = {**BASE_LEVELS, **(levels or {})}
        return {k: _aside(frame, k, lv[k], ctx) for k in WRAPPER_FACTORS if lv[k] != BASE_LEVELS[k]}

    def generate(self, frame: F, levels: dict[str, str], seed: str, ctx: dict[str, Any] | None,
                 state: str | None, feedback: str | None = None) -> LLMTurn:
        ctx = {k: v for k, v in (ctx or {}).items() if k in ("merchant", "amount")}
        base_text = self.template.realize(frame, {}, "base", ctx)
        tmpl_text = self.template.realize(frame, levels, seed, ctx)
        turn = LLMTurn(seed, base_text=base_text, template_text=tmpl_text)
        system, user = build_rephrase_prompt(frame, levels, ctx, state, base_text, feedback)
        try:
            reply = self.llm.invoke(system, user, salt=seed)
        except LLMUnavailable as e:
            turn.error = f"llm unavailable: {e}"
            return turn
        self.stats["generated"] += 1
        obj = parse_json_object(reply)
        if obj is None or not isinstance(obj.get("utterance"), str):
            turn.json_reply = False
            self.stats["non_json_replies"] += 1
            raw = _clean(reply)
        else:
            raw = _clean(obj["utterance"])
        turn.raw = raw
        turn.form = self.gates.form(raw).as_dict()
        turn.facts = self.gates.facts(raw, frame, ctx, levels, [base_text, tmpl_text]).as_dict()
        turn.final = self._surface(raw, frame, levels, seed, ctx)
        return turn

    def _surface(self, text: str, frame: F, levels: dict[str, str], seed: str, ctx: dict[str, Any] | None) -> str:
        n = {**BASE_LEVELS, **(levels or {})}["noise"]
        if n == "none" or not text:
            return text
        protect = set(PROTECT)
        for tok in protected_tokens(frame, levels, ctx):
            protect.update(re.findall(r"[a-z']+", tok.lower()))
        rng = random.Random(f"{seed}|noise")
        return typo(text, rng, protect=protect) if n == "typo" else asr_noise(text, rng, protect=protect)

    # ------------------------------------------------------------------ runner interface
    def realize(self, frame: F, levels: dict[str, str], seed: str, ctx: dict[str, Any] | None = None) -> str:
        ctx = dict(ctx or {})
        state = ctx.get("state")
        k = self._key(seed, frame, levels, state)
        if k in self.bank:
            self.stats["pinned_hits"] += 1
            return self.bank[k]
        # a turn the proof never saw: generate live, under the same gates and the reader check
        feedback = None
        for a in range(self.runtime_attempts):
            s = f"{seed}|rt{a}"
            t = self.generate(frame, levels, s, ctx, state, feedback)
            if t.error:
                break
            self.stats["runtime_generated"] += 1
            g3 = self.gates.one_clear_decision(t.final, frame, levels)
            same = self.ont.equivalent(frame, self.reader.read(t.final, state), state)
            if t.text_ok and g3.ok and same:
                self.stats["runtime_accepted"] += 1
                self.bank[k] = t.final
                return t.final
            feedback = "; ".join(t.reasons() + g3.reasons + ([] if same else ["the meaning changed"]))
        if self.fallback != "template":
            raise LLMUnavailable(f"no accepted LLM utterance for {seed}")
        self.stats["runtime_fallbacks"] += 1
        return self.template.realize(frame, levels, seed, {k2: v for k2, v in ctx.items() if k2 in ("merchant", "amount")})
