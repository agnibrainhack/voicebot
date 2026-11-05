"""Realizers: semantic frame + factor levels -> customer utterance (book Ch. 6.3.3).

Selecting a factor level is data; turning it into text is code. Each customer
turn is generated from its *frame*, never by editing another sentence, in three
layers:

1. core      the decision itself, phrased by entity_aliasing, paraphrase_depth,
             specificity, numeric_format, clarity, question_format, style=terse
             and self_correction;
2. wrappers  meaning-neutral sentences around the core: history, instruction
             conflict (reported speech or an instruction the bot must ignore),
             answer anchoring, confidence pressure, emotional frame, style, length;
3. surface   disfluency and typo / speech-recognition noise, with decision-
             critical tokens (negations, yes/no, digits) protected.

Base level for every factor is its first level, and the realisation at all-base
levels is the canonical text shown for a base case. Every realised turn is then
checked by the frame reader (reader.py); the realizer is never trusted.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..ontology import SemanticFrame
from .lexicon import Lexicon, cap, fill, low
from .noise import asr_noise, disfluency, number_to_words, spell_digits, typo

BASE_LEVELS = {
    "clarity": "clear", "entity_aliasing": "canonical", "specificity": "general",
    "numeric_format": "digits", "noise": "none", "disfluency": "none", "self_correction": "none",
    "paraphrase_depth": "none", "question_format": "interrogative", "emotional_frame": "neutral",
    "style": "formal", "history": "no_history", "instruction_conflict": "none",
    "answer_anchoring": "none", "confidence_pressure": "none", "length": "short",
}

_cap, _low = cap, low


@dataclass
class Realizer:
    """Deterministic, seed-stable realisation of frames under factor levels. Every phrase,
    template and wrapper comes from the domain lexicon (domain.yaml -> lexicon); this class
    only applies them."""

    lexicon: Lexicon | None = None

    def __post_init__(self):
        if self.lexicon is None:
            self.lexicon = Lexicon.current()

    def realize(self, frame: SemanticFrame, levels: dict[str, str], seed: str,
                context: dict[str, Any] | None = None) -> str:
        lv = {**BASE_LEVELS, **(levels or {})}
        rng = random.Random(seed)
        ctx = context or {}
        core = self._core(frame, lv, ctx)
        text = self._wrap(frame, core, lv, rng)
        return self._surface(frame, text, lv, rng)

    # ------------------------------------------------------------------ core
    def _num(self, n: int | float | str, lv: dict, money: bool = False) -> str:
        v = int(float(n))
        if lv["numeric_format"] == "spoken":
            return f"{number_to_words(v)} dollars" if money else number_to_words(v)
        return f"${v}" if money else str(v)

    def _depth(self, lv: dict) -> int:
        return self.lexicon["depths"].index(lv["paraphrase_depth"])

    def _obj(self, lv: dict, ctx: dict) -> str:
        objs = self.lexicon["objects"]
        if lv["specificity"] == "specific" and ctx.get("merchant"):
            amt = ctx.get("amount", "")
            if lv["numeric_format"] == "spoken" and amt:
                d, c = amt.split(".")
                amount = f"{number_to_words(int(d))} dollar {number_to_words(int(c))}"
            else:
                amount = f"${amt}"
            return fill(objs["specific"], amount=amount, merchant=ctx["merchant"])
        return objs["general"][lv["paraphrase_depth"]]

    def _core(self, f: SemanticFrame, lv: dict, ctx: dict) -> str:
        if f.act == "ASK_KB":
            q = self._kb_terse(f, lv) if lv["style"] == "terse" else self._kb_question(f, lv)
            return self._clarity(f, self._qform(f, q, lv), lv)
        spec = self.lexicon.acts.get(f.act)
        if spec is None:
            raise ValueError(f"no realizer for {f.act}")
        if "code" in spec:
            return self._code_act(f, spec, lv)
        if "slot" in spec:
            return self._value_act(f, spec, lv, ctx)
        if "slots" in spec:
            return self._clarity(f, self._qform(f, self._request(f, spec, lv), lv), lv)
        return self._fixed_act(f, spec, lv)

    def _fixed_act(self, f: SemanticFrame, spec: dict, lv: dict) -> str:
        say = spec["say"]
        if isinstance(say, str):                          # one sentence whatever the levels
            return say
        terse = lv["style"] == "terse"
        al = lv["entity_aliasing"]
        fields = {"phrase": spec["phrase"][al]} if "phrase" in spec else {}
        core = spec["terse"] if terse else fill(say[lv["paraphrase_depth"]], **fields)
        if spec.get("false_start") and lv["self_correction"] == "false_start" and not terse:
            core = fill(spec["false_start"], core_low=_low(core))
        return self._clarity(f, self._qform(f, core, lv), lv)

    def _code_act(self, f: SemanticFrame, spec: dict, lv: dict) -> str:
        code = f.get(spec["code"]["slot"])
        c = spell_digits(code) if lv["numeric_format"] == "spoken" else code
        terse = lv["style"] == "terse"
        if terse:
            core = fill(spec["terse"], code=c)
        else:
            ref = spec["ref"][lv["specificity"]]
            core = fill(spec["say"][lv["paraphrase_depth"]], ref=ref, code=c)
        if spec.get("false_start") and lv["self_correction"] == "false_start" and not terse:
            start = c[:2] if lv["numeric_format"] == "digits" else " ".join(c.split()[:2])
            core = fill(spec["false_start"], code_start=start, core_low=_low(core))
        return self._clarity(f, core, lv)

    def _value_act(self, f: SemanticFrame, spec: dict, lv: dict, ctx: dict) -> str:
        slot = spec["slot"]
        d = f.get(slot)
        al = lv["entity_aliasing"]
        if lv["style"] == "terse":
            core = spec["terse"][d]
        else:
            p = self.lexicon.phrase(d, al, obj=self._obj(lv, ctx))
            core = fill(spec["say"][d][lv["paraphrase_depth"]], phrase=p)
        corr = spec.get("correction")
        if corr and f.get("corrected_from"):
            first = self.lexicon.phrase(f.get("corrected_from"), "canonical", obj=corr["obj"])
            core = fill(corr["template"], first=first, core_low=_low(core.rstrip(".")))
        elif lv["self_correction"] == "false_start" and d in (spec.get("false_start") or {}):
            core = spec["false_start"][d] + _low(core)
        req = spec.get("with_request")
        if req:
            rspec = self.lexicon.acts[req]
            if f.get(rspec["slots"]["action"]):
                core = core.rstrip(".") + "." + self._request(f, rspec, lv, addition=True)
        return self._clarity(f, core, lv)

    def _request(self, f: SemanticFrame, spec: dict, lv: dict, addition: bool = False) -> str:
        al = lv["entity_aliasing"]
        act_v, opt_v = f.get(spec["slots"]["action"]), f.get(spec["slots"]["option"])
        action = self.lexicon.phrase(act_v, al)
        option = self.lexicon.phrase(opt_v, al) if opt_v else None
        tmpl = spec["as_addition"] if addition else spec
        if lv["style"] == "terse":
            return fill(tmpl["terse"], action=action, option_list=(", " + option) if option else "")
        tail = fill(spec["option_tail"], option=option) if option else ""
        return fill(tmpl["say"][lv["paraphrase_depth"]], action=action, tail=tail,
                    option_or_default=option or spec["option_default"])

    # -------------------------------------------------------------- policy questions
    def _kb_fields(self, f: SemanticFrame, lv: dict) -> dict[str, Any]:
        kbq, al = self.lexicon.kb, lv["entity_aliasing"]
        out: dict[str, Any] = {
            "entity": self.lexicon.phrase(f.get("entity"), al) if f.get("entity") else "",
            "rep": kbq.get("specific_suffix", "") if lv["specificity"] == "specific" else "",
        }
        if f.get("value") is not None:
            out["money"] = self._num(int(f.get("value")), lv, money=True)
        if f.get("options") is not None:
            opts = [self._num(o, lv, money=True) for o in f.get("options")]
            out["options"] = ", ".join(opts[:-1]) + f" or {opts[-1]}"
        out["days"] = self._num(f.get("days"), lv) if f.get("days") is not None else ""
        if f.get("name") is not None:
            out["fake"] = self._fake(f.get("name"), lv)
        if f.get("subject") is not None:
            out["subject"] = self.lexicon.phrase(f.get("subject"), al)
        return out

    def _kb_spec(self, f: SemanticFrame) -> dict[str, Any]:
        spec = self.lexicon.kb["intents"].get(f.get("intent"))
        if spec is None:
            raise ValueError(f.get("intent"))
        return spec

    @staticmethod
    def _variant(spec: dict, key: str, f: SemanticFrame, zero_key: str | None = None) -> Any:
        if zero_key and zero_key in spec and f.get("value") is not None and int(f.get("value")) == 0:
            return spec[zero_key]
        v = spec[key]
        if isinstance(v, dict):
            k = f.get(spec["by"])
            return v[k] if k in v else v["default"]
        return v

    def _kb_question(self, f: SemanticFrame, lv: dict) -> str:
        spec = self._kb_spec(f)
        variants = self._variant(spec, "say", f, "zero")
        return fill(variants[self._depth(lv)], **self._kb_fields(f, lv))

    def _fake(self, name: str, lv: dict) -> str:
        """A near-miss option name, with the noun varied by entity_aliasing like a real option."""
        kbq = self.lexicon.kb
        q = str(name).removesuffix(kbq.get("option_suffix", "")).replace("_", " ")
        return f"{q} " + kbq["option_nouns"][lv["entity_aliasing"]]

    def _kb_terse(self, f: SemanticFrame, lv: dict) -> str:
        spec = self._kb_spec(f)
        return fill(self._variant(spec, "terse", f, "terse_zero"), **self._kb_fields(f, lv))

    # -------------------------------------------------------------- modifiers
    _NO_CLARITY = ("END", "UNCLEAR", "DENY")

    def _clarity(self, f: SemanticFrame, core: str, lv: dict) -> str:
        c = lv["clarity"]
        if c == "clear" or f.act in self._NO_CLARITY or lv["style"] == "terse":
            return core
        spec = self.lexicon["clarity"][c]
        pre = spec["prefix"].get(f.act, spec["prefix"]["default"])
        post = ""
        sfx = (spec.get("suffix") or {}).get(f.act)
        if sfx:
            slot = (self.lexicon.acts.get(f.act) or {}).get("slot")
            e = sfx.get(f.get(slot)) if slot else None
            if e and not (e.get("unless") and f.get(e["unless"])):
                post = e["text"]
        if c == "indirect":
            if pre.endswith(", "):
                return pre + _low(core) + post
            return pre + core + post
        # downplayed: the decision is clear but framed as an aside (book: 'misleading')
        return pre + _low(core)

    def _qform(self, f: SemanticFrame, core: str, lv: dict) -> str:
        qf = lv["question_format"]
        if qf == "interrogative" or lv["style"] == "terse":
            return core
        if f.act == "ASK_KB":
            body = core.rstrip("?.").strip()
            body = body[:1].lower() + body[1:]
            return fill(self.lexicon["question_forms"][qf], body=body)
        forms = (self.lexicon.acts.get(f.act) or {}).get("question_forms")
        return forms[qf] if forms and qf in forms else core

    def _wrap(self, f: SemanticFrame, core: str, lv: dict, rng: random.Random) -> str:
        """Meaning-neutral sentences around the decision, from the lexicon's wrapper list."""
        pre, post = [], []
        intent = f.get("intent") if f.act == "ASK_KB" else None
        for w in self.lexicon.get("wrappers", []):
            level = lv.get(w["factor"])
            text = (w.get("text") or {}).get(level)
            if text is None:
                continue
            if any(lv.get(k) != v for k, v in (w.get("when") or {}).items()):
                continue
            if w.get("acts") and f.act not in w["acts"]:
                continue
            if intent is not None and intent in (w.get("skip_intents") or []):
                continue
            if isinstance(text, dict):
                if "by_intent" in text:
                    text = text["by_intent"].get(intent, text["default"])
                else:
                    text = text.get(f.act, text.get("default"))
                if text is None:
                    continue
            lines = list(text) if isinstance(text, list) else [text]
            if w["at"] == "first":
                pre[0:0] = lines
            elif w["at"] == "pre":
                pre.extend(lines)
            else:
                post.extend(lines)
        return " ".join(pre + [core] + post).strip()

    def _surface(self, f: SemanticFrame, text: str, lv: dict, rng: random.Random) -> str:
        d = lv["disfluency"]
        if d != "none":
            text = disfluency(text, rng, kind=d)
        n = lv["noise"]
        protect = set(self.lexicon.protect)
        if n == "typo":
            text = typo(text, rng, protect=protect)
        elif n == "asr":
            text = asr_noise(text, rng, protect=protect, confusions=self.lexicon.speech_confusions)
        return text
