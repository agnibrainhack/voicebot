"""Separate invariance validation for LLM-written enrichments.

The template arm and the LLM arm share the scenario-level test (``InvarianceValidator.
_conclude``: L1 truth identity, L3 oracle replay of the read-back cards, signature
equality). What differs is how far each turn is trusted before that test:

per turn, in order (all must hold; ``_accept_turn``)
  G0 form            the raw LLM text is one spoken line with no internal names;
  G1 facts           no figure or merchant the base meaning does not carry; every
                     decision value kept;
  G3 one decision    no competing decision cues beyond what the factor levels allow;
  G2 meaning (L2)    the rule reader reads the *final* utterance (after surface noise)
                     in its dialog state and recovers the base card, admissible there;
  G2' dual read      (dual_read="gate") an LLM reader that never saw the base also
                     recovers the base card.
  A failing turn is regenerated (``max_attempts``) with feedback naming the failed gate -
  never the answer. A turn that still fails makes the scenario *not materialised* for the
  LLM arm: it is excluded from runs and attribution and reported with its reasons.

instrument power on LLM text (``negative_controls``), reported apart from the template arm
  * deterministic meaning-changing edits of the accepted LLM utterances (negation,
    digit, delivery, disposition, figure and merchant swaps, yes/no flip) - each edited
    utterance goes through the SAME ``_accept_turn`` and must be rejected;
  * adversarial rewrites: the LLM is asked to change exactly one meaning element of an
    accepted utterance (swap the disposition, the delivery, a digit ...). Each rewrite
    must be rejected; accepted rewrites are listed for audit together with what both
    readers made of them (an accepted rewrite is either a gate miss or an adversary that
    failed to change the meaning - the audit list lets a person decide);
  * wrong state: the accepted LLM utterance answered in a state that does not accept
    its meaning must be rejected.

manipulation check: ``factor_realisation`` per turn -> realised levels per scenario
(a factor counts as realised when it is visible on at least half of the turns it applies
to). Attribution for the LLM arm uses the realised levels.
"""
from __future__ import annotations

import random
import re
import statistics
from collections import Counter
from typing import Any

from ...core import Scenario
from ...knowledge.encoders import cosine
from ..invariance import InvarianceValidator, ProofResult
from ..realizers import BASE_LEVELS
from .client import CachedLLM, LLMUnavailable, parse_json_object
from .extractor import LLMFrameExtractor
from .prompts import build_adversary_prompt
from .realizer import LLMRealizer, _clean

MAX_LLM_ATTEMPTS = 3


class LLMInvarianceValidator(InvarianceValidator):
    def __init__(self, store, oracle, reader, llm_realizer: LLMRealizer, template_validator: InvarianceValidator,
                 dual_read: str = "gate", extractor_llm: CachedLLM | None = None,
                 adversary_llm: CachedLLM | None = None, max_attempts: int = MAX_LLM_ATTEMPTS):
        super().__init__(store, oracle, reader, llm_realizer)
        self.llm_realizer = llm_realizer
        self.template_validator = template_validator
        # share the oracle paths with the template arm: both arms prove the same turns
        self._paths, self._states = template_validator._paths, template_validator._states
        self.dual_read = dual_read if extractor_llm is not None else "off"
        self.extractor = LLMFrameExtractor(extractor_llm, store) if extractor_llm is not None else None
        self.adversary = adversary_llm
        self.max_attempts = max_attempts
        self.gate_failures: Counter = Counter()
        self.dual_agree: list[bool] = []

    # ------------------------------------------------------------------ one turn
    def _accept_turn(self, text_raw: str, text_final: str, frame, st, ctx, levels, licensed: list[str],
                     check_form: bool = True) -> dict[str, Any]:
        """Every per-turn gate on one utterance; the controls call exactly this function."""
        g = self.llm_realizer.gates
        res: dict[str, Any] = {}
        reasons: list[str] = []
        if check_form:
            f0 = g.form(text_raw)
            res["G0_form"] = f0.ok
            reasons += f0.reasons
        f1 = g.facts(text_raw, frame, ctx, levels, licensed)
        res["G1_facts"] = f1.ok
        reasons += f1.reasons
        f3 = g.one_clear_decision(text_final, frame, levels)
        res["G3_one_decision"] = f3.ok
        reasons += f3.reasons
        read = self.reader.read(text_final, st)
        same = self.ont.equivalent(frame, read, st)
        res["G2_reader"] = same
        res["read"] = read
        if not same:
            ok_adm, why = self.ont.admissible(read, st)
            reasons.append("the rule reader recovers a different meaning: " + str(self.ont.canonicalize(read, st))
                           + ("" if ok_adm else f" ({why})"))
        if self.extractor is not None and self.dual_read in ("gate", "report"):
            dual = self.extractor.read(text_final, st, ctx)
            agree = dual is not None and self.ont.equivalent(frame, dual, st)
            res["G2_dual_read"] = agree
            res["dual_read"] = dual
            if self.dual_read == "gate" and not agree:
                reasons.append("the independent reader recovers a different meaning: "
                               + (str(self.ont.canonicalize(dual, st)) if dual is not None else "nothing"))
        gate_keys = [k for k in ("G0_form", "G1_facts", "G3_one_decision", "G2_reader") if k in res]
        if self.dual_read == "gate" and "G2_dual_read" in res:
            gate_keys.append("G2_dual_read")
        res["accepted"] = all(res[k] for k in gate_keys)
        res["reasons"] = reasons
        return res

    # ------------------------------------------------------------------ proof
    def prove(self, scn: Scenario) -> ProofResult:
        item = scn.base
        path = self.base_path(item)
        states = self._states.get(item.qid, [None] * len(path))
        levels = scn.factor_levels
        turns, read_frames, overrides, retries = [], [], {}, 0
        realised: dict[str, list[bool]] = {}
        unavailable = False
        for i, (frame, ctx) in enumerate(path):
            st = states[i]
            ctx = dict(ctx or {})
            feedback, chosen, attempts = None, None, []
            for attempt in range(self.max_attempts):
                seed = f"{scn.sid}|{i}" + (f"#{attempt}" if attempt else "")
                t = self.llm_realizer.generate(frame, levels, seed, ctx, st, feedback)
                if t.error:
                    attempts.append({"attempt": attempt, "error": t.error})
                    unavailable = True
                    break
                chk = self._accept_turn(t.raw, t.final, frame, st, ctx, levels, [t.base_text, t.template_text])
                attempts.append({"attempt": attempt, "raw": t.raw, "final": t.final, "json_reply": t.json_reply,
                                 **{k: v for k, v in chk.items() if k not in ("read", "dual_read")},
                                 "read": str(chk["read"]),
                                 "dual_read": str(chk.get("dual_read")) if "dual_read" in chk else None})
                for k in ("G0_form", "G1_facts", "G3_one_decision", "G2_reader", "G2_dual_read"):
                    if k in chk and not chk[k]:
                        self.gate_failures[k] += 1
                if "G2_dual_read" in chk:
                    self.dual_agree.append(bool(chk["G2_dual_read"]))
                chosen = (t, chk, seed, attempt)
                if chk["accepted"]:
                    break
                feedback = "; ".join(chk["reasons"][:4]) or "the meaning changed"
            if chosen is None:                            # LLM unavailable before any text
                base_text = self.template_validator.realizer.realize(frame, {}, "base", ctx)
                turns.append({"turn": i, "state": st, "frame": str(frame), "frame_obj": frame, "read_obj": None,
                              "base_text": base_text, "enriched_text": "", "card": self.ont.card(frame, st),
                              "read_card": self.ont.card(None, st), "read_back": "None", "equivalent": False,
                              "frame_distance": 1.0, "surface_cosine": 0.0, "arm": "llm", "attempts": attempts,
                              "llm_unavailable": True})
                read_frames.append(frame)
                continue
            t, chk, seed, attempt = chosen
            if attempt:
                overrides[i] = attempt
                retries += attempt
            if chk["accepted"]:
                self.llm_realizer.pin(seed, frame, levels, st, t.final)
            g = chk["read"]
            read_frames.append(self.ont.canonicalize(g, st))
            fr = self.llm_realizer.gates.factor_realisation(t.final, t.base_text, frame, levels, ctx,
                                                            self.llm_realizer.asides(frame, levels, ctx))
            for k, v in fr.items():
                if v["detected"] is not None:
                    realised.setdefault(k, []).append(bool(v["detected"]))
            turns.append({"turn": i, "state": st, "frame": str(frame), "frame_obj": frame, "read_obj": g,
                          "base_text": t.base_text, "template_text": t.template_text, "llm_raw": t.raw,
                          "enriched_text": t.final, "card": self.ont.card(frame, st), "read_card": self.ont.card(g, st),
                          "read_back": str(g), "equivalent": bool(chk["accepted"]),
                          "gates": {k: chk[k] for k in ("G0_form", "G1_facts", "G3_one_decision", "G2_reader",
                                                         "G2_dual_read") if k in chk},
                          "dual_read": str(chk.get("dual_read")) if "dual_read" in chk else None,
                          "reasons": chk["reasons"], "attempts": attempts, "factor_realisation": fr,
                          "frame_distance": self.ont.frame_distance(frame, g, st),
                          "surface_cosine": cosine(self.store.v.encode(t.base_text), self.store.v.encode(t.final)),
                          "template_cosine": cosine(self.store.v.encode(t.template_text), self.store.v.encode(t.final)),
                          "arm": "llm"})
        scn.metadata["seed_overrides"] = overrides
        rl = {k: (sum(v) * 2 >= len(v)) for k, v in realised.items() if v}
        realised_levels = {k: (lvl if rl.get(k, True) else BASE_LEVELS.get(k, lvl))
                           for k, lvl in levels.items()}
        scn.metadata["realised_levels"] = realised_levels
        scn.metadata["unrealised_factors"] = sorted(k for k, ok in rl.items() if not ok)
        extra = {"llm_unavailable": unavailable, "realised_levels": realised_levels,
                 "unrealised_factors": scn.metadata["unrealised_factors"],
                 "turns_needing_retry": sum(1 for t in turns if len(t.get("attempts", [])) > 1)}
        return self._conclude(scn, path, states, turns, read_frames, retries, arm="llm",
                              extra_ok=not unavailable, extra=extra)

    # ------------------------------------------------------------------ controls
    _SWAPS = [
        ("text_negation", r"\bI recognize\b", "I don't recognize"),
        ("text_negation", r"\bI don't recognize\b", "I recognize"),
        ("text_negation", r"\b(it's|it is|that's|that is) mine\b", r"\1 not mine"),
        ("text_negation", r"\b(it's|it is|that's) not mine\b", r"\1 mine"),
        ("text_delivery_word", r"\b(expedited|express|rush|overnight)\b", "standard"),
        ("text_delivery_word", r"\b(standard|regular|normal)\b", "expedited"),
        ("text_card_action", r"\b(secure|lock|freeze|block)\b", "replace"),
        ("text_card_action", r"\breplace\b", "lock"),
    ]

    def _llm_text_mutations(self, text: str, frame, ctx) -> list[tuple[str, str]]:
        out, seen = [], set()
        for kind, pat, rep in self._SWAPS:
            if kind in seen or not re.search(pat, text, flags=re.I):
                continue
            new = re.sub(pat, rep, text, count=1, flags=re.I)
            if new != text:
                out.append((kind, new))
                seen.add(kind)
        # a bare yes / no (the only cue in the turn) flipped: the state then reads the opposite meaning;
        # flipping a "yes" inside a longer sentence would only make it self-contradictory, not opposite
        bare = re.fullmatch(r"\W*(?:um,?\s*|uh,?\s*)?(yes|yeah|yep|sure|no|nope)\W*", text, flags=re.I)
        if bare:
            word = bare.group(1).lower()
            flip = "No." if word in ("yes", "yeah", "yep", "sure") else "Yes."
            out.append(("text_yes_no", flip))
        m = re.search(r"\b\d{4}\b", text)
        if m:
            d = m.group(0)
            out.append(("text_digit", text.replace(d, d[:3] + str((int(d[3]) + 3) % 10), 1)))
        m = re.search(r"\$?(\d+)(?!\d)", text)
        if m and not re.fullmatch(r"\d{4}", m.group(1)):
            n = int(m.group(1))
            out.append(("text_figure_shift", text[:m.start(1)] + str(n + 5) + text[m.end(1):]))
        own = str((ctx or {}).get("merchant") or "")
        if own and own.lower() in text.lower():
            other = next(m for m in ("Starbucks", "Shell", "Target") if m.lower() != own.lower())
            out.append(("text_merchant_swap", re.sub(re.escape(own), other, text, count=1, flags=re.I)))
        return out

    def negative_controls(self, proofs: list[ProofResult], scenarios: list[Scenario], per_type: int = 40,
                          adversarial: int = 20, seed: int = 11) -> dict[str, Any]:
        rng = random.Random(seed)
        by_sid = {s.sid: s for s in scenarios}
        stats: dict[str, dict[str, int]] = {}
        audit: list[dict[str, Any]] = []
        kg = self.ont.kg
        prompt_states = list(kg.prompt_states) if kg is not None else []

        def rec(kind, detected, row=None):
            s = stats.setdefault(kind, {"n": 0, "detected": 0})
            if s["n"] >= per_type:
                return False
            s["n"] += 1
            s["detected"] += int(detected)
            if not detected and row is not None:
                audit.append({"kind": kind, **row})
            return True

        accepted = [p for p in proofs if p.accepted]
        rng.shuffle(accepted)
        adv_left = adversarial if self.adversary is not None else 0
        adv = {"n": 0, "rejected": 0, "effective": 0, "rejected_effective": 0, "unavailable": 0}
        for p in accepted:
            scn = by_sid.get(p.sid)
            if scn is None:
                continue
            path = self.base_path(scn.base)
            for t in p.turns:
                i, st = t["turn"], t["state"]
                frame, ctx = path[i]
                text = t["enriched_text"]
                licensed = [t["base_text"], t["template_text"]]
                for kind, mt in self._llm_text_mutations(text, frame, ctx):
                    chk = self._accept_turn(mt, mt, frame, st, ctx, scn.factor_levels, licensed, check_form=False)
                    rec(kind, not chk["accepted"], {"sid": p.sid, "turn": i, "state": st, "original": text,
                                                     "edited": mt, "read": str(chk["read"])})
                # wrong state: same words answered where this meaning is not accepted
                wrong = [w for w in prompt_states if w != st and not kg.accepts(w, self.ont.canonicalize(frame, w).act)]
                if wrong:
                    w = wrong[rng.randrange(len(wrong))]
                    g = self.reader.read(text, w)
                    rec("wrong_state", not self.ont.equivalent(frame, g, w),
                        {"sid": p.sid, "turn": i, "state": w, "original": text, "edited": text, "read": str(g)})
                # adversarial rewrites by the LLM
                muts = self.template_validator._mutations(frame)
                if adv_left > 0 and muts:
                    kind, target = muts[rng.randrange(len(muts))]
                    spoken = scn.factor_levels.get("numeric_format") == "spoken"
                    system, user = build_adversary_prompt(text, target, st, ctx, spoken)
                    try:
                        obj = parse_json_object(self.adversary.invoke(system, user, salt=f"adv|{p.sid}|{i}|{kind}"))
                    except LLMUnavailable:
                        adv["unavailable"] += 1
                        adv_left = 0
                        continue
                    if not obj or not isinstance(obj.get("utterance"), str):
                        continue
                    adv_left -= 1
                    at = _clean(obj["utterance"])
                    chk = self._accept_turn(at, at, frame, st, ctx, scn.factor_levels, licensed + [at], check_form=False)
                    effective = self.ont.equivalent(target, chk["read"], st)
                    adv["n"] += 1
                    adv["rejected"] += int(not chk["accepted"])
                    adv["effective"] += int(effective)
                    adv["rejected_effective"] += int(effective and not chk["accepted"])
                    if chk["accepted"]:
                        audit.append({"kind": f"adversarial:{kind}", "sid": p.sid, "turn": i, "state": st,
                                      "original": text, "edited": at, "target": str(target), "read": str(chk["read"]),
                                      "dual_read": str(chk.get("dual_read"))})
        out = {k: {"n": v["n"], "detection_rate": v["detected"] / max(1, v["n"])} for k, v in sorted(stats.items())}
        total = sum(v["n"] for v in stats.values())
        out["_overall"] = {"n": total, "detection_rate": sum(v["detected"] for v in stats.values()) / max(1, total)}
        out["_adversarial"] = {**adv, "rejection_rate": adv["rejected"] / max(1, adv["n"]),
                               "rejection_rate_when_meaning_changed": adv["rejected_effective"] / max(1, adv["effective"]),
                               "note": "an accepted rewrite is a gate miss or an adversary that kept the meaning; "
                                       "see the audit list"}
        out["_audit"] = audit[:200]
        return out

    # ------------------------------------------------------------------ summary
    def summarize_llm(self, proofs: list[ProofResult], factor_names: list[str]) -> dict[str, Any]:
        base = self.summarize(proofs, factor_names)
        turns = [t for p in proofs for t in p.turns]
        att = [len(t.get("attempts", [])) for t in turns]
        fr: dict[str, dict[str, int]] = {}
        for t in turns:
            for k, v in (t.get("factor_realisation") or {}).items():
                if v["detected"] is None:
                    continue
                d = fr.setdefault(f"{k}={v['level']}", {"turns": 0, "realised": 0})
                d["turns"] += 1
                d["realised"] += int(v["detected"])
        not_mat = [p for p in proofs if not p.accepted]
        reasons = Counter()
        for p in not_mat:
            for t in p.turns:
                if not t["equivalent"]:
                    for r in t.get("reasons", [])[:2] or (["llm unavailable"] if t.get("llm_unavailable") else []):
                        reasons[re.sub(r"\d+", "#", r)[:90]] += 1
        tc = [t["template_cosine"] for t in turns if "template_cosine" in t]
        return {**base, "arm": "llm",
                "gate_failures_by_attempt": dict(self.gate_failures),
                "mean_attempts_per_turn": statistics.fmean(att) if att else None,
                "turns_first_attempt_accepted": sum(1 for t in turns if len(t.get("attempts", [])) == 1 and t["equivalent"]),
                "dual_read": {"mode": self.dual_read, "checks": len(self.dual_agree),
                              "agreement": (sum(self.dual_agree) / len(self.dual_agree)) if self.dual_agree else None},
                "llm_unavailable_scenarios": sum(1 for p in proofs if p.extra.get("llm_unavailable")),
                # the transport errors behind 'llm unavailable' (first few distinct messages)
                "llm_errors": Counter(a.get("error") for p in proofs for t in p.turns
                                      for a in t.get("attempts", []) if a.get("error")).most_common(5),
                "not_materialised": len(not_mat),
                "top_rejection_reasons": reasons.most_common(12),
                "factor_realisation": {k: {**v, "rate": v["realised"] / max(1, v["turns"])} for k, v in sorted(fr.items())},
                "scenarios_with_unrealised_factors": sum(1 for p in proofs if p.extra.get("unrealised_factors")),
                "cosine_to_template_text_mean": statistics.fmean(tc) if tc else None,
                "realizer": dict(self.llm_realizer.stats)}
