"""The invariance proof for a compact policy graph (enrich/invariance.py, layers L1-L4 unchanged).

What changes is only what the v1 validator read from charge-review specifics:

* the base path: every customer turn with the prompt state it answers and the loop item it is
  about (the realiser's context for ``specificity``), straight from the oracle's events;
* the negative controls, mined from the graph: a value swapped for another member of the slot's
  value set, a changed credential digit, confirm <-> deny, end -> a request for a person, a
  policy question about another entity, attribute or limit, the lexicon's value-word swap in the
  raw text, the same words answered in a state that does not accept them, an inserted turn.
"""
from __future__ import annotations

import random
import re
import statistics
from typing import Any

from ..enrich.invariance import InvarianceValidator
from ..ontology import SemanticFrame as F, sequence_signature


def _graph():
    from ..domain import current
    return current().graph


class GraphInvarianceValidator(InvarianceValidator):
    def base_path(self, item) -> list[tuple[F, dict[str, Any]]]:
        if item.qid not in self._paths:
            tr = self.oracle.run(item.metadata["spec"], item.metadata["fixture"])
            path, states = [], []
            for ev in tr.events:
                if ev["role"] == "user":
                    path.append((ev["frame"], {**(ev.get("item") or {}), "state": ev.get("state")}))
                    states.append(ev.get("state"))
            self._paths[item.qid] = path
            self._states[item.qid] = states
        return self._paths[item.qid]

    def _mutations(self, f: F) -> list[tuple[str, F]]:
        g = _graph()
        out: list[tuple[str, F]] = []
        sl = {k: (list(v) if isinstance(v, tuple) else v) for k, v in f.slots}
        a = f.act
        if a == "ASK_KB":
            if sl.get("intent") == "value":
                num = g.numeric_attributes()
                spec = num.get(sl["attr"]) or {}
                other = [m for m in spec.get("members", {}) if m != sl.get("entity")]
                if other:
                    out.append(("kb_entity_swap", F.make(a, **{**sl, "entity": other[0]})))
                attrs = [x for x, s in num.items() if x != sl["attr"] and sl.get("entity") in s["members"]]
                from ..domain import current
                asked = current().get("lexicon.kb_questions.attributes") or {}
                attrs = [x for x in attrs if not asked or x in asked]
                if attrs:
                    out.append(("kb_attr_swap", F.make(a, **{**sl, "attr": attrs[0]})))
            elif sl.get("intent") == "limit":
                keys = [k for k, lv in g.limits.items() if not lv.get("internal") and k != sl.get("key")]
                if keys:
                    out.append(("kb_limit_swap", F.make(a, **{**sl, "key": keys[0]})))
            return out
        if a == "CONFIRM":
            return [("confirm_to_deny", F.make("DENY"))]
        if a == "DENY":
            return [("deny_to_confirm", F.make("CONFIRM"))]
        if a == "END":
            return [("end_to_human", F.make((g.glob.get("human") or {}).get("act", "REQUEST_HUMAN")))]
        for s, sp in g.slot_specs(a).items():
            v = sl.get(s)
            if v is None or sp.get("correction_of"):
                continue
            if sp["set"] == "credential_code":
                out.append(("credential_digit", F.make(a, **{**sl, s: v[:-1] + str((int(v[-1]) + 3) % 10)})))
                continue
            others = [m for m in g.members(sp["set"]) if m != v]
            if others:
                kw = {**sl, s: others[0]}
                if sl.get("corrected_from") == others[0]:
                    kw.pop("corrected_from")
                out.append((f"swap_{s}", F.make(a, **kw)))
        return out

    def _text_mutations(self, text: str, frame: F | None = None) -> list[tuple[str, str]]:
        out = []
        act = frame.act if frame is not None else None
        g = _graph()
        has_code = act is not None and any(sp["set"] == "credential_code" for sp in g.slot_specs(act).values())
        m = re.search(r"\b\d{4}\b", text)
        if m and has_code:
            d = m.group(0)
            out.append(("text_digit", text.replace(d, d[:2] + d[3] + d[2], 1)))
        ctl = self.lexicon.get("controls") or {}
        if ctl.get("text_value_word"):
            word_re, other = ctl["text_value_word"]
            if len(re.findall(word_re, text)) == 1 and not re.search(r"\b" + re.escape(other) + r"\b", text):
                out.append(("text_value_word", re.sub(word_re, other, text, count=1)))
        if ctl.get("text_negation") and frame is not None and frame.act in self.lexicon.acts:
            pos_re, pos, neg_re, neg = ctl["text_negation"]
            slot = self.lexicon.acts[frame.act].get("slot")
            if slot and self.reader.cue_counts(text).get(slot) == 1:
                if re.search(pos_re, text):
                    out.append(("text_negation", re.sub(pos_re, neg, text, count=1)))
                elif re.search(neg_re, text):
                    out.append(("text_negation", re.sub(neg_re, pos, text, count=1)))
        return out

    def negative_controls(self, scenarios, per_type: int = 40, seed: int = 11) -> dict[str, Any]:
        rng = random.Random(seed)
        stats: dict[str, dict[str, Any]] = {}

        def rec(kind, detected, dist, gt_changed):
            s = stats.setdefault(kind, {"n": 0, "detected": 0, "distances": [], "gt_changed": 0, "gt_checked": 0})
            if s["n"] >= per_type:
                return
            s["n"] += 1
            s["detected"] += int(detected)
            if dist is not None:
                s["distances"].append(dist)
            if gt_changed is not None:
                s["gt_checked"] += 1
                s["gt_changed"] += int(gt_changed)

        order = list(scenarios)
        rng.shuffle(order)
        kg = self.ont.kg
        prompt_states = list(kg.prompt_states) if kg is not None else []
        for scn in order:
            item = scn.base
            path = self.base_path(item)
            states = self._states.get(item.qid, [None] * len(path))
            frames = [f for f, _ in path]
            spec, fx = item.metadata["spec"], item.metadata["fixture"]
            base_beh = self._behaviour(self.oracle.run(spec, fx, frames_override=frames))
            replay = lambda fr: self._behaviour(self.oracle.run(spec, fx, frames_override=fr)) != base_beh
            for i, (frame, ctx) in enumerate(path):
                st = states[i]
                for kind, mf in self._mutations(frame):
                    text = self.realizer.realize(mf, scn.factor_levels, f"neg|{scn.sid}|{i}|{kind}", ctx)
                    gr = self.ont.canonicalize(self.reader.read(text, st), st)
                    rec(kind, not self.ont.equivalent(frame, gr, st), self.ont.frame_distance(frame, gr, st),
                        replay(frames[:i] + [gr] + frames[i + 1:]))
                base_text = self.realizer.realize(frame, scn.factor_levels, f"{scn.sid}|{i}", ctx)
                for kind, mt in self._text_mutations(base_text, frame):
                    gr = self.ont.canonicalize(self.reader.read(mt, st), st)
                    rec(kind, not self.ont.equivalent(frame, gr, st), self.ont.frame_distance(frame, gr, st),
                        replay(frames[:i] + [gr] + frames[i + 1:]))
                wrong = [w for w in prompt_states if w != st and not kg.accepts(w, self.ont.canonicalize(frame, w).act)]
                if wrong:
                    w = wrong[rng.randrange(len(wrong))]
                    rec("wrong_state", not self.ont.equivalent(frame, self.reader.read(base_text, w), w), None, None)
            if frames:
                k = rng.randrange(len(frames) + 1)
                ins = frames[:k] + [F.make("CHITCHAT")] + frames[k:]
                ins_states = list(states[:k]) + [states[k - 1] if k else (prompt_states or [None])[0]] + list(states[k:])
                rec("insert_turn", sequence_signature(ins, ins_states) != sequence_signature(frames, states), None,
                    replay(ins))
        out = {}
        for k, s in sorted(stats.items()):
            out[k] = {"n": s["n"], "detection_rate": s["detected"] / max(1, s["n"]),
                      "mean_ontological_distance": statistics.fmean(s["distances"]) if s["distances"] else None,
                      "behaviour_changed_rate": (s["gt_changed"] / s["gt_checked"]) if s["gt_checked"] else None}
        total = sum(s["n"] for s in stats.values())
        out["_overall"] = {"n": total, "detection_rate": sum(s["detected"] for s in stats.values()) / max(1, total)}
        return out
