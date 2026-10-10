"""Proof that every enriched case is the same test as its base case.

For each enriched scenario we check four layers, all of which must hold:

L1  Ground-truth identity. The scenario carries its base's ground truth by
    construction; we verify it by content hash and by recomputing the oracle's
    answer for the base spec from scratch.
L2  Ontological frame equivalence, turn by turn. Every customer turn on the
    oracle's path is realised under the scenario's factor levels and read back by
    the ontology-grounded FrameReader. The read frame must equal the base frame
    after canonicalisation (alias -> canonical individual). The ontological
    signature of the whole conversation (hash of the canonical frame sequence)
    must therefore equal the base signature.
L3  Oracle re-derivation (the functionality check of Ch. 8.1.1, applied to the
    enriched transcript). The oracle replays the *read-back* frames of the
    enriched conversation; the outcome signature and every per-stage component
    must equal the base answer.
L4  Geometric corroboration (reported, not gating): v-space cosine between the
    base and enriched surface forms shows how far the presentation moved, and
    the graded ontological distance (u-space tension for disjoint values) shows
    how far the meaning moved: always 0 for accepted scenarios.

A turn that fails L2 is re-realised with a fresh seed (up to 4 times) and the
seed is pinned for the run; a scenario that still fails is rejected and
excluded from analysis, as GEODE discards questions it cannot vouch for.

Negative controls show the instrument has power: meaning-changing edits (a
swapped disposition, a changed card digit, an inserted turn, a negation
toggled in the raw text, ...) must be rejected.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from ..core import Scenario
from ..knowledge.encoders import cosine
from ..ontology import Ontology, SemanticFrame as F, sequence_signature

MAX_RETRIES = 4


def gt_hash(answer: Any, components: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({"a": answer, "c": components}, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class ProofResult:
    sid: str
    qid: str
    levels: dict[str, str]
    accepted: bool
    gt_identical: bool
    frames_equivalent: bool
    oracle_rederived: bool
    base_signature: str
    enriched_signature: str
    n_turns: int
    retries: int
    max_frame_distance: float
    mean_surface_cosine: float
    min_surface_cosine: float
    turns: list[dict[str, Any]] = field(default_factory=list)



def _state_of(bot_act) -> str | None:
    """The dialog state a customer turn answers: the bot act that prompted it."""
    if isinstance(bot_act, dict):
        return bot_act.get("act") or bot_act.get("type") or (str(bot_act) if bot_act else None)
    return str(bot_act) if bot_act else None

class InvarianceValidator:
    def __init__(self, store, oracle, reader, realizer):
        self.store, self.oracle, self.reader, self.realizer = store, oracle, reader, realizer
        self.ont = Ontology.from_store(store)
        self._paths: dict[str, list[tuple[F, dict]]] = {}
        self._states: dict[str, list[str | None]] = {}

    # ----------------------------------------------------------------- paths
    def base_path(self, item) -> list[tuple[F, dict[str, Any]]]:
        if item.qid not in self._paths:
            spec, fx = item.metadata["spec"], item.metadata["fixture"]
            tr = self.oracle.run(spec, fx)
            by_id = {t["transaction_id"]: t for t in fx["transactions"]}
            path, last_bot, states = [], {}, []
            for ev in tr.events:
                if ev["role"] == "bot":
                    last_bot = ev["acts"][-1] if ev["acts"] else {}
                elif ev["role"] == "user":
                    t = by_id.get(last_bot.get("txn"), {})
                    path.append((ev["frame"], {"merchant": t.get("merchant"), "amount": t.get("amount")}))
                    states.append(_state_of(last_bot))
            self._paths[item.qid] = path
            self._states[item.qid] = states
        return self._paths[item.qid]

    # ----------------------------------------------------------------- proof
    def prove(self, scn: Scenario) -> ProofResult:
        item = scn.base
        path = self.base_path(item)
        turns, read_frames, overrides, retries = [], [], {}, 0
        for i, (frame, ctx) in enumerate(path):
            base_text = self.realizer.realize(frame, {}, f"{item.qid}|{i}|base", ctx)
            ok, g, text = False, None, ""
            for attempt in range(MAX_RETRIES + 1):
                seed = f"{scn.sid}|{i}" + (f"#{attempt}" if attempt else "")
                text = self.realizer.realize(frame, scn.factor_levels, seed, ctx)
                g = self.reader.read(text)
                if self.ont.equivalent(frame, g):
                    ok = True
                    break
            if attempt:
                overrides[i] = attempt
                retries += attempt
            read_frames.append(g)
            turns.append({"turn": i, "state": self._states.get(item.qid, [None] * (i + 1))[i],
                          "frame": str(frame), "frame_obj": frame, "read_obj": g, "base_text": base_text,
                          "enriched_text": text,
                          "read_back": str(g), "equivalent": ok,
                          "frame_distance": self.ont.frame_distance(frame, g),
                          "surface_cosine": cosine(self.store.v.encode(base_text), self.store.v.encode(text))})
        scn.metadata["seed_overrides"] = overrides
        # L1: carried truth == recomputed truth
        recomputed = self.oracle.run(item.metadata["spec"], item.metadata["fixture"])
        rc = {f"expected_{k}": v for k, v in recomputed.components().items()}
        gt_ok = gt_hash(scn.base.answer, scn.base.components) == gt_hash(recomputed.signature(), rc)
        # L3: replay the read-back frames through the oracle
        replay = self.oracle.run(item.metadata["spec"], item.metadata["fixture"], frames_override=read_frames)
        rp = {f"expected_{k}": v for k, v in replay.components().items()}
        oracle_ok = replay.signature() == item.answer and rp == item.components
        frames_ok = all(t["equivalent"] for t in turns)
        base_sig = sequence_signature(f for f, _ in path)
        enr_sig = sequence_signature(read_frames)
        cos = [t["surface_cosine"] for t in turns] or [1.0]
        res = ProofResult(scn.sid, item.qid, dict(scn.factor_levels), frames_ok and oracle_ok and gt_ok,
                          gt_ok, frames_ok, oracle_ok, base_sig, enr_sig, len(turns), retries,
                          max([t["frame_distance"] for t in turns] or [0.0]),
                          statistics.fmean(cos), min(cos), turns)
        scn.metadata["proof"] = {"accepted": res.accepted, "signature": enr_sig, "retries": retries}
        return res

    # -------------------------------------------------------- negative controls
    def _mutations(self, f: F) -> list[tuple[str, F]]:
        out = []
        a = f.act
        swap_d = {"RECOGNIZED": "UNRECOGNIZED", "UNRECOGNIZED": "RECOGNIZED", "UNSURE": "RECOGNIZED"}
        swap_dl = {"standard_delivery": "expedited_delivery", "expedited_delivery": "standard_delivery"}
        swap_c = {"SECURE_CARD": "REPLACE_CARD", "REPLACE_CARD": "SECURE_CARD"}
        sl = {k: (list(v) if isinstance(v, tuple) else v) for k, v in f.slots}
        if a == "INFORM_DISPOSITION":
            out.append(("swap_disposition", F.make(a, **{**sl, "disposition": swap_d[sl["disposition"]]})))
            if "card_action" in sl:
                out.append(("drop_card_request", F.make(a, **{k: v for k, v in sl.items() if k not in ("card_action", "delivery")})))
                out.append(("swap_card_action", F.make(a, **{**sl, "card_action": swap_c[sl["card_action"]],
                                                          **({"delivery": None} if sl["card_action"] == "REPLACE_CARD" else
                                                             {"delivery": "standard_delivery"})})))
            elif sl["disposition"] == "UNRECOGNIZED":
                out.append(("add_card_request", F.make(a, **{**sl, "card_action": "SECURE_CARD"})))
        elif a == "PROVIDE_CREDENTIAL":
            c = sl["credential"]
            out.append(("credential_digit", F.make(a, credential=c[:-1] + str((int(c[-1]) + 3) % 10))))
        elif a == "CONFIRM":
            out.append(("confirm_to_deny", F.make("DENY")))
        elif a == "REQUEST_CARD_ACTION" and sl.get("delivery"):
            out.append(("swap_delivery", F.make(a, **{**sl, "delivery": swap_dl[sl["delivery"]]})))
        elif a == "END":
            out.append(("end_to_human", F.make("REQUEST_HUMAN")))
        elif a == "ASK_KB":
            i = sl["intent"]
            if "entity" in sl:
                out.append(("kb_entity_swap", F.make(a, **{**sl, "entity": swap_dl[sl["entity"]]})))
            if i == "value":
                out.append(("kb_attr_swap", F.make(a, **{**sl, "attr": "days" if sl["attr"] == "fee" else "fee"})))
            if i == "window_check":
                out.append(("kb_boundary_shift", F.make(a, **{**sl, "days": int(sl["days"]) + 1})))
            if i == "plausibility":
                out.append(("kb_value_shift", F.make(a, **{**sl, "value": str(int(sl["value"]) + 5)})))
            if i == "faster":
                out.append(("kb_intent_swap", F.make(a, intent="cheaper")))
        return out

    def _text_mutations(self, text: str, frame: F | None = None) -> list[tuple[str, str]]:
        """Raw-text edits that unambiguously change meaning. An edit is only generated when the
        utterance carries exactly one cue of the kind being edited, so the edited sentence is a
        coherent statement of the opposite decision, not a self-contradiction."""
        out = []
        act = frame.act if frame is not None else None
        cues = self.reader.cue_counts(text)
        if cues["disposition"] != 1 or act not in (None, "INFORM_DISPOSITION"):
            pass
        elif re.search(r"\bI recognize\b", text):
            out.append(("text_negation", re.sub(r"\bI recognize\b", "I don't recognize", text, count=1)))
        elif re.search(r"\bI don't recognize\b", text):
            out.append(("text_negation", re.sub(r"\bI don't recognize\b", "I recognize", text, count=1)))
        m = re.search(r"\b\d{4}\b", text)
        if m and act in (None, "PROVIDE_CREDENTIAL"):
            d = m.group(0)
            out.append(("text_digit", text.replace(d, d[:2] + d[3] + d[2], 1)))
        if cues["delivery"] == 1 and re.search(r"\bexpedited\b", text) and \
                (act is None or frame.get("delivery") or frame.get("entity")):
            out.append(("text_delivery_word", re.sub(r"\bexpedited\b", "standard", text, count=1)))
        return out

    @staticmethod
    def _behaviour(tr) -> tuple:
        """Expected bot behaviour: the outcome plus the sequence of bot acts (with policy answers)."""
        acts = tuple((a.get("act"), json.dumps(a.get("answer"), default=str), a.get("txn"))
                     for e in tr.events if e["role"] == "bot" for a in e["acts"])
        return tr.signature(), acts

    def negative_controls(self, scenarios: list[Scenario], per_type: int = 40, seed: int = 11) -> dict[str, Any]:
        rng = random.Random(seed)
        stats: dict[str, dict[str, Any]] = {}

        def rec(kind: str, detected: bool, dist: float | None, gt_changed: bool | None):  # noqa: C901
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
        for scn in order:
            item = scn.base
            path = self.base_path(item)
            frames = [f for f, _ in path]
            spec, fx = item.metadata["spec"], item.metadata["fixture"]
            base_beh = self._behaviour(self.oracle.run(spec, fx, frames_override=frames))
            replay = lambda fr: self._behaviour(self.oracle.run(spec, fx, frames_override=fr)) != base_beh
            for i, (frame, ctx) in enumerate(path):
                for kind, mf in self._mutations(frame):
                    text = self.realizer.realize(mf, scn.factor_levels, f"neg|{scn.sid}|{i}|{kind}", ctx)
                    g = self.reader.read(text)
                    detected = not self.ont.equivalent(frame, g)
                    rec(kind, detected, self.ont.frame_distance(frame, g), replay(frames[:i] + [g] + frames[i + 1:]))
                base_text = self.realizer.realize(frame, scn.factor_levels, f"{scn.sid}|{i}", ctx)
                for kind, mt in self._text_mutations(base_text, frame):
                    g = self.reader.read(mt)
                    rec(kind, not self.ont.equivalent(frame, g), self.ont.frame_distance(frame, g),
                        replay(frames[:i] + [g] + frames[i + 1:]))
            # sequence-level: an inserted off-task turn changes the ontological signature
            if frames:
                k = rng.randrange(len(frames) + 1)
                ins = frames[:k] + [F.make("CHITCHAT")] + frames[k:]
                rec("insert_turn", sequence_signature(ins) != sequence_signature(frames), None, replay(ins))
        out = {}
        for k, s in sorted(stats.items()):
            out[k] = {"n": s["n"], "detection_rate": s["detected"] / max(1, s["n"]),
                      "mean_ontological_distance": statistics.fmean(s["distances"]) if s["distances"] else None,
                      "behaviour_changed_rate": (s["gt_changed"] / s["gt_checked"]) if s["gt_checked"] else None}
        total = sum(s["n"] for s in stats.values())
        out["_overall"] = {"n": total, "detection_rate": sum(s["detected"] for s in stats.values()) / max(1, total)}
        return out

    # ------------------------------------------------------------- summary
    @staticmethod
    def summarize(proofs: list[ProofResult], factor_names: list[str]) -> dict[str, Any]:
        n = len(proofs)
        by_level: dict[str, dict[str, Any]] = {}
        for f in factor_names:
            for p in proofs:
                lv = p.levels.get(f)
                key = f"{f}={lv}"
                d = by_level.setdefault(key, {"scenarios": 0, "accepted": 0, "turns": 0, "cos": []})
                d["scenarios"] += 1
                d["accepted"] += int(p.accepted)
                d["turns"] += p.n_turns
                d["cos"].append(p.mean_surface_cosine)
        lvl = {k: {"scenarios": v["scenarios"], "acceptance": v["accepted"] / v["scenarios"],
                   "turns": v["turns"], "mean_surface_cosine": statistics.fmean(v["cos"])}
               for k, v in sorted(by_level.items())}
        return {
            "scenarios": n,
            "accepted": sum(p.accepted for p in proofs),
            "acceptance_rate": sum(p.accepted for p in proofs) / max(1, n),
            "L1_gt_identical": sum(p.gt_identical for p in proofs),
            "L2_frames_equivalent": sum(p.frames_equivalent for p in proofs),
            "L3_oracle_rederived": sum(p.oracle_rederived for p in proofs),
            "signatures_equal": sum(p.base_signature == p.enriched_signature for p in proofs),
            "turns_checked": sum(p.n_turns for p in proofs),
            "scenarios_needing_reseed": sum(p.retries > 0 for p in proofs),
            "max_frame_distance_accepted": max([p.max_frame_distance for p in proofs if p.accepted] or [0.0]),
            "surface_cosine_mean": statistics.fmean([p.mean_surface_cosine for p in proofs]) if proofs else None,
            "surface_cosine_min": min([p.min_surface_cosine for p in proofs]) if proofs else None,
            "by_level": lvl,
        }

    def evidence_triples(self, proofs: list[ProofResult], items: dict[str, Any]) -> list[tuple[str, str, str]]:
        tr = []
        seen = set()
        for p in proofs:
            it = items[p.qid]
            if p.qid not in seen:
                seen.add(p.qid)
                tr += [(f"cv:case_{p.qid}", "a", "cv:BaseCase"),
                       (f"cv:case_{p.qid}", "cv:frameSignature", json.dumps(p.base_signature)),
                       (f"cv:case_{p.qid}", "cv:groundTruthHash", json.dumps(gt_hash(it.answer, it.components)))]
            sid = p.sid.replace("~", "_")
            tr += [(f"cv:case_{sid}", "a", "cv:EnrichedCase"),
                   (f"cv:case_{sid}", "cv:derivedFrom", f"cv:case_{p.qid}"),
                   (f"cv:case_{sid}", "cv:frameSignature", json.dumps(p.enriched_signature)),
                   (f"cv:case_{sid}", "cv:groundTruthHash", json.dumps(gt_hash(it.answer, it.components))),
                   (f"cv:case_{sid}", "cv:invarianceAccepted", json.dumps(str(p.accepted).lower()))]
        return tr
