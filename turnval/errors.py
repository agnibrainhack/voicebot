"""Block 4 - error operators: one injected bot error per conversation, the two turns after it, and the labels.

A design row names the error family, its position (early / middle / late among the turns where it can happen), its
subtlety (one HRT or several) and whether the bot recovers in the next two turns. The conversation stops two turns
after the error.

Families and what they do (every operator records the HRTs / transition it breaks, so V1/V2 would flag it):
  transition_skip          the response of a later state (a step skipped)
  transition_wrong_branch  the state another tool outcome leads to (the bot did not follow its tool)
  transition_repeat        the previous state again, although it was answered
  transition_regress       an earlier state (going back)
  premature_handoff        transfers / ends the call when the flow continues
  missing_handoff          carries on when the customer asked for a person
  content_missing          the right state, with one (subtle) or several (blatant) MUST_CONVEY items left out
  content_forbidden        the right state, saying something a MUST_NOT forbids
  metadata_wrong           the right state, a spoken value that does not match its metadata field
  order_violation          the right state, required items in the wrong order
  guardrail_local          the right state, breaking a box guardrail (how it is said)
  guardrail_global         the right state, breaking an AOP-wide policy
  context_ignored          the customer's interjection / correction is not handled as the graph requires
  wrapper_misread          the customer's words mean one outcome, the bot acts on another (V1/V2 pass: only V3 sees it)
  hallucination            the right state plus an invented fact

Recovery: a state error recovers by giving the correct state's response on the next turn; a content error recovers by
giving correct responses. "stays_off" repeats the error kind on both following turns. Turns after the error are
labelled against the correct path; a turn back on the correct state is labelled correct.

Side negatives: extra erroneous responses at other turns, for the same context (not shown in the conversation): they
give the contrastive groups more negatives.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .scenario import HANDOFF_STATE, Conversation, GoldTurn, Simulator

STATE_FAMILIES = {"transition_skip", "transition_wrong_branch", "transition_repeat", "transition_regress",
                  "premature_handoff", "missing_handoff", "wrapper_misread"}
CONTENT_FAMILIES = {"content_missing", "content_forbidden", "metadata_wrong", "order_violation", "guardrail_local",
                    "guardrail_global", "hallucination", "context_ignored"}
ALTERNATIVES = {
    "set_4ssn_metadata.match": ["INPUT_ERROR", "ATTEMPT_ERROR", "PROFILE_SINGLE_MATCH", "PROFILE_MULTI_MATCH", "PROFILE_NO_MATCH"],
    "set_fullssn_metadata.match": ["INPUT_ERROR", "ATTEMPT_ERROR", "PROFILE_SINGLE_MATCH", "PROFILE_MULTI_MATCH", "PROFILE_NO_MATCH"],
    "set_cm15_metadata.match": ["INPUT_ERROR", "ATTEMPT_ERROR", "PROFILE_IDENTIFIED", "PROFILE_NOT_IDENTIFIED"],
    "set_cm5_metadata.match": ["INPUT_ERROR", "ATTEMPT_ERROR", "PROFILE_IDENTIFIED", "PROFILE_NOT_IDENTIFIED", "SEARCHING"],
    "collected_4SSN": ["INPUT_ERROR", "COLLECTED"], "collected_5CM": ["INPUT_ERROR", "COLLECTED"],
    "collected_zipcode": ["INPUT_ERROR", "COLLECTED"],
}
CUSTOMER_DRIVEN = {
    "card_confirm": [True, False],
    "check_fullssn_or_cm15.match": ["FULL_SSN", "CM_15", "NO_PREFERENCE_SHARED"],
    "check_OTP_channel.match": ["SMS", "EMAIL", "NO_PREFERENCE_SHARED"],
    "txn_answer": ["SEND_TO_AMEX", "PROVIDE_MORE_INFO", "TRANSACTION_NOT_VERIFIED"],
    "check_verification.match": [("VALIDATE", "VERIFICATION_PASSED"), ("VALIDATE", "repeatQuestion"),
                                 ("INPUT_USER_ERROR", None), ("NO_USER_RESPONSE", None)],
}
GENERIC_FORBIDDEN = {
    "User Identification": {"ref": "global_policies:UI_CLAUSE_035/B", "head": "Unrequired request",
                            "tail": "ask for a piece of information the AOP does not require (e.g. email address, mother's maiden name)"},
    "Charge Verification": {"ref": "global_policies:CV_CLAUSE_007/A", "head": "Anything else",
                            "tail": "ask the customer if there is anything else they would like to review or get help with"},
}


@dataclass
class BotSpec:
    """What one bot response must be: the correct one for a state, or an erroneous one."""
    kind: str                                   # gold | error
    state: str                                  # the state whose response this is (the wrong one for state errors)
    gold_state: str                             # the state the bot should be in (labels are judged against it)
    boxes: list[str]
    hrts: dict[str, Any]                        # the HRTs of `state` (what the writer follows)
    gold_hrts: dict[str, Any]                   # the HRTs of `gold_state` (what the turn is judged on)
    family: str = "none"
    operator: str = ""
    instruction: str = ""                       # for the error writer
    violations: list[str] = field(default_factory=list)   # atom refs / transitions broken
    label: int = 1


@dataclass
class ScriptTurn:
    index: int
    gold: GoldTurn                              # the correct turn this one is based on
    customer: dict[str, Any]                    # intent for the customer writer (plus react_to_error)
    bot: BotSpec                                # what the bot says in the conversation
    variants: list[BotSpec] = field(default_factory=list)  # same context, other responses (counterfactual rows)
    phase: str = "pre"                          # pre | error | post | clean


@dataclass
class Script:
    conv: Conversation
    turns: list[ScriptTurn]
    error: dict[str, Any]


def _gold_spec(t: GoldTurn) -> BotSpec:
    return BotSpec("gold", t.state, t.state, t.step["boxes"], t.hrts, t.hrts)


def _atoms(t: GoldTurn, kinds: set[str], spoken_only: bool = False) -> list[dict[str, Any]]:
    out = [a for a in t.hrts.get("atoms", []) if a.get("kind") in kinds and not a.get("response_conditional")]
    if spoken_only:
        out = [a for a in out if a.get("field") and a.get("check") != "derived_from_fields"]
    return out


def _pos_pick(cands: list[int], position: str, rng: random.Random) -> int | None:
    if not cands:
        return None
    frac = {"early": 0.15, "middle": 0.5, "late": 0.85}.get(position, rng.random())
    i = min(len(cands) - 1, max(0, round(frac * (len(cands) - 1))))
    return cands[i]


class Injector:
    def __init__(self, sim: Simulator, conv: Conversation, rng: random.Random):
        self.sim, self.conv, self.rng = sim, conv, rng
        self.turns = conv.turns

    # ------------------------------------------------------------------ eligibility per family
    def _speech_turns(self) -> list[int]:
        return [t.index for t in self.turns if t.state != HANDOFF_STATE]

    def eligible(self, family: str) -> list[int]:
        T = self.turns
        sp = self._speech_turns()
        if family == "transition_skip":
            return [i for i in sp if any(u.index > i and u.state not in (T[i].state, HANDOFF_STATE) for u in T)]
        if family == "transition_repeat":
            return [i for i in sp if i > 0 and T[i - 1].state not in (T[i].state, HANDOFF_STATE)]
        if family == "transition_regress":
            return [i for i in sp if i > 1 and any(u.state not in (T[i].state, HANDOFF_STATE) for u in T[:i - 1])]
        if family in ("transition_wrong_branch", "wrapper_misread"):
            table = ALTERNATIVES if family == "transition_wrong_branch" else CUSTOMER_DRIVEN
            return [t.index for t in T if t.customer.get("addressed") in table and t.index > 0]
        if family == "premature_handoff":
            return [i for i in sp if i < len(T) - 1]
        if family == "missing_handoff":
            return [t.index for t in T if t.state == HANDOFF_STATE and "asks_human" in t.context["flags"]]
        if family == "content_missing":
            return [i for i in sp if _atoms(T[i], {"convey", "global_must"})]
        if family == "content_forbidden":
            return sp
        if family == "metadata_wrong":
            return [i for i in sp if _atoms(T[i], {"metadata", "convey"}, spoken_only=True)]
        if family == "order_violation":
            return [i for i in sp if any(a.get("in_order") for a in T[i].hrts.get("atoms", []))]
        if family == "guardrail_local":
            return [i for i in sp if _atoms(T[i], {"guard_must", "guard_must_not"})]
        if family == "guardrail_global":
            return [i for i in sp if _atoms(T[i], {"global_must", "global_must_not"})]
        if family == "context_ignored":
            return [t.index for t in T if t.state != HANDOFF_STATE and (
                t.customer.get("interjection", "none") != "none" or t.customer.get("correction", "none") != "none")]
        if family == "hallucination":
            return sp
        return []

    # ------------------------------------------------------------------ one erroneous response
    def make(self, family: str, k: int, subtle: bool) -> BotSpec | None:
        t = self.turns[k]
        g = _gold_spec(t)
        base = dict(gold_state=t.state, gold_hrts=t.hrts, family=family, label=0)
        T = self.turns
        if family == "transition_skip":
            later = [u for u in T if u.index > k and u.state not in (t.state, HANDOFF_STATE)]
            u = later[0] if subtle else later[-1]
            return BotSpec("error", u.state, boxes=u.step["boxes"], hrts=u.hrts, operator="skip_ahead",
                           instruction=f"Skip the current step: respond as the bot would at the later state {u.state}.",
                           violations=[f"transition {t.state} -> {u.state}"], **base)
        if family == "transition_repeat":
            u = T[k - 1]
            return BotSpec("error", u.state, boxes=u.step["boxes"], hrts=u.hrts, operator="repeat_previous",
                           instruction="Repeat the previous question as if the customer had not answered it.",
                           violations=[f"repeat {u.state}"], **base)
        if family == "transition_regress":
            earlier = [u for u in T[:k - 1] if u.state not in (t.state, HANDOFF_STATE)]
            u = self.rng.choice(earlier)
            return BotSpec("error", u.state, boxes=u.step["boxes"], hrts=u.hrts, operator="go_back",
                           instruction=f"Go back to an earlier step ({u.state}) that was already completed.",
                           violations=[f"transition {t.state} -> {u.state}"], **base)
        if family in ("transition_wrong_branch", "wrapper_misread"):
            table = ALTERNATIVES if family == "transition_wrong_branch" else CUSTOMER_DRIVEN
            addr = t.customer["addressed"]
            alts = [a for a in table[addr] if a != t.customer.get("outcome")]
            self.rng.shuffle(alts)
            for a in alts:
                cf = self.sim.counterfactual(k, addr, a, set(t.context["flags"]) - {"first_id_request", "later_id_request"})
                if cf and cf["state"] not in (t.state,) and (cf["state"] != HANDOFF_STATE or family == "wrapper_misread"):
                    what = ("acts as if the customer's answer meant " if family == "wrapper_misread"
                            else "follows the outcome ") + str(a)
                    return BotSpec("error", cf["state"], boxes=cf["boxes"], hrts=cf["hrts"],
                                   operator=f"{addr}={a}", instruction=f"The bot {what} instead of {t.customer.get('outcome')}.",
                                   violations=[f"{addr}: {t.customer.get('outcome')} -> {a}", f"transition -> {cf['state']}"],
                                   **base)
            return None
        if family == "premature_handoff":
            hand = {"User Identification": "USE escalate_to_fraudCCP", "Charge Verification": "USE escalate_to_fraudCCP"}[self.conv.aop]
            if self.rng.random() < 0.3 and self.conv.aop == "Charge Verification":
                hand = "USE end_call"
            h = self.sim._handoff_hrts(hand, set())
            return BotSpec("error", HANDOFF_STATE, boxes=[HANDOFF_STATE], hrts=h, operator=hand,
                           instruction="Tell the customer you are transferring them / ending the call, although the flow continues.",
                           violations=[f"transition {t.state} -> {HANDOFF_STATE}"], **base)
        if family == "missing_handoff":
            cf = self.sim.counterfactual(k, t.customer.get("addressed"), t.customer.get("outcome"),
                                         set(t.context["flags"]) - {"asks_human"})
            if not cf or cf["state"] == HANDOFF_STATE:
                return None
            return BotSpec("error", cf["state"], boxes=cf["boxes"], hrts=cf["hrts"], operator="ignore_human_request",
                           instruction="Carry on with the flow and do not transfer, although the customer asked for a person.",
                           violations=["global ROUTING_OVERRIDE: human requested -> SWITCH Connect to Agent"], **base)
        # content errors: the right state, said wrongly
        spec = BotSpec("error", t.state, boxes=g.boxes, hrts=t.hrts, **base)
        if family == "content_missing":
            cands = _atoms(t, {"convey", "global_must"})
            n = 1 if subtle else min(3, len(cands))
            drop = self.rng.sample(cands, n)
            spec.operator = "drop"
            spec.instruction = "Leave out: " + "; ".join(f"[{a['head']}] {a.get('description') or a['tail']}" for a in drop)
            spec.violations = [a["ref"] for a in drop]
            return spec
        if family == "content_forbidden":
            cands = _atoms(t, {"forbid", "global_must_not", "guard_must_not"})
            if cands:
                a = self.rng.choice(cands)
                spec.instruction = f"Also say this forbidden thing: [{a['head']}] {a.get('head_meaning') or a['tail']}"
                spec.violations = [a["ref"]]
            else:
                a = GENERIC_FORBIDDEN[self.conv.aop]
                spec.instruction = f"Also {a['tail']}."
                spec.violations = [a["ref"]]
            spec.operator = "say_forbidden"
            return spec
        if family == "metadata_wrong":
            cands = _atoms(t, {"metadata", "convey"}, spoken_only=True)
            a = self.rng.choice(cands)
            spec.operator = "wrong_value"
            spec.instruction = (f"Say a wrong value for {a['field']} ({a['head']}): plausible but different from the "
                                f"true one" + ("" if subtle else ", clearly different") + ".")
            spec.violations = [a["ref"]]
            return spec
        if family == "order_violation":
            spec.operator = "reorder"
            spec.instruction = "Give the required items in a different order than the AOP requires (e.g. ask the question before the details)."
            spec.violations = [a["ref"] for a in t.hrts.get("atoms", []) if a.get("in_order")][:3]
            return spec
        if family == "guardrail_local":
            a = self.rng.choice(_atoms(t, {"guard_must", "guard_must_not"}))
            verb = "Do this, which the AOP forbids" if a["kind"] == "guard_must_not" else "Fail to do this, which the AOP requires"
            spec.operator = "break_local_guardrail"
            spec.instruction = f"{verb}: [{a['head']}] {a.get('description') or a['tail']}"
            spec.violations = [a["ref"]]
            return spec
        if family == "guardrail_global":
            a = self.rng.choice(_atoms(t, {"global_must", "global_must_not"}))
            verb = "Do this, which an AOP-wide policy forbids" if a["kind"] == "global_must_not" else "Fail to do this, which an AOP-wide policy requires"
            spec.operator = "break_global_policy"
            spec.instruction = f"{verb}: [{a['head']}] {a.get('description') or a['tail']}"
            spec.violations = [a["ref"]]
            return spec
        if family == "context_ignored":
            inter = t.customer.get("interjection", "none")
            corr = t.customer.get("correction", "none")
            trig = [a for a in t.hrts.get("atoms", []) if str(a.get("box", "")).startswith(("UI_CLAUSE_0", "CV_CLAUSE_00"))
                    or a.get("branch") not in ("A",)]
            if inter != "none":
                spec.operator = f"ignore_{inter}"
                spec.instruction = (f"Ignore what the customer raised ({inter.replace('_', ' ')}): "
                                    + ("engage with it and answer it fully instead of redirecting." if inter in
                                       ("off_topic", "new_concern", "multi_demand", "ungrounded_question") and not subtle
                                       else "do not acknowledge or handle it the way the policy requires; just carry on."))
            else:
                spec.operator = f"ignore_{corr}"
                spec.instruction = ("Act on what the customer said BEFORE they corrected themselves, ignoring the correction."
                                    if corr == "self_correct_in_turn" else
                                    "Respond as if the customer had not changed their earlier answer, contradicting what they just said.")
            spec.violations = [a["ref"] for a in trig][:4] or [f"context:{inter}/{corr}"]
            return spec
        if family == "hallucination":
            spec.operator = "invent_fact"
            spec.instruction = ("Add one invented fact the tools and AOP never gave (e.g. a made-up detail about the "
                                "charge, the account, a fee, a timeline or a promise)." if subtle else
                                "Add several invented facts and a promise the AOP never makes.")
            spec.violations = ["grounding"]
            return spec
        return None

    # ------------------------------------------------------------------ the script
    def script(self, row: dict[str, str], side_negatives: int = 2) -> Script:
        T = self.turns
        fam = row.get("error_family", "none")
        err: dict[str, Any] = {"family": fam, "family_effective": fam, "turn": None}
        turns: list[ScriptTurn] = []
        if fam == "none" or not T:
            turns = [ScriptTurn(t.index, t, dict(t.customer), _gold_spec(t), phase="clean") for t in T]
            self._side(turns, side_negatives)
            return Script(self.conv, turns, err)
        subtle = row.get("error_subtlety") == "subtle"
        chain = [fam] + [f for f in ("content_missing", "content_forbidden", "hallucination") if f != fam]
        spec, k = None, None
        for f in chain:
            k = _pos_pick(self.eligible(f), row.get("error_position", "middle"), self.rng)
            if k is None:
                continue
            spec = self.make(f, k, subtle)
            if spec:
                err["family_effective"] = f
                break
        if spec is None or k is None:
            turns = [ScriptTurn(t.index, t, dict(t.customer), _gold_spec(t), phase="clean") for t in T]
            err["family_effective"] = "none"
            self._side(turns, side_negatives)
            return Script(self.conv, turns, err)
        err.update({"turn": k, "operator": spec.operator, "violations": spec.violations, "state_error":
                    err["family_effective"] in STATE_FAMILIES})
        for t in T[:k]:
            turns.append(ScriptTurn(t.index, t, dict(t.customer), _gold_spec(t), phase="pre"))
        gold_k = _gold_spec(T[k])
        turns.append(ScriptTurn(k, T[k], dict(T[k].customer), spec, variants=[gold_k], phase="error"))
        recovers = row.get("error_recovery") == "recovers"
        state_err = err["state_error"]
        nxt = k + 1
        for j in range(2):                              # the two turns after the error
            idx = k + 1 + j
            if state_err:
                g = T[k] if (j == 0 and recovers) else (T[nxt] if nxt < len(T) else None)
                if j == 1 and recovers:
                    g = T[k + 1] if k + 1 < len(T) else None
                if g is None or (T[k].state == HANDOFF_STATE and recovers and j == 1):
                    break
                cust = {"addressed": None, "react_to_error": True, "intent": {"act": "reacts_to_agent"},
                        "interjection": "asks_human" if spec.operator == "ignore_human_request" else "none",
                        "correction": "none"} if j == 0 else dict(g.customer)
                if recovers:
                    bot = _gold_spec(g)
                else:
                    bot = BotSpec("error", spec.state, gold_state=g.state, boxes=spec.boxes, hrts=spec.hrts,
                                  gold_hrts=g.hrts, family=spec.family, operator=spec.operator + "/persists",
                                  instruction=spec.instruction + " (still)", violations=spec.violations, label=0)
                turns.append(ScriptTurn(idx, g, cust, bot, phase="post"))
                if g.state == HANDOFF_STATE:
                    break
            else:
                if nxt >= len(T):
                    break
                g = T[nxt]
                cust = dict(g.customer)
                if j == 0:
                    cust["react_to_error"] = True
                if recovers or g.state == HANDOFF_STATE:
                    bot = _gold_spec(g)
                else:
                    bot = self._make_any(spec.family, nxt, subtle) or _gold_spec(g)
                turns.append(ScriptTurn(idx, g, cust, bot, phase="post"))
                nxt += 1
        self._side(turns, side_negatives)
        return Script(self.conv, turns, err)

    def _make_any(self, family: str, k: int, subtle: bool) -> BotSpec | None:
        for f in (family, "content_missing", "hallucination"):
            if k in self.eligible(f):
                v = self.make(f, k, subtle)
                if v:
                    return v
        return None

    def _side(self, turns: list[ScriptTurn], n: int) -> None:
        """Extra wrong responses for the same context at clean turns (contrastive negatives, not in the dialogue)."""
        cands = [st for st in turns if st.phase in ("pre", "clean") and st.bot.kind == "gold"
                 and st.gold.state != HANDOFF_STATE]
        self.rng.shuffle(cands)
        fams = ["content_missing", "content_forbidden", "metadata_wrong", "hallucination", "premature_handoff",
                "transition_repeat", "guardrail_global"]
        for st in cands[:n]:
            for f in self.rng.sample(fams, len(fams)):
                if st.index in self.eligible(f):
                    v = self.make(f, st.index, self.rng.random() < 0.5)
                    if v:
                        v.operator += "/side"
                        st.variants.append(v)
                        break


def build_script(sim: Simulator, conv: Conversation, row: dict[str, str], seed: int, side_negatives: int = 2) -> Script:
    return Injector(sim, conv, random.Random(seed)).script(row, side_negatives)
