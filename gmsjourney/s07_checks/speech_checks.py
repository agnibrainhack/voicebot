"""Checking what the bot said: every claim against its source, every act for completeness.

Parts of this file:
  1. ``UtteranceVerifier``: each claim supported, contradicted, unsupported or unverifiable (account facts against
     the session, policy figures against the exact register, stances against the store); turn findings: words
     contradict the label, a labelled act not said, required information missing, an act out of flow.
  2. The same check for any compact-graph pack (``PackModel``, ``GenericReader``, ``GenericChecks``).
  3. The public names of the speech check, with the overview of how one bot turn is checked.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# =====================================================================================================================
# PART 1 - Verifying a bot turn: every fact it states, and whether it says what it should.
#
# Input   for each bot turn: the acts the bot labelled (each may carry its own ``text``; a bot that
#         only returns text has no labels), the text the customer hears, and the conversation so far
#         (the tool log and the customer's frames). The checker for one conversation keeps the rest
#         (the transaction last talked about, the prompts the customer is answering).
# Output  a ``TurnReport`` per turn:
#           verdicts  one ``ClaimVerdict`` per claim the words make:
#                       supported     the words match where the truth lives
#                       contradicted  they state something else (the wrong fee, the wrong answer recorded)
#                       unsupported   nothing backs them (claimed recorded / done when nothing succeeded,
#                                     a charge no tool returned)
#                       unverifiable  nothing to check against (advisory, never a failure)
#           findings  what is wrong with the turn as a whole:
#                       text_contradicts_label  the words say something other than the act label
#                       label_not_said          an act is labelled but the words do not express it
#                       missing                 a required piece of information is not said (R6, R19, R26 ...)
#                       not_allowed_here        the act does not follow from the dialog-state graph (flow)
#                       disclosed               an internal figure, or one the project never volunteers (policy.py)
#                       volunteered             in a policy answer, any fact besides the one asked (fails the turn)
#                       unread                  a clause no check could read (advisory)
#           policy clauses (policy.py) add claims: policy_figure, promise and policy, with the same verdicts
#         ``TurnReport.ok`` is True when no claim is contradicted or unsupported and there is no
#         finding other than flow and the advisory kinds (flow is reported separately: ``flow_ok``).
#
# Where each claim's truth is read:
#   merchant / amount / date / age_days    the transactions the tool returned           (SessionFacts.items)
#   disposition said to be recorded        a successful record_charge_response for it    (R12, R13, R33)
#   disposition stated otherwise           what the customer said                        (SessionFacts.caller)
#   action said to be done                 a successful submit_card_action               (R17)
#   action offered                         the options get_card_action_options returned  (R15)
#   fee / delivery_days / limit            the GMS exact register (lookup_enm), with the trained
#                                          geometry's distance score_triple(...) vs the relation's cap
#   stance                                 the GMS stance check (u-space tension vs calibrated cuts)
#   opening                                the required opening statement
#   recorded_count                         the number of successful records
#
# Usage (one conversation):
#     verifier = UtteranceVerifier(store, SpeechContract.from_graph(store.kg))
#     checker = verifier.conversation(session_date="2026-03-16")
#     report = checker.check(acts, text, tool_log, customer_frames)   # after every bot turn
#     checker.reports                                                 # every turn's report
#     checker.summary()                                               # counts for the run's row
# =====================================================================================================================

from ..s01_packs.names import N
from ..s01_packs import names as roles
from .speech_reading import SpeechContract
from .speech_reading import ANY, Claim, ReadUtterance, UtteranceReader, words_to_digits
from .speech_reading import SessionFacts

FAILING = ("contradicted", "unsupported")
def _reask_after() -> set[str]:                            # after these the bot may re-ask the prompt that was pending
    return {a for a in ("ASK_CLARIFY", "ANSWER_KB") if a in N.state}


# ------------------------------------------------------------------ results
@dataclass(frozen=True)
class ClaimVerdict:
    """The verdict on one claim: was what the words state true, and where was the truth read?"""
    claim: Claim
    status: str                   # supported | contradicted | unsupported | unverifiable
    truth: Any = None             # the true value, where it was known
    source: str = ""              # where the truth was read
    why: str = ""                 # one plain sentence
    geometry: dict | None = None  # GMS: distance of the claimed value vs the relation's cap (policy figures)
    steps: tuple = ()             # how the truth was found: the rules tried, in order ({rule, hit, detail})
    compare: dict | None = None   # the comparison made: {op, said, truth, result}

    @property
    def failed(self) -> bool:
        """True for contradicted and unsupported (the customer was told something false or baseless)."""
        return self.status in FAILING

    @property
    def basis(self) -> str:
        """Where the truth of this claim was read, as one of a few plain kinds (for reports):

          tool result             what a tool returned (a charge's merchant, amount, date; the options offered)
          successful write        a write that truly succeeded ("recorded", "done", the summary's records)
          what the customer said  the customer's own answer, said back
          GMS exact register      a policy figure (fee, delivery days, limit), from the GMS register
          GMS stance check        a rule's stance (R27: not proof of fraud), from the GMS stance check
          policy text             the required opening statement, word for word
          nothing to check        an unverifiable mention (never a failure)
        """
        f, mode = self.claim.field, self.claim.mode
        if self.status == "unverifiable":
            return "nothing to check"
        src = self.source or ""
        if src == "the customer's policy question":
            return "what the customer asked"
        if src == "what the customer said":
            return "what the customer said"
        if src.endswith(" result options") or " result " in src:
            return "tool result"
        if src.startswith("the returned charge"):
            return "tool result"
        if src.startswith("successful "):              # where the check read it (the generic checks name it)
            return "successful write"
        if src.startswith("exact register"):
            return "GMS exact register"
        if " result " in src:
            return "tool result"
        if f in ("merchant", "amount", "date", "age_days"):
            return "tool result"
        if f == "disposition":
            return "successful write" if mode == "recorded" else "what the customer said"
        if f in ("action", "delivery"):
            return "tool result" if mode == "offered" else "successful write"
        if f in ("fee", "delivery_days", "limit", "policy_figure"):
            return "GMS exact register"
        if f == "promise":
            return "graph actions"
        if f == "policy":
            return "policy text"
        if f == "stance":
            return "GMS stance check"
        if f == "recorded_count":
            return "successful write"
        return "policy text" if f == "opening" else "other"

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view: the claim's fields plus status, basis, truth, source, why (and geometry if any)."""
        return {**self.claim.as_dict(), "status": self.status, "basis": self.basis,
                "truth": list(self.truth) if isinstance(self.truth, tuple) else self.truth,
                "source": self.source, "why": self.why, **({"geometry": self.geometry} if self.geometry else {}),
                **({"steps": [dict(x) for x in self.steps]} if self.steps else {}),
                **({"compare": _jsonable(self.compare)} if self.compare else {})}


def _jsonable(v: Any) -> Any:
    """Tuples and sets as lists, recursively (for the JSON trace)."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in (sorted(v, key=str) if isinstance(v, (set, frozenset)) else v)]
    return v


@dataclass(frozen=True)
class Finding:
    """A problem with a turn as a whole (not with one claim): unfaithful, incomplete, or out of flow."""
    kind: str                     # text_contradicts_label | label_not_said | missing | not_allowed_here
    act: str | None
    detail: str
    rule: str | None = None       # the policy rule or PDF section the finding enforces

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view."""
        return {"kind": self.kind, "act": self.act, "detail": self.detail, "rule": self.rule}


@dataclass
class TurnReport:
    """Everything the speech check found in one bot turn.

    turn      the bot turn's index (0 = the opening; n = the reply to customer turn n-1)
    text      what the customer heard
    acts      the act labels the bot gave (empty for a text-only bot)
    heard     the acts the words express
    verdicts  one per claim the words make
    findings  faithfulness, completeness and flow problems of the turn
    side      the failing verdicts and findings (by index) that belong to a side step: the answer to a policy
              question or a clarification (s06_bots/turns.py). They are reported at their turn; a journey's grade
              leaves them out (``core_ok``), unless the case is that policy question
    """
    turn: int
    text: str
    acts: list[str]
    heard: list[str]
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    side: dict = field(default_factory=lambda: {"verdicts": set(), "findings": set()})
    trace: dict = field(default_factory=dict)   # how the check went, step by step (speech_trace.py; the playground)

    @property
    def flow_findings(self) -> list[Finding]:
        """The ``not_allowed_here`` findings (flow is graded on its own, as ``c_flow``)."""
        return [f for f in self.findings if f.kind == "not_allowed_here"]

    advisory: frozenset = frozenset({"unread"})                  # finding kinds that never fail the turn

    @property
    def ok(self) -> bool:
        """Every claim supported (or unverifiable) and the words say what the labels and the policy require."""
        return not any(v.failed for v in self.verdicts) and not [
            f for f in self.findings if f.kind != "not_allowed_here" and f.kind not in self.advisory]

    @property
    def core_ok(self) -> bool:
        """``ok``, leaving out what failed in a side step (a policy answer, a clarification)."""
        sv, sf = self.side["verdicts"], self.side["findings"]
        return not any(v.failed for i, v in enumerate(self.verdicts) if i not in sv) and not [
            f for i, f in enumerate(self.findings)
            if i not in sf and f.kind != "not_allowed_here" and f.kind not in self.advisory]

    @property
    def flow_ok(self) -> bool:
        """Every act of the turn is a transition the dialog-state graph allows."""
        return not self.flow_findings

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view (stored per turn in ``extras["speech"]["turns"]``)."""
        return {"turn": self.turn, "text": self.text, "acts": self.acts, "heard": self.heard, "ok": self.ok,
                "core_ok": self.core_ok, "flow_ok": self.flow_ok,
                "claims": [{**v.as_dict(), "sentence": v.claim.sentence,
                            **({"side": True} if i in self.side["verdicts"] else {})}
                           for i, v in enumerate(self.verdicts)],
                "findings": [{**f.as_dict(), **({"side": True} if i in self.side["findings"] else {})}
                             for i, f in enumerate(self.findings)],
                "advisory": sorted(self.advisory),
                **({"trace": _jsonable(self.trace)} if self.trace else {})}


# ------------------------------------------------------------------ the verifier
class UtteranceVerifier:
    """Checks utterances against a GMS store and the speech contract read from its policy graph.

    store     the knowlytix GMS store (needs lookup_enm, score_triple, tau, stance_check)
    contract  SpeechContract.from_graph(store.kg)
    """

    def __init__(self, store, contract: SpeechContract, corroborate: bool = True):
        self.store, self.c = store, contract
        self.reader = UtteranceReader(contract)
        self.generic = None
        if getattr(contract, "model", None) is not None:      # a compiled pack: fields from its graph
            self.reader = GenericReader(contract, contract.model)
            self.generic = GenericChecks(self, contract.model)
        self._geo: dict[tuple, dict | None] = {}
        # what the readers leave: figures in other words, promises, policy-sounding statements (policy.py)
        from .speech_policy import PolicyChecker, settings as _policy_settings
        kg = getattr(store, "kg", None)
        self.policy = PolicyChecker(contract, kg, getattr(kg, "graph", None)) if kg is not None else None
        self.advisory = frozenset(_policy_settings()["advisory"])
        if self.policy is not None and self.policy.cfg["judge"] == "llm":
            from .speech_policy import LLMPolicyJudge
            self.policy.judge = LLMPolicyJudge(self.policy.cfg.get("llm_model"))
        # knowlytix's own claim checker (AnswerVerifier, geometric mode: no LLM) as an independent second
        # reader of policy figures; it reports the store facts a sentence states correctly. It reads with the
        # store's tuned encoders (v from the embed loop, u from the contradiction SFT), the same ones the
        # geometry readings and the stance check use, so every GMS reading of a sentence is like for like.
        self._kx_verifier = None
        self._kx_seen: dict[str, list[tuple[str, str, str]]] = {}
        raw = getattr(store, "kx_raw", None)
        if corroborate and raw is not None:
            try:
                from knowlytix.knowledge.rag.verify import AnswerVerifier
                self._kx_verifier = AnswerVerifier(raw, None, mode="geometric", v_encoder=getattr(store, "v_ft", None),
                                                   u_encoder=getattr(store, "u_ft", None))
            except Exception:                                   # noqa: BLE001 - corroboration is optional
                self._kx_verifier = None

    def conversation(self, session_date: str | None = None) -> "ConversationChecker":
        """A checker for one conversation (it remembers what was talked about across turns)."""
        return ConversationChecker(self, session_date)

    # ---------------------------------------------------------------- one turn
    def check_turn(self, acts: list[dict[str, Any]], text: str, facts: SessionFacts, *, turn: int = 0,
                   last_item: str | None = None, pending_prompts: set[str] | None = None,
                   customer_act: str | None = None, question: Any = None,
                   presented: set[str] | None = None, held: dict | None = None) -> TurnReport:
        """Check one bot turn. ``acts`` are the labels (each may carry its own ``text``); with no labels the
        whole text is read and the acts it expresses stand in for them.

        last_item        the transaction talked about last (for sentences that name no merchant)
        pending_prompts  the prompts the customer was answering (for flow)
        customer_act     what the customer's last turn meant (for flow)
        question         the customer's policy question, when this turn answers one (ASK_KB frame)
        presented        transactions already presented: re-asking one need not repeat its facts (pdf.3:
                         do not ask the customer to repeat information without a reason)
        held             the oracle's acts held in the words (s06_bots/oracle_says.hold): each sentence is read
                         under the oracle's act it belongs to, the acts are the ones the turn is graded as having
                         said, and whether each act is in the words is the hold's verdict (no label of the agent's)"""
        labelled = [a for a in acts if a.get("act")]
        per_act = labelled and all("text" in a for a in labelled)
        segments = [(a, a.get("text") or "") for a in labelled] if per_act else [(None, text or "")]
        if held is not None:
            labelled = [a for a in held["acts"] if a.get("act")]
            per_act = True
            segments = [(lab, words) for lab, words in held["segments"]]
        report = TurnReport(turn, text or "", [a.get("act") for a in labelled], [])
        report.advisory = self.advisory
        presented = presented or set()
        asked = self.policy.asked_subjects(question) if self.policy is not None else set()
        segs = []
        for label, seg_text in segments:
            about = (label or {}).get("txn") or last_item
            if about is not None and not isinstance(about, (str, int, float)):   # a bot's malformed reference:
                about = json.dumps(about, sort_keys=True, default=str)           # matches no item, never crashes
            answering = question if (label is None or ("ANSWER_KB" in N.state and label.get("act") == N.state.ANSWER_KB)) \
                else None
            read = self.reader.read(seg_text, facts, about_item=about, question=answering)
            seg = {"act": (label or {}).get("act"), "label": {k: v for k, v in (label or {}).items() if k != "text"},
                   "text": seg_text, "about": about, "answering": bool(answering), "sentences": _explain(self.reader, read, facts),
                   "opening": any(c.field == "opening" for c in read.claims), "heard": list(read.heard),
                   "claims": [], "policy": [], "faithfulness": [], "completeness": []}
            segs.append(seg)
            report.heard += [h for h in read.heard if h not in report.heard]
            mine = []
            for cl in read.claims:
                mine.append(_with_steps(self._corroborated(self.verify_claim(cl, facts, label))))
            if self.policy is not None:
                act = (label or {}).get("act")
                for cv in self.policy.check(read):
                    if cv.outcome == "unread":
                        report.findings.append(Finding("unread", act, f"not read as policy: {cv.clause!r}"))
                        seg["policy"].append({"clause": cv.clause, "outcome": "unread"})
                        continue
                    v = ClaimVerdict(cv.as_claim(), cv.status, cv.truth, cv.cites or "", cv.why,
                                     steps=({"rule": "a policy statement", "hit": True,
                                             "detail": f"read by the policy reader; checked against {cv.cites or 'the policy graph'}"},))
                    mine.append(v)
                    seg["policy"].append({"clause": cv.clause, "outcome": cv.status})
                    why = self.policy.never(cv.subjects, cv.internal) if not cv.failed_for_disclosure(asked) else None
                    if why:
                        report.findings.append(Finding("disclosed", act, f"{cv.clause!r}: {why}"))
                if answering is not None:                  # a policy answer: what else did it tell?
                    for v in mine:
                        if v.status in ("supported", "contradicted") and self.policy.volunteered(v.claim, answering):
                            report.findings.append(Finding("volunteered", act, f"asked about {', '.join(sorted(asked))}; "
                                                           f"also told {v.claim.field} of {v.claim.about}: {v.claim.words!r}"))
                for v in mine:                             # a reader's figure about a never-volunteer subject
                    if v.claim.field not in ("policy_figure", "promise", "policy") and v.claim.about not in asked:
                        why = self.policy.never((v.claim.about,), False) if v.claim.about else None
                        if why:
                            report.findings.append(Finding("disclosed", act, f"{v.claim.words!r}: {why}"))
            seg["claims"] = list(range(len(report.verdicts), len(report.verdicts) + len(mine)))
            report.verdicts += mine
            if label is not None and held is not None:     # the hold already decided whether the act is in the words
                row = next((r for r in held["says"] if r["act"] == label.get("act")), None)
                if row is not None:
                    seg["faithfulness"].append({"check": f"the words express {row['act']} (the oracle's act)",
                                                "ok": row["status"] == "held", "detail": row["detail"]})
                report.findings += self._completeness(label["act"], label, read, facts, about, presented,
                                                      seg["completeness"])
            elif label is not None:
                report.findings += self._faithfulness(label, read, seg["faithfulness"])
                report.findings += self._completeness(label["act"], label, read, facts, about, presented,
                                                      seg["completeness"])
            elif not labelled:
                for act in read.heard:
                    report.findings += self._completeness(act, {}, read, facts, about, presented, seg["completeness"])
        names = report.acts or report.heard
        flow: list[dict] = []
        report.findings += self._flow(names, pending_prompts or set(), customer_act, first=turn == 0, log=flow)
        from .speech_trace import readers
        report.trace = {"split": "oracle" if held is not None else "per_act" if per_act else "whole",
                        "readers": readers(self.reader),
                        **({"says": {k: v for k, v in held.items() if k != "segments"}} if held is not None else {}),
                        "segments": segs, "flow": flow, "flow_acts": list(names),
                        "pending": sorted(pending_prompts or ()), "customer_act": customer_act}
        return report

    # ---------------------------------------------------------------- one claim
    def verify_claim(self, cl: Claim, facts: SessionFacts, label: dict[str, Any] | None = None) -> ClaimVerdict:
        """Grade one claim against where its truth lives (the table in the module docstring).

        cl     the claim, as the reader produced it
        facts  the session so far
        label  the act label the claim's words belong to, if the bot gave one (a fee with no option named
               is then about the label's delivery option)
        """
        if self.generic is not None:
            return self.generic.verify_claim(cl, facts, label)
        f = cl.field
        if f in ("merchant", "amount", "date", "age_days"):
            return self._item_claim(cl, facts)
        if f == "disposition":
            if cl.mode == "recorded":
                w = facts.write_for(roles.RECORD, cl.about)
                if w is None:
                    return ClaimVerdict(cl, "unsupported", None, roles.RECORD,
                                        f"said to be recorded, but no record for {cl.about} succeeded (R13)")
                truth = w["args"].get("disposition")
                return self._compare(cl, truth, f"successful record_charge_response({cl.about})")
            truth = facts.caller.get("disposition")
            if truth is None:
                return ClaimVerdict(cl, "unverifiable", None, "", "the customer has not answered yet")
            return self._compare(cl, truth, "what the customer said")
        if f == "action":
            if cl.mode == "done":
                w = facts.write_for(roles.ACTION, None)
                if w is None:
                    return ClaimVerdict(cl, "unsupported", None, roles.ACTION,
                                        "said to be done, but no card action succeeded (R17)")
                return self._compare(cl, w["args"].get("action"), "successful submit_card_action")
            if cl.mode == "offered":
                if facts.options is None:
                    return ClaimVerdict(cl, "unsupported", None, roles.OPTIONS,
                                        "offered before any card option was returned (R15)")
                ok = cl.value in facts.options
                return ClaimVerdict(cl, "supported" if ok else "contradicted", facts.options,
                                    "get_card_action_options result",
                                    "" if ok else "not among the options the tool returned (R15)")
            return ClaimVerdict(cl, "unverifiable", None, "", "mentioned, not asserted")
        if f == "delivery":
            if cl.mode == "done":
                w = facts.write_for(roles.ACTION, None)
                if w is None:
                    return ClaimVerdict(cl, "unverifiable", None, "", "no card action has succeeded")
                return self._compare(cl, w["args"].get("delivery"), "successful submit_card_action")
            return ClaimVerdict(cl, "unverifiable", None, "", "a delivery option named")
        if f in ("fee", "delivery_days", "limit"):
            return self._policy_figure(cl, facts, label)
        if f == "stance":
            sp = self.c.stances[cl.about]
            chk = self.store.stance_check(*sp.fact, cl.value)
            status = {"supported": "supported", "contradicted": "contradicted"}.get(chk.get("verdict"), "unverifiable")
            return ClaimVerdict(cl, status, sp.stance, f"GMS stance check of {cl.about} {sp.fact}",
                                f"tension {chk.get('energy')}", {"tension": chk.get("energy")})
        if f == "opening":
            return ClaimVerdict(cl, "supported", self.c.opening, "required opening statement")
        if f == "recorded_count":
            n = len(facts.writes.get(roles.RECORD, []))
            return ClaimVerdict(cl, "supported" if cl.value == n else "contradicted", n,
                                "successful record_charge_response calls")
        return ClaimVerdict(cl, "unverifiable", None, "", f"no check for {f}")

    def _compare(self, cl: Claim, truth: Any, source: str) -> ClaimVerdict:
        """supported when the claimed value equals ``truth`` (case-insensitive), else contradicted."""
        ok = str(cl.value).lower() == str(truth).lower()
        return ClaimVerdict(cl, "supported" if ok else "contradicted", truth, source,
                            "" if ok else f"says {cl.value}, the truth is {truth}",
                            compare={"op": "equals, ignoring case", "said": str(cl.value).lower(),
                                     "truth": str(truth).lower(), "result": ok})

    def _item_claim(self, cl: Claim, facts: SessionFacts) -> ClaimVerdict:
        """An account fact must be what the tool returned for that transaction (R7, R24)."""
        if not facts.items:
            return ClaimVerdict(cl, "unsupported", None, roles.ITEMS,
                                "an account fact, but no transaction was returned (R7)")
        same = lambda a, b: _num_eq(a, b) if cl.field in ("amount", "age_days") else str(a).lower() == str(b).lower()  # noqa: E731
        if cl.about in facts.items:
            truth = facts.items[cl.about].get(cl.field)
            if truth is None:
                return ClaimVerdict(cl, "unverifiable", None, "", f"the tool returned no {cl.field}")
            if same(cl.value, truth):
                return ClaimVerdict(cl, "supported", truth, f"tool result: {cl.about}.{cl.field}")
            other = next((t for t, it in facts.items.items() if same(cl.value, it.get(cl.field))), None)
            why = f"{cl.about}'s {cl.field} is {truth}" + (f"; {cl.value} belongs to {other}" if other else "")
            return ClaimVerdict(cl, "contradicted", truth, f"tool result: {cl.about}.{cl.field}", why)
        other = next((t for t, it in facts.items.items() if same(cl.value, it.get(cl.field))), None)
        if other:
            return ClaimVerdict(cl, "supported", cl.value, f"tool result: {other}.{cl.field}")
        return ClaimVerdict(cl, "unsupported", None, roles.ITEMS,
                            f"no returned transaction has {cl.field} {cl.value} (R7)")

    def _policy_figure(self, cl: Claim, facts: SessionFacts, label: dict[str, Any] | None) -> ClaimVerdict:
        """A policy figure must be the exact register's value (lookup_enm). The GMS geometry is reported
        alongside: the claimed value's distance from the fact vs the relation's learned cap."""
        if cl.field == "limit":
            truth = self.store.lookup_enm("limit", cl.about)
            geo = self._geometry(cl.about, "has_value", cl.value)
            ok = _num_eq(cl.value, truth)
            return ClaimVerdict(cl, "supported" if ok else "contradicted", truth, f"exact register limit/{cl.about}",
                                "" if ok else f"{cl.about} is {truth}", geo)
        option = None if cl.about == ANY else cl.about or (label or {}).get("delivery") or facts.caller.get("delivery")
        options = self.c.value_sets.get("delivery", [])
        if cl.field == "fee":
            if option is None:
                hits = [o for o in options if _num_eq(cl.value, self.store.lookup_enm("delivery_fee_usd", o))]
                if hits:
                    return ClaimVerdict(cl, "supported", cl.value, f"exact register delivery_fee_usd/{hits[0]}")
                return ClaimVerdict(cl, "unsupported", None, "exact register delivery_fee_usd",
                                    "no delivery option has this fee")
            truth = self.store.lookup_enm("delivery_fee_usd", option)
            ok = _num_eq(cl.value, truth)
            return ClaimVerdict(cl, "supported" if ok else "contradicted", truth,
                                f"exact register delivery_fee_usd/{option}", "" if ok else f"{option} costs ${truth}",
                                self._geometry(option, "has_fee_usd", cl.value))
        lo, hi = cl.value
        rng = {o: (self.store.lookup_enm("delivery_days_min", o), self.store.lookup_enm("delivery_days_max", o))
               for o in options}
        if option is None:
            hits = [o for o, (a, b) in rng.items() if a <= lo and hi <= b]
            if hits:
                return ClaimVerdict(cl, "supported", rng[hits[0]], f"exact register delivery_days/{hits[0]}")
            return ClaimVerdict(cl, "unsupported", None, "exact register delivery_days",
                                "no delivery option takes this long")
        a, b = rng[option]
        ok = a <= lo and hi <= b
        return ClaimVerdict(cl, "supported" if ok else "contradicted", (a, b), f"exact register delivery_days/{option}",
                            "" if ok else f"{option} takes {a if a == b else f'{a}-{b}'} business days",
                            self._geometry(option, "has_business_days_max", hi))

    def _corroborated(self, v: ClaimVerdict) -> ClaimVerdict:
        """Add knowlytix's own reading of a fee or delivery-time sentence: does its AnswerVerifier find the
        same fact, with the same value, supported in the store? (It finds the store's facts that a sentence
        states correctly, so a wrong figure is simply not found.)

        AnswerVerifier reads one sentence, without the dialog. Only a sentence it can read on its own is
        passed: one that names the option and gives the figure as a number. "No, it's $0." (the option is in
        the customer's question) and "standard delivery is free" (no number) are left unchecked by it, so
        the agreement count measures knowlytix where it had something to read."""
        if self._kx_verifier is None or v.claim.field not in ("fee", "delivery_days") or v.claim.about in (None, ANY):
            return v
        sent = v.claim.sentence
        names_option = any(m.value == v.claim.about
                           for m in self.reader._values_named(words_to_digits(sent).lower(), "delivery"))
        if not names_option or not any(ch.isdigit() for ch in words_to_digits(v.claim.words)):
            return v
        if sent not in self._kx_seen:
            try:
                rep = self._kx_verifier.verify(sent)
                self._kx_seen[sent] = [(x.triple.head, x.triple.relation, str(x.triple.tail)) for x in rep.verdicts
                                       if x.status == "supported"]
            except Exception:                                   # noqa: BLE001
                self._kx_seen[sent] = []
        rels = ("has_fee_usd",) if v.claim.field == "fee" else ("has_business_days_min", "has_business_days_max")
        vals = [v.claim.value] if v.claim.field == "fee" else list(v.claim.value)
        found = any(h == v.claim.about and r in rels and _num_eq(t, val)
                    for h, r, t in self._kx_seen[sent] for val in vals)
        geo = dict(v.geometry or {})
        geo["knowlytix"] = "confirmed" if found else "not found"
        from dataclasses import replace
        return replace(v, geometry=geo)

    def _geometry(self, head: str, relation: str, value: Any) -> dict | None:
        """The GMS reading of a claimed figure: geodesic distance of (head, relation, value) from the fact,
        against the relation's learned cap radius. Inside the cap = grounded (book Ch. 3.4, 9)."""
        key = (head, relation, str(value))
        if key not in self._geo:
            try:
                d = float(self.store.score_triple(head, relation, str(_num(value))))
                cap = float(self.store.tau(relation))
                self._geo[key] = {"distance": round(d, 4), "cap": round(cap, 4), "inside_cap": d <= cap}
            except Exception:                                   # noqa: BLE001 - geometry is reported, never decides
                self._geo[key] = None
        return self._geo[key]

    # ---------------------------------------------------------------- the turn as a whole
    def _faithfulness(self, label: dict[str, Any], read: ReadUtterance, log: list | None = None) -> list[Finding]:
        """Do the words say what the label claims?"""
        if self.generic is not None:
            return self.generic.faithfulness(label, read, log)
        act = label["act"]
        spec = self.c.acts.get(act)
        out = []
        if spec and (spec.cues or spec.says) and act not in read.heard and read.text.strip():
            out.append(Finding("label_not_said", act, f"labelled {act} but the words do not express it: "
                                                     f"{read.text!r}"))
        for fld in ("disposition", "action", "delivery", "fee"):
            want = label.get(fld)
            if want in (None, ""):
                continue
            said = {str(c.value) for c in read.of(fld)}
            if fld == "fee":
                said = {str(_num(c.value)) for c in read.of(fld)}
                want = _num(want)
            if said and str(want) not in said:
                out.append(Finding("text_contradicts_label", act, f"label {fld}={want}, the words say "
                                                                  f"{', '.join(sorted(said))}"))
        txn = label.get("txn")
        named = {c.about for c in read.of("merchant")}
        if txn and named and txn not in named:
            out.append(Finding("text_contradicts_label", act, f"labelled about {txn}, the words name "
                                                              f"{', '.join(sorted(named))}"))
        for tid, disp in (label.get("recorded") or {}).items():
            said = {c.value for c in read.of("disposition") if c.about == tid}
            if said and disp not in said:
                out.append(Finding("text_contradicts_label", act, f"summary label {tid}={disp}, the words say "
                                                                  f"{', '.join(sorted(said))}"))
        return out

    def _completeness(self, act: str, label: dict[str, Any], read: ReadUtterance, facts: SessionFacts,
                      about: str | None, presented: set[str] = frozenset(), log: list | None = None) -> list[Finding]:
        """Does the act say everything the policy requires it to say (when there is something to say)?
        A transaction's facts are required the first time it is presented, not when it is re-asked."""
        if self.generic is not None:
            return self.generic.completeness(act, label, read, facts, log)
        spec = self.c.acts.get(act)
        if spec is None:
            return []
        out = []
        for fs in spec.says:
            if not fs.required:
                continue
            kind = fs.source.split(".")[0]
            if kind == "writes":
                out += self._summary_complete(act, fs, read, facts)
                continue
            if kind == "item" and (about is None or about not in facts.items or about in presented):
                continue                                  # nothing to say about, or already presented
            if kind == "caller" and facts.resolve(fs.source).value is None and not label.get(fs.field):
                continue
            if not read.of(fs.field):
                out.append(Finding("missing", act, f"{act} must say the {fs.field} ({fs.source})", spec.source))
        return out

    def _summary_complete(self, act: str, fs, read: ReadUtterance, facts: SessionFacts) -> list[Finding]:
        """A summary must name every response that was truly recorded and every action that truly completed
        (pdf.2F, R19)."""
        tool = fs.source.split(".", 1)[1]
        out = []
        if tool == roles.RECORD:
            named = {c.about for c in read.of("merchant")} | {c.about for c in read.of("disposition")}
            for w in facts.writes.get(tool, []):
                tid = w["args"].get("transaction_id")
                if tid not in named:
                    out.append(Finding("missing", act, f"{act} does not mention the recorded response for {tid}",
                                       spec_source(self.c, act)))
        elif tool == roles.ACTION:
            said = {c.value for c in read.of("action")}
            for w in facts.writes.get(tool, []):
                if w["args"].get("action") not in said:
                    out.append(Finding("missing", act, f"{act} does not mention the completed action "
                                                       f"{w['args'].get('action')}", spec_source(self.c, act)))
        return out

    def _flow(self, names: list[str], pending: set[str], customer_act: str | None, first: bool,
              log: list | None = None) -> list[Finding]:
        """Is each act where the dialog-state graph allows it? The first act of a turn must follow from a
        prompt the customer was answering, on what they said (or be allowed from anywhere: a person on
        request, a policy answer, a clarification, the turn limit); each next act must follow automatically
        from the one before (or re-ask the pending prompt after a clarification or a policy answer)."""
        log = log if log is not None else []
        if not names:
            log.append({"check": "nothing to place in the dialog", "ok": True, "detail": "no act labelled or heard"})
            return []
        out = []
        c = self.c
        if first:
            ok = not ("OPENING" in N.state and N.state.OPENING in c.acts and names[0] != N.state.OPENING)
            log.append({"check": f"{names[0]} starts the call", "ok": ok,
                        "detail": "the call must start with the opening (R1)" if not ok else "the first turn opens the call"})
            if not ok:
                out.append(Finding("not_allowed_here", names[0], "the call must start with the opening (R1)", "R1"))
        else:
            a0 = names[0]
            via = next((f"{p} goes to {a0} when the customer means {customer_act}" for p in sorted(pending)
                        if a0 in c.next_states(p, customer_act)), None) \
                or (f"{a0} is allowed from anywhere on {customer_act}" if a0 in c.global_acts.get(customer_act or "", set()) else None) \
                or (f"{a0} is allowed from anywhere" if a0 in c.global_acts["any"] else None) \
                or (f"{a0} re-asks the prompt the customer was answering" if a0 in pending else None)
            log.append({"check": f"{a0} follows what the customer said", "ok": bool(via),
                        "detail": via or f"from {', '.join(sorted(pending)) or 'no pending prompt'} on {customer_act}, "
                                         f"the graph does not go to {a0}"})
            if not via:
                out.append(Finding("not_allowed_here", a0, f"{a0} does not follow from {sorted(pending)} on "
                                                           f"{customer_act}"))
        for a, b in zip(names, names[1:]):
            auto = b in c.next_states(a, "auto")
            reask = a in _reask_after() and (b in pending or b in c.prompt_states)
            ok = auto or reask
            log.append({"check": f"{b} follows {a}", "ok": ok,
                        "detail": f"{a} goes on to {b} by itself (an automatic step in the graph)" if auto else
                        f"after {a} the agent may re-ask a prompt" if reask else f"the graph has no step from {a} to {b}"})
            if not ok:
                out.append(Finding("not_allowed_here", b, f"{b} does not follow {a}"))
        return out


def _explain(reader, read: ReadUtterance, facts) -> list[dict]:
    """Each sentence as the reader saw it (speech_trace.py); a failure here never fails the check."""
    from .speech_trace import explain_sentence
    out = []
    for s in read.sentences:
        try:
            out.append(explain_sentence(reader, s, facts))
        except Exception as e:                                  # noqa: BLE001 - the trace is for reading only
            out.append({"text": s, "error": f"{type(e).__name__}: {e}"})
    return out


def _with_steps(v: ClaimVerdict) -> ClaimVerdict:
    """A verdict from a check that keeps no steps of its own: one step, where its truth was read."""
    if v.steps:
        return v
    from dataclasses import replace
    return replace(v, steps=({"rule": v.basis, "hit": True, "detail": v.source or v.why},))


# ------------------------------------------------------------------ one conversation
class ConversationChecker:
    """Checks every bot turn of one conversation, remembering what was talked about.

    check(acts, text, tool_log, customer_frames) -> TurnReport   after each bot turn
    reports                                                        every turn's report so far
    summary()                                                      counts for the run's result row
    """

    def __init__(self, verifier: UtteranceVerifier, session_date: str | None = None):
        self.v, self.session_date = verifier, session_date
        self.reports: list[TurnReport] = []
        self.last_item: str | None = None
        self.pending: set[str] = set()
        self.presented: set[str] = set()          # transactions whose facts were stated when presented

    def hold(self, expected: list[dict] | None, text: str, tool_log: list[dict[str, Any]], customer_frames: list[Any],
             question: Any = None, answer_act: str | None = None, ended: bool | None = None) -> dict[str, Any]:
        """The oracle's acts for this turn held in the agent's words (s06_bots/oracle_says.hold); ``ended``: whether
        the agent ended the call with this reply."""
        from ..s06_bots.oracle_says import hold
        facts = SessionFacts.from_run(tool_log, customer_frames, self.session_date, self.v.c.success_values)
        return hold(self.v, expected, text, facts, question=question, store=self.v.store, answer_act=answer_act,
                    ended=ended)

    def check(self, acts: list[dict[str, Any]], text: str, tool_log: list[dict[str, Any]],
              customer_frames: list[Any], held: dict | None = None) -> TurnReport:
        """Check one bot turn, then remember what it presented and asked.

        acts             the bot's act labels for the turn ([] for a text-only bot)
        text             what the customer hears
        tool_log         the backend's tool log so far (everything the bot did before speaking)
        customer_frames  the customer's frames so far (the last one is what this turn answers)
        held             the oracle's acts held in the words (``hold``): the turn is read and graded by them
        """
        facts = SessionFacts.from_run(tool_log, customer_frames, self.session_date, self.v.c.success_values)
        last = customer_frames[-1] if customer_frames else None
        customer_act = getattr(last, "act", None)
        rep = self.v.check_turn(acts, text, facts, turn=len(self.reports), last_item=self.last_item,
                                pending_prompts=self.pending, customer_act=customer_act,
                                question=last if customer_act == N.act.ASK_KB else None, presented=self.presented,
                                held=held)
        if held is not None:
            acts = list(held["acts"])
            self._mark_side(rep, [{**lab, "text": words} for lab, words in held["segments"] if lab])
        else:
            self._mark_side(rep, acts)
        try:
            from .speech_trace import facts_snapshot
            rep.trace["facts"] = facts_snapshot(facts, list(customer_frames))
        except Exception as e:                                  # noqa: BLE001 - the trace is for reading only
            rep.trace["facts"] = {"error": f"{type(e).__name__}: {e}"}
        self.reports.append(rep)
        for a in acts:                             # a presentation, said or not, counts once
            if a.get("txn") and "PRESENT_TXN" in N.state and a.get("act") == N.state.PRESENT_TXN:
                self.presented.add(a["txn"])
        named = [a.get("txn") for a in acts if a.get("txn")] + \
                [v.claim.about for v in rep.verdicts if v.claim.field in ("merchant", "item")]
        if named:
            self.last_item = named[-1]
        names = rep.acts or rep.heard
        prompts = {a for a in names if a in self.v.c.prompt_states}
        if prompts:
            self.pending = prompts
        return rep

    @staticmethod
    def _mark_side(rep: "TurnReport", acts: list[dict[str, Any]]) -> None:
        """Mark the failures that belong to a side step: a finding about a side act, a failing claim whose sentence
        is in the words of a side act (a policy answer, a clarification)."""
        from ..s06_bots.turns import side_acts
        side = side_acts()
        norm = lambda t: " ".join(str(t or "").lower().split())          # noqa: E731
        texts = [norm(a.get("text")) for a in acts if a.get("act") in side and a.get("text")]
        for i, v in enumerate(rep.verdicts):
            sent = norm(getattr(v.claim, "sentence", ""))
            if v.failed and sent and any(sent in t for t in texts):
                rep.side["verdicts"].add(i)
        for i, f in enumerate(rep.findings):
            if f.act in side:
                rep.side["findings"].add(i)

    def summary(self) -> dict[str, Any]:
        """Counts for the run's result row: claims by verdict, unfaithful / missing / flow findings, how many
        policy figures knowlytix's AnswerVerifier also read (and agreed on), and the grades ``c_speech`` (every turn
        ok, leaving out the failures of side steps: a policy answer, a clarification), ``c_speech_strict`` (every
        turn ok) and ``c_flow`` (every turn in flow). The counts leave the side failures out; ``speech_side``
        counts them."""
        vs = [v for r in self.reports for i, v in enumerate(r.verdicts) if i not in r.side["verdicts"]]
        fs = [f for r in self.reports for i, f in enumerate(r.findings) if i not in r.side["findings"]]
        side_n = sum(1 for r in self.reports for i, v in enumerate(r.verdicts) if i in r.side["verdicts"] and v.failed) + \
            sum(1 for r in self.reports for i, f in enumerate(r.findings)
                if i in r.side["findings"] and f.kind not in r.advisory)
        count = lambda s: sum(v.status == s for v in vs)  # noqa: E731
        kinds = lambda k: sum(f.kind == k for f in fs)    # noqa: E731
        kx = [v for v in vs if (v.geometry or {}).get("knowlytix")]
        agree = sum((v.status == "supported") == (v.geometry["knowlytix"] == "confirmed") for v in kx)
        adv = self.v.advisory
        policy = [v for v in vs if v.claim.field in ("policy_figure", "promise", "policy")]
        return {"speech_claims": len(vs), "speech_supported": count("supported"),
                "speech_policy_checked": len(policy), "speech_policy_failed": sum(v.failed for v in policy),
                "speech_unread": kinds("unread"), "speech_volunteered": kinds("volunteered"),
                "speech_volunteered_blocking": 0 if "volunteered" in adv else kinds("volunteered"),
                "speech_disclosed": kinds("disclosed"),
                "speech_kx_checked": len(kx), "speech_kx_agree": agree,
                "speech_contradicted": count("contradicted"), "speech_unsupported": count("unsupported"),
                "speech_unverifiable": count("unverifiable"), "speech_unfaithful": kinds("text_contradicts_label")
                + kinds("label_not_said"), "speech_missing": kinds("missing"),
                "speech_flow": kinds("not_allowed_here"),
                "speech_side": side_n,
                "c_speech": int(all(r.core_ok for r in self.reports)),
                "c_speech_strict": int(all(r.ok for r in self.reports)),
                "c_flow": int(all(r.flow_ok for r in self.reports))}

    def problems(self, limit: int = 12) -> list[dict[str, Any]]:
        """Every failing claim and finding, with the turn and words, for failure reports."""
        out = []
        for r in self.reports:
            where = turn_where(r.turn)
            for i, v in enumerate(r.verdicts):
                if v.failed:
                    out.append({**({"side": True} if i in r.side["verdicts"] else {}),
                                "turn": r.turn, "where": where, "kind": v.status, "field": v.claim.field,
                                "words": v.claim.words,
                                "said": v.claim.value if not isinstance(v.claim.value, tuple) else list(v.claim.value),
                                "truth": v.truth if not isinstance(v.truth, tuple) else list(v.truth),
                                "why": v.why, "sentence": v.claim.sentence})
            for i, f in enumerate(r.findings):
                if f.kind in r.advisory:                  # advisory (unread, volunteered): counted, not a problem
                    continue
                out.append({**({"side": True} if i in r.side["findings"] else {}),
                            "turn": r.turn, "where": where, "kind": f.kind, "act": f.act, "why": f.detail,
                            "rule": f.rule})
        return out[:limit]


# ------------------------------------------------------------------ helpers
def turn_where(turn: int) -> str:
    """A bot turn in the turn-by-turn table's numbering: bot turn 0 is the opening, bot turn n answers
    customer turn n-1."""
    return "the opening" if turn == 0 else f"the reply to customer turn {turn - 1}"


def spec_source(contract: SpeechContract, act: str) -> str | None:
    """The PDF section that says what ``act`` must say (pdf.2B ...): the rule a finding cites."""
    spec = contract.acts.get(act)
    return spec.source if spec else None


def _num(v: Any) -> Any:
    """A number as int when whole (15.0 -> 15), else as float; anything else unchanged."""
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def _num_eq(a: Any, b: Any) -> bool:
    """Numeric equality ("45.67" == 45.67); False when either side is not a number."""
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


# =====================================================================================================================
# PART 2 - The speech check for any pack compiled from a compact graph: what the bot says, read and checked against the
#
# graph, the GMS and the session, with nothing named in code.
#
# The charge-review reader knows its own fields (merchants, amounts, dispositions, card actions, delivery fees).
# For a compiled pack every field comes from the graph:
#
# * **values**: every member of every value set a customer slot or a tool result takes, worded as the graph's
#   aliases and the pack lexicon's bot phrasings say; a mention is a claim on that value set, *offered* (in a question
#   or an offer), *done* (said as completed) or *stated*;
# * **figures**: a money figure is a value of the pack's money attribute (``*_usd``), a span of business days a value
#   of its days attribute, each about the member named nearest (else the one the customer chose); a number next to a
#   limit's unit ("18 years", "4 cards") is that limit;
# * **the opening statement**.
#
# Each claim is checked where its truth lives:
#
#     a value said to be done          a successful write of a tool that takes that value (the graph's write tools)
#     a value a tool returned          that tool's latest result (an ineligibility reason)
#     a value the customer gave        what the customer said
#     a figure                         the exact register (``lookup_enm``), the GMS geometry reported alongside
#     a limit                          the exact register's limit
#
# Completeness reads each act's required fields by where their truth lives (``says`` bindings: ``caller.<slot>``,
# ``result.<tool>.<field>``, ``policy.<attr>:caller.<slot>``, ``policy.limit:<key>``, ``text.opening``).
# =====================================================================================================================

from . import speech_reading as _r
from .speech_reading import BUSINESS_DAYS, MONEY, _dedupe, _nearest, _num as _read_num, _sentences, _squash, fallback


def _kind(attr: str) -> str:
    if attr.endswith("_usd"):
        return "money"
    if "days" in attr:
        return "days"
    return "count"


UP_TO = re.compile(r"\b(?:up to|at most|within|no more than|as long as|maximum of|max(?:imum)?)\s*$")
AT_LEAST = re.compile(r"\b(?:at least|as few as|no fewer than|no less than|minimum of|min(?:imum)?)\s*$")


class PackModel:
    """What the generic check needs to know about a compiled pack, read from its compact graph."""

    def __init__(self, g, lexicon: dict[str, Any] | None = None):
        self.g = g
        self.numeric = g.numeric_attributes()                         # attr -> {set, members}
        self.money = [a for a in self.numeric if _kind(a) == "money"]
        self.days = [a for a in self.numeric if _kind(a) == "days"]
        slot_sets = {sp["set"] for a in (g.data.get("acts") or {}) for sp in g.slot_specs(a).values()
                     if sp["set"] in g.values}
        self.returned = {}                                             # value set -> (tool, field)
        self.offered = {}                                              # value set -> (tool, field): the options a tool lists
        for t, sp in g.tools.items():
            for f, vs in (sp.get("returns") or {}).items():
                if isinstance(vs, str) and vs in g.values:
                    self.returned[vs] = (t, f)
                elif isinstance(vs, dict) and vs.get("subset_of") in g.values:
                    self.offered[vs["subset_of"]] = (t, f)
        self.value_sets = [vs for vs in g.values if vs in slot_sets or vs in self.returned]
        self.slot_of_set = {}
        for a in (g.data.get("acts") or {}):
            for sl, sp in g.slot_specs(a).items():
                self.slot_of_set.setdefault(sp["set"], sl)
        self.write_args = {}                                           # value set -> [(tool, arg)]
        for t, sp in g.tools.items():
            if sp.get("role") == "write":
                for arg, vs in (sp.get("args") or {}).items():
                    self.write_args.setdefault(vs, []).append((t, arg))
        self.units = {k: str(lv.get("unit") or "") for k, lv in g.limits.items() if not lv.get("internal")}
        lex = lexicon or {}
        self.phrases: dict[str, list[str]] = {}
        for vs in self.value_sets:
            for m in g.members(vs):
                ws = list(g.aliases(m)) + [m.lower().replace("_", " ")]
                ent = (lex.get("values") or {}).get(m) or {}
                ws += list((ent.get("say") or {}).values()) + list(ent.get("cues") or [])
                self.phrases[m] = sorted({w.lower() for w in ws if w and "{" not in w}, key=len, reverse=True)


LIMIT_CUE = re.compile(r"\b(limit|limited|up to|at most|maximum|max|no more than|only|cap|allowed|allows?|within|"
                       r"covers?|look ?back|per session|per call|in one call|minimum|min|at least|must be|or older|"
                       r"requires?|required|need to be|needs to be)\b")


def _unit_words() -> dict[str, list[str]]:
    """The pack's other words for a limit's unit (domain.yaml ``bot.unit_words``: transactions -> [charges])."""
    from ..s01_packs.loading import current
    try:
        return {k: list(v or []) for k, v in ((current().section("bot") or {}).get("unit_words") or {}).items()}
    except Exception:                                          # noqa: BLE001 - a pack without a bot section
        return {}


def _item_regex() -> re.Pattern | None:
    """The pack's wording of a returned item (``bot.item``: "the {merchant} charge for ${amount}") as a pattern, so a
    charge the words name that no tool returned is read too (a made-up charge)."""
    from ..s01_packs.loading import current
    try:
        tpl = (current().section("bot") or {}).get("item")
    except Exception:                                          # noqa: BLE001
        return None
    if not tpl or "{merchant}" not in tpl:
        return None
    parts = re.split(r"(\{\w+\})", str(tpl).lower())
    rx = ""
    for p in parts:
        if p == "{merchant}":
            rx += r"(?P<merchant>[a-z0-9][\w&'.-]*(?: [a-z0-9][\w&'.-]*){0,2}?)"
        elif p.startswith("{"):
            rx += r"[\w.,$-]+"
        else:
            rx += re.escape(p)
    try:
        return re.compile(rx)
    except re.error:
        return None


class GenericReader(UtteranceReader):
    """Reads a compiled pack's bot turns into claims on its value sets, figures and limits."""

    def __init__(self, contract, model: PackModel):
        super().__init__(contract)
        self.m = model
        self._value_phrases = {vs: sorted({(p, v) for v in model.g.members(vs) for p in model.phrases.get(v, [])},
                                          key=lambda x: -len(x[0])) for vs in model.value_sets}
        self._item_re = _item_regex()
        self._limit_res = {}
        words = _unit_words()
        for k, unit in model.units.items():
            if unit:
                stems = [re.escape(w.rstrip("s")).replace("_", "[ _]") for w in [unit, *words.get(unit, [])]]
                alt = "|".join(dict.fromkeys(stems))        # "calendar_days": "30 calendar days"; "transactions": "charges"
                self._limit_res[k] = [re.compile(rf"\b(\d+)\s+(?:\w+\s+)?(?:{alt})s?\b")]

    def read(self, text: str, facts, about_item: str | None = None, question: Any = None) -> ReadUtterance:
        out = ReadUtterance(text=text or "", sentences=_sentences(text or ""))
        if self._opening and self._opening in _squash(text or ""):
            out.claims.append(Claim("opening", True, self.c.opening, text))
        seen: set = set()                                       # (field, about) of the figures said so far
        for sent in out.sentences:
            new = self._sentence(sent, facts, question, seen)
            seen.update((c.field, c.about) for c in new)
            out.claims += new
        out.heard = self._heard(out)
        return out

    def _sentence(self, sent: str, facts, question: Any = None, seen: set | None = None) -> list[Claim]:
        low = words_to_digits(sent).lower()
        asks = bool(_r.QUESTION_OR_CONDITION.search(sent.strip()))
        mode = "offered" if asks or _r.OFFER.search(low) else "done" if _r.COMPLETION.search(low) \
            else "recorded" if _r.RECORDING_VERBS.search(low) else "states"
        claims: list[Claim] = []
        # the charges the tools returned, named by their merchant (or their id): "the Amazon charge for $45.67"
        items = []
        for tid, it in (getattr(facts, "items", None) or {}).items():
            for name in dict.fromkeys([str(it.get("merchant") or "").lower(), str(tid).lower()]):
                if name:
                    items += [(m.start(), m.end(), tid, m.group(0)) for m in
                              re.finditer(r"(?<![\w$])" + re.escape(name) + r"(?![\w])", low)]
        for mt in self._item_re.finditer(low) if self._item_re is not None else ():
            if not any(s0 < mt.end() and mt.start() < e0 for s0, e0, _, _ in items):   # a charge no tool returned
                items.append((mt.start("merchant"), mt.end("merchant"), None, mt.group("merchant")))
        items.sort(key=lambda x: x[0])
        named: dict[str, list] = {vs: self._values_named(low, vs) for vs in self.m.value_sets}
        for vs, ms in named.items():
            for mt in ms:                            # a value said of a named charge is about that charge
                near = max((x for x in items if x[0] <= mt.start), default=None, key=lambda x: x[0])
                claims.append(Claim(vs, mt.value, mt.words, sent, near[2] if near else None, mode))
        asked = question.get("entity") if question is not None else None

        def about(attr: str, pos: int) -> str | None:
            vs = self.m.numeric[attr]["set"]
            hit = _nearest(named.get(vs) or [], pos)
            if hit or asked:
                return hit or fallback(asked, attr, seen if seen is not None else set())
            return facts.caller.get(self.m.slot_of_set.get(vs, ""))

        for s0, _, tid, w in items:
            claims.append(Claim("item", tid if tid is not None else w, w, sent, tid, mode))
        days_spans = [(m.start(), m.end()) for m in BUSINESS_DAYS.finditer(low)]
        for mt in MONEY.finditer(low):
            v = _read_num(mt.group(1) or mt.group(2))
            if items:                                  # money said of a named charge is that charge's amount
                near = min(items, key=lambda x: abs(x[0] - mt.start()))
                opts = [m for attr in self.m.money[:1] for m in named.get(self.m.numeric[attr]["set"]) or []]
                if not opts or abs(near[0] - mt.start()) < min(abs(m.start - mt.start()) for m in opts):
                    claims.append(Claim("amount", v, mt.group(0), sent, near[2]))
                    continue
            for attr in self.m.money[:1]:
                claims.append(Claim(attr, v, mt.group(0), sent, about(attr, mt.start())))
        fm = _r.FREE.search(low)
        if fm and self.m.money:
            attr = self.m.money[0]
            if named.get(self.m.numeric[attr]["set"]):
                claims.append(Claim(attr, 0, fm.group(0), sent, about(attr, fm.start())))
        for mt in BUSINESS_DAYS.finditer(low):
            lo, hi = int(mt.group(1)), int(mt.group(2) or mt.group(1))
            before = low[max(0, mt.start() - 24):mt.start()]
            bound = "max" if UP_TO.search(before) and lo == hi else "min" if AT_LEAST.search(before) and lo == hi else None
            pick = [a for a in self.m.days if bound and a.endswith("_" + bound)] or self.m.days[:1]
            for attr in pick[:1]:                      # "up to 7 business days" is the maximum, "at least 5" the minimum
                claims.append(Claim(attr, (lo, hi), mt.group(0), sent, about(attr, mt.start())))
        if LIMIT_CUE.search(low):                      # a count is a limit only when said as one ("up to 4")
            for key, regs in self._limit_res.items():
                for rx in regs:
                    for mt in rx.finditer(low):
                        if any(s <= mt.start() < e for s, e in days_spans):
                            continue
                        claims.append(Claim("limit", int(mt.group(1)), mt.group(0), sent, key))
        return _dedupe(claims)

    def _heard(self, r: ReadUtterance) -> list[str]:
        low = words_to_digits(r.text).lower()
        said = {c.field for c in r.claims}
        heard = []
        for act, spec in self.c.acts.items():
            need = [self.c.claim_field(f) for f in spec.says if f.required]
            need = [x for x in need if x]
            if any(q.lower() in low for q in spec.cues) or (need and all(x in said for x in need)):
                heard.append(act)
        return heard


class _Steps:
    """The rules a claim's check tried, in order: each one that did not apply, then the one that did."""

    def __init__(self):
        self.steps: list[dict] = []

    def no(self, rule: str, detail: str = "") -> None:
        self.steps.append({"rule": rule, "hit": False, "detail": detail})

    def hit(self, v: ClaimVerdict, rule: str, detail: str = "", compare: dict | None = None) -> ClaimVerdict:
        from dataclasses import replace
        self.steps.append({"rule": rule, "hit": True, "detail": detail})
        return replace(v, steps=tuple(self.steps), compare=compare if compare is not None else v.compare)


class GenericChecks:
    """Claim checks and completeness for a compiled pack (used by UtteranceVerifier when the contract is generic)."""

    def __init__(self, verifier, model: PackModel):
        self.v, self.m = verifier, model

    def _asks_for(self, state: str) -> set[str]:
        """The value sets a prompt state asks the customer to choose from (its accepted acts' slots)."""
        from ..s01_packs.names import graph_names
        return {sp["set"] for a in graph_names().accepts.get(state, ()) for sp in self.m.g.slot_specs(a).values()
                if sp.get("set")}

    def verify_claim(self, cl: Claim, facts, label: dict | None = None):
        """Grade one claim: the first rule that applies says where its truth is (each rule tried is kept in
        ``steps``, the comparison made in ``compare``, for the speech check page)."""
        f, m = cl.field, self.m
        store = self.v.store
        st = _Steps()
        if f != "opening":
            st.no("the required opening statement", "not the opening")
        else:
            return st.hit(ClaimVerdict(cl, "supported", self.v.c.opening, "required opening statement"),
                          "the required opening statement", "the words contain the policy's opening, word for word",
                          {"op": "contains, word for word", "said": cl.words, "truth": self.v.c.opening, "result": True})
        if f in ("item", "amount"):                     # a charge the tools returned, and its amount
            it = facts.items.get(str(cl.about)) if getattr(facts, "items", None) else None
            if f == "item":
                if it is not None:
                    return st.hit(ClaimVerdict(cl, "supported", cl.value, f"the returned charge {cl.about}"),
                                  "a charge the tools returned", f"\u201c{cl.words}\u201d names {cl.about}, a charge the tools returned",
                                  {"op": "is one of the returned charges", "said": cl.value, "truth": sorted(facts.items),
                                   "result": True})
                return st.hit(ClaimVerdict(cl, "unsupported", sorted(facts.items), "the returned charges",
                                           f"no tool returned a charge from {cl.words}"),
                              "a charge the tools returned", f"\u201c{cl.words}\u201d names no charge the tools returned",
                              {"op": "is one of the returned charges", "said": cl.words, "truth": sorted(facts.items),
                               "result": False})
            truth = (it or {}).get("amount")
            if truth is None:
                return st.hit(ClaimVerdict(cl, "unsupported", None, "the returned charges",
                                           f"no amount returned for {cl.about}" if cl.about else
                                           "the amount of a charge no tool returned"),
                              "the amount of a returned charge", "the tools returned no amount for it")
            ok = _num_eq(cl.value, truth)
            return st.hit(ClaimVerdict(cl, "supported" if ok else "contradicted", truth, f"the returned charge {cl.about}",
                                       "" if ok else f"says {cl.value}, the charge's amount is {truth}"),
                          "the amount of a returned charge", f"the tools returned {cl.about} with amount {truth}",
                          {"op": "equals, as numbers", "said": cl.value, "truth": truth, "result": ok})
        st.no("a returned charge or its amount", "not about a charge")
        if f == "limit":
            truth = store.lookup_enm("limit", cl.about)
            ok = _num_eq(cl.value, truth)
            return st.hit(ClaimVerdict(cl, "supported" if ok else "contradicted", truth, f"exact register limit/{cl.about}",
                                       "" if ok else f"{cl.about} is {truth}", self.v._geometry(cl.about, "has_value", cl.value)),
                          "a policy limit", f"the GMS exact register: lookup_enm('limit', {cl.about!r}) = {truth}",
                          {"op": "equals, as numbers", "said": cl.value, "truth": truth, "result": ok})
        st.no("a policy limit", "not a limit")
        if f in m.numeric:
            members = list(m.numeric[f]["members"])
            truth_of = lambda x: store.lookup_enm(f, x)                       # noqa: E731
            rng = isinstance(cl.value, tuple)
            fits = (lambda t: _num_eq(cl.value, t)) if not rng else \
                (lambda t: cl.value[0] <= float(t) and float(t) <= cl.value[1] or _num_eq(cl.value[1], t))
            op = "a range that holds the register value" if rng else "equals, as numbers"
            who = None if cl.about == ANY else cl.about or (label or {}).get(m.slot_of_set.get(m.numeric[f]["set"], ""))
            rule = f"a policy figure ({f})"
            if who in members:
                truth = truth_of(who)
                ok = fits(truth)
                return st.hit(ClaimVerdict(cl, "supported" if ok else "contradicted", truth, f"exact register {f}/{who}",
                                           "" if ok else f"{who}: {f} is {truth}",
                                           self.v._geometry(who, f"has_{f}", cl.value if not rng else cl.value[1])),
                              rule, f"about {who}: the GMS exact register lookup_enm({f!r}, {who!r}) = {truth}",
                              {"op": op, "said": cl.value, "truth": truth, "result": ok})
            hits = [x for x in members if fits(truth_of(x))]
            if hits:
                return st.hit(ClaimVerdict(cl, "supported", cl.value, f"exact register {f}/{hits[0]}"), rule,
                              f"about nothing named: some member has it ({hits[0]}: {truth_of(hits[0])})",
                              {"op": f"{op}, for any of {', '.join(members)}", "said": cl.value,
                               "truth": {x: truth_of(x) for x in members}, "result": True})
            return st.hit(ClaimVerdict(cl, "unsupported", None, f"exact register {f}", f"no member has this {f}"), rule,
                          f"about nothing named: no member of {m.numeric[f]['set']} has it",
                          {"op": f"{op}, for any of {', '.join(members)}", "said": cl.value,
                           "truth": {x: truth_of(x) for x in members}, "result": False})
        st.no("a policy figure", "not a number the policy register holds")
        if f in m.value_sets:
            summary = [v for k, v in (label or {}).items() if k not in ("act", "text") and isinstance(v, dict)]
            rule = "a summary of what was written"
            if summary and f in m.write_args:           # a summary of what was written: each value must be one
                writes = [(a, w) for t, a in m.write_args[f] for w in facts.writes.get(t, [])]
                done = {str((w.get("args") or {}).get(a)) for a, w in writes}
                tools = ", ".join(t for t, _ in m.write_args[f])
                mine = {str((w.get("args") or {}).get(a)) for a, w in writes
                        if cl.about is not None and str(cl.about) in map(str, (w.get("args") or {}).values())}
                if mine:                                # said of a named charge: what was written for that charge
                    ok = str(cl.value) in mine
                    return st.hit(ClaimVerdict(cl, "supported" if ok else "contradicted", sorted(mine),
                                               f"successful {tools} for {cl.about}",
                                               "" if ok else f"says {cl.value} for {cl.about}, the write was {', '.join(sorted(mine))}"),
                                  rule, f"a summary naming {cl.about}: the truth is what a successful {tools} wrote for it",
                                  {"op": "is one of", "said": str(cl.value), "truth": sorted(mine), "result": ok})
                ok = str(cl.value) in done
                return st.hit(ClaimVerdict(cl, "supported" if ok else "contradicted", sorted(done), f"successful {tools}",
                                           "" if ok else f"no successful {tools} wrote {cl.value}"),
                              rule, f"the act's label carries a record ({', '.join(k for k, v in (label or {}).items() if isinstance(v, dict))}) "
                                    f"and {tools} writes {f}: the truth is every value a successful {tools} wrote",
                              {"op": "is one of", "said": str(cl.value), "truth": sorted(done), "result": ok})
            st.no(rule, "the act's label carries no record" if not summary else f"no tool writes {f}")
            rule = "said to be done"
            if cl.mode == "done" and f in m.write_args:
                done = [(t, a, w) for t, a in m.write_args[f] for w in facts.writes.get(t, [])]
                tools = ", ".join(t for t, _ in m.write_args[f])
                if not done:
                    return st.hit(ClaimVerdict(cl, "unsupported", None, tools, f"said to be done, but no {tools} succeeded"),
                                  rule, f"the sentence says it is done: the truth is a successful {tools}, and none succeeded",
                                  {"op": "is one of", "said": str(cl.value), "truth": [], "result": False})
                vals = {(w.get("args") or {}).get(a) for _, a, w in done}
                ok = cl.value in vals
                return st.hit(ClaimVerdict(cl, "supported" if ok else "contradicted", sorted(map(str, vals)),
                                           f"successful {done[-1][0]}", "" if ok else f"the write was for {', '.join(map(str, vals))}"),
                              rule, f"the sentence says it is done: the truth is what a successful {tools} wrote",
                              {"op": "is one of", "said": str(cl.value), "truth": sorted(map(str, vals)), "result": ok})
            st.no(rule, f"the sentence is put as {cl.mode!r}, not done" if cl.mode != "done" else f"no tool writes {f}")
            rule = "said to be recorded"
            if cl.mode == "recorded" and f in m.write_args:   # "I've noted ...": a successful write must be behind it
                tools = ", ".join(t for t, _ in m.write_args[f])
                writes = [(a, w) for t, a in m.write_args[f] for w in facts.writes.get(t, [])]
                if cl.about is not None:
                    writes = [(a, w) for a, w in writes if str(cl.about) in map(str, (w.get("args") or {}).values())]
                vals = sorted({str((w.get("args") or {}).get(a)) for a, w in writes})
                if not vals:
                    return st.hit(ClaimVerdict(cl, "unsupported", None, tools,
                                               f"said to be recorded, but no successful {tools}"
                                               + (f" for {cl.about}" if cl.about else "")),
                                  rule, f"a recording word: the truth is a successful {tools}, and none succeeded",
                                  {"op": "is one of", "said": str(cl.value), "truth": [], "result": False})
                if str(cl.value) not in vals:
                    return st.hit(ClaimVerdict(cl, "contradicted", vals, f"successful {tools}",
                                               f"says {cl.value} was recorded, the record is {', '.join(vals)}"),
                                  rule, f"a recording word: the truth is what a successful {tools} wrote",
                                  {"op": "is one of", "said": str(cl.value), "truth": vals, "result": False})
                st.no(rule, f"a successful {tools} wrote {cl.value}: now held to what the customer said too")
            else:
                st.no(rule, "no recording word" if cl.mode != "recorded" else f"no tool writes {f}")
            rule = "an option the tools offered"
            if cl.mode == "offered" and f in m.offered and m.offered[f][0] in facts.results:
                tool, fld = m.offered[f]
                opts = [str(x) for x in (facts.results.get(tool, {}).get(fld) or [])]
                ok = str(cl.value) in opts
                return st.hit(ClaimVerdict(cl, "supported" if ok else "unsupported", opts, f"{tool} result {fld}",
                                           "" if ok else f"offers {cl.value}; {tool} offered only {', '.join(opts) or 'nothing'}"),
                              rule, f"an option put to the customer: the truth is the options {tool} returned",
                              {"op": "is one of the options offered", "said": str(cl.value), "truth": opts, "result": ok})
            st.no(rule, "not put as an offer" if cl.mode != "offered" else
                  f"no tool lists the {f} options" if f not in m.offered else f"{m.offered[f][0]} was not called")
            rule = "a value a tool returned"
            if f in m.returned:
                tool, fld = m.returned[f]
                truth = facts.results.get(tool, {}).get(fld)
                if truth is None:
                    return st.hit(ClaimVerdict(cl, "unsupported" if cl.mode == "states" else "unverifiable", None, tool,
                                               f"{tool} returned no {fld}"), rule, f"{tool} returns {f}, but it returned no {fld}")
                return st.hit(self.v._compare(cl, truth, f"{tool} result {fld}"), rule,
                              f"{tool} returns {f}: the truth is its result's {fld} = {truth}")
            st.no(rule, f"no tool returns {f}")
            slot = m.slot_of_set.get(f)
            truth = facts.caller.get(slot) if slot else None
            asked = (label or {}).get("act")
            states_one = any(isinstance(v, str) and m.g.member_set(v) == f for k, v in (label or {}).items()
                             if k not in ("act", "text"))       # a confirmation names its value: not a choice offered
            rule = "a choice the prompt offers"
            if (cl.mode == "offered" and asked and not states_one and f in self._asks_for(asked)
                    and cl.value in m.g.members(f)):
                return st.hit(ClaimVerdict(cl, "supported", cl.value, f"{asked} asks for a {f}",
                                           "a choice the prompt offers (the graph: it accepts this value set)"),
                              rule, f"put as an offer, and {asked} asks the customer for a {f} (the graph)",
                              {"op": f"is a member of {f}", "said": str(cl.value), "truth": list(m.g.members(f)), "result": True})
            st.no(rule, "not put as an offer" if cl.mode != "offered" else
                  f"{asked} names its own value" if states_one else f"{asked or 'this act'} does not ask for a {f}")
            rule = "the subject the customer asked about"
            if getattr(facts, "asked", None) is not None and str(cl.value) == str(facts.asked):
                return st.hit(ClaimVerdict(cl, "supported", facts.asked, "the customer's policy question",
                                           "the option the customer asked about"),
                              rule, f"this reply answers the customer's question about {facts.asked} "
                                    f"(the customer's meaning card: ASK_KB entity={facts.asked})",
                              {"op": "is the subject of the question", "said": str(cl.value), "truth": facts.asked,
                               "result": True})
            st.no(rule, "the customer's last line is not a policy question" if getattr(facts, "asked", None) is None
                  else f"the customer asked about {facts.asked}")
            rule = "what the customer said"
            if truth is None or cl.mode == "offered":
                return st.hit(ClaimVerdict(cl, "unverifiable", None, "", "mentioned, not asserted" if truth else
                                           "the customer has not said it yet"),
                              "nothing to compare", f"put as an offer: mentioned, not asserted" if truth is not None else
                              f"the customer's meaning cards have no {slot or f} yet")
            return st.hit(self.v._compare(cl, truth, "what the customer said"), rule,
                          f"the customer's meaning card gave {slot} = {truth} (facts.caller.{slot})")
        st.no("a value set", f"{f} is not a value set")
        return st.hit(ClaimVerdict(cl, "unverifiable", None, "", f"no check for {f}"), "no check for this field", "")

    def completeness(self, act: str, label: dict, read: ReadUtterance, facts, log: list | None = None) -> list:
        """The required fields of ``act`` (its ``says``) the words leave out. ``log`` gets one entry per field."""
        spec = self.v.c.acts.get(act)
        log = log if log is not None else []
        if spec is None:
            log.append({"check": f"{act} has no required wording in the policy", "ok": True, "detail": ""})
            return []
        out = []
        if not spec.says:
            log.append({"check": f"{act} must say nothing in particular", "ok": True, "detail": "its says list is empty"})
        for fs in spec.says:
            name = f"{act} says the {fs.field}"
            if not fs.required:
                log.append({"check": name, "ok": True, "skip": True, "detail": "optional"})
                continue
            want = self.v.c.claim_field(fs)
            src = fs.source
            if want is None:
                log.append({"check": name, "ok": True, "skip": True,
                            "detail": f"{src}: free wording the reader does not read as a claim"})
                continue
            if src.startswith("caller.") and facts.caller.get(src.split(".", 1)[1]) is None and not label.get(fs.field):
                log.append({"check": name, "ok": True, "skip": True, "detail": f"the customer has not said the {src[7:]} yet"})
                continue
            if fs.field in label and label[fs.field] in (None, ""):
                log.append({"check": name, "ok": True, "skip": True, "detail": "the act has no value for it"})
                continue                              # the act has no value for it (a card action with no delivery)
            if src.startswith("result."):
                tool, _, fld = src[7:].partition(".")
                if facts.results.get(tool, {}).get(fld) is None:
                    log.append({"check": name, "ok": True, "skip": True, "detail": f"{tool} returned no {fld}"})
                    continue
            said = read.of(want)
            log.append({"check": name, "ok": bool(said), "detail": (f"read as a {want} claim: " + ", ".join(
                f"\u201c{c.words}\u201d" for c in said)) if said else f"no {want} claim in the words ({src})"})
            if not said:
                out.append(Finding("missing", act, f"{act} must say the {fs.field} ({src})", spec.source))
        return out


def claim_field(model: PackModel, binding: str) -> str | None:
    """The claim field a ``says`` binding is read as (None: free text the reader does not read)."""
    kind, _, rest = binding.partition(".")
    if kind == "text" and rest == "opening":
        return "opening"
    if kind == "policy":
        what, _, _ = rest.partition(":")
        return "limit" if what.startswith("limit") else what
    if kind == "caller":
        vs = model.g.slot_set(rest)
        return vs if vs in model.value_sets else None
    if kind == "result":
        tool, _, fld = rest.partition(".")
        vs = (model.g.tools.get(tool, {}).get("returns") or {}).get(fld)
        return vs if isinstance(vs, str) and vs in model.value_sets else None
    if kind == "attribute":
        return None
    return None


def _faithfulness(self, label: dict, read: ReadUtterance, log: list | None = None) -> list:
    """Do the words say what the label claims? A labelled value of a value set must be the one the words say
    (when they name one); a labelled act must be heard. ``log`` gets one entry per check made."""
    act = label["act"]
    spec = self.v.c.acts.get(act)
    log = log if log is not None else []
    out = []
    # an act is judged only when the words could show it: a cue, or a field it must say and has a value for
    # (TRANSFER's reason is optional; a card action with no delivery has no delivery to say)
    need = [self.v.c.claim_field(f) for f in (spec.says if spec else []) if f.required and label.get(f.field) is not None]
    need = [x for x in need if x]
    judgeable = spec and (spec.cues or need)
    said = {c.field for c in read.claims}
    heard = act in read.heard or (need and all(x in said for x in need))
    how = []
    if spec and spec.cues:
        how.append("a cue word (" + ", ".join(f"\u201c{q}\u201d" for q in spec.cues[:6]) + ("..." if len(spec.cues) > 6 else "") + ")")
    if need:
        how.append("a claim for every field it must say (" + ", ".join(need) + ")")
    if not judgeable:
        log.append({"check": f"the words express {act}", "ok": True, "skip": True,
                    "detail": f"{act} has no cue words and nothing it must say: words cannot show it, so it is not judged"})
    else:
        log.append({"check": f"the words express {act}", "ok": bool(heard) or not read.text.strip(),
                    "detail": ("heard: " if heard else "not heard: no ") + " or ".join(how)})
    if judgeable and not heard and read.text.strip():
        out.append(Finding("label_not_said", act, f"labelled {act} but the words do not express it: {read.text!r}"))
    for fld, want in label.items():
        if fld in ("act", "text") or want in (None, ""):
            continue
        vs = self.m.g.member_set(str(want)) if isinstance(want, str) else None
        if vs in self.m.value_sets:
            said = {str(c.value) for c in read.of(vs)}
        elif isinstance(want, (int, float)) and self.m.money:
            said = {str(_num(c.value)) for c in read.of(self.m.money[0])}
            want = _num(want)
        else:
            continue
        name = f"the words agree with the label's {fld} = {want}"
        if not said:
            log.append({"check": name, "ok": True, "skip": True, "detail": f"the words name no {vs or fld}"})
        else:
            log.append({"check": name, "ok": str(want) in said, "detail": "the words say " + ", ".join(sorted(said))})
        if said and str(want) not in said:
            out.append(Finding("text_contradicts_label", act, f"label {fld}={want}, the words say {', '.join(sorted(said))}"))
    return out


GenericChecks.faithfulness = _faithfulness


# =====================================================================================================================
# PART 3 - Validating what the bot actually says, sentence by sentence, against the policy graph and the GMS.
#
# Why
# ---
# A bot hands the harness two things every turn: act labels (``CONFIRM_RECORDED(T1, UNRECOGNIZED)``)
# and the words the customer hears. Outcome and trajectory grading read the labels and the tool log.
# The words were never re-read, so a bot that labels everything correctly but *tells* the customer
# something else ("I've noted that you recognize the Amazon charge", "express delivery is $25") passed.
# This package reads the words on their own and checks every fact they state.
#
# The flow for one bot turn
# -------------------------
# 1. ``SessionFacts`` (session.py) holds what is true in this conversation so far: the transactions
#    the tool returned, every write that truly succeeded, the card options offered, and what the
#    customer said (as the runner's reader understood it).
# 2. ``UtteranceReader`` (reader.py) reads the text, ignoring the labels, into ``Claim``s: one fact
#    each ("amount = 45.67 about T1", "fee = 25 about expedited_delivery", "disposition recorded =
#    RECOGNIZED about T1"), plus the acts its words express.
# 3. ``UtteranceVerifier`` (verify.py) checks each claim where its truth lives:
#      account facts       -> SessionFacts (what the tools returned / what was written)
#      policy facts        -> the GMS: exact register (lookup_enm), with the trained geometry's
#                             distance (score_triple vs the relation's cap) reported alongside
#      rule stances        -> the GMS stance check (u-space tension, calibrated cuts)
#      the customer        -> what the customer said
#    and checks the turn as a whole: does the text say what the labels claim (faithfulness), does
#    each act say what the policy requires it to say (completeness), and is each act allowed at
#    this point of the dialog-state graph (flow).
# 4. The result, a ``TurnReport``, lists every claim with its verdict and every finding.
#
# What it reads from the policy graph (policy/charge_review_kg.json, kind "speech")
# ----------------------------------------------------------------------------------
# * ``ACT says FIELD``  (from = where the truth lives, required)   e.g. PRESENT_TXN says date, from item.date
# * ``ACT spoken_cue W`` words that mark an act with no content     e.g. ASK_MORE_HELP "anything else"
# * ``VALUE spoken_as W`` how the bot words a value                 e.g. UNRECOGNIZED "isn't yours"
# * ``TOOL returns_field F``                                        e.g. get_review_transactions date
# * ``RULE asserted_by / denied_by W``                              e.g. R27 "means fraud" / "not proof of fraud"
# plus the graph's value sets, synonyms, opening statement and dialog-state transitions.
#
# Usage
# -----
#     from gmsjourney.s07_checks.speech_checks import SpeechContract, SessionFacts, UtteranceVerifier
#
#     verifier = UtteranceVerifier(store, SpeechContract.from_graph(store.kg))
#     facts = SessionFacts.from_run(tool_log, customer_frames, session_date="2026-03-16")
#     report = verifier.check_turn(acts, text, facts, previous_acts=prev, customer_act="INFORM_DISPOSITION")
#     report.ok              # True when every claim is supported and there is no finding
#     report.as_dict()       # every claim, its verdict and truth, every finding (for reports)
#
# The campaign runs this on every bot turn of every scenario (s06_bots/harness.py) and grades the run on
# it (s07_checks/scoring.py: ``c_speech`` and the ``speech_*`` columns).
# =====================================================================================================================

from .speech_reading import ActContract, FieldSpec
from .speech_reading import Truth

__all__ = ["ActContract", "FieldSpec", "SpeechContract", "Claim", "ReadUtterance", "UtteranceReader",
           "SessionFacts", "Truth", "ClaimVerdict", "Finding", "TurnReport", "UtteranceVerifier"]
