"""Verifying a bot turn: every fact it states, and whether it says what it should.

Input   for each bot turn: the acts the bot labelled (each may carry its own ``text``; a bot that
        only returns text has no labels), the text the customer hears, and the conversation so far
        (the tool log and the customer's frames). The checker for one conversation keeps the rest
        (the transaction last talked about, the prompts the customer is answering).
Output  a ``TurnReport`` per turn:
          verdicts  one ``ClaimVerdict`` per claim the words make:
                      supported     the words match where the truth lives
                      contradicted  they state something else (the wrong fee, the wrong answer recorded)
                      unsupported   nothing backs them (claimed recorded / done when nothing succeeded,
                                    a charge no tool returned)
                      unverifiable  nothing to check against (advisory, never a failure)
          findings  what is wrong with the turn as a whole:
                      text_contradicts_label  the words say something other than the act label
                      label_not_said          an act is labelled but the words do not express it
                      missing                 a required piece of information is not said (R6, R19, R26 ...)
                      not_allowed_here        the act does not follow from the dialog-state graph (flow)
        ``TurnReport.ok`` is True when no claim is contradicted or unsupported and there is no
        finding other than flow (flow is reported separately: ``flow_ok``).

Where each claim's truth is read:
  merchant / amount / date / age_days    the transactions the tool returned           (SessionFacts.items)
  disposition said to be recorded        a successful record_charge_response for it    (R12, R13, R33)
  disposition stated otherwise           what the customer said                        (SessionFacts.caller)
  action said to be done                 a successful submit_card_action               (R17)
  action offered                         the options get_card_action_options returned  (R15)
  fee / delivery_days / limit            the GMS exact register (lookup_enm), with the trained
                                         geometry's distance score_triple(...) vs the relation's cap
  stance                                 the GMS stance check (u-space tension vs calibrated cuts)
  opening                                the required opening statement
  recorded_count                         the number of successful records

Usage (one conversation):
    verifier = UtteranceVerifier(store, SpeechContract.from_graph(store.kg))
    checker = verifier.conversation(session_date="2026-03-16")
    report = checker.check(acts, text, tool_log, customer_frames)   # after every bot turn
    checker.reports                                                 # every turn's report
    checker.summary()                                               # counts for the run's row
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract import SpeechContract
from .reader import Claim, ReadUtterance, UtteranceReader, words_to_digits
from .session import SessionFacts

FAILING = ("contradicted", "unsupported")
REASK_AFTER = {"ASK_CLARIFY", "ANSWER_KB"}     # after these the bot may re-ask the prompt that was pending


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
        if f in ("merchant", "amount", "date", "age_days"):
            return "tool result"
        if f == "disposition":
            return "successful write" if mode == "recorded" else "what the customer said"
        if f in ("action", "delivery"):
            return "tool result" if mode == "offered" else "successful write"
        if f in ("fee", "delivery_days", "limit"):
            return "GMS exact register"
        if f == "stance":
            return "GMS stance check"
        if f == "recorded_count":
            return "successful write"
        return "policy text" if f == "opening" else "other"

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view: the claim's fields plus status, basis, truth, source, why (and geometry if any)."""
        return {**self.claim.as_dict(), "status": self.status, "basis": self.basis,
                "truth": list(self.truth) if isinstance(self.truth, tuple) else self.truth,
                "source": self.source, "why": self.why, **({"geometry": self.geometry} if self.geometry else {})}


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
    """
    turn: int
    text: str
    acts: list[str]
    heard: list[str]
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def flow_findings(self) -> list[Finding]:
        """The ``not_allowed_here`` findings (flow is graded on its own, as ``c_flow``)."""
        return [f for f in self.findings if f.kind == "not_allowed_here"]

    @property
    def ok(self) -> bool:
        """Every claim supported (or unverifiable) and the words say what the labels and the policy require."""
        return not any(v.failed for v in self.verdicts) and not [f for f in self.findings if f.kind != "not_allowed_here"]

    @property
    def flow_ok(self) -> bool:
        """Every act of the turn is a transition the dialog-state graph allows."""
        return not self.flow_findings

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready view (stored per turn in ``extras["speech"]["turns"]``)."""
        return {"turn": self.turn, "text": self.text, "acts": self.acts, "heard": self.heard, "ok": self.ok,
                "flow_ok": self.flow_ok, "claims": [v.as_dict() for v in self.verdicts],
                "findings": [f.as_dict() for f in self.findings]}


# ------------------------------------------------------------------ the verifier
class UtteranceVerifier:
    """Checks utterances against a GMS store and the speech contract read from its policy graph.

    store     the knowlytix GMS store (needs lookup_enm, score_triple, tau, stance_check)
    contract  SpeechContract.from_graph(store.kg)
    """

    def __init__(self, store, contract: SpeechContract, corroborate: bool = True):
        self.store, self.c = store, contract
        self.reader = UtteranceReader(contract)
        self._geo: dict[tuple, dict | None] = {}
        # knowlytix's own claim checker (AnswerVerifier, geometric mode: no LLM) as an independent second
        # reader of policy figures; it reports the store facts a sentence states correctly
        self._kx_verifier = None
        self._kx_seen: dict[str, list[tuple[str, str, str]]] = {}
        raw = getattr(store, "kx_raw", None)
        if corroborate and raw is not None:
            try:
                from knowlytix.knowledge.rag.verify import AnswerVerifier
                self._kx_verifier = AnswerVerifier(raw, None, mode="geometric")
            except Exception:                                   # noqa: BLE001 - corroboration is optional
                self._kx_verifier = None

    def conversation(self, session_date: str | None = None) -> "ConversationChecker":
        """A checker for one conversation (it remembers what was talked about across turns)."""
        return ConversationChecker(self, session_date)

    # ---------------------------------------------------------------- one turn
    def check_turn(self, acts: list[dict[str, Any]], text: str, facts: SessionFacts, *, turn: int = 0,
                   last_item: str | None = None, pending_prompts: set[str] | None = None,
                   customer_act: str | None = None, question: Any = None,
                   presented: set[str] | None = None) -> TurnReport:
        """Check one bot turn. ``acts`` are the labels (each may carry its own ``text``); with no labels the
        whole text is read and the acts it expresses stand in for them.

        last_item        the transaction talked about last (for sentences that name no merchant)
        pending_prompts  the prompts the customer was answering (for flow)
        customer_act     what the customer's last turn meant (for flow)
        question         the customer's policy question, when this turn answers one (ASK_KB frame)
        presented        transactions already presented: re-asking one need not repeat its facts (pdf.3:
                         do not ask the customer to repeat information without a reason)"""
        labelled = [a for a in acts if a.get("act")]
        per_act = labelled and all("text" in a for a in labelled)
        segments = [(a, a.get("text") or "") for a in labelled] if per_act else [(None, text or "")]
        report = TurnReport(turn, text or "", [a.get("act") for a in labelled], [])
        presented = presented or set()
        for label, seg_text in segments:
            about = (label or {}).get("txn") or last_item
            answering = question if (label is None or label.get("act") == "ANSWER_KB") else None
            read = self.reader.read(seg_text, facts, about_item=about, question=answering)
            report.heard += [h for h in read.heard if h not in report.heard]
            for cl in read.claims:
                report.verdicts.append(self._corroborated(self.verify_claim(cl, facts, label)))
            if label is not None:
                report.findings += self._faithfulness(label, read)
                report.findings += self._completeness(label["act"], label, read, facts, about, presented)
            elif not labelled:
                for act in read.heard:
                    report.findings += self._completeness(act, {}, read, facts, about, presented)
        names = report.acts or report.heard
        report.findings += self._flow(names, pending_prompts or set(), customer_act, first=turn == 0)
        return report

    # ---------------------------------------------------------------- one claim
    def verify_claim(self, cl: Claim, facts: SessionFacts, label: dict[str, Any] | None = None) -> ClaimVerdict:
        """Grade one claim against where its truth lives (the table in the module docstring).

        cl     the claim, as the reader produced it
        facts  the session so far
        label  the act label the claim's words belong to, if the bot gave one (a fee with no option named
               is then about the label's delivery option)
        """
        f = cl.field
        if f in ("merchant", "amount", "date", "age_days"):
            return self._item_claim(cl, facts)
        if f == "disposition":
            if cl.mode == "recorded":
                w = facts.write_for("record_charge_response", cl.about)
                if w is None:
                    return ClaimVerdict(cl, "unsupported", None, "record_charge_response",
                                        f"said to be recorded, but no record for {cl.about} succeeded (R13)")
                truth = w["args"].get("disposition")
                return self._compare(cl, truth, f"successful record_charge_response({cl.about})")
            truth = facts.caller.get("disposition")
            if truth is None:
                return ClaimVerdict(cl, "unverifiable", None, "", "the customer has not answered yet")
            return self._compare(cl, truth, "what the customer said")
        if f == "action":
            if cl.mode == "done":
                w = facts.write_for("submit_card_action", None)
                if w is None:
                    return ClaimVerdict(cl, "unsupported", None, "submit_card_action",
                                        "said to be done, but no card action succeeded (R17)")
                return self._compare(cl, w["args"].get("action"), "successful submit_card_action")
            if cl.mode == "offered":
                if facts.options is None:
                    return ClaimVerdict(cl, "unsupported", None, "get_card_action_options",
                                        "offered before any card option was returned (R15)")
                ok = cl.value in facts.options
                return ClaimVerdict(cl, "supported" if ok else "contradicted", facts.options,
                                    "get_card_action_options result",
                                    "" if ok else "not among the options the tool returned (R15)")
            return ClaimVerdict(cl, "unverifiable", None, "", "mentioned, not asserted")
        if f == "delivery":
            if cl.mode == "done":
                w = facts.write_for("submit_card_action", None)
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
            n = len(facts.writes.get("record_charge_response", []))
            return ClaimVerdict(cl, "supported" if cl.value == n else "contradicted", n,
                                "successful record_charge_response calls")
        return ClaimVerdict(cl, "unverifiable", None, "", f"no check for {f}")

    def _compare(self, cl: Claim, truth: Any, source: str) -> ClaimVerdict:
        """supported when the claimed value equals ``truth`` (case-insensitive), else contradicted."""
        ok = str(cl.value).lower() == str(truth).lower()
        return ClaimVerdict(cl, "supported" if ok else "contradicted", truth, source,
                            "" if ok else f"says {cl.value}, the truth is {truth}")

    def _item_claim(self, cl: Claim, facts: SessionFacts) -> ClaimVerdict:
        """An account fact must be what the tool returned for that transaction (R7, R24)."""
        if not facts.items:
            return ClaimVerdict(cl, "unsupported", None, "get_review_transactions",
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
        return ClaimVerdict(cl, "unsupported", None, "get_review_transactions",
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
        option = cl.about or (label or {}).get("delivery") or facts.caller.get("delivery")
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
        if self._kx_verifier is None or v.claim.field not in ("fee", "delivery_days") or not v.claim.about:
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
        return ClaimVerdict(v.claim, v.status, v.truth, v.source, v.why, geo)

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
    def _faithfulness(self, label: dict[str, Any], read: ReadUtterance) -> list[Finding]:
        """Do the words say what the label claims?"""
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
                      about: str | None, presented: set[str] = frozenset()) -> list[Finding]:
        """Does the act say everything the policy requires it to say (when there is something to say)?
        A transaction's facts are required the first time it is presented, not when it is re-asked."""
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
        if tool == "record_charge_response":
            named = {c.about for c in read.of("merchant")} | {c.about for c in read.of("disposition")}
            for w in facts.writes.get(tool, []):
                tid = w["args"].get("transaction_id")
                if tid not in named:
                    out.append(Finding("missing", act, f"{act} does not mention the recorded response for {tid}",
                                       spec_source(self.c, act)))
        elif tool == "submit_card_action":
            said = {c.value for c in read.of("action")}
            for w in facts.writes.get(tool, []):
                if w["args"].get("action") not in said:
                    out.append(Finding("missing", act, f"{act} does not mention the completed action "
                                                       f"{w['args'].get('action')}", spec_source(self.c, act)))
        return out

    def _flow(self, names: list[str], pending: set[str], customer_act: str | None, first: bool) -> list[Finding]:
        """Is each act where the dialog-state graph allows it? The first act of a turn must follow from a
        prompt the customer was answering, on what they said (or be allowed from anywhere: a person on
        request, a policy answer, a clarification, the turn limit); each next act must follow automatically
        from the one before (or re-ask the pending prompt after a clarification or a policy answer)."""
        if not names:
            return []
        out = []
        c = self.c
        if first:
            if "OPENING" in c.acts and names[0] != "OPENING":
                out.append(Finding("not_allowed_here", names[0], "the call must start with the opening (R1)", "R1"))
        else:
            a0 = names[0]
            ok = (any(a0 in c.next_states(p, customer_act) for p in pending)
                  or a0 in c.global_acts.get(customer_act or "", set()) or a0 in c.global_acts["any"]
                  or a0 in pending)
            if not ok:
                out.append(Finding("not_allowed_here", a0, f"{a0} does not follow from {sorted(pending)} on "
                                                           f"{customer_act}"))
        for a, b in zip(names, names[1:]):
            ok = b in c.next_states(a, "auto") or (a in REASK_AFTER and (b in pending or b in c.prompt_states))
            if not ok:
                out.append(Finding("not_allowed_here", b, f"{b} does not follow {a}"))
        return out


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

    def check(self, acts: list[dict[str, Any]], text: str, tool_log: list[dict[str, Any]],
              customer_frames: list[Any]) -> TurnReport:
        """Check one bot turn, then remember what it presented and asked.

        acts             the bot's act labels for the turn ([] for a text-only bot)
        text             what the customer hears
        tool_log         the backend's tool log so far (everything the bot did before speaking)
        customer_frames  the customer's frames so far (the last one is what this turn answers)
        """
        facts = SessionFacts.from_run(tool_log, customer_frames, self.session_date, self.v.c.success_values)
        last = customer_frames[-1] if customer_frames else None
        customer_act = getattr(last, "act", None)
        rep = self.v.check_turn(acts, text, facts, turn=len(self.reports), last_item=self.last_item,
                                pending_prompts=self.pending, customer_act=customer_act,
                                question=last if customer_act == "ASK_KB" else None, presented=self.presented)
        self.reports.append(rep)
        for a in acts:                             # a presentation, said or not, counts once
            if a.get("act") == "PRESENT_TXN" and a.get("txn"):
                self.presented.add(a["txn"])
        named = [a.get("txn") for a in acts if a.get("txn")] + \
                [v.claim.about for v in rep.verdicts if v.claim.field == "merchant"]
        if named:
            self.last_item = named[-1]
        names = rep.acts or rep.heard
        prompts = {a for a in names if a in self.v.c.prompt_states}
        if prompts:
            self.pending = prompts
        return rep

    def summary(self) -> dict[str, Any]:
        """Counts for the run's result row: claims by verdict, unfaithful / missing / flow findings, how many
        policy figures knowlytix's AnswerVerifier also read (and agreed on), and the two grades
        ``c_speech`` (every turn ok) and ``c_flow`` (every turn in flow)."""
        vs = [v for r in self.reports for v in r.verdicts]
        fs = [f for r in self.reports for f in r.findings]
        count = lambda s: sum(v.status == s for v in vs)  # noqa: E731
        kinds = lambda k: sum(f.kind == k for f in fs)    # noqa: E731
        kx = [v for v in vs if (v.geometry or {}).get("knowlytix")]
        agree = sum((v.status == "supported") == (v.geometry["knowlytix"] == "confirmed") for v in kx)
        return {"speech_claims": len(vs), "speech_supported": count("supported"),
                "speech_kx_checked": len(kx), "speech_kx_agree": agree,
                "speech_contradicted": count("contradicted"), "speech_unsupported": count("unsupported"),
                "speech_unverifiable": count("unverifiable"), "speech_unfaithful": kinds("text_contradicts_label")
                + kinds("label_not_said"), "speech_missing": kinds("missing"),
                "speech_flow": kinds("not_allowed_here"),
                "c_speech": int(all(r.ok for r in self.reports)), "c_flow": int(all(r.flow_ok for r in self.reports))}

    def problems(self, limit: int = 12) -> list[dict[str, Any]]:
        """Every failing claim and finding, with the turn and words, for failure reports."""
        out = []
        for r in self.reports:
            where = turn_where(r.turn)
            for v in r.verdicts:
                if v.failed:
                    out.append({"turn": r.turn, "where": where, "kind": v.status, "field": v.claim.field,
                                "words": v.claim.words,
                                "said": v.claim.value if not isinstance(v.claim.value, tuple) else list(v.claim.value),
                                "truth": v.truth if not isinstance(v.truth, tuple) else list(v.truth),
                                "why": v.why, "sentence": v.claim.sentence})
            for f in r.findings:
                out.append({"turn": r.turn, "where": where, "kind": f.kind, "act": f.act, "why": f.detail,
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

