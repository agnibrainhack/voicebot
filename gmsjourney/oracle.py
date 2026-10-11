"""The policy oracle: expected behaviour, computed rather than annotated (Ch. 12.4).

``PolicyOracle`` is an executable reading of the policy graph. It reads every
limit from the exact numeric register (lookup_enm), the opening line and rules
from the store, and the admissible transitions from the store's `enables`
edges; it then plays the journey against the user simulator on the same backend
fixture the bot will face. The resulting ``Trace`` is the ground truth: outcome
signature, per-stage decisions, which rules fired and which workflow edges
were traversed.

The oracle consumes semantic frames only. It never reads text, so its answer
depends on what the customer *meant*, which is exactly what the invariance
proof (enrich/invariance.py) shows enrichment leaves unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .journey import JourneySpec, UserSimulator, build_fixture
from .kb_truth import kb_answer
from .ontology import SemanticFrame
from .sut.tools import ToolEnvironment, ToolError, ToolGateway
from .trace import Trace


class _Stop(Exception):
    pass


class _TurnLimit(Exception):
    pass


@dataclass
class Limits:
    max_transactions_per_session: int
    lookback_days: int
    max_auth_attempts: int
    max_clarification_retries: int
    max_unsure_retries: int
    max_turns: int
    expedited_fee: float

    @classmethod
    def from_store(cls, store) -> "Limits":
        L = lambda k: store.lookup_enm("limit", k)
        return cls(L("max_transactions_per_session"), L("lookback_days"), L("max_auth_attempts"),
                   L("max_clarification_retries"), L("max_unsure_retries"), L("max_turns"),
                   store.lookup_enm("delivery_fee_usd", "expedited_delivery"))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class PolicyOracle:
    def __init__(self, store):
        self.store = store
        self.limits = Limits.from_store(store)
        self.opening = store.required_language.get("opening", "")

    # ------------------------------------------------------------------ public
    def run(self, spec: JourneySpec, fixture: dict[str, Any] | None = None,
            frames_override: list[SemanticFrame] | None = None,
            gateway_factory=None) -> Trace:
        """Play the journey. `frames_override` replays a fixed frame sequence instead of
        the simulator (used to re-derive ground truth from an enriched transcript)."""
        fixture = fixture or build_fixture(spec)
        env = ToolEnvironment(fixture)
        tools = gateway_factory(env) if gateway_factory else env
        sim = UserSimulator(spec, fixture, self.limits.lookback_days)
        replay = list(frames_override) if frames_override is not None else None
        tr = Trace()
        self._tr, self._tools, self._sim, self._replay = tr, tools, sim, replay
        self._phase = "service"
        try:
            self._journey()
        except _Stop:
            pass
        tr.tool_log = list(env.log)
        return tr

    # ----------------------------------------------------------------- plumbing
    def _bot(self, *acts: dict[str, Any]) -> None:
        self._tr.events.append({"role": "bot", "acts": list(acts)})

    def _ask(self, *acts: dict[str, Any]) -> SemanticFrame:
        """Emit bot acts that expect a reply; return the customer's frame."""
        if self._tr.turns >= self.limits.max_turns:
            raise _TurnLimit()
        self._bot(*acts)
        if self._replay is not None:
            frame = self._replay.pop(0) if self._replay else None
        else:
            frame = self._sim.respond(acts[-1])
        if frame is None:
            self._tr.notes.append("customer_silent")
            self._tr.ended = True
            raise _Stop()
        self._tr.turns += 1
        self._tr.user_frames.append(frame)
        self._tr.events.append({"role": "user", "frame": frame})
        return frame

    def _call(self, tool: str, **args: Any) -> Any:
        try:
            res = self._tools.call(tool, **args)
        except ToolError as e:                 # a failed call is part of the expected trajectory too
            self._tr.events.append({"role": "tool", "tool": tool, "args": args, "result": {"error": str(e)},
                                    "ok": False})
            raise
        self._tr.events.append({"role": "tool", "tool": tool, "args": args, "result": res})
        return res

    def _transfer(self, reason: str, rule: str) -> None:
        self._tr.rules_fired.add(rule)
        acts = []
        rec = self._recorded_claims()
        if rec:
            self._tr.rules_fired.add("R19")
            acts.append({"act": "SUMMARIZE", "recorded": rec})
        acts.append({"act": "TRANSFER", "reason": reason})
        self._bot(*acts)
        try:
            self._call("transfer_to_human", reason=reason)
        except ToolError:
            pass
        raise _Stop()

    def _recorded_claims(self) -> dict[str, str]:
        return dict(self._tr.claims_recorded)

    def _answer_kb(self, frame: SemanticFrame) -> dict[str, Any]:
        truth = kb_answer(self.store, frame)
        self._tr.kb.append({"frame": frame, "answer": truth.answer, "correct": True})
        self._tr.rules_fired.update({"R25"} | ({"R27"} if frame.get("intent") == "fraud" else set()))
        return {"act": "ANSWER_KB", "intent": frame.get("intent"), "answer": truth.answer}

    # ----------------------------------------------------------------- journey
    def _journey(self) -> None:
        try:
            self._authenticate()
            txns = self._retrieve()
            self._review(txns)
        except _TurnLimit:
            if self._phase == "finish":
                self._tr.ended = True
                self._bot({"act": "SUMMARIZE", "recorded": self._recorded_claims()}, {"act": "END"})
                raise _Stop()
            self._transfer("turn_limit", "R28")

    def _authenticate(self) -> None:
        tr = self._tr
        tr.rules_fired.update({"R1", "R2"})
        attempts, unclear = 0, 0
        acts: tuple = ({"act": "OPENING", "text": self.opening}, {"act": "ASK_AUTH"})
        while True:
            f = self._ask(*acts)
            if f.act == "REQUEST_HUMAN":
                self._transfer("customer_request", "R20")
            if f.act == "ASK_KB":
                acts = (self._answer_kb(f), {"act": "ASK_AUTH"})
                continue
            if f.act != "PROVIDE_CREDENTIAL":
                unclear += 1
                if unclear > self.limits.max_clarification_retries:
                    self._transfer("ambiguity_persists", "R23")
                tr.rules_fired.add("R22")
                acts = ({"act": "ASK_CLARIFY"}, {"act": "ASK_AUTH"})
                continue
            try:
                res = self._call("authenticate_customer", credential=f.get("credential"))
            except ToolError:
                self._transfer("auth_unavailable", "R5")
            if res.get("status") == "AUTHENTICATED":
                tr.rules_fired.add("R3")
                return
            attempts += 1
            if attempts >= self.limits.max_auth_attempts:
                self._transfer("auth_failed", "R4")
            acts = ({"act": "ASK_AUTH_RETRY"},)

    def _retrieve(self) -> list[dict[str, Any]]:
        tr = self._tr
        try:
            res = self._call("get_review_transactions")
        except ToolError:
            self._transfer("retrieve_unavailable", "R14")
        txns = res.get("transactions", [])
        tr.rules_fired.update({"R7", "R24"})
        within = [t for t in txns if t["age_days"] <= self.limits.lookback_days]
        if len(within) != len(txns) or any(t["age_days"] == self.limits.lookback_days for t in txns):
            tr.rules_fired.add("R30")
        if not within:
            tr.rules_fired.add("R8")
            self._phase = "finish"
            f = self._ask({"act": "EXPLAIN_NO_TXNS"}, {"act": "ASK_MORE_HELP"})
            while f.act == "ASK_KB":
                f = self._ask(self._answer_kb(f), {"act": "ASK_MORE_HELP"})
            if f.act == "REQUEST_HUMAN":
                self._transfer("customer_request", "R20")
            tr.ended = True
            self._bot({"act": "END"})
            raise _Stop()
        return within

    def _review(self, within: list[dict[str, Any]]) -> None:
        tr, L = self._tr, self.limits
        to_review = within[: L.max_transactions_per_session]
        more = len(within) > L.max_transactions_per_session
        prefix: list[dict[str, Any]] = []
        card_done = False
        for idx, txn in enumerate(to_review):
            tid = txn["transaction_id"]
            tr.presented.append(tid)
            tr.rules_fired.add("R6")
            final, card = self._capture(tid, prefix)
            prefix = []
            # record before advancing (R12); claim only on SUCCESS with matching echo (R13, R33)
            try:
                res = self._call("record_charge_response", transaction_id=tid, disposition=final)
            except ToolError:
                self._transfer("record_failed", "R14")
            tr.rules_fired.add("R12")
            if res.get("status") != "SUCCESS" or res.get("transaction_id") != tid:
                tr.rules_fired.update({"R13", "R33"} if res.get("transaction_id") != tid else {"R13"})
                self._transfer("record_failed", "R14")
            tr.claims_recorded[tid] = final
            prefix.append({"act": "CONFIRM_RECORDED", "txn": tid, "disposition": final})
            if final == "UNSURE":
                self._transfer("unsure_twice", "R11")
            if final == "UNRECOGNIZED" and card and not card_done:
                card_done = True
                prefix = self._card_action(card, prefix, remaining=idx < len(to_review) - 1)
        if more:
            tr.rules_fired.add("R31")
            self._bot(*prefix, {"act": "EXPLAIN_LIMIT"})
            self._transfer("transaction_limit", "R31")
        self._phase = "finish"
        tr.rules_fired.add("R19")
        f = self._ask(*prefix, {"act": "SUMMARIZE", "recorded": self._recorded_claims()},
                      {"act": "ASK_MORE_HELP"})
        while f.act == "ASK_KB":
            f = self._ask(self._answer_kb(f), {"act": "ASK_MORE_HELP"})
        if f.act == "REQUEST_HUMAN":
            self._transfer("customer_request", "R20")
        tr.ended = True
        self._bot({"act": "END"})
        raise _Stop()

    def _capture(self, tid: str, prefix: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        tr, L = self._tr, self.limits
        unsure, unclear = 0, 0
        acts = (*prefix, {"act": "PRESENT_TXN", "txn": tid})
        while True:
            f = self._ask(*acts)
            if f.act == "ASK_KB":
                acts = (self._answer_kb(f), {"act": "PRESENT_TXN", "txn": tid})
                continue
            if f.act == "REQUEST_HUMAN":
                self._transfer("customer_request", "R20")
            if f.act == "INFORM_DISPOSITION":
                d = f.get("disposition")
                if f.get("corrected_from"):
                    tr.rules_fired.add("R21")
                if d in ("RECOGNIZED", "UNRECOGNIZED"):
                    tr.rules_fired.add("R9")
                    card = {k: f.get(k) for k in ("card_action", "delivery") if f.get(k)}
                    return d, card
                if d == "UNSURE":
                    unsure += 1
                    tr.rules_fired.add("R10")
                    if unsure > L.max_unsure_retries:
                        return "UNSURE", {}
                    acts = ({"act": "PROVIDE_CONTEXT", "txn": tid},)
                    continue
            unclear += 1
            tr.rules_fired.add("R22")
            if unclear > L.max_clarification_retries:
                self._transfer("ambiguity_persists", "R23")
            acts = ({"act": "ASK_CLARIFY", "txn": tid},)

    def _card_action(self, card: dict[str, Any], prefix: list[dict[str, Any]], remaining: bool) -> list[dict[str, Any]]:
        tr = self._tr
        try:
            res = self._call("get_card_action_options")
        except ToolError:
            self._transfer("card_options_unavailable", "R32")
        action, delivery = card.get("card_action"), card.get("delivery")
        tr.rules_fired.add("R15")
        if action not in res.get("options", []):
            return prefix + [{"act": "INFORM_NO_CARD_ACTION"}]
        unclear = 0
        while True:
            acts = list(prefix)
            if action == "REPLACE_CARD" and delivery == "expedited_delivery":
                tr.rules_fired.add("R26")
                acts.append({"act": "DISCLOSE_FEE", "fee": self.limits.expedited_fee, "delivery": delivery})
            acts.append({"act": "CONFIRM_ACTION", "action": action, "delivery": delivery})
            prefix = []
            f = self._ask(*acts)
            if f.act == "CONFIRM":
                break
            if f.act == "REQUEST_CARD_ACTION":
                action = f.get("card_action") or action
                delivery = f.get("delivery") or delivery
                continue
            if f.act == "DENY":
                return [{"act": "ACK_DECLINED"}]
            if f.act == "REQUEST_HUMAN":
                self._transfer("customer_request", "R20")
            if f.act == "ASK_KB":
                prefix = [self._answer_kb(f)]
                continue
            unclear += 1
            if unclear > self.limits.max_clarification_retries:
                self._transfer("ambiguity_persists", "R23")
        tr.rules_fired.add("R16")
        try:
            res = self._call("submit_card_action", action=action, delivery=delivery)
        except ToolError:
            self._transfer("submit_failed", "R29")
        if res.get("status") != "SUCCESS" or res.get("action") != action:
            tr.rules_fired.add("R17")
            self._transfer("submit_failed", "R29")
        tr.rules_fired.add("R17")
        if remaining:
            tr.rules_fired.add("R18")
        tr.claims_actions.append((action, delivery))
        return [{"act": "ACTION_DONE", "action": action, "delivery": delivery}]
