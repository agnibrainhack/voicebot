"""The policy oracle: expected behaviour, computed rather than annotated (Ch. 12.4).

``PolicyOracle`` is an executable reading of the policy. It walks the journey the domain
pack prescribes (domain.yaml -> flow) as a sequence of service-bot patterns:

  authenticate   ask for the credential, call the auth tool, retry up to a limit
  fetch          call the read tool, keep the items inside the review window
  per item       present it, capture the caller's answer (clarify / unsure retries),
                 write it with the write tool, confirm it, and run the follow-up action
                 (options -> disclose a figure -> confirm -> submit) once when the
                 answer calls for it; past the per-session cap, explain the limit
  finish         summarise what was written, ask whether anything else is needed, end
  handoff        any time: a request for a person, a failed or unavailable tool, a
                 retry limit or the turn limit hands over (with a summary first)

Every act name, tool, argument, rule id and limit comes from the domain pack, and every
limit and figure is read from the exact numeric register (lookup_enm). The oracle plays the
journey against the user simulator on the same backend fixture the bot will face; the
resulting ``Trace`` is the ground truth: outcome signature, per-stage decisions, which rules
fired and which workflow edges were traversed.

The oracle consumes semantic frames only. It never reads text, so its answer depends on what
the customer *meant*, which is exactly what the invariance proof (enrich/invariance.py) shows
enrichment leaves unchanged.
"""
from __future__ import annotations

from typing import Any

from .journey import JourneySpec, UserSimulator, build_fixture
from .kb_truth import kb_answer
from .ontology import SemanticFrame
from .sut.tools import ToolEnvironment, ToolError
from .trace import Trace


class _Stop(Exception):
    pass


class _TurnLimit(Exception):
    pass


class Limits:
    """Every limit of the policy graph plus the named figures (domain.yaml -> figures), read from
    exact numeric memory. Attributes are the limit / figure names (``limits.max_turns``)."""

    def __init__(self, **values: Any):
        self.__dict__.update(values)

    @classmethod
    def from_store(cls, store) -> "Limits":
        from .domain import current
        dom = current()
        cat = dom.get("figures.limits.enm_category", "limit")
        vals = {k: store.lookup_enm(cat, k) for k in dom.limit_keys()}
        for name, sp in (dom.get("figures.named") or {}).items():
            vals[name] = store.lookup_enm(sp["enm_category"], sp["key"])
        return cls(**vals)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class PolicyOracle:
    def __init__(self, store):
        from .domain import current
        dom = current()
        self.store = store
        self.flow = dom.section("flow")
        self.tools = dom.section("tools")
        items = dom.get("fixtures.items") or {}
        self.id_field, self.age_field = items.get("id_field"), items.get("age_field")
        self.limits = Limits.from_store(store)
        self.opening = store.required_language.get(self.flow["opening"]["text_from"], "")

    def L(self, name: str) -> Any:
        return getattr(self.limits, name)

    def _ok(self, tool: str) -> str:
        sp = self.tools[tool]
        return sp["check"]["success"] if sp.get("check") else sp["status"]["ok"]

    def run(self, spec: JourneySpec, fixture: dict[str, Any] | None = None,
            frames_override: list[SemanticFrame] | None = None,
            gateway_factory=None) -> Trace:
        """Play the journey. `frames_override` replays a fixed frame sequence instead of
        the simulator (used to re-derive ground truth from an enriched transcript)."""
        fixture = fixture or build_fixture(spec)
        env = ToolEnvironment(fixture)
        tools = gateway_factory(env) if gateway_factory else env
        sim = UserSimulator(spec, fixture, self.L(self.flow["fetch"]["window"]["limit"]))
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
        if self._tr.turns >= self.L(self.flow["turn_limit"]["limit"]):
            raise _TurnLimit()
        self._bot(*acts)
        if self._replay is not None:
            frame = self._replay.pop(0) if self._replay else None
        else:
            frame = self._sim.respond(acts[-1])
        if frame is None:
            self._tr.notes.append(self.flow["silent_note"])
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
        h = self.flow["handoff"]
        self._tr.rules_fired.add(rule)
        acts = []
        rec = self._recorded_claims()
        if rec:
            self._tr.rules_fired.add(h["summary"]["rule"])
            acts.append({"act": h["summary"]["act"], h["summary"]["field"]: rec})
        acts.append({"act": h["act"], "reason": reason})
        self._bot(*acts)
        arg = next(iter(self.tools[h["tool"]].get("args") or {"reason": ""}))
        try:
            self._call(h["tool"], **{arg: reason})
        except ToolError:
            pass
        raise _Stop()

    def _give_up(self, spec: dict[str, str]) -> None:
        self._transfer(spec["reason"], spec["rule"])

    def _recorded_claims(self) -> dict[str, str]:
        return dict(self._tr.claims_recorded)

    def _answer_kb(self, frame: SemanticFrame) -> dict[str, Any]:
        q = self.flow["question"]
        truth = kb_answer(self.store, frame)
        self._tr.kb.append({"frame": frame, "answer": truth.answer, "correct": True})
        self._tr.rules_fired.update(set(q["rules"]) | set((q.get("intent_rules") or {}).get(frame.get("intent"), [])))
        return {"act": q["answer"], "intent": frame.get("intent"), "answer": truth.answer}

    def _is_human(self, f) -> bool:
        return f.act == self.flow["human"]["act"]

    def _to_human(self) -> None:
        self._transfer(self.flow["human"]["reason"], self.flow["human"]["rule"])

    def _is_question(self, f) -> bool:
        return f.act == self.flow["question"]["act"]

    # ----------------------------------------------------------------- journey
    def _journey(self) -> None:
        try:
            self._authenticate()
            items = self._fetch()
            self._per_item(items)
        except _TurnLimit:
            if self._phase == "finish":
                fin = self.flow["finish"]
                self._tr.ended = True
                self._bot({"act": fin["summary"]["act"], fin["summary"]["field"]: self._recorded_claims()},
                          {"act": fin["end"]})
                raise _Stop()
            tl = self.flow["turn_limit"]
            self._transfer(tl["reason"], tl["rule"])

    def _authenticate(self) -> None:
        tr, A, cl = self._tr, self.flow["authenticate"], self.flow["clarify"]
        op = self.flow["opening"]
        tr.rules_fired.update(op["rules"])
        attempts, unclear = 0, 0
        acts: tuple = ({"act": op["act"], "text": self.opening}, {"act": A["ask"]})
        while True:
            f = self._ask(*acts)
            if self._is_human(f):
                self._to_human()
            if self._is_question(f):
                acts = (self._answer_kb(f), {"act": A["ask"]})
                continue
            if f.act != A["act"]:
                unclear += 1
                if unclear > self.L(cl["limit"]):
                    self._give_up(cl["give_up"])
                tr.rules_fired.add(cl["rule"])
                acts = ({"act": cl["act"]}, {"act": A["ask"]})
                continue
            try:
                res = self._call(A["tool"], **{arg: f.get(slot) for arg, slot in A["args"].items()})
            except ToolError:
                self._give_up(A["unavailable"])
            if res.get("status") == self._ok(A["tool"]):
                tr.rules_fired.update(A["success_rules"])
                return
            attempts += 1
            if attempts >= self.L(A["limit"]):
                self._give_up(A["exhausted"])
            acts = ({"act": A["retry"]},)

    def _fetch(self) -> list[dict[str, Any]]:
        tr, Fe = self._tr, self.flow["fetch"]
        try:
            res = self._call(Fe["tool"])
        except ToolError:
            self._give_up(Fe["unavailable"])
        items = res.get(self.tools[Fe["tool"]]["key"], [])
        tr.rules_fired.update(Fe["rules"])
        look = self.L(Fe["window"]["limit"])
        within = [t for t in items if t[self.age_field] <= look]
        if len(within) != len(items) or any(t[self.age_field] == look for t in items):
            tr.rules_fired.add(Fe["window"]["rule"])
        if not within:
            tr.rules_fired.add(Fe["empty"]["rule"])
            self._phase = "finish"
            more = self.flow["finish"]["more_help"]
            f = self._ask({"act": Fe["empty"]["act"]}, {"act": more})
            self._finish_reply(f)
        return within

    def _finish_reply(self, f) -> None:
        """The caller's reply to 'anything else?': answer questions, hand over on request, end."""
        more = self.flow["finish"]["more_help"]
        while self._is_question(f):
            f = self._ask(self._answer_kb(f), {"act": more})
        if self._is_human(f):
            self._to_human()
        self._tr.ended = True
        self._bot({"act": self.flow["finish"]["end"]})
        raise _Stop()

    def _per_item(self, within: list[dict[str, Any]]) -> None:
        tr, P = self._tr, self.flow["per_item"]
        W, FU = P["write"], P.get("follow_up")
        cap = self.L(P["cap"]["limit"])
        to_review = within[:cap]
        more = len(within) > cap
        prefix: list[dict[str, Any]] = []
        follow_done = False
        for idx, item in enumerate(to_review):
            tid = item[self.id_field]
            tr.presented.append(tid)
            tr.rules_fired.add(P["present"]["rule"])
            final, request = self._capture(tid, prefix)
            prefix = []
            # write before advancing; claim only on success with a matching echo
            try:
                res = self._call(W["tool"], **{W["args"]["item"]: tid, W["args"]["value"]: final})
            except ToolError:
                self._give_up(W["failed"])
            tr.rules_fired.add(W["rule"])
            if res.get("status") != self._ok(W["tool"]) or res.get(W["echo"]) != tid:
                tr.rules_fired.update(W["failed"]["echo_mismatch_rules"] if res.get(W["echo"]) != tid
                                      else W["failed"]["rules"])
                self._give_up(W["failed"])
            tr.claims_recorded[tid] = final
            prefix.append({"act": W["confirm"]["act"], W["confirm"]["item_field"]: tid,
                           W["confirm"]["value_field"]: final})
            if final == W["terminal"]["value"]:
                self._give_up(W["terminal"])
            if FU and final == FU["when_value"] and request and not follow_done:
                follow_done = True
                prefix = self._follow_up(request, prefix, remaining=idx < len(to_review) - 1)
        if more:
            tr.rules_fired.add(P["cap"]["rule"])
            self._bot(*prefix, {"act": P["cap"]["act"]})
            self._give_up(P["cap"])
        self._phase = "finish"
        fin = self.flow["finish"]
        tr.rules_fired.add(fin["summary"]["rule"])
        f = self._ask(*prefix, {"act": fin["summary"]["act"], fin["summary"]["field"]: self._recorded_claims()},
                      {"act": fin["more_help"]})
        self._finish_reply(f)

    def _capture(self, tid: str, prefix: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        tr, P, cl = self._tr, self.flow["per_item"], self.flow["clarify"]
        C, pres = P["capture"], P["present"]
        unsure, unclear = 0, 0
        acts = (*prefix, {"act": pres["act"], pres["field"]: tid})
        while True:
            f = self._ask(*acts)
            if self._is_question(f):
                acts = (self._answer_kb(f), {"act": pres["act"], pres["field"]: tid})
                continue
            if self._is_human(f):
                self._to_human()
            if f.act == C["act"]:
                d = f.get(C["slot"])
                if f.get("corrected_from"):
                    tr.rules_fired.add(C["correction_rule"])
                if d in C["clear"]:
                    tr.rules_fired.add(C["clear_rule"])
                    request = {k: f.get(k) for k in C["request_slots"] if f.get(k)}
                    return d, request
                U = C["unsure"]
                if d == U["value"]:
                    unsure += 1
                    tr.rules_fired.add(U["rule"])
                    if unsure > self.L(U["limit"]):
                        return U["value"], {}
                    acts = ({"act": U["act"], pres["field"]: tid},)
                    continue
            unclear += 1
            tr.rules_fired.add(cl["rule"])
            if unclear > self.L(cl["limit"]):
                self._give_up(cl["give_up"])
            acts = ({"act": cl["act"], pres["field"]: tid},)

    def _follow_up(self, request: dict[str, Any], prefix: list[dict[str, Any]], remaining: bool) -> list[dict[str, Any]]:
        tr, FU = self._tr, self.flow["per_item"]["follow_up"]
        O, D, C, S = FU["options"], FU["disclose"], FU["confirm"], FU["submit"]
        try:
            res = self._call(O["tool"])
        except ToolError:
            self._give_up(O["unavailable"])
        action, option = request.get(FU["action_slot"]), request.get(FU["option_slot"])
        tr.rules_fired.add(O["rule"])
        if action not in res.get(O["key"], []):
            return prefix + [{"act": FU["not_offered"]["act"]}]
        unclear = 0
        while True:
            acts = list(prefix)
            if all({FU["action_slot"]: action, FU["option_slot"]: option}.get(k) == v for k, v in D["when"].items()):
                tr.rules_fired.add(D["rule"])
                acts.append({"act": D["act"], D["figure_field"]: self.L(D["figure"]), D["option_field"]: option})
            acts.append({"act": C["act"], C["action_field"]: action, C["option_field"]: option})
            prefix = []
            f = self._ask(*acts)
            if f.act == C["accept"]:
                break
            if f.act == C["change"]:
                action = f.get(FU["action_slot"]) or action
                option = f.get(FU["option_slot"]) or option
                continue
            if f.act == C["decline"]:
                return [{"act": FU["declined"]["act"]}]
            if self._is_human(f):
                self._to_human()
            if self._is_question(f):
                prefix = [self._answer_kb(f)]
                continue
            unclear += 1
            if unclear > self.L(C["limit"]):
                self._give_up(self.flow["clarify"]["give_up"])
        tr.rules_fired.add(S["rule"])
        try:
            res = self._call(S["tool"], **{S["args"]["action"]: action, S["args"]["option"]: option})
        except ToolError:
            self._give_up(S["failed"])
        if res.get("status") != self._ok(S["tool"]) or res.get(S["echo"]) != action:
            tr.rules_fired.add(S["done_rule"])
            self._give_up(S["failed"])
        tr.rules_fired.add(S["done_rule"])
        if remaining:
            tr.rules_fired.add(S["remaining_rule"])
        tr.claims_actions.append((action, option))
        done = FU["done"]
        return [{"act": done["act"], done["action_field"]: action, done["option_field"]: option}]
