"""Drive a bot through a scenario and observe it as a Trace (book Ch. 9.3).

The bot is an interface, not an implementation. Anything with

    start() -> BotTurn(text, acts, ended)
    respond(user_text) -> BotTurn(text, acts, ended)

can be tested. ``acts`` (structured dialog acts) are optional: when a bot only
returns text, ``BotActClassifier`` infers the act from the text so the user
simulator knows what is being asked. Tool calls are observed at the mock backend
(ToolEnvironment / ToolGateway), so components, outcome and trajectory are read
from what the bot actually did, not from what it says it did.

After every bot turn the runner also checks what the bot SAID (gmsjourney/speech): the words are
read on their own, every fact they state is checked against the tool results, the customer's turns
and the GMS, and against the bot's own act labels (``extras["speech"]``, graded as ``c_speech``).

Per user turn the runner:
  1. asks the simulator for the next semantic frame given the bot's act,
  2. realises it under the scenario's factor levels with the scenario's enrichment arm
     (template or LLM; pinned seed / pinned proved text from the proof),
  3. re-reads it with the FrameReader (runtime invariance monitor),
  4. sends the text to the bot and times the reply (per-turn latency / length budgets).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core import Scenario, SUTResult
from ..journey import UserSimulator
from ..kb_truth import judge_kb_answer
from ..ontology import Ontology
from ..trace import Trace, new_trace
from .reference_bot import BotTurn, ReferenceBot
from .tools import FaultProfile, ToolEnvironment, ToolGateway
from ..names import N
from .. import roles

MAX_USER_TURNS = 45


class BotActClassifier:
    """Infer the dialog act of a text-only bot turn (for bots without structured acts)."""
    @property
    def PATTERNS(self) -> list[tuple[str, str]]:                 # noqa: N802 (read when classifying)
        return [
            (N.state.TRANSFER, r"\b(transfer|connect you|representative will)\b"),
            (N.state.ASK_AUTH_RETRY, r"\b(didn't match|try again|not match)\b"),
            (N.state.ASK_AUTH, r"\b(last four|verify|authenticate|digits)\b"),
            (N.state.DISCLOSE_FEE, r"\$\s?\d+.*\bfee\b|\bfee\b.*\$\s?\d+"),
            (N.state.CONFIRM_ACTION, r"\b(shall i|should i|go ahead\?|confirm)\b"),
            (N.state.PROVIDE_CONTEXT, r"\bdays ago\b"),
            (N.state.PRESENT_TXN, r"\b(do you recognize|recognise)\b"),
            (N.state.ASK_CLARIFY, r"\b(didn't catch|say that again|sorry\?)\b"),
            (N.state.EXPLAIN_NO_TXNS, r"\bno (charges|transactions)\b"),
            (N.state.ASK_MORE_HELP, r"\b(anything else)\b"),
            (N.state.END, r"\b(goodbye|bye)\b"),
        ]

    def classify(self, text: str, merchants: dict[str, str]) -> dict[str, Any]:
        t = text.lower()
        act = "UNKNOWN"
        for name, pat in self.PATTERNS:
            if re.search(pat, t):
                act = name
                break
        txn = None
        for tid, m in merchants.items():
            if m.lower() in t:
                txn = tid
        out = {"act": act}
        if txn:
            out["txn"] = txn
        if act in (N.state.DISCLOSE_FEE, N.state.CONFIRM_ACTION) and re.search(r"expedited|express|rush", t):
            out["delivery"] = roles.FEE_OPTION
        return out


@dataclass
class JourneyRunner:
    store: Any
    realizer: Any
    reader: Any
    bot_factory: Callable[[Any], Any]
    use_structured_acts: bool = True
    classifier: BotActClassifier = field(default_factory=BotActClassifier)
    governance: Any = None            # governance.monitor.GovernanceConfig -> runtime gates on every run
    arm_realizers: dict[str, Any] | None = None   # enrichment arm -> realizer (scenario.metadata['arm'])
    budgets: Any = None               # evaluation.constraints.ConstraintBudget -> per-turn budgets in every row
    check_speech: bool = True         # read and verify every bot utterance (gmsjourney/speech)

    def __post_init__(self) -> None:
        self.ont = Ontology.from_store(self.store)
        from ..domain import current
        self.graph = current().graph                 # a compact pack: graph-generic items, outcomes and claims
        self.lookback = self.store.lookup_enm("limit", N.limit.lookback_days) if self.graph is None else None
        self.speech = None
        kg = getattr(self.store, "kg", None)
        if self.check_speech and kg is not None and kg.where("says"):
            from ..speech import SpeechContract, UtteranceVerifier
            self.speech = UtteranceVerifier(self.store, SpeechContract.from_graph(kg))

    def run(self, scn: Scenario, faults: list[FaultProfile] | None = None,
            oracle_path: list | None = None) -> SUTResult:
        item = scn.base
        spec, fx = item.metadata["spec"], item.metadata["fixture"]
        env = ToolEnvironment(json.loads(json.dumps(fx)))
        tools = ToolGateway(env, list(faults)) if faults else env
        monitor = None
        if self.governance is not None:
            from ..governance.monitor import GovernedTools, RuntimeMonitor
            if self.graph is not None:
                from ..graph.governance import GraphRuntimeMonitor as RuntimeMonitor   # noqa: N811
            monitor = RuntimeMonitor(self.store, self.reader, self.governance, run_id=scn.sid)
            tools = GovernedTools(tools, monitor)
        bot = self.bot_factory(tools)
        sim = UserSimulator(spec, fx, self.lookback)
        if self.graph is None:
            merchants = {t["transaction_id"]: t["merchant"] for t in fx["transactions"]}
            by_id = {t["transaction_id"]: t for t in fx["transactions"]}
        else:
            from ..graph.runner import item_index
            by_id = item_index(fx)
            merchants = {}
        overrides = scn.metadata.get("seed_overrides", {})
        realizer = (self.arm_realizers or {}).get(scn.metadata.get("arm", "template"), self.realizer)
        bot_stats: list[dict[str, Any]] = []
        tr = new_trace()
        transcript: list[dict[str, Any]] = []
        violations: list[dict[str, Any]] = []
        current_txn = None
        last_kb = None
        speech = self.speech.conversation(fx.get("session_date")) if self.speech is not None else None
        t0 = time.perf_counter()
        turn: BotTurn = bot.start()
        bot_stats.append(_turn_stats(turn.text, t0))
        for i in range(MAX_USER_TURNS + 1):
            acts = turn.acts if (self.use_structured_acts and turn.acts) else \
                ([self.classifier.classify(turn.text, merchants)] if self.graph is None else [{"act": "UNKNOWN"}])
            said = None
            if speech is not None:           # what the customer hears, checked before they answer
                said = speech.check(turn.acts if self.use_structured_acts else [], turn.text, env.log, tr.user_frames)
            tr.events.append({"role": "bot", "acts": acts, "text": turn.text})
            transcript.append({"role": "bot", "text": turn.text, "acts": [a.get("act") for a in acts]})
            if monitor is not None:
                monitor.on_bot(turn.text, turn.acts if self.use_structured_acts else None)
                monitor.on_speech(said)      # the speech check's findings, sealed live into the audit
            self._observe_acts(tr, acts, last_kb, turn.text)
            if turn.ended or i == MAX_USER_TURNS:
                break
            act = dict(acts[-1])
            if self.graph is None:                    # one review per item: the customer answers about the current one
                for a in acts:
                    if a.get("act") == N.state.PRESENT_TXN and a.get("txn"):
                        current_txn = a["txn"]
                act.setdefault("current_txn", current_txn)
                if act.get("act") in (N.state.ASK_CLARIFY, N.state.ANSWER_KB, N.state.PROVIDE_CONTEXT) and not act.get("txn"):
                    act["txn"] = current_txn
            else:                                     # an act naming a returned item makes it the current one
                for a in acts:
                    for v in a.values():
                        if isinstance(v, str) and v in by_id:
                            current_txn = v
            frame = sim.respond(act)
            if frame is None:
                break
            t = by_id.get(act.get("txn") or current_txn, {})
            st = sim.last_state if self.graph is not None else act.get("act")   # the dialog state this turn answers
            ctx = {**t, "state": st} if self.graph is not None else \
                {"merchant": t.get("merchant"), "amount": t.get("amount"), "state": st}
            pinned = oracle_path is not None and i < len(oracle_path) and oracle_path[i][0] == frame
            att = overrides.get(i, 0) if pinned else 0
            seed = f"{scn.sid}|{i}" + (f"#{att}" if att else "")
            text = realizer.realize(frame, scn.factor_levels, seed, ctx)
            read = self.reader.read(text, st)
            if not self.ont.equivalent(frame, read, st):
                violations.append({"turn": i, "state": st, "frame": str(frame), "text": text, "read": str(read)})
            tr.turns += 1
            tr.user_frames.append(frame)
            tr.events.append({"role": "user", "frame": frame, "text": text})
            transcript.append({"role": "user", "text": text, "frame": str(frame), "state": st})
            last_kb = frame if frame.act == N.act.ASK_KB else None
            if monitor is not None:
                monitor.on_user(text, st)
            n0 = len(env.log)
            t0 = time.perf_counter()
            turn = bot.respond(text)
            bot_stats.append(_turn_stats(turn.text, t0))
            for c in env.log[n0:]:          # tool calls caused by this user turn, in order
                tr.events.append({"role": "tool", "tool": c["tool"], "args": c["args"], "result": c.get("result")})
        tr.tool_log = list(env.log)
        tr.ended = bool(turn.ended) and not tr.transferred
        if hasattr(tr, "finish"):
            tr.finish()                              # outcome acts and success claims, read from what was said
        comps = tr.components()
        status = "escalated" if tr.transferred else ("ok" if tr.ended else "failed")
        return SUTResult(
            answer=tr.signature(), components=comps,
            trajectory=[{"tool": c["tool"], "ok": c.get("ok", True)} for c in tr.tool_log],
            status=status, escalation_trigger=tr.transfer_reason,
            extras={"trace": tr, "transcript": transcript, "invariance_violations": violations,
                    "bot_turn_stats": bot_stats, "arm": scn.metadata.get("arm", "template"),
                    "injected": list(getattr(tools, "injected", [])),
                    "governance": monitor.finish() if monitor is not None else None,
                    "speech": None if speech is None else {"summary": speech.summary(), "problems": speech.problems(),
                                                           "turns": [r.as_dict() for r in speech.reports]}})

    # ------------------------------------------------------------------ helpers
    def _observe_acts(self, tr: Trace, acts: list[dict], last_kb, text: str) -> None:
        if self.graph is not None:                   # outcomes and claims are read at the end (trace.finish)
            for a in acts:
                if a.get("act") == N.state.ANSWER_KB and last_kb is not None:
                    ans = a.get("text", text)
                    tr.kb.append({"frame": str(last_kb), "answer": ans, **judge_kb_answer(self.store, last_kb, ans)})
            return
        for a in acts:
            k = a.get("act")
            if k == N.state.PRESENT_TXN and a.get("txn") and (not tr.presented or tr.presented[-1] != a["txn"]) \
                    and a["txn"] not in tr.presented:
                tr.presented.append(a["txn"])
            elif k == N.state.CONFIRM_RECORDED and a.get("txn"):
                tr.claims_recorded[a["txn"]] = a.get("disposition")
            elif k == N.state.SUMMARIZE:
                for t, d in (a.get("recorded") or {}).items():
                    tr.claims_recorded[t] = d
                for act in a.get("actions") or []:
                    tr.claims_actions.append(tuple(act))
            elif k == N.state.ACTION_DONE:
                tr.claims_actions.append((a.get("action"), a.get("delivery")))
            elif k == N.state.ANSWER_KB and last_kb is not None:
                ans = a.get("text", text)
                verdict = judge_kb_answer(self.store, last_kb, ans)
                tr.kb.append({"frame": str(last_kb), "answer": ans, **verdict})


def _turn_stats(text: str, t0: float) -> dict[str, Any]:
    return {"latency_ms": (time.perf_counter() - t0) * 1000.0, "words": len((text or "").split()),
            "chars": len(text or "")}


def make_reference_factory(store, mutations: frozenset = frozenset(), config: dict | None = None):
    sentences = [s for _, s in store.kb_sentences]

    def factory(tools):
        from .reference_bot import DEFAULT_CONFIG
        return ReferenceBot(tools, sentences, dict(config or DEFAULT_CONFIG), frozenset(mutations))
    return factory


class CallableBotAdapter:
    """Wrap your own bot: provide start() and respond(text) callables that return text
    (and optionally a list of act dicts). Tool calls must go to the `tools` object the
    factory receives, so point your bot's tool client at it."""

    def __init__(self, start_fn: Callable[[], Any], respond_fn: Callable[[str], Any]):
        self._start, self._respond = start_fn, respond_fn

    @staticmethod
    def _wrap(out: Any) -> BotTurn:
        if isinstance(out, BotTurn):
            return out
        if isinstance(out, dict):
            return BotTurn(out.get("text", ""), out.get("acts", []), bool(out.get("ended")))
        return BotTurn(str(out), [], False)

    def start(self) -> BotTurn:
        return self._wrap(self._start())

    def respond(self, text: str) -> BotTurn:
        return self._wrap(self._respond(text))


class HTTPBotAdapter:
    """Drive a deployed bot over HTTP. Expects POST {session, text} -> {text, acts?, ended?}.
    The bot's backend must be pointed at the mock tools for the session (fixture id)."""

    def __init__(self, url: str, session: str, timeout: float = 30.0):
        self.url, self.session, self.timeout = url, session, timeout

    def _post(self, payload: dict) -> BotTurn:
        import urllib.request
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return CallableBotAdapter._wrap(json.loads(r.read()))

    def start(self) -> BotTurn:
        return self._post({"session": self.session, "event": "start"})

    def respond(self, text: str) -> BotTurn:
        return self._post({"session": self.session, "text": text})
