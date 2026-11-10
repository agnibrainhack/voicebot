"""Driving a bot through a case and observing it.

The harness plays the customer (the reactive simulator answers whatever the bot asks), passes the bot's tool calls
to the mock bank, checks every turn as it happens, and returns a ``Trace``.

Parts of this file:
  1. ``JourneyRunner`` and the bot adapters (an in-process callable, an HTTP endpoint): the full multi-turn loop
     for a scenario, with realised (reworded) customer text, runtime monitoring and the speech check per turn.
  2. ``GraphRunner``: the same for a journey mined from a compact policy graph.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# =====================================================================================================================
# PART 1 - Drive a bot through a scenario and observe it as a Trace (book Ch. 9.3).
#
# The bot is an interface, not an implementation. Anything with
#
#     start() -> BotTurn(text, acts, ended)
#     respond(user_text) -> BotTurn(text, acts, ended)
#
# can be tested. ``acts`` (structured dialog acts) are optional: when a bot only
# returns text, ``BotActClassifier`` infers the act from the text so the user
# simulator knows what is being asked. Tool calls are observed at the mock backend
# (ToolEnvironment / ToolGateway), so components, outcome and trajectory are read
# from what the bot actually did, not from what it says it did.
#
# After every bot turn the runner also checks what the bot SAID (s07_checks/speech_*.py): the words are
# read on their own, every fact they state is checked against the tool results, the customer's turns
# and the GMS, and against the bot's own act labels (``extras["speech"]``, graded as ``c_speech``).
#
# Per user turn the runner:
#   1. asks the simulator for the next semantic frame given the bot's act,
#   2. realises it under the scenario's factor levels with the scenario's enrichment arm
#      (template or LLM; pinned seed / pinned proved text from the proof),
#   3. re-reads it with the FrameReader (runtime invariance monitor),
#   4. sends the text to the bot and times the reply (per-turn latency / length budgets).
# =====================================================================================================================

from ..s05_rewording.catalog import Scenario, SUTResult
from ..s04_cases.journey import UserSimulator
from ..s03_knowledge.questions import judge_kb_answer
from ..s04_cases.meaning import Ontology
from .tools import Trace, new_trace
from .reference import BotTurn, ReferenceBot
from .tools import FaultProfile, ToolEnvironment, ToolGateway
from ..s01_packs.names import N
from ..s01_packs import names as roles

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
        from ..s07_checks.speech_reading import plain
        t = plain(text).lower()                                  # "didn’t" reads as "didn't"
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
    """Runs a bot through one scenario: realises every customer turn under the scenario's levels, plays the
    customer, passes tool calls to the mock bank (with any injected faults), checks each turn (runtime monitor,
    speech check, budgets) and returns the result with its Trace.

    The customer is ``scripted`` or ``adaptive``. Scripted, it says the base journey's lines in order (the lines the
    customer says in the graph's own conversation, ``paths.base_path``, in the states they answer), whatever the bot
    replies: every bot turn meets exactly the input the graph met, and a deviation never changes what the customer
    says next. The customer plays its part to the end: when the bot ends the call early (hangs up, transfers) the
    remaining lines are still said and every bot reply recorded, and ``extras["bot_ended_at"]`` is the line before
    which the bot ended it (None: it did not end early). It stops when the base journey has no more lines. Adaptive,
    the customer answers whatever the bot actually asked (``UserSimulator.respond``) and stops when the bot ends.
    """
    store: Any
    realizer: Any
    reader: Any
    bot_factory: Callable[[Any], Any]
    use_structured_acts: bool = True
    classifier: BotActClassifier = field(default_factory=BotActClassifier)
    governance: Any = None            # governance.monitor.GovernanceConfig -> runtime gates on every run
    arm_realizers: dict[str, Any] | None = None   # enrichment arm -> realizer (scenario.metadata['arm'])
    budgets: Any = None               # evaluation.constraints.ConstraintBudget -> per-turn budgets in every row
    check_speech: bool = True         # read and verify every bot utterance (s07_checks/speech_*.py)
    customer: str = "adaptive"        # "scripted": the base journey's lines, whatever the bot says; "adaptive": answers the bot
    paths: Any = None                 # base journeys for the scripted customer (a validator: base_path, base_states)

    def __post_init__(self) -> None:
        self.ont = Ontology.from_store(self.store)
        from ..s01_packs.loading import current
        self.graph = current().graph                 # a compact pack: graph-generic items, outcomes and claims
        self.lookback = self.store.lookup_enm("limit", N.limit.lookback_days) if self.graph is None else None
        self.speech = None
        kg = getattr(self.store, "kg", None)
        if self.check_speech and kg is not None and kg.where("says"):
            from ..s07_checks.speech_reading import SpeechContract
            from ..s07_checks.speech_checks import UtteranceVerifier
            self.speech = UtteranceVerifier(self.store, SpeechContract.from_graph(kg))

    def run(self, scn: Scenario, faults: list[FaultProfile] | None = None,
            oracle_path: list | None = None, oracle_states: list | None = None) -> SUTResult:
        item = scn.base
        spec, fx = item.metadata["spec"], item.metadata["fixture"]
        env = ToolEnvironment(json.loads(json.dumps(fx)))
        tools = ToolGateway(env, list(faults)) if faults else env
        monitor = None
        if self.governance is not None:
            from ..s07_checks.runtime import GovernedTools, RuntimeMonitor
            if self.graph is not None:
                from ..s07_checks.runtime import GraphRuntimeMonitor as RuntimeMonitor  # noqa: N811
            monitor = RuntimeMonitor(self.store, self.reader, self.governance, run_id=scn.sid)
            tools = GovernedTools(tools, monitor)
        bot = self.bot_factory(tools)
        sim = UserSimulator(spec, fx, self.lookback)
        script = self._script(scn, oracle_path, oracle_states)   # scripted customer: the base journey's lines
        if script is not None and oracle_path is None:
            oracle_path = [(f, c) for f, c, _ in script]          # its lines are the proved, pinned wording
        if self.graph is None:
            merchants = {t["transaction_id"]: t["merchant"] for t in fx["transactions"]}
            by_id = {t["transaction_id"]: t for t in fx["transactions"]}
        else:
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
        ended_at = None                              # the customer line before which the bot ended the call
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
            if i == MAX_USER_TURNS or (turn.ended and script is None):
                break
            if turn.ended and ended_at is None and i < len(script):   # the bot hung up early: the customer plays on
                ended_at = i
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
            if script is not None:                   # the base journey's next line, whatever the bot just said
                if i >= len(script):
                    break
                frame, pctx, st = script[i]
                ctx = {**pctx, "state": st}
            else:
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
                    "customer": "scripted" if script is not None else "adaptive",
                    "script_lines": len(script) if script is not None else None,
                    "bot_ended_at": ended_at,
                    "bot_turn_stats": bot_stats, "arm": scn.metadata.get("arm", "template"),
                    "injected": list(getattr(tools, "injected", [])),
                    "governance": monitor.finish() if monitor is not None else None,
                    "speech": None if speech is None else {"summary": speech.summary(), "problems": speech.problems(),
                                                           "turns": [r.as_dict() for r in speech.reports]}})

    # ------------------------------------------------------------------ helpers
    def _script(self, scn: Scenario, oracle_path: list | None, oracle_states: list | None) -> list | None:
        """The scripted customer's lines for this scenario: [(frame, context, state)], or None (adaptive customer,
        or no base journey to script from)."""
        if self.customer != "scripted":
            return None
        path, states = oracle_path, oracle_states
        if self.paths is not None:
            path = path if path is not None else self.paths.base_path(scn.base)
            states = states if states is not None else self.paths.base_states(scn.base)
        if path is None:
            return None
        states = list(states or [])
        return [(f, dict(c or {}), states[i] if i < len(states) and states[i] is not None else (c or {}).get("state"))
                for i, (f, c) in enumerate(path)]

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
    """A bot factory for the charge-review reference bot (with the given planted defects)."""
    sentences = [s for _, s in store.kb_sentences]

    def factory(tools):
        from .reference import DEFAULT_CONFIG
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


# =====================================================================================================================
# PART 2 - Driving any bot through a journey mined from a compact policy graph (book Ch. 9.3).
#
# The same per-turn loop as s06_bots/harness.py: the simulator answers the bot's act with the next
# meaning card of the script, the card is realised under the scenario's factor levels (pinned seed
# from the proof), read back by the frame reader (runtime invariance monitor) and sent to the bot;
# tool calls are observed at the mock backend. What the bot *says* is read from its acts: success
# claims (states that ``claims`` a tool), outcome acts (states marked ``outcome``) and policy
# answers, so the trace is scored exactly as the oracle's.
# =====================================================================================================================

from ..s04_cases.factors import GraphSimulator
from ..s07_checks.scoring import GraphTrace


def _graph():
    from ..s01_packs.loading import current
    return current().graph


def item_index(fixture: dict[str, Any]) -> dict[str, dict]:
    """item id -> item, for every list of items a scripted tool returns (context for realisation)."""
    g = _graph()
    out = {}
    for t, sp in g.tools.items():
        for fld, rs in (sp.get("returns") or {}).items():
            if not isinstance(rs, dict) or "items" not in rs:
                continue
            for res in fixture.get(f"tool:{t}") or []:
                for it in (res.get(fld) or []) if isinstance(res, dict) else []:
                    out[str(it.get(rs["id"]))] = it
    return out


@dataclass
class GraphRunner(JourneyRunner):
    """The JourneyRunner for a journey mined from a compact policy graph."""
    def __post_init__(self) -> None:
        self.ont = Ontology.from_store(self.store)

    def run(self, scn: Scenario, faults: list[FaultProfile] | None = None,
            oracle_path: list | None = None, oracle_states: list | None = None) -> SUTResult:
        g = _graph()
        item = scn.base
        spec, fx = item.metadata["spec"], item.metadata["fixture"]
        env = ToolEnvironment(json.loads(json.dumps(fx)))
        tools = ToolGateway(env, list(faults)) if faults else env
        monitor = None
        if self.governance is not None:
            from ..s07_checks.runtime import GovernedTools
            from ..s07_checks.runtime import GraphRuntimeMonitor
            monitor = GraphRuntimeMonitor(self.store, self.reader, self.governance, run_id=scn.sid)
            tools = GovernedTools(tools, monitor)
        bot = self.bot_factory(tools)
        sim = GraphSimulator(spec, fx)
        script = self._script(scn, oracle_path, oracle_states)   # scripted customer: the base journey's lines
        if script is not None and oracle_path is None:
            oracle_path = [(f, c) for f, c, _ in script]          # its lines are the proved, pinned wording
        items = item_index(fx)
        overrides = scn.metadata.get("seed_overrides", {})
        realizer = (self.arm_realizers or {}).get(scn.metadata.get("arm", "template"), self.realizer)
        bot_stats: list[dict[str, Any]] = []
        tr = GraphTrace()
        transcript: list[dict[str, Any]] = []
        violations: list[dict[str, Any]] = []
        current = None
        last_kb = None
        t0 = time.perf_counter()
        ended_at = None                              # the customer line before which the bot ended the call
        turn = bot.start()
        bot_stats.append(_turn_stats(turn.text, t0))
        for i in range(MAX_USER_TURNS + 1):
            acts = turn.acts if (self.use_structured_acts and turn.acts) else [{"act": "UNKNOWN"}]
            tr.events.append({"role": "bot", "acts": acts, "text": turn.text})
            transcript.append({"role": "bot", "text": turn.text, "acts": [a.get("act") for a in acts]})
            if monitor is not None:
                monitor.on_bot(turn.text, turn.acts if self.use_structured_acts else None)
            self._observe(tr, acts, last_kb, turn.text)
            for a in acts:
                for v in a.values():
                    if isinstance(v, str) and v in items:
                        current = items[v]
            if i == MAX_USER_TURNS or (turn.ended and script is None):
                break
            if turn.ended and ended_at is None and i < len(script):   # the bot hung up early: the customer plays on
                ended_at = i
            act = dict(acts[-1])
            if script is not None:                   # the base journey's next line, whatever the bot just said
                if i >= len(script):
                    break
                frame, pctx, st = script[i]
                ctx = {**pctx, "state": st}
            else:
                frame = sim.respond(act)
                if frame is None:
                    break
                st = sim.last_state
                ctx = {**(current or {}), "state": st}
            pinned = oracle_path is not None and i < len(oracle_path) and oracle_path[i][0] == frame
            att = overrides.get(i, 0) if pinned else 0
            seed = f"{scn.sid}|{i}" + (f"#{att}" if att else "")
            text = realizer.realize(frame, scn.factor_levels, seed, ctx)
            read = self.reader.read(text, st)
            if not self.ont.equivalent(frame, read, st):
                violations.append({"turn": i, "state": st, "frame": str(frame), "text": text, "read": str(read)})
            tr.turns += 1
            tr.user_frames.append(frame)
            tr.events.append({"role": "user", "frame": frame, "text": text, "state": st})
            transcript.append({"role": "user", "text": text, "frame": str(frame), "state": st})
            last_kb = frame if frame.act == (g.glob.get("question") or {}).get("act") else None
            if monitor is not None:
                monitor.on_user(text, st)
            n0 = len(env.log)
            t0 = time.perf_counter()
            turn = bot.respond(text)
            bot_stats.append(_turn_stats(turn.text, t0))
            for c in env.log[n0:]:
                tr.events.append({"role": "tool", "tool": c["tool"], "args": c["args"], "result": c.get("result"),
                                  "ok": c.get("ok", True)})
        tr.tool_log = list(env.log)
        tr.ended = bool(turn.ended) and not tr.transferred
        comps = tr.components()
        status = "escalated" if tr.transferred else ("ok" if tr.ended else "failed")
        return SUTResult(
            answer=tr.signature(), components=comps,
            trajectory=[{"tool": c["tool"], "ok": c.get("ok", True)} for c in tr.tool_log],
            status=status, escalation_trigger=tr.transfer_reason,
            extras={"trace": tr, "transcript": transcript, "invariance_violations": violations,
                    "customer": "scripted" if script is not None else "adaptive",
                    "script_lines": len(script) if script is not None else None,
                    "bot_ended_at": ended_at,
                    "bot_turn_stats": bot_stats, "arm": scn.metadata.get("arm", "template"),
                    "injected": list(getattr(tools, "injected", [])),
                    "governance": monitor.finish() if monitor is not None else None})

    def _observe(self, tr: GraphTrace, acts: list[dict], last_kb, text: str) -> None:
        g = _graph()
        q = g.glob.get("question") or {}
        for a in acts:
            k = a.get("act")
            s = g.states.get(k) or {}
            if s.get("outcome"):
                tr.told.append(dict(a))
            if s.get("claims"):
                tr.claims.append((s["claims"], dict(a)))
            if k == q.get("answer") and last_kb is not None:
                ans = a.get("text") or text
                verdict = judge_kb_answer(self.store, last_kb, ans)
                tr.kb.append({"frame": str(last_kb), "answer": ans, **verdict})
