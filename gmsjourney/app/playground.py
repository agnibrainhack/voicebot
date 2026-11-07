"""The playground: talk to the bot under test yourself, one turn at a time, with every turn checked as you go.

A session is one conversation on one case. The case sets the mock bank (the items, which tools fail or decline) and
offers its customer lines - the scripted words of each turn and, for cases of a finished run, the reworded versions
that were proved to mean the same. The tester clicks a line or types their own. For every bot turn the session
shows:

* the tool calls the bot made, with their results (the mock bank's log);
* what the graph expects at this turn: the oracle replays the customer's turns so far (each read into its meaning
  card by the frame reader) and gives the tools it would call and the acts it would say; the turn matches when the
  tools (with their key arguments) and the decision acts are the same (turn_compare's rules); after the first
  mismatch the conversations may legitimately differ, so later turns are marked "after divergence";
* the speech check of what the bot said (claims supported, contradicted or unsupported; policy statements;
  findings such as a missing or unfaithful act), as the campaign grades it.

Everything runs on the journey's symbolic store, compiled from the graph in about a second: no knowledge-store
training and no licence. Its figures and rules are the graph's own, so the checks agree with a campaign's on what is
true; the campaign's trained store adds the geometry checks on top.

Cases come from a finished run of the journey (``out/case_specs.jsonl``; older runs are rebuilt from report.json
and kept only where the rebuilt case has the run's exact expected outcome) or, without a run, from the project:
the default case and one case per factor level.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_SESSIONS = 24
MAX_TURNS = 60


# ----------------------------------------------------------------------------- the journey's symbolic world
class Bench:
    """The active journey's store, oracle, reader, realizer and speech verifier (built once per journey)."""

    def __init__(self) -> None:
        from ..domain import current
        from ..enrich.reader import FrameReader
        from ..enrich.realizers import Realizer
        from ..ontology import Ontology
        from ..oracle import PolicyOracle
        from ..packstore import symbolic_store
        self.dom = current()
        self.pack = self.dom.name
        self.store = symbolic_store()
        self.oracle = PolicyOracle(self.store)
        self.reader = FrameReader(self.store)
        self.realizer = Realizer()
        self.ont = Ontology.from_store(self.store)
        self.speech = None
        kg = getattr(self.store, "kg", None)
        if kg is not None and kg.where("says"):
            from ..speech import SpeechContract, UtteranceVerifier
            self.speech = UtteranceVerifier(self.store, SpeechContract.from_graph(kg))
        self.cases_cache: dict[str, list[dict]] = {}
        self.dropped: dict[str, int] = {}

    def reference_factory(self):
        if self.dom.graph is None:
            from ..sut.adapter import make_reference_factory
            return make_reference_factory(self.store), "the journey's reference bot"
        from ..graph.bot import make_graph_factory
        return make_graph_factory(self.store), "the journey's reference bot (follows the graph)"


_BENCHES: dict[str, Bench] = {}


def bench(fresh: bool = False) -> Bench:
    """The active journey's bench (call with the journey active)."""
    from ..domain import current
    name = current().name
    if fresh or name not in _BENCHES:
        _BENCHES[name] = Bench()
    return _BENCHES[name]


# ----------------------------------------------------------------------------- cases
def _frame(d: dict):
    from ..ontology import SemanticFrame as F
    return F.make(d["act"], **(d.get("slots") or {}))


def _lines(B: Bench, qid: str, spec, fx) -> tuple[list[dict], Any]:
    """The scripted customer lines of a case (the original wording, as the campaign realises it) and its trace."""
    tr = B.oracle.run(spec, fx)
    out, i = [], 0
    for ev in tr.events:
        if ev.get("role") != "user":
            continue
        fr = ev["frame"]
        st = ev.get("state")
        ctx = {**(ev.get("item") or {}), "state": st}
        try:
            text = B.realizer.realize(fr, {}, f"{qid}~base|{i}", ctx)
        except Exception:                               # noqa: BLE001 - a frame the realiser cannot word
            text = str(fr)
        card = B.ont.card(fr, st) if st else {"act": fr.act, "slots": dict(fr.slots)}
        out.append({"turn": i, "state": st, "frame": str(fr), "card": {"act": card.get("act"), "slots": card.get("slots")},
                    "text": text, "reworded": []})
        i += 1
    return out, tr


def _case(B: Bench, qid: str, label: str, spec, category: str | None = None, source: str = "",
          reworded: dict[int, list[dict]] | None = None) -> dict:
    from ..journey import build_fixture, primary_category
    fx = build_fixture(spec)
    lines, tr = _lines(B, qid, spec, fx)
    for ln in lines:
        ln["reworded"] = (reworded or {}).get(ln["turn"], [])
    sig = json.loads(tr.signature())
    return {"id": qid, "label": label, "category": category or primary_category(spec), "source": source,
            "spec": spec, "fixture": fx, "lines": lines, "expected": sig, "turns": len(lines),
            "rules": sorted(tr.rules_fired), "levels": {k: v for k, v in spec.dims().items()}}


def factor_cases(B: Bench) -> list[dict]:
    """Without a run: the default case, then one case per non-default factor level (as Preview shows them)."""
    from ..journey import JourneySpec, is_valid
    from ..project import Project
    proj = Project.load(B.dom.project_path)
    out = [_case(B, "default", "Every factor at its default level", JourneySpec(), source="project")]
    for name, f in proj.factors.items():
        for lv in f.levels:
            if lv == f.default:
                continue
            try:
                sp = JourneySpec(**{name: lv})
            except TypeError:
                continue
            if not is_valid(sp):
                continue
            try:
                out.append(_case(B, f"{name}={lv}", f"{name} = {lv}", sp, source="project"))
            except Exception:                           # noqa: BLE001 - a level the oracle cannot play alone
                continue
    return out


def run_cases(B: Bench, out_dir: Path) -> list[dict]:
    """The generated cases of a finished run, rebuilt exactly (case_specs.jsonl) or from report.json and kept only
    where the rebuilt case reproduces the run's expected outcome. Reworded versions come from the run's dataset."""
    from ..journey import DIMENSIONS, JourneySpec, policy_question_case
    out_dir = Path(out_dir)
    tc = [json.loads(x) for x in (out_dir / "dataset" / "test_cases.jsonl").read_text(encoding="utf-8").splitlines()
          if x.strip()] if (out_dir / "dataset" / "test_cases.jsonl").exists() else []
    rew: dict[str, dict[int, list[dict]]] = {}
    base_rows: dict[str, dict] = {}
    for c in tc:
        if c["case_type"] == "base":
            base_rows[c["base_id"]] = c
            continue
        if not c.get("proof_accepted"):
            continue
        changed = c.get("enrichment_changed") or ""
        for s in c.get("trajectory") or []:
            if s.get("actor") == "customer" and s.get("text"):
                rew.setdefault(c["base_id"], {}).setdefault(s["turn"], []).append(
                    {"case": c["case_id"], "text": s["text"], "how": changed})
    specs = []
    if (out_dir / "case_specs.jsonl").exists():
        for x in (out_dir / "case_specs.jsonl").read_text(encoding="utf-8").splitlines():
            if not x.strip():
                continue
            d = json.loads(x)
            lv = {k: v for k, v in d["levels"].items() if k in DIMENSIONS}
            specs.append((d["qid"], d.get("category"), JourneySpec(kb_frames=tuple(_frame(f) for f in d.get("kb_frames") or []),
                                                                   extra_kb=int(d.get("extra_kb") or 0), **lv), d.get("signature")))
    else:
        R = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        for b in R.get("bases") or []:
            row = base_rows.get(b["qid"]) or {}
            card = row.get("policy_question_card")
            kb = (_frame(card),) if card else ()
            lv = {k: v for k, v in (b.get("relevant") or {}).items() if k in DIMENSIONS}
            try:
                sp = policy_question_case(kb[0]) if (kb and not lv) else JourneySpec(kb_frames=kb, **lv)
            except (TypeError, IndexError):
                continue
            specs.append((b["qid"], b.get("category"), sp, b.get("expected")))
    out = []
    run_cases.dropped = 0
    for qid, cat, sp, sig in specs:
        try:
            c = _case(B, qid, (base_rows.get(qid) or {}).get("base_case", "").split(" || ")[0].split("] ", 1)[-1][:90] or qid,
                      sp, cat, "run", rew.get(qid))
        except Exception:                               # noqa: BLE001 - the project changed since the run
            run_cases.dropped += 1
            continue
        if sig is not None and c["expected"] != sig:
            run_cases.dropped += 1
            continue
        out.append(c)
    return out


def cases(source: str, runs_root: Path | None = None) -> dict[str, Any]:
    """``source``: "project", or a run id of the active journey ("latest": its newest finished run, else project)."""
    B = bench()
    run_id = None
    if source != "project" and runs_root is not None:
        cands = sorted((d for d in Path(runs_root).iterdir() if d.is_dir() and (d / "out" / "report.json").exists()
                        and json.loads((d / "job.json").read_text()).get("pack") == B.pack), reverse=True) \
            if Path(runs_root).exists() else []
        if source == "latest":
            run_id = cands[0].name if cands else None
        elif any(d.name == source for d in cands):
            run_id = source
    key = run_id or "project"
    if key not in B.cases_cache:
        B.cases_cache[key] = run_cases(B, Path(runs_root) / run_id / "out") if run_id else factor_cases(B)
        B.dropped[key] = getattr(run_cases, "dropped", 0) if run_id else 0
        if run_id and not B.cases_cache[key]:          # nothing rebuilt: fall back to the project's cases
            B.cases_cache.pop(key)
            key, run_id = "project", None
            B.cases_cache.setdefault(key, factor_cases(B))
    return {"source": run_id or "project", "dropped": B.dropped.get(key, 0),
            "cases": [{k: v for k, v in c.items() if k not in ("spec", "fixture")} for c in B.cases_cache[key]]}


def find_case(source: str, case_id: str) -> dict:
    B = bench()
    for c in B.cases_cache.get(source, []):
        if c["id"] == case_id:
            return c
    raise KeyError(f"case {case_id} (load the case list first)")


# ----------------------------------------------------------------------------- a conversation
def _describe(acts: list[dict]) -> list[str]:
    from ..turn_compare import describe_act
    return [describe_act(a) for a in acts if not str(a.get("act", "")).startswith("(")]


def _tools(log: list[dict]) -> list[dict]:
    from ..turn_compare import describe_tool
    out = []
    for c in log:
        res = c.get("result")
        status = (res.get("status") or res.get("error")) if isinstance(res, dict) else res
        out.append({"tool": c["tool"], "args": c.get("args") or {}, "status": status, "ok": c.get("ok", True),
                    "text": describe_tool(c)})
    return out


def _speech(rep) -> dict | None:
    if rep is None:
        return None
    d = rep.as_dict()
    claims = [{"field": c.get("field"), "words": c.get("words"), "status": c.get("status"), "truth": c.get("truth"),
               "why": c.get("why"), "basis": c.get("basis")} for c in d["claims"]]
    adv = set(rep.advisory)
    finds = [{**f, "advisory": f["kind"] in adv, "flow": f["kind"] == "not_allowed_here"} for f in d["findings"]]
    return {"ok": rep.ok, "flow_ok": rep.flow_ok, "heard": d.get("heard"), "claims": claims, "findings": finds}


@dataclass
class Session:
    id: str
    pack: str
    case: dict
    bot_kind: str
    bot_name: str
    bot: Any
    env: Any
    checker: Any
    llm: Any = None
    frames: list = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    diverged_at: int | None = None
    ended: bool = False
    created: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ what the graph expects
    def _expected(self) -> tuple[list[dict], list[dict]]:
        from ..turn_compare import segment
        B = bench()
        tr = B.oracle.run(self.case["spec"], json.loads(json.dumps(self.case["fixture"])), frames_override=list(self.frames))
        opening, turns = segment(tr.events)
        return opening, turns

    def _bot_turn(self, turn, n0: int, customer: dict | None) -> dict:
        from ..turn_compare import _tool_key, normalize_acts
        acts = list(turn.acts or [])
        tools = _tools(self.env.log[n0:])
        rep = None
        if self.checker is not None:
            rep = self.checker.check(acts, turn.text, self.env.log, list(self.frames))
        opening, exp_turns = self._expected()
        i = len(self.frames) - 1
        if customer is None:                            # the opening
            e_acts = [a for a in opening if not str(a.get("act", "")).startswith("(")]
            e_tools = []
            match = normalize_acts(e_acts) == normalize_acts(acts)
            tools_match = True
        else:
            e = exp_turns[i] if i < len(exp_turns) else None
            e_acts = e["acts"] if e else []
            e_tools = [{"tool": x["tool"], "args": x.get("args") or {}, "status": (x.get("result") or {}).get("status")
                        if isinstance(x.get("result"), dict) else x.get("result"), "text": None} for x in (e["tools"] if e else [])]
            from ..turn_compare import describe_tool
            for x, raw in zip(e_tools, e["tools"] if e else []):
                x["text"] = describe_tool(raw)
            tools_match = [_tool_key(x) for x in (e["tools"] if e else [])] == \
                [_tool_key({"tool": t["tool"], "args": t["args"]}) for t in tools]
            match = e is not None and tools_match and normalize_acts(e_acts) == normalize_acts(acts)
        after = self.diverged_at is not None
        if not match and self.diverged_at is None:
            self.diverged_at = len(self.turns)
        self.ended = bool(turn.ended) or self.ended
        out = {"n": len(self.turns), "customer": customer, "bot": {"text": turn.text, "acts": _describe(acts),
               "act_names": [a.get("act") for a in acts], "ended": bool(turn.ended)},
               "tools": tools, "expected": {"tools": [x["text"] for x in e_tools], "acts": _describe(e_acts),
                                           "ends": (not exp_turns or i >= len(exp_turns) - 1) and customer is not None
                                           and bool(e_acts) and str(e_acts[-1].get("act")) in ("END", "TRANSFER")},
               "match": match, "tools_match": tools_match, "after_divergence": after, "speech": _speech(rep)}
        self.turns.append(out)
        return out

    def start(self) -> dict:
        n0 = len(self.env.log)
        turn = self.bot.start()
        return self._bot_turn(turn, n0, None)

    def say(self, text: str) -> dict:
        if self.ended:
            raise ValueError("the conversation has ended; start a new one")
        if len(self.turns) >= MAX_TURNS:
            raise ValueError(f"{MAX_TURNS} turns is the playground's limit; start a new conversation")
        text = (text or "").strip()
        if not text:
            raise ValueError("type what the customer says, or pick a line")
        B = bench()
        last = (self.turns[-1]["bot"]["act_names"] or [None])[-1] if self.turns else None
        frame = B.reader.read(text, last)
        card = B.ont.card(frame, last) if last else {"act": frame.act, "slots": dict(frame.slots)}
        self.frames.append(frame)
        n0 = len(self.env.log)
        turn = self.bot.respond(text)
        return self._bot_turn(turn, n0, {"text": text, "state": last, "read": str(frame),
                                         "card": {"act": card.get("act"), "slots": card.get("slots")}})

    def view(self) -> dict:
        return {"id": self.id, "pack": self.pack, "case": self.case["id"], "case_label": self.case["label"],
                "bot": self.bot_kind, "bot_name": self.bot_name, "turns": self.turns, "ended": self.ended,
                "diverged_at": self.diverged_at, "next_line": self._next_line(),
                "llm": ({"stats": self.llm.stats.as_dict(), "calls": self.llm.llm.report()} if self.llm is not None else None)}

    def _next_line(self) -> int | None:
        """The scripted line that answers the bot's last question (the next turn of the case), if any."""
        n = len(self.frames)
        return n if n < len(self.case["lines"]) and not self.ended else None


_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()


def start(source: str, case_id: str, sut: dict | None) -> dict:
    """A new conversation on a case with the reference bot (``sut`` None or kind "reference") or the LLM bot."""
    from ..sut.tools import ToolEnvironment
    from .sut import make_factory
    B = bench()
    case = find_case(source, case_id)
    env = ToolEnvironment(json.loads(json.dumps(case["fixture"])))
    fac, name, llm = make_factory(sut, B.store)
    kind = "llm" if fac is not None else "reference"
    if fac is None:
        fac, name = B.reference_factory()
    checker = B.speech.conversation(case["fixture"].get("session_date")) if B.speech is not None else None
    s = Session(uuid.uuid4().hex[:12], B.pack, case, kind, name, fac(env), env, checker, llm)
    _SESSIONS[s.id] = s
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    s.start()
    return s.view()


def get(sid: str) -> Session:
    s = _SESSIONS.get(sid)
    if s is None:
        raise KeyError(f"conversation {sid} (it may have been closed; start a new one)")
    _SESSIONS.move_to_end(sid)
    return s
