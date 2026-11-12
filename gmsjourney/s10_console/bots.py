"""The bot under test as the console configures it, and the playground.

Parts of this file:
  1. ``make_factory``: the reference bot, or the LLM bot (its model reads the customer, the journey's graph decides,
     a model words the acts) from a console job's ``sut`` settings; ``suggested`` shows what the models are asked;
     ``check_gateway`` makes one call.
  2. The playground (``start`` / ``Session.say``): talk to the bot on one case; every turn is compared with what the
     graph expects at that point and its words go through the speech check.
  3. Saved conversations: a Playground conversation kept as a test case, replayed against any bot.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =====================================================================================================================
# PART 1 - The bot under test, as the console configures it: the journey's reference bot, or the LLM bot.
#
# A console job carries ``sut``::
#
#     {"kind": "reference"}                                    the pack's reference bot (the default)
#     {"kind": "llm", "provider": "gateway" | "local", "model": "gpt-41", "local_path": "", "vllm_url": "",
#      "mode": "live" | "replay", "temperature": 0.0,
#      "wording": "same" | "gateway" | "local" | "template", "wording_model": "", "wording_local_path": "",
#      "wording_vllm_url": "", "voice": "warm and brief"}
#
# The LLM bot (s06_bots/llm_bot.py) is structured: its model reads each customer turn (kind, intent, slots from what
# the state accepts), the journey's graph decides the tools and the acts, and a model words the acts (each line
# checked against its facts). ``provider`` / ``model`` / ``local_path`` / ``vllm_url`` choose the reading model:
# a gateway deployment, or a local model on vLLM (from a folder, in the test run's own process, or through a vLLM
# server; the Playground takes it only through a server). ``wording`` chooses the wording model apart from it:
# "same" (the reading model), a gateway deployment ``wording_model``, a local model (``wording_local_path`` /
# ``wording_vllm_url``), or "template" (no model: the journey's own templates). ``voice`` is the tone the wording
# model is asked for; it never changes what is said.
# =====================================================================================================================

WORDING_CHOICES = ("same", "gateway", "local", "template")


def suggested(store=None) -> dict[str, Any]:
    """What the LLM bot's models are asked, for the active journey: the two fixed prompts and the menu of its first
    question (what the reading model is offered there)."""
    from ..s01_packs.loading import current
    from ..s06_bots.llm_bot import READ_SYSTEM, STRUCTURED_VERSION, WORD_SYSTEM, journey_title, LLMReader, StructuredStats
    dom = current()
    out = {"pack": dom.name, "title": journey_title(), "reading": READ_SYSTEM, "wording": WORD_SYSTEM,
           "version": STRUCTURED_VERSION, "graph": dom.graph is not None}
    if dom.graph is not None:
        g = dom.graph
        first = next((s for s in g.states if g.kind(s) == "ask"), None)
        if first:
            out["example_state"] = first
            out["example_menu"] = LLMReader(g, None, StructuredStats()).menu(first)
    return out


def wording_where(sut: dict[str, Any]) -> str:
    """Where the bot's words are made: "same" (the reading model), "gateway", "local" or "template" (no model)."""
    w = str((sut or {}).get("wording") or "").strip().lower()
    return w if w in WORDING_CHOICES else "same"


def _local_llm(path: Any, url: Any, s, tag: str, who: str):
    """(CachedLLM on a local vLLM model, where it runs), or RuntimeError naming ``who`` when it cannot be used."""
    from ..llm import CachedLLM
    from ..s06_bots.llm_bot import LocalChat, fake_model, status
    lp, lu = str(path or "").strip(), str(url or "").strip()
    st = status(lp, lu)
    if not st["usable"]:
        raise RuntimeError(f"{who} local model cannot be used: {st['detail']}")
    fake = fake_model()
    chat = fake if fake is not None else LocalChat(lp or None, temperature=s.temperature, max_tokens=s.max_tokens, url=lu or None)
    s.model = getattr(chat, "name", "") or f"fake-{type(chat).__name__}"
    return CachedLLM(s, model=chat, tag=tag), "vLLM" + (" server" if st["url"] else ", this machine")


def _gateway_llm(s, tag: str):
    from ..llm import CachedLLM
    from ..s06_bots.llm_bot import fake_model
    fake = fake_model()
    return CachedLLM(s, model=fake, tag=tag) if fake is not None else CachedLLM(s, tag=tag)


def settings_for(sut: dict[str, Any], part: str = "read"):
    """The model-client settings of a console job's ``sut`` block: for reading a turn (a small JSON object: 200
    tokens; temperature 0) or for wording a bot turn (400 tokens; the job's temperature). JSON mode either way, so a
    reply is never prose that needs a second call. The gateway counts max_tokens against the deployment's quota."""
    from ..llm import LLMSettings
    sut = sut or {}
    if part == "word":
        model = str(sut.get("wording_model") or "") if wording_where(sut) == "gateway" else str(sut.get("model") or "")
        return LLMSettings(model=model, temperature=float(sut.get("temperature") or 0.0), max_tokens=400,
                           mode=str(sut.get("mode") or "live"), backend="gateway", json_mode=True)
    return LLMSettings(model=str(sut.get("model") or ""), temperature=0.0, max_tokens=200,
                       mode=str(sut.get("mode") or "live"), backend="gateway", json_mode=True)


def make_factory(sut: dict[str, Any] | None, store=None):
    """(bot_factory or None for the reference bot, display name, the LLM factory or None)."""
    sut = sut or {}
    if sut.get("kind") != "llm":
        return None, None, None
    from ..llm import DEFAULT_MODEL
    from ..s06_bots.llm_bot import fake_model, make_structured_factory
    s = settings_for(sut, "read")
    if sut.get("provider") == "local":                 # gpt-oss / Llama on vLLM
        reader, where = _local_llm(sut.get("local_path"), sut.get("vllm_url"), s, "sut", "the bot's")
    else:
        reader, where = _gateway_llm(s, "sut"), ""
    w = wording_where(sut)
    ws = settings_for(sut, "word")
    if w == "template":
        words, wname = None, "the journey's templates"
    elif w == "local":
        words, _ = _local_llm(sut.get("wording_local_path"), sut.get("wording_vllm_url"), ws, "sut-word", "the wording")
        wname = f"{ws.model} (local, vLLM)"
    elif w == "gateway":
        words = _gateway_llm(ws, "sut-word")
        same = sut.get("provider") != "local" and (ws.model or DEFAULT_MODEL) == (s.model or DEFAULT_MODEL)
        wname = "the same model" if same else (ws.model or DEFAULT_MODEL)
    else:                                              # the reading model words too (its own settings: 400 tokens)
        if sut.get("provider") == "local":
            words, _ = _local_llm(sut.get("local_path"), sut.get("vllm_url"), ws, "sut-word", "the bot's")
        else:
            words = _gateway_llm(ws, "sut-word")
        wname = "the same model"
    fac = make_structured_factory(store, reader_llm=reader, wording_llm=words, voice=str(sut.get("voice") or ""))
    rname = (s.model or DEFAULT_MODEL) + (f", {where}" if where else "")
    name = (f"LLM bot ({rname} reads; graph decides; {wname} words"
            + (", replay" if s.mode == "replay" else "") + (", stand-in model" if fake_model() is not None else "") + ")")
    return fac, name, fac


def check_gateway(model: str = "", timeout: float = 30.0) -> dict[str, Any]:
    """One tiny call through the gateway (tools/check_llm.py in one function): ok, endpoint, seconds, the reply or
    the exact error. A configured stand-in model (GMSJOURNEY_SUT_FAKE_MODEL) is reported instead of calling out."""
    from ..s06_bots.llm_bot import fake_model
    if fake_model() is not None:
        return {"ok": True, "fake": True, "detail": "a stand-in model is configured (GMSJOURNEY_SUT_FAKE_MODEL); "
                                                    "the gateway is not called"}
    from ..llm import LLMSettings, LLMUnavailable, _messages, make_chat_model
    s = LLMSettings(model=model, timeout=timeout, backend="gateway")
    try:
        chat = make_chat_model(s)
    except LLMUnavailable as e:
        return {"ok": False, "model": s.model, "detail": f"no credential: {e}"}
    except Exception as e:                         # noqa: BLE001
        return {"ok": False, "model": s.model, "detail": f"{type(e).__name__}: {e}"}
    url = chat._url() if hasattr(chat, "_url") else None
    t0 = time.time()
    try:
        out = chat.invoke(_messages("You are a terse assistant.", 'Reply with exactly this JSON and nothing else: {"ok": true}'))
    except Exception as e:                         # noqa: BLE001 - report anything
        return {"ok": False, "model": s.model, "endpoint": url, "seconds": round(time.time() - t0, 2),
                "detail": f"{type(e).__name__}: {str(e)[:400]}"}
    return {"ok": True, "model": s.model, "endpoint": url, "seconds": round(time.time() - t0, 2),
            "detail": f"replied {str(getattr(out, 'content', ''))[:80]!r}"}


# =====================================================================================================================
# PART 2 - The playground: talk to the bot under test yourself, one turn at a time, with every turn checked as you go.
#
# A session is one conversation on one case of any journey. The case sets the mock bank (the items, which tools fail or
# decline) and its base journey (the customer's lines in the graph's own conversation of the case). At every turn the tester chooses how the customer answers:
#
# * stick to the journey: the base journey's next line, whatever the bot said (as a test run's scripted customer says
#   it, so a bot mistake does not change the case);
# * talk freely: the tester types anything (a line that means the journey's next one counts as it; anything else is an
#   extra turn and the journey's line stays next).
#
# Either way the words are read into a meaning card and the turn is validated the same way. For every bot turn the
# session shows, as the answer sheet does:
#
# * the customer side: the dialog state answered, what it accepts, how it reads a bare yes or no, its rules and
#   limits; the meaning card read from the words, whether the state admits it, the card as triples; and, for free
#   talk, the journey's next line and whether the typed one means the same;
# * the agent side: the graph's hops for this turn (state before, trigger, tool calls with results, condition, state
#   after, rules with their text, the graph triple and its source), the tools and acts the graph expects, and the
#   bot's own tools, acts and words;
# * the tool calls the bot made, with their results (the mock bank's log);
# * what the graph expects at this turn: the oracle replays the customer's turns so far (each read into its meaning
#   card by the frame reader) and gives the tools it would call and the acts it would say; the turn matches when the
#   tools (with their key arguments) and the decision acts are the same (turn_compare's rules); every turn that does
#   not match is flagged as a deviation, with the reasons (the first one is where the bot left the journey);
# * the speech check of what the bot said (claims supported, contradicted or unsupported; policy statements;
#   findings such as a missing or unfaithful act), as the campaign grades it.
#
# Everything runs on the journey's symbolic store, compiled from the graph in about a second: no knowledge-store
# training and no licence. Its figures and rules are the graph's own, so the checks agree with a campaign's on what is
# true; the campaign's trained store adds the geometry checks on top.
#
# Cases come from a finished run of the journey (``out/case_specs.jsonl``; older runs are rebuilt from report.json
# and kept only where the rebuilt case has the run's exact expected outcome) or, without a run, from the project:
# the default case and one case per factor level.
# =====================================================================================================================

MAX_SESSIONS = 24
MAX_TURNS = 60


# ----------------------------------------------------------------------------- the journey's symbolic world
class Bench:
    """The active journey's store, oracle, reader, realizer and speech verifier (built once per journey)."""

    def __init__(self) -> None:
        from ..s01_packs.loading import current
        from ..s05_rewording.wording import FrameReader
        from ..s05_rewording.wording import Realizer
        from ..s04_cases.meaning import Ontology
        from ..s02_graph.flow import GraphOracle as PolicyOracle
        from ..s03_knowledge.store import symbolic_store
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
            from ..s07_checks.speech_reading import SpeechContract
            from ..s07_checks.speech_checks import UtteranceVerifier
            self.speech = UtteranceVerifier(self.store, SpeechContract.from_graph(kg))
        self.cases_cache: dict[str, list[dict]] = {}
        self.dropped: dict[str, int] = {}

    def reference_factory(self):
        if self.dom.graph is None:
            from ..s06_bots.harness import make_reference_factory
            return make_reference_factory(self.store), "the journey's reference bot"
        from ..s06_bots.reference import make_graph_factory
        return make_graph_factory(self.store), "the journey's reference bot (follows the graph)"


_BENCHES: dict[str, Bench] = {}


def bench(fresh: bool = False) -> Bench:
    """The active journey's bench (call with the journey active)."""
    from ..s01_packs.loading import current
    name = current().name
    if fresh or name not in _BENCHES:
        _BENCHES[name] = Bench()
    return _BENCHES[name]


def reset(pack: str | None = None) -> None:
    """Forget the bench of ``pack`` (every bench if None): its graph changed."""
    if pack is None:
        _BENCHES.clear()
    else:
        _BENCHES.pop(pack, None)


# ----------------------------------------------------------------------------- cases
def _frame(d: dict):
    from ..s04_cases.meaning import SemanticFrame as F
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
    from ..s04_cases.journey import build_fixture, primary_category
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
    from ..s04_cases.design import JourneyBaseGenerator
    from ..s04_cases.journey import JourneySpec, build_fixture, is_valid
    from ..s01_packs.project import Project
    from .studio import reach
    proj = Project.load(B.dom.project_path)
    out = [_case(B, "default", "Every factor at its default level", JourneySpec(), source="project")]
    play = lambda sp: B.oracle.run(sp, build_fixture(sp))          # noqa: E731
    gen = JourneyBaseGenerator(B.oracle, kb_pool=proj.padding_questions())   # questions and turn padding, as a run adds them
    for name, f in proj.factors.items():
        for lv in f.levels:
            if lv == f.default:
                continue
            try:
                if B.dom.graph is not None and not f.role:  # a level the default call never reaches gets a partner
                    levels, partner = reach(proj, play, name, lv)
                else:                                       # triples graphs: relevance rules say where a level matters
                    levels, partner = {name: lv}, None
                sp = JourneySpec(**levels)
            except TypeError:
                continue
            if not is_valid(sp):
                continue
            label = f"{name} = {lv}" + (f", with {partner} so the call gets there" if partner else
                                         " (the call does not get there from the defaults; generated cases combine it "
                                         "with levels that do)" if partner == "" else "")
            done = sp
            if f.role in ("questions", "turn_target"):  # a policy-question level asks one; a turn level pads the call
                try:
                    done = gen.finalize(sp)
                except Exception:                       # noqa: BLE001
                    done = sp
            if done is None:
                label += " (its turn target cannot be met from here)"
            else:
                sp = done
            try:
                out.append(_case(B, f"{name}={lv}", label, sp, source="project"))
            except Exception:                           # noqa: BLE001 - a level the oracle cannot play alone
                continue
    return out


def run_cases(B: Bench, out_dir: Path) -> list[dict]:
    """The generated cases of a finished run, rebuilt exactly (case_specs.jsonl) or from report.json and kept only
    where the rebuilt case reproduces the run's expected outcome. Reworded versions come from the run's dataset."""
    from ..s04_cases.journey import DIMENSIONS, JourneySpec, policy_question_case
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
    """One case of the loaded case list, by source and id."""
    B = bench()
    for c in B.cases_cache.get(source, []):
        if c["id"] == case_id:
            return c
    raise KeyError(f"case {case_id} (load the case list first)")


# ----------------------------------------------------------------------------- a conversation
def _describe(acts: list[dict]) -> list[str]:
    from ..s09_reports.cases import describe_act
    return [describe_act(a) for a in acts if not str(a.get("act", "")).startswith("(")]


def _tools(log: list[dict]) -> list[dict]:
    from ..s09_reports.cases import describe_tool
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
    d = rep.as_dict()                                # every claim in full: the speech check page replays them
    adv = set(rep.advisory)
    finds = [{**f, "advisory": f["kind"] in adv, "flow": f["kind"] == "not_allowed_here"} for f in d["findings"]]
    return {"ok": rep.ok, "core_ok": d.get("core_ok", rep.ok), "flow_ok": rep.flow_ok, "heard": d.get("heard"),
            "claims": d["claims"], "findings": finds}


@dataclass
class Session:
    """One playground conversation: the case, the bot, the mock bank, the speech checker and every turn so far."""
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
    deviations: list[int] = field(default_factory=list)
    ended: bool = False
    created: float = field(default_factory=time.time)
    judges: bool = False                                # the two LLM judges read every bot turn beside the rules
    ended_turn: int | None = None                       # the turn at which the bot ended the call
    base: list | None = None                            # the base journey's customer lines (built on first use)
    base_i: int = 0                                     # the next of them
    last_acts: list = field(default_factory=list)
    source: str = "project"
    replay_of: str | None = None

    # ------------------------------------------------------------------ the journey's customer
    def _base_lines(self) -> list[tuple]:
        """The base journey: the customer's lines in the graph's own conversation of this case, in order, each with
        its context and the state it answers [(frame, context, state)]. Built once per conversation."""
        if self.base is None:
            B = bench()
            tr = B.oracle.run(self.case["spec"], json.loads(json.dumps(self.case["fixture"])))
            by_id = {t.get("transaction_id"): t for t in (self.case["fixture"].get("transactions") or [])}
            lines, last_bot = [], {}
            for ev in tr.events:
                if ev["role"] == "bot":
                    last_bot = ev["acts"][-1] if ev.get("acts") else {}
                elif ev["role"] == "user":
                    state = ev.get("state") or last_bot.get("act") or last_bot.get("type")
                    item = ev.get("item") or {k: v for k, v in (by_id.get(last_bot.get("txn")) or {}).items()
                                              if k in ("merchant", "amount")}
                    lines.append((ev["frame"], dict(item or {}), state))
            self.base = lines
        return self.base

    def _journey_frame(self, consume: bool):
        """The base journey's next line (None: the base journey has no more lines). The journey does not follow the
        bot: it moves on when its line is said, or when a typed line means the same; anything else is an extra turn."""
        lines = self._base_lines()
        if self.base_i >= len(lines):
            return None
        fr = lines[self.base_i][0]
        if consume:
            self.base_i += 1
        return fr

    def _words(self, frame, base_line: int | None = None) -> str:
        """``frame`` in words; a base-journey line (``base_line``) is worded in its own context and state."""
        B = bench()
        state = self.last_acts[-1].get("act") if self.last_acts else None
        ctx = {"state": state}
        lines = self._base_lines()
        if base_line is not None and base_line < len(lines):
            ctx = {**lines[base_line][1], "state": lines[base_line][2] or state}
        try:
            return B.realizer.realize(frame, {}, f"{self.case['id']}~play|{len(self.frames)}", ctx)
        except Exception:                               # noqa: BLE001
            return str(frame)

    def journey_line(self) -> dict | None:
        """The journey's next customer line, without using it up: {text, card} or None. The customer plays its part to
        the end: after the bot has ended the call, the base journey's remaining lines are still offered."""
        fr = self._journey_frame(consume=False)
        if fr is None:
            return None
        return {"text": self._words(fr, self.base_i), "read": str(fr)}

    # ------------------------------------------------------------------ the answer sheet of one turn
    def _state_sheet(self, state: str | None) -> dict | None:
        kg = getattr(bench().store, "kg", None)
        if kg is None or not state:
            return None
        y, n = kg.interprets(state)
        return {"state": state, "stage": kg.stage(state), "asks": state in kg.prompt_states,
                "accepts": kg.accepted_acts(state), "yes_means": y, "no_means": n,
                "rules": [{"id": r, "text": kg.rule_text(r)} for r in kg.rules_for(state)],
                "limits": [{"key": k, "value": kg.limit(k)} for k in kg.limits_for(state)]}

    def _hops(self, i: int | None) -> list[dict]:
        """The graph transitions the oracle took in customer turn ``i`` (None: the opening), each with its condition,
        rules and graph triple. A transition on a customer act (anything but "auto" or the turn-limit "any") starts a
        new customer turn."""
        from ..s09_reports.cases import graph_edge
        kg = getattr(bench().store, "kg", None)
        turns = [[]]                                    # turns[0]: the opening; turns[k]: customer turn k-1
        for src, dst, on in getattr(self, "_trace").flow_path:
            if on not in ("auto", "any"):
                turns.append([])
            turns[-1].append((src, dst, on))
        seg = turns[0] if i is None else (turns[i + 1] if i + 1 < len(turns) else [])
        out = []
        for src, dst, on in seg:
            e = graph_edge(kg, src, dst, on)
            q = (e or {}).get("qualifiers") or {}
            out.append({"from": src, "to": dst, "trigger": on, "condition": q.get("when") or q.get("guard"),
                        "rules": [{"id": r, "text": kg.rule_text(r)} for r in q.get("rules") or []] if kg else [],
                        "triple": (e or {}).get("id"), "source": (e or {}).get("source"), "evidence": (e or {}).get("evidence")})
        return out

    def _expected(self) -> tuple[list[dict], list[dict]]:
        from ..s09_reports.cases import segment
        B = bench()
        tr = B.oracle.run(self.case["spec"], json.loads(json.dumps(self.case["fixture"])), frames_override=list(self.frames))
        self._trace = tr
        opening, turns = segment(tr.events)
        return opening, turns

    def _bot_turn(self, turn, n0: int, customer: dict | None) -> dict:
        from ..s09_reports.cases import _tool_key, normalize_acts
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
            from ..s09_reports.cases import describe_tool
            for x, raw in zip(e_tools, e["tools"] if e else []):
                x["text"] = describe_tool(raw)
            tools_match = [_tool_key(x) for x in (e["tools"] if e else [])] == \
                [_tool_key({"tool": t["tool"], "args": t["args"]}) for t in tools]
            match = e is not None and tools_match and normalize_acts(e_acts) == normalize_acts(acts)
        after = self.diverged_at is not None
        speech = _speech(rep)
        why = []
        hung_up = customer is not None and self.ended          # the bot had ended the call before this line
        if hung_up:
            why.append(f"the bot had already ended the call (at {'the opening' if self.ended_turn == 0 else f'turn {self.ended_turn}'}); "
                       "the customer plays the rest of the journey")
        if customer is not None and i >= len(exp_turns):
            why.append("the graph's conversation has already ended here, but the bot carried on")
        else:
            if not tools_match:
                why.append("tools: the graph calls " + (", ".join(x["text"] for x in e_tools) or "no tool") +
                           "; the bot called " + (", ".join(x["text"] for x in tools) or "no tool"))
            if normalize_acts(e_acts) != normalize_acts(acts):
                why.append("acts: the graph says " + (", ".join(_describe(e_acts)) or "nothing") +
                           "; the bot said " + (", ".join(_describe(acts)) or "nothing"))
        if speech and not speech["ok"]:
            bad = [f"{c['status']} {c['field']} '{c['words']}'" for c in speech["claims"] if c["status"] in ("contradicted", "unsupported")]
            bad += [f"{f['kind']} {f.get('act') or ''}".strip() for f in speech["findings"] if not f["advisory"]]
            if bad:
                why.append("words: " + "; ".join(bad))
        deviated = not match or bool(speech and not speech["ok"]) or (hung_up and bool(e_acts or e_tools))
        if deviated:
            self.deviations.append(len(self.turns))
            if self.diverged_at is None:
                self.diverged_at = len(self.turns)
        if turn.ended and not self.ended:
            self.ended_turn = len(self.turns)
        self.ended = bool(turn.ended) or self.ended
        self.last_acts = [a for a in acts if not str(a.get("act", "")).startswith("(")] or self.last_acts
        state = customer.get("state") if customer else None
        sheet = {"state": self._state_sheet(state), "hops": self._hops(None if customer is None else i)}
        out = {"n": len(self.turns), "customer": customer, "bot": {"text": turn.text, "acts": _describe(acts),
               "act_names": [a.get("act") for a in acts], "ended": bool(turn.ended)},
               "tools": tools, "expected": {"tools": [x["text"] for x in e_tools], "acts": _describe(e_acts),
                                           "ends": (not exp_turns or i >= len(exp_turns) - 1) and customer is not None
                                           and bool(e_acts) and str(e_acts[-1].get("act")) in ("END", "TRANSFER")},
               "match": match, "tools_match": tools_match, "after_divergence": after, "speech": speech,
               "deviated": deviated, "first_deviation": deviated and self.diverged_at == len(self.turns),
               "why": why, "sheet": sheet,
               "reading": _reading(getattr(getattr(self.bot, "reader", None), "last", None)) if self.bot_kind == "llm" else None}
        if self.judges:
            out["judges"] = self._judge(out, e_tools, e_acts, acts, tools)
        self.turns.append(out)
        return out

    def _quota(self) -> dict | None:
        """The bot's gateway deployment's quota as the pacer knows it (tokens a minute, calls at once, pushbacks)."""
        m = getattr(getattr(self.llm, "llm", None), "model", None) if self.llm is not None else None
        url = m._url() if hasattr(m, "_url") else None
        if not url:
            return None
        from ..llm import gateway_quota
        return gateway_quota().get(url)

    def _judge(self, out: dict, e_tools: list, e_acts: list, acts: list, tools: list) -> dict:
        """The two LLM judges on this turn (s07_checks/llm_judges.py), beside the rules' verdict."""
        c = out.get("customer") or {}
        hist = []
        for t in self.turns:
            if t.get("customer"):
                hist.append(f"customer: {t['customer'].get('text')}")
            hist.append(f"bot: {t['bot'].get('text') or ', '.join(t['bot'].get('acts') or [])}")
        sheet = (out.get("sheet") or {}).get("state") or {}
        try:
            return judges().judge_turn({
                "key": f"{self.pack}|{self.case['id']}|{len(self.turns)}|{c.get('text')}|{out['bot'].get('text')}",
                "state": c.get("state") or "(call opens)", "customer_text": c.get("text") or "(the call opens)",
                "customer_card": c.get("card"), "expected_tools": "; ".join(x["text"] for x in e_tools),
                "expected_acts": "; ".join(_describe(e_acts)), "actual_tools": "; ".join(t["text"] for t in tools),
                "actual_acts": "; ".join(_describe(acts)), "bot_text": out["bot"].get("text"), "history": hist,
                "rules": {r["id"]: r["text"] for r in sheet.get("rules") or []}, "rules_ok": not out["deviated"]})
        except Exception as e:                          # noqa: BLE001 - a judge never breaks the conversation
            na = {"verdict": "unavailable", "reason": f"{type(e).__name__}: {str(e)[:160]}", "first_wrong_turn": None}
            from ..s07_checks.llm_judges import combine
            return {"a": na, "b": na, "combined": combine(not out["deviated"], na, na), "model": None}

    def start(self) -> dict:
        n0 = len(self.env.log)
        turn = _clean_turn(self.bot.start())
        return self._bot_turn(turn, n0, None)

    def say(self, text: str = "", mode: str = "free") -> dict:
        """The customer's next turn. ``mode`` "journey": the base journey's next line, whatever the bot said (``text``
        is ignored); "free": ``text`` as typed (it counts as the journey's line when it means the same, otherwise it is
        an extra turn and the journey's line stays next). Both are read into a meaning card and validated the same way."""
        from ..s09_reports.cases import frame_hrt
        if self.ended and mode != "replay" and self._journey_frame(consume=False) is None:
            raise ValueError("the bot has ended the call and the base journey has no more customer lines; start a new one")
        if len(self.turns) >= MAX_TURNS:
            raise ValueError(f"{MAX_TURNS} turns is the playground's limit; start a new conversation")
        B = bench()
        last = (self.turns[-1]["bot"]["act_names"] or [None])[-1] if self.turns else None
        if mode == "journey":
            k = self.base_i
            jf = self._journey_frame(consume=True)
            if jf is None:
                raise ValueError("the base journey has no more customer lines (its call has ended); talk freely instead")
            text = self._words(jf, k)
            would = None
        else:                                           # "free" as typed, or "replay": a saved conversation's line
            text = (text or "").strip()
            if not text:
                raise ValueError("type what the customer says, or stick to the journey")
            jf = self._journey_frame(consume=False)
            would = {"text": self._words(jf, self.base_i), "read": str(jf)} if jf is not None else None
        frame = B.reader.read(text, last)
        card = B.ont.card(frame, last) if last else {"act": frame.act, "slots": dict(frame.slots), "admissible": True}
        # The graph is given the reading as said (frame), not the state's reading of a bare yes/no (card): show what it
        # was given, and the state's reading beside it, so "Means" never disagrees with what the graph expects.
        state_reading = None
        if (card.get("act"), card.get("slots")) != (frame.act, dict(frame.slots)):
            state_reading = {"act": card.get("act"), "slots": card.get("slots")}
            card = {**card, "act": frame.act, "slots": dict(frame.slots)}
        if would is not None:
            wc = B.ont.card(jf, last) if last else {"act": jf.act, "slots": dict(jf.slots)}
            mine = state_reading or card                # meaning compared as read in the state, as the proofs do
            would["same_meaning"] = (wc.get("act"), wc.get("slots")) == (mine.get("act"), mine.get("slots"))
            if would["same_meaning"]:                   # said in other words: the journey moves on
                self.base_i += 1
        self.frames.append(frame)
        n0 = len(self.env.log)
        turn = _clean_turn(self.bot.respond(text))
        return self._bot_turn(turn, n0, {"text": text, "mode": mode, "state": last, "read": str(frame),
                                         "card": {"act": card.get("act"), "slots": card.get("slots")},
                                         "state_reading": state_reading,
                                         "admissible": card.get("admissible", True), "why_not": card.get("why_not"),
                                         "triples": [list(x) for x in frame_hrt(frame, B.ont, f"turn{len(self.frames)}", last)],
                                         "journey_would_say": would})

    def view(self) -> dict:
        return {"id": self.id, "pack": self.pack, "case": self.case["id"], "case_label": self.case["label"],
                "source": self.source, "replay_of": self.replay_of,
                "bot": self.bot_kind, "bot_name": self.bot_name, "turns": self.turns, "ended": self.ended,
                "ended_turn": self.ended_turn,
                "diverged_at": self.diverged_at, "deviations": list(self.deviations), "next_line": self._next_line(),
                "journey_line": self.journey_line(), "judges": self.judges,
                "judges_model": judges().model if self.judges else None,
                "gateway_quota": self._quota(),
                "llm": ({"stats": self.llm.stats.as_dict(), "calls": self.llm.llm.report(),
                         "wording": self.llm.wording_llm.report() if getattr(self.llm, "wording_llm", None) is not None else None}
                        if self.llm is not None else None)}

    def speech_cards(self) -> list[dict]:
        """This conversation's speech checks as one run of the speech check page (speech_playground.html): the
        shape of a test run's dataset/case_results.jsonl rows, ``[{sid, arm, correct, speech_turns}]``."""
        from ..s07_checks.speech_checks import turn_where
        cards = []
        for t in self.turns:
            sp = t.get("speech")
            if not sp:
                continue
            cards.append({"turn": t["n"], "where": turn_where(t["n"]), "customer": (t.get("customer") or {}).get("text"),
                          "text": t["bot"]["text"], "acts": t["bot"].get("act_names") or [], "heard": sp.get("heard"),
                          "ok": sp["ok"], "core_ok": sp.get("core_ok", sp["ok"]), "flow_ok": sp["flow_ok"],
                          "claims": sp["claims"], "findings": sp["findings"]})
        return [{"sid": f"{self.case['id']} · Playground conversation {self.id}", "arm": "base", "correct": None,
                 "speech_turns": cards}]

    def _next_line(self) -> int | None:
        """The scripted line that answers the bot's last question (the next turn of the case), if any."""
        n = len(self.frames)
        return n if n < len(self.case["lines"]) and not self.ended else None


_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()


from ..s06_bots.harness import _clean as _clean_turn    # a bot's malformed references graded, not fatal

_JUDGES: list = []


def judges():
    """The Playground's two LLM judges (built once, on the testing LLM: the console's Judges' LLM setting)."""
    if not _JUDGES:
        from ..s07_checks.llm_judges import Judges
        _JUDGES.append(Judges())
    return _JUDGES[0]


def _reading(last: dict | None) -> dict | None:
    """How the LLM bot read the customer's last turn: its kind, and why a reading was not used."""
    if not last:
        return None
    return {"kind": last.get("kind"), "rejected": last.get("why"), "read": last.get("read")}


def reset_judges() -> None:
    """Forget the judges (the testing LLM changed)."""
    _JUDGES.clear()


def start(source: str, case_id: str, sut: dict | None, judge: bool = False) -> dict:
    """A new conversation on a case with the reference bot (``sut`` None or kind "reference") or the LLM bot; with
    ``judge`` the two LLM judges read every bot turn beside the rules."""
    return _open(source, case_id, sut, judge=judge).view()


def _open(source: str, case_id: str, sut: dict | None, fac=None, name: str | None = None, judge: bool = False) -> Session:
    """A new conversation, opened: the bot's first turn is taken. ``fac``/``name``: a bot factory already built (a test
    run's), used instead of building one from ``sut``."""
    from ..s06_bots.tools import ToolEnvironment
    if fac is None and (sut or {}).get("kind") == "llm" and (sut or {}).get("provider") == "local":
        from ..s06_bots.llm_bot import status
        if not status(sut.get("local_path"), sut.get("vllm_url"))["url"]:   # a chat never loads a model into the console
            raise ValueError("in the Playground a local model needs a vLLM server URL (loading the model here would take "
                             "the console's process); give the server, or use it in a test run")
    if fac is None and (sut or {}).get("kind") == "llm" and wording_where(sut) == "local":
        from ..s06_bots.llm_bot import status
        if not status(sut.get("wording_local_path"), sut.get("wording_vllm_url"))["url"]:
            raise ValueError("in the Playground the wording's local model needs a vLLM server URL; give the server, "
                             "or use it in a test run")
    B = bench()
    case = find_case(source, case_id)
    env = ToolEnvironment(json.loads(json.dumps(case["fixture"])))
    if fac is not None:
        llm = fac if hasattr(fac, "stats") and hasattr(fac, "llm") else None     # an LLM bot's factory keeps its stats
        kind = "llm" if llm is not None else "reference"
    else:
        fac, name, llm = make_factory(sut, B.store)
        kind = "llm" if fac is not None else "reference"
        if fac is None:
            fac, name = B.reference_factory()
    checker = B.speech.conversation(case["fixture"].get("session_date")) if B.speech is not None else None
    s = Session(uuid.uuid4().hex[:12], B.pack, case, kind, name or "the bot under test", fac(env), env, checker, llm,
                source=source, judges=bool(judge))
    _SESSIONS[s.id] = s
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    s.start()
    return s


def get(sid: str) -> Session:
    """A playground conversation by id (most recently used are kept)."""
    s = _SESSIONS.get(sid)
    if s is None:
        raise KeyError(f"conversation {sid} (it may have been closed; start a new one)")
    _SESSIONS.move_to_end(sid)
    return s


# =====================================================================================================================
# PART 3 - Saved conversations: a Playground conversation kept as a test case.
#
# A conversation that found something (or one worth keeping) is saved with the journey, in
# ``<pack>/saved_conversations.json``: the case it started from and every customer line in order. Saving keeps the
# customer's side only; the bot's side is what is being tested. Replaying sends the same lines, in the same order, to
# a bot (the Playground's choice, or a test run's bot under test) and validates every turn exactly as the Playground
# does. Every test run replays the journey's saved conversations after its steps (runs.py), so a deviation found by
# hand is checked again on every later bot.
# =====================================================================================================================

def _saved_file() -> Path:
    return Path(bench().dom.dir) / "saved_conversations.json"


def saved_list() -> list[dict]:
    """The active journey's saved conversations (newest first)."""
    f = _saved_file()
    try:
        items = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    except (OSError, ValueError):
        items = []
    return sorted(items, key=lambda x: x.get("created", ""), reverse=True)


def _write_saved(items: list[dict]) -> None:
    f = _saved_file()
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(f)


def save_conversation(sid: str, name: str = "", note: str = "") -> dict:
    """Keep conversation ``sid`` as a test case of its journey; returns the saved record."""
    s = get(sid)
    lines = [{"text": t["customer"]["text"], "mode": t["customer"].get("mode", "free")} for t in s.turns if t.get("customer")]
    if not lines:
        raise ValueError("say at least one customer line before saving the conversation")
    first = s.diverged_at
    rec = {"id": uuid.uuid4().hex[:10], "name": (name or "").strip()[:120] or
           (f"{s.case['id']}: deviation at {'the opening' if first == 0 else f'turn {first}'}" if first is not None
            else f"{s.case['id']}: {len(lines)} customer line{'s' if len(lines) != 1 else ''}"),
           "note": (note or "").strip()[:1000], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "source": s.source, "case": s.case["id"], "case_label": s.case["label"], "lines": lines,
           "saved_with": {"bot": s.bot_name, "deviations": list(s.deviations), "first_deviation": first,
                          "why": s.turns[first]["why"] if first is not None and first < len(s.turns) else [],
                          "ended": s.ended}}
    _write_saved(saved_list() + [rec])
    return rec


def delete_saved(cid: str) -> None:
    items = saved_list()
    keep = [x for x in items if x["id"] != cid]
    if len(keep) == len(items):
        raise KeyError(f"saved conversation {cid}")
    _write_saved(keep)


def replay(rec: dict, sut: dict | None = None, runs_root: Path | None = None, fac=None, name: str | None = None) -> Session:
    """Send a saved conversation's customer lines, in order, to a bot; the Session holds every validated turn."""
    src = rec.get("source") or "project"
    loaded = cases(src, runs_root)                      # the case list the conversation started from
    try:
        find_case(loaded["source"], rec["case"])
        src = loaded["source"]
    except KeyError:
        src = cases("project", runs_root)["source"]     # that run is gone: the project may still have the case
        find_case(src, rec["case"])                     # KeyError: the case no longer exists
    s = _open(src, rec["case"], sut, fac=fac, name=name)
    s.replay_of = rec["id"]
    for ln in rec.get("lines") or []:               # every saved line, even after the bot has ended the call
        if len(s.turns) >= MAX_TURNS:
            break
        s.say(ln["text"], "replay")
    return s


def replay_result(rec: dict, s: Session | None, error: str | None = None) -> dict:
    """One saved conversation's replay, summarised: passed when every line was sent and no turn deviated."""
    base = {"id": rec["id"], "name": rec.get("name"), "case": rec.get("case"), "lines": len(rec.get("lines") or []),
            "saved_with": rec.get("saved_with")}
    if s is None:
        return {**base, "passed": False, "error": error or "could not be replayed"}
    sent = sum(1 for t in s.turns if t.get("customer"))
    first = s.diverged_at
    out = {**base, "passed": not s.deviations and sent == base["lines"], "sent": sent, "deviations": list(s.deviations),
           "first_deviation": first, "why": s.turns[first]["why"] if first is not None else [], "ended": s.ended,
           "bot": s.bot_name, "session": s.id}
    if s.ended_turn is not None and s.ended_turn < len(s.turns) - 1:
        out["why"] = out["why"] + [f"the bot ended the call at turn {s.ended_turn}, before the customer had said all "
                                   f"{base['lines']} lines"]
    return out


def replay_all(sut: dict | None = None, runs_root: Path | None = None, fac=None, name: str | None = None,
               on_each=None) -> list[dict]:
    """Replay every saved conversation of the active journey; ``on_each(i, n, result)`` after each."""
    items = saved_list()
    out = []
    for i, rec in enumerate(items):
        try:
            res = replay_result(rec, replay(rec, sut, runs_root, fac=fac, name=name))
        except KeyError:
            res = replay_result(rec, None, f"its case {rec.get('case')} no longer exists in the project")
        except Exception as e:                          # noqa: BLE001 - one bad replay must not stop the others
            res = replay_result(rec, None, f"{type(e).__name__}: {str(e)[:200]}")
        out.append(res)
        if on_each is not None:
            on_each(i + 1, len(items), res)
    return out
