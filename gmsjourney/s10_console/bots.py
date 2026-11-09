"""The bot under test as the console configures it, and the playground.

Parts of this file:
  1. ``extract``: text from a file a tester attaches to the LLM bot's prompt (text, Markdown, PDF, Word, Excel).
  2. ``make_factory``: the reference bot, or an LLM bot (gateway or local model, critique layer on by default),
     from a console job's ``sut`` settings; ``suggested`` gives the suggested prompt; ``check_gateway`` one call.
  3. The playground (``start`` / ``Session.say``): talk to the bot on one case; every turn is compared with what the
     graph expects at that point and its words go through the speech check.
"""
from __future__ import annotations

import csv
import io
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =====================================================================================================================
# PART 1 - Text from a file a tester attaches to the LLM bot's prompt (an SOP, a policy PDF, an FAQ sheet).
#
# ``extract(name, data) -> {"name", "kind", "chars", "text", "truncated", "note"}``. Plain text formats are decoded;
# PDF pages are read with pypdf (pdfplumber as a fallback); Word files with python-docx (paragraphs and tables);
# spreadsheets with openpyxl (one line per row). Anything else is refused with the formats that work. The text is
# capped (``MAX_CHARS``) so a large manual cannot crowd out the procedure in the prompt; the cap is reported.
# =====================================================================================================================

MAX_BYTES = 8 * 1024 * 1024
MAX_CHARS = 120_000
TEXT = {".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".csv", ".tsv", ".html", ".htm", ".xml", ".rst", ".log"}
FORMATS = "text (.txt .md .json .yaml .csv .html), PDF, Word (.docx) or Excel (.xlsx)"


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
        if any(x.strip() for x in pages):
            return "\n\n".join(f"[page {i + 1}]\n{x.strip()}" for i, x in enumerate(pages) if x.strip())
    except ImportError:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(f"[page {i + 1}]\n{x.strip()}" for i, x in enumerate(pages) if x.strip())
    except ImportError as e:
        raise ValueError("reading PDF needs pypdf or pdfplumber (pip install pypdf)") from e


def _docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as e:
        raise ValueError("reading Word files needs python-docx (pip install python-docx)") from e
    d = docx.Document(io.BytesIO(data))
    out = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            out.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(out)


def _xlsx(data: bytes) -> str:
    try:
        import openpyxl
    except ImportError as e:
        raise ValueError("reading Excel files needs openpyxl (pip install openpyxl)") from e
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"[sheet {ws.title}]")
        for row in ws.iter_rows(values_only=True):
            if any(c is not None and str(c).strip() for c in row):
                out.append(" | ".join("" if c is None else str(c) for c in row))
    return "\n".join(out)


def extract(name: str, data: bytes) -> dict[str, Any]:
    """Text from an attached file (text, Markdown, JSON, YAML, CSV, PDF, Word, Excel), capped in size, or an error
    that says why the file cannot be used.
    """
    name = Path(name or "attachment").name
    if len(data) > MAX_BYTES:
        raise ValueError(f"{name} is {len(data) / 1e6:.1f} MB; attach a file under {MAX_BYTES // (1024 * 1024)} MB")
    ext = Path(name).suffix.lower()
    if ext in TEXT:
        text, kind = _decode(data), ext.lstrip(".")
        if ext == ".json":                          # pretty, so the model reads keys and values
            try:
                text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
            except ValueError:
                pass
        elif ext in (".csv", ".tsv"):
            rows = list(csv.reader(io.StringIO(text), delimiter="\t" if ext == ".tsv" else ","))
            text = "\n".join(" | ".join(r) for r in rows)
    elif ext == ".pdf":
        text, kind = _pdf(data), "pdf"
    elif ext == ".docx":
        text, kind = _docx(data), "docx"
    elif ext in (".xlsx", ".xlsm"):
        text, kind = _xlsx(data), "xlsx"
    else:
        raise ValueError(f"{name}: this format cannot be read as text; attach {FORMATS}")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ValueError(f"{name} has no readable text (a scanned PDF needs OCR first)")
    cut = len(text) > MAX_CHARS
    return {"name": name, "kind": kind, "chars": min(len(text), MAX_CHARS), "text": text[:MAX_CHARS],
            "truncated": cut, "note": f"only the first {MAX_CHARS:,} of {len(text):,} characters are used" if cut else ""}


# =====================================================================================================================
# PART 2 - The bot under test, as the console configures it: the reference bot of the journey, or an LLM bot on the model
#
# gateway with the tester's prompt and attached files.
#
# A console job carries ``sut``::
#
#     {"kind": "reference"}                                    the pack's reference bot (the default)
#     {"kind": "llm", "provider": "gateway" | "local", "model": "gpt-41", "mode": "live" | "replay", "temperature": 0.0,
#      "max_steps": 6, "critique": true, "instructions": "...", "references": [{"name": "faq.pdf", "text": "..."}]}
#
# ``critique`` (default on) adds the critique layer: a second call of the same model reviews every step the bot
# proposes (each tool call and each reply) against the procedure and approves it or replaces it with a corrected step.
#
# ``provider`` "local" runs Llama (or any chat model) with vLLM from the folder in GMSJOURNEY_SUT_LOCAL_MODEL
# (s06_bots/llm_bot.py); campaign runs only, the Playground refuses it.
#
# ``instructions`` is the editable part of the prompt (who the bot is, how it behaves, the operating procedure); the
# references are added after it as reference material; the tools, dialog acts and reply format the harness needs are
# always appended (``sut.llm_bot.protocol_text``) and are not editable.
# =====================================================================================================================

def suggested(store=None) -> dict[str, Any]:
    """The suggested prompt for the active journey, and the fixed part the harness adds."""
    from ..s01_packs.loading import current
    from ..s06_bots.llm_bot import (SUT_PROMPT_VERSION, act_inventory, journey_title, protocol_text, suggested_instructions, transfer_reasons)
    dom = current()
    src = dom.transcription_path
    return {"pack": dom.name, "title": journey_title(), "instructions": suggested_instructions(),
            "protocol": protocol_text(act_inventory(store), transfer_reasons(store)), "version": SUT_PROMPT_VERSION,
            "policy_source": (f"the policy transcription ({src.name})" if src is not None and src.exists()
                              else f"the policy graph ({dom.graph_path.name}) as text")}


def critique_on(sut: dict[str, Any]) -> bool:
    """The critique layer is on unless the job turns it off."""
    v = (sut or {}).get("critique", True)
    return v not in (False, "off", "false", 0, "0", None)


def settings_for(sut: dict[str, Any]):
    """The model-client settings of a console job's ``sut`` block."""
    from ..llm import LLMSettings
    return LLMSettings(model=str(sut.get("model") or ""), temperature=float(sut.get("temperature") or 0.0),
                       max_tokens=int(sut.get("max_tokens") or 900), mode=str(sut.get("mode") or "live"))


def make_factory(sut: dict[str, Any] | None, store=None):
    """(bot_factory or None for the reference bot, display name, the LLM factory or None)."""
    sut = sut or {}
    if sut.get("kind") != "llm":
        return None, None, None
    from ..s06_bots.llm_bot import fake_model, make_llm_factory
    s = settings_for(sut)
    llm, where = None, "gateway"
    if sut.get("provider") == "local":              # Llama on vLLM, on this machine (campaign runs only)
        from ..llm import CachedLLM
        from ..s06_bots.llm_bot import LocalChat, status
        st = status()
        if not st["usable"]:
            raise RuntimeError(f"the local model cannot be used: {st['detail']}")
        chat = LocalChat(temperature=s.temperature, max_tokens=s.max_tokens)
        s.model = chat.name
        llm, where = CachedLLM(s, model=chat, tag="sut"), "vLLM" + (" server" if st["url"] else ", this machine")
    fac = make_llm_factory(store, s, llm=llm, instructions=sut.get("instructions") or None,
                           references=[r for r in (sut.get("references") or []) if r.get("text")],
                           max_steps=int(sut.get("max_steps") or 6), critique=critique_on(sut))
    name = f"LLM bot ({fac.llm.settings.model}" + (f", {where}" if llm is not None else "") + \
        (", replay" if s.mode == "replay" else "") + (", stand-in model" if llm is None and fake_model() is not None else "") + \
        (", with critique" if fac.critique else "") + ")"
    return fac, name, fac


def check_gateway(model: str = "", timeout: float = 30.0) -> dict[str, Any]:
    """One tiny call through the gateway (tools/check_llm.py in one function): ok, endpoint, seconds, the reply or
    the exact error. A configured stand-in model (GMSJOURNEY_SUT_FAKE_MODEL) is reported instead of calling out."""
    from ..s06_bots.llm_bot import fake_model
    if fake_model() is not None:
        return {"ok": True, "fake": True, "detail": "a stand-in model is configured (GMSJOURNEY_SUT_FAKE_MODEL); "
                                                    "the gateway is not called"}
    from ..llm import LLMSettings, LLMUnavailable, _messages, make_chat_model
    s = LLMSettings(model=model, timeout=timeout)
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
# PART 3 - The playground: talk to the bot under test yourself, one turn at a time, with every turn checked as you go.
#
# A session is one conversation on one case. The case sets the mock bank (the items, which tools fail or decline) and
# offers its customer lines - the scripted words of each turn and, for cases of a finished run, the reworded versions
# that were proved to mean the same. The tester clicks a line or types their own. For every bot turn the session
# shows:
#
# * the tool calls the bot made, with their results (the mock bank's log);
# * what the graph expects at this turn: the oracle replays the customer's turns so far (each read into its meaning
#   card by the frame reader) and gives the tools it would call and the acts it would say; the turn matches when the
#   tools (with their key arguments) and the decision acts are the same (turn_compare's rules); after the first
#   mismatch the conversations may legitimately differ, so later turns are marked "after divergence";
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
    d = rep.as_dict()
    claims = [{"field": c.get("field"), "words": c.get("words"), "status": c.get("status"), "truth": c.get("truth"),
               "why": c.get("why"), "basis": c.get("basis")} for c in d["claims"]]
    adv = set(rep.advisory)
    finds = [{**f, "advisory": f["kind"] in adv, "flow": f["kind"] == "not_allowed_here"} for f in d["findings"]]
    return {"ok": rep.ok, "flow_ok": rep.flow_ok, "heard": d.get("heard"), "claims": claims, "findings": finds}


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
    ended: bool = False
    created: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ what the graph expects
    def _expected(self) -> tuple[list[dict], list[dict]]:
        from ..s09_reports.cases import segment
        B = bench()
        tr = B.oracle.run(self.case["spec"], json.loads(json.dumps(self.case["fixture"])), frames_override=list(self.frames))
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
        if not match and self.diverged_at is None:
            self.diverged_at = len(self.turns)
        self.ended = bool(turn.ended) or self.ended
        out = {"n": len(self.turns), "customer": customer, "bot": {"text": turn.text, "acts": _describe(acts),
               "act_names": [a.get("act") for a in acts], "ended": bool(turn.ended)},
               "tools": tools, "expected": {"tools": [x["text"] for x in e_tools], "acts": _describe(e_acts),
                                           "ends": (not exp_turns or i >= len(exp_turns) - 1) and customer is not None
                                           and bool(e_acts) and str(e_acts[-1].get("act")) in ("END", "TRANSFER")},
               "match": match, "tools_match": tools_match, "after_divergence": after, "speech": _speech(rep),
               "critique": self._revisions_since()}
        self.turns.append(out)
        return out

    def _revisions_since(self) -> list[dict]:
        """The steps the critique layer corrected during this turn (LLM bot with critique on)."""
        st = getattr(self.llm, "stats", None) if self.llm is not None else None
        if st is None or not hasattr(st, "revisions"):
            return []
        revs = st.revisions()
        new, self._rev0 = revs[getattr(self, "_rev0", 0):], len(revs)
        return new

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
                "llm": ({"stats": self.llm.stats.as_dict(), "calls": self.llm.llm.report(),
                         "critique": bool(getattr(self.llm, "critique", False))} if self.llm is not None else None)}

    def _next_line(self) -> int | None:
        """The scripted line that answers the bot's last question (the next turn of the case), if any."""
        n = len(self.frames)
        return n if n < len(self.case["lines"]) and not self.ended else None


_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()


def start(source: str, case_id: str, sut: dict | None) -> dict:
    """A new conversation on a case with the reference bot (``sut`` None or kind "reference") or the LLM bot."""
    from ..s06_bots.tools import ToolEnvironment
    if (sut or {}).get("kind") == "llm" and (sut or {}).get("provider") == "local":
        raise ValueError("the local model is for campaign runs only; in the Playground use the gateway or the reference bot")
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
    """A playground conversation by id (most recently used are kept)."""
    s = _SESSIONS.get(sid)
    if s is None:
        raise KeyError(f"conversation {sid} (it may have been closed; start a new one)")
    _SESSIONS.move_to_end(sid)
    return s
