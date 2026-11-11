"""The console's bot-under-test choice, the case explorer and the playground. No knowledge store, no gateway: the
LLM bot runs on a scripted stand-in model (tests/fake_sut_model.py).

    python tests/test_console_features.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import socket
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gmsjourney.s01_packs import loading as domain  # noqa: E402

FAKE = "fake_sut_model:HappyPath"


def _with_fake(fn):
    def run():
        old = os.environ.get("GMSJOURNEY_SUT_FAKE_MODEL")
        os.environ["GMSJOURNEY_SUT_FAKE_MODEL"] = FAKE
        try:
            fn()
        finally:
            if old is None:
                os.environ.pop("GMSJOURNEY_SUT_FAKE_MODEL", None)
            else:
                os.environ["GMSJOURNEY_SUT_FAKE_MODEL"] = old
            domain.set_current("charge_review")
    run.__name__ = fn.__name__
    return run


# ----------------------------------------------------------------------------- the LLM bot's prompt, any journey
def test_llm_bot_prompt_is_built_from_the_active_journey():
    from gmsjourney.s06_bots.llm_bot import (act_inventory, build_system_prompt, protocol_text, suggested_instructions, transfer_reasons)
    try:
        domain.set_current("supplementary_card")
        acts = act_inventory()
        assert "ASK_RELATIONSHIP" in acts and "PRESENT_TXN" not in acts           # the pack's own dialog states
        assert acts["CONFIRM_APPLICATION"][1] == ["relationship", "card_product"]   # fields from the graph's carries
        reasons = transfer_reasons()
        assert "manual_review" in reasons and "card_options_unavailable" not in reasons
        assert "submit_card_action" not in reasons["submit_failed"]                 # no other journey's tool named
        proto = protocol_text()
        assert "submit_supplementary_application" in proto and "SPOUSE | DOMESTIC_PARTNER" in proto
        assert "Supplementary" in suggested_instructions()
        sp = build_system_prompt(instructions="You are a test bot.", references=[{"name": "faq.txt", "text": "Premium costs more."}])
        assert sp.startswith("You are a test bot.") and "REFERENCE MATERIAL: faq.txt" in sp and "HOW TO REPLY" in sp
        assert sp.index("faq.txt") < sp.index("BACKEND TOOLS")                       # references before the protocol
        domain.set_current("charge_review")
        assert "OPERATING PROCEDURE" in build_system_prompt() and "card_options_unavailable" in transfer_reasons()
    finally:
        domain.set_current("charge_review")


# ----------------------------------------------------------------------------- attachments
def test_attachments_are_read_as_text_and_refused_clearly():
    from gmsjourney.s10_console.bots import MAX_CHARS, extract
    r = extract("faq.md", "# FAQ\n\nStandard delivery is free.".encode())
    assert r["kind"] == "md" and "free" in r["text"] and not r["truncated"]
    assert "a | b" in extract("t.csv", b"a,b\n1,2\n")["text"]
    assert '"fee": 15' in extract("x.json", b'{"fee":15}')["text"]
    big = extract("big.txt", ("x" * (MAX_CHARS + 10)).encode())
    assert big["truncated"] and big["chars"] == MAX_CHARS and big["note"]
    import docx
    d = docx.Document()
    d.add_paragraph("Expedited delivery costs $15.")
    buf = io.BytesIO()
    d.save(buf)
    assert "$15" in extract("sop.docx", buf.getvalue())["text"]
    for name, data in (("pic.png", b"\x89PNG"), ("empty.txt", b"   ")):
        try:
            extract(name, data)
            raise AssertionError(f"{name} should be refused")
        except ValueError as e:
            assert name in str(e)


# ----------------------------------------------------------------------------- case explorer
def _fake_run(root: Path) -> Path:
    out = root / "out"
    (out / "dataset").mkdir(parents=True)
    traj = lambda texts: [{"actor": "bot", "acts": ["OPENING", "ASK_AUTH"], "state": "ASK_AUTH"}] + [   # noqa: E731
        {"actor": "customer", "turn": i, "state": "ASK_AUTH", "card": {"act": "PROVIDE_CREDENTIAL", "slots": {"credential": "4821"}}, "text": t}
        for i, t in enumerate(texts)]
    rows = [{"case_id": "J000~base", "case_type": "base", "arm": "base", "base_id": "J000", "category": "happy", "tags": ["t"],
             "n_turns": 1, "base_case": "T0 [ASK_AUTH] 4821.", "trajectory": traj(["4821."]), "expected_outcome": {"end": "ENDED"},
             "meaning_cards": [], "rules_fired": ["R1"], "states_visited": []},
            {"case_id": "J000~0001", "case_type": "enriched", "arm": "template", "base_id": "J000", "category": "happy", "tags": ["t"],
             "n_turns": 1, "trajectory": traj(["um, four eight two one"]), "enrichment": {"noise": "asr", "style": "terse"},
             "enrichment_changed": "noise=asr", "proof_accepted": True, "proof_status": "accepted", "proof_layers": "L1=True",
             "surface_cosine_mean": 0.3}]
    (out / "dataset" / "test_cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (out / "dataset" / "case_results.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"sid": "J000~base", "correct": 1, "status": "ok", "failure_codes": [], "turns": 1},
        {"sid": "J000~0001", "correct": 0, "status": "failed", "failure_codes": ["OUTCOME_INCORRECT"], "turns": 1})))
    (out / "scenarios.jsonl").write_text(json.dumps({"sid": "J000~0001", "turns": [{"turn": 0, "equivalent": True, "read_back": "PROVIDE_CREDENTIAL(credential=4821)", "surface_cosine": 0.3}]}))
    (out / "report.json").write_text(json.dumps({"results": {"bot": "LLM bot (x)"}, "bases": [{"qid": "J000", "relevant": {"auth": "ok", "items": "one"}, "rules": ["R1"]}],
                                                 "coverage": {"pairs_required": 1, "pairs_covered": 1, "rules_covered": ["R1"], "rules_missing": ["R2"]}}))
    return out


def test_case_explorer_reads_a_runs_cases_back():
    from gmsjourney.s10_console.runs import detail, index
    out = _fake_run(Path(tempfile.mkdtemp()))
    I = index(out)
    assert I["totals"] == {"cases": 1, "reworded": 1, "proved": 1, "tested": 1, "accuracy_original": 1.0, "accuracy_reworded": 0.0}
    c = I["cases"][0]
    assert c["levels"] == {"auth": "ok", "items": "one"} and c["reworded_results"][0]["codes"] == ["OUTCOME_INCORRECT"]
    assert I["wording"] == {"noise": {"asr": 1}, "style": {"terse": 1}} and I["coverage"]["rules_missing"] == ["R2"]
    D = detail(out, "J000")
    base, rew = D["versions"]
    assert base["type"] == "base" and base["turns"][0]["text"] == "4821."
    assert rew["changed"] == ["noise"] and rew["turns"][0]["equivalent"] is True and rew["result"]["correct"] is False


def test_verdict_puts_expected_beside_actual_and_explains_the_gate():
    from gmsjourney.s09_reports.verdict import _plain_reason, _speech, build
    root = Path(tempfile.mkdtemp())
    out = _fake_run(root)
    R = json.loads((out / "report.json").read_text())
    R["release_gate"] = {"base_level": {"passed": False, "tiers": {"action_taking": {"reasons": ["accuracy 0.500 < 0.95", "1 critical failures > 0"]}}},
                         "enriched_template": {"passed": True, "tiers": {}}}
    R["results"]["base_level"] = {"accuracy": 1.0}
    (out / "report.json").write_text(json.dumps(R))
    (out / "failures.jsonl").write_text(json.dumps({"sid": "J000~0001", "why": {
        "expected": {"end": "ENDED", "writes": [["record", {"id": "T1", "answer": "NO"}]]},
        "actual": {"end": "TRANSFERRED", "transfer_reason": "ambiguity", "writes": []},
        "stages": {"record": {"expected": "record:SUCCESS", "actual": "-"}}}}))
    (out / "dataset" / "turn_by_turn.csv").write_text(
        "case_id,turn,state,customer_text,expected_tools,actual_tools,expected_bot,actual_bot_acts,actual_bot_text,speech_check,turn_match,after_divergence\n"
        "J000~0001,0,ASK_AUTH,um four eight,auth() -> OK,-,PRESENT_TXN(T1),ASK_CLARIFY,Sorry?,unread X: y ; contradicted fee '$9',False,False\n")
    (root / "job.json").write_text(json.dumps({"pack": "charge_review_graph", "id": "r1", "code": {"commit": "abc123", "uncommitted_changes": False}}))
    V = build(root)
    assert not V["ready"] and V["gates"][0]["reasons"][0] == "correct 50% of runs, needs 95% (action-taking cases)"
    assert V["risks"][0]["code"] == "OUTCOME_INCORRECT" and V["risks"][0]["what"] and V["fingerprint"]["code"]["commit"] == "abc123"
    fail = V["cases"][0]                                       # failed runs first
    assert fail["sid"] == "J000~0001" and not fail["correct"]
    assert fail["expected"]["writes"] == ["record(id=T1, answer=NO)"] and fail["actual"]["end"] == "TRANSFERRED"
    assert fail["stages"][0]["stage"] == "record" and fail["first_difference"] == "0"
    t = fail["turns"][0]
    assert t["exp_tools"] == "auth() -> OK" and t["bot_tools"] == "" and t["speech"] == "contradicted fee '$9'"   # advisory notes dropped
    ok = V["cases"][1]
    assert ok["correct"] and ok["actual"] == ok["expected"]
    assert _plain_reason("within-budget rate 0.333 < 0.95").startswith("33% of runs within") and _speech("ok") == ""


def test_verdict_is_written_as_files():
    from gmsjourney.s09_reports.verdict import write
    out = _fake_run(Path(tempfile.mkdtemp()))
    w = write(out)
    assert w["runs"] == 2 and w["failed"] == 1 and (out / "verdict.json").exists()
    page = (out / "verdict.html").read_text()
    assert "Not ready to release" in page or "Ready to release" in page
    assert "id='case-J000~0001'" in page and "Expected outcome (policy graph)" in page


# ----------------------------------------------------------------------------- a local model for the bot (vLLM)
class _Out:
    def __init__(self, text):
        self.outputs = [type("O", (), {"text": text, "token_ids": [1, 2]})()]
        self.prompt_token_ids = [1, 2, 3]


class _Engine:
    """Stands in for vllm.LLM: chat(messages, params) -> [RequestOutput]."""
    def __init__(self):
        self.seen = []

    def chat(self, messages, params):
        self.seen.append(messages)
        return [_Out('{"say": [{"act": "END", "text": "Goodbye."}], "end": true}')]


def test_local_model_status_chat_and_the_playground_server_rule():
    from gmsjourney.s06_bots.llm_bot import ENV, LocalChat, status
    from gmsjourney.llm import _messages
    old = {k: os.environ.pop(k, None) for k in (ENV, "GMSJOURNEY_VLLM_URL")}
    try:
        assert not status()["usable"] and ENV in status()["detail"]
        folder = Path(tempfile.mkdtemp()) / "Meta-Llama-3.1-8B-Instruct"
        folder.mkdir()
        os.environ[ENV] = str(folder)
        assert not status()["folder_ok"] and "not a model folder" in status()["detail"]
        (folder / "config.json").write_text("{}")
        st = status()
        assert st["folder_ok"] and st["name"] == "Meta-Llama-3.1-8B-Instruct"
        eng = _Engine()
        chat = LocalChat(engine=eng)
        r = chat.invoke(_messages("You are a bot.", '{"conversation": []}'))
        assert '"END"' in r.content and r.usage_metadata == {"input_tokens": 3, "output_tokens": 2}
        os.environ.pop(ENV)                              # a folder the tester types works the same as the variable
        assert status(str(folder))["folder_ok"] and status(str(folder))["name"] == "Meta-Llama-3.1-8B-Instruct"
        assert "does not exist" in status(str(folder) + "-missing")["detail"]
        assert status(str(folder), "http://localhost:8000/v1")["usable"] and LocalChat(str(folder), url="http://x/v1").url == "http://x/v1"
        os.environ[ENV] = str(folder)
        assert [m["role"] for m in eng.seen[0]] == ["system", "user"] and chat.name == "local-Meta-Llama-3.1-8B-Instruct"
        from gmsjourney.s10_console import bots as PG  # the Playground never loads the local model: a server only
        try:
            PG.start("project", "default", {"kind": "llm", "provider": "local", "local_path": str(folder)})
            raise AssertionError("the playground must refuse a local model without a vLLM server")
        except ValueError as e:
            assert "vLLM server" in str(e)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_local_model_through_a_vllm_server_runs_the_llm_bot():
    """GMSJOURNEY_VLLM_URL: the bot's model calls go to an OpenAI-compatible vLLM server (a stub here)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from fake_sut_model import HappyPath
    from types import SimpleNamespace
    from gmsjourney.s06_bots.llm_bot import ENV
    model = HappyPath()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):                                    # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msgs = [SimpleNamespace(content=m["content"]) for m in body["messages"]]
            text = model.invoke(msgs).content
            out = json.dumps({"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    port = _port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = {k: os.environ.pop(k, None) for k in (ENV, "GMSJOURNEY_VLLM_URL", "GMSJOURNEY_SUT_FAKE_MODEL", "GMSJOURNEY_CACHE")}
    os.environ.update({ENV: "llama-8b-test", "GMSJOURNEY_VLLM_URL": f"http://127.0.0.1:{port}/v1"})
    from gmsjourney.s03_knowledge import build as backend
    keep_cache, backend.CACHE = backend.CACHE, Path(tempfile.mkdtemp(prefix="gms_llm_cache_"))   # every call reaches the server
    try:
        domain.set_current("charge_review_graph")
        from gmsjourney.s10_console.bots import make_factory
        from gmsjourney.s06_bots.tools import ToolEnvironment
        from gmsjourney.s10_console import bots as PG
        PG.cases("project", None)
        case = PG.find_case("project", "default")
        fac, name, _ = make_factory({"kind": "llm", "provider": "local", "mode": "live"}, PG.bench().store)
        assert "llama-8b-test" in name and "vLLM server" in name
        env = ToolEnvironment(json.loads(json.dumps(case["fixture"])))
        bot = fac(env)
        assert bot.start().acts[-1]["act"] == "ASK_AUTH"
        turn = bot.respond("The last four digits of my card are 4821.")
        assert [c["tool"] for c in env.log][:2] == ["authenticate_customer", "get_review_transactions"] and turn.acts
        assert fac.llm.report()["calls"] >= 3 and model.calls >= 3
        assert fac.critique and model.critiques >= 2          # the critique reviewed the replies on the same server
        assert fac.stats.critique_skipped >= 1                # the lookups ran unreviewed (scope "key", the default)
    finally:
        backend.CACHE = keep_cache
        srv.shutdown()
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        domain.set_current("charge_review")


# ----------------------------------------------------------------------------- playground
@_with_fake
def test_playground_plays_a_case_with_the_reference_bot_and_the_llm_bot():
    from gmsjourney.s10_console import bots as PG
    domain.set_current("supplementary_card")
    L = PG.cases("project", None)
    assert L["source"] == "project" and L["cases"][0]["id"] == "default" and len(L["cases"]) > 5
    case = L["cases"][0]
    v = PG.start("project", "default", None)
    assert v["bot"] == "reference" and v["turns"][0]["match"]
    s = PG.get(v["id"])
    for ln in case["lines"]:                         # the scripted lines: the reference bot does what the graph expects
        t = s.say(ln["text"])
        assert t["match"], (ln, t["expected"], t["bot"])
        assert t["speech"] is None or t["speech"]["ok"], t["speech"]
    assert s.view()["ended"] and s.view()["diverged_at"] is None
    try:
        s.say("hello?")
        raise AssertionError("a finished conversation takes no more turns")
    except ValueError:
        pass
    domain.set_current("charge_review_graph")        # the stand-in model plays charge review
    PG.cases("project", None)
    v = PG.start("project", "default", {"kind": "llm", "instructions": "Be brief.", "references": [{"name": "a.txt", "text": "x"}]})
    assert v["bot"] == "llm" and "stand-in" in v["bot_name"] and v["llm"]["stats"]["turns"] == 1
    s = PG.get(v["id"])
    t = s.say("The last four digits of my card are 4821.")
    assert [x["tool"] for x in t["tools"]][:2] == ["authenticate_customer", "get_review_transactions"] and t["match"]
    assert "Be brief." in s.llm.system_prompt and "a.txt" in s.llm.system_prompt


# ----------------------------------------------------------------------------- the API
def _port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


@_with_fake
def test_console_api_for_the_bot_cases_and_playground():
    from gmsjourney.s10_console.server import make_server
    runs = Path(tempfile.mkdtemp())
    port = _port()
    srv = make_server(port, "127.0.0.1", runs)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
    try:
        code, sg = call("api/journey/supplementary_card/sut")
        assert code == 200 and "OPERATING PROCEDURE" in sg["instructions"] and "HOW TO REPLY" in sg["protocol"]
        code, a = call("api/attach", {"name": "n.txt", "data": base64.b64encode(b"hello").decode()})
        assert code == 200 and a["text"] == "hello"
        code, e = call("api/attach", {"name": "n.exe", "data": ""})
        assert code == 400 and "n.exe" in e["error"]
        code, ck = call("api/llm/check", {"model": ""})
        assert code == 200 and ck["ok"] and ck["fake"]
        rid = "20990101-000000-charge_review_graph"            # a finished run with cases, for the explorer
        (runs / rid).mkdir()
        (runs / rid / "job.json").write_text(json.dumps({"pack": "charge_review_graph", "id": rid}))
        _fake_run(runs / rid)
        code, I = call(f"api/runs/{rid}/cases")
        assert code == 200 and I["totals"]["cases"] == 1
        code, D = call(f"api/runs/{rid}/cases/J000")
        assert code == 200 and len(D["versions"]) == 2
        code, V = call(f"api/runs/{rid}/verdict")
        assert code == 200 and len(V["cases"]) == 2 and V["fingerprint"]["journey"] == "charge_review_graph"
        code, L = call("api/play/charge_review_graph/cases?source=project")
        assert code == 200 and L["cases"]
        code, v = call("api/play/charge_review_graph/start", {"source": "project", "case": L["cases"][0]["id"], "sut": {"kind": "reference"}})
        assert code == 200 and v["turns"]
        code, r = call(f"api/play/s/{v['id']}/say", {"text": L["cases"][0]["lines"][0]["text"]})
        assert code == 200 and r["turn"]["match"] and len(r["turns"]) == 2
        code, e = call("api/play/s/nope/say", {"text": "x"})
        assert code == 404
        # a run with the LLM bot keeps its prompt and files beside the job, not in it
        from gmsjourney.s10_console.runs import Runs
        R = Runs(runs)
        from gmsjourney.s10_console import runs as runs_mod
        import subprocess
        from types import SimpleNamespace
        real = runs_mod.subprocess                          # no worker started; git still answers for provenance
        runs_mod.subprocess = SimpleNamespace(Popen=lambda *a, **k: type("P", (), {"poll": lambda self: 0, "pid": 0})(),
                                              run=subprocess.run, SubprocessError=subprocess.SubprocessError, STDOUT=subprocess.STDOUT)
        try:
            rid2 = R.start({"pack": "charge_review_graph", "sut": {"kind": "llm", "instructions": "secret sauce",
                                                                   "references": [{"name": "f.txt", "text": "abc"}]}})
        finally:
            runs_mod.subprocess = real
        job = json.loads((runs / rid2 / "job.json").read_text())
        assert len(job.get("graph_sha") or "") == 12                     # provenance recorded at start
        assert job["sut"]["references"] == [{"name": "f.txt", "chars": 3}] and "secret sauce" not in json.dumps(job)
        assert json.loads((runs / rid2 / "sut.json").read_text())["instructions"] == "secret sauce"
    finally:
        srv.shutdown()


# ----------------------------------------------------------------------------- progress, saved conversations, comparison
def test_progress_reports_reach_a_sink_from_bars_phases_and_llm_calls():
    from gmsjourney import progress
    from gmsjourney.s07_checks.scoring import progress_bar
    got = []
    progress.set_sink(lambda **e: got.append(e))
    try:
        b = progress_bar(3, "SUT · base cases", show=False)        # counted even with no terminal bar
        for _ in range(3):
            b.update(1)
        b.set_postfix_str("correct 3/3")
        b.close()
        bars = [e for e in got if e["key"] == "sut_base_cases"]
        assert bars[0]["done"] == 0 and bars[-1]["done"] == 3 and bars[-1]["total"] == 3 and bars[-1]["final"]
        assert bars[-1]["label"] == "Bot runs, original wording" and bars[-1]["note"] == "correct 3/3"
        progress.phase("gms", 0, 2, "first", phases=["first", "second"], expect_s=4.0)
        progress.phase("gms", 2, 2, "done")
        ph = [e for e in got if e["key"] == "gms"]
        assert ph[0]["phases"] == ["first", "second"] and ph[0]["expect_s"] == 4.0 and ph[-1]["done"] == ph[-1]["total"]
        for outcome in ("cached", "live", "error"):
            progress.llm_call("sut", outcome, 0.1, 5)
        progress.llm_flush()
        llm = [e for e in got if e["key"] == "llm"][-1]
        assert (llm["done"], llm["cached"], llm["live"], llm["errors"]) == (3, 1, 1, 1) and llm["by_tag"]["sut"]["calls"] == 3
    finally:
        progress.set_sink(None)
    assert not progress.active()
    assert progress_bar(3, "x", show=False) is None                 # nobody listening, no terminal: no bar at all


def test_the_run_step_hook_reads_both_step_counts():
    import re
    from gmsjourney.s08_campaign import pipeline
    src = Path(pipeline.__file__).read_text()
    pat = re.search(r're\.match\(r"(.+?)", str\(msg\)\)', src).group(1)
    assert re.match(pat, "[1/13] building the store").group(1) == "1"
    assert re.match(pat, "[5b/13] LLM arm").group(1) == "5b"
    assert re.match(pat, "[10/12] mutation analysis").group(1) == "10"


@_with_fake
def test_a_playground_conversation_saved_as_a_test_case_is_replayed():
    from gmsjourney.s10_console import bots as PG
    domain.set_current("charge_review_graph")
    f = PG._saved_file()
    before = f.read_text() if f.exists() else None
    try:
        PG.cases("project", None)
        s = PG.get(PG.start("project", "default", None)["id"])
        s.say("", "journey")
        s.say("No, that's not mine, I never made that purchase.", "free")
        rec = PG.save_conversation(s.id, "", "keep this")
        assert rec["lines"][0]["mode"] == "journey" and rec["lines"][1]["text"].startswith("No, that's not mine")
        assert rec["name"].startswith("default:") and rec["note"] == "keep this" and rec["saved_with"]["first_deviation"] is None
        assert [x["id"] for x in PG.saved_list()][0] == rec["id"]
        r = PG.replay(rec)                                           # the reference bot: every turn as the graph says
        assert r.replay_of == rec["id"] and [t["customer"]["mode"] for t in r.turns if t["customer"]] == ["replay", "replay"]
        assert not r.deviations
        res = PG.replay_all({"kind": "llm", "critique": False})       # the stand-in model deviates: flagged on replay
        mine = next(x for x in res if x["id"] == rec["id"])
        assert not mine["passed"] and mine["first_deviation"] is not None and mine["why"] and mine["sent"] == 2, mine
        off = PG.replay(rec, {"kind": "llm", "critique": False})      # the same turns as a live conversation
        on = PG.replay(rec, {"kind": "llm", "critique": True})
        assert not off.turns[2]["tools_match"] and on.turns[2]["tools_match"]   # it records the wrong answer; critique fixes it
        PG.delete_saved(rec["id"])
        assert rec["id"] not in [x["id"] for x in PG.saved_list()]
        try:
            PG.save_conversation(PG.start("project", "default", None)["id"])
            raise AssertionError("a conversation with no customer line cannot be saved")
        except ValueError:
            pass
    finally:
        if before is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(before)


def test_two_runs_are_compared_case_by_case():
    import csv
    from gmsjourney.s10_console.runs import compare
    root = Path(tempfile.mkdtemp())
    cols = ["sid", "base", "correct", "failure_codes", "c_record", "c_said"]
    for rid, rows, head, sha in (
            ("r1", [["a~base", "A", "True", "", "1", "1"], ["b~base", "B", "False", "wrong_record", "0", "1"],
                    ["c~base", "C", "True", "", "1", "1"], ["x~base", "X", "True", "", "1", "1"]],
             {"accuracy_original": 0.75, "gate": False, "bot": "one"}, "aaa"),
            ("r2", [["a~base", "A", "False", "said_wrong", "1", "0"], ["b~base", "B", "True", "", "1", "1"],
                    ["c~base", "C", "True", "", "1", "1"]],
             {"accuracy_original": 0.67, "gate": True, "bot": "two"}, "bbb")):
        d = root / rid / "out"
        d.mkdir(parents=True)
        (root / rid / "job.json").write_text(json.dumps({"pack": "charge_review_graph", "seed": 42}))
        (root / rid / "headline.json").write_text(json.dumps(head))
        (d / "report.json").write_text(json.dumps({"provenance": {"graph_sha": sha}}))
        with open(d / "results_base_level.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(rows)
    C = compare(root, "r1", "r2")
    assert [c["sid"] for c in C["now_failing"]] == ["a~base"] and C["now_failing"][0]["codes_b"] == "said_wrong"
    assert [c["sid"] for c in C["now_passing"]] == ["b~base"] and C["still_passing"] == 1 and not C["still_failing"]
    assert C["shared"] == 3 and C["only_a"] == 1 and C["only_b"] == 0
    assert any("graph changed" in c for c in C["caveats"]) and any("only in r1" in c for c in C["caveats"])
    st = {x["stage"]: (x["a"], x["b"]) for x in C["stages"]}
    assert st["said"] == (1.0, 2 / 3) and st["record"] == (2 / 3, 1.0)
    assert C["gates"][0]["a"] is False and C["gates"][0]["b"] is True
    try:
        compare(root, "r1", "nope")
        raise AssertionError("an unknown run is refused")
    except KeyError:
        pass


def test_the_graph_path_preview_says_what_opening_would_do():
    from gmsjourney.s10_console.graphs import peek_graph
    p = peek_graph("gmsjourney/policy/supplementary_card/supplementary_card_kg.json")
    assert p["ok"] and p["kind"] == "compact graph" and p["counts"]["states"] > 5 and "supplementary_card" in p["action"]
    assert "supplementary_card" in " ".join(peek_graph("gmsjourney/policy/supp")["suggest"])
    assert not peek_graph("README.md")["ok"] and not peek_graph("")["ok"]
    t = peek_graph("gmsjourney/policy/charge_review/charge_review_kg.json")
    assert t["kind"] == "triples graph" and t["ok"] and t["counts"]["triples"] > 100


def test_confusion_is_unclear_not_no_and_curly_apostrophes_match_cues():
    from gmsjourney.s07_checks.speech_reading import plain, words_to_digits
    from gmsjourney.s10_console import bots as PG
    assert plain("I\u2019m sorry, I didn\u2019t catch \u201cthat\u201d \u2014 ok\u2026") == 'I\'m sorry, I didn\'t catch "that" - ok...'
    assert "didn't catch" in words_to_digits("I didn\u2019t catch that").lower()
    for pack in ("charge_review", "charge_review_graph", "supplementary_card"):
        domain.set_current(pack)
        r = PG.bench().reader
        for t in ("I don't really understand what you are saying", "I don\u2019t understand", "what do you mean?", "I'm confused"):
            assert r.read(t, None).act == "UNCLEAR", (pack, t, r.read(t, None))
        assert r.read("No.", None).act == "DENY" and r.read("No, don't do that.", None).act == "DENY"
    domain.set_current("charge_review")
    r = PG.bench().reader
    assert str(r.read("not mine, I don't understand why it's there", None)) == "INFORM_DISPOSITION(disposition=UNRECOGNIZED)"
    assert str(r.read("I don't know", None)) == "INFORM_DISPOSITION(disposition=UNSURE)"


def test_the_words_check_hears_a_clarification_written_with_curly_apostrophes():
    from gmsjourney.s10_console import bots as PG
    domain.set_current("charge_review")
    PG.cases("project", None)
    s = PG.get(PG.start("project", "default", None)["id"])
    s.say("", "journey")
    s.say("I don\u2019t really understand what you are saying", "free")
    c = s.turns[-1]["customer"]
    assert c["card"]["act"] == "UNCLEAR" and c["state_reading"] is None
    chk = PG.bench().speech.conversation(s.case["fixture"].get("session_date"))
    rep = chk.check([{"act": "ASK_CLARIFY", "txn": "T1"}], "I\u2019m sorry, I didn\u2019t catch whether you recognize the "
                    "Amazon charge for $45.67 on March 13th. Do you recognize this charge?", s.env.log, list(s.frames))
    assert "ASK_CLARIFY" in rep.heard and not [f for f in rep.findings if f.kind == "label_not_said"]
    s2 = PG.get(PG.start("project", "default", None)["id"])                 # a bare "No.": shown as the graph got it
    s2.say("", "journey")
    s2.say("No.", "free")
    c2 = s2.turns[-1]["customer"]
    assert c2["card"]["act"] == "DENY" and c2["state_reading"]["slots"] == {"disposition": "UNRECOGNIZED"}
    assert s2.turns[-1]["expected"]["acts"] == ["ASK_CLARIFY(T1)"]


def test_the_playground_journey_line_follows_the_base_journey_not_the_bot():
    from gmsjourney.s10_console import bots as PG
    domain.set_current("charge_review_graph")
    PG.cases("project", None)
    s = PG.get(PG.start("project", "default", None)["id"])
    base = [str(f) for f, _, _ in s._base_lines()]
    assert s.journey_line()["read"] == base[0]
    s.say("", "journey")
    assert s.base_i == 1 and s.journey_line()["read"] == base[1]
    s.say("Sorry, what was that?", "free")                       # an extra turn: the journey's line stays next
    w = s.turns[-1]["customer"]["journey_would_say"]
    assert s.base_i == 1 and w["read"] == base[1] and not w["same_meaning"]
    s.say("Yes, that one is mine.", "free")                        # the journey's line in other words: it moves on
    assert s.turns[-1]["customer"]["journey_would_say"]["same_meaning"] and s.base_i == 2
    while s.journey_line() and not s.ended:
        s.say("", "journey")
    assert s.base_i == len(base) and s.journey_line() is None
    if not s.ended:
        try:
            s.say("", "journey")
            raise AssertionError("a journey line past the base journey")
        except ValueError as e:
            assert "no more customer lines" in str(e)


def test_confusion_in_many_wordings_and_the_customer_says_it_in_several_ways():
    from gmsjourney.s01_packs import loading
    from gmsjourney.s05_rewording.wording import SemanticFrame as F
    from gmsjourney.s10_console import bots as PG
    confused = ["I don't get what you mean", "Sorry, I didn't understand the question", "You lost me there", "Huh?",
                "Can you rephrase that?", "What are you asking me?", "That makes no sense", "That doesn't make any sense",
                "Could you clarify?", "No idea what you're talking about", "Not sure what you mean", "Come again?"]
    for pack in ("charge_review", "charge_review_graph", "supplementary_card"):
        domain.set_current(pack)
        assert loading.lint(loading.current()) == [], pack            # every variant reads back as UNCLEAR
        r = PG.bench().reader
        assert [t for t in confused if r.read(t, None).act != "UNCLEAR"] == [], pack
        rz = PG.bench().realizer
        lines = {rz.realize(F.make("UNCLEAR"), {"paraphrase_depth": d}, f"case{i}") for d in ("light", "deep") for i in range(40)}
        assert len(lines) >= 8, (pack, lines)                         # four ways per reworded depth
        assert {rz.realize(F.make("UNCLEAR"), {"paraphrase_depth": "none"}, f"case{i}") for i in range(10)} == {"Hmm, sorry, what was that?"}
    domain.set_current("charge_review")
    r = PG.bench().reader                                             # answers that merely contain such words stay answers
    assert str(r.read("not mine, I don't understand why it's there", None)) == "INFORM_DISPOSITION(disposition=UNRECOGNIZED)"
    assert str(r.read("I'm not sure about that one", None)) == "INFORM_DISPOSITION(disposition=UNSURE)"
    assert r.read("I don't understand, can I talk to a real person?", None).act == "REQUEST_HUMAN"
    assert r.read("Can you explain the expedited fee?", None).act == "ASK_KB"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)


def test_a_test_run_scripts_the_customer_by_default_and_the_run_page_can_turn_it_off():
    from gmsjourney.s08_campaign.pipeline import CampaignConfig
    from gmsjourney.s10_console import runs
    assert CampaignConfig().customer == "scripted" and "customer" in runs.OPTIONS
    html = (Path(runs.__file__).parent / "ui.html").read_text()
    assert "Customer follows the base journey" in html and 'customer: F.scripted_customer === false ? "adaptive" : "scripted"' in html
    with tempfile.TemporaryDirectory() as d:
        try:
            runs.Runs(Path(d)).start({"pack": "charge_review", "customer": "whatever"})
            raise AssertionError("an unknown customer was accepted")
        except ValueError as e:
            assert "scripted" in str(e)


# ---------------------------------------------------------------------------- the two LLM judges, the tools' LLM
class _FakeJudge:
    """A stand-in judge: calls a run incorrect when the bot transferred (and records what it was shown)."""

    def __init__(self, who):
        self.who, self.seen = who, []

    def invoke(self, msgs):
        from types import SimpleNamespace
        u = msgs[-1].content
        self.seen.append(u)
        bad = "TRANSFER" in u.upper() if self.who == "a" else "transfer" in u.lower()
        return SimpleNamespace(content='{"verdict": "%s", "first_wrong_turn": null, "reason": "stand-in %s"}'
                               % ("incorrect" if bad else "correct", self.who))


def _judges():
    from gmsjourney.llm import CachedLLM, LLMSettings
    from gmsjourney.s07_checks.llm_judges import Judges
    d = tempfile.mkdtemp()
    fa, fb = _FakeJudge("a"), _FakeJudge("b")
    return Judges(llm_a=CachedLLM(LLMSettings(cache_dir=d, backend="gateway"), model=fa, tag="judge_a"),
                  llm_b=CachedLLM(LLMSettings(cache_dir=d, backend="gateway"), model=fb, tag="judge_b")), fa, fb


def test_two_llm_judges_sit_beside_the_rules_and_see_different_material():
    from gmsjourney.s07_checks.llm_judges import combine, summarize
    c = lambda v: {"verdict": v}                                              # noqa: E731
    assert combine(True, c("correct"), c("correct"))["status"] == "agree"
    assert combine(True, c("incorrect"), c("incorrect"))["status"] == "judges disagree with the rules"
    assert combine(False, c("correct"), c("incorrect"))["status"] == "judges split"
    assert combine(True, c("unavailable"), c("unavailable"))["status"] == "judges unavailable"
    J, fa, fb = _judges()
    case = {"case_id": "X~base", "rules_ok": True, "expected_outcome": {"end": "END"}, "actual_outcome": {"end": "TRANSFER"},
            "turns": [{"turn": 0, "state": "ASK_AUTH", "customer_text": "4821", "customer_card": "{}", "expected_tools": "auth()",
                       "expected_bot": "PRESENT_TXN", "actual_tools": "auth()", "actual_bot_acts": "TRANSFER", "actual_bot_text": "Transferring you."}],
            "rules": {"R1": "Verify the caller first."}}
    r = J.judge_case(case)
    assert r["a"]["verdict"] == "incorrect" and r["b"]["verdict"] == "incorrect"
    assert r["combined"]["status"] == "judges disagree with the rules"
    assert "REQUIRED" in fa.seen[0] and "REQUIRED" not in fb.seen[0] and "R1: Verify the caller first." in fb.seen[0]
    s = summarize([r])
    assert s["judged"] == 1 and s["a_agrees_with_rules"] == 0.0 and s["to_review"][0]["case_id"] == "X~base"
    again = J.judge_case(case)                                                # cached: no second call
    assert again["a"] == r["a"] and len(fa.seen) == 1


def test_the_playground_shows_both_judges_on_every_bot_turn():
    from gmsjourney.s10_console import bots as PG
    domain.set_current("charge_review_graph")
    J, fa, fb = _judges()
    PG._JUDGES[:] = [J]
    try:
        PG.cases("project", None)
        s = PG.get(PG.start("project", "default", None, judge=True)["id"])
        assert s.turns[0]["judges"]["combined"]["status"] in ("agree", "judges disagree with the rules", "judges split")
        s.say("", "journey")
        t = s.turns[-1]
        assert t["judges"]["a"]["verdict"] in ("correct", "incorrect") and t["judges"]["combined"]["rules"] == ("incorrect" if t["deviated"] else "correct")
        assert "Customer said" in fa.seen[-1] and "REQUIRED" in fa.seen[-1]
        assert "conversation so far" in fb.seen[-1] and "REQUIRED" not in fb.seen[-1]
        s.judges = False
        s.say("", "journey")
        assert "judges" not in s.turns[-1] and s.view()["judges"] is False
    finally:
        PG.reset_judges()


def test_the_test_tools_llm_can_run_on_a_local_model():
    from gmsjourney import llm as L
    from gmsjourney.s10_console import bots as PG
    keep = {k: os.environ.get(k) for k in ("GMSJOURNEY_LLM_BACKEND", "GMSJOURNEY_LLM_LOCAL_MODEL", "GMSJOURNEY_LLM_VLLM_URL")}
    try:
        L.use_backend("local", "/models/Llama-3.1-8B-Instruct", "http://localhost:8000/v1")
        s = L.LLMSettings(model="gpt-41")
        assert s.backend == "local" and s.model == "local-Llama-3.1-8B-Instruct" and s.vllm_url.endswith("/v1")
        assert s.public()["where"].startswith("local model")
        chat = L.make_chat_model(s)
        assert type(chat).__name__ == "LocalChat" and chat.url == "http://localhost:8000/v1"
        assert PG.settings_for({"kind": "llm", "model": "gpt-41"}).backend == "gateway"     # the bot keeps its own
        L.use_backend("local", "", "")
        try:
            L.make_chat_model(L.LLMSettings())
            raise AssertionError("an unusable local model was accepted")
        except L.LLMUnavailable as e:
            assert "local model" in str(e)
        L.use_backend("gateway")
        assert L.LLMSettings().backend == "gateway"
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_the_sphere_is_placed_by_the_trained_stores_angular_distances():
    import numpy as np
    from types import SimpleNamespace
    from gmsjourney.s03_knowledge.build import export_entity_points
    from gmsjourney.s10_console import graphs as GR
    dom = domain.set_current("charge_review_graph")
    names, _edges, sec, _ty = GR._entity_graph(dom.kg)
    rng = np.random.default_rng(0)
    cen = {s: rng.normal(size=24) for s in set(sec.values())}
    W = np.stack([cen[sec[n]] + 0.3 * rng.normal(size=24) for n in names])
    model = SimpleNamespace(state_dict=lambda: {"rel_emb.weight": np.zeros((4, 24)), "ent_emb.weight": W})
    kx = SimpleNamespace(adapter=SimpleNamespace(entity_to_idx={n: i for i, n in enumerate(names)}), model=model)
    d = tempfile.mkdtemp()
    info = export_entity_points(kx, d)
    assert info["entities"] == len(names) and info["source"] == "model.ent_emb.weight"
    V = GR.build(dom, store_dir=d)
    assert V["placement"] == "gms" and V["fit"]["angle_correlation"] > 0.8, V["fit"]
    assert "angular distances" in V["note"]
    assert GR.build(dom, store_dir=tempfile.mkdtemp())["placement"] == "graph"


def test_the_answer_sheet_lists_every_case_its_answer_key_and_its_rewordings():
    from gmsjourney.s02_graph.compile import compile_graph
    from gmsjourney.s02_graph.interpreter import GraphOracle
    from gmsjourney.s04_cases.factors import mine_paths, to_item
    from gmsjourney.s05_rewording.catalog import Scenario
    from gmsjourney.s05_rewording.proof import GraphInvarianceValidator
    from gmsjourney.s05_rewording.wording import FrameReader, Realizer
    from gmsjourney.s09_reports.answer_sheet import write_answer_sheet
    dom = domain.set_current("supplementary_card")
    store = compile_graph(dom.graph_path).store
    oracle = GraphOracle(store)
    bases = [to_item(sp, oracle, f"G{i:03d}") for i, (sp, _) in enumerate(mine_paths(oracle)[:3])]
    val = GraphInvarianceValidator(store, oracle, FrameReader(store), Realizer())
    proofs = [val.prove(Scenario(f"{b.qid}~0001", b, {}, "enriched")) for b in bases]
    d = tempfile.mkdtemp()
    info = write_answer_sheet(d, oracle=oracle, store=store, validator=val, bases=bases, proofs=proofs, journey=dom.name)
    html_ = (Path(d) / "answer_sheet.html").read_text()
    assert info == {"file": "answer_sheet.html", "cases": 3, "rewordings": 3, "accepted": 3}
    assert all(b.qid in html_ for b in bases) and "A correct bot must, in order" in html_ and "proved: same answer key" in html_


def test_the_rewording_llm_and_the_judges_llm_are_chosen_separately():
    from gmsjourney import llm as L
    from gmsjourney.s10_console import server as SV
    keys = [k for k in os.environ if k.startswith("GMSJOURNEY_LLM_")]
    keep = {k: os.environ[k] for k in keys}
    old = SV._TOOLS_LLM
    SV._TOOLS_LLM = Path(tempfile.mkdtemp()) / "console_llm.json"
    try:
        SV.save_tools_llm({"role": "enrichment", "backend": "gateway", "model": "gpt-55"})
        out = SV.save_tools_llm({"role": "testing", "backend": "local", "local_path": "/models/Qwen2.5-7B", "vllm_url": "http://127.0.0.1:9/v1"})
        assert out["enrichment"]["backend"] == "gateway" and out["enrichment"]["model"] == "gpt-55"
        assert out["testing"]["backend"] == "local" and out["testing"]["usable"]
        assert L.LLMSettings(role="enrichment").model == "gpt-55" and L.LLMSettings(role="enrichment").backend == "gateway"
        assert L.LLMSettings(role="testing").model == "local-Qwen2.5-7B"
        from gmsjourney.s07_checks.llm_judges import Judges
        assert Judges().a.settings.backend == "local"                      # the judges follow the testing choice
        saved = json.loads(SV._TOOLS_LLM.read_text())
        assert set(saved) == {"enrichment", "testing"}
        try:
            SV.save_tools_llm({"role": "everything", "backend": "local"})
            raise AssertionError("an unknown role was accepted")
        except ValueError:
            pass
        html = (Path(SV.__file__).parent / "ui.html").read_text()
        assert 'llmPick("enrichment", "rf-enr", "Rewording LLM")' in html and 'llmPick("testing", "rf-jdg"' in html
        assert 'llmPick("testing", "pg-jdg"' in html
    finally:
        SV._TOOLS_LLM = old
        for k in [k for k in os.environ if k.startswith("GMSJOURNEY_LLM_")]:
            os.environ.pop(k)
        os.environ.update(keep)


def test_the_validation_report_shows_the_rules_and_both_judges_for_every_run():
    from gmsjourney.s07_checks.diagnosis import render_bot_validation_html
    rows = [{"sid": "J000~base", "arm": "base", "correct": 1, "workflow_adherent": 1, "outcome_correct": 1, "c_speech": 1},
            {"sid": "J001~base", "arm": "base", "correct": 0, "workflow_adherent": 1, "outcome_correct": 0, "c_speech": 1,
             "c_transfer": 0, "failure_codes": ["ESCALATION_MISSED"]}]
    j = {"J000~base": {"a": {"verdict": "correct", "reason": "fine"}, "b": {"verdict": "incorrect", "reason": "promised a refund"},
                       "combined": {"status": "judges split"}},
         "J001~base": {"a": {"verdict": "incorrect", "reason": "no transfer"}, "b": {"verdict": "incorrect", "reason": "x"},
                       "combined": {"status": "agree"}}}
    page = render_bot_validation_html({"results": {"bot": "B"}, "llm_judges": {"model": "local-X (local model)"}}, rows, j)
    assert "Three verdicts per run" in page and "judge A, the auditor" in page and "promised a refund" in page
    assert "stages wrong: transfer" in page and "ESCALATION_MISSED" in page and "local-X" in page
    off = render_bot_validation_html({"results": {"bot": "B"}}, rows, {})
    assert "were off for this run" in off


def test_the_playground_customer_plays_on_after_the_bot_ends_the_call():
    from gmsjourney.s10_console import bots as PG
    domain.set_current("charge_review_graph")
    PG.cases("project", None)
    s = PG.get(PG.start("project", "default", None)["id"])
    s.say("", "journey")
    s.say("I want to talk to a real person.", "free")                 # the bot hands over: the call ends here
    assert s.ended and s.ended_turn == len(s.turns) - 1
    left = len(s._base_lines()) - s.base_i
    assert left > 0 and s.journey_line() is not None                  # the journey still has the customer's lines
    s.say("", "journey")
    t = s.turns[-1]
    assert any("already ended the call" in w for w in t["why"]) and t["deviated"]
    while s.journey_line() is not None:
        s.say("", "journey")
    assert s.base_i == len(s._base_lines())
    try:
        s.say("", "journey")
        raise AssertionError("a line past the end of the journey after a hang-up")
    except ValueError as e:
        assert "no more customer lines" in str(e)
    assert s.view()["ended_turn"] == 2


def test_gateway_clients_share_one_token_one_connection_pool_and_what_the_deployment_rejects():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from gmsjourney import llm as L
    seen = {"posts": 0, "ports": set(), "rejected": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"                                  # keep-alive

        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["posts"] += 1
            seen["ports"].add(self.client_address[1])
            if any(m["role"] == "system" for m in body["messages"]):    # this deployment rejects the system role
                seen["rejected"] += 1
                out, code = json.dumps({"error": "the system role is not supported"}).encode(), 400
            else:
                out, code = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode(), 200
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    calls = []
    real = L._fetch_token
    L._fetch_token = lambda s: (calls.append(1), "tok")[1]
    L._TOKENS.clear()
    L._ADAPTED.clear()
    keep = os.environ.pop("AEXP_ACCESS_TOKEN", None)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1/{{model}}/chat"
        mk = lambda: L.make_chat_model(L.LLMSettings(url=url, backend="gateway", send_sampling_params=False))   # noqa: E731
        bot, critic, judge = mk(), mk(), mk()
        for c in (bot, critic, judge, bot, critic, judge):
            assert c.invoke(L._messages("sys", "hi")).content == "ok"
        assert len(calls) == 1                                          # one token for every client
        assert seen["rejected"] == 1                                    # learnt once, not once per client
        assert len(seen["ports"]) == 1                                  # one kept-alive connection
        assert critic.system_role is False and judge.system_role is False
    finally:
        L._fetch_token = real
        L._TOKENS.clear()
        L._ADAPTED.clear()
        if keep is not None:
            os.environ["AEXP_ACCESS_TOKEN"] = keep
        L._session().close()                                            # drop the kept-alive connection
        srv.shutdown()


def test_identical_conversations_are_judged_once():
    J, fa, fb = _judges()
    case = {"turns": [{"turn": 0, "state": "S", "customer_text": "hi", "expected_tools": "-", "expected_bot": "A",
                       "actual_tools": "-", "actual_bot_acts": "A", "actual_bot_text": "hello"}],
            "expected_outcome": {"end": "END"}, "actual_outcome": {"end": "END"}, "rules": {}, "rules_ok": True}
    J.judge_case({**case, "case_id": "J001~base"})
    r = J.judge_case({**case, "case_id": "J001~0004"})                  # the same conversation in another run
    assert len(fa.seen) == 1 and len(fb.seen) == 1 and r["case_id"] == "J001~0004"


def test_meaning_proofs_are_reused_only_when_nothing_they_depend_on_changed():
    from gmsjourney.s02_graph.compile import compile_graph
    from gmsjourney.s02_graph.interpreter import GraphOracle
    from gmsjourney.s04_cases.factors import mine_paths, to_item
    from gmsjourney.s05_rewording import proof_cache as PC
    from gmsjourney.s05_rewording.catalog import Scenario
    from gmsjourney.s05_rewording.proof import GraphInvarianceValidator
    from gmsjourney.s05_rewording.wording import FrameReader, Realizer
    dom = domain.set_current("supplementary_card")
    store = compile_graph(dom.graph_path).store
    oracle = GraphOracle(store)
    bases = [to_item(sp, oracle, f"G{i:03d}") for i, (sp, _) in enumerate(mine_paths(oracle)[:3])]
    val = GraphInvarianceValidator(store, oracle, FrameReader(store), Realizer())
    lv = {}
    scns = lambda: [Scenario(f"{b.qid}~0001", b, dict(lv), "enriched") for b in bases]   # noqa: E731
    root = tempfile.mkdtemp()
    first = PC.ProofCache(root, dom, val)
    fresh = [first.prove(s) for s in scns()]
    assert first.stats()["proved"] == 3 and first.stats()["reused"] == 0
    again = PC.ProofCache(root, dom, val)
    s2 = scns()
    reused = [again.prove(s) for s in s2]
    assert again.stats()["reused"] == 3 and again.stats()["proved"] == 0
    for a, b, s in zip(fresh, reused, s2):
        assert PC.encode(a, {}) == PC.encode(b, {})                                 # the same proof, turn for turn
        assert b.turns[0]["frame_obj"] == a.turns[0]["frame_obj"] and s.metadata["proof"]["accepted"] == a.accepted
        assert "seed_overrides" in s.metadata
    other = [Scenario(f"{b.qid}~0002", b, dict(lv), "enriched") for b in bases]   # another case: proved
    third = PC.ProofCache(root, dom, val)
    [third.prove(s) for s in other]
    assert third.stats()["proved"] == 3
    off = PC.ProofCache(root, dom, val, reuse=False)                               # reuse off: proved again
    [off.prove(s) for s in scns()]
    assert off.stats()["proved"] == 3
    real = PC.pack_fingerprint
    PC.pack_fingerprint = lambda d: "an-edited-journey"                            # the journey's files changed
    try:
        changed = PC.ProofCache(root, dom, val)
        [changed.prove(s) for s in scns()]
        assert changed.stats()["proved"] == 3 and changed.dir != first.dir
    finally:
        PC.pack_fingerprint = real


def test_a_reasoning_model_on_vllm_gets_a_thinking_budget_and_low_effort():
    from gmsjourney.s06_bots.llm_bot import LocalChat, is_reasoning_model
    assert is_reasoning_model("openai/gpt-oss-20b") and not is_reasoning_model("/models/Llama-3.1-8B")
    b = LocalChat("openai/gpt-oss-20b", temperature=0.0, max_tokens=300, url="http://x/v1")._body([])
    assert b["reasoning_effort"] == "low" and b["max_tokens"] == 300 + 1024 and b["model"] == "openai/gpt-oss-20b"
    plain = LocalChat("/models/Llama-3.1-8B", temperature=0.0, max_tokens=300, url="http://x/v1")._body([])
    assert "reasoning_effort" not in plain and plain["max_tokens"] == 300
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "gpt_oss_20b_vllm.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "vllm.entrypoints.openai.api_server" in src and "api/llm/settings" in src and "save_tools_llm" in src
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "cell", "exec")                       # every cell is valid Python


def test_a_vllm_server_error_is_shown_and_an_unsupported_field_is_dropped():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from gmsjourney.s06_bots.llm_bot import LocalChat
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(sorted(body))
            if "reasoning_effort" in body:                                   # an older server: rejects the field
                code, out = 400, {"message": "extra fields not permitted: reasoning_effort"}
            elif body["messages"][-1]["content"] == "boom":
                code, out = 500, {"message": "HarmonyError: error downloading or loading vocab file"}
            else:
                code, out = 200, {"choices": [{"message": {"content": "OK", "reasoning_content": "hmm"}}]}
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        from gmsjourney.llm import _messages
        _m = lambda t: _messages("Be brief.", t)                               # noqa: E731
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        chat = LocalChat("openai/gpt-oss-20b", temperature=0.0, max_tokens=50, url=url)
        assert chat.invoke(_m("hi")).content == "OK"
        assert "reasoning_effort" in seen[0] and "reasoning_effort" not in seen[1]     # retried once without it
        chat.invoke(_m("again"))
        assert "reasoning_effort" not in seen[2]                                      # and not sent again
        try:
            chat.invoke(_m("boom"))
            raise AssertionError("a server error was swallowed")
        except RuntimeError as e:
            assert "HTTP 500" in str(e) and "vocab file" in str(e)
    finally:
        srv.shutdown()


def test_a_bot_that_writes_a_reference_as_an_object_is_graded_not_crashed():
    from gmsjourney.s06_bots.harness import _clean
    from gmsjourney.s06_bots.llm_bot import _ref
    from gmsjourney.s10_console import bots as PG
    assert _ref({"transaction_id": "T1", "merchant": "Amazon"}) == "T1" and _ref({"T1": "RECOGNIZED"}) == {"T1": "RECOGNIZED"}
    domain.set_current("charge_review_graph")
    PG.cases("project", None)
    plain = PG.get(PG.start("project", "default", None)["id"])
    odd = PG.get(PG.start("project", "default", None)["id"])
    real = odd.bot.respond

    def as_objects(text):                                    # the same bot, writing every reference as an object
        t = real(text)
        for a in t.acts:
            for k, v in list(a.items()):
                if k != "act" and isinstance(v, str) and v.startswith("T"):
                    a[k] = {"transaction_id": v, "merchant": "?"}
        return t
    odd.bot.respond = as_objects
    for s in (plain, odd):
        while s.journey_line() is not None and not (s.ended and s.journey_line() is None):
            s.say("", "journey")
    assert [t["deviated"] for t in odd.turns] == [t["deviated"] for t in plain.turns]
    t = _clean(type("T", (), {"acts": [{"act": "PRESENT_TXN", "txn": {"id": "T2"}}]})())
    assert t.acts[0]["txn"] == "T2"


def test_a_throttling_gateway_slows_every_client_down_instead_of_failing_the_run():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from gmsjourney import llm as L
    state = {"n": 0, "always": False}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                state["n"] += 1
                busy = state["always"] or state["n"] <= 3                    # the first calls are throttled
            out = json.dumps({"error": "rate limited"} if busy else {"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(429 if busy else 200)
            if busy:
                self.send_header("Retry-After", "0.2")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    keep = {k: os.environ.get(k) for k in ("AEXP_ACCESS_TOKEN", "GMSJOURNEY_LLM_RETRY_SECONDS")}
    os.environ["AEXP_ACCESS_TOKEN"] = "tok"
    L._PACERS.clear()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1/{{model}}/chat"
        d = tempfile.mkdtemp()
        llm = L.CachedLLM(L.LLMSettings(url=url, backend="gateway", cache_dir=d, send_sampling_params=False), tag="sut")
        with ThreadPoolExecutor(4) as ex:                                       # four conversations at once
            out = list(ex.map(lambda i: llm.invoke("sys", f"turn {i}", "x"), range(8)))
        assert out == ["ok"] * 8
        p = next(iter(L._PACERS.values()))                                     # the deployment's pacer
        assert p.throttled >= 3 and not llm._tripped
        state["always"] = True                                                  # now it never lets up
        os.environ["GMSJOURNEY_LLM_RETRY_SECONDS"] = "1"
        for i in range(10):                                                     # more than the breaker's 8
            try:
                llm.invoke("sys", f"more {i}", "x")
                raise AssertionError("a call succeeded against a gateway that only throttles")
            except L.LLMThrottled as e:
                assert "busy" in str(e) or "kept answering 429" in str(e)
        assert not llm._tripped                                                 # busy is not down
    finally:
        L._PACERS.clear()
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        L._session().close()
        srv.shutdown()


def test_the_pacer_learns_the_deployments_token_quota_and_spaces_big_calls():
    import time as _t
    from gmsjourney import llm as L
    p = L._Pacer(4)
    p.observe({"x-ratelimit-limit-tokens": "600", "x-ratelimit-remaining-tokens": "12",
               "x-ratelimit-limit-requests": "60"})
    assert p.tpm == 600 and p.quota()["source"] == "x-ratelimit-limit-tokens" and p.quota()["requests_per_minute"] == 60
    t0 = _t.time()
    p.acquire(10)                                                    # 12 left: admitted at once
    p.release()
    p.acquire(10)                                                    # 2 left, 10/s refill: waits about 0.8 s
    p.release()
    assert 0.5 < _t.time() - t0 < 3.0
    big = L.estimate_tokens([{"content": "x" * 20000}, {"content": "y" * 1300}], 500)
    small = L.estimate_tokens([{"content": "x" * 1500}, {"content": "y" * 300}], 220)
    assert big > 6000 and small < 800                                # a bot step costs ten rewording calls


def test_json_mode_is_asked_for_and_dropped_where_the_deployment_rejects_it():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from gmsjourney import llm as L
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append("response_format" in body)
            bad = "response_format" in body
            out = json.dumps({"error": {"message": "Unrecognized request argument: response_format"}} if bad else
                             {"choices": [{"message": {"content": '{"say": []}'}}]}).encode()
            self.send_response(400 if bad else 200)
            self.send_header("x-ratelimit-limit-tokens", "30000")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    keep = os.environ.get("AEXP_ACCESS_TOKEN")
    os.environ["AEXP_ACCESS_TOKEN"] = "tok"
    L._ADAPTED.clear()
    L._PACERS.clear()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1/{{model}}/chat"
        c = L.make_chat_model(L.LLMSettings(url=url, backend="gateway", json_mode=True, max_tokens=500))
        assert c.invoke(L._messages("Reply in JSON.", "hi")).content == '{"say": []}'
        assert seen == [True, False] and "JSON mode (response_format) dropped" in c.adaptations
        assert L.gateway_quota()[c._url()]["tokens_per_minute"] == 30000
        from gmsjourney.s10_console.bots import settings_for
        b = settings_for({"kind": "llm", "model": "gpt-41"})
        assert b.json_mode and b.max_tokens == 500
    finally:
        L._ADAPTED.clear()
        L._PACERS.clear()
        if keep is None:
            os.environ.pop("AEXP_ACCESS_TOKEN", None)
        else:
            os.environ["AEXP_ACCESS_TOKEN"] = keep
        L._session().close()
        srv.shutdown()


@_with_fake
def test_the_critics_model_is_chosen_apart_from_the_bots():
    """The critique runs on the bot's model, a gateway deployment or a local vLLM model, whatever the bot runs on."""
    from gmsjourney.s10_console import bots as B
    from gmsjourney.s06_bots.llm_bot import ENV
    assert B.critic_where({"provider": "gateway", "critic_model": "gpt-55"}) == "gateway"      # an older job
    assert B.critic_where({"provider": "local", "critic_model": "gpt-55"}) == "same"
    assert B.critic_where({"critic_provider": "local"}) == "local"
    base = {"kind": "llm", "model": "gpt-41", "critique": True}
    fac, name, _ = B.make_factory({**base, "critic_provider": "same"})
    assert fac.critic_llm is fac.llm and "critique by the same model" in name
    fac, name, _ = B.make_factory({**base, "critic_provider": "gateway", "critic_model": "gpt-41"})
    assert fac.critic_llm is fac.llm                                       # the bot's own deployment: one model
    fac, name, _ = B.make_factory({**base, "critic_provider": "gateway", "critic_model": "gpt-55"})
    assert fac.critic_llm is not fac.llm and fac.critic_llm.settings.model == "gpt-55" and "critique by gpt-55" in name
    old = os.environ.pop(ENV, None)
    try:
        folder = Path(tempfile.mkdtemp()) / "gpt-oss-20b"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        loc = {**base, "critic_provider": "local", "critic_local_path": str(folder), "critic_vllm_url": "http://127.0.0.1:9/v1"}
        fac, name, _ = B.make_factory(loc)                                 # a gateway bot reviewed by a local model
        assert fac.critic_llm is not fac.llm and "(local, vLLM)" in name and fac.critic_llm.tag == "critic"
        try:
            B.make_factory({**base, "critic_provider": "local", "critic_local_path": str(folder) + "-missing"})
            raise AssertionError("an unusable local critic must be refused")
        except RuntimeError as e:
            assert "critic's local model" in str(e)
        try:                                                               # the Playground needs a server for it
            B.start("project", "default", {**loc, "critic_vllm_url": ""})
            raise AssertionError("the playground must refuse a local critic without a vLLM server")
        except ValueError as e:
            assert "critic's local model needs a vLLM server" in str(e)
        fac, name, _ = B.make_factory({**base, "critique": False, "critic_provider": "local"})
        assert fac.critic_llm is None and "no critique" in name
    finally:
        if old is not None:
            os.environ[ENV] = old


@_with_fake
def test_the_critique_reviews_only_the_steps_in_its_scope():
    """Every step, the account changes and replies (the default), or the replies only: fewer calls with each."""
    from gmsjourney.s10_console import bots as PG
    from gmsjourney.s06_bots.tools import write_tools
    domain.set_current("charge_review_graph")
    case = PG.cases("project", None)["cases"][0]
    seen = {}
    for scope in ("all", "key", "replies"):
        v = PG.start("project", "default", {"kind": "llm", "model": "gpt-41", "critique": True, "critique_scope": scope})
        s = PG.get(v["id"])
        for ln in case["lines"]:
            if s.view()["ended"]:
                break
            s.say(ln["text"])
        st = s.llm.stats.as_dict()
        seen[scope] = (st["critiques"], st["critique_skipped"])
        assert ("of every step" if scope == "all" else "of account changes and replies" if scope == "key" else "of replies only") in s.bot_name
    assert seen["all"][1] == 0 and seen["all"][0] > seen["key"][0] >= seen["replies"][0] > 0, seen
    assert sum(seen["all"]) == sum(seen["key"]) == sum(seen["replies"]), seen   # every step is reviewed or skipped
    assert write_tools() and PG.critique_scope({}) == "key" and PG.critique_scope({"critique_scope": "all"}) == "all"
