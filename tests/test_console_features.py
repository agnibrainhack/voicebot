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
        assert fac.critique and model.critiques >= 3          # the critique layer reviewed every step on the same server
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
