"""The journey console's API: the journeys and devices it offers, a journey's graph and project, a live check and a
preview, the runs folder, and what a step summary reads. No campaign is run.

    python tests/test_app.py
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmsjourney import domain  # noqa: E402
from gmsjourney.app import make_server  # noqa: E402
from gmsjourney.app.runs import check_device, devices  # noqa: E402
from gmsjourney.app.summaries import STEPS, summarize  # noqa: E402


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve():
    port, runs = _port(), tempfile.mkdtemp()
    srv = make_server(port, "127.0.0.1", runs)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, (json.loads(r.read()) if "json" in r.headers["Content-Type"] else r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
    return srv, call


def test_devices_list_cpu_and_four_cuda_slots():
    d = devices()
    ids = [o["id"] for o in d["options"]]
    assert ids[:2] == ["auto", "cpu"] and all(f"cuda:{i}" in ids for i in range(4))
    assert check_device("cpu") is None and check_device("gpu7")
    if not d["cuda"]:
        assert check_device("cuda:0")                       # refused with a reason when torch sees no GPU


def test_console_api_reads_checks_and_previews_a_journey():
    srv, call = _serve()
    try:
        code, page = call("")
        assert code == 200 and b"Journey Console" in page and b'"api/meta"' in page    # relative: works under a proxy
        code, meta = call("api/meta")
        names = {p["name"]: p for p in meta["packs"]}
        assert {"charge_review", "supplementary_card", "charge_review_graph"} <= set(names)
        assert names["charge_review_graph"]["loops"] == 1 and names["charge_review_graph"]["compiled"]
        code, J = call("api/journey/supplementary_card")
        assert code == 200 and J["pack"] == "supplementary_card"
        assert J["domain"]["states"] and J["domain"]["tools"] and J["project"]["factors"]
        assert J["fixture_keys"] and not J["issues"]
        P = J["project"]
        f = P["factors"][0]
        lv = next(k for k in f["levels"] if str(k) != str(f["default"]))
        code, pv = call("api/journey/supplementary_card/preview", {"project": P, "levels": {f["name"]: lv}})
        assert code == 200 and pv["conversation"] and pv["signature"]["end"]
        bad = json.loads(json.dumps(P))
        bad["factors"][0]["anchor"] = {"tool": "no_such_tool"}
        code, chk = call("api/journey/supplementary_card/check", {"project": bad})
        assert code == 200 and any("no_such_tool" in i for i in chk["issues"])
        code, y = call("api/journey/supplementary_card/yaml", {"project": P})
        assert "factors:" in y["yaml"]
        code, runs = call("api/runs")
        assert runs == {"runs": [], "active": None}
        code, err = call("api/runs", {"pack": "supplementary_card", "device": "cuda:9"})
        assert code == 400 and "cuda:9" in err["error"]
        code, _ = call("runs/../../etc/passwd")
        assert code == 404
    finally:
        srv.shutdown()
        domain.set_current("charge_review")


def test_every_step_has_a_summary_and_skips_cleanly():
    assert [s[0] for s in STEPS] == [str(i) for i in range(1, 13)]
    for sid, *_ in STEPS:
        out = summarize(sid, {})                             # a step that did not run reads as nothing, not a crash
        assert set(out) >= {"metrics", "tables", "notes"}


def test_a_busy_port_falls_back_and_the_url_can_be_set():
    import os
    from gmsjourney.app import url_for
    from gmsjourney.app.server import free_port
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        assert free_port(taken) != taken
    os.environ["GMSJOURNEY_CONSOLE_URL"] = "https://hub.example/user/me/proxy/{port}/"
    try:
        assert url_for(8790) == "https://hub.example/user/me/proxy/8790/"
    finally:
        del os.environ["GMSJOURNEY_CONSOLE_URL"]
    assert url_for(8790, proxy=False) == "http://127.0.0.1:8790/"


def test_a_run_outlives_the_server_that_started_it():
    """A worker still running after a kernel restart is reported running (its pid is in its start event), and a run
    whose worker is gone without a last event is 'stopped'."""
    import json
    import os
    import subprocess
    import time
    from gmsjourney.app.runs import Runs
    root = Path(tempfile.mkdtemp())
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        for rid, pid in (("r-live", live.pid), ("r-gone", 2 ** 22 + 7)):
            (root / rid).mkdir()
            (root / rid / "job.json").write_text(json.dumps({"pack": "supplementary_card", "id": rid}))
            (root / rid / "events.jsonl").write_text(json.dumps({"type": "start", "pid": pid, "t": time.time()}) + "\n")
        runs = Runs(root)                                   # a fresh server: it started neither
        assert runs.status("r-live") == "running" and runs.active() == "r-live"
        assert runs.status("r-gone") == "stopped"
        assert runs.cancel("r-live") and live.wait(10) is not None
        assert runs.status("r-live") == "cancelled"
    finally:
        if live.poll() is None:
            os.killpg(live.pid, 9)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
