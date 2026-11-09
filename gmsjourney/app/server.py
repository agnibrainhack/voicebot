"""The journey console's HTTP server: one page (ui.html) and a small JSON API.

    GET  /                                  the console
    GET  /api/meta                          the journeys (domain packs), the compute devices, the runs folder
    GET  /api/journey/<pack>                what the graph says and what the project configures (studio.Studio.state)
    POST /api/journey/<pack>/<action>       check | preview | save | yaml | draft   (body: {project, levels, questions})
    GET  /api/runs                          every run, newest first
    POST /api/runs                          start a run   (body: {pack, device, max_cases, ...})
    GET  /api/runs/<id>?since=N             a run's status and its events from N on
    POST /api/runs/<id>/cancel              stop a run
    GET  /runs/<id>/<file>                  a run's outputs (report.html, dataset/..., ...)
    GET  /api/runs/<id>/cases[/<case>]      a finished run's generated cases (cases.py): the index, or one case
    GET  /api/runs/<id>/verdict             the one-page verdict (verdict.py): gates, risks, fingerprint, case by case
    GET  /api/journey/<pack>/sut            the suggested prompt for an LLM bot, and the fixed part the harness adds
    POST /api/attach                        text from a file for the LLM bot's prompt (body: {name, data: base64})
    POST /api/llm/check                     one tiny call through the model gateway (body: {model})
    GET  /api/play/<pack>/cases?source=S    the playground's cases: "latest" run, a run id, or "project"
    POST /api/play/<pack>/start             a conversation (body: {source, case, sut})
    GET  /api/play/s/<sid>                  a conversation so far
    POST /api/play/s/<sid>/say              the customer's next turn (body: {text})
    POST /api/graph/open                    open a graph by path: its journey, or a new journey made from it (body: {path})
    GET  /api/graph/<pack>/sections         the graph section by section, and its core segments (graphs.py)
    GET  /api/graph/<pack>/edit             the graph's own entries, for the editor
    POST /api/graph/<pack>/check            lint an edited graph without saving it (body: {items} or {triples})
    POST /api/graph/<pack>/save             save it as <name>.vN.json beside the original and switch to it
    POST /api/graph/<pack>/use              switch the journey to one of its graph versions (body: {file})
    GET  /api/graph/<pack>/sphere           the unit-sphere view of the graph (sphere.py)

The graph and project calls share one process-wide active journey, so they are serialised; campaign runs happen
in their own processes (runs.py).
"""
from __future__ import annotations

import contextlib
import json
import mimetypes
import socket
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .runs import ROOT, Runs, devices

UI = Path(__file__).with_name("ui.html")
_LOCK = threading.RLock()
_STUDIOS: dict[str, Any] = {}


def packs() -> list[dict[str, Any]]:
    """Every journey under gmsjourney/policy/: its name, title, graph and whether it has a project yet."""
    from ..domain import current, load, set_current
    keep = current().name
    out = []
    for d in sorted((ROOT / "gmsjourney" / "policy").iterdir()):
        if not d.is_dir() or not (d / "domain.yaml").exists():
            continue
        try:
            dom = load(d.name)
            g = dom.graph
            out.append({"name": d.name, "title": (g.title if g is not None else None) or d.name.replace("_", " ").title(),
                        "graph": dom.graph_path.name, "path": str(dom.graph_path.relative_to(ROOT)),
                        "compiled": g is not None,
                        "loops": len(g.loops()) if g is not None else 0,
                        "has_project": dom.project_path.exists()})
        except Exception as e:                     # noqa: BLE001
            out.append({"name": d.name, "title": d.name, "error": f"{type(e).__name__}: {e}"})
    set_current(keep)
    return out


@contextlib.contextmanager
def _on(pack: str):
    """Work on one journey, then put back the journey that was active: the console shares the process (a Jupyter
    kernel) with whatever else runs there, and must not leave its own choice behind."""
    from ..domain import current, set_current
    with _LOCK:
        keep = current().name
        try:
            yield set_current(pack)
        finally:
            if current().name != keep:
                set_current(keep)


def _studio(pack: str, fresh: bool = False):
    from ..studio import Studio
    with _on(pack):
        if fresh or pack not in _STUDIOS:
            _STUDIOS[pack] = Studio(pack=pack)
        return _STUDIOS[pack]


def _draft(pack: str) -> dict[str, Any]:
    from ..domain import set_current
    from ..project import dump_yaml
    from ..project_draft import HEADER, draft
    with _on(pack) as dom:
        if dom.project_path.exists():
            return {"drafted": False, "path": str(dom.project_path)}
        raw = draft(dom)
        dom.project_path.write_text(HEADER.format(name=raw["project"]["name"], path=dom.project_path, pack=pack)
                                    + "\n" + dump_yaml(raw))
        _STUDIOS.pop(pack, None)
        return {"drafted": True, "path": str(dom.project_path)}


def make_server(port: int = 8790, host: str = "127.0.0.1", runs_dir: str | Path | None = None) -> ThreadingHTTPServer:
    runs = Runs(runs_dir or ROOT / "output" / "console_runs")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _err(self, e: Exception, code: int = 400) -> None:
            self._json({"error": f"{e}" if isinstance(e, (ValueError, RuntimeError, KeyError)) else f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()}, code)

        def do_GET(self):                           # noqa: N802
            u = urlparse(self.path)
            parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
            try:
                if not parts:
                    return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
                if parts == ["api", "meta"]:
                    from ..sut.local_llm import status as local_status
                    return self._json({"packs": packs(), "devices": devices(), "runs_dir": str(runs.root),
                                       "active_run": runs.active(), "sut_local": local_status()})
                if parts[:2] == ["api", "journey"] and len(parts) == 3:
                    st = _studio(parts[2], fresh=parse_qs(u.query).get("fresh") == ["1"])
                    with _on(parts[2]):
                        return self._json({"pack": parts[2], **st.state()})
                if parts == ["api", "runs"]:
                    return self._json({"runs": runs.list(), "active": runs.active()})
                if parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "verdict":
                    from ..verdict import build
                    return self._json(build(runs._dir(parts[2])))
                if parts[:2] == ["api", "runs"] and len(parts) in (4, 5) and parts[3] == "cases":
                    from . import cases as C
                    out = runs._dir(parts[2]) / "out"
                    return self._json(C.index(out) if len(parts) == 4 else C.detail(out, parts[4]))
                if parts[:2] == ["api", "journey"] and len(parts) == 4 and parts[3] == "sut":
                    from .sut import suggested
                    with _on(parts[2]):
                        return self._json(suggested())
                if parts[:2] == ["api", "play"] and len(parts) == 4 and parts[3] == "cases":
                    from . import playground as PG
                    with _on(parts[2]):
                        return self._json(PG.cases((parse_qs(u.query).get("source") or ["latest"])[0], runs.root))
                if parts[:3] == ["api", "play", "s"] and len(parts) == 4:
                    from . import playground as PG
                    sess = PG.get(parts[3])
                    with _on(sess.pack):
                        return self._json(sess.view())
                if parts[:2] == ["api", "graph"] and len(parts) == 4:
                    from . import graphs as GR
                    with _on(parts[2]) as dom:
                        if parts[3] == "sections":
                            return self._json({**GR.sections(dom), "segments": GR.core_segments(dom),
                                               "lexicon_drafted": getattr(dom, "lexicon_drafted", None) or []})
                        if parts[3] == "edit":
                            return self._json(GR.editable(dom))
                        if parts[3] == "sphere":
                            from . import sphere
                            return self._json(sphere.build(dom))
                if parts[:2] == ["api", "runs"] and len(parts) == 3:
                    since = int((parse_qs(u.query).get("since") or ["0"])[0])
                    return self._json(runs.events(parts[2], since))
                if parts[0] == "runs" and len(parts) >= 3:
                    p = runs.file(parts[1], "/".join(parts[2:]))
                    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    if ctype.startswith("text/"):
                        ctype += "; charset=utf-8"
                    return self._send(200, p.read_bytes(), ctype)
            except KeyError as e:
                return self._json({"error": f"not found: {e}"}, 404)
            except Exception as e:                  # noqa: BLE001
                return self._err(e, 500)
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):                          # noqa: N802
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if parts[:2] == ["api", "journey"] and len(parts) == 4:
                    pack, action = parts[2], parts[3]
                    if action == "draft":
                        return self._json(_draft(pack))
                    st = _studio(pack)
                    with _on(pack):
                        raw = body.get("project") or st.project.raw
                        if action == "check":
                            return self._json(st.check(raw))
                        if action == "preview":
                            return self._json(st.preview(raw, body.get("levels") or {}, body.get("questions") or []))
                        if action == "save":
                            return self._json(st.save(raw))
                        if action == "yaml":
                            from ..studio import to_yaml
                            return self._json({"yaml": to_yaml(raw)})
                if parts == ["api", "graph", "open"]:
                    from . import graphs as GR
                    with _LOCK:
                        out = GR.open_graph(str(body.get("path") or ""))
                        _STUDIOS.pop(out["pack"], None)
                        return self._json(out)
                if parts[:2] == ["api", "graph"] and len(parts) == 4 and parts[3] in ("check", "save", "use"):
                    from . import graphs as GR
                    pack = parts[2]
                    with _on(pack) as dom:
                        if parts[3] == "check":
                            orig, doc, fmt = GR.build_doc(dom, body)
                            return self._json({"changes": GR.diff(orig, doc, fmt), **GR.check_candidate(dom, doc)})
                        if parts[3] == "save":
                            out = GR.save_version(dom, body)
                        else:
                            f = str(body.get("file") or "")
                            if f not in {v["file"] for v in GR.versions(dom)}:
                                raise ValueError(f"{f!r} is not a version of this journey's graph")
                            out = {"switched": GR.set_graph(dom.dir, f), "file": f}
                    _STUDIOS.pop(pack, None)
                    from .playground import reset as _pg_reset
                    _pg_reset(pack)
                    return self._json(out)
                if parts == ["api", "attach"]:
                    import base64
                    from .attachments import extract
                    return self._json(extract(str(body.get("name") or ""), base64.b64decode(body.get("data") or "")))
                if parts == ["api", "llm", "check"]:
                    from .sut import check_gateway
                    if body.get("provider") == "local":
                        from ..sut.local_llm import status as local_status
                        st = local_status()
                        return self._json({"ok": st["usable"], "detail": st["detail"], "local": True})
                    return self._json(check_gateway(str(body.get("model") or "")))
                if parts[:2] == ["api", "play"] and len(parts) == 4 and parts[3] == "start":
                    from . import playground as PG
                    with _on(parts[2]):
                        return self._json(PG.start(str(body.get("source") or "project"), str(body.get("case") or ""),
                                                   body.get("sut")))
                if parts[:3] == ["api", "play", "s"] and len(parts) == 5 and parts[4] == "say":
                    from . import playground as PG
                    sess = PG.get(parts[3])
                    with _on(sess.pack):
                        turn = sess.say(str(body.get("text") or ""))
                        return self._json({"turn": turn, **sess.view()})
                if parts == ["api", "runs"]:
                    if not body.get("pack"):
                        raise ValueError("choose a journey to run")
                    return self._json({"id": runs.start(body)})
                if parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "cancel":
                    return self._json({"cancelled": runs.cancel(parts[2])})
            except KeyError as e:
                return self._json({"error": f"not found: {e}"}, 404)
            except Exception as e:                  # noqa: BLE001
                return self._err(e)
            return self._send(404, b"not found", "text/plain")

    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"journey console: listening on {host}: anyone who can reach this port can edit projects and start "
              "campaigns (there is no login); prefer 127.0.0.1 with port forwarding or a Jupyter proxy")
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), H)


def free_port(port: int, host: str = "127.0.0.1", tries: int = 20) -> int:
    """``port`` if nothing listens there, else the next free one (another kernel may hold the console's port)."""
    for p in range(port, port + tries):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise OSError(f"no free port in {port}-{port + tries - 1}")


_SERVERS: dict[int, ThreadingHTTPServer] = {}


def start(port: int = 8790, host: str = "127.0.0.1", runs_dir: str | Path | None = None) -> str:
    """Start the console in a background thread (from Jupyter or Python); returns its URL. Running it again returns
    the same server; if another process holds ``port``, the next free port is used (and said)."""
    if _SERVERS:
        p = port if port in _SERVERS else next(iter(_SERVERS))
        return f"http://{host}:{p}/"
    p = free_port(port, host)
    if p != port:
        print(f"journey console: port {port} is in use (another kernel?); using {p}")
    srv = make_server(p, host, runs_dir)
    threading.Thread(target=srv.serve_forever, daemon=True, name=f"journey-console-{p}").start()
    _SERVERS[p] = srv
    return f"http://{host}:{p}/"


def running() -> dict[int, str]:
    """The consoles this process serves: port -> URL."""
    return {p: f"http://{s.server_address[0]}:{p}/" for p, s in _SERVERS.items()}


def stop(port: int | None = None) -> None:
    """Stop the console (the one on ``port``, or every one this process started). Runs in progress keep going."""
    if port is None:
        for p in list(_SERVERS):
            stop(p)
        return
    srv = _SERVERS.pop(port, None)
    if srv is not None:
        srv.shutdown()
        srv.server_close()
