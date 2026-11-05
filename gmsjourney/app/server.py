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

The graph and project calls share one process-wide active journey, so they are serialised; campaign runs happen
in their own processes (runs.py).
"""
from __future__ import annotations

import json
import mimetypes
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
                        "graph": dom.graph_path.name, "compiled": g is not None,
                        "loops": len(g.loops()) if g is not None else 0,
                        "has_project": dom.project_path.exists()})
        except Exception as e:                     # noqa: BLE001
            out.append({"name": d.name, "title": d.name, "error": f"{type(e).__name__}: {e}"})
    set_current(keep)
    return out


def _studio(pack: str, fresh: bool = False):
    from ..domain import set_current
    from ..studio import Studio
    with _LOCK:
        set_current(pack)
        if fresh or pack not in _STUDIOS:
            _STUDIOS[pack] = Studio(pack=pack)
        return _STUDIOS[pack]


def _draft(pack: str) -> dict[str, Any]:
    from ..domain import set_current
    from ..project import dump_yaml
    from ..project_draft import HEADER, draft
    with _LOCK:
        dom = set_current(pack)
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
                    return self._json({"packs": packs(), "devices": devices(), "runs_dir": str(runs.root),
                                       "active_run": runs.active()})
                if parts[:2] == ["api", "journey"] and len(parts) == 3:
                    st = _studio(parts[2], fresh=parse_qs(u.query).get("fresh") == ["1"])
                    with _LOCK:
                        from ..domain import set_current
                        set_current(parts[2])
                        return self._json({"pack": parts[2], **st.state()})
                if parts == ["api", "runs"]:
                    return self._json({"runs": runs.list(), "active": runs.active()})
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
                    with _LOCK:
                        from ..domain import set_current
                        set_current(pack)
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

    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), H)


_SERVERS: dict[int, ThreadingHTTPServer] = {}


def start(port: int = 8790, host: str = "127.0.0.1", runs_dir: str | Path | None = None) -> str:
    """Start the console in a background thread (from Jupyter or Python); returns its URL."""
    if port in _SERVERS:
        return f"http://{host}:{port}/"
    srv = make_server(port, host, runs_dir)
    threading.Thread(target=srv.serve_forever, daemon=True, name=f"journey-console-{port}").start()
    _SERVERS[port] = srv
    return f"http://{host}:{port}/"


def stop(port: int = 8790) -> None:
    srv = _SERVERS.pop(port, None)
    if srv is not None:
        srv.shutdown()
        srv.server_close()
