"""The journey console: a local web app to configure a journey and run its test campaign.

From Jupyter (or any Python session)::

    from gmsjourney import s10_console as console
    console.show()                            # start it (background thread) and show it in the notebook
    console.stop()                            # stop it; runs in progress keep going

or ``url = console.start(port=8790)`` for the URL alone. From a shell: ``python -m gmsjourney.s10_console --port 8790``.

Where the page is reachable from depends on where the kernel runs. ``show`` works it out:

1. JupyterHub: ``<prefix>proxy/<port>/`` (jupyter-server-proxy);
2. a Jupyter server whose ``/proxy/<port>/`` reaches the console (jupyter-server-proxy installed there): that path,
   relative to the Jupyter page, so it works wherever the browser is;
3. otherwise ``http://127.0.0.1:<port>/``: right when the browser runs on the same machine as the kernel; from
   anywhere else forward the port (``ssh -L 8790:localhost:8790 <host>``).

``GMSJOURNEY_CONSOLE_URL`` (with ``{port}``) overrides all three. The campaign itself always runs in its own process;
the console only starts it and follows its events.
"""
from __future__ import annotations

import glob
import json
import os
import urllib.request


from .server import make_server, running, start, stop


def _jupyter_proxy(port: int) -> str | None:
    """``<base_url>proxy/<port>/`` when the Jupyter server running this kernel proxies to the console, else None.
    Read from the server's runtime file (no jupyter_server needed in the kernel) and checked with one request."""
    try:
        from jupyter_core.paths import jupyter_runtime_dir
    except ImportError:
        return None
    infos = []
    for f in glob.glob(os.path.join(jupyter_runtime_dir(), "jpserver-*.json")):
        try:
            infos.append(json.load(open(f)))
        except (OSError, ValueError):
            continue
    mine = [i for i in infos if i.get("pid") == os.getppid()] or (infos if len(infos) == 1 else [])
    for info in mine:
        base = str(info.get("base_url") or "/")
        probe = f"{str(info.get('url') or '').rstrip('/')}/proxy/{port}/api/meta"
        if not probe.startswith("http"):
            continue
        req = urllib.request.Request(probe, headers={"Authorization": f"token {info.get('token', '')}"})
        try:
            with urllib.request.urlopen(req, timeout=4) as r:
                if r.status == 200 and b"packs" in r.read(4096):
                    return f"{base.rstrip('/')}/proxy/{port}/"
        except Exception:                          # noqa: BLE001 - no proxy there
            continue
    return None


def url_for(port: int, host: str = "127.0.0.1", proxy: bool | None = None) -> str:
    """The URL a browser should open (see the module docstring for the order)."""
    if os.environ.get("GMSJOURNEY_CONSOLE_URL"):
        return os.environ["GMSJOURNEY_CONSOLE_URL"].format(port=port)
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
    if prefix and proxy is not False:
        return f"{prefix.rstrip('/')}/proxy/{port}/"
    if proxy is not False:
        found = _jupyter_proxy(port)
        if found:
            return found
        if proxy:
            return f"/proxy/{port}/"
    return f"http://{host}:{port}/"


def show(port: int = 8790, height: int = 960, host: str = "127.0.0.1", runs_dir=None, proxy: bool | None = None):
    """Start the console (if it is not running yet) and display it in the notebook, with a link to open it in its
    own browser tab (it has more room there). Returns the embedded frame."""
    from IPython.display import HTML, IFrame, display
    direct = start(port, host, runs_dir)
    used = int(direct.rstrip("/").rsplit(":", 1)[1])
    url = url_for(used, host, proxy)
    via = "through Jupyter" if url.startswith("/") or "/proxy/" in url else "directly"
    tip = ("" if via == "through Jupyter" else
           " If the frame below stays blank, your browser is not on the kernel's machine: forward the port "
           f"(<code>ssh -L {used}:localhost:{used} &lt;host&gt;</code>) or install <code>jupyter-server-proxy</code> "
           "where Jupyter runs.")
    display(HTML(f'<div style="font:14px system-ui;margin:4px 0 8px;line-height:1.5">Journey Console is running '
                 f'({via}): <a href="{url}" target="_blank" rel="noopener"><b>open it in a new tab</b></a> for more '
                 f'room, or use it below. Stop it with <code>console.stop()</code>; a campaign in progress keeps '
                 f'running.{tip}</div>'))
    return IFrame(url, width="100%", height=height)


__all__ = ["start", "stop", "show", "running", "url_for", "make_server"]
