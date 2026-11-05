"""The journey console: a local web app to configure a journey and run its test campaign.

From Jupyter (or any Python session)::

    from gmsjourney import app
    url = app.start(port=8790)            # a background thread; the campaign itself runs in its own process
    from IPython.display import IFrame
    IFrame(url, width="100%", height=900)

or simply ``app.show()``. From a shell: ``python -m gmsjourney.app --port 8790``. On a remote Jupyter, ``app.show()``
goes through jupyter-server-proxy when it finds one (JupyterHub), or forward the port (ssh -L 8790:localhost:8790).
"""
from __future__ import annotations

import os

from .server import make_server, start, stop


def show(port: int = 8790, height: int = 960, host: str = "127.0.0.1", runs_dir=None, proxy: bool | None = None):
    """Start the console and display it in the notebook.

    On JupyterHub (or with ``proxy=True``) the page is shown through jupyter-server-proxy at ``<prefix>proxy/<port>/``,
    so it works when the kernel runs on a remote GPU machine; otherwise straight from ``http://host:port/``.
    """
    from IPython.display import IFrame
    url = start(port, host, runs_dir)
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
    if proxy or (proxy is None and prefix):
        url = f"{(prefix or '/').rstrip('/')}/proxy/{port}/"
    print(f"journey console: {url}")
    return IFrame(url, width="100%", height=height)


__all__ = ["start", "stop", "show", "make_server"]
