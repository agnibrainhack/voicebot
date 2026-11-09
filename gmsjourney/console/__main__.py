"""python -m gmsjourney.console [--port 8790] [--host 127.0.0.1] [--runs DIR]
"""
from __future__ import annotations

import argparse


from .server import make_server


def main(argv=None) -> None:
    """Command line: serve the console until interrupted."""
    ap = argparse.ArgumentParser(prog="python -m gmsjourney.console", description="the journey console")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", default=None, help="folder for campaign runs (default: output/console_runs)")
    a = ap.parse_args(argv)
    srv = make_server(a.port, a.host, a.runs)
    print(f"journey console: http://{a.host}:{a.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
