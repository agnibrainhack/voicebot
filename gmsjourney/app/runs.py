"""The console's campaign runs: start one in its own process, read its events, list past runs, cancel.

Each run is a folder ``<root>/<id>/`` holding ``job.json``, ``events.jsonl`` (worker.py), ``console.log`` (everything
the process printed), ``headline.json`` when it finished, and the campaign outputs under ``out/``.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def devices() -> dict[str, Any]:
    """The compute devices this machine offers: cpu, and each CUDA GPU torch can see (with its name and memory)."""
    out: dict[str, Any] = {"options": [{"id": "auto", "label": "Auto (GPU when available)", "available": True},
                                       {"id": "cpu", "label": "CPU", "available": True}], "cuda": 0, "torch": None}
    try:
        import torch
        out["torch"] = torch.__version__
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        out["cuda"] = n
        for i in range(max(n, 4)):                 # cuda:0-3 always listed; unavailable ones are marked
            if i < n:
                p = torch.cuda.get_device_properties(i)
                out["options"].append({"id": f"cuda:{i}", "label": f"cuda:{i} · {p.name} · {p.total_memory / 2**30:.0f} GB",
                                       "available": True})
            else:
                out["options"].append({"id": f"cuda:{i}", "label": f"cuda:{i}", "available": False})
    except Exception as e:                         # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["options"] += [{"id": f"cuda:{i}", "label": f"cuda:{i}", "available": False} for i in range(4)]
    return out


def _provenance(pack: str) -> dict[str, Any]:
    from ..verdict import provenance
    return provenance(pack)


def check_device(dev: str | None) -> str | None:
    """None when the device can be used, else why not."""
    if not dev or dev in ("auto", "cpu"):
        return None
    if not re.fullmatch(r"cuda(:\d+)?", dev):
        return f"unknown device {dev!r}: use auto, cpu or cuda:N"
    opts = {o["id"]: o for o in devices()["options"]}
    if dev == "cuda":
        return None if devices()["cuda"] else "no CUDA GPU is visible to torch on this machine"
    if not (opts.get(dev) or {}).get("available"):
        return f"{dev} is not available on this machine ({devices()['cuda']} CUDA GPU(s) visible)"
    return None


class Runs:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ start / cancel
    def active(self) -> str | None:
        """The run in progress: one this server started, or one still alive from before a restart (its worker's
        pid is in its start event)."""
        with self.lock:
            for rid, p in list(self.procs.items()):
                if p.poll() is None:
                    return rid
        for run in sorted((d for d in self.root.iterdir() if d.is_dir()), reverse=True)[:5]:
            if self.status(run.name) == "running":
                return run.name
        return None

    def start(self, job: dict[str, Any]) -> str:
        busy = self.active()
        if busy:
            raise RuntimeError(f"run {busy} is still going: wait for it or cancel it first")
        why = check_device(job.get("device"))
        if why:
            raise ValueError(why)
        if (job.get("sut") or {}).get("provider") == "local" and (job.get("sut") or {}).get("kind") == "llm":
            from ..sut.local_llm import status as local_status
            st = local_status()
            if not st["usable"]:
                raise ValueError(f"the local model cannot be used: {st['detail']}")
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^a-z0-9_]+", "-", job["pack"].lower())
        run = self.root / rid
        run.mkdir(parents=True)
        sut = dict(job.get("sut") or {})
        if sut.get("kind") == "llm":                # the prompt and attachments live beside the job, not in it
            (run / "sut.json").write_text(json.dumps(sut, indent=1, ensure_ascii=False), encoding="utf-8")
            if sut.get("provider") == "local":
                from ..sut.local_llm import status as local_status
                sut["model"] = f"local-{local_status()['name']}"
                (run / "sut.json").write_text(json.dumps(sut, indent=1, ensure_ascii=False), encoding="utf-8")
            job = {**job, "sut": {"kind": "llm", "provider": sut.get("provider") or "gateway", "model": sut.get("model") or "",
                                  "mode": sut.get("mode") or "live",
                                  "custom_prompt": bool(sut.get("custom_prompt")),
                                  "references": [{"name": r.get("name"), "chars": len(r.get("text") or "")}
                                                 for r in sut.get("references") or []]}}
        job = {**job, **_provenance(job["pack"])}
        (run / "job.json").write_text(json.dumps({**job, "id": rid, "created": time.time()}, indent=1))
        env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        if job.get("device") and job["device"] != "auto":
            env["GMSJOURNEY_DEVICE"] = job["device"]
        log = open(run / "console.log", "w")
        # its own process group, so cancelling stops everything the campaign started
        p = subprocess.Popen([sys.executable, "-m", "gmsjourney.app.worker", str(run)], cwd=str(ROOT), env=env,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with self.lock:
            self.procs[rid] = p
        return rid

    def cancel(self, rid: str) -> bool:
        """Stop a run and every process it started (its process group); works for a run started before a restart."""
        with self.lock:
            p = self.procs.get(rid)
        pid = p.pid if p is not None and p.poll() is None else (self._pid(rid) if self.status(rid) == "running" else None)
        if pid is None:
            return False
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                break
            for _ in range(50):
                if not _alive(pid) or (p is not None and p.poll() is not None):
                    break
                time.sleep(0.2)
            if not _alive(pid) or (p is not None and p.poll() is not None):
                break
        with open(self.root / rid / "events.jsonl", "a") as f:
            f.write(json.dumps({"type": "cancelled", "t": time.time()}) + "\n")
        return True

    # ------------------------------------------------------------------ reading
    def _dir(self, rid: str) -> Path:
        if not re.fullmatch(r"[0-9A-Za-z_\-]+", rid) or not (self.root / rid).is_dir():
            raise KeyError(rid)
        return self.root / rid

    def events(self, rid: str, since: int = 0) -> dict[str, Any]:
        run = self._dir(rid)
        lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if (run / "events.jsonl").exists() else []
        evs = []
        for ln in lines[since:]:
            try:
                evs.append(json.loads(ln))
            except json.JSONDecodeError:              # a line still being written
                break
        return {"id": rid, "status": self.status(rid, lines), "events": evs, "next": since + len(evs)}

    def status(self, rid: str, lines: list[str] | None = None) -> str:
        run = self._dir(rid)
        if lines is None:
            lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if (run / "events.jsonl").exists() else []
        last = json.loads(lines[-1]).get("type") if lines else None
        for ln in reversed(lines):
            t = json.loads(ln).get("type")
            if t in ("done", "error", "cancelled"):
                return {"done": "finished", "error": "failed", "cancelled": "cancelled"}[t]
        with self.lock:
            p = self.procs.get(rid)
        if p is not None and p.poll() is None:
            return "running"
        pid = self._pid(rid, lines)
        if p is None and pid and _alive(pid):
            return "running"                       # started before this server (a kernel restart): still going
        return "stopped" if last else "starting"

    def _pid(self, rid: str, lines: list[str] | None = None) -> int | None:
        """The worker's pid, from the run's start event."""
        if lines is None:
            f = self.root / rid / "events.jsonl"
            lines = f.read_text(encoding="utf-8").splitlines()[:1] if f.exists() else []
        for ln in lines[:1]:
            try:
                return int(json.loads(ln).get("pid") or 0) or None
            except (ValueError, TypeError):
                return None
        return None

    def list(self) -> list[dict[str, Any]]:
        out = []
        for run in sorted((d for d in self.root.iterdir() if d.is_dir() and (d / "job.json").exists()), reverse=True):
            job = json.loads((run / "job.json").read_text())
            head = json.loads((run / "headline.json").read_text()) if (run / "headline.json").exists() else None
            if head is not None and "gate_reworded" not in head and (run / "out" / "report.json").exists():
                try:                                       # runs from before the reworded gate was shown: once
                    rep = json.loads((run / "out" / "report.json").read_text())
                    head["gate_reworded"] = ((rep.get("release_gate") or {}).get("enriched_template") or {}).get("passed")
                    (run / "headline.json").write_text(json.dumps(head, default=str))
                except (OSError, ValueError):
                    pass
            files = [f for f in ("report.html", "bot_validation.html", "case_comparison.html") if (run / "out" / f).exists()]
            out.append({"id": run.name, "job": job, "status": self.status(run.name), "headline": head, "files": files,
                        "elapsed": _elapsed(run / "events.jsonl")})
        return out

    def file(self, rid: str, rel: str) -> Path:
        """A file of the run's outputs (no escaping the run folder)."""
        base = (self._dir(rid) / "out").resolve()
        p = (base / rel).resolve()
        if base not in p.parents and p != base or not p.is_file():
            raise KeyError(rel)
        return p


def _alive(pid: int) -> bool:
    """True while a process with this pid exists (and is not a zombie)."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _elapsed(events) -> float | None:
    """Seconds from a run's first event to its last (for the run list and time estimates)."""
    try:
        lines = events.read_text(encoding="utf-8").splitlines()
        return round(json.loads(lines[-1])["t"] - json.loads(lines[0])["t"], 1) if len(lines) > 1 else None
    except (OSError, ValueError, KeyError, IndexError):
        return None
