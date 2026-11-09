"""Progress of the long parts of a test run, for whoever is watching.

Every long loop reports here: the bot runs (cases played), the rewording proofs, the resilience and planted-defect
runs, the knowledge store's training phases, and every LLM call (``llm.CachedLLM``). On a terminal a report is a tqdm
bar, as before; in the console's worker (``s10_console/runs.py``) a *sink* is installed with ``set_sink`` and every
report becomes a ``progress`` event the run page draws as a bar.

Analogy: the departures board. Each loop is a flight that posts its own status (boarding 40 of 130); the board does
not fly anything, it only shows what the flights report, and a terminal without a board (no sink) still has the
announcements (tqdm).

    bar(total, desc, show=..., key=..., unit=...)   a counted loop; update(n), set_postfix_str(s), close()
    phase(key, i, n, label, ...)                     a sequence of named phases (the store's training)
    llm_call(tag, outcome, seconds, tokens)          one LLM call: "cached", "live" or "error"
    set_sink(fn)                                     fn(**event) receives every report (None turns it off)

Reports never raise: a broken sink must not stop a test run.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

_sink: Callable[..., None] | None = None
_lock = threading.Lock()


def set_sink(fn: Callable[..., None] | None) -> None:
    """Send every progress report to ``fn(**event)`` (the console's worker writes them as events)."""
    global _sink
    _sink = fn
    _LLM.reset()


def report(key: str, done: float, total: float | None = None, label: str = "", unit: str = "", **detail: Any) -> None:
    """One progress report: ``done`` of ``total`` ``unit`` for the bar ``key`` (shown as ``label``)."""
    fn = _sink
    if fn is None:
        return
    try:
        fn(key=key, done=done, total=total, label=label, unit=unit, **detail)
    except Exception:                                     # noqa: BLE001 - progress must never stop the work
        pass


def active() -> bool:
    """True when someone is listening (a sink is installed)."""
    return _sink is not None


# ---------------------------------------------------------------------------------------------------------------- bars
class Bar:
    """A counted loop: a tqdm bar on a terminal (when ``show`` and tqdm is installed) and progress reports."""

    def __init__(self, iterable=None, total: int | None = None, desc: str = "", show: bool = True,
                 key: str | None = None, unit: str = "", label: str | None = None):
        self.it = iterable
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self.total, self.n, self.post = total, 0, ""
        self.key = key or _slug(desc) or "work"
        self.label = label or desc
        self.unit = unit
        self.t0 = time.time()
        self.tq = None
        if show:
            try:
                from tqdm.auto import tqdm
                self.tq = tqdm(total=total, desc=desc, dynamic_ncols=True)
            except ImportError:
                pass
        self._send()

    def _send(self, final: bool = False) -> None:
        report(self.key, self.n, self.total, self.label, self.unit, note=self.post,
               elapsed=round(time.time() - self.t0, 1), final=final)

    def __iter__(self):
        for x in (self.it if self.it is not None else []):
            yield x
            self.update(1)
        self.close()

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.tq is not None:
            self.tq.update(n)
        self._send()

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        self.post = s
        if self.tq is not None:
            self.tq.set_postfix_str(s, refresh=refresh)
        self._send()

    def close(self) -> None:
        if self.tq is not None:
            self.tq.close()
            self.tq = None
        self._send(final=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def bar(total: int | None = None, desc: str = "", show: bool = True, key: str | None = None, unit: str = "",
        iterable=None, label: str | None = None) -> Bar:
    """A counted loop's bar (see ``Bar``)."""
    return Bar(iterable, total=total, desc=desc, show=show, key=key, unit=unit, label=label)


def _slug(s: str) -> str:
    import re
    return re.sub(r"_+", "_", "".join(c if c.isalnum() else "_" for c in (s or "").lower())).strip("_")[:40]


# ------------------------------------------------------------------------------------------------------------ phases
def phase(key: str, i: int, n: int, label: str, *, phases: list[str] | None = None, expect_s: float | None = None,
          **detail: Any) -> None:
    """Phase ``i`` (0-based) of ``n`` started: ``label``. ``expect_s`` is how long it took last time (for an estimate);
    ``phases`` names them all (sent once with the first). ``i == n`` means every phase is done."""
    report(key, i, n, label, "phases", phase=True, phases=phases, expect_s=expect_s, **detail)


class StepCounter:
    """Counts optimizer steps while training runs inside a library we cannot instrument (torch's global hook), so a
    long phase shows that it is moving. A no-op without torch or the hook."""

    def __init__(self, key: str):
        self.key, self.n, self.h, self.last = key, 0, None, 0.0

    def __enter__(self):
        self.n = 0                                        # counted per phase
        try:
            from torch.optim.optimizer import register_optimizer_step_post_hook
            self.h = register_optimizer_step_post_hook(self._tick)
        except Exception:                                 # noqa: BLE001 - old torch, no torch: nothing to count
            self.h = None
        return self

    def _tick(self, *_a) -> None:
        self.n += 1
        now = time.time()
        if now - self.last >= 0.5:
            self.last = now
            report(self.key + ".steps", self.n, None, "optimizer steps", "steps", sub=True)

    def __exit__(self, *exc):
        if self.h is not None:
            self.h.remove()
        if self.n:
            report(self.key + ".steps", self.n, None, "optimizer steps", "steps", sub=True, final=True)


# --------------------------------------------------------------------------------------------------------------- LLM
class _LLMCounter:
    """Every LLM call of the run, by cache tag: how many, how many from the cache, failures, time and tokens."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.tags: dict[str, dict[str, float]] = {}
        self.last = 0.0
        self.t0 = time.time()

    def add(self, tag: str, outcome: str, seconds: float = 0.0, tokens: int = 0) -> None:
        if _sink is None:
            return
        with _lock:
            d = self.tags.setdefault(tag or "llm", {"calls": 0, "cached": 0, "live": 0, "errors": 0, "seconds": 0.0,
                                                    "tokens": 0})
            d["calls"] += 1
            d[{"cached": "cached", "live": "live"}.get(outcome, "errors")] += 1
            d["seconds"] += seconds
            d["tokens"] += tokens
            now = time.time()
            if now - self.last < 0.5 and outcome != "error":
                return
            self.last = now
            tot = {k: sum(t[k] for t in self.tags.values()) for k in ("calls", "cached", "live", "errors", "tokens")}
            snap = {t: dict(v) for t, v in self.tags.items()}
        report("llm", tot["calls"], None, "LLM calls", "calls", llm=True, by_tag=snap, cached=tot["cached"],
               live=tot["live"], errors=tot["errors"], tokens=tot["tokens"], elapsed=round(time.time() - self.t0, 1))

    def flush(self) -> None:
        if _sink is None or not self.tags:
            return
        with _lock:
            tot = {k: sum(t[k] for t in self.tags.values()) for k in ("calls", "cached", "live", "errors", "tokens")}
            snap = {t: dict(v) for t, v in self.tags.items()}
        report("llm", tot["calls"], None, "LLM calls", "calls", llm=True, by_tag=snap, cached=tot["cached"],
               live=tot["live"], errors=tot["errors"], tokens=tot["tokens"], elapsed=round(time.time() - self.t0, 1),
               final=True)


_LLM = _LLMCounter()


def llm_call(tag: str, outcome: str, seconds: float = 0.0, tokens: int = 0) -> None:
    """Count one LLM call (``outcome``: cached, live or error)."""
    _LLM.add(tag, outcome, seconds, tokens)


def llm_flush() -> None:
    """Send the LLM counts now (at the end of a step, past the throttle)."""
    _LLM.flush()
