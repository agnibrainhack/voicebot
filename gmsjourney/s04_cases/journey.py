"""Journey content: what a test case *is*, before any presentation factor.

The content factors, their levels and what each level changes come from the project configuration
(``gmsjourney/policy/<pack>/project.yaml``, read by ``s01_packs/project.py``); their names and values are checked against
the policy graph. A level is a patch to two scripts:

* the **bank fixture** (what the mock backend returns: authentication results, the charges and their ages,
  the card options, the submit result, tool failures), and
* the **customer script** (what the simulated customer says in reply to each bot act: per script a queue
  of meaning cards, then a default).

The oracle (``s02_graph/flow.py``, ``GraphOracle``) replays any combination of levels by walking the policy graph against
those two scripts, so a new factor or level needs no new oracle code.

Every factor here changes the correct behaviour of the bot, which is the book's test for "base, not
factor" (Ch. 1.2): a knob that would move the answer belongs to the base.

* ``JourneySpec``      one case: a level per factor, plus the policy questions the customer asks.
* ``relevant_dims``    the factors that can influence this case's ground truth (others read N/A).
* ``is_valid``         the project's constraints between factors.
* ``build_fixture``    the bank fixture for a case.
* ``UserSimulator``    the customer, answering each bot act from the case's scripts. It is reactive, so a
                       test does not break when a bot phrases or orders its prompts differently.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from typing import Any


from ..s02_graph.flow import evaluate
from .meaning import SemanticFrame as F
from ..s01_packs.project import NA, CaseView, Project, parse_frame, project
from ..s01_packs.names import N

_P: Project = project()
DIMENSIONS: dict[str, list[Any]] = _P.dimensions
CREDENTIAL: str = str(_P.fixture.get("credential", ""))
SESSION_DATE: str = str(_P.fixture.get("session_date", date.today().isoformat()))
MERCHANTS: list[tuple[str, str]] = [tuple(m) for m in _P.fixture.get("merchants", [])]
REVIEW_AGES: list[int] = list(_P.fixture.get("review_ages", []))


def activate(proj: Project) -> None:
    """Build every case from ``proj`` from now on (the studio previews unsaved edits this way)."""
    global _P, CREDENTIAL, SESSION_DATE
    _P = proj
    DIMENSIONS.clear()
    DIMENSIONS.update(proj.dimensions)
    CREDENTIAL = str(proj.fixture.get("credential", ""))
    SESSION_DATE = str(proj.fixture.get("session_date", date.today().isoformat()))
    MERCHANTS[:] = [tuple(m) for m in proj.fixture.get("merchants", [])]
    REVIEW_AGES[:] = list(proj.fixture.get("review_ages", []))
    import sys
    gen = sys.modules.get("gmsjourney.s04_cases.design")
    if gen is not None:                              # the case generator reads the same project
        gen._activate(proj)


def active_project() -> Project:
    """The project the case functions currently read (set by ``activate``)."""
    return _P


def _on_pack_switch() -> None:
    activate(project())


from ..s01_packs.loading import on_switch as _on_switch  # noqa: E402

_on_switch(_on_pack_switch)


class JourneySpec:
    """One case: a level for every configured factor, plus the customer's policy questions.

    ``JourneySpec(n_txn=2, disposition="mixed")`` - unnamed factors take their default level. Factor levels
    read as attributes (``spec.n_txn``)."""

    __slots__ = ("_levels", "kb_frames", "extra_kb")

    def __init__(self, kb_frames: tuple = (), extra_kb: int = 0, **levels: Any):
        unknown = set(levels) - set(DIMENSIONS)
        if unknown:
            raise TypeError(f"unknown factor(s) {sorted(unknown)}; the project defines {list(DIMENSIONS)}")
        d = dict(_P.defaults)
        d.update(levels)
        object.__setattr__(self, "_levels", tuple((k, d[k]) for k in DIMENSIONS))
        object.__setattr__(self, "kb_frames", tuple(kb_frames))
        object.__setattr__(self, "extra_kb", int(extra_kb))

    def __getattr__(self, name: str) -> Any:
        for k, v in object.__getattribute__(self, "_levels"):
            if k == name:
                return v
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("JourneySpec is immutable; use with_()")

    def __eq__(self, other) -> bool:
        return isinstance(other, JourneySpec) and (self._levels, self.kb_frames, self.extra_kb) == \
            (other._levels, other.kb_frames, other.extra_kb)

    def __hash__(self) -> int:
        return hash((self._levels, self.kb_frames, self.extra_kb))

    def __repr__(self) -> str:
        lv = ", ".join(f"{k}={v!r}" for k, v in self._levels)
        return f"JourneySpec({lv}, kb_frames={self.kb_frames!r}, extra_kb={self.extra_kb})"

    def key(self) -> str:
        d: dict[str, Any] = dict(self._levels)
        d["extra_kb"] = self.extra_kb
        d["kb_frames"] = [f.canonical() for f in self.kb_frames]
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def dims(self) -> dict[str, Any]:
        return dict(self._levels)

    def with_(self, **kw: Any) -> "JourneySpec":
        d = self.dims()
        kb, extra = kw.pop("kb_frames", self.kb_frames), kw.pop("extra_kb", self.extra_kb)
        d.update(kw)
        return JourneySpec(kb_frames=kb, extra_kb=extra, **d)


def _view(spec: JourneySpec, n_items: int = 0, limits: Any = None) -> CaseView:
    return CaseView(_P, spec.dims(), n_items, limits)


def relevant_dims(spec: JourneySpec) -> dict[str, Any]:
    """The factors that can influence ground truth for this case (others are N/A). A factor with
    ``relevant_when: reached`` is relevant when the case reaches it: walking the case through the graph, its
    anchor is reached (default level) or everything its level scripts is used (another level)."""
    reach = None
    if any(f.relevant_when == "reached" for f in _P.factors.values()):
        info = _reach(spec)
        reach = info.__getitem__
    v = CaseView(_P, spec.dims(), 0, None, reach)
    return {k: (lv if v.f_relevant(k) else NA) for k, lv in spec.dims().items()}


_REACH: dict[tuple, dict[str, bool]] = {}


def _reach(spec: JourneySpec) -> dict[str, bool]:
    """factor -> reached, from one walk of the case on the pack's symbolic store (cached per case)."""
    key = (id(_P), spec.key())
    if key in _REACH:
        return _REACH[key]
    from ..s03_knowledge.store import pack_oracle
    from ..s01_packs.project import fixture_keys
    oracle = pack_oracle()
    fx = build_fixture(spec)
    sim = UserSimulator(spec, fx, getattr(oracle.limits, "lookback_days", None))
    tr = oracle.run(spec, fx, responder=sim.respond)
    called = {c["tool"] for c in tr.tool_log}
    keys = fixture_keys()
    out = {}
    for f in _P.factors.values():
        lv = getattr(spec, f.name)
        L = f.levels.get(lv)
        if lv == f.default or L is None or (not L.fixture and not L.customer):
            a = f.anchor
            out[f.name] = (a.get("tool") in called) if a.get("tool") else \
                (sim.answered.get(a["state"], 0) > 0) if a.get("state") else True
            continue
        ok = all(keys.get(k, (k.partition(":")[2] if k.startswith("tool:") else None, ""))[0] in called
                 for k in L.fixture)
        ok = ok and (not L.customer or f.name in sim.used_by)
        out[f.name] = ok
    if len(_REACH) > 50_000:
        _REACH.clear()
    _REACH[key] = out
    return out


def is_valid(spec: JourneySpec) -> bool:
    """The project's constraints between factors."""
    v = _view(spec)
    return all(evaluate(c, v) for c in _P.constraints)


def primary_category(spec: JourneySpec) -> str:
    """The base category the case's distinguishing content exercises (the project's first matching rule)."""
    v = _view(spec)
    return next((name for name, when in _P.categories if evaluate(when, v)), "uncategorized")


def tags_for(spec: JourneySpec) -> list[str]:
    """Every project tag that applies to the case, with its category."""
    v = _view(spec)
    return sorted({primary_category(spec)} | {name for name, when in _P.tags if evaluate(when, v)})


def policy_question_case(question: F) -> JourneySpec:
    """The case a single policy question is asked in (project ``generator.policy_question_case``)."""
    return JourneySpec(kb_frames=(question,), **(_P.generator.get("policy_question_case") or {}))


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------
def build_fixture(spec: JourneySpec) -> dict[str, Any]:
    """The mock bank for a case: the project's defaults, patched by every factor's level in order.

    Every key is a fixture field of the pack's tools (``project.fixture_keys``): for a compact pack the
    results of the 1st, 2nd, ... call of each tool (``tool:<name>``). A pack whose items are listed in the
    fixture (``merchants``) gets them built here: ``items`` how many, their ages, an extra out-of-window one."""
    fxc = _P.fixture
    fx: dict[str, Any] = {"session_date": SESSION_DATE, "credential": CREDENTIAL}
    settings: dict[str, Any] = copy.deepcopy(fxc.get("defaults") or {})
    for f in _P.factors.values():
        lv = f.levels.get(getattr(spec, f.name))
        if lv is not None:
            settings.update(copy.deepcopy(lv.fixture))
    if not MERCHANTS:                                # no item list: the settings are the fixture
        fx.update(_loop_fixture(settings))
        return fx
    n = int(settings.pop("items", 0) or 0)
    last_age = settings.pop("last_item_age", None)
    extra = settings.pop("extra_item", None)
    prefix = fxc.get("item_id_prefix", "T")
    txns = []
    for i in range(n):
        m, amt = MERCHANTS[i]
        age = REVIEW_AGES[i]
        if last_age is not None and i == n - 1:
            age = int(last_age)
        txns.append({"transaction_id": f"{prefix}{i + 1}", "merchant": m, "amount": amt, "date": _date(age),
                     "age_days": age})
    if extra:
        m, amt = MERCHANTS[int(extra.get("merchant", len(MERCHANTS) - 1))]
        age = int(extra["age"])
        txns.insert(min(int(extra.get("at", 0)), len(txns)),
                    {"transaction_id": extra.get("id", f"{prefix}{len(txns) + 1}"), "merchant": m, "amount": amt,
                     "date": _date(age), "age_days": age})
    fx["auth_results"] = settings.pop("auth_results", [N.status.AUTHENTICATED])
    fx["transactions"] = txns
    fx["card_options"] = settings.pop("card_options", [])
    fx["submit_result"] = settings.pop("submit_result", N.status.SUCCESS)
    fx.update(settings)
    return fx


def _date(age_days: int) -> str:
    """The calendar date of a charge that is ``age_days`` old on SESSION_DATE (ISO, e.g. 2026-03-13)."""
    return (date.fromisoformat(SESSION_DATE) - timedelta(days=age_days)).isoformat()


def _loop_fixture(settings: dict[str, Any]) -> dict[str, Any]:
    """A compiled pack that loops over a returned list: the list the bank returns, built from the pack's item
    records (``items`` how many, default 1; ``last_item_age``; ``extra_item_age``), unless a level scripts that tool
    itself; a tool returning some members of a value set returns all of them by default."""
    from ..s01_packs.loading import current
    g = current().graph
    if g is None:
        return settings
    out = dict(settings)
    lps = g.loops()
    if lps:
        from .factors import loop_items
        lp = lps[0]
        n = int(out.pop("items", 1) or 0)
        last, extra = out.pop("last_item_age", None), out.pop("extra_item_age", None)
        key = f"tool:{lp['tool']}"
        if key not in out:
            sp = g.tools[lp["tool"]]
            spec = (sp.get("returns") or {})[lp["field"]]
            items = loop_items(spec, n, last, extra)
            out[key] = [{"status": sp["ok"] if items else (sp.get("fail") or [sp["ok"]])[0], lp["field"]: items}]
    for t, sp in g.tools.items():                    # some members of a value set: all of them by default
        for fld, typ in (sp.get("returns") or {}).items():
            if isinstance(typ, dict) and "subset_of" in typ and f"tool:{t}" not in out:
                full = list(current().get(f"fixtures.options.{typ['subset_of']}") or g.members(typ["subset_of"]))
                out[f"tool:{t}"] = [{"status": sp["ok"], fld: full}]
    return out


def loop_ids(fixture: dict[str, Any]) -> list[str]:
    """The ids of the items a compiled pack's loop will handle: the list its tool returns, inside the window."""
    from ..s01_packs.loading import current
    g = current().graph
    lps = g.loops() if g is not None else []
    if not lps:
        return []
    lp = lps[0]
    res = (fixture.get(f"tool:{lp['tool']}") or [{}])[0]
    items = res.get(lp["field"]) if isinstance(res, dict) else None
    out = []
    for x in items or []:
        ok = True
        for fld, lim in lp["where"].items():
            v = x.get(fld)
            if v is None or ("max" in lim and v > g.limit_value(lim["max"])) or ("min" in lim and v < g.limit_value(lim["min"])):
                ok = False
        if ok:
            out.append(x.get(lp["id"]))
    return out


def reviewable_ids(fixture: dict[str, Any], lookback: int | None) -> list[str]:
    """The ids of the items a customer is asked about: inside the lookback window (or a loop's items)."""
    if "transactions" not in fixture:                # a compiled pack's loop (none: no per-item scripts)
        return loop_ids(fixture)
    if lookback is None:
        return []
    return [t["transaction_id"] for t in fixture.get("transactions", []) if t["age_days"] <= lookback]


# --------------------------------------------------------------------------
# The customer
# --------------------------------------------------------------------------
def _reasks() -> set[str]:
    """Acts after which the prompt that was pending is still open (a clarification, a policy answer)."""
    return {a for a in ("ASK_CLARIFY", "ANSWER_KB") if a in N.state}


class _Say:
    """One queued meaning card: said once (or every time, ``sticky``), only when the bot act matches ``if_bot``."""
    __slots__ = ("frame", "sticky", "if_bot", "owner", "also")

    def __init__(self, frame: F, sticky: bool = False, if_bot: dict | None = None, owner: str | None = None):
        self.frame, self.sticky, self.if_bot, self.owner = frame, sticky, dict(if_bot or {}), owner
        self.also: tuple[str, ...] = ()              # factors that added slots to this frame (used when it is said)


class UserSimulator:
    """The customer of one case: answers each bot act from the case's scripts (project ``customer``)."""

    def __init__(self, spec: JourneySpec, fixture: dict[str, Any], lookback: int = 30):
        self.spec, self.fixture, self.lookback = spec, fixture, lookback
        self.emitted: list[F] = []
        self.answered: dict[str, int] = {}          # bot act -> how many times the customer answered it
        self.last_state: str | None = None          # the prompt the customer last answered
        self.used_by: set[str] = set()              # factors whose scripted frames were said
        self._item: Any = None                      # the item the customer was last asked about (per-item scripts)
        from ..s01_packs.loading import current
        self._compiled = current().graph is not None
        cfg = _P.customer
        self.answers: dict[str, str] = dict(cfg.get("answers") or {})
        self.scripts: dict[str, dict[str, Any]] = dict(cfg.get("scripts") or {})
        self.ids = reviewable_ids(fixture, lookback)
        cred = fixture.get("credential", CREDENTIAL)
        self._frame = lambda t: parse_frame(t, cred)                     # noqa: E731
        self.queues: dict[tuple[str, Any], list[_Say]] = {}
        self._last: dict[tuple[str, Any], F] = {}
        for name, sc in self.scripts.items():
            if sc.get("per_item"):
                for tid in self.ids:
                    self.queues[(name, tid)] = [_Say(self._frame(t)) for t in sc.get("item_script") or []]
            else:
                self.queues[(name, None)] = []
        view = _view(spec, len(self.ids))
        for f in _P.factors.values():
            lv = f.levels.get(getattr(spec, f.name))
            for op in (lv.customer if lv is not None else []):
                self._apply(op, view, f.name)
        q = cfg.get("questions") or {}
        if spec.kb_frames and q:
            self._apply({"script": q.get("script"), "item": q.get("item", 0), "position": q.get("position", "start"),
                         "frames": list(spec.kb_frames)}, view)

    # ------------------------------------------------------------------ building the scripts
    def _keys(self, op: dict[str, Any], view: CaseView) -> list[tuple[str, Any]]:
        sc = op.get("script")
        if not self.scripts.get(sc, {}).get("per_item"):
            return [(sc, None)]
        item = op.get("item", 0)
        if item == "all":                            # every item answers this way
            return [(sc, tid) for tid in self.ids]
        idx = evaluate(item, view) if isinstance(item, str) else int(item)
        return [(sc, self.ids[idx])] if 0 <= idx < len(self.ids) else []

    def _apply(self, op: dict[str, Any], view: CaseView, owner: str | None = None) -> None:
        for key in self._keys(op, view):
            q = self.queues.setdefault(key, [])
            if op.get("add_slots"):
                to = self._frame(op["to"]) if op.get("to") else None
                for s in q:
                    if to is None or (s.frame.act == to.act and all(s.frame.get(k) == v for k, v in to.slots)):
                        s.frame = F.make(s.frame.act, **{**dict(s.frame.slots), **op["add_slots"]})
                        if owner and owner != s.owner:
                            s.also = (*s.also, owner)
                continue
            frames = list(op.get("frames", ()))
            if op.get("questions"):                  # this case's policy questions, asked here
                frames += list(self.spec.kb_frames)
            new = [_Say(f, bool(op.get("sticky")), op.get("if_bot"), owner) for f in frames] or \
                  [_Say(self._frame(t), bool(op.get("sticky")), op.get("if_bot"), owner) for t in op.get("say") or []]
            pos = op.get("position", "append")
            if pos == "replace":
                q[:] = new
            elif pos == "start":
                q[:0] = new
            else:
                q.extend(new)

    # ------------------------------------------------------------------ answering
    def _emit(self, f: F) -> F:
        self.emitted.append(f)
        return f

    def respond(self, bot_act: dict[str, Any]) -> F | None:
        act = bot_act.get("act")
        if act not in self.answers and act in _reasks():            # a clarification or a policy answer that
            act = bot_act.get("state") or self.last_state          # did not re-ask: answer the pending prompt
        script = self.answers.get(act)
        if script is None:
            return None                              # terminal or unknown: the customer stays silent
        self.last_state = act
        self.answered[act] = self.answered.get(act, 0) + 1
        sc = self.scripts.get(script, {})
        if sc.get("per_item"):
            tid = bot_act.get("txn") or bot_act.get("current_txn") or \
                next((v for v in bot_act.values() if isinstance(v, str) and v in self.ids), None) or \
                (self._item if self._compiled else None)   # a re-ask, or a prompt about the item in hand (a
            #                                           confirmation that names no item): the item last asked about
            self._item = tid
            key = (script, tid)
            if key not in self.queues:               # a transaction the case never scripted
                return self._emit(self._frame(sc["unknown_item"])) if sc.get("unknown_item") else None
        else:
            key = (script, None)
        q = self.queues.get(key, [])
        for i, s in enumerate(q):
            if s.if_bot and any(bot_act.get(k) != v for k, v in s.if_bot.items()):
                continue
            if not s.sticky:
                q.pop(i)
            for o in (s.owner, *s.also):
                if o:
                    self.used_by.add(o)
            return self._remember(key, sc, s.frame)
        if not sc.get("per_item"):
            ex = sc.get("exhausted") or {}
            if ex.get("repeat_last_answer") and key in self._last:     # the customer sticks to their last answer
                return self._emit(self._last[key])
            return self._emit(self._frame(sc["default"])) if sc.get("default") else None
        ex = sc.get("exhausted") or {}                # the bot keeps asking after the script ran out
        last = self._last.get(key)
        if last is not None:
            return self._emit(last)
        return self._emit(self._frame(ex["otherwise"])) if ex.get("otherwise") else None

    def _remember(self, key, sc: dict[str, Any], f: F) -> F:
        ex = sc.get("exhausted") or {}
        if ex.get("repeat_last") and f.act == ex["repeat_last"]:
            self._last[key] = f
        if ex.get("repeat_last_answer") and f.act not in (ex.get("not_repeated") or ()):
            self._last[key] = f
        return self._emit(f)
