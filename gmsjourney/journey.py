"""Journey content: what a test case *is*, before any presentation factor.

``JourneySpec`` holds the content dimensions of one base case. Every dimension changes the
correct behaviour of the bot, which is the book's test for "base, not factor" (Ch. 1.2): a
knob that would move the answer belongs to the base. Tool unavailability, ambiguity that
persists, a customer correction that flips the answer and the turn limit all live here for
that reason, even though a generic catalog files some of them as presentation factors.

The dimensions, their defaults, which combinations are invalid, which dimensions are
relevant to a case, how a case becomes a backend fixture and what the simulated customer
says all come from the domain pack (domain.yaml -> journey, fixtures); this module only
applies them.

``build_fixture`` turns a spec into a deterministic backend fixture. ``UserSimulator``
answers each bot act with the next semantic frame the spec dictates. It is reactive, so a
test does not break when a bot phrases or orders its prompts differently; it only needs to
know what the bot is asking for.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .ontology import SemanticFrame as F

NA = "N/A"


# --------------------------------------------------------------------------
# The journey model of the active domain
# --------------------------------------------------------------------------
class JourneyModel:
    def __init__(self, dom):
        self.dom = dom
        self.cfg = dom.section("journey")
        self.fixtures = dom.section("fixtures")
        # one dict object per domain: the generator may narrow a dimension temporarily
        self.dimensions: dict[str, list[Any]] = {k: list(v) for k, v in self.cfg["dimensions"].items()}
        self.defaults: dict[str, Any] = dict(self.cfg["defaults"])

    @staticmethod
    def frame(spec: dict[str, Any]) -> F:
        return F.make(spec["act"], **{k: v for k, v in spec.items() if k != "act"})


_MODELS: dict[int, JourneyModel] = {}


def model() -> JourneyModel:
    from .domain import current
    dom = current()
    if id(dom) not in _MODELS:
        _MODELS[id(dom)] = JourneyModel(dom)
    return _MODELS[id(dom)]


def __getattr__(name: str):                       # module constants of the active domain
    if name.startswith("__"):
        raise AttributeError(name)
    if name == "DIMENSIONS":
        return model().dimensions
    fx = model().fixtures
    if name == "MERCHANTS":
        return [(r["merchant"], r["amount"]) for r in fx["items"]["records"]]
    if name == "REVIEW_AGES":
        return list(fx["items"]["ages"])
    if name == "CREDENTIAL":
        return fx["credential"]
    raise AttributeError(name)


def holds(cond: dict[str, Any] | None, values: dict[str, Any]) -> bool:
    """A journey condition: {dim: [levels]} / {dim: {not: [levels]}} (all hold), {any: [...]},
    {not: condition}. An absent condition holds."""
    if not cond:
        return True
    for k, v in cond.items():
        if k == "any":
            if not any(holds(c, values) for c in v):
                return False
        elif k == "not":
            if holds(v, values):
                return False
        elif isinstance(v, dict) and "not" in v:
            if values.get(k) in v["not"]:
                return False
        elif values.get(k) not in v:
            return False
    return True


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------
class JourneySpec:
    """One base case: a level per dimension, plus the policy questions asked before the first
    answer and the padding used to hit a turn target. Immutable; compare and hash by value."""

    __slots__ = ("_dims", "kb_frames", "extra_kb")

    def __init__(self, kb_frames: tuple = (), extra_kb: int = 0, **dims: Any):
        m = model()
        unknown = set(dims) - set(m.dimensions)
        if unknown:
            raise TypeError(f"JourneySpec: unknown dimension(s) {sorted(unknown)}")
        object.__setattr__(self, "_dims", {k: dims.get(k, m.defaults[k]) for k in m.dimensions})
        object.__setattr__(self, "kb_frames", tuple(kb_frames))
        object.__setattr__(self, "extra_kb", extra_kb)

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_dims")
        if name in d:
            return d[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("JourneySpec is immutable; use with_()")

    def key(self) -> str:
        d = dict(self._dims)
        d["extra_kb"] = self.extra_kb
        d["kb_frames"] = [f.canonical() for f in self.kb_frames]
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def dims(self) -> dict[str, Any]:
        return dict(self._dims)

    def with_(self, **kw: Any) -> "JourneySpec":
        kb = kw.pop("kb_frames", self.kb_frames)
        extra = kw.pop("extra_kb", self.extra_kb)
        return JourneySpec(kb_frames=kb, extra_kb=extra, **{**self._dims, **kw})

    def __eq__(self, other) -> bool:
        return isinstance(other, JourneySpec) and self._dims == other._dims and \
            self.kb_frames == other.kb_frames and self.extra_kb == other.extra_kb

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self._dims.items()]
        parts += [f"kb_frames={self.kb_frames!r}", f"extra_kb={self.extra_kb!r}"]
        return f"JourneySpec({', '.join(parts)})"


def relevant_dims(spec: JourneySpec) -> dict[str, Any]:
    """The dimensions that actually influence ground truth for this spec (others are N/A)."""
    m = model()
    d = spec.dims()
    out: dict[str, Any] = {}
    for step in m.cfg["relevance"]:
        if step.get("add") and holds(step.get("if"), d):
            for k in step["add"]:
                out[k] = d[k]
        if step.get("stop") and holds(step["stop"], d):
            break
    return {**{k: NA for k in m.dimensions}, **out}


def is_valid(spec: JourneySpec) -> bool:
    """Static constraints between content dimensions."""
    d = spec.dims()
    return not any(holds(c, d) for c in model().cfg.get("invalid", []))


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------
def _resolve(v: Any) -> Any:
    if isinstance(v, dict) and set(v) == {"fixture"}:
        return copy.deepcopy(model().fixtures[v["fixture"]])
    return copy.deepcopy(v)


def _items_step() -> dict[str, Any]:
    return next(st["items"] for st in model().cfg["fixture"] if "items" in st)


def build_fixture(spec: JourneySpec) -> dict[str, Any]:
    m = model()
    d = spec.dims()
    fx: dict[str, Any] = {}
    for step in m.cfg["fixture"]:
        if "set" in step:
            fx.update({k: _resolve(v) for k, v in step["set"].items()})
        elif "dimension" in step:
            fx.update({k: _resolve(v) for k, v in (step["levels"].get(d[step["dimension"]]) or {}).items()})
        elif "items" in step:
            it, cfg = step["items"], m.fixtures["items"]
            recs, ages, prefix = cfg["records"], cfg["ages"], cfg["id_prefix"]
            idf, agef = cfg["id_field"], cfg["age_field"]
            n = d[it["count"]]
            win = (it.get("window_levels") or {}).get(d.get(it.get("window"))) or {}
            items = []
            for i in range(n):
                age = ages[i]
                if "last_age" in win and i == n - 1:
                    age = win["last_age"]
                items.append({idf: f"{prefix}{i + 1}", **recs[i], agef: age})
            if "extra" in win:
                x = win["extra"]
                items.insert(min(x["insert_at"], len(items)),
                             {idf: f"{prefix}{x['id']}", **recs[x["record"]], agef: x["age"]})
            fx[it["field"]] = items
    return fx


def reviewable_ids(fixture: dict[str, Any], lookback: int) -> list[str]:
    cfg = model().fixtures["items"]
    return [t[cfg["id_field"]] for t in fixture[_items_step()["field"]] if t[cfg["age_field"]] <= lookback]


# --------------------------------------------------------------------------
# User simulator
# --------------------------------------------------------------------------
def _card_slots(spec: JourneySpec) -> dict[str, Any]:
    req = model().cfg["customer"]["request"]
    return dict(req["levels"].get(spec.dims()[req["dimension"]], {}))


def disposition_script(spec: JourneySpec, index: int) -> list[F]:
    """Frames the customer gives for the index-th presented item."""
    m = model()
    ans = m.cfg["customer"]["answers"]
    if index > 0:
        return [m.frame(a) for a in ans["later"]]
    level = spec.dims()[ans["dimension"]]
    if level not in ans["levels"]:
        raise ValueError(level)
    out = []
    for a in ans["levels"][level]:
        a = dict(a)
        if a.pop("with_request", False):
            a.update(_card_slots(spec))
        out.append(m.frame(a))
    return out


@dataclass
class UserSimulator:
    spec: JourneySpec
    fixture: dict[str, Any]
    lookback: int = 30
    queues: dict[str, list[F]] = field(default_factory=dict)
    switched: bool = False
    emitted: list[F] = field(default_factory=list)
    _last_disp: dict[str, F] = field(default_factory=dict)
    _human_used: bool = False

    def __post_init__(self) -> None:
        m = model()
        cu = m.cfg["customer"]
        ids = reviewable_ids(self.fixture, self.lookback)
        hm = cu.get("human_mid")
        human_at = min(hm["item"], len(ids) - 1) if (hm and holds(hm["when"], self.spec.dims()) and ids) else None
        for i, tid in enumerate(ids):
            q: list[F] = []
            if i == 0:
                q += list(self.spec.kb_frames)
            if human_at == i:
                q.append(m.frame(cu["human"]))
            q += disposition_script(self.spec, i)
            self.queues[tid] = q

    def _emit(self, f: F) -> F:
        self.emitted.append(f)
        return f

    def respond(self, bot_act: dict[str, Any]) -> F | None:
        m = model()
        cu = m.cfg["customer"]
        asks = cu["asks"]
        act = bot_act.get("act")
        if act in asks["credential"]:
            if cu.get("at_credential_human") and holds(cu["at_credential_human"], self.spec.dims()):
                return self._emit(m.frame(cu["human"]))
            cr = cu["credential"]
            return self._emit(F.make(cr["act"], **{cr["slot"]: m.fixtures["credential"]}))
        if act in asks["item"]:
            tid = bot_act.get(cu["item_field"]) or bot_act.get(cu["item_field_alt"])
            q = self.queues.get(tid)
            ans = cu["answers"]
            if q is None:                       # an item the spec never scripted
                return self._emit(m.frame(ans["unscripted"]))
            if q:
                f = q.pop(0)
                if f.act == ans["exhausted"]["repeat_last"]:
                    self._last_disp[tid] = f
                return self._emit(f)
            # queue exhausted but the bot keeps asking: repeat the latest clear answer
            return self._emit(self._last_disp.get(tid, m.frame(ans["exhausted"]["else"])))
        if act in asks["confirm"]:
            sw = cu.get("switch")
            if (sw and all(bot_act.get(k) == v for k, v in sw["if_bot"].items()) and holds(sw["when"], self.spec.dims())
                    and not self.switched):
                self.switched = True
                return self._emit(m.frame(sw["say"]))
            return self._emit(m.frame(cu["confirm"]))
        if act in asks["wrap_up"]:
            return self._emit(m.frame(cu["wrap_up"]))
        return None                              # terminal or unknown: customer stays silent
