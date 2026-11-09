"""The project configuration (``gmsjourney/policy/<pack>/project.yaml``): the choices the policy graph cannot make.

``Project`` reads the file; ``check(project, domain)`` validates every name and value in it against the
policy graph (a customer frame must be an act the graph declares with slots and values it allows, a
bank status must be one of the tool's statuses, an expression must parse, ...).

    python -m gmsjourney.s01_packs.project check [path]     -> lists every problem, exit 1 if any
    python -m gmsjourney.s01_packs.project show [path]      -> the factors, their levels and what each changes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


from ..s02_graph.flow import FlowError, evaluate, parse
from ..s04_cases.meaning import SemanticFrame
from .names import N

PACKAGE = Path(__file__).resolve().parents[1]            # the gmsjourney package folder
FILE = "project.yaml"                    # in every domain pack folder (gmsjourney/policy/<pack>/project.yaml)
ENV = "GMSJOURNEY_PROJECT"
NA = "N/A"

_FRAME = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*(?:\((.*)\))?\s*$")


def parse_frame(text: str, credential: str | None = None) -> SemanticFrame:
    """``ACT(slot=value, ...)`` -> a meaning card (``$credential`` is the fixture's credential)."""
    m = _FRAME.match(str(text))
    if not m:
        raise ValueError(f"not a customer frame: {text!r} (write ACT(slot=value, ...))")
    slots: dict[str, Any] = {}
    for part in [p.strip() for p in (m.group(2) or "").split(",") if p.strip()]:
        k, _, v = part.partition("=")
        v = v.strip().strip("'\"")
        if v == "$credential":
            v = credential
        slots[k.strip()] = v
    return SemanticFrame.make(m.group(1), **slots)


@dataclass
class Level:
    """One level of a factor: the fixture and customer behaviour it sets, plus its case-building hints
    (boundary value, number of questions, target turn, exact count, transfer to avoid or reach).
    """
    name: Any
    fixture: dict[str, Any] = field(default_factory=dict)
    customer: list[dict[str, Any]] = field(default_factory=list)
    boundary: bool = False
    questions: int = 0
    turn_target: str | None = None
    exact: bool = False
    avoid_transfer: str | None = None
    transfer: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Factor:
    """One factor of the project: its name, what it is about, default level, levels, when it is relevant, its role
    and the decision point (tool or state) it is about.
    """
    name: str
    about: str
    default: Any
    levels: dict[Any, Level]
    relevant_when: str | None = None
    role: str | None = None
    anchor: dict[str, Any] = field(default_factory=dict)   # {tool: T} or {state: S}: the decision point it is about


REACHED = "reached"        # relevant_when: the case reaches the factor's anchor (decided by walking the case)


@dataclass
class Project:
    """A journey's ``project.yaml``: the policy graph, the base fixture and customer, the factors, constraints,
    categories, generator settings and tags.
    """
    path: Path
    raw: dict[str, Any]
    name: str
    policy_graph: Path
    fixture: dict[str, Any]
    customer: dict[str, Any]
    factors: dict[str, Factor]
    constraints: list[str]
    categories: list[tuple[str, str]]
    generator: dict[str, Any]
    tags: list[tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Project":
        p = Path(path or default_path())
        raw = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}      # a pack without one: no factors yet
        factors: dict[str, Factor] = {}
        for f in raw.get("factors", []):
            levels = {}
            for lv, spec in (f.get("levels") or {}).items():
                spec = spec or {}
                levels[lv] = Level(lv, dict(spec.get("fixture") or {}), list(spec.get("customer") or []),
                                   bool(spec.get("boundary")), int(spec.get("questions") or 0), spec.get("turn_target"),
                                   bool(spec.get("exact")), spec.get("avoid_transfer"), spec.get("transfer"), spec)
            factors[f["name"]] = Factor(f["name"], f.get("about", ""), f.get("default", next(iter(levels), None)),
                                        levels, f.get("relevant_when"), f.get("role"), dict(f.get("anchor") or {}))
        proj = raw.get("project") or {}
        graph = pack_of(p).graph_path
        return cls(p, raw, proj.get("name", ""), graph, dict(raw.get("fixture") or {}), dict(raw.get("customer") or {}),
                   factors, list(raw.get("constraints") or []),
                   [(c["name"], str(c["when"])) for c in raw.get("categories") or []], dict(raw.get("generator") or {}),
                   [(c["name"], str(c["when"])) for c in raw.get("tags") or []])

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path or self.path)
        out.write_text(dump_yaml(self.raw))
        return out

    # ------------------------------------------------------------------ helpers
    @property
    def dimensions(self) -> dict[str, list[Any]]:
        return {f.name: list(f.levels) for f in self.factors.values()}

    @property
    def defaults(self) -> dict[str, Any]:
        return {f.name: f.default for f in self.factors.values()}

    def factor_with_role(self, role: str) -> Factor | None:
        return next((f for f in self.factors.values() if f.role == role), None)

    def boundaries(self) -> set[str]:
        return {f"{f.name}={lv}" for f in self.factors.values() for lv, L in f.levels.items() if L.boundary}

    def frame(self, text: str) -> SemanticFrame:
        return parse_frame(text, self.fixture.get("credential"))

    def padding_questions(self) -> list[SemanticFrame]:
        return [self.frame(t) for t in self.generator.get("padding_questions") or []]


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data):              # a value used twice is written twice, never as &id / *id
        return True


def dump_yaml(data: dict[str, Any]) -> str:
    """YAML text for a project dict: keys in order, no anchors, unicode kept, 120 columns."""
    return yaml.dump(data, Dumper=_NoAliases, sort_keys=False, allow_unicode=True, width=120, default_flow_style=None)


def default_path() -> Path:
    """``GMSJOURNEY_PROJECT``, else the active domain pack's project.yaml."""
    if os.environ.get(ENV):
        return Path(os.environ[ENV])
    from .loading import current
    return current().dir / FILE


def pack_of(path: str | Path):
    """The domain pack a project file belongs to: its folder when that is a pack, else the active pack."""
    from .loading import CONFIG_FILE, current, load
    d = Path(path).resolve().parent
    cur = current()
    if d == cur.dir.resolve() or not (d / CONFIG_FILE).exists():
        return cur
    return load(d)


@lru_cache(maxsize=8)
def _load_cached(path: str) -> Project:
    return Project.load(path)


def project(path: str | Path | None = None) -> Project:
    """The project in use (``GMSJOURNEY_PROJECT``, else the active pack's project.yaml), cached."""
    return _load_cached(str(Path(path or default_path()).resolve()))


def reset_cache() -> None:
    """Forget loaded projects (after a project file changed)."""
    _load_cached.cache_clear()


# ----------------------------------------------------------------------------- expressions over a case
class CaseView:
    """What a factor expression can ask about one case."""
    FUNCTIONS = {"level", "rel", "relevant", "items", "min", "max", "limit"}

    def __init__(self, proj: Project, levels: dict[str, Any], n_items: int = 0, limits: Any = None,
                 reach=None):
        self.proj, self.levels, self.n_items, self.limits = proj, levels, n_items, limits
        self._rel: dict[str, bool] = {}
        self._reach = reach                              # factor -> reached? (for relevant_when: reached)

    def f_level(self, name: str) -> Any:
        if name not in self.levels:
            raise FlowError(f"unknown factor {name!r}")
        return self.levels[name]

    def f_relevant(self, name: str) -> bool:
        if name not in self._rel:
            self._rel[name] = False                      # a cycle reads as irrelevant
            when = self.proj.factors[name].relevant_when
            if when == REACHED:
                self._rel[name] = bool(self._reach(name)) if self._reach is not None else True
            else:
                self._rel[name] = bool(evaluate(when, self))
        return self._rel[name]

    def f_rel(self, name: str) -> Any:
        return self.f_level(name) if self.f_relevant(name) else NA

    def f_items(self) -> int:
        return self.n_items

    def f_min(self, *a):
        return min(a)

    def f_max(self, *a):
        return max(a)

    def f_limit(self, key: str) -> Any:
        if self.limits is None:
            raise FlowError("limit() needs the policy's limits")
        return getattr(self.limits, key) if not isinstance(self.limits, dict) else self.limits[key]


# ----------------------------------------------------------------------------- validation against the graph
def fixture_keys() -> dict[str, tuple[str, str]]:
    """Mock-backend fixture key -> (tool, what its value is), read from the pack's tool specs (domain.yaml ->
    tools, or mined from a compact graph): ``statuses`` the results of the 1st, 2nd, ... call, ``status`` one
    status, ``items`` the list a read returns, ``call`` the call number that fails."""
    from ..s06_bots.tools import tool_specs
    out: dict[str, tuple[str, str]] = {}
    for t, sp in tool_specs().items():
        for key, kind in (("results_field", "statuses"), ("unavailable_flag", "status"), ("result_field", "status"),
                          ("field", "items"), ("fail_on_call_field", "call")):
            if sp.get(key):
                out[sp[key]] = (t, kind)
    out.update({k: v for k, v in loop_fixture_keys().items() if k not in out})
    return out


LOOP_KEYS = {"items": "count", "last_item_age": "age", "extra_item_age": "age"}


def loop_fixture_keys() -> dict[str, tuple[str, str]]:
    """The bank settings of a pack that loops over a returned list (its first loop): ``items`` how many,
    ``last_item_age`` the age of the last one, ``extra_item_age`` one more after them (outside the window)."""
    from .loading import current
    g = current().graph
    lps = g.loops() if g is not None else []
    if not lps:
        return {}
    lp = lps[0]
    keys = {"items": (lp["tool"], "count")}
    if lp.get("age"):
        keys.update({"last_item_age": (lp["tool"], "age"), "extra_item_age": (lp["tool"], "age")})
    return keys


def __getattr__(name: str):                     # FIXTURE_TOOL: the map of the active pack
    if name == "FIXTURE_TOOL":
        return {k: v for k, v in fixture_keys().items() if v[1] in ("statuses", "status")}
    raise AttributeError(name)


def check(proj: Project, domain) -> list[str]:
    """Every problem in the project file, read against the policy graph's domain."""
    issues: list[str] = []
    cred = proj.fixture.get("credential")

    def chk_frame(where: str, text: str, script: str | None = None) -> None:
        try:
            fr = parse_frame(text, cred)
        except ValueError as e:
            issues.append(f"{where}: {e}")
            return
        if fr.act not in domain.customer_acts:
            issues.append(f"{where}: {fr.act} is not a customer act in the graph ({', '.join(domain.customer_acts)})")
            return
        spec = domain.slots.get(fr.act, {})
        for k, v in fr.slots:
            if k not in spec:
                issues.append(f"{where}: {fr.act} has no slot {k!r} (slots: {', '.join(spec) or 'none'})")
                continue
            vs = spec[k].get("values_from")
            if vs in domain.value_sets and v not in domain.value_sets[vs]:
                issues.append(f"{where}: {v!r} is not in value set {vs} ({', '.join(domain.value_sets[vs])})")
        if script:
            states = [s for s, sc in (proj.customer.get("answers") or {}).items() if sc == script and s in domain.accepts]
            if states and not any(fr.act in domain.accepts[s] for s in states):
                issues.append(f"{where}: {fr.act} is not accepted in {', '.join(states)} (the states script "
                              f"{script!r} answers)")

    def chk_expr(where: str, expr: str | None, fns=CaseView.FUNCTIONS) -> None:
        if expr:
            try:
                parse(str(expr), frozenset(fns))
            except FlowError as e:
                issues.append(f"{where}: {e}")

    # customer scripts
    answers = proj.customer.get("answers") or {}
    scripts = proj.customer.get("scripts") or {}
    for state, sc in answers.items():
        if state not in domain.states:
            issues.append(f"customer.answers: {state} is not a dialog state in the graph")
        if sc not in scripts:
            issues.append(f"customer.answers: {state} uses unknown script {sc!r}")
    for sc, spec in scripts.items():
        for k in ("default", "unknown_item"):
            if spec.get(k):
                chk_frame(f"customer.scripts.{sc}.{k}", spec[k], sc)
        for i, t in enumerate(spec.get("item_script") or []):
            chk_frame(f"customer.scripts.{sc}.item_script[{i}]", t, sc)
        ex = spec.get("exhausted") or {}
        if ex.get("otherwise"):
            chk_frame(f"customer.scripts.{sc}.exhausted.otherwise", ex["otherwise"], sc)
        if ex.get("repeat_last") and ex["repeat_last"] not in domain.customer_acts:
            issues.append(f"customer.scripts.{sc}.exhausted.repeat_last: {ex['repeat_last']} is not a customer act")
    q = proj.customer.get("questions") or {}
    if q and q.get("script") not in scripts:
        issues.append(f"customer.questions.script: unknown script {q.get('script')!r}")
    # fixture vocabulary
    lookback = domain.limits.get(N.limit.lookback_days) if proj.fixture.get("review_ages") else None
    for i, age in enumerate(proj.fixture.get("review_ages") or []):
        if lookback is not None and age > lookback:
            issues.append(f"fixture.review_ages[{i}] = {age} is outside the review window (lookback_days = {lookback})")
    if len(proj.fixture.get("review_ages") or []) > len(proj.fixture.get("merchants") or []):
        issues.append("fixture: more review_ages than merchants")
    # factors
    names = set(proj.factors)
    for f in proj.factors.values():
        where = f"factors.{f.name}"
        if f.default not in f.levels:
            issues.append(f"{where}.default: {f.default!r} is not one of its levels")
        if f.relevant_when == REACHED:
            if f.anchor.get("tool") and f.anchor["tool"] not in domain.tools:
                issues.append(f"{where}.anchor: {f.anchor['tool']!r} is not a tool in the graph")
            elif f.anchor.get("state") and f.anchor["state"] not in domain.prompt_states:
                issues.append(f"{where}.anchor: {f.anchor['state']!r} is not a prompt state in the graph")
        else:
            chk_expr(f"{where}.relevant_when", f.relevant_when)
        for ref in re.findall(r"(?:level|rel|relevant)\('([^']+)'\)", str(f.relevant_when or "")):
            if ref not in names:
                issues.append(f"{where}.relevant_when: unknown factor {ref!r}")
        for lv, L in f.levels.items():
            w = f"{where}.levels.{lv}"
            for k, v in L.fixture.items():
                if k in fixture_keys() and fixture_keys()[k][1] in ("statuses", "status"):
                    tool, _ = fixture_keys()[k]
                    allowed = domain.tool_outcomes(tool) if tool in domain.tools else ()
                    for x in (v if isinstance(v, list) else [v]):
                        x = x.get("status") if isinstance(x, dict) else x          # a scripted result object
                        if allowed and x not in allowed:
                            issues.append(f"{w}.fixture.{k}: {x!r} is not a status of {tool} ({', '.join(allowed)})")
                elif k == "card_options":
                    allowed = domain.value_sets.get("card_action", ())
                    for x in v or []:
                        if x not in allowed:
                            issues.append(f"{w}.fixture.card_options: {x!r} is not in value set card_action")
                elif k not in fixture_keys() and k not in ("items", "last_item_age", "extra_item", "record_fail_on_call"):
                    issues.append(f"{w}.fixture: unknown mock-bank setting {k!r}")
            for i, op in enumerate(L.customer):
                sc = op.get("script")
                if sc not in scripts:
                    issues.append(f"{w}.customer[{i}].script: unknown script {sc!r}")
                for j, t in enumerate(op.get("say") or []):
                    chk_frame(f"{w}.customer[{i}].say[{j}]", t, sc)
                if op.get("to"):
                    chk_frame(f"{w}.customer[{i}].to", op["to"], sc)
                for k, v in (op.get("add_slots") or {}).items():
                    chk_frame(f"{w}.customer[{i}].add_slots", f"INFORM_DISPOSITION({k}={v})" if op.get("to") is None
                              else f"{parse_frame(op['to'], cred).act}({k}={v})", sc)
                if isinstance(op.get("item"), str) and op["item"] != "all":      # all: every item
                    chk_expr(f"{w}.customer[{i}].item", op["item"])
                for k in (op.get("if_bot") or {}):
                    if k not in {"delivery", "action", "txn", "fee", "disposition"}:
                        issues.append(f"{w}.customer[{i}].if_bot: unknown bot-act field {k!r}")
            if L.turn_target:
                chk_expr(f"{w}.turn_target", L.turn_target)
            for k in ("avoid_transfer", "transfer"):
                v = getattr(L, k)
                if v and v not in domain.triggers:
                    issues.append(f"{w}.{k}: {v!r} is not an escalation trigger in the graph")
    for i, c in enumerate(proj.constraints):
        chk_expr(f"constraints[{i}]", c)
    for i, (n, w) in enumerate(proj.categories):
        chk_expr(f"categories[{i}] ({n})", w)
    for i, (n, w) in enumerate(proj.tags):
        chk_expr(f"tags[{i}] ({n})", w)
    pq = proj.generator.get("policy_question_case") or {}
    for k, v in pq.items():
        if k not in proj.factors:
            issues.append(f"generator.policy_question_case: unknown factor {k!r}")
        elif v not in proj.factors[k].levels:
            issues.append(f"generator.policy_question_case: {v!r} is not a level of {k}")
    for i, t in enumerate(proj.generator.get("padding_questions") or []):
        chk_frame(f"generator.padding_questions[{i}]", t)
    for r in proj.generator.get("fault_only_rules") or []:
        if r not in domain.rules:
            issues.append(f"generator.fault_only_rules: {r} is not a rule in the graph")
    issues += _check_evaluation(proj.raw.get("evaluation") or {})
    for on, states in ((proj.raw.get("speech") or {}).get("also_allowed") or {}).items():
        if on != "any" and on not in domain.customer_acts:
            issues.append(f"speech.also_allowed: {on!r} is not a customer act (or 'any')")
        for st in states or []:
            if st not in domain.states:
                issues.append(f"speech.also_allowed.{on}: {st!r} is not a dialog state in the graph")
    from ..s07_checks.speech_reading import SIGNALS
    sig = (proj.raw.get("speech") or {}).get("signals") or {}
    for name in SIGNALS:
        words = sig.get(name)
        if not words or not isinstance(words, list):
            issues.append(f"speech.signals.{name}: needs a list of words")
            continue
        for w in words:
            try:
                re.compile(str(w))
            except re.error as e:
                issues.append(f"speech.signals.{name}: {w!r} is not a valid pattern ({e})")
    for name in sig:
        if name not in SIGNALS:
            issues.append(f"speech.signals: unknown cue list {name!r} (lists: {', '.join(SIGNALS)})")
    issues += _check_speech_policy((proj.raw.get("speech") or {}).get("policy") or {})
    return issues


POLICY_FINDINGS = ("unread", "volunteered", "disclosed", "text_contradicts_label", "label_not_said", "missing")


def _check_speech_policy(sp: dict[str, Any]) -> list[str]:
    """speech.policy: advisory finding kinds, never_volunteer subjects (limits and values of the graph), the judge and
    marker lists (s07_checks/speech_policy.py)."""
    from .loading import current
    from ..s07_checks.speech_policy import MARKERS
    issues = []
    for k in sp:
        if k not in ("advisory", "never_volunteer", "judge", "llm_model", "markers"):
            issues.append(f"speech.policy: unknown key {k!r} (advisory, never_volunteer, judge, llm_model, markers)")
    for k in sp.get("advisory") or []:
        if k not in POLICY_FINDINGS:
            issues.append(f"speech.policy.advisory: {k!r} is not a finding kind ({', '.join(POLICY_FINDINGS)})")
    if sp.get("judge") not in (None, "graph", "llm"):
        issues.append(f"speech.policy.judge: {sp.get('judge')!r} must be graph or llm")
    names = {t["head"] for t in current().kg.triples}
    for x in sp.get("never_volunteer") or []:
        if x not in names:
            issues.append(f"speech.policy.never_volunteer: {x!r} is not a limit or value in the graph")
    for k, v in (sp.get("markers") or {}).items():
        if k not in MARKERS:
            issues.append(f"speech.policy.markers: unknown list {k!r} (lists: {', '.join(MARKERS)})")
        elif not isinstance(v, list) or not v:
            issues.append(f"speech.policy.markers.{k}: needs a list of words")
    return issues


def _check_evaluation(ev: dict[str, Any]) -> list[str]:
    """Severities, tier promotions and release-gate profiles against the codes and tiers the scorer emits."""
    from dataclasses import fields

    from ..s07_checks.scoring import CODES, LEVELS, TIERS, TierProfile
    out = []
    sev = ev.get("severity") or {}
    for c in CODES:
        if c not in sev:
            out.append(f"evaluation.severity: no severity for failure code {c}")
    for c, v in sev.items():
        if c not in CODES:
            out.append(f"evaluation.severity: {c!r} is not a failure code the scorer emits")
        elif v not in LEVELS:
            out.append(f"evaluation.severity.{c}: {v!r} is not one of {LEVELS}")
    for t, m in (ev.get("tier_escalation") or {}).items():
        if t not in TIERS:
            out.append(f"evaluation.tier_escalation: unknown risk tier {t!r} (tiers: {TIERS})")
        for a, b in (m or {}).items():
            if a not in LEVELS or b not in LEVELS:
                out.append(f"evaluation.tier_escalation.{t}: {a!r} -> {b!r} is not a pair of severities")
    allowed = {f.name for f in fields(TierProfile)} - {"tier"}
    for t, p in (ev.get("release_gate") or {}).items():
        if t not in TIERS:
            out.append(f"evaluation.release_gate: unknown risk tier {t!r} (tiers: {TIERS})")
        for k in p or {}:
            if k not in allowed:
                out.append(f"evaluation.release_gate.{t}: unknown setting {k!r} (settings: {sorted(allowed)})")
        for c in (p or {}).get("failure_budget") or {}:
            if c not in CODES:
                out.append(f"evaluation.release_gate.{t}.failure_budget: {c!r} is not a failure code")
    return out


def derived(domain) -> dict[str, Any]:
    """What the graph suggests as factors and levels (shown next to the configured ones in the studio)."""
    out: dict[str, Any] = {"tool_outcomes": {}, "limit_boundaries": {}, "customer_choices": {}}
    for t in domain.tools.values():
        out["tool_outcomes"][t.name] = {"success": t.success, "failures": list(t.failures), "on_failure": t.on_failure}
    for k, v in domain.limits.items():
        if isinstance(v, (int, float)):
            out["limit_boundaries"][k] = {"at": v, "just_over": v + 1, "just_under": v - 1}
    for s in domain.prompt_states:
        out["customer_choices"][s] = domain.options_for(s)
    return out


def main(argv=None) -> None:
    """Command line: ``check`` a project against its graph, or ``show`` it."""
    ap = argparse.ArgumentParser(prog="python -m gmsjourney.s01_packs.project")
    ap.add_argument("cmd", choices=["check", "show"])
    ap.add_argument("path", nargs="?", default=None)
    a = ap.parse_args(argv)
    from .names import GraphNames
    from .loading import set_current
    path = a.path or default_path()
    set_current(pack_of(path))                       # the project's own pack: its tools, loops and names
    proj = Project.load(path)
    dom = GraphNames.from_kg(pack_of(proj.path).kg)
    if a.cmd == "check":
        issues = check(proj, dom)
        for i in issues:
            print("-", i)
        print(f"{proj.path}: {'ok' if not issues else f'{len(issues)} problem(s)'}")
        sys.exit(1 if issues else 0)
    print(json.dumps({f.name: {"default": f.default, "levels": list(f.levels), "relevant_when": f.relevant_when}
                      for f in proj.factors.values()}, indent=1, default=str))




from .loading import on_switch as _on_switch  # noqa: E402

_on_switch(reset_cache)


if __name__ == "__main__":
    main()
