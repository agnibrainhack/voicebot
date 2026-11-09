"""The project studio: a local page to read the policy graph and edit the project configuration.

    python -m gmsjourney.s10_console.studio [--pack supplementary_card] [--project path/to/project.yaml] [--port 8765]

or, from a notebook:  ``url = studio.start("supplementary_card")`` and show it in an IFrame.

then open http://127.0.0.1:8765. The page shows the graph's core (dialog states and their transitions,
tools and their statuses, limits, value sets, rules, escalation triggers), what the harness derives from
it (the nomenclature, the outcomes and customer choices the factors cover or miss), and the project's own
choices: the mock bank, the simulated customer, the journey factors and their levels, constraints,
categories and tags. Any combination of levels can be previewed: the oracle walks the graph against it
and the page shows the expected conversation. Save checks the file against the graph and writes it (the
previous version is kept as ``<file>.bak``).

Runs on the symbolic store compiled straight from the graph (about a second; no GMS training, no licence).
Listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import yaml


from ..s04_cases import journey
from ..s01_packs.names import GraphNames
from ..s02_graph.flow import FlowSpec, GraphOracle
from ..s02_graph.triples import PolicyKG
from ..s01_packs.project import NA, Project, check, derived, dump_yaml
from ..s01_packs.names import N
from ..s01_packs import names as roles

UI = Path(__file__).resolve().parent / "studio_ui.html"

SECTION_NOTES = {
    "project": "the project and the policy graph it reads",
    "fixture": "the mock bank every case runs against; factor levels patch it",
    "customer": "how the simulated customer answers each bot act; factor levels patch the scripts",
    "factors": "the content factors of a journey; each level is a patch (applied in this order)",
    "constraints": "every case must satisfy these",
    "categories": "the first that applies names a case's base category",
    "tags": "every tag that applies is attached to a case",
    "generator": "how the case generator pads calls and which rules only tool faults reach",
    "evaluation": "severity of each failure code, the tier promotions and the release gate of each risk tier",
    "speech": "speech-check leniencies the graph does not state, and the plain-English cues the reader looks for",
}


def to_yaml(raw: dict[str, Any]) -> str:
    """The project as YAML, a short note above each section (comments in a hand-edited file are not kept)."""
    out = ["# gmsjourney project configuration (written by the studio; check it with: python -m gmsjourney.s01_packs.project check)",
           "# Names and values are checked against the policy graph. Expressions use level(f), rel(f), relevant(f),",
           "# items(), min, max, limit(key). Customer frames are ACT(slot=value, ...); $credential is the fixture's.", ""]
    for key, val in raw.items():
        note = SECTION_NOTES.get(key)
        out.append(f"# {'-' * 90}")
        if note:
            out.append(f"# {key}: {note}")
        out.append(dump_yaml({key: val}).rstrip())
        out.append("")
    return "\n".join(out) + "\n"


from ..s03_knowledge.store import symbolic_store  # noqa: E402  (re-exported: tests and tools import it from here)


class Studio:
    """One journey's project in the console: the graph's view (states, tools, acts, rules, limits), the project,
    checks as you edit, coverage, a Preview of any level, and lossless save.
    """
    def __init__(self, project_path: str | None = None, pack: str | None = None):
        from ..s01_packs.loading import current, set_current
        from ..s01_packs.project import pack_of
        dom = set_current(pack) if pack else current()
        self.project = Project.load(project_path or dom.project_path)
        self.pack = pack_of(self.project.path)
        self.kg = self.pack.kg
        self.domain = GraphNames.from_kg(self.kg)
        self.store = symbolic_store(self.pack.graph_path, self.kg)
        self.flow = FlowSpec.from_kg(self.kg)

    # ------------------------------------------------------------------ views
    def domain_view(self) -> dict[str, Any]:
        D, kg = self.domain, self.kg
        states = []
        for s in D.states:
            states.append({
                "name": s, "stage": D.stage_of.get(s), "prompts": s in D.prompt_states,
                "accepts": list(D.accepts.get(s, ())), "yes": (D.yes_no.get(s) or (None, None))[0],
                "no": (D.yes_no.get(s) or (None, None))[1],
                "rules": kg.tails(s, "applies_rule"), "carries": [{"field": f, "from": e} for f, e, _ in self.flow.carries.get(s, [])],
                "out": [{"dst": t.dst, "on": t.on, "when": t.when, "match": t.match, "guard": t.guard, "do": list(t.do),
                         "say": list(t.say), "rules": list(t.rules)} for t in self.flow.out.get(s, [])],
            })
        tools = [{"name": t.name, "stage": t.stage, "success": t.success, "failures": list(t.failures),
                  "on_failure": t.on_failure, "args": [{"name": a, "from": e} for a, e in self.flow.args.get(t.name, [])],
                  "returns": t.returns} for t in D.tools.values()]
        acts = [{"name": a, "slots": {k: {**q, "values": list(D.value_sets.get(q.get("values_from"), ()))}
                                      for k, q in D.slots.get(a, {}).items()}} for a in D.customer_acts]
        return {
            "title": D.title, "sha": self.kg.sha256[:12], "summary": D.summary(), "stages": list(D.stages),
            "states": states, "tools": tools, "acts": acts, "limits": D.limits,
            "internal_limits": self._internal_limits(),
            "value_sets": {k: [{"value": v, "aliases": len(D.aliases.get(v, ()))} for v in vals]
                           for k, vals in D.value_sets.items()},
            "rules": [{"id": r.rid, "stance": r.stance, "text": r.text, "fact": r.fact, "source": r.source}
                      for r in D.rules.values()],
            "triggers": [{"name": t.name, "condition": t.condition, "rule": t.rule, "states": list(t.states)}
                         for t in D.triggers.values()],
            "names": {k: list(v) for k, v in D.all_names().items()},
            "fees": {o: {"fee": self.store.lookup_enm("delivery_fee_usd", o)}
                     for o in D.value_sets.get("delivery", ())} if D.value_sets.get("delivery") and
            getattr(self.pack, "graph", None) is None else {},        # charge review's fees; a compiled pack: figures
            "has_items": bool(D.item_tool) and getattr(self.pack, "graph", None) is None,   # charge review's charges
            "figures": self._figures(),
            "loop": self._loop(),
        }

    def _loop(self) -> dict[str, Any] | None:
        """A compiled pack's loop over a returned list: what it reads, its cap and window, and the item records."""
        g = getattr(self.pack, "graph", None)
        lps = g.loops() if g is not None else []
        if not lps:
            return None
        lp = lps[0]
        spec = (g.tools[lp["tool"]].get("returns") or {})[lp["field"]]
        fx = self.pack.get(f"fixtures.items.{spec['items']}") or {}
        return {"state": lp["state"], "tool": lp["tool"], "field": lp["field"], "prompts": lp["prompts"],
                "cap": {lp["cap"]: g.limit_value(lp["cap"])} if lp.get("cap") else None,
                "window": {f: {k: (v, g.limit_value(v)) for k, v in lim.items()} for f, lim in lp["where"].items()},
                "id": lp["id"], "age": lp["age"], "records": fx.get("records") or [], "ages": fx.get("ages") or [],
                "id_prefix": fx.get("id_prefix", "I")}

    @staticmethod
    def _internal_limits() -> list[str]:
        """Limits the graph marks internal (a compact graph's ``internal: true``): never told to the customer."""
        from ..s01_packs.loading import current
        g = current().graph
        return sorted(k for k, lv in ((g.limits if g is not None else {}) or {}).items() if lv.get("internal"))

    def _figures(self) -> dict[str, dict[str, Any]]:
        """value -> attribute -> the figure the policy states (a pack compiled from a compact graph)."""
        g = getattr(self.pack, "graph", None)
        if g is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for attr, spec in g.numeric_attributes().items():
            for m in spec["members"]:
                out.setdefault(m, {})[attr] = g.attr(m, attr)
        return out

    def coverage(self, proj: Project) -> dict[str, Any]:
        """What the configured levels reach: graph outcomes and customer choices covered or missed, and the
        transitions, rules and escalation triggers one-level-at-a-time variations exercise."""
        from ..s01_packs.project import FIXTURE_TOOL
        D = self.domain
        statuses: dict[str, set] = {t: set() for t in D.tools}
        said: set[tuple[str, str]] = set()
        answers = proj.customer.get("answers") or {}
        for f in proj.factors.values():
            for L in f.levels.values():
                for k, v in L.fixture.items():
                    if k in FIXTURE_TOOL:                    # a status, or a result object carrying one
                        statuses[FIXTURE_TOOL[k][0]].update(x.get("status") if isinstance(x, dict) else x
                                                            for x in (v if isinstance(v, list) else [v]))
                    if k == "record_fail_on_call":
                        statuses[roles.RECORD].add(N.status.FAILED)
                    if k == "card_options":
                        statuses[roles.OPTIONS].add(N.status.OPTIONS if v else N.status.NONE)
                    if k == "items":
                        statuses[roles.ITEMS].add(N.status.TRANSACTIONS if v else N.status.EMPTY)
                for op in L.customer:
                    for t in op.get("say") or []:
                        act = t.split("(")[0].strip()
                        for st, sc in answers.items():
                            if sc == op.get("script") and st in D.prompt_states:
                                said.add((st, act))
        for t in D.tools.values():                          # what every case gets by default
            if t.success:
                statuses[t.name].add(t.success)
        sc_default = {}
        for name, sc in (proj.customer.get("scripts") or {}).items():
            for k in ("default", "unknown_item"):
                if sc.get(k):
                    sc_default.setdefault(name, set()).add(sc[k].split("(")[0].strip())
            for t in sc.get("item_script") or []:
                sc_default.setdefault(name, set()).add(t.split("(")[0].strip())
        for st, sc in answers.items():
            for act in sc_default.get(sc, ()):
                said.add((st, act))
        for st in D.prompt_states:                         # policy questions are asked by the generator
            if N.act.ASK_KB in D.accepts.get(st, ()) and (proj.customer.get("questions") or {}).get("script") == answers.get(st):
                said.add((st, N.act.ASK_KB))
        tool_gaps = {t: [s for s in D.tool_outcomes(t) if s not in statuses[t]] for t in D.tools}
        choice_gaps = {st: [a for a in D.accepts.get(st, ()) if (st, a) not in said] for st in D.prompt_states}
        # every level, in a few valid cases around it (other factors drawn at random)
        import random
        journey.activate(proj)
        oracle = GraphOracle(self.store, spec=self.flow)
        rng = random.Random(0)
        qrole = proj.factor_with_role("questions")
        pool = proj.padding_questions()
        took, rules, triggers, per_level = set(), set(), set(), {}
        dims = proj.dimensions
        for f in proj.factors.values():
            for lv in f.levels:
                runs, reach = 0, {"ends": {}, "transfers": {}}
                for _ in range(60):
                    if runs >= 8:
                        break
                    d = {k: rng.choice(v) for k, v in dims.items()}
                    d[f.name] = lv
                    spec = journey.JourneySpec(**d)
                    if not journey.is_valid(spec):
                        continue
                    if qrole is not None and pool and (proj.factors[qrole.name].levels[getattr(spec, qrole.name)].questions):
                        spec = spec.with_(kb_frames=(rng.choice(pool),))
                    try:
                        tr = oracle.run(spec, journey.build_fixture(spec))
                    except Exception as e:               # noqa: BLE001
                        reach["error"] = str(e)
                        break
                    runs += 1
                    took |= {f"{a}>{b}:{o}" for a, b, o in tr.flow_path}
                    rules |= tr.rules_fired
                    if tr.transfer_reason:
                        triggers.add(tr.transfer_reason)
                        reach["transfers"][tr.transfer_reason] = reach["transfers"].get(tr.transfer_reason, 0) + 1
                    reach["ends"][tr.end_state] = reach["ends"].get(tr.end_state, 0) + 1
                reach["cases"] = runs
                per_level[f"{f.name}={lv}"] = reach
        all_tr = {f"{t.src}>{t.dst}:{t.on}" for trs in self.flow.out.values() for t in trs}
        return {"tool_outcomes_missing": {k: v for k, v in tool_gaps.items() if v},
                "customer_choices_missing": {k: v for k, v in choice_gaps.items() if v},
                "transitions_reached": len(took & all_tr), "transitions_total": len(all_tr),
                "transitions_not_reached": sorted(all_tr - took),
                "rules_not_reached": sorted(set(D.rules) - rules),
                "triggers_not_reached": sorted(set(D.triggers) - triggers), "per_level": per_level}

    def state(self) -> dict[str, Any]:
        return {"path": str(self.project.path), "project": self.project.raw, "domain": self.domain_view(),
                "derived": derived(self.domain), "issues": check(self.project, self.domain),
                "coverage": self.coverage(self.project), "fixture_keys": self.fixture_keys()}

    def fixture_keys(self) -> dict[str, str]:
        """The bank settings a level can patch: charge review's named settings, or tool:<name> for a pack
        compiled from a compact graph (what that tool returns, call by call)."""
        if getattr(self.pack, "graph", None) is None:
            return FIXTURE_KEYS
        from ..s01_packs.project import fixture_keys
        return {k: f"{t} returns these, call by call (a status, or a result object with its fields)"
                for k, (t, _) in fixture_keys().items()}

    # ------------------------------------------------------------------ actions
    def _project(self, raw: dict[str, Any]) -> Project:
        normalize(raw)
        tmp = Path(self.project.path).with_suffix(".studio-check.yaml")
        tmp.write_text(dump_yaml(raw))
        try:
            return Project.load(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def check(self, raw: dict[str, Any]) -> dict[str, Any]:
        proj = self._project(raw)
        return {"issues": check(proj, self.domain), "coverage": self.coverage(proj)}

    def preview(self, raw: dict[str, Any], levels: dict[str, Any], questions: list[str]) -> dict[str, Any]:
        proj = self._project(raw)
        journey.activate(proj)
        levels = {k: _coerce(v, proj.factors[k].levels) for k, v in levels.items() if k in proj.factors}
        qs = tuple(proj.frame(q) for q in questions if q.strip())
        oracle = GraphOracle(self.store, spec=self.flow)
        partner = None
        if len(levels) == 1 and not qs and self.pack.graph is not None:   # one level: make sure the call gets there
            (name, lv), = levels.items()
            levels, partner = reach(proj, lambda sp: oracle.run(sp, journey.build_fixture(sp)), name, lv)
        spec = journey.JourneySpec(kb_frames=qs, **levels)
        fx = journey.build_fixture(spec)
        tr = oracle.run(spec, fx)
        conv = []
        for e in tr.events:
            if e["role"] == "bot":
                conv.append({"role": "bot", "acts": e["acts"]})
            elif e["role"] == "user":
                conv.append({"role": "customer", "frame": frame_text(e["frame"])})
            else:
                conv.append({"role": "tool", "tool": e["tool"], "args": e.get("args"), "result": e.get("result"),
                             "ok": e.get("ok", True)})
        rel = journey.relevant_dims(spec)
        return {"spec": spec.dims(), "valid": journey.is_valid(spec), "relevant": {k: v != NA for k, v in rel.items()},
                "category": journey.primary_category(spec), "tags": journey.tags_for(spec), "fixture": fx,
                "conversation": conv, "signature": json.loads(tr.signature()), "rules": sorted(tr.rules_fired),
                "turns": tr.turns, "path": [list(p) for p in tr.flow_path], "notes": tr.notes,
                "components": tr.components(), "with": partner}

    def save(self, raw: dict[str, Any]) -> dict[str, Any]:
        proj = self._project(raw)                       # normalizes raw in place
        issues = check(proj, self.domain)
        p = Path(self.project.path)
        if p.exists():
            shutil.copyfile(p, p.with_suffix(p.suffix + ".bak"))
        p.write_text(to_yaml(raw))
        self.project = Project.load(p)
        journey.activate(self.project)
        return {"saved": str(p), "issues": issues}


FIXTURE_KEYS = {
    "auth_results": "authenticate_customer returns these statuses, call by call",
    "transactions_status": "get_review_transactions status (UNAVAILABLE: the call fails)",
    "items": "how many reviewable charges the bank holds",
    "last_item_age": "age in days of the last charge (edge of the window)",
    "extra_item": "one more charge {id, merchant (index), age, at (position)}",
    "record_fail_on_call": "the n-th record_charge_response returns FAILED",
    "record_status": "record_charge_response status (UNAVAILABLE: the call fails)",
    "card_options": "the card actions get_card_action_options returns",
    "card_options_status": "get_card_action_options status (UNAVAILABLE: the call fails)",
    "submit_result": "submit_card_action status",
    "submit_status": "submit_card_action availability (UNAVAILABLE: the call fails)",
}


def level_site(proj: Project, name: str, lv: Any) -> tuple[set[str], set[str]]:
    """Where a level acts: the prompts whose customer answer it changes, the tools whose answer it sets (else the
    factor's decision point). Empty when the project does not say (factors written with relevance expressions)."""
    f = proj.factors[name]
    L = f.levels.get(lv)
    states: set[str] = set()
    tools: set[str] = set()
    answers = proj.customer.get("answers") or {}
    for op in (L.customer if L is not None else []):
        states |= {st for st, sc in answers.items() if sc == op.get("script")}
    for k in (L.fixture if L is not None else {}):
        if str(k).startswith("tool:"):
            tools.add(str(k)[5:])
    if not states and not tools:
        a = f.anchor or {}
        if a.get("state"):
            states.add(a["state"])
        if a.get("tool"):
            tools.add(a["tool"])
    return states, tools


def _visits(tr) -> tuple[set[str], set[str]]:
    """The prompts the customer answered and the tools called in a trace."""
    states = {e.get("state") for e in tr.events if e.get("role") == "user"}
    tools = {e.get("tool") for e in tr.events if e.get("role") == "tool"}
    return states, tools


def reach(proj: Project, run, name: str, lv: Any, base: dict[str, Any] | None = None,
          budget: int = 1500) -> tuple[dict[str, Any], str | None]:
    """The levels to play so that the call gets to where ``name = lv`` acts. With every other factor at its default
    the call can miss it (a person asked for at the retry prompt, which the default call never reaches); then the
    first other level, or pair of levels, that takes the call there is added. Returns (levels, "other = level" or
    None when nothing needed adding); "" when no combination within the budget gets there. ``run(spec)`` plays a
    JourneySpec and returns its trace."""
    base = dict(base or {})
    want_s, want_t = level_site(proj, name, lv)
    if not want_s and not want_t:
        return {**base, name: lv}, None
    runs = [0]

    def gets_there(d) -> bool:
        sp = journey.JourneySpec(**d)
        if not journey.is_valid(sp):
            return False
        runs[0] += 1
        st, tl = _visits(run(sp))
        return bool(want_s & st or want_t & tl)

    others = [(g, lv2) for g, f in proj.factors.items() if g != name and g not in base and not f.role
              for lv2 in f.levels if lv2 != f.default]
    try:
        d_st, d_tl = _visits(run(journey.JourneySpec(**base)))     # levels that act where the default call goes first
        here = lambda o: bool(level_site(proj, *o)[0] & d_st or level_site(proj, *o)[1] & d_tl)   # noqa: E731
        others.sort(key=lambda o: not here(o))
    except Exception:                                    # noqa: BLE001
        pass
    try:
        if gets_there({**base, name: lv}):
            return {**base, name: lv}, None
        for g, lv2 in others:
            if gets_there({**base, g: lv2, name: lv}):
                return {**base, g: lv2, name: lv}, f"{g} = {lv2}"
        for i, (g1, v1) in enumerate(others):
            for g2, v2 in others[i + 1:]:
                if g2 == g1:
                    continue
                if runs[0] >= budget:
                    raise StopIteration
                if gets_there({**base, g1: v1, g2: v2, name: lv}):
                    return {**base, g1: v1, g2: v2, name: lv}, f"{g1} = {v1} and {g2} = {v2}"
    except Exception:                                    # noqa: BLE001 - out of budget, or a combination the oracle cannot play
        pass
    return {**base, name: lv}, ""


def _num(v: Any) -> Any:
    return int(v) if isinstance(v, str) and v.lstrip("-").isdigit() else v


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Undo what JSON does to the project in the browser: level names like 0, 1, 2 come back as strings."""
    for f in raw.get("factors") or []:
        f["levels"] = {_num(k): v for k, v in (f.get("levels") or {}).items()}
        for lv in f["levels"].values():               # the editor adds empty patches while drawing
            for k in ("fixture", "customer"):
                if isinstance(lv, dict) and k in lv and not lv[k]:
                    del lv[k]
        f["default"] = _num(f.get("default"))
    pq = (raw.get("generator") or {}).get("policy_question_case") or {}
    for k in list(pq):
        pq[k] = _num(pq[k])
    return raw


def frame_text(f) -> str:
    """A meaning card as the config writes it: ACT(slot=value, ...)."""
    return f"{f.act}({', '.join(f'{k}={v}' for k, v in f.slots)})"


def _coerce(v: Any, levels: dict) -> Any:
    for lv in levels:
        if str(lv) == str(v):
            return lv
    return v


def make_server(studio: Studio, host: str = "127.0.0.1", port: int = 8765):
    """The studio's HTTP server (not started)."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                     # quiet
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

        def do_GET(self):                               # noqa: N802
            if self.path in ("/", "/index.html"):
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                try:
                    return self._json(studio.state())
                except Exception as e:                  # noqa: BLE001
                    return self._json({"error": str(e), "trace": traceback.format_exc()}, 500)
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):                              # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                raw = body.get("project") or studio.project.raw
                if self.path == "/api/check":
                    return self._json(studio.check(raw))
                if self.path == "/api/preview":
                    return self._json(studio.preview(raw, body.get("levels") or {}, body.get("questions") or []))
                if self.path == "/api/save":
                    return self._json(studio.save(raw))
                if self.path == "/api/yaml":
                    return self._json({"yaml": to_yaml(raw)})
            except Exception as e:                      # noqa: BLE001
                return self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}, 400)
            return self._send(404, b"not found", "text/plain")

    from http.server import ThreadingHTTPServer
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), H)


def serve(studio: Studio, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve until Ctrl-C (command line)."""
    httpd = make_server(studio, host, port)
    print(f"gmsjourney studio: http://{host}:{port}  (project: {studio.project.path}; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


_RUNNING: dict[int, Any] = {}


def start(pack: str | None = None, project: str | None = None, port: int = 8765, host: str = "127.0.0.1") -> str:
    """Start the studio in a background thread (for a notebook) and return its URL. Starting again on the same port
    replaces the running one (switching the pack it edits)."""
    import threading
    stop(port)
    httpd = make_server(Studio(project, pack=pack), host, port)
    threading.Thread(target=httpd.serve_forever, daemon=True, name=f"gmsjourney-studio-{port}").start()
    _RUNNING[port] = httpd
    return f"http://{host}:{port}/"


def stop(port: int = 8765) -> None:
    """Stop a studio started with ``start``."""
    httpd = _RUNNING.pop(port, None)
    if httpd is not None:
        httpd.shutdown()
        httpd.server_close()


def main(argv=None) -> None:
    """Command line: serve the standalone project studio for one pack."""
    ap = argparse.ArgumentParser(prog="python -m gmsjourney.s10_console.studio")
    ap.add_argument("--pack", default=None, help="domain pack (name or folder); default: $GMSJOURNEY_DOMAIN / charge_review")
    ap.add_argument("--project", default=None, help="project file (default: the pack's project.yaml)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    serve(Studio(a.project, pack=a.pack), port=a.port)


if __name__ == "__main__":
    main()
