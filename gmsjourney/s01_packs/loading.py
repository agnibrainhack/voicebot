"""Loading a journey pack, and checking that its configuration agrees with its policy graph.

A *journey pack* is one folder under ``gmsjourney/policy/<name>/``: the policy graph (``*.json``), ``domain.yaml``
(how customers word things, how the reference bot words its lines, the mock tools), ``project.yaml`` (the test
plan) and the enrichment catalog layer (``application.yaml``, ``custom_factors.yaml``). Think of the pack as a
cartridge: the engine is the same for every journey, the cartridge says which journey it plays.

Parts of this file:
  1. ``Domain`` and ``load`` / ``set_current`` / ``current``: read a pack (a compact graph's sections are mined
     from the graph, with ``domain.yaml`` laid on top) and make it the active one; ``on_switch`` hooks reset the
     caches that belong to the previous pack; ``lint`` lists every disagreement between config and graph.
  2. The lint checks (``CHECKS``): lexicon against the graph's values and acts (with a read-back round trip),
     tools, figures, fixtures, rules, flow and journeys. Each returns a list of problems; empty means clean.

Run ``python -m gmsjourney.s01_packs.loading <pack>`` to lint a pack from a shell.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# =====================================================================================================================
# PART 1 - Domain packs: one policy graph plus one ``domain.yaml`` per bot under test.
#
# A domain pack is a folder under ``gmsjourney/policy/`` (or anywhere on disk)::
#
#     policy/<domain>/
#         <graph>.json          the policy graph (facts, dialog states, tools, rules, aliases)
#         domain.yaml           how to test it (roles, flow, tools, fixtures, figures, lexicon,
#                               kb, rules, journey, enrichment, campaign)
#         application.yaml      the enrichment catalog layer (factor selection and levels)
#         custom_factors.yaml   factors this application adds to the official catalogs
#         <transcription>.md    optional: the policy as structured markdown
#
# The graph is the single source of truth for what is true and allowed; ``domain.yaml`` only
# names graph elements, gives them roles and adds what a graph cannot carry. Every component
# reads the same loaded ``Domain`` (``domain.current()``), so the ontology, journeys,
# enrichment, oracle and runtime validation cannot drift apart. ``lint()`` checks every
# cross-reference between the config and its graph before a run.
# =====================================================================================================================

POLICY_ROOT = Path(__file__).resolve().parents[1] / "policy"
DEFAULT_DOMAIN = "charge_review"
CONFIG_FILE = "domain.yaml"


class DomainError(ValueError):
    """A domain pack is missing, malformed, or does not match its graph."""
    pass


@dataclass
class Domain:
    """One loaded domain pack: its folder, ``domain.yaml`` settings, the graph (compact or compiled) and helpers to
    read sections, limits and file paths.
    """
    name: str
    dir: Path
    config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ files
    @property
    def graph_path(self) -> Path:
        return self.dir / self.section("domain")["graph"]

    @property
    def project_path(self) -> Path:
        """The pack's project file: fixture, customer, journey factors, scoring and speech settings."""
        return self.dir / "project.yaml"

    @property
    def transcription_path(self) -> Path | None:
        t = self.section("domain").get("transcription")
        return (self.dir / t) if t else None

    @property
    def catalog_dir(self) -> Path:
        """Where application.yaml / custom_factors.yaml live (the official catalogs are shared)."""
        return self.dir

    # ------------------------------------------------------------------ config access
    def section(self, name: str) -> dict[str, Any]:
        return self.config.get(name) or {}

    def get(self, dotted: str, default: Any = None) -> Any:
        """``domain.get("roles.item.name")``: a nested config value, or ``default``."""
        cur: Any = self.config
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    @property
    def root(self) -> str:
        return self.section("domain")["root"]

    @property
    def engine(self) -> str:
        """``graph`` for a compact policy graph (schema v2, s02_graph/compact.py): the dialog, journeys,
        checks and scoring are mined from the graph. ``patterns`` for a v1 graph driven by the
        flow patterns of domain.yaml."""
        return "graph" if self.graph is not None else "patterns"

    @property
    def graph(self):
        """The compact policy graph (graph.model.PolicyGraph) of a v2 pack, else None."""
        return self.__dict__.get("_graph")

    # ------------------------------------------------------------------ figures
    def limit_keys(self) -> list[str]:
        """The graph's limits, in graph order (is_a <figures.limits.class>)."""
        cls = self.get("figures.limits.class", "limit")
        return [t["head"] for t in self.kg.triples if t["relation"] == "is_a" and t["tail"] == cls]

    def enm_for_relation(self) -> dict[str, str]:
        """Numeric relation -> the ENM category that backs it."""
        return {r: sp["enm_category"] for r, sp in (self.get("figures.relations") or {}).items()}

    def kx_enm_map(self) -> dict[str, tuple[str, str]]:
        """ENM category -> (knowlytix register type, key format)."""
        f = self.section("figures")
        out = {f["limits"]["enm_category"]: (f["limits"]["kx"]["type"], f["limits"]["kx"]["key"])}
        for sp in (f.get("relations") or {}).values():
            out[sp["enm_category"]] = (sp["kx"]["type"], sp["kx"]["key"])
        return out

    # ------------------------------------------------------------------ the graph
    @property
    def kg(self):
        if "_kg" not in self.__dict__:
            if self.graph is not None:              # a compact graph, compiled into the full one (flow, speech)
                from ..s02_graph.compile import compile_pack
                self.__dict__["_kg"] = compile_pack(self.graph)
            else:
                from ..s02_graph.triples import PolicyKG
                self.__dict__["_kg"] = PolicyKG.load(self.graph_path)
        return self.__dict__["_kg"]

    def __repr__(self) -> str:
        return f"Domain({self.name!r}, {self.dir})"


# ---------------------------------------------------------------------- loading
def _resolve_dir(spec: str | Path | None) -> Path:
    if spec is None or spec == "":
        spec = os.environ.get("GMSJOURNEY_DOMAIN") or DEFAULT_DOMAIN
    p = Path(spec)
    if p.suffix == ".json" and p.exists():          # a graph file: its folder is the pack
        return p.resolve().parent
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve().parent
    if p.is_dir():
        return p.resolve()
    cand = POLICY_ROOT / str(spec)
    if cand.is_dir():
        return cand
    raise DomainError(f"no domain pack {spec!r}: expected a folder with {CONFIG_FILE} "
                      f"(looked in {p} and {cand})")


def load(spec: str | Path | None = None) -> Domain:
    """A domain by name (``charge_review``), folder, graph file or config file;
    default: $GMSJOURNEY_DOMAIN, else ``charge_review``."""
    d = _resolve_dir(spec)
    cfg_path = d / CONFIG_FILE
    if not cfg_path.exists():
        raise DomainError(f"{d} has no {CONFIG_FILE}")
    config = yaml.safe_load(cfg_path.read_text()) or {}
    gfile = (config.get("domain") or {}).get("graph")
    graph = None
    if gfile:
        from ..s02_graph.compact import PolicyGraph, deep_merge, derive_config, is_v2
        if is_v2(d / gfile):
            # a compact policy graph: every section it implies is mined from it, domain.yaml on top
            graph = PolicyGraph.load(d / gfile)
            config = deep_merge(derive_config(graph, gfile), config)
            # a value, act or slot member the graph has and the lexicon does not yet word gets drafted wording,
            # so an edited graph runs while its wording is written (lexicon_draft.fill_lexicon)
            from .drafting import fill_lexicon
            config["lexicon"], drafted = fill_lexicon(config.get("lexicon"), graph)
            # and a state that speaks with no line of the bot's gets a drafted one (lexicon_draft.fill_bot_says)
            from .drafting import fill_bot_says
            bot = dict(config.get("bot") or {})
            bot["says"], said = fill_bot_says(bot.get("says"), graph)
            config["bot"], drafted = bot, drafted + said
    dom = Domain(name=(config.get("domain") or {}).get("id", d.name), dir=d, config=config)
    if graph is not None:
        dom.__dict__["_graph"] = graph
        dom.__dict__["lexicon_drafted"] = drafted
    if not dom.graph_path.exists():
        raise DomainError(f"{cfg_path}: graph {dom.graph_path.name} not found")
    return dom


_CURRENT: Domain | None = None


def current() -> Domain:
    """The domain every component reads (loaded on first use)."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = load()
    return _CURRENT


def set_current(dom: Domain | str | Path | None) -> Domain:
    """Make a pack the active one (by name, folder, graph or config file). Everything that reads the active
    pack (graph names, project, lexicon, tools) reads the new one from here on."""
    global _CURRENT
    _CURRENT = dom if isinstance(dom, Domain) else load(dom)
    for hook in _ON_SWITCH:
        hook()
    return _CURRENT


_ON_SWITCH: list = []


def on_switch(fn):
    """Register a cache reset to run whenever the active pack changes."""
    _ON_SWITCH.append(fn)
    return fn


# ---------------------------------------------------------------------- lint
def lint(dom: Domain | None = None) -> list[str]:
    """Every problem found between the config and its graph (empty = clean)."""
    dom = dom or current()
    errs: list[str] = []
    d = dom.section("domain")
    for key in ("id", "graph", "root"):
        if key not in d:
            errs.append(f"domain.{key} missing")
    try:
        kg = dom.kg
    except Exception as e:                                # noqa: BLE001 - reported, not raised
        return errs + [f"graph does not load: {e}"]
    errs += [f"graph: {e}" for e in kg.validate()]
    heads = {t["head"] for t in kg.triples} | {t["tail"] for t in kg.triples if isinstance(t["tail"], str)}
    if d.get("root") and d["root"] not in heads:
        errs.append(f"domain.root {d['root']!r} is not an entity of the graph")
    if dom.transcription_path is not None and not dom.transcription_path.exists():
        errs.append(f"domain.transcription {dom.transcription_path.name} not found")
    if not (dom.catalog_dir / "application.yaml").exists():
        errs.append("application.yaml (enrichment catalog layer) not found in the domain folder")
    for chk in CHECKS:
        if dom.engine in getattr(chk, "engines", ("patterns", "graph")):
            errs += chk(dom, kg)
    return errs


def main(argv: list[str] | None = None) -> int:
    """Command line: lint a domain pack against its policy graph (exit 1 when there are problems)."""
    import argparse
    ap = argparse.ArgumentParser(description="Lint a domain pack against its policy graph.")
    ap.add_argument("domain", nargs="?", default=None)
    a = ap.parse_args(argv)
    dom = load(a.domain)
    errs = lint(dom)
    print(f"{dom.name} ({dom.dir}): " + ("clean" if not errs else f"{len(errs)} problem(s)"))
    for e in errs:
        print("  -", e)
    return 1 if errs else 0


# =====================================================================================================================
# PART 2 - Cross-checks between a domain pack's config and its policy graph (run by ``domain.lint``).
#
# Each check returns a list of problems; an empty list means the section agrees with the graph.
# =====================================================================================================================


CHECKS: list = []           # every section check, run in order by domain.lint


def check(fn):
    """Register a lint check (a function ``(domain, graph) -> list of problems``)."""
    CHECKS.append(fn)
    return fn


def patterns_only(fn):
    """A check of a hand-written section; for a graph pack (engine graph) that section is mined
    from the graph, so the check does not apply."""
    fn.engines = ("patterns",)
    return fn


def _value_sets(kg) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for t in kg.triples:
        if t["relation"] == "member_of":
            out.setdefault(t["tail"], []).append(t["head"])
    return out


def _dialog_acts(kg) -> set[str]:
    return {t["tail"] for t in kg.triples if t["relation"] == "accepts_act"}


@check
def lexicon_matches_graph(dom: Domain, kg) -> list[str]:
    """Lint: every lexicon slot is a value set of the graph with exactly that set's members, and the aliasing levels
    and depths are consistent.
    """
    lx = dom.section("lexicon")
    if not lx:
        return ["lexicon section missing"]
    errs: list[str] = []
    sets = _value_sets(kg)
    levels = lx.get("aliasing_levels", [])
    depths = lx.get("depths", [])
    # every slot is a graph value set, and its values are exactly that set's members
    for slot, spec in (lx.get("slots") or {}).items():
        if slot not in sets:
            errs.append(f"lexicon.slots.{slot}: not a value set of the graph (member_of)")
            continue
        missing = set(sets[slot]) - set(spec.get("order", []))
        extra = set(spec.get("order", [])) - set(sets[slot])
        if missing:
            errs.append(f"lexicon.slots.{slot}: graph members not listed: {sorted(missing)}")
        if extra:
            errs.append(f"lexicon.slots.{slot}: not members of the graph's value set: {sorted(extra)}")
        if spec.get("pick") not in ("final_with_correction", "latest_first_mention", "latest_last_mention"):
            errs.append(f"lexicon.slots.{slot}.pick: unknown rule {spec.get('pick')!r}")
    # every value: a phrase per aliasing level and reader cues that compile
    for v, spec in (lx.get("values") or {}).items():
        for lvl in levels:
            if lvl not in (spec.get("say") or {}):
                errs.append(f"lexicon.values.{v}.say: no {lvl!r} phrase")
        for c in spec.get("cues", []):
            try:
                re.compile(c)
            except re.error as e:
                errs.append(f"lexicon.values.{v}.cues: {c!r} does not compile ({e})")
        if not any(v in m for m in sets.values()):
            errs.append(f"lexicon.values.{v}: not a member of any graph value set")
    # acts: graph dialog acts (a realizer-only distractor such as CHITCHAT may have no cues)
    gacts = _dialog_acts(kg)
    for a, spec in (lx.get("acts") or {}).items():
        if a not in gacts and (spec.get("cues") or spec.get("slot") or spec.get("slots") or spec.get("code")):
            errs.append(f"lexicon.acts.{a}: the graph does not accept this act in any state")
        say = spec.get("say")
        if isinstance(say, dict) and "slot" not in spec:
            for d in depths:
                if d not in say:
                    errs.append(f"lexicon.acts.{a}.say: no {d!r} template")
    for a in gacts - set(lx.get("acts") or {}) - {"ASK_KB"}:
        errs.append(f"lexicon.acts: graph act {a} has no templates")
    rd = lx.get("reading") or {}
    for a in list(rd.get("first", [])) + list(rd.get("then", [])) + list(rd.get("last_cue", [])):
        if not ((lx.get("acts") or {}).get(a) or {}).get("cues"):
            errs.append(f"lexicon.reading: {a} has no cues")
    return errs


@check
def lexicon_round_trip(dom: Domain, kg) -> list[str]:
    """Every value phrase and act template, at every aliasing level and paraphrase depth, must
    read back as the meaning it was generated from."""
    if not dom.section("lexicon"):
        return []
    from ..s05_rewording.wording import Lexicon
    from ..s05_rewording.wording import FrameReader
    from ..s05_rewording.wording import Realizer
    from ..s04_cases.meaning import SemanticFrame as F

    if lexicon_matches_graph(dom, kg):
        return []                       # structural problems first; the round trip needs a sound lexicon
    lx = Lexicon(dom.section("lexicon"))
    aliases: dict[str, list[str]] = {}
    for t in kg.triples:
        if t["relation"] == "has_alias":
            aliases.setdefault(t["head"], []).append(t["tail"])
    stub = type("Stub", (), {"aliases": aliases, "kg": kg})()
    reader, realizer = FrameReader(stub, lx), Realizer(lx)
    frames: list[Any] = []
    for a, spec in lx.acts.items():
        if "slot" in spec:
            frames += [F.make(a, **{spec["slot"]: v}) for v in lx.slots[spec["slot"]]["order"]]
        elif "slots" in spec:
            act_slot, opt_slot = spec["slots"]["action"], spec["slots"]["option"]
            cond = lx.slots[opt_slot].get("only_if") or {}
            for v in lx.slots[act_slot]["order"]:
                frames.append(F.make(a, **{act_slot: v}))
                if all(({act_slot: v}).get(k) == w for k, w in cond.items()):
                    frames += [F.make(a, **{act_slot: v, opt_slot: o}) for o in lx.slots[opt_slot]["order"]]
        elif "code" in spec:
            frames.append(F.make(a, **{spec["code"]["slot"]: "4821"}))
        elif isinstance(spec.get("say"), dict) and spec.get("cues"):
            frames.append(F.make(a))
    errs = []
    for f in frames:
        for lvl in lx["aliasing_levels"]:
            for d in lx["depths"]:
                try:
                    text = realizer.realize(f, {"entity_aliasing": lvl, "paraphrase_depth": d}, "lint")
                    got = reader.read(text)
                except (KeyError, IndexError, ValueError) as e:
                    errs.append(f"lexicon round trip: {f} at {lvl}/{d} fails: {type(e).__name__}: {e}")
                    continue
                if got != f:
                    errs.append(f"lexicon round trip: {f} at {lvl}/{d} -> {text!r} reads as {got}")
    return errs


@check
@patterns_only
def tools_match_graph(dom: Domain, kg) -> list[str]:
    """Lint: every graph tool has an entry (and only graph tools), each with a valid role and arguments."""
    tools = dom.section("tools")
    if not tools:
        return ["tools section missing"]
    errs = []
    graph_tools = {t["head"] for t in kg.triples if t["relation"] == "is_a" and t["tail"] == "tool"}
    for t in graph_tools - set(tools):
        errs.append(f"tools: graph tool {t} has no entry")
    for t, sp in tools.items():
        if t not in graph_tools:
            errs.append(f"tools.{t}: not a tool of the graph (is_a tool)")
        if sp.get("role") not in ("auth", "read", "write", "handoff"):
            errs.append(f"tools.{t}.role: {sp.get('role')!r} is not auth | read | write | handoff")
        if sp.get("behaviour") not in ("attempts", "list", "echo"):
            errs.append(f"tools.{t}.behaviour: {sp.get('behaviour')!r} is not attempts | list | echo")
        for arg, vset in (sp.get("arg_values") or {}).items():
            if arg not in (sp.get("args") or {}):
                errs.append(f"tools.{t}.arg_values: {arg} is not an argument of the tool")
            if vset not in _value_sets(kg):
                errs.append(f"tools.{t}.arg_values.{arg}: {vset!r} is not a value set of the graph")
        values = {x["tail"] for x in kg.triples if x["head"] == t and x["relation"] in ("success_value", "failure_value")}
        used = set((sp.get("status") or {}).values()) | set(sp.get("default_results") or [])
        if sp.get("check"):
            used |= {sp["check"]["success"], sp["check"]["else"]}
        for v in used - values:
            errs.append(f"tools.{t}: status {v!r} is not a success_value / failure_value of the graph")
    if sum(sp.get("role") == "auth" for sp in tools.values()) > 1:
        errs.append("tools: more than one auth tool")
    return errs


@check
@patterns_only
def figures_match_graph(dom: Domain, kg) -> list[str]:
    """Lint: every limit and named figure exists in the graph and has a value."""
    f = dom.section("figures")
    if not f:
        return ["figures section missing"]
    errs = []
    limits = dom.limit_keys()
    if not limits:
        errs.append(f"figures.limits: the graph has no is_a {f['limits'].get('class')!r} entities")
    valued = {t["head"] for t in kg.triples if t["relation"] == "has_value"}
    for k in limits:
        if k not in valued:
            errs.append(f"figures.limits: limit {k} has no has_value in the graph")
    rels = {t["relation"] for t in kg.triples}
    for r in (f.get("relations") or {}):
        if r not in rels:
            errs.append(f"figures.relations.{r}: not a relation of the graph")
    cats = {f["limits"]["enm_category"]} | {sp["enm_category"] for sp in (f.get("relations") or {}).values()}
    for name, sp in (f.get("named") or {}).items():
        if sp.get("enm_category") not in cats:
            errs.append(f"figures.named.{name}: unknown enm_category {sp.get('enm_category')!r}")
        elif sp["enm_category"] != f["limits"]["enm_category"]:
            rel = next((r for r, x in f["relations"].items() if x["enm_category"] == sp["enm_category"]), None)
            if rel and not any(t["head"] == sp.get("key") and t["relation"] == rel for t in kg.triples):
                errs.append(f"figures.named.{name}: the graph states no {rel} for {sp.get('key')!r}")
    return errs


@check
@patterns_only
def fixtures_match_graph(dom: Domain, kg) -> list[str]:
    """Lint: the mock bank's fixtures (card options, items, records) use values the graph knows."""
    fx = dom.section("fixtures")
    if not fx:
        return ["fixtures section missing"]
    errs = []
    sets = _value_sets(kg)
    for v in fx.get("card_options", []):
        if not any(v in m for m in sets.values()):
            errs.append(f"fixtures.card_options: {v} is not a member of any graph value set")
    items = (fx.get("items") or {})
    if len(items.get("ages", [])) > len(items.get("records", [])):
        errs.append("fixtures.items: more ages than records")
    return errs


@check
@patterns_only
def rules_match_graph(dom: Domain, kg) -> list[str]:
    """Lint: every rule entry names a graph rule, a known rule kind, and tools and limits that exist."""
    r = dom.section("rules")
    if not r:
        return ["rules section missing"]
    errs = []
    tools = dom.section("tools")
    graph_rules = {t["head"] for t in kg.triples if t["relation"] == "is_a" and t["tail"] == "rule"}
    limits = set(dom.limit_keys()) | set((dom.get("figures.named") or {}))
    known = {"no_call_after_handoff", "requires_auth", "max_calls", "item_scope", "value_from_caller",
             "terminal_value", "offered_only", "disclose_then_confirm", "handoff_when_requested", "turn_limit",
             "all_items_written", "terminal_value_handed_off", "claim_after_success", "forbidden_stance", "pii"}

    def rule_ok(rid, where):
        for part in str(rid).split("/"):
            if part not in graph_rules and part not in ("PII", "ENM"):
                errs.append(f"rules.{where}: {part} is not a rule of the graph")

    def walk(entries, where):
        for i, sp in enumerate(entries or []):
            w = f"{where}[{i}] {sp.get('check')}"
            if sp.get("check") not in known:
                errs.append(f"rules.{w}: unknown check template")
            for k in ("rule", "rule_missing", "rule_mismatch", "confirm_rule"):
                if sp.get(k):
                    rule_ok(sp[k], w)
            for k in ("tool", "source"):
                if sp.get(k) and sp[k] not in tools:
                    errs.append(f"rules.{w}: {k} {sp[k]!r} is not a tool")
            for lim in [sp.get("limit"), sp.get("figure"), (sp.get("window") or {}).get("limit"),
                        (sp.get("cap") or {}).get("limit") if isinstance(sp.get("cap"), dict) else sp.get("cap")]:
                if lim and lim not in limits:
                    errs.append(f"rules.{w}: {lim!r} is not a limit or named figure")
            for k in ("pattern", "unless"):
                if sp.get(k):
                    try:
                        re.compile(sp[k])
                    except re.error as e:
                        errs.append(f"rules.{w}.{k}: does not compile ({e})")
    walk(r.get("calls"), "calls")
    walk(r.get("completion"), "completion")
    walk(r.get("speech"), "speech")
    if not any(sp.get("role") == "handoff" for sp in tools.values()):
        errs.append("rules: the call checks need a handoff tool (tools.*.role: handoff)")
    return errs


@check
def no_boolean_keys(dom: Domain, kg) -> list[str]:
    """YAML reads unquoted yes / no / on / off as booleans: as a key or a level that is always a mistake."""
    errs = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, bool):
                    errs.append(f"{path}: key {k!r} was read as a boolean; quote it (\"yes\" / \"no\")")
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                if isinstance(v, bool) and path.startswith("journey"):
                    errs.append(f"{path}[{i}]: level {v!r} was read as a boolean; quote it")
                walk(v, f"{path}[{i}]")
    walk(dom.config, "")
    return [e.lstrip(".") for e in errs]


def _states(kg) -> set[str]:
    return {t["head"] for t in kg.triples if t["relation"] == "leads_to"} | \
           {t["tail"] for t in kg.triples if t["relation"] == "leads_to"}


@check
@patterns_only
def flow_matches_graph(dom: Domain, kg) -> list[str]:
    """Lint: every state, act, rule, tool and figure named in the flow section exists in the graph."""
    fl = dom.section("flow")
    if not fl:
        return ["flow section missing"]
    errs = []
    states, acts = _states(kg), _dialog_acts(kg)
    rules = {t["head"] for t in kg.triples if t["relation"] == "is_a" and t["tail"] == "rule"}
    tools = dom.section("tools")
    figures = set(dom.limit_keys()) | set(dom.get("figures.named") or {})
    roles = {t: sp.get("role") for t, sp in tools.items()}

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}"
                if k == "act" and isinstance(v, str) and v not in states and v not in acts:
                    errs.append(f"flow{p}: {v!r} is neither a dialog state nor a customer act of the graph")
                elif k in ("ask", "retry", "more_help", "end", "answer") and isinstance(v, str) and v not in states:
                    errs.append(f"flow{p}: {v!r} is not a dialog state of the graph")
                elif k in ("rule", "clear_rule", "correction_rule", "done_rule", "remaining_rule") and v not in rules:
                    errs.append(f"flow{p}: {v!r} is not a rule of the graph")
                elif k in ("rules", "success_rules", "echo_mismatch_rules"):
                    errs.extend(f"flow{p}: {r!r} is not a rule of the graph" for r in v if r not in rules)
                elif k == "tool" and v not in tools:
                    errs.append(f"flow{p}: {v!r} is not a tool")
                elif k in ("limit", "figure") and v not in figures:
                    errs.append(f"flow{p}: {v!r} is not a limit or named figure")
                walk(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk({k: v for k, v in fl.items() if k not in ("workflow", "stage_names", "stages")}, "")
    for pat, role in (("authenticate", "auth"), ("fetch", "read"), ("handoff", "handoff")):
        t = (fl.get(pat) or {}).get("tool")
        if t and roles.get(t) != role:
            errs.append(f"flow.{pat}.tool: {t} has role {roles.get(t)!r}, expected {role!r}")
    w = (fl.get("per_item") or {}).get("write", {}).get("tool")
    if w and roles.get(w) != "write":
        errs.append(f"flow.per_item.write.tool: {w} has role {roles.get(w)!r}, expected 'write'")
    names = set((fl.get("stage_names") or {}).values())
    if set(fl.get("stages", [])) != names:
        errs.append(f"flow.stages and flow.stage_names disagree: {sorted(set(fl.get('stages', [])) ^ names)}")
    wf = fl.get("workflow") or {}
    nodes = {t["head"] for t in kg.triples if t["relation"] == "enables"} | \
            {t["tail"] for t in kg.triples if t["relation"] == "enables"}
    for k in ("start", "summary", "end"):
        if wf.get(k) not in nodes:
            errs.append(f"flow.workflow.{k}: {wf.get(k)!r} is not a workflow node (enables) of the graph")
    return errs


@check
@patterns_only
def journey_is_consistent(dom: Domain, kg) -> list[str]:
    """Lint: every journey dimension has a default and its levels are consistent."""
    j = dom.section("journey")
    if not j:
        return ["journey section missing"]
    errs = []
    dims = j.get("dimensions") or {}
    for d in dims:
        if d not in (j.get("defaults") or {}):
            errs.append(f"journey.defaults: no default for {d}")
    for d, v in (j.get("defaults") or {}).items():
        if d not in dims:
            errs.append(f"journey.defaults: {d} is not a dimension")

    def cond(c, where):
        if not c:
            return
        for k, v in c.items():
            if k == "any":
                for x in v:
                    cond(x, where)
            elif k == "not":
                cond(v, where)
            elif k not in dims:
                errs.append(f"journey.{where}: {k!r} is not a dimension")
            else:
                levels = v["not"] if isinstance(v, dict) else v
                for lv in levels:
                    if lv not in dims[k] and lv != "N/A":
                        errs.append(f"journey.{where}: {k}={lv!r} is not a level")
    for i, c in enumerate(j.get("invalid", [])):
        cond(c, f"invalid[{i}]")
    for i, st in enumerate(j.get("relevance", [])):
        cond(st.get("if"), f"relevance[{i}].if")
        cond(st.get("stop"), f"relevance[{i}].stop")
        for d in st.get("add", []):
            if d not in dims:
                errs.append(f"journey.relevance[{i}].add: {d!r} is not a dimension")
    added = {d for st in j.get("relevance", []) for d in st.get("add", [])}
    for d in set(dims) - added:
        errs.append(f"journey.relevance: dimension {d} is never relevant")
    for key in ("categories", "tags"):
        for i, c in enumerate(j.get(key, [])):
            cond(c.get("when"), f"{key}[{i}]")
    if j.get("categories") and j["categories"][-1].get("when"):
        errs.append("journey.categories: the last category needs no 'when' (the fallback)")
    for b in j.get("boundaries", []):
        d, _, lv = b.partition("=")
        if d not in dims or not any(str(x) == lv for x in dims[d]):
            errs.append(f"journey.boundaries: {b!r} is not a dimension=level")
    items = (dom.get("fixtures.items") or {})
    count_dim = j.get("item_count")
    if count_dim in dims and max(dims[count_dim]) > len(items.get("ages", [])):
        errs.append(f"fixtures.items: {count_dim} goes up to {max(dims[count_dim])} but only "
                    f"{len(items.get('ages', []))} ages are given")
    for st in j.get("fixture", []):
        if "dimension" in st:
            for lv in st["levels"]:
                if lv not in dims.get(st["dimension"], []):
                    errs.append(f"journey.fixture: {st['dimension']}={lv!r} is not a level")
    acts = _dialog_acts(kg)
    for i, f in enumerate(j.get("padding", [])):
        if f.get("act") not in acts:
            errs.append(f"journey.padding[{i}]: {f.get('act')!r} is not a customer act")
    return errs


if __name__ == "__main__":
    raise SystemExit(main())
