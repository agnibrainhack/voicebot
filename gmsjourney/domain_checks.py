"""Cross-checks between a domain pack's config and its policy graph (run by ``domain.lint``).

Each check returns a list of problems; an empty list means the section agrees with the graph.
"""
from __future__ import annotations

import re
from typing import Any

from .domain import Domain

CHECKS: list = []           # every section check, run in order by domain.lint


def check(fn):
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
    from .enrich.lexicon import Lexicon
    from .enrich.reader import FrameReader
    from .enrich.realizers import Realizer
    from .ontology import SemanticFrame as F

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
