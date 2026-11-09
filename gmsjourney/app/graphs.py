"""Open a policy graph by path, show it section by section, check its core segments and save edited versions.

* ``open_graph(path)``  a graph file inside a journey folder opens that journey (switching the journey to that
  graph file if it is another version); a compact graph from anywhere else becomes a new journey: its folder
  under gmsjourney/policy/, a drafted customer lexicon (lexicon_draft), the enrichment catalog copied from a
  template journey and a drafted project.
* ``sections(dom)``     every triple of the graph the engine reads (the compiled graph, so both formats look the
  same) in one of seven sections: used by the customer, by the bot, by both, defining the flow, rules,
  knowledge and figures, other.
* ``core_segments(dom)`` what every graph needs for the validations a campaign runs, with what is missing and a
  suggested addition.
* ``editable(dom)`` / ``save_version(pack, ...)`` the graph's own entries (compact: one JSON entry per value set,
  act, tool, state, rule, limit; triples: one row per triple), and a new ``<name>.vN.json`` beside the original
  with what changed. The original file is never written.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .runs import ROOT

POLICY_ROOT = ROOT / "gmsjourney" / "policy"
DEFAULT_GRAPH = "gmsjourney/policy/charge_review/charge_review_kg.json"
TEMPLATE_PACK = "supplementary_card"           # the enrichment catalog a new journey starts from

SECTIONS = [
    ("customer", "Used by the customer",
     "What a caller can say: the acts the bot listens for, the values they name and the words for them. "
     "Every customer turn is worded from this, and read back from it."),
    ("bot", "Used by the bot",
     "What the bot says and does: its prompts and lines, the tools it calls, their arguments, results and "
     "statuses. The tool-contract, speech and claim checks read this."),
    ("both", "Used by both",
     "Where the two meet: the acts each bot state accepts, how a bare yes or no is read there, and the value "
     "sets that pass from the customer's words into tool arguments."),
    ("flow", "Defines the flow",
     "How the call moves: transitions, stages, handoff triggers and the states they fire in. Journeys, "
     "coverage and the flow checks are walked from this."),
    ("rules", "Rules",
     "What the bot must and must not do, where each rule applies and what justifies each handoff. Every "
     "rule-break check and mutant comes from here."),
    ("knowledge", "Knowledge and figures",
     "Limits, fees, durations and the policy text the bot answers questions from. The figure checks and the "
     "policy-question cases come from here."),
    ("other", "Other", "Domain context and anything the sections above do not claim."),
]
_TITLES = {k: t for k, t, _ in SECTIONS}

_REL = {
    "customer": {"has_alias", "spoken_as", "member_of", "has_slot"},
    "bot": {"says", "prompts_customer", "spoken_cue", "calls_tool", "has_arg", "returns_field", "success_value",
            "failure_value", "on_failure", "echoes", "has_role", "claims_tool", "is_outcome", "carries", "when_used"},
    "both": {"accepts_act", "interprets_yes_as", "interprets_no_as"},
    "flow": {"leads_to", "enables", "precedes", "precedes_two", "in_stage", "starts_dialog", "fires_in_state",
             "transfer_when", "transfers_on", "limited_by", "has_condition", "has_stage"},
    "rules": {"has_stance", "constrains", "applies_rule", "justified_by", "asserted_by", "denied_by"},
    "knowledge": {"has_value", "has_unit", "states", "requires_fee_confirmation", "delivered_by", "has_allowed"},
}
_BY_REL = {r: s for s, rels in _REL.items() for r in rels}
_ASK_FIELDS = {"intent", "attr", "entity", "key", "corrected_from"}


# ---------------------------------------------------------------------- paths and journeys
def resolve(path: str) -> Path:
    p = Path(str(path or DEFAULT_GRAPH).strip().strip('"').strip("'")).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _is_v2(p: Path) -> bool:
    from ..graph.model import is_v2
    return is_v2(p)


def _pack_graph(pack_dir: Path) -> str | None:
    import yaml
    cfg = yaml.safe_load((pack_dir / "domain.yaml").read_text()) or {}
    return (cfg.get("domain") or {}).get("graph")


def set_graph(pack_dir: Path, filename: str) -> bool:
    """Point the journey at another graph file in its folder (only domain.yaml's ``graph:`` line changes)."""
    cfg = pack_dir / "domain.yaml"
    text = cfg.read_text()
    m = re.search(r"(?m)^domain:\s*\n((?:[ \t]+.*\n|\s*\n)*)", text)
    if not m or not re.search(r"(?m)^[ \t]+graph:", m.group(1)):
        raise ValueError(f"{rel(cfg)} has no domain.graph line to switch")
    block = re.sub(r"(?m)^([ \t]+graph:)[ \t]*\S.*$", lambda mm: f"{mm.group(1)} {filename}", m.group(1), count=1)
    new = text[:m.start(1)] + block + text[m.end(1):]
    if new != text:
        cfg.write_text(new)
        return True
    return False


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s or "journey"


def _unique(slug: str) -> str:
    out, n = slug, 2
    while (POLICY_ROOT / out).exists():
        out, n = f"{slug}_{n}", n + 1
    return out


def new_journey(src: Path, name: str | None = None) -> dict[str, Any]:
    """A journey folder for a compact graph from outside the journey folders."""
    import yaml
    from ..graph.model import PolicyGraph
    from ..lexicon_draft import draft_bot_says, draft_lexicon
    g = PolicyGraph.load(src)
    slug = _unique(_slug(name or g.data.get("id") or src.stem))
    d = POLICY_ROOT / slug
    d.mkdir(parents=True)
    try:
        gname = src.name if src.name.endswith(".json") else src.stem + ".json"
        shutil.copy(src, d / gname)
        title = g.data.get("title") or slug.replace("_", " ").title()
        head = (f"# {title}\n#\n# Made by the journey console from {src.name}. The policy graph states the policy and the "
                "dialog; every section a\n# campaign reads (tools, figures, rules, flow, journeys, checks) is mined from "
                "it. The lexicon (how customers\n# word things) and bot.says (how the reference bot words its lines) below are first drafts from "
                "the\n# graph (gmsjourney/lexicon_draft.py), checked by playing the project's cases: replace their plain "
                "wording\n# with the way your customers and your bot speak.\n\n")
        body = {"domain": {"id": slug, "graph": gname, "title": title}, "lexicon": draft_lexicon(g),
                "bot": {"says": draft_bot_says(g)}}
        (d / "domain.yaml").write_text(head + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=110))
        tpl = POLICY_ROOT / TEMPLATE_PACK
        app = (tpl / "application.yaml").read_text()
        app = re.sub(r"(?m)^(application:\s*).*$", lambda m: m.group(1) + slug, app, count=1)
        (d / "application.yaml").write_text(app)
        shutil.copy(tpl / "custom_factors.yaml", d / "custom_factors.yaml")
        from ..domain import load
        from ..project import dump_yaml
        from ..project_draft import HEADER, draft
        dom = load(slug)                         # not made active: switching runs hooks that read the project
        raw = draft(dom)
        dom.project_path.write_text(HEADER.format(name=raw["project"]["name"], path=dom.project_path, pack=slug)
                                    + "\n" + dump_yaml(raw))
        from .server import _on
        with _on(slug) as dom:
            from ..domain import lint
            problems = lint(dom)
            wording = check_wording()
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return {"pack": slug, "folder": rel(d), "graph": rel(d / gname), "problems": problems[:20], "wording": wording}


def check_wording(max_cases: int = 60) -> dict[str, Any]:
    """Play the active journey's project cases with its reference bot and the speech check: how many turns, how
    many the graph agrees with, and every bot line whose words fail the check (with an example). A drafted
    journey is checked this way when it is made."""
    from . import playground as PG
    PG.reset()
    cases = PG.cases("project", None)["cases"][:max_cases]
    turns = flow = 0
    bad: dict[str, dict] = {}
    for c in cases:
        sess = PG.get(PG.start("project", c["id"], None)["id"])
        ts = list(sess.turns)
        for ln in c["lines"]:
            try:
                ts.append(sess.say(ln["text"]))
            except ValueError:
                break
        for t in ts:
            turns += 1
            flow += not t["match"]
            sp = t.get("speech")
            if sp and not sp["ok"]:
                why = [f["detail"] for f in sp["findings"] if not f["advisory"]] or ["a claim did not check out"]
                for a in t["bot"]["act_names"]:
                    bad.setdefault(a, {"act": a, "turns": 0, "example": t["bot"]["text"], "why": why[0]})["turns"] += 1
    return {"cases": len(cases), "turns": turns, "off_flow": flow, "speech_failures": sorted(bad.values(), key=lambda x: -x["turns"])}


def open_graph(path: str) -> dict[str, Any]:
    """Open ``path``: the journey that holds it (switched to that file), or a new journey made from it."""
    p = resolve(path)
    if not p.exists():
        raise ValueError(f"no file at {path}")
    if p.suffix.lower() != ".json":
        raise ValueError("a policy graph is a .json file (a compact gmsjourney.policy_graph/2 graph, or a triples graph)")
    try:
        data = json.loads(p.read_text())
    except ValueError as e:
        raise ValueError(f"{p.name} is not valid JSON: {e}") from None
    pack_dir = p.parent
    inside = pack_dir.parent == POLICY_ROOT and (pack_dir / "domain.yaml").exists()
    if inside:
        switched = False
        if _pack_graph(pack_dir) != p.name:
            v2_now = _is_v2(pack_dir / (_pack_graph(pack_dir) or ""))
            if v2_now != _is_v2(p):
                raise ValueError(f"{p.name} is in a different graph format from the journey's current graph; open it "
                                 "from its own folder instead")
            switched = set_graph(pack_dir, p.name)
        return {"pack": pack_dir.name, "graph": rel(p), "created": False, "switched": switched}
    if _is_v2(p):
        out = new_journey(p)
        return {**out, "created": True, "switched": False}
    if isinstance(data, dict) and isinstance(data.get("triples"), list):
        raise ValueError(f"{p.name} is a triples graph (the format built from a PDF). A triples graph needs a full "
                         "domain.yaml written for it, so it can only be opened from a journey folder that has one; a "
                         "compact graph (gmsjourney.policy_graph/2) can be opened from anywhere and becomes a new journey.")
    raise ValueError(f"{p.name} is not a policy graph (no schema gmsjourney.policy_graph/2 and no triples)")


def versions(dom) -> list[dict[str, Any]]:
    """The graph files in the journey's folder that belong to the same graph (the original and its versions)."""
    cur = dom.graph_path.name
    base = re.sub(r"\.v\d+$", "", Path(cur).stem)
    out = []
    for f in sorted(dom.dir.glob(f"{base}*.json")):
        if f.stem != base and not re.fullmatch(re.escape(base) + r"\.v\d+", f.stem):
            continue
        out.append({"file": f.name, "path": rel(f), "active": f.name == cur, "original": f.stem == base})
    out.sort(key=lambda v: int(re.sub(r"\D", "", v["file"].rsplit(".v", 1)[-1]) or 0) if not v["original"] else -1)
    return out


# ---------------------------------------------------------------------- sections
def _types(kg) -> dict[str, str]:
    ty: dict[str, str] = {}
    for t in kg.where("is_a"):
        ty.setdefault(t["head"], t["tail"])
    for t in kg.where("member_of"):
        ty.setdefault(t["head"], "value")
        ty.setdefault(t["tail"], "value_set")
    return ty


def shared_slots(kg) -> set[str]:
    """Slot names a customer act fills and a tool takes as an argument."""
    slots = {t["tail"] for t in kg.where("has_slot")} - _ASK_FIELDS
    args = {t["tail"] for t in kg.where("has_arg")}
    return slots & args


def classify(kg) -> dict[str, list[dict]]:
    ty = _types(kg)
    shared = shared_slots(kg)
    member_set = {t["head"]: t["tail"] for t in kg.where("member_of")}
    out: dict[str, list[dict]] = {k: [] for k, _, _ in SECTIONS}
    for t in kg.triples:
        r, h = t["relation"], t["head"]
        s = _BY_REL.get(r)
        if r in ("member_of", "has_alias", "spoken_as") and (member_set.get(h) in shared or t["tail"] in shared):
            s = "both"
        elif r in ("has_slot", "has_arg") and t["tail"] in shared:
            s = "both"
        elif r == "is_a":
            s = {"dialog_act": "customer", "dialog_state": "flow", "tool": "bot", "tool_step": "bot",
                 "rule": "rules", "limit": "knowledge", "escalation_trigger": "flow",
                 "delivery_option": "knowledge", "value_set": "customer"}.get(t["tail"], "other")
        elif s is None:
            hty = ty.get(h)
            if r == "has_text":
                s = {"rule": "rules", "limit": "knowledge", "tool": "bot", "dialog_state": "bot"}.get(hty, "knowledge")
            elif h == "bot" or t.get("kind") == "policy" and hty == "rule":
                s = "rules"
            elif t.get("tail_type") == "number" or re.match(r"has_(.*_)?(usd|days|fee|age|max|min)", r):
                s = "knowledge"
            elif hty in ("tool", "tool_step"):
                s = "bot"
            elif hty == "dialog_state":
                s = "flow"
            else:
                s = "other"
        out[s].append(t)
    return out


def sections(dom, cap: int = 600) -> dict[str, Any]:
    kg = dom.kg
    ty = _types(kg)
    groups = classify(kg)
    shared = sorted(shared_slots(kg))
    out = []
    for key, title, about in SECTIONS:
        ts = groups[key]
        ents: dict[str, str] = {}
        for t in ts:
            ents.setdefault(t["head"], ty.get(t["head"], ""))
            if t.get("tail_type", "entity") == "entity" and isinstance(t["tail"], str) and t["tail"] in ty:
                ents.setdefault(t["tail"], ty.get(t["tail"], ""))
        if key == "both":
            for s in shared:
                ents.setdefault(s, "shared slot")
        out.append({"key": key, "title": title, "about": about, "count": len(ts),
                    "relations": dict(Counter(t["relation"] for t in ts).most_common()),
                    "entities": [{"name": n, "type": v} for n, v in sorted(ents.items(), key=lambda x: (x[1], x[0]))][:200],
                    "triples": [{"id": t.get("id"), "head": t["head"], "relation": t["relation"],
                                 "tail": t["tail"] if not isinstance(t["tail"], (dict, list)) else json.dumps(t["tail"]),
                                 "kind": t.get("kind")} for t in ts[:cap]],
                    "more": max(0, len(ts) - cap)})
    return {"pack": dom.dir.name, "graph": rel(dom.graph_path), "format": "compact" if dom.graph is not None else "triples",
            "title": (dom.graph.title if dom.graph is not None else kg.data.get("title")) or dom.dir.name,
            "triples": len(kg.triples), "shared_slots": shared, "sections": out, "versions": versions(dom)}


# ---------------------------------------------------------------------- core segments
def _seg(id_, name, needed, ok, detail, add=None, triples=None):
    return {"id": id_, "name": name, "needed_for": needed, "ok": bool(ok), "detail": detail,
            "add": add or {}, "triples": triples or []}


def core_segments(dom) -> list[dict[str, Any]]:
    """What every policy graph needs for the validations a campaign runs, and what is missing."""
    kg = dom.kg
    g = dom.graph
    ty = _types(kg)
    acts = {h for h, t in ty.items() if t == "dialog_act"}
    states = set(kg.states)
    tools = {h for h, t in ty.items() if t == "tool"}
    limits = {h for h, t in ty.items() if t == "limit"}
    rules = {h for h, t in ty.items() if t == "rule"}
    slots = defaultdict(set)
    for t in kg.where("has_slot"):
        slots[t["head"]].add(t["tail"])
    args = defaultdict(set)
    for t in kg.where("has_arg"):
        args[t["head"]].add(t["tail"])
    says = {t["head"] for t in kg.where("says")} | {t["head"] for t in kg.where("spoken_cue")}
    accepts = defaultdict(set)
    for t in kg.where("accepts_act"):
        accepts[t["head"]].add(t["tail"])
    nexts = defaultdict(set)
    for t in kg.where("leads_to"):
        nexts[t["head"]].add(t["tail"])
    start = next((t["head"] for t in kg.where("starts_dialog")), "OPENING" if "OPENING" in states else None)
    out = []

    out.append(_seg("opening", "Opening statement", "the opening-statement rule and every journey's first turn",
                    start and start in says, f"start state: {start or 'none'}" + ("" if start in says else " (says nothing)"),
                    {"opening": "This call may be recorded. How can I help you today?",
                     "states.OPENING": {"stage": "opening", "say": {"text": "$opening"}, "go": "ASK_AUTH"}},
                    [{"head": "OPENING", "relation": "says", "tail": "opening"}]))

    cred_acts = [a for a, s in slots.items() if any(x.startswith("credential") for x in s)]
    auth_tools = [t for t, a in args.items() if any(x.startswith("credential") for x in a)]
    out.append(_seg("auth", "Caller verification", "the no-account-tool-before-verification rule, wrong-code journeys",
                    cred_acts and auth_tools,
                    f"credential act: {', '.join(cred_acts) or 'missing'}; verification tool: {', '.join(auth_tools) or 'missing'}",
                    {"acts.PROVIDE_CREDENTIAL": {"credential": "credential_code"},
                     "tools.authenticate_customer": {"stage": "authentication", "role": "auth", "args": {"credential": "credential_code"},
                                                     "ok": "AUTHENTICATED", "fail": ["FAILED"],
                                                     "check": {"arg": "credential", "fixture": "credential"},
                                                     "text": "Verifies the caller."}},
                    [{"head": "PROVIDE_CREDENTIAL", "relation": "has_slot", "tail": "credential"},
                     {"head": "authenticate_customer", "relation": "has_arg", "tail": "credential"}]))

    auth_lim = [x for x in limits if "auth" in x]
    out.append(_seg("auth_limit", "Verification attempt limit", "retry journeys and the limit-plus-one mutant",
                    auth_lim, ", ".join(auth_lim) or "no limit on verification attempts",
                    {"limits.max_auth_attempts": {"value": 2, "unit": "attempts", "text": "At most 2 authentication attempts; then transfer."}},
                    [{"head": "max_auth_attempts", "relation": "is_a", "tail": "limit"},
                     {"head": "max_auth_attempts", "relation": "has_value", "tail": 2}]))

    value_acts = [a for a, s in slots.items() if s - _ASK_FIELDS - {x for x in s if x.startswith("credential")}]
    alias = defaultdict(int)
    for t in kg.where("has_alias"):
        alias[t["head"]] += 1
    said = {x for a in value_acts for x in slots[a]} - _ASK_FIELDS
    members = [t["head"] for t in kg.where("member_of") if t["tail"] in said]
    bare = [m for m in members if not alias.get(m)]
    out.append(_seg("customer_acts", "Customer acts with values", "case generation: every value of every act is a case",
                    value_acts and not bare,
                    f"{len(value_acts)} act(s) carry values; " + (f"values with no aliases: {', '.join(bare[:8])}" if bare else
                                                                   f"all {len(members)} values have aliases"),
                    {"values.<set>": {"<VALUE>": {"aliases": ["how a customer says it", "another way"]}},
                     "acts.<INFORM_X>": {"<slot>": "<set>"}},
                    [{"head": "<VALUE>", "relation": "has_alias", "tail": "how a customer says it"}]))

    asks = [s for s in states if accepts.get(s)]
    unknown = sorted({a for s in asks for a in accepts[s]} - acts)
    out.append(_seg("prompts", "Bot prompts that accept acts", "the flow walk: each prompt is where a customer turn lands",
                    asks and not unknown,
                    f"{len(asks)} prompt state(s)" + (f"; accept unknown acts: {', '.join(unknown)}" if unknown else ""),
                    {"states.ASK_X": {"stage": "<stage>", "on": {"INFORM_X": "NEXT_STATE"}, "rules": []}}))

    no_status = [t for t in tools if not kg.where("success_value", head=t)
                 or (not kg.where("failure_value", head=t) and "transfer" not in t and "handoff" not in t)]
    out.append(_seg("tool_status", "Tools with success and failure results",
                    "the tool-contract checks, tool-failure journeys and the status-as-success mutant",
                    tools and not no_status,
                    f"{len(tools)} tool(s)" + (f"; missing a success or failure value: {', '.join(no_status)}" if no_status else ""),
                    {"tools.<tool>": {"ok": "SUCCESS", "fail": ["FAILED"]}},
                    [{"head": "<tool>", "relation": "success_value", "tail": "SUCCESS"},
                     {"head": "<tool>", "relation": "failure_value", "tail": "FAILED"}]))

    confirm = [s for s in asks if "CONFIRM" in accepts[s]]
    out.append(_seg("confirm", "Confirmation before an action", "the confirm-before-acting rule and skipped-confirmation mutants",
                    confirm, ", ".join(sorted(confirm)) or "no prompt accepts CONFIRM",
                    {"states.CONFIRM_X": {"stage": "<stage>", "on": {"CONFIRM": "DO_X", "DENY": "ACK_DECLINED"}}}))

    echo = kg.where("echoes")
    mismatch = kg.where("treat_echo_mismatch_as")
    out.append(_seg("echo", "Echo check on a write", "the ignored-echo mutant (the bot trusting a result that does not match)",
                    echo or mismatch, (", ".join(sorted({t['head'] for t in echo})) + " echoes what it wrote") if echo else
                    ("a rule treats an echo mismatch as " + str(mismatch[0]["tail"])) if mismatch else "no tool echoes what it wrote",
                    {"tools.<write tool>": {"echo": "<argument it must return unchanged>"}},
                    [{"head": "<write tool>", "relation": "echoes", "tail": "<argument>"}]))

    trig = sorted(h for h, t in ty.items() if t == "escalation_trigger")
    unjust = [x for x in trig if not kg.where("justified_by", head=x)]
    out.append(_seg("transfer", "Handoff with justified reasons", "handoff journeys and the transfer-reason check",
                    trig and not unjust,
                    f"{len(trig)} handoff reason(s)" + (f"; with no rule behind them: {', '.join(unjust)}" if unjust else ""),
                    {"transfers": {"state": "TRANSFER", "tool": "transfer_to_human", "arg": "reason", "then": "END",
                                   "reasons": {"<reason>": "<rule id>"}}},
                    [{"head": "<reason>", "relation": "justified_by", "tail": "<rule id>"}]))

    human = "REQUEST_HUMAN" in acts or bool(g is not None and (g.glob.get("human") or {}).get("act"))
    out.append(_seg("human", "Customer asks for a person", "the handoff-on-request rule and its journeys", human,
                    "REQUEST_HUMAN is accepted everywhere" if human else "no act for asking for a person",
                    {"global.human": {"act": "REQUEST_HUMAN", "then": "!customer_requested_human"}}))

    clar = [x for x in limits if "clarif" in x or "unsure" in x]
    unclear = "UNCLEAR" in acts
    out.append(_seg("unclear", "Unclear answers and a retry limit", "noise and ambiguity cases, the ambiguity handoff",
                    unclear and clar, ("UNCLEAR act; " if unclear else "no UNCLEAR act; ") + (", ".join(clar) or "no clarification limit"),
                    {"limits.max_clarification_retries": {"value": 1, "unit": "retries", "text": "One clarification, then transfer.", "internal": True},
                     "global.unclear": {"act": "UNCLEAR", "clarify": "ASK_CLARIFY", "limit": "max_clarification_retries",
                                        "rule": "<rule id>", "else": "!ambiguity_persists"}}))

    turns = [x for x in limits if "turn" in x]
    out.append(_seg("turn_limit", "Turn limit", "long-conversation cases and the turn-limit handoff", turns,
                    ", ".join(turns) or "no limit on the number of turns",
                    {"limits.max_turns": {"value": 20, "unit": "turns", "text": "Transfer after 20 customer turns.", "internal": True},
                     "global.turn_limit": {"limit": "max_turns", "rule": "<rule id>", "else": "!turn_limit", "wrap_up": "END"}}))

    kb = kg.where("states")
    q = "ASK_KB" in acts
    out.append(_seg("questions", "Policy questions and their answers", "policy-question cases and the knowledge-truth check",
                    q and kb, f"{len(kb)} policy sentence(s)" + ("" if q else "; no ASK_KB act"),
                    {"knowledge": ["<a sentence of policy the bot may quote>"],
                     "global.question": {"act": "ASK_KB", "answer": "ANSWER_KB", "rule": "<rule id>"}},
                    [{"head": "knowledge_base", "relation": "states", "tail": "<policy sentence>"}]))

    stance = {t["head"] for t in kg.where("has_stance")}
    applied = {t["tail"] for t in kg.where("applies_rule")} | {t["tail"] for t in kg.where("justified_by")}
    applied |= {str(t.get("source", "")).split(".", 1)[1] for t in kg.triples                # a rule's own check
                if t["head"] == "bot" and str(t.get("source", "")).startswith("rules.")}
    if g is not None:
        applied |= {v.get("rule") for v in g.glob.values() if isinstance(v, dict) and v.get("rule")}
    idle = sorted(rules - applied)
    out.append(_seg("rules", "Rules with a stance, each checked somewhere", "every rule-break check and the rule mutants",
                    rules and rules <= stance and not idle,
                    f"{len(rules)} rule(s)" + (f"; checked nowhere: {', '.join(idle[:10])}" if idle else "")
                    + (f"; no must / must_not: {', '.join(sorted(rules - stance)[:10])}" if rules - stance else ""),
                    {"rules.R<n>": ["must", "What the bot must do.", {"states": ["bot", "<relation>", "<object>"]}]}))

    unvalued = sorted(x for x in limits if not kg.where("has_value", head=x))
    unit = sorted(x for x in limits if not kg.where("has_unit", head=x))
    out.append(_seg("figures", "Figures with values and units", "the figure checks (exact numbers the bot quotes)",
                    limits and not unvalued and not unit,
                    f"{len(limits)} limit(s)" + (f"; no value: {', '.join(unvalued)}" if unvalued else "")
                    + (f"; no unit: {', '.join(unit)}" if unit else ""),
                    {"limits.<name>": {"value": 0, "unit": "<unit>", "text": "<the sentence it comes from>"}}))

    reach, todo = {start} if start else set(), [start] if start else []
    while todo:
        for n in nexts.get(todo.pop(), ()):
            if n not in reach:
                reach.add(n)
                todo.append(n)
    lost = sorted(states - reach - {"ANSWER_KB", "ASK_CLARIFY", "TRANSFER"})
    out.append(_seg("reach", "Every state reachable, the call can end", "journey enumeration and coverage",
                    "END" in states and "END" in reach and not lost,
                    ("END reachable" if "END" in reach else "END not reachable") + (f"; unreachable: {', '.join(lost)}" if lost else ""),
                    {"states.END": {"stage": "closing", "say": {}}}))

    worded = says | set(((dom.section("bot") or {}).get("says") or {}))
    if g is not None:                                     # a compact graph: a prompt or line of the bot
        speakers = {n for n, s in g.states.items() if not s.get("call") and (s.get("on") or s.get("say") is not None)}
    else:
        speakers = states if kg.where("says") else set()
    speak = sorted(speakers - worded - {"ANSWER_KB"})
    drafted = sorted(x.split(".", 2)[2] for x in (getattr(dom, "lexicon_drafted", None) or []) if x.startswith("bot.says."))
    if speak:
        detail = f"no wording for: {', '.join(speak[:10])}"
    elif drafted:
        detail = (f"drafted wording for {len(drafted)} state(s) ({', '.join(drafted[:6])}{', ...' if len(drafted) > 6 else ''}); "
                  "write your own under bot.says in domain.yaml when you can")
    else:
        detail = "every speaking state has its wording"
    seg = _seg("speech", "What each state says", "how the reference bot words its lines (the speech check reads what "
               "each state must say from the graph, so it runs either way)", not speak, detail,
               {}, [{"head": "<STATE>", "relation": "says", "tail": "<field>"}])
    seg["advisory"] = g is not None                       # compact graphs: wording only, the checks need none
    seg["drafted"] = drafted
    out.append(seg)

    from ..domain_checks import lexicon_matches_graph
    try:
        lx = lexicon_matches_graph(dom, kg)
    except Exception as e:                                  # noqa: BLE001
        lx = [f"{type(e).__name__}: {e}"]
    out.append(_seg("lexicon", "Customer wording for every value and act", "case wording and the read-back proof",
                    not lx, "; ".join(lx[:4]) or "every value and act is worded",
                    {}, []))
    return out


# ---------------------------------------------------------------------- editing
def _compact_section(key: str, child: str | None, g) -> str:
    if key == "values":
        shared = set()
        acts = {sp["set"] for a in g.acts for sp in g.slot_specs(a).values()}
        tools = {vs for t in g.tools.values() for vs in (t.get("args") or {}).values()}
        shared = acts & tools
        return "both" if child in shared else "customer" if child in acts else "bot"
    return {"acts": "customer", "tools": "bot", "opening": "bot", "credential_hint": "bot",
            "states": "flow", "transfers": "flow", "global": "flow", "rules": "rules",
            "knowledge": "knowledge", "limits": "knowledge", "facts": "knowledge"}.get(key, "other")


_SPLIT = ("values", "acts", "tools", "states", "rules", "limits", "facts", "global")


def compact_items(data: dict, g) -> list[dict[str, Any]]:
    items = []
    for k, v in data.items():
        if k in _SPLIT and isinstance(v, dict):
            for c, cv in v.items():
                items.append({"path": f"{k}.{c}", "section": _compact_section(k, c, g), "value": cv})
        else:
            items.append({"path": k, "section": _compact_section(k, None, g), "value": v})
    return items


def editable(dom) -> dict[str, Any]:
    p = dom.graph_path
    data = json.loads(p.read_text())
    base = {"pack": dom.dir.name, "file": p.name, "path": rel(p), "versions": versions(dom),
            "sections": [{"key": k, "title": t} for k, t, _ in SECTIONS]}
    if dom.graph is not None:
        return {**base, "format": "compact", "items": compact_items(data, dom.graph)}
    groups = classify(dom.kg)
    sec = {t["id"]: s for s, ts in groups.items() for t in ts}
    rows = [{**t, "section": sec.get(t["id"], "other")} for t in data["triples"]]
    return {**base, "format": "triples", "triples": rows,
            "sources": sorted((data.get("source") or {}).get("sections", {}))}


def apply_items(items: list[dict]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for it in items:
        path, v = str(it["path"]), it.get("value")
        if "." in path and path.split(".", 1)[0] in _SPLIT:
            k, c = path.split(".", 1)
            data.setdefault(k, {})[c] = v
        else:
            data[path] = v
    return data


def _next_version(dom) -> Path:
    base = re.sub(r"\.v\d+$", "", dom.graph_path.stem)
    n = 2
    while (dom.dir / f"{base}.v{n}.json").exists():
        n += 1
    return dom.dir / f"{base}.v{n}.json"


def _triples_doc(orig: dict, rows: list[dict], note: str) -> dict[str, Any]:
    from ..knowledge.policy_kg import _content_hash
    doc = copy.deepcopy(orig)
    secs = doc.setdefault("source", {}).setdefault("sections", {})
    used = 0
    clean = []
    seen = set()
    n = max([int(re.sub(r"\D", "", str(t.get("id") or "")) or 0) for t in orig.get("triples", [])] + [0])
    for r in rows:
        t = {k: v for k, v in r.items() if k != "section"}
        if not t.get("head") or not t.get("relation"):
            continue
        if not t.get("id") or t["id"] in seen:
            n += 1
            t["id"] = f"T{n:04d}"
        seen.add(t["id"])
        t.setdefault("tail_type", "number" if isinstance(t.get("tail"), (int, float)) else "entity")
        t.setdefault("kind", "policy")
        if not t.get("source") or t["source"] not in secs:
            t["source"] = "user_edit"
            used += 1
        clean.append(t)
    if used:
        secs.setdefault("user_edit", note or "Entries added in the journey console.")
    doc["triples"] = clean
    doc.pop("sha256", None)
    doc["sha256"] = _content_hash(doc)
    return doc


def diff(old: dict, new: dict, fmt: str) -> list[dict[str, Any]]:
    out = []
    if fmt == "triples":
        a = {t["id"]: t for t in old.get("triples", [])}
        b = {t["id"]: t for t in new.get("triples", [])}
        key = lambda t: (t["head"], t["relation"], json.dumps(t["tail"], sort_keys=True))
        for i in b.keys() - a.keys():
            out.append({"change": "added", "path": i, "after": f"{b[i]['head']} {b[i]['relation']} {b[i]['tail']}"})
        for i in a.keys() - b.keys():
            out.append({"change": "removed", "path": i, "before": f"{a[i]['head']} {a[i]['relation']} {a[i]['tail']}"})
        for i in a.keys() & b.keys():
            if key(a[i]) != key(b[i]):
                out.append({"change": "changed", "path": i, "before": f"{a[i]['head']} {a[i]['relation']} {a[i]['tail']}",
                            "after": f"{b[i]['head']} {b[i]['relation']} {b[i]['tail']}"})
        return sorted(out, key=lambda x: x["path"])
    def walk(v, path, acc, depth):
        if isinstance(v, dict) and v and depth < 3:
            for k, x in v.items():
                walk(x, f"{path}.{k}" if path else str(k), acc, depth + 1)
        else:
            acc[path] = json.dumps(v, sort_keys=True, ensure_ascii=False)
        return acc
    flat = lambda d: walk(d, "", {}, 0)
    fa, fb = flat(old), flat(new)
    for p in sorted(fb.keys() - fa.keys()):
        out.append({"change": "added", "path": p, "after": fb[p]})
    for p in sorted(fa.keys() - fb.keys()):
        out.append({"change": "removed", "path": p, "before": fa[p]})
    for p in sorted(fa.keys() & fb.keys()):
        if fa[p] != fb[p]:
            out.append({"change": "changed", "path": p, "before": fa[p], "after": fb[p]})
    return out


def check_candidate(dom, doc: dict) -> dict[str, Any]:
    """Load ``doc`` as the journey's graph in a scratch copy of the journey folder and lint it there."""
    from ..domain import current, lint, load, set_current
    tmp = Path(tempfile.mkdtemp(prefix="gms_graph_"))
    try:
        d = tmp / dom.dir.name
        d.mkdir()
        for f in dom.dir.iterdir():
            if f.is_file() and f.suffix in (".yaml", ".yml", ".md", ".txt"):
                shutil.copy(f, d / f.name)
        (d / dom.graph_path.name).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        keep = current()
        try:
            cand = set_current(load(d))
            problems = lint(cand)
            segs = core_segments(cand)
        except Exception as e:                          # noqa: BLE001
            return {"loads": False, "problems": [f"{type(e).__name__}: {e}"], "segments": []}
        finally:
            set_current(keep)
        return {"loads": True, "problems": problems, "segments": segs}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_doc(dom, body: dict) -> tuple[dict, dict, str]:
    orig = json.loads(dom.graph_path.read_text())
    if dom.graph is not None:
        if "items" in body:
            doc = apply_items(body["items"])
        else:
            doc = body.get("graph")
        if not isinstance(doc, dict):
            raise ValueError("send the edited graph as items (or the whole graph)")
        return orig, doc, "compact"
    return orig, _triples_doc(orig, body.get("triples") or [], str(body.get("note") or "")), "triples"


def save_version(dom, body: dict) -> dict[str, Any]:
    orig, doc, fmt = build_doc(dom, body)
    changes = diff(orig, doc, fmt)
    if not changes:
        raise ValueError("nothing changed: edit an entry before saving a version")
    chk = check_candidate(dom, doc)
    if not chk["loads"]:
        raise ValueError("the edited graph does not load: " + "; ".join(chk["problems"][:3]))
    out = _next_version(dom)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    before = dom.graph_path.name
    set_graph(dom.dir, out.name)
    return {"file": out.name, "path": rel(out), "from": before, "changes": changes, "problems": chk["problems"],
            "segments": chk["segments"]}
