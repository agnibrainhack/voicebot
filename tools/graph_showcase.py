"""The policy graph as an explorable map for senior leadership and large rooms (one self-contained HTML page).

    python tools/graph_showcase.py                                    # -> docs/POLICY_GRAPH_SHOWCASE.html
    python tools/graph_showcase.py --campaign output/campaign_llm     # the same base cases as that campaign
    python tools/graph_showcase.py --cases none                       # graph only (no store, no oracle)

The map opens clean: only the steps of a call, laid out by stage, and no path drawn. Nothing is revealed
until someone asks:
* hover a step: its own moves appear (where a call can come from and go to from there), labelled;
* click a step: the side panel lists the REAL call paths through it - the paths the policy oracle takes in
  the base test cases, identical ones grouped - each with what it means in plain English, how it ends and how
  many test cases follow it. Filter by ending (completed by the bot / hand-over to a human);
* hover a path to preview it on the map; click it to walk it step by step (a token moves along it; the caption
  says what the caller did, where the bot goes, why, and quotes the policy rule behind it).

Only the policy graph is drawn: its 19 dialog states and its leads_to transitions, nothing else. A real path is
shown only if every move in it is one of those transitions (same from-state, to-state and trigger; the graph's
"any" trigger only for the turn-limit hand-over it describes); test cases that take a move the graph does not have
are counted and named on the page, not drawn. Hovering a step or a move shows every triple the graph file records
about it (stage, acts it accepts, yes/no readings, rules with stance and text, limits with values, tools with their
results, escalation triggers, conditions, workflow edges), with triple ids, source sections and source quotes.

Real paths come from running the policy oracle on the base cases, rebuilt exactly as the answer sheet does
(with --campaign, its base-case settings are read from report.json so paths match its answer sheet). With
--cases none, or if the store cannot be loaded, the panel shows example routes the graph allows (shortest
routes through each move), clearly labelled as examples.
Keys: Esc clear, <- / -> step through the chosen path, F full screen.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- the words on the page
STAGE_ORDER = ["opening", "authenticate", "retrieve", "capture", "record", "card_action", "finish", "transfer", "end"]
STAGE_LABELS = {"opening": "Greet", "authenticate": "Verify the caller", "retrieve": "Find the charges",
                "capture": "Review each charge", "record": "Record the answer", "card_action": "Protect the card",
                "finish": "Wrap up", "transfer": "Human hand-over", "end": "Call ends"}
STATE_LABELS = {
    "OPENING": "Greeting", "ASK_AUTH": "Ask for card digits", "ASK_AUTH_RETRY": "Digits didn't match",
    "PRESENT_TXN": "Read out a charge", "ANSWER_KB": "Answer a policy question",
    "ASK_CLARIFY": "Ask to repeat", "PROVIDE_CONTEXT": "Give more detail",
    "CONFIRM_RECORDED": "Confirm it's recorded", "DISCLOSE_FEE": "Disclose the fee",
    "CONFIRM_ACTION": "Confirm the card action", "ACTION_DONE": "Card action done",
    "ACK_DECLINED": "Caller declined", "INFORM_NO_CARD_ACTION": "No card action available",
    "EXPLAIN_LIMIT": "Explain a limit", "EXPLAIN_NO_TXNS": "No charges to review",
    "SUMMARIZE": "Summarise the call", "ASK_MORE_HELP": "Anything else?",
    "TRANSFER": "Hand over to a human", "END": "Call ends",
}
ACT_LABELS = {
    "PROVIDE_CREDENTIAL": "The caller gives the last four card digits",
    "INFORM_DISPOSITION": "The caller says whether the charge is theirs",
    "REQUEST_CARD_ACTION": "The caller asks to secure or replace the card",
    "CONFIRM": "The caller says yes", "DENY": "The caller says no",
    "REQUEST_HUMAN": "The caller asks for a person", "ASK_KB": "The caller asks a policy question",
    "UNCLEAR": "The caller's answer is unclear", "END": "The caller says that's all",
    "auto": "The bot moves on by itself", "any": "At any point", "always": "Always",
    "call start": "The call starts",
}


# ---------------------------------------------------------------- the graph
def load_graph(kg_path: Path) -> dict:
    d = json.loads(kg_path.read_text(encoding="utf-8"))
    T = d["triples"]
    by_head: dict[str, list[dict]] = {}
    for t in T:
        by_head.setdefault(t["head"], []).append(t)
    tails = lambda h, r: [t["tail"] for t in by_head.get(h, []) if t["relation"] == r]
    states = list(dict.fromkeys(t["head"] for t in T if t["relation"] == "is_a" and t["tail"] == "dialog_state"))
    rules = {t["head"] for t in T if t["relation"] == "is_a" and t["tail"] == "rule"}
    nodes = []
    for s in states:
        isa = next(t for t in by_head[s] if t["relation"] == "is_a" and t["tail"] == "dialog_state")
        nodes.append({"id": s, "label": STATE_LABELS.get(s, s.replace("_", " ").title()),
                      "stage": (tails(s, "in_stage") or ["other"])[0],
                      "waits": (tails(s, "prompts_customer") or ["no"])[0] == "yes",
                      "accepts": [ACT_LABELS.get(a, a) for a in tails(s, "accepts_act")],
                      "desc": isa.get("evidence") or "", "source": isa.get("source") or ""})
    edges = []
    for t in T:
        if t["relation"] != "leads_to":
            continue
        q = t.get("qualifiers") or {}
        on = q.get("on") or "always"
        rtext = lambda r: next((x["tail"] for x in by_head.get(r, []) if x["relation"] == "has_text"), "")
        edges.append({"id": t["id"], "src": t["head"], "dst": t["tail"], "on": on,
                      "on_label": ACT_LABELS.get(on, on), "when": q.get("when") or "always",
                      "rules": [{"id": r, "text": rtext(r)} for r in q.get("rules") or []]})
    stage_order = [s for s in STAGE_ORDER if any(n["stage"] == s for n in nodes)]
    stage_order += sorted({n["stage"] for n in nodes} - set(stage_order))
    return {"nodes": nodes, "edges": edges,
            "stages": [{"id": s, "label": STAGE_LABELS.get(s, s.replace("_", " ").title())} for s in stage_order],
            "facts": {"states": len(nodes), "moves": len(edges), "rules": len(rules)},
            # everything in the graph file, for the hover details (nothing on the page is added to it)
            "triples": T, "relations": d.get("relations") or {},
            "sections": ((d.get("source") or {}).get("sections") or {}),
            "document": (d.get("source") or {}).get("document"), "graph_title": d.get("title")}


REASON_EN = {"customer_request": "the caller asked for a person", "auth_failed": "verification failed twice",
             "auth_unavailable": "the verification system is down",
             "retrieve_unavailable": "the charges could not be retrieved", "record_failed": "an answer could not be recorded",
             "unsure_twice": "the caller stayed unsure", "ambiguity_persists": "the answer stayed unclear",
             "transaction_limit": "too many charges for one call", "card_options_unavailable": "card options are unavailable",
             "submit_failed": "the card action could not be submitted", "turn_limit": "the call reached its length limit"}
# what reaching a step says about the call (first time only), in the order it happens
EVENT_EN = {"ASK_AUTH_RETRY": "wrong card digits at first", "EXPLAIN_NO_TXNS": "no charges to review",
            "ANSWER_KB": "a policy question", "ASK_CLARIFY": "an unclear answer", "PROVIDE_CONTEXT": "an unsure caller",
            "EXPLAIN_LIMIT": "a policy limit explained", "DISCLOSE_FEE": "an expedited card, fee disclosed",
            "CONFIRM_ACTION": "a card action requested", "ACTION_DONE": "card action completed",
            "ACK_DECLINED": "card action declined", "INFORM_NO_CARD_ACTION": "no card action available"}


def headline(hops: list[dict]) -> str:
    """What happens on a path, in a few words, read off the path itself (not the test set-up)."""
    seen, ev = set(), []
    import re
    n_charges = len({m for h in hops for t in h["tools"] if t.startswith("record_charge_response(")
                     for m in re.findall(r"transaction_id=(\w+)", t)})
    for h in hops:
        d = h["dst"]
        if d in EVENT_EN and d not in seen:
            seen.add(d)
            if d == "CONFIRM_ACTION" and "DISCLOSE_FEE" in seen:
                continue
            ev.append(EVENT_EN[d])
        if d == "PRESENT_TXN" and "charges" not in seen:
            seen.add("charges")
            ev.append("a charge read out, not yet answered" if not n_charges else
                      "one charge answered" if n_charges == 1 else f"{n_charges} charges answered")
    return ", ".join(ev)


# ---------------------------------------------------------------- real paths (the oracle on the base cases)
def _trace_hops(tr, kg) -> list[dict]:
    """Every dialog-state move of one oracle trace, in order: {src, dst, on, tools}. Same split as the answer
    sheet (narrate.turn_hops): the customer's act triggers a turn's first move, the bot's own moves are 'auto'."""
    from gmsjourney.narrate import tool_line, turn_hops
    opening, groups = [], []
    for ev in tr.events:
        if ev["role"] == "user":
            groups.append({"on": ev["frame"].act, "steps": []})
        elif ev["role"] == "tool":
            (groups[-1]["steps"] if groups else opening).append(("tool", ev))
        else:
            (groups[-1]["steps"] if groups else opening).extend(("act", a) for a in ev.get("acts") or [])
    out, cur = [], None
    for st_on, steps in [(None, opening)] + [(g["on"], g["steps"]) for g in groups]:
        for h in turn_hops(kg, cur, st_on, steps):
            if h["src"] is None or h["dst"] is None:        # the call's first state / a turn that stays put
                cur = h["dst"] or cur
                continue
            out.append({"src": h["src"], "dst": h["dst"], "on": h["on"], "tools": [tool_line(x) for x in h["tools"]]})
            cur = h["dst"]
    return out


def real_paths(policy: Path, campaign: Path | None, graph: dict, log=print) -> list[dict]:
    from gmsjourney.bases.journey import JourneyBaseGenerator, to_item
    from gmsjourney.bases.kb import mine_kb_items
    from gmsjourney.journey import JourneySpec
    from gmsjourney.oracle import PolicyOracle
    from gmsjourney.pipeline import build_gms
    from gmsjourney.enrich.llm.prompts import describe_meaning
    from tools.answer_sheet import DIM_EN

    cfg = {}
    if campaign and (campaign / "report.json").exists():
        cfg = json.loads((campaign / "report.json").read_text(encoding="utf-8")).get("config", {}) or {}
    seed = 42
    log("loading the knowledge store (reloaded from the cache when one matches) ...")
    store, _rep, info = build_gms(policy, backend_choice="auto", seed=seed, log=lambda *a: None)
    oracle, kg = PolicyOracle(store), getattr(store, "kg", None)
    items, _ = mine_kb_items(store, seed, extended=cfg.get("kb_questions") == "extended")
    admitted = [k for k in items if k.admission != "rejected"]
    mtl = cfg.get("max_turn_limit_cases", 2) if cfg else 2
    gen = JourneyBaseGenerator(oracle, kb_pool=[k.frame for k in admitted], seed=seed, max_turn_limit_cases=mtl)
    cover, rep = gen.covering()
    extra = gen.closure(cover, store, rep, extra_traces=[
        oracle.run(JourneySpec(n_txn=1, kb_mid="ask", kb_frames=(k.frame,))) for k in admitted])
    specs = list({s.key(): s for s in cover + extra}.values())
    items_ = [(to_item(sp, oracle, f"J{i:03d}"), None) for i, sp in enumerate(specs)]
    items_ += [(to_item(JourneySpec(n_txn=1, kb_mid="ask", kb_frames=(k.frame,)), oracle, k.qid,
                        category=k.category, kb_item=k), k) for k in admitted]
    e_index: dict[tuple, str] = {}
    for e in graph["edges"]:
        e_index.setdefault((e["src"], e["dst"], e["on"]), e["id"])
    groups: dict[tuple, dict] = {}
    for it, k in items_:
        tr = oracle.run(it.metadata["spec"], it.metadata["fixture"])
        hops = _trace_hops(tr, kg)
        xfer = [c for c in tr.tool_log if c.get("tool") == "transfer_to_human"]
        if hops and hops[-1]["dst"] == "TRANSFER" and xfer:  # the hand-over the trace made closes the call
            from gmsjourney.narrate import tool_line
            hops.append({"src": "TRANSFER", "dst": "END", "on": "auto", "tools": [tool_line(xfer[-1])]})
        outcome = json.loads(it.answer) if isinstance(it.answer, str) else (it.answer or {})
        if k is not None:
            tests = [f"a policy question mid-call: the caller {describe_meaning(k.frame)}"
                     + (" (the policy has no answer, so the bot must say so)" if k.probe else "")]
        else:
            rel = {d: v for d, v in (it.metadata.get("relevant") or {}).items() if v not in (None, "N/A")}
            tests = [x for x in (DIM_EN.get(d, {}).get(v) for d, v in rel.items()) if x]
        key = tuple((h["src"], h["dst"], h["on"]) for h in hops)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"hops": [{**h, "edge": e_index.get((h["src"], h["dst"], h["on"]))
                                         or (e_index.get((h["src"], h["dst"], "any"))   # the graph's "any" move is
                                             if tr.transfer_reason == "turn_limit" else None)}  # the turn limit
                                        for h in hops],
                               "cases": [], "tests": [], "end": outcome.get("end"),
                               "reason": tr.transfer_reason or outcome.get("transfer_reason"),
                               "per_case": [tests], "per_case_all": []}
        g["cases"].append(it.qid)
        g["per_case_all"].append(tests)
        for t in tests:
            if t not in g["tests"]:
                g["tests"].append(t)
    # what the path's test cases set up: only what ALL of them share (a covering-array case sets many factors at
    # once, most of which this path never reaches); many policy questions folded into one line
    routine = {DIM_EN["auth"]["ok"], DIM_EN["n_txn"][1], DIM_EN["disposition"]["all_recognized"]}
    for g in groups.values():
        kb = [t for t in g["tests"] if t.startswith("a policy question mid-call")]
        shared = [t for t in g.pop("per_case")[0] if all(t in c for c in g["per_case_all"])]
        g.pop("per_case_all")
        if len(kb) > 1:
            kb = [f"a policy question mid-call ({len(kb)} different questions, e.g. the caller "
                  f"{kb[0].split('the caller ', 1)[-1]})"]
        g["tests"] = kb + [t for t in shared if t not in kb and t not in routine and
                           not t.startswith("a policy question mid-call")]
        g["headline"] = headline(g["hops"])
    out = sorted(groups.values(), key=lambda g: (-len(g["cases"]), len(g["hops"])))
    # only paths whose every move is a transition of the policy graph are shown; the rest are counted, not drawn
    kept = [g for g in out if all(h["edge"] for h in g["hops"])]
    dropped = [g for g in out if not all(h["edge"] for h in g["hops"])]
    for i, g in enumerate(kept):
        g["id"] = f"P{i + 1:02d}"
    graph["left_out"] = {"cases": sum(len(g["cases"]) for g in dropped), "paths": len(dropped),
                         "case_ids": [c for g in dropped for c in g["cases"]],
                         "moves": sorted({f'{h["src"]} --{h["on"]}--> {h["dst"]}' for g in dropped
                                          for h in g["hops"] if not h["edge"]})}
    log(f"{len(items_)} base cases -> {len(out)} distinct real paths (backend {info.get('selected')}); "
        f"{len(kept)} shown, {len(dropped)} left out ({graph['left_out']['cases']} cases take a move "
        f"that is not in the policy graph: {graph['left_out']['moves']})")
    return kept


def example_paths(graph: dict) -> list[dict]:
    """Graph only: for each move, the shortest allowed route start -> move -> end of call."""
    out_e: dict[str, list[dict]] = {}
    for e in graph["edges"]:
        out_e.setdefault(e["src"], []).append(e)
    start = graph["nodes"][0]["id"]

    def shortest(a: str, goal: str) -> list[dict] | None:
        q, seen = deque([(a, [])]), {a}
        while q:
            s, p = q.popleft()
            if s == goal:
                return p
            for e in out_e.get(s, []):
                if e["dst"] not in seen:
                    seen.add(e["dst"])
                    q.append((e["dst"], p + [e]))
        return None
    paths, seen_keys = [], set()
    for e in graph["edges"]:
        head, tail = shortest(start, e["src"]), shortest(e["dst"], "END")
        if head is None or tail is None:
            continue
        p = head + [e] + tail
        key = tuple(x["id"] for x in p)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        paths.append({"hops": [{"edge": x["id"], "src": x["src"], "dst": x["dst"], "on": x["on"], "tools": []}
                               for x in p], "cases": [], "tests": [],
                      "end": "TRANSFERRED" if any(x["dst"] == "TRANSFER" for x in p) else "COMPLETED",
                      "reason": None})
        paths[-1]["headline"] = headline(paths[-1]["hops"])
    for i, p in enumerate(paths):
        p["id"] = f"A{i + 1:02d}"
    return paths


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0f1318;--surface:#161b22;--surface-2:#1d242d;--line:#2b3440;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#8e98a4;
--normal:#3987e5;--recovery:#199e70;--human:#e66767;--node:#1f2731;--node-hi:#26313d;--glow:#ffd36e;
--font:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif}
:root[data-theme="light"]{color-scheme:light;--bg:#f4f5f7;--surface:#fcfcfb;--surface-2:#eef0f3;--line:#d5dae1;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6b7280;
--normal:#2a78d6;--recovery:#1baf7a;--human:#e34948;--node:#ffffff;--node-hi:#eef4fd;--glow:#b86e00}
@media (prefers-color-scheme:light){:root:not([data-theme="dark"]){color-scheme:light;--bg:#f4f5f7;--surface:#fcfcfb;--surface-2:#eef0f3;--line:#d5dae1;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6b7280;
--normal:#2a78d6;--recovery:#1baf7a;--human:#e34948;--node:#ffffff;--node-hi:#eef4fd;--glow:#b86e00}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font)}
body{padding:clamp(12px,2vw,26px) 16px}
.wrap{max-width:1800px;margin:0 auto}
header{display:flex;flex-wrap:wrap;gap:.8rem 2rem;align-items:flex-end;justify-content:space-between;margin-bottom:.9rem}
h1{margin:0;font-size:clamp(1.6rem,3vw,2.6rem);font-weight:700;letter-spacing:-.01em;line-height:1.1}
.sub{color:var(--ink-2);font-size:clamp(.95rem,1.25vw,1.15rem);margin:.35rem 0 0;max-width:82ch}
.facts{color:var(--muted);font-size:.88rem;margin-top:.3rem}
.tools{display:flex;gap:.5rem}
button{font:inherit;cursor:pointer;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:.42rem .8rem}
button:hover{border-color:var(--normal)}
button:focus-visible,.node:focus-visible{outline:2px solid var(--glow);outline-offset:2px}
.main{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:14px;align-items:start}
@media (max-width:1100px){.main{grid-template-columns:1fr}}
.map{background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden;position:relative}
.hint{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);background:var(--surface-2);border:1px solid var(--line);border-radius:999px;
  padding:.4rem 1rem;color:var(--ink-2);font-size:.95rem;pointer-events:none;transition:opacity .3s;white-space:nowrap}
svg{display:block;width:100%;height:auto}
.lane-t{fill:var(--muted);font-size:14px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.lane{fill:var(--surface-2);opacity:.55}
.edge{fill:none;stroke-width:2.2;opacity:0;transition:opacity .25s,stroke-width .2s}
.edge.normal{stroke:var(--normal)} .edge.recovery{stroke:var(--recovery);stroke-dasharray:7 6} .edge.human{stroke:var(--human);stroke-width:2.8}
.edge.gap{stroke:var(--muted);stroke-dasharray:2 5}
.edge.near{opacity:.6}
.edge.path{opacity:.9;stroke-width:4}
.edge.walked{opacity:1;stroke-width:5.5}
.lbl{font-size:12.5px;fill:var(--ink-2);opacity:0;pointer-events:none;transition:opacity .2s;paint-order:stroke;stroke:var(--surface);stroke-width:4px}
.lbl.on{opacity:1}
.node rect{fill:var(--node);stroke:var(--line);stroke-width:1.5;transition:fill .2s,stroke .2s}
.node{transition:opacity .2s;cursor:pointer}
.node.waits rect{stroke-width:2}
.node:not(.waits) rect{stroke-dasharray:5 4}
.node text{fill:var(--ink);font-size:16px;font-weight:600;pointer-events:none}
.node .code{fill:var(--muted);font-size:11px;font-weight:500;font-family:ui-monospace,Consolas,monospace}
.node.focus rect,.node:hover rect{stroke:var(--glow);stroke-width:2.5}
.node.onpath rect{fill:var(--node-hi);stroke:var(--normal);stroke-dasharray:none}
.node.walked rect{fill:var(--node-hi);stroke:var(--glow);stroke-width:3;stroke-dasharray:none}
.node.faded{opacity:.25}
.token{fill:var(--glow);filter:drop-shadow(0 0 8px var(--glow));opacity:0}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:1rem;max-height:calc(100vh - 28px);overflow:auto;position:sticky;top:12px}
.panel h2{margin:.1rem 0 0;font-size:1.35rem}
.panel .m{color:var(--muted);font-size:.85rem;margin:.3rem 0}
.panel .d{color:var(--ink-2);font-size:.95rem;margin:.45rem 0 .6rem;line-height:1.45}
.filters{display:flex;gap:.4rem;flex-wrap:wrap;margin:.7rem 0 .3rem}
.filters button{font-size:.82rem;padding:.25rem .6rem}
.filters button[aria-pressed="true"]{border-color:var(--glow);box-shadow:0 0 0 1px var(--glow) inset}
.pcard{display:block;width:100%;text-align:left;margin:.45rem 0;padding:.6rem .7rem;border-radius:10px;background:var(--surface-2)}
.pcard b{display:block;font-size:.97rem;line-height:1.35;font-weight:600}
.pcard span{display:block;color:var(--muted);font-size:.8rem;margin-top:.25rem}
.pcard[aria-pressed="true"]{border-color:var(--glow);box-shadow:0 0 0 1px var(--glow) inset}
.dot{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;margin-right:.35rem}
.caption{margin-top:12px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:1rem 1.2rem;display:none}
.caption.on{display:block}
.caption h3{margin:0 0 .25rem;font-size:clamp(1.1rem,1.7vw,1.5rem);line-height:1.25}
.caption .meta{color:var(--muted);font-size:.9rem;line-height:1.4}
.caption ol{margin:.7rem 0 0;padding-left:1.5rem;display:grid;gap:.3rem}
.caption li{color:var(--ink-2);line-height:1.45;font-size:clamp(.92rem,1.1vw,1.05rem);opacity:.55;transition:opacity .2s}
.caption li.done{opacity:.85}
.caption li.now{color:var(--ink);opacity:1}
.caption li .r{color:var(--muted)}
.ctrl{display:flex;gap:.5rem;margin-top:.8rem}
.legend{display:flex;flex-wrap:wrap;gap:.8rem 1.3rem;color:var(--ink-2);font-size:.88rem;margin:.7rem 0 0}
.legend i{display:inline-block;width:26px;height:0;border-top:3px solid;vertical-align:middle;margin-right:.4rem}
.hit{fill:none;stroke:transparent;stroke-width:16;pointer-events:none;cursor:help}
.hit.live{pointer-events:stroke}
.edge.hot{opacity:1!important;stroke-width:5.5}
.tip.wide{max-width:min(1240px,97vw);width:min(1240px,97vw)}
.tip .cols3{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr) minmax(0,1fr);gap:1rem}
@media (max-width:900px){.tip .cols3{grid-template-columns:1fr}}
.tip .h{font-size:1.02rem}
.tip code{font-family:ui-monospace,Consolas,monospace;font-size:.82em;color:var(--muted)}
.tip .tid{font-family:ui-monospace,Consolas,monospace;font-size:.72rem;color:var(--muted);white-space:nowrap}
.tip .sec{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:.5rem 0 .2rem}
.tip .cols{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:.9rem}
.tip table.kv{border-collapse:collapse;width:100%;font-size:.8rem;margin-top:.35rem}
.tip table.kv th{text-align:left;vertical-align:top;color:var(--muted);font-weight:500;padding:.12rem .5rem .12rem 0;white-space:nowrap}
.tip table.kv td{padding:.12rem 0;line-height:1.3}
.tip table.kv td>div{margin:.05rem 0}
.tip ul.ev{margin:0;padding-left:1rem;font-size:.78rem;line-height:1.35;color:var(--ink-2)}
.tip .sectext{font-size:.76rem;line-height:1.35;color:var(--ink-2)}
.tip{position:fixed;z-index:10;max-width:480px;overflow:hidden;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.65rem .8rem;font-size:.88rem;line-height:1.45;box-shadow:0 10px 30px rgba(0,0,0,.35);pointer-events:none;display:none;color:var(--ink)}
.tip b{font-weight:600}
.tip b.h{display:block;font-size:1.02rem;margin-bottom:.15rem}
.tip .m{color:var(--muted);font-size:.8rem}
@media (prefers-reduced-motion:reduce){.edge,.node,.node rect,.lbl,.caption li{transition:none}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 id="title"></h1>
      <p class="sub">Every step a call can reach, stage by stage. Hover a step to see where a call can go from there. Click it to see the real calls that pass through it, and what each one means.</p>
      <div class="facts" id="facts"></div>
    </div>
    <div class="tools">
      <button id="theme" type="button">Light / dark</button>
      <button id="full" type="button">Full screen</button>
    </div>
  </header>
  <div class="main">
    <div>
      <div class="map"><svg id="map" role="img" aria-label="Policy graph of the voice bot"></svg><div class="hint" id="hint">Hover a step to see its moves · click it to see the real calls through it</div></div>
      <div class="caption" id="cap" aria-live="polite"></div>
      <div class="legend">
        <span><i style="border-color:var(--normal)"></i>Normal flow</span>
        <span><i style="border-color:var(--recovery);border-top-style:dashed"></i>Recovery (retry, clarify, explain)</span>
        <span><i style="border-color:var(--human);border-top-width:4px"></i>Hand-over to a human</span>
        <span>Solid box: the bot waits for the caller · dashed box: the bot acts on its own</span>
      </div>
    </div>
    <aside class="panel" id="panel"></aside>
  </div>
</div>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const cap1 = s => s ? s[0].toUpperCase() + s.slice(1) : s;
$("#title").textContent = D.title; document.title = D.title;
const tip = $("#tip");
function moveTip(ev) { const w = tip.offsetWidth, h = tip.offsetHeight; let x = ev.clientX + 16, y = ev.clientY + 16;
  if (x + w > innerWidth - 8) x = Math.max(8, ev.clientX - w - 16); if (y + h > innerHeight - 8) y = Math.max(8, innerHeight - h - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px"; }
const N = Object.fromEntries(D.nodes.map(n => [n.id, n])), E = Object.fromEntries(D.edges.map(e => [e.id, e]));
const REAL = D.paths_kind === "real";
$("#facts").textContent = `${D.facts.states} steps · ${D.facts.moves} allowed moves · ${D.facts.rules} policy rules`
  + (REAL ? ` · ${D.paths.reduce((a, p) => a + p.cases.length, 0)} test cases` : "");

// ---- layout: one column per stage (sub-columns where the most common call has two steps in a stage)
const spine = new Set(D.spine), happy = new Set(D.happy);
const spineIn = {}; D.spine.forEach(id => { const st = N[id].stage; (spineIn[st] = spineIn[st] || []).push(id); });
const units = D.stages.map(s => Math.max(1, (spineIn[s.id] || []).length));
const W = 1680, unitW = W / units.reduce((a, b) => a + b, 0), nodeW = Math.min(178, unitW - 14), nodeH = 54, top0 = 86, rowH = 92;
const stageX = {}, stageW = {}; let acc = 0;
D.stages.forEach((s, i) => { stageX[s.id] = acc; stageW[s.id] = units[i] * unitW; acc += units[i] * unitW; });
const rowsUsed = {}, pos = {};
const ordered = [...D.nodes].sort((a, b) => (spine.has(b.id) - spine.has(a.id)) || (happy.has(b.id) - happy.has(a.id)));
for (const n of ordered) {
  let r, x;
  if (spine.has(n.id)) { r = 0; x = stageX[n.stage] + spineIn[n.stage].indexOf(n.id) * unitW + (unitW - nodeW) / 2; }
  else { r = n.id === "TRANSFER" ? 5 : (rowsUsed[n.stage] ?? 1); if (n.id !== "TRANSFER") rowsUsed[n.stage] = r + 1;
         x = stageX[n.stage] + (stageW[n.stage] - nodeW) / 2; }
  pos[n.id] = { x, y: top0 + r * rowH, r };
}
const H = top0 + (Math.max(...Object.values(pos).map(p => p.r)) + 1) * rowH + 50;
const svg = $("#map"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };
function fit(t, max) { let size = parseFloat(getComputedStyle(t).fontSize) || 16; while (t.getComputedTextLength() > max && size > 9) { size -= .5; t.style.fontSize = size + "px"; } }
D.stages.forEach((s, i) => {
  if (i % 2 === 0) el("rect", { x: stageX[s.id], y: 0, width: stageW[s.id], height: H, class: "lane" });
  const t = el("text", { x: stageX[s.id] + stageW[s.id] / 2, y: 36, "text-anchor": "middle", class: "lane-t" }); t.textContent = s.label; fit(t, stageW[s.id] - 22);
});
const kind = (src, dst) => (dst === "TRANSFER" || src === "TRANSFER") ? "human" : ((happy.has(src) && happy.has(dst)) ? "normal" : "recovery");
function pathFor(src, dst, k = 0, n = 1) {        // k-th of n parallel transitions between the same two steps
  const a = pos[src], b = pos[dst]; if (!a || !b) return "";
  const o = (k - (n - 1) / 2) * 16;
  if (src === dst) { const x = a.x + nodeW * .7, y = a.y, h = 46 + k * 14; return `M${x - 22},${y} C${x - 30},${y - h} ${x + 30},${y - h} ${x + 22},${y}`; }
  if (b.x > a.x + 1) { const x1 = a.x + nodeW, y1 = a.y + nodeH / 2 + o * .35, x2 = b.x, y2 = b.y + nodeH / 2 + o * .35, dx = (x2 - x1) * .45;
    return `M${x1},${y1} C${x1 + dx},${y1 + o * 1.6} ${x2 - dx},${y2 + o * 1.6} ${x2},${y2}`; }
  if (Math.abs(b.x - a.x) <= 1) { const x1 = a.x + nodeW, y1 = a.y + nodeH / 2, x2 = b.x + nodeW, y2 = b.y + nodeH / 2, bulge = 36 + Math.abs(y2 - y1) * .15 + k * 18;
    return `M${x1},${y1} C${x1 + bulge},${y1} ${x2 + bulge},${y2} ${x2},${y2}`; }
  const x1 = a.x + nodeW / 2 + o, y1 = a.y, x2 = b.x + nodeW / 2 + o, y2 = b.y, lift = Math.min(34, 22 + (a.x - b.x) / unitW * 4) + k * 12;
  return `M${x1},${y1} C${x1},${y1 - lift} ${x2},${y2 - lift} ${x2},${y2}`;
}
const short = on => on === "auto" || on === "always" ? "" : on === "UNCLEAR" ? "unclear answer" : on === "any" ? "any time"
  : (D.act_labels[on] || on).replace(/^The caller /, "");
const gE = el("g", {}), gL = el("g", {}), gH = el("g", {}), gN = el("g", {});
const edgeEl = {}, lblEl = {}, hitEl = {};
const pairN = {}, pairK = {};
for (const e of D.edges) { const k = e.src + ">" + e.dst; pairK[e.id] = pairN[k] = (pairN[k] ?? -1) + 1; }
for (const e of D.edges) {
  const d = pathFor(e.src, e.dst, pairK[e.id], pairN[e.src + ">" + e.dst] + 1); if (!d) continue;
  const p = edgeEl[e.id] = el("path", { d, class: "edge " + kind(e.src, e.dst) }, gE);
  const mid = p.getPointAtLength(p.getTotalLength() / 2);
  const t = lblEl[e.id] = el("text", { x: mid.x, y: mid.y - 7, "text-anchor": "middle", class: "lbl" }, gL);
  t.textContent = short(e.on);
  const h = hitEl[e.id] = el("path", { d, class: "hit" }, gH);  // a wide invisible target, live only while the move is shown
  h.addEventListener("mouseenter", ev => { overEdge = e.id; clearTimeout(leaveT); p.classList.add("hot"); tipEdge(ev, e); });
  h.addEventListener("mousemove", moveTip);
  h.addEventListener("mouseleave", () => { overEdge = null; p.classList.remove("hot"); hideTip(); leaveSoon(); });
}
const hopKey = h => h.edge;
let overEdge = null, overNode = null, leaveT = null;
function leaveSoon() { clearTimeout(leaveT); leaveT = setTimeout(() => { if (!overEdge && !overNode) hover(null); }, 350); }
const nodeEl = {};
for (const n of D.nodes) {
  const p = pos[n.id];
  const g = el("g", { class: "node" + (n.waits ? " waits" : ""), transform: `translate(${p.x},${p.y})`, tabindex: 0, role: "button", "aria-label": n.label }, gN);
  el("rect", { width: nodeW, height: nodeH, rx: 12 }, g);
  const t = el("text", { x: nodeW / 2, y: 24, "text-anchor": "middle" }, g); t.textContent = n.label; fit(t, nodeW - 14);
  const c = el("text", { x: nodeW / 2, y: 42, "text-anchor": "middle", class: "code" }, g); c.textContent = n.id; fit(c, nodeW - 14);
  g.addEventListener("mouseenter", ev => { overNode = n.id; clearTimeout(leaveT); hover(n.id); tipNode(ev, n); });
  g.addEventListener("mousemove", moveTip);
  g.addEventListener("mouseleave", () => { overNode = null; hideTip(); leaveSoon(); });
  g.addEventListener("focus", () => hover(n.id)); g.addEventListener("blur", () => hover(null));
  g.addEventListener("click", () => select(n.id));
  g.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(n.id); } });
  nodeEl[n.id] = g;
}
const token = el("circle", { r: 11, class: "token" });

// ---- hover details: every triple of the policy graph about the step / the move (nothing added)
const byHead = {}, byTail = {};
for (const t of D.triples) { (byHead[t.head] = byHead[t.head] || []).push(t); (byTail[t.tail] = byTail[t.tail] || []).push(t); }
const facts = (h, r) => (byHead[h] || []).filter(t => t.relation === r);
const one = (h, r) => (facts(h, r)[0] || {}).tail;
const enablesByW = {};
for (const t of D.triples) if (t.relation === "enables" && t.qualifiers && t.qualifiers.edge) enablesByW[t.qualifiers.edge] = t;
const tid = t => `<span class="tid">${esc(t.id)}${t.source ? " · " + esc(t.source) : ""}${t.gms ? "" : " · not in GMS"}</span>`;
function entity(x) {                                  // a tail, with what the graph says about it
  const s = String(x), H = byHead[s] || [];
  const isa = (H.find(t => t.relation === "is_a") || {}).tail;
  if (isa === "rule") return `<b>${esc(s)}</b> <i>${esc(one(s, "has_stance") || "")}</i>${one(s, "constrains") ? " · on " + esc(one(s, "constrains")) : ""}: ${esc(one(s, "has_text") || "")}`;
  if (isa === "limit") { const v = facts(s, "has_value")[0]; return `<b>${esc(s)}</b> = ${esc(v ? v.tail : "?")} ${esc(one(s, "has_unit") || "")}${one(s, "has_text") ? " — " + esc(one(s, "has_text")) : ""}`; }
  if (isa === "tool") {
    const bits = [["when used", "when_used"], ["stage", "has_stage"], ["success", "success_value"], ["failure", "failure_value"], ["on failure", "on_failure"]]
      .map(([l, r]) => { const v = facts(s, r).map(t => t.tail); return v.length ? `${l}: ${esc(v.join(" / "))}` : ""; }).filter(Boolean);
    return `<b>${esc(s)}</b>${bits.length ? " — " + bits.join("; ") : ""}`;
  }
  if (isa === "dialog_act") return `<b>${esc(s)}</b>${D.act_labels[s] ? " — " + esc(D.act_labels[s]) : ""}`;
  return `<b>${esc(s)}</b>`;
}
const relDesc = r => D.relations[r] ? ` title="${esc(D.relations[r])}"` : "";
function evidence(ts) {                               // the source sentences behind these triples, one per section quote
  const seen = new Set(), out = [];
  for (const t of ts) { const k = (t.source || "") + "|" + (t.evidence || ""); if (!t.evidence || seen.has(k)) continue; seen.add(k);
    out.push(`<li><span class="tid">${esc(t.source || "")}</span> “${esc(t.evidence)}”</li>`); }
  return out.length ? `<div class="sec">Source quotes</div><ul class="ev">${out.join("")}</ul>` : "";
}
function tipNode(ev, n) {
  const own = (byHead[n.id] || []).filter(t => t.relation !== "leads_to");
  const rels = [...new Set(own.map(t => t.relation))];
  const into = (byTail[n.id] || []).filter(t => t.relation !== "leads_to");
  const outs = D.edges.filter(e => e.src === n.id), ins = D.edges.filter(e => e.dst === n.id);
  const through = D.paths.filter(p => p.hops.some(h => h.src === n.id || h.dst === n.id)).length;
  const row = r => `<tr><th${relDesc(r)}>${esc(r.replace(/_/g, " "))}</th><td>${own.filter(t => t.relation === r).map(t => `<div>${entity(t.tail)} ${tid(t)}</div>`).join("")}</td></tr>`;
  tip.innerHTML = `<b class="h">${esc(n.label)} <code>${esc(n.id)}</code></b>`
    + `<div class="m">${n.waits ? "The bot waits for the caller here" : "The bot acts on its own here"} · ${outs.length} move${outs.length === 1 ? "" : "s"} out, ${ins.length} in`
    + (through ? ` · ${through} ${REAL ? "real path" : "example route"}${through === 1 ? "" : "s"} (click)` : "") + `</div>`
    + `<div class="cols3"><div><div class="sec">In the policy graph</div><table class="kv">`
    + rels.filter(r => r !== "applies_rule").map(row).join("")
    + into.map(t => `<tr><th${relDesc(t.relation)}>${esc(t.relation.replace(/_/g, " "))} (from)</th><td><div>${entity(t.head)} ${tid(t)}</div></td></tr>`).join("")
    + `</table></div><div><div class="sec">Rules applied here</div><table class="kv">${rels.includes("applies_rule") ? row("applies_rule").replace(/<th[^>]*>applies rule<\/th>/, "") : "<tr><td>—</td></tr>"}</table></div>`
    + `<div>${evidence(own.concat(into))}</div></div>`;
  tip.classList.add("wide"); tip.style.display = "block"; fitTip(); moveTip(ev);
}
function tipEdge(ev, e) {
  const t = D.triples.find(x => x.id === e.id), q = t.qualifiers || {};
  const used = D.paths.filter(p => p.hops.some(h => h.edge === e.id));
  const nCases = used.reduce((a, p) => a + p.cases.length, 0);
  const other = Object.keys(q).filter(k => !["on", "when", "rules", "edges"].includes(k));
  tip.innerHTML = `<b class="h">${esc(N[e.src].label)} → ${esc(N[e.dst].label)}</b>`
    + `<div class="m"><code>${esc(e.src)}</code> leads_to <code>${esc(e.dst)}</code> · triple ${tid(t)} · kind ${esc(t.kind || "")}`
    + (REAL ? ` · ${nCases ? `taken in ${nCases} test case${nCases === 1 ? "" : "s"}` : "no test case takes it"}` : "") + `</div>`
    + `<table class="kv"><tr><th${relDesc("leads_to")}>on</th><td>${entity(q.on || "always")}${["auto", "any", "always"].includes(q.on) ? " — " + esc(D.act_labels[q.on] || "") : ""}</td></tr>`
    + `<tr><th>when</th><td>${esc(q.when || "always")}</td></tr>`
    + `<tr><th>rules</th><td>${(q.rules || []).map(r => `<div>${entity(r)}</div>`).join("") || "—"}</td></tr>`
    + `<tr><th>workflow edges</th><td>${(q.edges || []).map(w => { const x = enablesByW[w]; return `<div><b>${esc(w)}</b>${x ? `: ${esc(x.head)} enables ${esc(x.tail)} when ${esc((x.qualifiers || {}).condition || "")} ${tid(x)}` : ""}</div>`; }).join("") || "—"}</td></tr>`
    + other.map(k => `<tr><th>${esc(k)}</th><td>${esc(JSON.stringify(q[k]))}</td></tr>`).join("")
    + `</table>` + evidence([t])
    + (t.source && D.sections[t.source] ? `<div class="sec">Section ${esc(t.source)} of ${esc(D.document || "the policy")}</div><div class="sectext">${esc(D.sections[t.source])}</div>` : "");
  tip.classList.remove("wide"); tip.style.display = "block"; fitTip(); moveTip(ev);
}
function fitTip() {                                   // a long card is scaled down to fit the screen, never cut
  let z = 1; tip.style.zoom = 1;
  while (tip.scrollHeight * z > innerHeight - 16 && z > .55) { z -= .05; tip.style.zoom = z; }
}
const hideTip = () => { tip.style.display = "none"; };

// ---- state
let hoverId = null, sel = null, chosen = null, preview = null, at = -1, filter = "all", timer = null;
const pathById = Object.fromEntries(D.paths.map(p => [p.id, p]));
function paint() {
  const showId = preview || chosen, P = pathById[showId], walking = P && showId === chosen && !preview;
  for (const k in edgeEl) { edgeEl[k].classList.remove("near", "path", "walked"); lblEl[k].classList.remove("on"); }
  for (const id in nodeEl) nodeEl[id].classList.remove("onpath", "walked", "faded", "focus");
  $("#hint").style.opacity = (hoverId || sel) ? 0 : 1;
  if (P) {
    const onN = new Set(P.hops.flatMap(h => [h.src, h.dst]));
    for (const id in nodeEl) nodeEl[id].classList.toggle("faded", !onN.has(id));
    P.hops.forEach((h, i) => {
      const k = hopKey(h), done = walking && i <= at;
      if (!edgeEl[k]) return;
      if (walking && i > at) { edgeEl[k].classList.add("near"); return; }
      edgeEl[k].classList.add(done ? "walked" : "path"); lblEl[k].classList.add("on");
      nodeEl[h.src].classList.add(done ? "walked" : "onpath");
      nodeEl[h.dst].classList.add(done ? "walked" : "onpath");
    });
    if (walking && at < 0) nodeEl[P.hops[0].src].classList.add("walked");
  } else {
    const f = hoverId || sel;
    if (f) for (const e of D.edges) if (e.src === f || e.dst === f) { edgeEl[e.id].classList.add("near"); lblEl[e.id].classList.add("on"); }
  }
  for (const k in hitEl) hitEl[k].classList.toggle("live", ["near", "path", "walked"].some(c => edgeEl[k].classList.contains(c)));
  if (sel) nodeEl[sel].classList.add("focus");
  if (hoverId && !P) nodeEl[hoverId].classList.add("focus");
}
function hover(id) { hoverId = id; paint(); }

// ---- what a path means
const actText = h => h.edge ? E[h.edge].on_label : (D.act_labels[h.on] || h.on);
function meaning(p) {
  const title = cap1(p.headline || (p.end === "TRANSFERRED" ? "Hand-over before any charge is read out" : "Ends before any charge is read out"));
  let decisive = null;                              // the move that set the ending: last one before the wrap-up
  for (const h of p.hops) if (!["SUMMARIZE", "TRANSFER", "ASK_MORE_HELP", "END"].includes(h.src)) decisive = h;
  const handed = p.end === "TRANSFERRED";
  const ending = handed ? "Hand-over to a human" + (p.reason ? `: ${D.reason_en[p.reason] || String(p.reason).replace(/_/g, " ")}` : "") : "Completed by the bot";
  let why = "";
  if (decisive) {
    const e = decisive.edge ? E[decisive.edge] : null;
    why = `${actText(decisive).replace(/^The bot moves on by itself$/, "the bot moves on")}${e && e.when !== "always" ? ` and ${e.when}` : ""}, so the bot goes to “${N[decisive.dst]?.label || decisive.dst}”`;
  }
  return { title, ending, why: cap1(why), handed };
}
function hopLine(h) {
  const e = h.edge ? E[h.edge] : null;
  const first = ["auto", "always"].includes(h.on) ? "The bot" : actText(h) + "; the bot";
  const rule = e && e.rules.find(r => r.text);
  return `${esc(first)} moves to <b>${esc(N[h.dst]?.label || h.dst)}</b>`
    + (e && e.when !== "always" ? `, because ${esc(e.when)}` : "")
    + (h.tools.length ? ` <span class="r">(${esc(h.tools.join("; "))})</span>` : "")
    + (rule ? `. <span class="r">Policy ${esc(rule.id)}: “${esc(rule.text)}”</span>` : ".")
    + ` <span class="r">(${esc(h.edge)})</span>`;
}

// ---- side panel
function renderPanel() {
  const P = $("#panel");
  if (!sel) {
    P.innerHTML = `<h2>Explore real calls</h2><p class="d">Click any step on the map to see the ${REAL ? "real call paths that pass through it — the routes the policy oracle takes in our test cases" : "routes the policy allows through it"}, what each one means and how it ends.</p>`
      + `<p class="d">Hover a path to see it on the map; click it to walk through it, step by step, with the policy rule behind every move.</p>`
      + `<p class="d">Hover a step, or a move once it is shown, for everything the policy graph records about it: its rules, tools, limits, conditions and the source quotes.</p>`
      + (REAL ? "" : `<p class="m">Graph-only mode: these are example routes, not test cases.</p>`)
      + (D.left_out && D.left_out.cases ? `<p class="m">Only states and transitions of the policy graph are drawn. ${D.left_out.cases} test case${D.left_out.cases === 1 ? "" : "s"} (${esc(D.left_out.case_ids.join(", "))}) take a move the graph does not have (${esc(D.left_out.moves.join("; "))}), so ${D.left_out.cases === 1 ? "it is" : "they are"} left out.</p>` : "");
    return;
  }
  const n = N[sel];
  let list = D.paths.filter(p => p.hops.some(h => h.src === sel || h.dst === sel));
  const total = list.length;
  if (filter === "human") list = list.filter(p => p.end === "TRANSFERRED"); else if (filter === "done") list = list.filter(p => p.end !== "TRANSFERRED");
  P.innerHTML = `<div class="m">Step</div><h2>${esc(n.label)}</h2>`
    + (n.desc ? `<p class="d">${esc(n.desc)} <span class="m">${esc(n.source)}</span></p>` : "")
    + (n.accepts.length ? `<p class="m">Here the caller may: ${n.accepts.map(s => esc(s.replace(/^The caller /, ""))).join(" · ")}</p>` : "")
    + `<div class="filters" role="group" aria-label="How the call ends">`
    + [["all", "Any ending"], ["done", "Completed by the bot"], ["human", "Hand-over"]].map(([k, t]) => `<button type="button" data-f="${k}" aria-pressed="${filter === k}">${t}</button>`).join("")
    + `</div><p class="m">${list.length}${list.length !== total ? ` of ${total}` : ""} ${REAL ? "real call path" : "example route"}${list.length === 1 ? "" : "s"} through this step</p>`
    + (list.length ? "" : `<p class="d">${total ? "None with this ending." : (REAL ? "No test case reaches this step." : "No route reaches this step.")}</p>`)
    + list.map(p => { const m = meaning(p);
        return `<button class="pcard" type="button" data-p="${p.id}" aria-pressed="${chosen === p.id}"><b>${esc(m.title)}</b>`
          + `<span><i class="dot" style="background:var(${m.handed ? "--human" : "--normal"})"></i>${esc(m.ending)} · ${p.hops.length} moves${REAL ? ` · ${p.cases.length} test case${p.cases.length === 1 ? "" : "s"}` : ""}</span></button>`; }).join("")
    + `<p class="m" style="margin-top:.8rem">Esc clears the map</p>`;
  P.querySelectorAll("[data-f]").forEach(b => b.onclick = () => { filter = b.dataset.f; renderPanel(); });
  P.querySelectorAll(".pcard").forEach(b => {
    b.onmouseenter = () => { preview = b.dataset.p; paint(); };
    b.onmouseleave = () => { preview = null; paint(); };
    b.onclick = () => choose(b.dataset.p);
  });
}
function resetWalk() { stopWalk(); chosen = null; preview = null; at = -1; $("#cap").classList.remove("on"); token.style.opacity = 0; }
function select(id) { resetWalk(); sel = id; filter = "all"; renderPanel(); paint(); }
function clearAll() { resetWalk(); sel = null; renderPanel(); paint(); }

// ---- walking a chosen path
function renderCaption() {
  const p = pathById[chosen]; if (!p) return;
  const m = meaning(p);
  $("#cap").innerHTML = `<h3>${esc(m.title)}</h3><div class="meta"><i class="dot" style="background:var(${m.handed ? "--human" : "--normal"})"></i><b>${esc(m.ending)}.</b> ${esc(m.why)}.`
    + (p.tests.length ? `<br>What the test cases set up: ${esc(p.tests.join("; "))}.` : "")
    + (REAL ? ` <br>Followed by ${p.cases.length} test case${p.cases.length === 1 ? "" : "s"}: ${esc(p.cases.slice(0, 8).join(", "))}${p.cases.length > 8 ? ", …" : ""}` : " <br>An example route the policy allows (not a test case).")
    + `</div><ol>${p.hops.map((h, i) => `<li class="${i === at ? "now" : i < at ? "done" : ""}">${hopLine(h)}</li>`).join("")}</ol>`
    + `<div class="ctrl"><button type="button" id="bk">◀ Back</button><button type="button" id="pl">${timer ? "❚❚ Pause" : "▶ Play"}</button><button type="button" id="nx">Next ▶</button><button type="button" id="cl">Close</button></div>`;
  $("#bk").onclick = () => { stopWalk(); step(-1); }; $("#nx").onclick = () => { stopWalk(); step(1); };
  $("#pl").onclick = playWalk; $("#cl").onclick = () => { resetWalk(); renderPanel(); paint(); };
  $("#cap").classList.add("on");
}
function choose(id) { resetWalk(); chosen = id; renderPanel(); renderCaption(); paint(); playWalk(); }
function step(d) {
  const p = pathById[chosen]; if (!p) return;
  at = Math.max(-1, Math.min(p.hops.length - 1, at + d)); renderCaption(); paint();
  if (at >= 0) animate(edgeEl[hopKey(p.hops[at])]);
  else { const s = pos[p.hops[0].src]; token.style.opacity = 1; token.setAttribute("cx", s.x + nodeW / 2); token.setAttribute("cy", s.y + nodeH / 2); }
}
function animate(path) {
  if (!path) return; token.style.opacity = 1;
  const len = path.getTotalLength(), t0 = performance.now(), dur = matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 750;
  (function f(t) { const k = Math.min(1, (t - t0) / dur), q = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2, pt = path.getPointAtLength(len * q);
    token.setAttribute("cx", pt.x); token.setAttribute("cy", pt.y); if (k < 1) requestAnimationFrame(f); })(t0);
}
function playWalk() {
  if (timer) { stopWalk(); renderCaption(); return; }
  const p = pathById[chosen]; if (!p) return;
  if (at >= p.hops.length - 1) at = -1;
  timer = setInterval(() => { if (at >= p.hops.length - 1) { stopWalk(); renderCaption(); return; } step(1); }, 1700);
  step(1);
}
function stopWalk() { clearInterval(timer); timer = null; }

addEventListener("keydown", ev => {
  if (ev.key === "Escape") clearAll();
  else if (ev.key === "ArrowRight" && chosen) { stopWalk(); step(1); }
  else if (ev.key === "ArrowLeft" && chosen) { stopWalk(); step(-1); }
  else if ((ev.key === "f" || ev.key === "F") && !ev.target.closest("button")) $("#full").click();
});
svg.addEventListener("click", ev => { if (ev.target === svg || ev.target.classList.contains("lane")) clearAll(); });
$("#full").onclick = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.();
$("#theme").onclick = () => { const r = document.documentElement, dark = getComputedStyle(r).colorScheme.includes("dark");
  r.dataset.theme = dark ? "light" : "dark"; try { localStorage.setItem("pg-theme", r.dataset.theme); } catch (e) {} };
try { const t = localStorage.getItem("pg-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
renderPanel(); paint();
</script>
</body>
</html>
"""


def build_page(policy: Path, title: str, cases: str, campaign: Path | None, log=print) -> tuple[str, dict]:
    graph = load_graph(policy)
    paths, kind = [], "example"
    if cases != "none":
        try:
            paths, kind = real_paths(policy, campaign, graph, log), "real"
        except Exception as e:                              # noqa: BLE001 - fall back to graph-only routes
            log(f"could not run the oracle on the base cases ({type(e).__name__}: {str(e)[:160]}); "
                f"showing example routes from the graph instead")
    if not paths:
        paths, kind = example_paths(graph), "example"
    # layout only (nothing is highlighted): the most common completed path runs straight across the top
    done = [p for p in paths if p["end"] != "TRANSFERRED"] or paths
    common = max(done, key=lambda p: (len(p["cases"]), -len(p["hops"])))
    spine = list(dict.fromkeys([common["hops"][0]["src"]] + [h["dst"] for h in common["hops"]]))
    happy = set(spine)
    for p in sorted(done, key=lambda p: -len(p["cases"]))[:4]:
        happy |= {h["src"] for h in p["hops"]} | {h["dst"] for h in p["hops"]}
    happy -= {"TRANSFER"}
    data = {**graph, "title": title, "paths": paths, "paths_kind": kind, "spine": spine, "happy": sorted(happy),
            "act_labels": ACT_LABELS, "reason_en": REASON_EN}
    html = PAGE.replace("__TITLE__", title.replace("<", "&lt;")).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/"))
    return html, data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default=str(ROOT / "gmsjourney" / "policy" / "charge_review_kg.json"))
    ap.add_argument("--campaign", default=None, help="campaign output dir: use its base-case settings (report.json)")
    ap.add_argument("--cases", default="auto", choices=["auto", "none"],
                    help="auto: real paths from the oracle on the base cases; none: graph-only example routes")
    ap.add_argument("--out", default=str(ROOT / "docs" / "POLICY_GRAPH_SHOWCASE.html"))
    ap.add_argument("--title", default="How our charge-review voice bot follows policy")
    a = ap.parse_args()
    html, data = build_page(Path(a.policy), a.title, a.cases, Path(a.campaign) if a.campaign else None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    assert all(h["edge"] for p in data["paths"] for h in p["hops"]), "a shown path uses a move outside the graph"
    print(f"wrote {out} ({len(html) // 1024} KB): {len(data['paths'])} {data['paths_kind']} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
