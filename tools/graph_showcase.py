"""The policy graph as a presentation: one self-contained HTML page for senior leadership and large rooms.

    python tools/graph_showcase.py                                   # -> docs/POLICY_GRAPH_SHOWCASE.html
    python tools/graph_showcase.py --policy path/to/kg.json --out showcase.html --title "Charge review"

What the audience sees
* the conversation as a map: one column per stage of the call, the smooth path running straight across the
  top, recovery moves below it and every hand-over to a human converging on one place;
* headline numbers (states, allowed moves, policy rules, back-end tools), all counted from the graph;
* "stories": pick one (a smooth call, a lost card, a caller who wants a person, ...) and a token walks the
  path while a caption says, in plain English, what the caller did, what the bot does and which policy rule
  makes it do so, quoting the policy document;
* hover any step or move for the detail (rules, what the caller may say there, the tools it uses).

Presenter keys: -> / <- next / previous step, space play / pause, 1-9 pick a story, F full screen.
Everything is read from the policy graph JSON (dialog states, leads_to transitions with their conditions,
rules and source quotes). The page has no external dependencies - it works offline on any laptop.
Edit STORIES / STATE_LABELS / ACT_LABELS below to tell the stories of another policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- the words on the slide
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
}
# each story: (title, one-line pitch, [(from, to, trigger or None), ...]) - every hop is checked against the graph
STORIES = [
    ("A smooth call", "Verified, one charge reviewed and recorded, nothing else needed.",
     [("OPENING", "ASK_AUTH", None), ("ASK_AUTH", "PRESENT_TXN", "PROVIDE_CREDENTIAL"),
      ("PRESENT_TXN", "CONFIRM_RECORDED", "INFORM_DISPOSITION"), ("CONFIRM_RECORDED", "SUMMARIZE", None),
      ("SUMMARIZE", "ASK_MORE_HELP", None), ("ASK_MORE_HELP", "END", "END")]),
    ("“Not mine — send me a new card, fast”", "Fraud reported, expedited replacement with the fee disclosed first.",
     [("OPENING", "ASK_AUTH", None), ("ASK_AUTH", "PRESENT_TXN", "PROVIDE_CREDENTIAL"),
      ("PRESENT_TXN", "CONFIRM_RECORDED", "INFORM_DISPOSITION"), ("CONFIRM_RECORDED", "DISCLOSE_FEE", None),
      ("DISCLOSE_FEE", "CONFIRM_ACTION", None), ("CONFIRM_ACTION", "ACTION_DONE", "CONFIRM"),
      ("ACTION_DONE", "SUMMARIZE", None), ("SUMMARIZE", "ASK_MORE_HELP", None), ("ASK_MORE_HELP", "END", "END")]),
    ("A question in the middle", "The caller asks about delivery fees, the bot answers and returns to the charge.",
     [("OPENING", "ASK_AUTH", None), ("ASK_AUTH", "PRESENT_TXN", "PROVIDE_CREDENTIAL"),
      ("PRESENT_TXN", "ANSWER_KB", "ASK_KB"), ("ANSWER_KB", "PRESENT_TXN", None),
      ("PRESENT_TXN", "CONFIRM_RECORDED", "INFORM_DISPOSITION"), ("CONFIRM_RECORDED", "SUMMARIZE", None),
      ("SUMMARIZE", "ASK_MORE_HELP", None), ("ASK_MORE_HELP", "END", "END")]),
    ("“Let me talk to a person”", "Half-way through, the caller asks for a human: summarise, then hand over.",
     [("OPENING", "ASK_AUTH", None), ("ASK_AUTH", "PRESENT_TXN", "PROVIDE_CREDENTIAL"),
      ("PRESENT_TXN", "CONFIRM_RECORDED", "INFORM_DISPOSITION"), ("CONFIRM_RECORDED", "PRESENT_TXN", None),
      ("PRESENT_TXN", "SUMMARIZE", "REQUEST_HUMAN"), ("SUMMARIZE", "TRANSFER", None), ("TRANSFER", "END", None)]),
    ("Wrong digits, twice", "Two failed verifications: the bot never discusses the account and hands over.",
     [("OPENING", "ASK_AUTH", None), ("ASK_AUTH", "ASK_AUTH_RETRY", "PROVIDE_CREDENTIAL"),
      ("ASK_AUTH_RETRY", "TRANSFER", "PROVIDE_CREDENTIAL"), ("TRANSFER", "END", None)]),
    ("An unsure caller", "“I'm not sure”: the bot gives more detail once, then records the clear answer.",
     [("OPENING", "ASK_AUTH", None), ("ASK_AUTH", "PRESENT_TXN", "PROVIDE_CREDENTIAL"),
      ("PRESENT_TXN", "PROVIDE_CONTEXT", "INFORM_DISPOSITION"),
      ("PROVIDE_CONTEXT", "CONFIRM_RECORDED", "INFORM_DISPOSITION"), ("CONFIRM_RECORDED", "SUMMARIZE", None),
      ("SUMMARIZE", "ASK_MORE_HELP", None), ("ASK_MORE_HELP", "END", "END")]),
]


def build_data(kg_path: Path, title: str) -> dict:
    d = json.loads(kg_path.read_text(encoding="utf-8"))
    T = d["triples"]
    by_head: dict[str, list[dict]] = {}
    for t in T:
        by_head.setdefault(t["head"], []).append(t)
    tails = lambda h, r: [t["tail"] for t in by_head.get(h, []) if t["relation"] == r]
    states = [t["head"] for t in T if t["relation"] == "is_a" and t["tail"] == "dialog_state"]
    states = list(dict.fromkeys(states))
    rules = {t["head"]: t["tail"] for t in T if t["relation"] == "has_text" and t["head"][:1] == "R"}
    tools = sorted({t["head"] for t in T if t["relation"] == "is_a" and t["tail"] == "tool"})
    try:                                                    # the state's own description, when the package is there
        from gmsjourney.knowledge.policy_kg import STATES as SPECS
        desc = {k: (v.evidence, v.source) for k, v in SPECS.items()}
    except Exception:                                       # noqa: BLE001 - optional enrichment
        desc = {}
    nodes = []
    for s in states:
        stage = (tails(s, "in_stage") or ["other"])[0]
        nodes.append({"id": s, "label": STATE_LABELS.get(s, s.replace("_", " ").title()), "stage": stage,
                      "waits": (tails(s, "prompts_customer") or ["no"])[0] == "yes",
                      "accepts": [ACT_LABELS.get(a, a) for a in tails(s, "accepts_act")],
                      "rules": [{"id": r, "text": rules.get(r, "")} for r in tails(s, "applies_rule")],
                      "tools": tails(s, "calls_tool"),
                      "desc": (desc.get(s) or ("", ""))[0], "source": (desc.get(s) or ("", ""))[1]})
    edges = []
    for t in T:
        if t["relation"] != "leads_to":
            continue
        q = t.get("qualifiers") or {}
        edges.append({"id": t["id"], "src": t["head"], "dst": t["tail"], "on": q.get("on") or "always",
                      "on_label": ACT_LABELS.get(q.get("on") or "always", q.get("on")),
                      "when": q.get("when") or "always",
                      "rules": [{"id": r, "text": rules.get(r, "")} for r in q.get("rules") or []],
                      "edges": q.get("edges") or [], "source": t.get("source"), "evidence": t.get("evidence")})
    # stories: resolve every hop to a real transition (exact trigger first); drop a story with a missing hop
    stories, warnings = [], []
    for title_, pitch, hops in STORIES:
        ids = []
        for src, dst, on in hops:
            cand = [e for e in edges if e["src"] == src and e["dst"] == dst]
            e = next((x for x in cand if on and x["on"] == on), cand[0] if cand else None)
            if e is None:
                warnings.append(f"story '{title_}': no transition {src} -> {dst} in the graph; story skipped")
                ids = None
                break
            ids.append(e["id"])
        if ids:
            stories.append({"title": title_, "pitch": pitch, "edges": ids})
    spine = []                                              # the smooth path: the first story's states
    if stories:
        e_by = {e["id"]: e for e in edges}
        first = stories[0]["edges"]
        spine = [e_by[first[0]]["src"]] + [e_by[i]["dst"] for i in first]
    happy = set(spine)
    for s in stories[1:2]:                                  # the second story is a "good" path too
        e_by = {e["id"]: e for e in edges}
        happy |= {e_by[i]["src"] for i in s["edges"]} | {e_by[i]["dst"] for i in s["edges"]}
    stage_order = [s for s in STAGE_ORDER if any(n["stage"] == s for n in nodes)]
    stage_order += sorted({n["stage"] for n in nodes} - set(stage_order))
    sources = sorted({e["source"] for e in edges if e.get("source")} |
                     {t.get("source") for t in T if t.get("source", "").startswith("pdf")})
    return {
        "title": title, "nodes": nodes, "edges": edges, "stories": stories, "spine": spine,
        "happy": sorted(happy), "stages": [{"id": s, "label": STAGE_LABELS.get(s, s.replace("_", " ").title())}
                                           for s in stage_order],
        "stats": {"states": len(nodes), "moves": len(edges), "rules": len(rules), "tools": len(tools),
                  "sections": len({s for s in sources if s and s.startswith("pdf")}),
                  "to_human": sum(1 for e in edges if e["dst"] == "TRANSFER")},
        "warnings": warnings,
    }


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
--normal:#2a78d6;--recovery:#1baf7a;--human:#e34948;--node:#ffffff;--node-hi:#eef4fd;--glow:#d48a00}
@media (prefers-color-scheme:light){:root:not([data-theme="dark"]){color-scheme:light;--bg:#f4f5f7;--surface:#fcfcfb;--surface-2:#eef0f3;--line:#d5dae1;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6b7280;
--normal:#2a78d6;--recovery:#1baf7a;--human:#e34948;--node:#ffffff;--node-hi:#eef4fd;--glow:#d48a00}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font)}
body{padding:clamp(12px,2vw,28px) 16px}
.wrap{max-width:1700px;margin:0 auto}
header{display:flex;flex-wrap:wrap;gap:1rem 2rem;align-items:flex-end;justify-content:space-between}
h1{margin:0;font-size:clamp(1.6rem,3.2vw,2.8rem);font-weight:700;letter-spacing:-.01em;line-height:1.1}
.sub{color:var(--ink-2);font-size:clamp(.95rem,1.3vw,1.2rem);margin:.35rem 0 0;max-width:70ch}
.tools{display:flex;gap:.5rem}
button{font:inherit;cursor:pointer;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:.45rem .85rem}
button:hover{border-color:var(--normal)}
button:focus-visible{outline:2px solid var(--glow);outline-offset:2px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:1.1rem 0}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem}
.stat b{display:block;font-size:clamp(1.8rem,3.4vw,2.9rem);line-height:1;font-variant-numeric:tabular-nums}
.stat span{color:var(--ink-2);font-size:.95rem}
.stories{display:flex;flex-wrap:wrap;gap:.5rem;margin:.4rem 0 .8rem}
.story{text-align:left;max-width:260px}
.story small{display:block;color:var(--muted);font-size:.78rem;margin-top:.1rem}
.story[aria-pressed="true"]{border-color:var(--glow);box-shadow:0 0 0 1px var(--glow) inset}
.stage-wrap{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden}
svg{display:block;width:100%;height:auto}
.lane-t{fill:var(--muted);font-size:14px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.lane{fill:var(--surface-2);opacity:.55}
.edge{fill:none;stroke-width:2;opacity:.22;transition:opacity .25s,stroke-width .25s}
.edge.normal{stroke:var(--normal)} .edge.recovery{stroke:var(--recovery);stroke-dasharray:7 6} .edge.human{stroke:var(--human);stroke-width:2.6}
.edge.hit{stroke:transparent;stroke-width:16;opacity:1;fill:none;cursor:help}
.edge.on{opacity:1;stroke-width:4}
.edge.dim{opacity:.06}
.node rect{fill:var(--node);stroke:var(--line);stroke-width:1.5;transition:fill .25s,stroke .25s}
.node.waits rect{stroke-width:2}
.node:not(.waits) rect{stroke-dasharray:5 4}
.node.happy rect{stroke:var(--normal)}
.node.human rect{stroke:var(--human)}
.node text{fill:var(--ink);font-size:16px;font-weight:600;pointer-events:none}
.node .code{fill:var(--muted);font-size:11px;font-weight:500;font-family:ui-monospace,Consolas,monospace}
.node{cursor:help}
.node.visited rect{fill:var(--node-hi);stroke:var(--glow);stroke-width:2.5;stroke-dasharray:none}
.node.dim{opacity:.35}
.token{fill:var(--glow);filter:drop-shadow(0 0 8px var(--glow))}
.caption{margin-top:12px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:1rem 1.2rem;min-height:7.5rem;display:grid;gap:.35rem}
.caption .step{color:var(--muted);font-size:.9rem;letter-spacing:.04em;text-transform:uppercase}
.caption .what{font-size:clamp(1.15rem,1.9vw,1.65rem);font-weight:600;line-height:1.3}
.caption .why{color:var(--ink-2);font-size:clamp(.95rem,1.25vw,1.15rem);line-height:1.45}
.caption .why q{font-style:italic}
.controls{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin-top:.6rem}
.legend{display:flex;flex-wrap:wrap;gap:1rem 1.4rem;color:var(--ink-2);font-size:.9rem;margin:.7rem 0 0}
.legend i{display:inline-block;width:28px;height:0;border-top:3px solid;vertical-align:middle;margin-right:.4rem}
.tip{position:fixed;z-index:10;max-width:380px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.7rem .85rem;font-size:.9rem;line-height:1.45;box-shadow:0 10px 30px rgba(0,0,0,.35);pointer-events:none;display:none}
.tip b{display:block;font-size:1rem;margin-bottom:.2rem}
.tip .m{color:var(--muted);font-size:.8rem}
.foot{color:var(--muted);font-size:.85rem;margin-top:1rem}
@media (prefers-reduced-motion:reduce){.edge,.node rect{transition:none}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 id="title"></h1>
      <p class="sub">Every move the voice bot is allowed to make, drawn from the policy document. Pick a story to watch a call travel through it.</p>
    </div>
    <div class="tools">
      <button id="theme" type="button" title="Light / dark">Light / dark</button>
      <button id="full" type="button" title="Full screen (F)">Full screen</button>
    </div>
  </header>
  <div class="stats" id="stats"></div>
  <div class="stories" id="stories" role="group" aria-label="Stories"></div>
  <div class="stage-wrap"><svg id="map" role="img" aria-label="Policy graph of the voice bot"></svg></div>
  <div class="caption" aria-live="polite">
    <div class="step" id="cap-step"></div>
    <div class="what" id="cap-what"></div>
    <div class="why" id="cap-why"></div>
  </div>
  <div class="controls">
    <button id="prev" type="button">&#9664; Back</button>
    <button id="play" type="button">&#9654; Play</button>
    <button id="next" type="button">Next &#9654;</button>
    <button id="all" type="button">Show every move</button>
  </div>
  <div class="legend">
    <span><i style="border-color:var(--normal)"></i>Normal flow</span>
    <span><i style="border-color:var(--recovery);border-top-style:dashed"></i>Recovery (retry, clarify, explain)</span>
    <span><i style="border-color:var(--human);border-top-width:4px"></i>Hand-over to a human</span>
    <span>Solid box: the bot waits for the caller &nbsp;·&nbsp; dashed box: the bot acts on its own</span>
  </div>
  <p class="foot" id="foot"></p>
</div>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("#title").textContent = D.title;
document.title = D.title;

// ---- headline numbers
const S = D.stats;
$("#stats").innerHTML = [
  [S.states, "conversation steps"], [S.moves, "allowed moves between them"], [S.rules, "policy rules enforced"],
  [S.tools, "back-end systems called"], [S.to_human, "safe exits to a human"]
].map(([n, t]) => `<div class="stat"><b>${n}</b><span>${t}</span></div>`).join("");
$("#foot").textContent = `Built from the policy graph: ${S.moves} transitions, each with its condition, the rules behind it and the policy text it came from (${S.sections} sections of the operating procedure).`
  + (D.warnings.length ? "  Note: " + D.warnings.join("; ") : "");

// ---- layout: one column per stage; the smooth path runs straight across the top
// a stage gets one sub-column per smooth-path step it holds, so the smooth path never doubles back
const spine = new Set(D.spine), happy = new Set(D.happy);
const spineIn = {}; D.spine.forEach(id => { const st = D.nodes.find(n => n.id === id).stage; (spineIn[st] = spineIn[st] || []).push(id); });
const units = D.stages.map(s => Math.max(1, (spineIn[s.id] || []).length));
const W = 1680, unitW = W / units.reduce((a, b) => a + b, 0), nodeW = Math.min(178, unitW - 14), nodeH = 54, top0 = 86, rowH = 92;
const stageX = {}, stageW = {}; let acc = 0;
D.stages.forEach((s, i) => { stageX[s.id] = acc; stageW[s.id] = units[i] * unitW; acc += units[i] * unitW; });
const col = {}; D.stages.forEach((s, i) => col[s.id] = i);
const rowsUsed = {}, pos = {};
const ordered = [...D.nodes].sort((a, b) => (spine.has(b.id) - spine.has(a.id)) || (happy.has(b.id) - happy.has(a.id)));
for (const n of ordered) {
  const c = col[n.stage] ?? 0;
  let r, x;
  if (spine.has(n.id)) {
    r = 0; const k = spineIn[n.stage].indexOf(n.id);
    x = stageX[n.stage] + k * unitW + (unitW - nodeW) / 2;
  } else {
    r = n.id === "TRANSFER" ? 5 : Math.max(1, rowsUsed[c] ?? 1);
    if (n.id !== "TRANSFER") rowsUsed[c] = r + 1;
    x = stageX[n.stage] + (stageW[n.stage] - nodeW) / 2;
  }
  pos[n.id] = { x, y: top0 + r * rowH, c: x / unitW, r };
}
const maxRow = Math.max(...Object.values(pos).map(p => p.r));
const H = top0 + (maxRow + 1) * rowH + 20;
const svg = $("#map");
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
const NS = "http://www.w3.org/2000/svg";
function fit(t, max) {                                   // shrink a label until it fits its box
  let size = parseFloat(getComputedStyle(t).fontSize) || 16;
  while (t.getComputedTextLength() > max && size > 9) { size -= 0.5; t.style.fontSize = size + "px"; }
}
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };

D.stages.forEach((s, i) => {
  if (i % 2 === 0) el("rect", { x: stageX[s.id], y: 0, width: stageW[s.id], height: H, class: "lane" });
  const t = el("text", { x: stageX[s.id] + stageW[s.id] / 2, y: 36, "text-anchor": "middle", class: "lane-t" }); t.textContent = s.label;
  fit(t, stageW[s.id] - 22);
});

const kind = e => e.dst === "TRANSFER" ? "human" : ((happy.has(e.src) && happy.has(e.dst)) ? "normal" : "recovery");
function pathFor(e) {
  const a = pos[e.src], b = pos[e.dst];
  if (!a || !b) return "";
  if (e.src === e.dst) {                                  // a loop: a small arc above the box
    const x = a.x + nodeW * .7, y = a.y;
    return `M${x - 22},${y} C${x - 30},${y - 46} ${x + 30},${y - 46} ${x + 22},${y}`;
  }
  if (b.x > a.x + 1) {                                        // forward: right side to left side
    const x1 = a.x + nodeW, y1 = a.y + nodeH / 2, x2 = b.x, y2 = b.y + nodeH / 2, dx = (x2 - x1) * .45;
    return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
  }
  if (Math.abs(b.x - a.x) <= 1) {                                      // same stage: an arc on the right
    const x1 = a.x + nodeW, y1 = a.y + nodeH / 2, x2 = b.x + nodeW, y2 = b.y + nodeH / 2, bulge = 36 + Math.abs(y2 - y1) * .15;
    return `M${x1},${y1} C${x1 + bulge},${y1} ${x2 + bulge},${y2} ${x2},${y2}`;
  }
  const x1 = a.x + nodeW / 2, y1 = a.y, x2 = b.x + nodeW / 2, y2 = b.y, lift = Math.min(34, 22 + (a.x - b.x) / unitW * 4);   // backward: over the top
  return `M${x1},${y1} C${x1},${y1 - lift} ${x2},${y2 - lift} ${x2},${y2}`;
}
const gEdges = el("g", {}), gHits = el("g", {}), gNodes = el("g", {});
const edgeEl = {};
for (const e of D.edges) {
  const d = pathFor(e); if (!d) continue;
  edgeEl[e.id] = el("path", { d, class: "edge " + kind(e) }, gEdges);
  const hit = el("path", { d, class: "edge hit" }, gHits);
  hit.addEventListener("mousemove", ev => tipEdge(ev, e)); hit.addEventListener("mouseleave", hideTip);
}
const nodeEl = {};
for (const n of D.nodes) {
  const p = pos[n.id];
  const g = el("g", { class: "node" + (n.waits ? " waits" : "") + (happy.has(n.id) ? " happy" : "") + (n.id === "TRANSFER" ? " human" : ""),
                       transform: `translate(${p.x},${p.y})`, tabindex: 0, role: "button", "aria-label": n.label }, gNodes);
  el("rect", { width: nodeW, height: nodeH, rx: 12 }, g);
  const t = el("text", { x: nodeW / 2, y: 24, "text-anchor": "middle" }, g); t.textContent = n.label; fit(t, nodeW - 14);
  const c = el("text", { x: nodeW / 2, y: 42, "text-anchor": "middle", class: "code" }, g); c.textContent = n.id; fit(c, nodeW - 14);
  g.addEventListener("mousemove", ev => tipNode(ev, n)); g.addEventListener("mouseleave", hideTip);
  g.addEventListener("focus", ev => { const r = g.getBoundingClientRect(); tipNode({ clientX: r.right, clientY: r.top }, n); });
  g.addEventListener("blur", hideTip);
  nodeEl[n.id] = g;
}
const token = el("circle", { r: 11, class: "token", cx: -50, cy: -50 });

// ---- tooltips
const tip = $("#tip");
function showTip(ev, html) { tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(ev.clientX + 16, innerWidth - tip.offsetWidth - 8), y = Math.min(ev.clientY + 16, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px"; }
function hideTip() { tip.style.display = "none"; }
const rulesHtml = rs => rs.filter(r => r.text).map(r => `<div><b style="display:inline;font-size:.85rem">${esc(r.id)}</b> ${esc(r.text)}</div>`).join("");
function tipNode(ev, n) {
  showTip(ev, `<b>${esc(n.label)}</b><div class="m">${esc(n.id)} · ${n.waits ? "the bot waits for the caller" : "the bot acts on its own"}</div>`
    + (n.desc ? `<div style="margin:.35rem 0">“${esc(n.desc)}” <span class="m">${esc(n.source)}</span></div>` : "")
    + (n.accepts.length ? `<div class="m" style="margin-top:.3rem">The caller may:</div><div>${n.accepts.map(esc).join(" · ")}</div>` : "")
    + (n.tools.length ? `<div class="m" style="margin-top:.3rem">Systems used next:</div><div>${n.tools.map(esc).join(", ")}</div>` : "")
    + (n.rules.length ? `<div class="m" style="margin-top:.3rem">Rules in force here:</div>${rulesHtml(n.rules.slice(0, 5))}` : ""));
}
function tipEdge(ev, e) {
  const a = D.nodes.find(n => n.id === e.src), b = D.nodes.find(n => n.id === e.dst);
  showTip(ev, `<b>${esc(a.label)} → ${esc(b.label)}</b><div>${esc(e.on_label)}${e.when && e.when !== "always" ? " — when " + esc(e.when) : ""}</div>`
    + rulesHtml(e.rules) + (e.evidence ? `<div class="m" style="margin-top:.3rem">Policy ${esc(e.source)}: “${esc(e.evidence)}”</div>` : ""));
}

// ---- stories
let story = 0, step = -1, timer = null, showAll = false;
const E = Object.fromEntries(D.edges.map(e => [e.id, e]));
const N = Object.fromEntries(D.nodes.map(n => [n.id, n]));
$("#stories").innerHTML = D.stories.map((s, i) =>
  `<button class="story" type="button" data-i="${i}" aria-pressed="${i === 0}"><b>${i + 1}. ${esc(s.title)}</b><small>${esc(s.pitch)}</small></button>`).join("");
document.querySelectorAll(".story").forEach(b => b.addEventListener("click", () => pick(+b.dataset.i)));

function render() {
  const s = D.stories[story], ids = s.edges.slice(0, step + 1);
  const visitedNodes = new Set([E[s.edges[0]].src, ...ids.map(i => E[i].dst)]);
  const inStory = new Set(s.edges);
  for (const id in edgeEl) {
    edgeEl[id].classList.toggle("on", ids.includes(id));
    edgeEl[id].classList.toggle("dim", !showAll && !inStory.has(id));
  }
  for (const id in nodeEl) {
    nodeEl[id].classList.toggle("visited", step >= 0 ? visitedNodes.has(id) : id === E[s.edges[0]].src);
    nodeEl[id].classList.toggle("dim", !showAll && !s.edges.some(i => E[i].src === id || E[i].dst === id));
  }
  if (step < 0) {
    const p = pos[E[s.edges[0]].src];
    token.setAttribute("cx", p.x + nodeW / 2); token.setAttribute("cy", p.y + nodeH / 2);
    $("#cap-step").textContent = `Story ${story + 1} of ${D.stories.length} · ${s.edges.length} moves`;
    $("#cap-what").textContent = s.title;
    $("#cap-why").innerHTML = esc(s.pitch) + " &nbsp;<span style='color:var(--muted)'>Press ▶ Play or → to walk through it.</span>";
    return;
  }
  const e = E[s.edges[step]];
  const what = (e.on === "auto" || e.on === "always" ? "The bot" : esc(e.on_label) + " → the bot") + " moves to <b>" + esc(N[e.dst].label) + "</b>";
  const rule = e.rules.find(r => r.text);
  $("#cap-step").textContent = `${s.title} · step ${step + 1} of ${s.edges.length}`;
  $("#cap-what").innerHTML = what + ".";
  $("#cap-why").innerHTML = (e.when && e.when !== "always" ? "Because " + esc(e.when) + ". " : "")
    + (rule ? `Policy rule <b>${esc(rule.id)}</b>: <q>${esc(rule.text)}</q> ` : "")
    + (e.evidence && !(rule && rule.text.replace(/\W/g, "").includes(e.evidence.replace(/\W/g, "")))
       ? `<span style="color:var(--muted)">(policy ${esc(e.source)}: “${esc(e.evidence)}”)</span>`
       : (e.source ? `<span style="color:var(--muted)">(policy ${esc(e.source)})</span>` : ""));
  animate(e);
}
function animate(e) {
  const path = edgeEl[e.id]; if (!path) return;
  const len = path.getTotalLength(), t0 = performance.now(), dur = matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 900;
  (function f(t) { const k = Math.min(1, (t - t0) / dur), pt = path.getPointAtLength(len * (k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2));
    token.setAttribute("cx", pt.x); token.setAttribute("cy", pt.y); if (k < 1) requestAnimationFrame(f); })(t0);
}
function pick(i) { story = i; step = -1; stop(); document.querySelectorAll(".story").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.i === i))); render(); }
function next() { const s = D.stories[story]; if (step < s.edges.length - 1) { step++; render(); } else stop(); }
function prev() { if (step > -1) { step--; render(); } }
function play() { if (timer) return stop(); if (step >= D.stories[story].edges.length - 1) step = -1;
  $("#play").innerHTML = "&#10073;&#10073; Pause"; next(); timer = setInterval(() => { next(); if (step >= D.stories[story].edges.length - 1) stop(); }, 2600); }
function stop() { clearInterval(timer); timer = null; $("#play").innerHTML = "&#9654; Play"; }
$("#next").onclick = () => { stop(); next(); };
$("#prev").onclick = () => { stop(); prev(); };
$("#play").onclick = play;
$("#all").onclick = () => { showAll = !showAll; $("#all").textContent = showAll ? "Focus on the story" : "Show every move"; render(); };
$("#full").onclick = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.();
$("#theme").onclick = () => { const r = document.documentElement, dark = getComputedStyle(r).colorScheme.includes("dark");
  r.dataset.theme = dark ? "light" : "dark"; try { localStorage.setItem("pg-theme", r.dataset.theme); } catch (e) {} };
try { const t = localStorage.getItem("pg-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
addEventListener("keydown", ev => {
  if (ev.target.closest && ev.target.closest("input,textarea")) return;
  if (ev.key === "ArrowRight") { stop(); next(); } else if (ev.key === "ArrowLeft") { stop(); prev(); }
  else if (ev.key === " ") { ev.preventDefault(); play(); } else if (ev.key.toLowerCase() === "f") $("#full").click();
  else if (/^[1-9]$/.test(ev.key) && +ev.key <= D.stories.length) pick(+ev.key - 1);
});
render();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default=str(ROOT / "gmsjourney" / "policy" / "charge_review_kg.json"),
                    help="the policy graph JSON (dialog states + leads_to transitions)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "POLICY_GRAPH_SHOWCASE.html"))
    ap.add_argument("--title", default="How our charge-review voice bot follows policy")
    a = ap.parse_args()
    data = build_data(Path(a.policy), a.title)
    html = PAGE.replace("__TITLE__", a.title.replace("<", "&lt;")).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    for w in data["warnings"]:
        print("warning:", w)
    s = data["stats"]
    print(f"wrote {out} ({len(html) // 1024} KB): {s['states']} states, {s['moves']} moves, {s['rules']} rules, "
          f"{len(data['stories'])} stories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
