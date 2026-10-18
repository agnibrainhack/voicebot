"""The state-transition graph of a policy graph file (charge_review_kg.json) as one self-contained HTML page.

    python tools/graph_showcase.py                       # -> docs/POLICY_GRAPH_SHOWCASE.html
    python tools/graph_showcase.py --policy other_kg.json --out other.html

Everything on the page comes from the graph file and nothing else: no store, no oracle, no test cases, no
rewording. The states are the heads of `is_a dialog_state` triples and the arrows are the `leads_to` triples,
drawn with their original ids and labelled with their `on` qualifier, exactly as written. Colours and line
styles only sort the arrows by that `on` value (a customer act, `auto`, `any`); boxes are solid or dashed by the
state's `prompts_customer` value.

The details panel on the right shows, on hover (click to pin, Esc to unpin):
* a state: every triple whose head or tail is that state, every field of each triple as stored (id, relation,
  tail, tail_type, kind, state, source, evidence, gms, qualifiers), the triples of the entities it points to
  (rules, limits, tools, acts, ...) and the text of every source section it cites;
* a transition: every field and qualifier of its `leads_to` triple, the triples of the rules it names, the
  workflow `enables` triples whose edge it names, and the text of its source section;
* nothing selected: the file's own header (title, schema, source document, checksums), the relation
  descriptions and how many triples there are of each relation.
The layout is computed by Graphviz (dot) when it is installed (`apt install graphviz` / `conda install graphviz`),
Wheel scrolls down the flow, Ctrl/⌘ + wheel zooms, drag pans, F for full screen.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_graph(kg_path: Path) -> dict:
    d = json.loads(kg_path.read_text(encoding="utf-8"))
    T = d["triples"]
    first = lambda h, r: next((t["tail"] for t in T if t["head"] == h and t["relation"] == r), None)
    states = list(dict.fromkeys(t["head"] for t in T if t["relation"] == "is_a" and t["tail"] == "dialog_state"))
    nodes = [{"id": s, "stage": first(s, "in_stage"), "prompts_customer": first(s, "prompts_customer")} for s in states]
    edges = [{"id": t["id"], "src": t["head"], "dst": t["tail"], "on": (t.get("qualifiers") or {}).get("on")}
             for t in T if t["relation"] == "leads_to"]
    stages = list(dict.fromkeys(n["stage"] for n in nodes))
    header = {k: v for k, v in d.items() if k != "triples"}
    return {"nodes": nodes, "edges": edges, "stages": stages, "triples": T, "header": header,
            "file": kg_path.name}


def dot_layout(graph: dict) -> dict | None:
    """Graphviz (dot) positions: node centres, each transition's spline and label point, in SVG units (y down).
    None when dot is not installed or fails (the page then lays the states out by stage itself)."""
    if not shutil.which("dot"):
        return None
    q = lambda x: '"' + str(x).replace('"', '\\"') + '"'
    lines = ["digraph G {", "rankdir=TB; nodesep=0.3; ranksep=0.55; splines=true; forcelabels=true;",
             'node [shape=box fixedsize=true width=2.35 height=0.5 fontname="Helvetica" fontsize=14];',
             'edge [fontname="Helvetica" fontsize=11 arrowsize=0.8];']
    lines += [f"{q(n['id'])};" for n in graph["nodes"]]
    lines += [f"{q(e['src'])} -> {q(e['dst'])} [id={q(e['id'])} xlabel={q(e['on'])}];" for e in graph["edges"]]
    lines.append("}")
    try:
        g = json.loads(subprocess.run(["dot", "-Tjson"], input="\n".join(lines), capture_output=True, text=True,
                                      timeout=60, check=True).stdout)
    except Exception:                                       # noqa: BLE001 - fall back to the built-in layout
        return None
    x0, y0, x1, y1 = (float(v) for v in g["bb"].split(","))
    pad = 24
    tx = lambda x: float(x) - x0 + pad
    ty = lambda y: (y1 - float(y)) + pad
    nodes = {o["name"]: {"cx": tx(o["pos"].split(",")[0]), "cy": ty(o["pos"].split(",")[1])}
             for o in g.get("objects", []) if "pos" in o}
    edges = {}
    for e in g.get("edges", []):
        pts, end = [], None
        for tok in e["pos"].split():
            if tok.startswith("e,"):
                end = tok[2:].split(",")
            elif not tok.startswith("s,"):
                pts.append(tok.split(","))
        P = [(tx(a), ty(b)) for a, b in pts]
        d = f"M{P[0][0]:.1f},{P[0][1]:.1f}" + "".join(
            f" C{P[k][0]:.1f},{P[k][1]:.1f} {P[k + 1][0]:.1f},{P[k + 1][1]:.1f} {P[k + 2][0]:.1f},{P[k + 2][1]:.1f}"
            for k in range(1, len(P) - 2, 3))
        if end:                                             # dot stops the spline short of the arrow head
            d += f" L{tx(end[0]):.1f},{ty(end[1]):.1f}"
        lx, ly = (e.get("xlp") or e.get("lp") or "0,0").split(",")
        edges[e["id"]] = {"d": d, "lx": tx(lx), "ly": ty(ly)}
    if len(nodes) != len(graph["nodes"]) or len(edges) != len(graph["edges"]):
        return None
    return {"w": x1 - x0 + 2 * pad, "h": y1 - y0 + 2 * pad, "node_w": 2.35 * 72, "node_h": 0.5 * 72,
            "nodes": nodes, "edges": edges, "engine": "graphviz"}


def layered_layout(graph: dict, node_w: float = 210, node_h: float = 52, node_gap: float = 90,
                   rank_gap: float = 78, lane_gap: float = 13) -> dict:
    """A layered (Sugiyama-style) top-to-bottom layout in plain Python, so the page looks the same on every
    machine, with or without Graphviz. Steps: break cycles with a depth-first search from the first state
    (back arrows are drawn upwards), rank every state by its longest path from the start, give every arrow
    that spans several rows a lane point on each row it crosses, order each row to reduce crossings
    (barycentre sweeps, best of 40), place rows so that connected states line up, then draw each arrow as a
    smooth curve through its lane points, with arrow ends spread along the box edges.
    node_gap: space between boxes in a row; rank_gap: space between rows; lane_gap: space between arrow lanes."""
    ids = [n["id"] for n in graph["nodes"]]
    E = [(e["id"], e["src"], e["dst"]) for e in graph["edges"]]
    succ = {v: [] for v in ids}
    for _, a, b in E:
        if a != b and b not in succ[a]:
            succ[a].append(b)
    state, back = {}, set()
    for root in ids:                                        # iterative DFS: arrows to a state on the stack are back
        if root in state:
            continue
        stack = [(root, iter(succ[root]))]
        state[root] = 1
        while stack:
            v, it = stack[-1]
            w = next(it, None)
            if w is None:
                state[v] = 2
                stack.pop()
            elif state.get(w) == 1:
                back.add((v, w))
            elif w not in state:
                state[w] = 1
                stack.append((w, iter(succ[w])))
    dag = [(eid, b, a, True) if (a, b) in back else (eid, a, b, False) for eid, a, b in E if a != b]
    indeg, ch = {v: 0 for v in ids}, {v: [] for v in ids}
    for _, a, b, _ in dag:
        indeg[b] += 1
        ch[a].append(b)
    rank, queue = {v: 0 for v in ids}, [v for v in ids if indeg[v] == 0]
    while queue:
        v = queue.pop(0)
        for w in ch[v]:
            rank[w] = max(rank[w], rank[v] + 1)
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    for v in ids:                                           # a state with no way in sits just above its children
        kids = [rank[w] for w in ch[v]]
        if kids and not any(b == v for _, _, b, _ in dag):
            rank[v] = min(kids) - 1
    R = max(rank.values())
    layers = {r: [] for r in range(R + 1)}
    for v in ids:
        layers[rank[v]].append(v)
    chains, width, lane_row = {}, {v: node_w for v in ids}, {}
    for eid, a, b, rev in dag:                              # lane points for arrows crossing rows
        chain = [a]
        for r in range(rank[a] + 1, rank[b]):
            d = f"{eid}#{r}"
            layers[r].append(d)
            width[d] = 0
            lane_row[d] = r
            chain.append(d)
        chain.append(b)
        chains[eid] = (chain, rev)
    up, down = {}, {}
    for chain, _ in chains.values():
        for u, w in zip(chain, chain[1:]):
            down.setdefault(u, []).append(w)
            up.setdefault(w, []).append(u)

    def crossings(order):
        pos = {v: i for r in order for i, v in enumerate(order[r])}
        n = 0
        for r in range(R):
            segs = [(pos[u], pos[w]) for u in order[r] for w in down.get(u, [])]
            n += sum(1 for i, (a1, b1) in enumerate(segs) for a2, b2 in segs[i + 1:] if (a1 - a2) * (b1 - b2) < 0)
        return n
    order = {r: list(layers[r]) for r in layers}
    best, best_n = {r: list(o) for r, o in order.items()}, crossings(order)
    for it in range(40):
        pos = {v: i for r in order for i, v in enumerate(order[r])}
        rows = range(1, R + 1) if it % 2 == 0 else range(R - 1, -1, -1)
        nb = up if it % 2 == 0 else down
        for r in rows:
            def key(v):
                ps = [pos[u] for u in nb.get(v, [])]
                return (sum(ps) / len(ps)) if ps else pos[v]
            order[r].sort(key=key)
            pos.update({v: i for i, v in enumerate(order[r])})
        n = crossings(order)
        if n < best_n:
            best, best_n = {r: list(o) for r, o in order.items()}, n
    order = best

    def sep(u, w):                                          # centre-to-centre distance between neighbours in a row
        return (width[u] + width[w]) / 2 + (node_gap if width[u] and width[w] else lane_gap)
    x = {}
    for r in order:
        cx = 0.0
        for i, v in enumerate(order[r]):
            cx = cx if i == 0 else cx + sep(order[r][i - 1], v)
            x[v] = cx

    def pack(row, want):
        n = len(row)
        a = list(want)
        for i in range(1, n):
            a[i] = max(a[i], a[i - 1] + sep(row[i - 1], row[i]))
        b = list(want)
        for i in range(n - 2, -1, -1):
            b[i] = min(b[i], b[i + 1] - sep(row[i], row[i + 1]))
        c = [(p + q) / 2 for p, q in zip(a, b)]
        for i in range(1, n):
            c[i] = max(c[i], c[i - 1] + sep(row[i - 1], row[i]))
        return c
    for it in range(16):                                    # line connected states up, row by row, both ways
        rows = range(1, R + 1) if it % 2 == 0 else range(R - 1, -1, -1)
        nb = up if it % 2 == 0 else down
        for r in rows:
            row = order[r]
            want = []
            for v in row:
                ps = [x[u] for u in nb.get(v, [])]
                want.append(sum(ps) / len(ps) if ps else x[v])
            for v, nx in zip(row, pack(row, want)):
                x[v] = nx
    pad = 40
    minx = min(x[v] - width[v] / 2 for v in x)
    for v in x:
        x[v] = x[v] - minx + pad
    row_y = lambda r: pad + r * (node_h + rank_gap) + node_h / 2
    y = {v: row_y(lane_row[v] if v in lane_row else rank[v]) for v in x}
    # arrow ends: spread along the bottom (out) and top (in) of each box, ordered by where the arrow goes
    ports = {}
    for eid, (chain, rev) in chains.items():
        a, b = chain[0], chain[-1]
        ports.setdefault((a, "bot"), []).append((x[chain[1]], eid))
        ports.setdefault((b, "top"), []).append((x[chain[-2]], eid))
    port_x = {}
    for (v, side), lst in ports.items():
        lst.sort()
        n = len(lst)
        for i, (_, eid) in enumerate(lst):
            port_x[(eid, side)] = x[v] + (0 if n == 1 else (i / (n - 1) - .5) * node_w * .78)
    edges, loops = {}, {}
    for eid, a, b in E:
        if a == b:                                          # a state's arrow to itself: a loop on its right side
            k = loops[a] = loops.get(a, -1) + 1
            rx, cy = x[a] + node_w / 2, y[a]
            d = (f"M{rx:.1f},{cy - 9:.1f} C{rx + 46 + 14 * k:.1f},{cy - 34 - 6 * k:.1f} "
                 f"{rx + 46 + 14 * k:.1f},{cy + 34 + 6 * k:.1f} {rx:.1f},{cy + 9:.1f}")
            edges[eid] = {"d": d}
            continue
        chain, rev = chains[eid]
        pts = [(port_x[(eid, "bot")], y[chain[0]] + node_h / 2)] + [(x[d], y[d]) for d in chain[1:-1]] + \
              [(port_x[(eid, "top")], y[chain[-1]] - node_h / 2)]
        if rev:                                             # a back arrow is drawn from its real head upwards
            pts = pts[::-1]
        d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            ym = (y0 + y1) / 2
            d += f" C{x0:.1f},{ym:.1f} {x1:.1f},{ym:.1f} {x1:.1f},{y1:.1f}"
        edges[eid] = {"d": d}
    W = max(x[v] + width[v] / 2 for v in x) + pad + 60
    H = pad * 2 + (R + 1) * node_h + R * rank_gap
    return {"w": W, "h": H, "node_w": node_w, "node_h": node_h,
            "nodes": {v: {"cx": x[v], "cy": y[v]} for v in ids}, "edges": edges, "engine": "built-in"}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0f1318;--surface:#161b22;--surface-2:#1d242d;--line:#2b3440;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#8e98a4;
--act:#3987e5;--auto:#199e70;--any:#d95926;--node:#1f2731;--glow:#ffd36e;--mono:ui-monospace,Consolas,"SFMono-Regular",monospace;
--font:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif}
:root[data-theme="light"]{color-scheme:light;--bg:#f4f5f7;--surface:#fcfcfb;--surface-2:#eef0f3;--line:#d5dae1;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6b7280;
--act:#2a78d6;--auto:#1baf7a;--any:#eb6834;--node:#ffffff;--glow:#b86e00}
@media (prefers-color-scheme:light){:root:not([data-theme="dark"]){color-scheme:light;--bg:#f4f5f7;--surface:#fcfcfb;--surface-2:#eef0f3;--line:#d5dae1;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6b7280;
--act:#2a78d6;--auto:#1baf7a;--any:#eb6834;--node:#ffffff;--glow:#b86e00}}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font-family:var(--font)}
body{display:flex;flex-direction:column;padding:10px 16px 12px;gap:8px}
header{display:flex;flex-wrap:wrap;gap:.4rem 1.5rem;align-items:baseline;justify-content:space-between}
h1{margin:0;font-size:clamp(1.15rem,1.9vw,1.6rem);font-weight:700;letter-spacing:-.01em}
.facts{color:var(--muted);font-size:.86rem;font-family:var(--mono)}
.tools{display:flex;gap:.4rem;margin-left:auto}
button{font:inherit;font-size:.86rem;cursor:pointer;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:.3rem .65rem}
button:hover{border-color:var(--act)}
button:focus-visible,.node:focus-visible{outline:2px solid var(--glow);outline-offset:2px}
.main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) clamp(320px,27vw,460px);gap:12px}
@media (max-width:900px){.main{grid-template-columns:1fr;grid-template-rows:70vh auto}}
.map{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;min-height:0}
svg{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
svg.panning{cursor:grabbing}
.legend{order:9;flex-basis:100%;display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.76rem;color:var(--ink-2);
  font-family:var(--mono)}
.legend i{display:inline-block;width:24px;height:0;border-top:3px solid;vertical-align:middle;margin-right:.35rem}
.lane-t{fill:var(--muted);font-size:13px;font-family:var(--mono)}
.lane{fill:var(--surface-2);opacity:.55}
.edge{fill:none;stroke-width:1.8;opacity:.55;stroke-linecap:round;transition:opacity .2s,stroke-width .15s}
.edge.act{stroke:var(--act)} .edge.auto{stroke:var(--auto);stroke-dasharray:6 5} .edge.any{stroke:var(--any);stroke-dasharray:1.5 4.5;stroke-width:2.4}
.edge.near{opacity:1;stroke-width:2.8}
.edge.hot{opacity:1;stroke-width:4.2}
.edge.dim{opacity:.06}
.lbl{pointer-events:none;transition:opacity .2s;opacity:.95}
.lbl text{font-size:11px;fill:var(--ink-2);font-family:var(--mono)}
.lbl .pill{fill:var(--surface);stroke:var(--line);stroke-width:1}
.lbl.dim{opacity:.05}
.lbl.hot text{fill:var(--ink);font-weight:700}
.lbl.hot .pill{stroke:var(--glow)}
.hit{fill:none;stroke:transparent;stroke-width:14;pointer-events:stroke;cursor:pointer}
.hit.off{pointer-events:none}
.node{cursor:pointer;transition:opacity .2s}
.node>rect:first-child{fill:var(--node);stroke:var(--line);stroke-width:1.5;filter:drop-shadow(0 2px 6px rgba(0,0,0,.18))}
.node .tab{fill:var(--act);opacity:.9}
.node.no .tab{fill:var(--auto)}
.node.no>rect:first-child{stroke-dasharray:6 4;stroke:var(--muted)}
.node text{pointer-events:none;font-family:var(--mono)}
.node .nid{fill:var(--ink);font-size:14px;font-weight:700}
.node .nst{fill:var(--muted);font-size:10.5px}
.node.focus>rect:first-child{stroke:var(--glow);stroke-width:2.5;stroke-dasharray:none}
.node.faded{opacity:.25}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.8rem .9rem;overflow:auto;min-height:0;font-size:.86rem;line-height:1.4}
.panel h2{margin:.1rem 0 .1rem;font-size:1.15rem;font-family:var(--mono);word-break:break-word}
.panel .k{color:var(--muted);font-size:.78rem}
.panel .pin{float:right;font-size:.72rem;color:var(--glow);font-family:var(--mono)}
.sec{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:.9rem 0 .3rem;border-bottom:1px solid var(--line);padding-bottom:.15rem}
.tr{padding:.35rem 0;border-bottom:1px dashed var(--line)}
.tr .tl{font-family:var(--mono);font-size:.84rem;word-break:break-word}
.rel{color:var(--muted)} .val{color:var(--ink);font-weight:600}
.meta{font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-top:.1rem;word-break:break-word}
.q{font-family:var(--mono);font-size:.76rem;margin:.15rem 0 0 .6rem;color:var(--ink-2);word-break:break-word}
.ev{font-size:.78rem;color:var(--ink-2);margin-top:.15rem}
.sub{margin:.25rem 0 .1rem .8rem;padding-left:.5rem;border-left:2px solid var(--line);font-family:var(--mono);font-size:.72rem;color:var(--ink-2);word-break:break-word}
.sub div{margin:.08rem 0}
.sectext{font-size:.8rem;color:var(--ink-2);white-space:pre-wrap;margin:.2rem 0 .5rem}
table.kv{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.76rem}
table.kv td{padding:.15rem .3rem .15rem 0;vertical-align:top;word-break:break-word}
table.kv td:first-child{color:var(--muted);white-space:nowrap}
@media (prefers-reduced-motion:reduce){.edge,.lbl,.node{transition:none}}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="facts" id="facts"></div>
  <div class="legend" id="legend"></div>
  <div class="tools">
    <button id="zin" type="button" aria-label="Zoom in">+</button>
    <button id="zout" type="button" aria-label="Zoom out">−</button>
    <button id="fitw" type="button">Fit width</button>
    <button id="fitb" type="button">Whole graph</button>
    <button id="theme" type="button">Light / dark</button>
    <button id="full" type="button">Full screen</button>
  </div>
</header>
<div class="main">
  <div class="map">
    <svg id="map" role="img" aria-label="State-transition graph"></svg>
  </div>
  <aside class="panel" id="panel" aria-live="polite"></aside>
</div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const val = v => esc(v !== null && typeof v === "object" ? JSON.stringify(v) : v);
$("#title").textContent = D.title; document.title = D.title;
$("#facts").textContent = `${D.file} · ${D.nodes.length} is_a dialog_state · ${D.edges.length} leads_to · ${D.triples.length} triples`;
const N = Object.fromEntries(D.nodes.map(n => [n.id, n]));
const kindOf = e => e.on === "auto" || e.on === "any" ? e.on : "act";
const acts = [...new Set(D.edges.filter(e => kindOf(e) === "act").map(e => e.on))];
$("#legend").innerHTML = `<span><i style="border-color:var(--act)"></i>on = ${esc(acts.join(", "))}</span>`
  + `<span><i style="border-color:var(--auto);border-top-style:dashed"></i>on = auto</span>`
  + `<span><i style="border-color:var(--any);border-top-style:dotted;border-top-width:4px"></i>on = any</span>`
  + `<span>solid box, blue tab: prompts_customer = yes · dashed box, green tab: no</span>`;

// ---- layout: computed when the page was built (built-in layered layout, or Graphviz)
const svg = $("#map"), NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };
function fit(t, max) { let size = parseFloat(getComputedStyle(t).fontSize) || 14; while (t.getComputedTextLength() > max && size > 8) { size -= .5; t.style.fontSize = size + "px"; } }
const L = D.layout, pos = {}, world = el("g", {});
const W = L.w, H = L.h, nodeW = L.node_w, nodeH = L.node_h;
for (const n of D.nodes) pos[n.id] = { x: L.nodes[n.id].cx - nodeW / 2, y: L.nodes[n.id].cy - nodeH / 2 };
const defs = el("defs", {});
for (const k of ["act", "auto", "any"]) {
  const m = el("marker", { id: "mk-" + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 9, markerHeight: 9, orient: "auto-start-reverse", markerUnits: "userSpaceOnUse" }, defs);
  el("path", { d: "M0,1 L10,5 L0,9 z", style: `fill:var(--${k})` }, m);
}
const gE = el("g", {}, world), gL = el("g", {}, world), gH = el("g", {}, world), gN = el("g", {}, world);
const edgeEl = {}, lblEl = {}, hitEl = {}, nodeEl = {};
for (const e of D.edges) {
  const d = L.edges[e.id].d;
  const p = edgeEl[e.id] = el("path", { d, class: "edge " + kindOf(e), "marker-end": `url(#mk-${kindOf(e)})` }, gE);
  const lg = lblEl[e.id] = el("g", { class: "lbl" }, gL);
  el("rect", { rx: 6, class: "pill" }, lg);
  el("text", { "text-anchor": "middle" }, lg).textContent = e.on;
  const h = hitEl[e.id] = el("path", { d, class: "hit" }, gH);
  h.addEventListener("mouseenter", () => setHover({ t: "edge", id: e.id }));
  h.addEventListener("mouseleave", () => setHover(null));
  h.addEventListener("click", ev => { if (!dragged) { pin({ t: "edge", id: e.id }); ev.stopPropagation(); } });
}
for (const n of D.nodes) {
  const p = pos[n.id];
  const g = nodeEl[n.id] = el("g", { class: "node" + (n.prompts_customer === "no" ? " no" : ""), transform: `translate(${p.x},${p.y})`, tabindex: 0, role: "button", "aria-label": n.id }, gN);
  el("rect", { width: nodeW, height: nodeH, rx: 12 }, g);
  el("rect", { width: 5, height: nodeH - 18, x: 9, y: 9, rx: 2.5, class: "tab" }, g);
  const t = el("text", { x: nodeW / 2 + 5, y: nodeH / 2 - 2, "text-anchor": "middle", class: "nid" }, g); t.textContent = n.id; fit(t, nodeW - 34);
  const st = el("text", { x: nodeW / 2 + 5, y: nodeH / 2 + 14, "text-anchor": "middle", class: "nst" }, g); st.textContent = "in_stage: " + (n.stage ?? "");
  g.addEventListener("mouseenter", () => setHover({ t: "node", id: n.id }));
  g.addEventListener("mouseleave", () => setHover(null));
  g.addEventListener("focus", () => setHover({ t: "node", id: n.id })); g.addEventListener("blur", () => setHover(null));
  g.addEventListener("click", ev => { if (!dragged) { pin({ t: "node", id: n.id }); ev.stopPropagation(); } });
  g.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pin({ t: "node", id: n.id }); } });
}

// ---- trigger labels: slide each along its arrow to the first spot that overlaps no box and no other label
(function placeLabels() {
  const boxes = D.nodes.map(n => ({ x0: pos[n.id].x - 4, y0: pos[n.id].y - 4, x1: pos[n.id].x + nodeW + 4, y1: pos[n.id].y + nodeH + 4 }));
  const hit = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  const placed = [], tries = [.5, .42, .58, .34, .66, .26, .74, .18, .82, .12, .88];
  const sorted = [...D.edges].sort((a, b) => edgeEl[a.id].getTotalLength() - edgeEl[b.id].getTotalLength());
  for (const e of sorted) {
    const g = lblEl[e.id], t = g.querySelector("text"), r = g.querySelector("rect"), p = edgeEl[e.id], len = p.getTotalLength();
    const w = t.getComputedTextLength() + 12, h = 17;
    let best = null;
    for (const f of tries) {
      const q = p.getPointAtLength(len * f), bx = { x0: q.x - w / 2, y0: q.y - h / 2, x1: q.x + w / 2, y1: q.y + h / 2 };
      const cost = boxes.filter(b => hit(bx, b)).length * 10 + placed.filter(b => hit(bx, b)).length;
      if (!best || cost < best.cost) best = { cost, q, bx };
      if (cost === 0) break;
    }
    placed.push(best.bx);
    t.setAttribute("x", best.q.x); t.setAttribute("y", best.q.y + 4);
    r.setAttribute("x", best.bx.x0); r.setAttribute("y", best.bx.y0); r.setAttribute("width", w); r.setAttribute("height", h);
  }
})();

// ---- zoom and pan (viewBox)
let vb = { x: 0, y: 0, w: W, h: H }, dragged = false;
const setVB = () => svg.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.w} ${vb.h}`);
function fitView() { vb = { x: 0, y: 0, w: W, h: H }; setVB(); }
function fitWidth() {                                 // the default: full width, readable size; scroll down the flow
  const r = svg.getBoundingClientRect(), w = Math.max(W, r.width / 1.25);
  vb = { x: (W - w) / 2, y: 0, w, h: w * r.height / r.width }; setVB();
}
function clampVB() { const m = 60; vb.y = Math.min(Math.max(vb.y, -m), Math.max(-m, H - vb.h + m)); vb.x = Math.min(Math.max(vb.x, -W), W); }
function zoomAt(f, cx, cy) {
  const r = svg.getBoundingClientRect(), s = Math.max(vb.w / r.width, vb.h / r.height);
  const ox = (r.width - vb.w / s) / 2, oy = (r.height - vb.h / s) / 2;
  const px = vb.x + ((cx ?? r.left + r.width / 2) - r.left - ox) * s, py = vb.y + ((cy ?? r.top + r.height / 2) - r.top - oy) * s;
  const nw = Math.min(W * 2, Math.max(W / 8, vb.w * f)), k = nw / vb.w;
  vb = { x: px - (px - vb.x) * k, y: py - (py - vb.y) * k, w: nw, h: vb.h * k }; setVB();
}
svg.addEventListener("wheel", ev => {                // wheel scrolls; Ctrl/⌘ + wheel (or pinch) zooms
  ev.preventDefault();
  if (ev.ctrlKey || ev.metaKey) { zoomAt(ev.deltaY > 0 ? 1.12 : 1 / 1.12, ev.clientX, ev.clientY); return; }
  const r = svg.getBoundingClientRect(), s = Math.max(vb.w / r.width, vb.h / r.height);
  vb.x += (ev.shiftKey ? ev.deltaY : ev.deltaX) * s; if (!ev.shiftKey) vb.y += ev.deltaY * s; clampVB(); setVB();
}, { passive: false });
let drag = null;
svg.addEventListener("pointerdown", ev => { drag = { x: ev.clientX, y: ev.clientY, vb: { ...vb } }; dragged = false; });
addEventListener("pointermove", ev => {
  if (!drag) return;
  const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
  if (!dragged && Math.hypot(dx, dy) < 5) return;
  dragged = true; svg.classList.add("panning");
  const r = svg.getBoundingClientRect(), s = Math.max(vb.w / r.width, vb.h / r.height);
  vb.x = drag.vb.x - dx * s; vb.y = drag.vb.y - dy * s; setVB();
});
addEventListener("pointerup", () => { drag = null; svg.classList.remove("panning"); setTimeout(() => { dragged = false; }, 0); });
svg.addEventListener("click", () => { if (!dragged) pin(null); });
$("#zin").onclick = () => zoomAt(1 / 1.3); $("#zout").onclick = () => zoomAt(1.3); $("#fitb").onclick = fitView; $("#fitw").onclick = fitWidth;
fitWidth(); addEventListener("resize", () => fitWidth());

// ---- the details panel: triples exactly as stored in the file
const byHead = {}, byTail = {}, byId = {};
for (const t of D.triples) { (byHead[t.head] = byHead[t.head] || []).push(t); (byTail[String(t.tail)] = byTail[String(t.tail)] || []).push(t); byId[t.id] = t; }
const SHOWN = ["head", "relation", "tail", "qualifiers", "evidence"];
function tripleHTML(t, opts = {}) {
  const meta = Object.keys(t).filter(k => !SHOWN.includes(k)).map(k => `${esc(k)}=${val(t[k])}`).join(" · ");
  const q = t.qualifiers ? Object.entries(t.qualifiers).map(([k, v]) => `<div>${esc(k)}: ${val(v)}</div>`).join("") : "";
  return `<div class="tr"><div class="tl">${opts.head ? `<span class="val">${val(t.head)}</span> ` : ""}<span class="rel">${esc(t.relation)}</span> <span class="val">${val(t.tail)}</span></div>`
    + (q ? `<div class="q">${q}</div>` : "")
    + (t.evidence != null ? `<div class="ev">evidence: “${esc(t.evidence)}”</div>` : "")
    + `<div class="meta">${meta}</div>` + (opts.extra || "") + `</div>`;
}
function entityHTML(name, skip) {                    // the triples of an entity a triple points to (one level)
  const ts = (byHead[String(name)] || []).filter(t => !(skip && skip(t)));
  if (!ts.length) return "";
  return `<div class="sub">${ts.map(t => `<div>${esc(t.relation)} ${val(t.tail)}${t.qualifiers ? " " + val(t.qualifiers) : ""} <span class="meta">${esc(t.id)} · ${esc(t.source || "")}</span></div>`).join("")}</div>`;
}
function sectionsHTML(sources) {
  const sec = (D.header.source || {}).sections || {};
  const list = [...new Set(sources.filter(Boolean))].filter(s => sec[s] != null);
  return list.length ? `<div class="sec">Source sections cited (source.sections)</div>` + list.map(s => `<div class="k">${esc(s)}</div><div class="sectext">${esc(sec[s])}</div>`).join("") : "";
}
const stateSkip = t => t.relation === "leads_to";
function nodeHTML(id) {
  const own = byHead[id] || [], into = byTail[id] || [];
  const outs = own.filter(t => t.relation === "leads_to"), ins = into.filter(t => t.relation === "leads_to");
  const other = own.filter(t => t.relation !== "leads_to"), otherIn = into.filter(t => t.relation !== "leads_to");
  return `<h2>${esc(id)}</h2><div class="k">${own.length} triples with head = ${esc(id)} · ${into.length} with tail = ${esc(id)}</div>`
    + `<div class="sec">head = ${esc(id)}</div>` + other.map(t => tripleHTML(t, { extra: entityHTML(t.tail, t2 => t2.relation === "enables" || t2.relation === "has_alias") })).join("")
    + `<div class="sec">leads_to from ${esc(id)} (${outs.length})</div>` + outs.map(t => tripleHTML(t)).join("")
    + `<div class="sec">tail = ${esc(id)} (${into.length})</div>` + ins.concat(otherIn).map(t => tripleHTML(t, { head: true })).join("")
    + sectionsHTML(own.concat(into).map(t => t.source));
}
function edgeHTML(id) {
  const t = byId[id], q = t.qualifiers || {};
  const wf = D.triples.filter(x => x.relation === "enables" && x.qualifiers && (q.edges || []).includes(x.qualifiers.edge));
  return `<h2>${esc(t.head)} → ${esc(t.tail)}</h2><div class="k">triple ${esc(t.id)}</div>`
    + `<div class="sec">the leads_to triple</div><table class="kv">` + Object.keys(t).map(k => `<tr><td>${esc(k)}</td><td>${val(t[k])}</td></tr>`).join("") + `</table>`
    + ((q.rules || []).length ? `<div class="sec">rules named in qualifiers.rules</div>` + q.rules.map(r => `<div class="tr"><div class="tl"><span class="val">${esc(r)}</span></div>${entityHTML(r)}</div>`).join("") : "")
    + (wf.length ? `<div class="sec">enables triples with qualifiers.edge in qualifiers.edges</div>` + wf.map(x => tripleHTML(x, { head: true })).join("") : "")
    + sectionsHTML([t.source].concat(wf.map(x => x.source)));
}
function overviewHTML() {
  const h = D.header, counts = {};
  for (const t of D.triples) counts[t.relation] = (counts[t.relation] || 0) + 1;
  const src = { ...(h.source || {}) }; const sections = src.sections || {}; delete src.sections;
  return `<h2>${esc(h.title || D.file)}</h2><div class="k">Hover a state or an arrow to see its triples here; click to pin, Esc or click the background to unpin.</div>`
    + `<div class="sec">file header</div><table class="kv">` + Object.keys(h).filter(k => !["source", "relations", "title"].includes(k)).map(k => `<tr><td>${esc(k)}</td><td>${val(h[k])}</td></tr>`).join("")
    + Object.keys(src).map(k => `<tr><td>source.${esc(k)}</td><td>${val(src[k])}</td></tr>`).join("") + `</table>`
    + `<div class="sec">relations (descriptions as written)</div><table class="kv">` + Object.entries(h.relations || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("") + `</table>`
    + `<div class="sec">triples per relation (${D.triples.length} in all)</div><table class="kv">` + Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join("") + `</table>`
    + `<div class="sec">source.sections (${Object.keys(sections).length})</div>` + Object.entries(sections).map(([k, v]) => `<div class="k">${esc(k)}</div><div class="sectext">${esc(v)}</div>`).join("");
}

// ---- hover / pin
let hover = null, pinned = null, hoverT = null;
function setHover(x) { clearTimeout(hoverT); if (x) { hover = x; render(); } else hoverT = setTimeout(() => { hover = null; render(); }, 120); }
function pin(x) { pinned = x; render(); }
let shown = null;
function render() {
  const f = hover || pinned, key = f ? f.t + ":" + f.id : "";
  for (const id in edgeEl) { edgeEl[id].classList.remove("near", "hot", "dim"); lblEl[id].classList.remove("dim", "hot"); hitEl[id].classList.remove("off"); }
  for (const id in nodeEl) nodeEl[id].classList.remove("focus", "faded");
  if (f && f.t === "node") {
    const nb = new Set([f.id]);
    for (const e of D.edges) if (e.src === f.id || e.dst === f.id) { edgeEl[e.id].classList.add("near"); nb.add(e.src); nb.add(e.dst); }
      else { edgeEl[e.id].classList.add("dim"); lblEl[e.id].classList.add("dim"); hitEl[e.id].classList.add("off"); }
    for (const id in nodeEl) nodeEl[id].classList.toggle("faded", !nb.has(id));
    nodeEl[f.id].classList.add("focus");
  } else if (f && f.t === "edge") {
    const e = D.edges.find(x => x.id === f.id);
    for (const x of D.edges) if (x.id !== e.id) { edgeEl[x.id].classList.add("dim"); lblEl[x.id].classList.add("dim"); }
    edgeEl[e.id].classList.add("hot"); lblEl[e.id].classList.add("hot");
    for (const id in nodeEl) nodeEl[id].classList.toggle("faded", id !== e.src && id !== e.dst);
  }
  if (key !== shown) {
    shown = key;
    const P = $("#panel");
    P.innerHTML = (pinned && f === pinned ? `<span class="pin">pinned · Esc to unpin</span>` : "") + (!f ? overviewHTML() : f.t === "node" ? nodeHTML(f.id) : edgeHTML(f.id));
    P.scrollTop = 0;
  }
}
addEventListener("keydown", ev => {
  if (ev.key === "Escape") pin(null);
  else if ((ev.key === "f" || ev.key === "F") && !ev.target.closest("button")) $("#full").click();
});
$("#full").onclick = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.();
$("#theme").onclick = () => { const r = document.documentElement, dark = getComputedStyle(r).colorScheme.includes("dark");
  r.dataset.theme = dark ? "light" : "dark"; try { localStorage.setItem("pg-theme", r.dataset.theme); } catch (e) {} };
try { const t = localStorage.getItem("pg-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
render();
</script>
</body>
</html>
"""


def build_page(policy: Path, title: str | None, layout: str = "builtin", **spacing) -> tuple[str, dict]:
    graph = load_graph(policy)
    lay = dot_layout(graph) if layout == "graphviz" else None
    data = {**graph, "title": title or f"{graph['header'].get('title') or graph['file']}: state-transition graph",
            "layout": lay or layered_layout(graph, **spacing)}
    html = PAGE.replace("__TITLE__", data["title"].replace("<", "&lt;")).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/"))
    return html, data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default=str(ROOT / "gmsjourney" / "policy" / "charge_review_kg.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "POLICY_GRAPH_SHOWCASE.html"))
    ap.add_argument("--title", default=None, help="page title (default: the file's title + ': state-transition graph')")
    ap.add_argument("--layout", default="builtin", choices=["builtin", "graphviz"],
                    help="builtin (default, the same on every machine) or graphviz (needs the dot program)")
    ap.add_argument("--node-gap", type=float, default=90, help="space between boxes in a row (built-in layout)")
    ap.add_argument("--rank-gap", type=float, default=78, help="space between rows (built-in layout)")
    ap.add_argument("--lane-gap", type=float, default=13, help="space between arrow lanes (built-in layout)")
    a = ap.parse_args()
    html, data = build_page(Path(a.policy), a.title, a.layout, node_gap=a.node_gap, rank_gap=a.rank_gap,
                            lane_gap=a.lane_gap)
    # nothing on the page is invented: every state, arrow and label is a value from the file
    T = data["triples"]
    assert [n["id"] for n in data["nodes"]] == list(dict.fromkeys(
        t["head"] for t in T if t["relation"] == "is_a" and t["tail"] == "dialog_state"))
    assert [e["id"] for e in data["edges"]] == [t["id"] for t in T if t["relation"] == "leads_to"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) // 1024} KB): {len(data['nodes'])} states, {len(data['edges'])} transitions, "
          f"{len(T)} triples, layout {data['layout']['engine']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
