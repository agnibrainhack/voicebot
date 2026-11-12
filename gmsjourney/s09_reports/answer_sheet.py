"""The answer sheet of a test run: every generated case and what a correct bot must do, written right after the cases
and their rewordings are proved (campaign step 5, before any bot is run).

For each base case the sheet shows:

* the case: its category, the decisions it exercises, and how the call must end (end state, transfer reason, what must
  be recorded, what the customer must be told), all from the oracle's own run of the case on the policy graph;
* the call opening: what the bot must do before the customer speaks;
* turn by turn: the dialog state, what the customer says (the original wording) and what it means (the meaning
  card), what a correct bot must do next (its tool calls with their key arguments and its acts, in the order they
  happen), and the policy rules
  of that state;
* every reworded version of the case (both arms, template and LLM), with the factor levels it changed and whether the
  proof accepted it: each reworded turn beside the original, marked as read back to the same meaning or not. An
  accepted version has the same answer key, turn for turn; a rejected one never reaches the bot and is shown for audit.

Nothing on the sheet is written by hand or by a model: it is read off the oracle's trace, the policy graph and the
proofs. Analogy: the marking scheme a teacher writes before the exam is sat, with every permitted way the questions
may be asked.

    write_answer_sheet(out_dir, oracle=..., store=..., validator=..., bases=..., proofs=..., llm_proofs=...)
        -> {"file": "answer_sheet.html", "cases": n, "rewordings": m, "accepted": k}
"""
from __future__ import annotations

import copy
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

E = lambda s: html.escape("" if s is None else str(s), quote=True)  # noqa: E731

CSS = """
:root{--bg:#f3f5f7;--surface:#fff;--ink:#141c24;--muted:#596877;--line:#d5dce3;--soft:#e7ecf0;--accent:#1d5fa8;
--ok:#1c7a4a;--okbg:#e2f2e9;--bad:#b3261e;--badbg:#fbe9e7;--graph:#eef3fa;--code:#f2f5f8;color-scheme:light;
--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10161d;--surface:#161e27;--ink:#e6edf3;
--muted:#93a3b3;--line:#2b3743;--soft:#222c36;--accent:#7fb2ec;--ok:#5cc98d;--okbg:#16281f;--bad:#f2837a;--badbg:#2e1a19;
--graph:#18222e;--code:#131a22;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#10161d;--surface:#161e27;--ink:#e6edf3;--muted:#93a3b3;--line:#2b3743;--soft:#222c36;
--accent:#7fb2ec;--ok:#5cc98d;--okbg:#16281f;--bad:#f2837a;--badbg:#2e1a19;--graph:#18222e;--code:#131a22;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.55;padding:0 16px}
.wrap{max-width:1320px;margin:0 auto;padding:28px 0 60px}
h1{font-size:1.9rem;margin:0 0 .3rem;line-height:1.15}h2{font-size:1.25rem;margin:1.6rem 0 .5rem}
h3{font-size:1.05rem;margin:0;display:flex;flex-wrap:wrap;gap:.5rem;align-items:baseline}
.lede{color:var(--muted);max-width:78ch;margin:.2rem 0 1rem}
code,.mono{font-family:var(--mono);font-size:.82em}code{background:var(--code);border:1px solid var(--soft);border-radius:4px;padding:0 .3em;overflow-wrap:anywhere}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:140px}
.tile b{display:block;font-size:1.4rem}.tile span{color:var(--muted);font-size:.85rem}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
.toolbar input{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);min-width:320px}
.toc{display:flex;flex-wrap:wrap;gap:6px}.toc a{font-family:var(--mono);font-size:.78rem;padding:2px 8px;border:1px solid var(--line);border-radius:999px;color:var(--ink);text-decoration:none;background:var(--surface)}
.case{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}
.meta{color:var(--muted);font-size:.86rem;margin:.3rem 0 .6rem}
.tag{border-radius:4px;padding:.05rem .45rem;font-family:var(--mono);font-size:.7rem;border:1px solid var(--line);color:var(--muted)}
.tag.ok{background:var(--okbg);color:var(--ok);border-color:transparent}.tag.no{background:var(--badbg);color:var(--bad);border-color:transparent}
.end{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;background:var(--graph);border-radius:8px;padding:8px 12px;font-size:.88rem;margin:.4rem 0 .8rem}
.end dt{color:var(--muted)}.end dd{margin:0;font-family:var(--mono);font-size:.82rem;overflow-wrap:anywhere}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{border-top:1px solid var(--soft);padding:7px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:500;font-size:.78rem}
td.n{font-family:var(--mono);color:var(--muted);width:2.5rem}td.must{background:var(--graph)}
td.must div{font-family:var(--mono);font-size:.8rem;overflow-wrap:anywhere}.rule{color:var(--muted);font-size:.8rem}
details{margin-top:.8rem}summary{cursor:pointer;color:var(--accent)}
.var{border:1px solid var(--soft);border-radius:8px;padding:8px 10px;margin:8px 0}.var.rej{opacity:.75}
.vh{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;font-size:.85rem;margin-bottom:4px}
.eq{font-size:.78rem;margin-left:6px}.eq.ok{color:var(--ok)}.eq.no{color:var(--bad)}
.hidden{display:none}
@media print{body{background:#fff}.toolbar,.toc{display:none}.case{break-inside:avoid-page}}
"""


def _named(text: str, merchants: dict | None) -> str:
    """Charge ids in a line replaced by merchant and amount (Amazon $45.67)."""
    from .cases import _label
    if not merchants:
        return text
    pat = re.compile(r"(?<![\w$])(" + "|".join(re.escape(k) for k in sorted(merchants, key=len, reverse=True)) + r")(?!\w)")
    return pat.sub(lambda m: _label(m.group(1), merchants), text)


def _outcome_rows(answer: Any, merchants: dict | None = None) -> list[tuple[str, str]]:
    try:
        o = json.loads(answer) if isinstance(answer, str) else (answer or {})
    except ValueError:
        return [("outcome", str(answer))]
    if not isinstance(o, dict):
        return [("outcome", str(o))]
    from .verdict import _outcome
    n = _outcome(o)
    rows = [("ends", n["end"] or "?")]
    if n["transfer_reason"]:
        rows.append(("transfer reason", n["transfer_reason"]))
    rows.append(("records", _named("; ".join(n["writes"]), merchants) or "nothing"))
    if n["told"]:
        rows.append(("tells the customer", "; ".join(map(str, n["told"]))))
    for k, v in o.items():                                # a pack's own outcome fields
        if k not in ("end", "transfer_reason", "writes", "told") and v not in (None, [], {}, ""):
            rows.append((k.replace("_", " "), json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)))
    return rows


def _card(c: dict | None) -> str:
    if not c:
        return ""
    sl = c.get("slots") or {}
    return f"{c.get('act')}({', '.join(f'{k}={v}' for k, v in sl.items())})" if sl else str(c.get("act"))


def write_answer_sheet(out_dir: str | Path, *, oracle, store, validator, bases: list, proofs: list,
                       llm_proofs: list | None = None, journey: str = "", factor_names: list | None = None,
                       out_name: str = "answer_sheet.html") -> dict[str, Any]:
    """Write ``answer_sheet.html`` in ``out_dir`` (see the module docstring) and return what it holds."""
    from .cases import describe_act, describe_tool, items_named, segment
    kg = getattr(store, "kg", None)
    ont = validator.ont
    by_base: dict[str, list] = defaultdict(list)
    for p in list(proofs) + list(llm_proofs or []):
        by_base[p.qid].append(p)
    for v in by_base.values():
        v.sort(key=lambda p: (not p.accepted, p.arm != "template", p.sid))

    def rules(st: str | None) -> str:
        if kg is None or not st:
            return ""
        try:
            rs = kg.rules_for(st)
        except Exception:                                 # noqa: BLE001 - a state the graph does not know
            return ""
        return "".join(f'<div class="rule"><code>{E(r)}</code> {E(kg.rule_text(r))}</div>' for r in rs)

    blocks, toc = [], []
    n_var = n_ok = 0
    for b in bases:
        spec, fx = b.metadata["spec"], b.metadata["fixture"]
        tr = oracle.run(spec, copy.deepcopy(fx))
        merchants = items_named(fx, tr.events)          # charges by merchant and amount
        opening, turns = segment(tr.events)
        path = validator.base_path(b)
        states = validator.base_states(b) if hasattr(validator, "base_states") else [None] * len(path)
        vs = by_base.get(b.qid, [])
        base_text = {}
        if vs:
            base_text = {t["turn"]: t.get("base_text") for t in vs[0].turns}
        rows = []
        for i, t in enumerate(turns):
            frame, ctx = path[i] if i < len(path) else (t.get("frame"), {})
            st = states[i] if i < len(states) and states[i] is not None else t.get("state")
            words = base_text.get(i)
            if words is None and frame is not None:
                try:
                    words = validator.realizer.realize(frame, {}, f"{b.qid}|{i}|base", ctx)
                except Exception:                         # noqa: BLE001
                    words = str(frame)
            card = ont.card(frame, st) if frame is not None else {}
            must = [f"<div>calls {E(describe_tool(x, merchants))}</div>" if k == "call" else           # in the order they happen
                    f"<div>{E(describe_act(x, merchants))}</div>" for k, x in t["steps"]]
            rows.append(f'<tr><td class="n">{i}</td><td><code>{E(st)}</code></td><td>“{E(words)}”</td>'
                        f'<td><code>{E(_card(card))}</code></td>'
                        f'<td class="must">{"".join(must) or "<div>(nothing: the call has ended)</div>"}</td>'
                        f'<td>{rules(st)}</td></tr>')
        op = " ; ".join(describe_act(a, merchants) for a in opening if not str(a.get("act", "")).startswith("("))
        var_html = []
        for p in vs:
            n_var += 1
            n_ok += bool(p.accepted)
            lv = ", ".join(f"{k}={v}" for k, v in (p.levels or {}).items())
            tl = "".join(f'<tr><td class="n">{t["turn"]}</td><td>“{E(t.get("enriched_text"))}”'
                         f'<span class="eq {"ok" if t.get("equivalent") else "no"}">'
                         f'{"✓ same meaning" if t.get("equivalent") else "✗ read as " + E(t.get("read_back"))}</span></td></tr>'
                         for t in p.turns)
            var_html.append(f'<div class="var{"" if p.accepted else " rej"}"><div class="vh"><b class="mono">{E(p.sid)}</b>'
                            f'<span class="tag">{E(p.arm)}</span>'
                            f'<span class="tag {"ok" if p.accepted else "no"}">{"proved: same answer key" if p.accepted else "rejected: never reaches the agent"}</span>'
                            f'<span class="meta" style="margin:0">{E(lv)}</span></div>'
                            f'<div class="tw"><table><tbody>{tl}</tbody></table></div></div>')
        cat = b.metadata.get("category") or getattr(b, "base", "") or ""
        tags = b.metadata.get("tags") or []
        rel = b.metadata.get("relevant") or {}
        rel_txt = ", ".join(f"{k}={v}" for k, v in rel.items()) if isinstance(rel, dict) else ", ".join(map(str, rel))
        end = "".join(f"<dt>{E(k)}</dt><dd>{E(v)}</dd>" for k, v in _outcome_rows(b.answer, merchants))
        q = getattr(b, "question", None) or b.metadata.get("question")
        blocks.append(
            f'<section class="case" id="{E(b.qid)}" data-text="{E((b.qid + " " + str(cat) + " " + " ".join(map(str, tags))).lower())}">'
            f'<h3><span class="mono">{E(b.qid)}</span> <span class="tag">{E(cat)}</span>'
            + "".join(f'<span class="tag">{E(t)}</span>' for t in [x for x in tags if x != cat][:6])
            + f'<span class="meta" style="margin:0">{len(turns)} customer turns · {len(vs)} reworded versions '
              f'({sum(p.accepted for p in vs)} proved)</span></h3>'
            + (f'<div class="meta">Decisions exercised: <code>{E(rel_txt)}</code></div>' if rel_txt else "")
            + (f'<div class="meta">Policy question: {E(q)}</div>' if q and isinstance(q, str) else "")
            + f'<dl class="end">{end}</dl>'
            + (f'<div class="meta">Opening statement: the agent must <code>{E(op)}</code></div>' if op else "")
            + '<div class="tw"><table><thead><tr><th>#</th><th>Dialog state</th><th>The customer says (original wording)</th>'
              '<th>Which means</th><th>A correct agent must, in order</th><th>Rules of the state</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>"
            + (f'<details><summary>The same case in other words: {len(vs)} reworded versions</summary>{"".join(var_html)}</details>'
               if vs else "")
            + "</section>")
        toc.append(f'<a href="#{E(b.qid)}">{E(b.qid)}</a>')
    doc = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answer sheet</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<h1 id="top">Answer sheet{(" · " + E(journey)) if journey else ""}</h1>
<p class="lede">Every case generated for this test run, before any agent is run: what the customer says and means at each
turn, what a correct agent must do (its tool calls with their key arguments and its acts, in the order they happen) and how the call must end, all read
off the policy graph's own run of the case. Under each case, its reworded versions: a proved version has the same answer
key turn for turn; a rejected one is never sent to the agent.</p>
<div class="tiles"><div class="tile"><b>{len(bases)}</b><span>cases</span></div><div class="tile"><b>{n_var}</b><span>reworded versions</span></div>
<div class="tile"><b>{n_ok}</b><span>proved to mean the same</span></div><div class="tile"><b>{n_var - n_ok}</b><span>rejected</span></div></div>
<div class="toolbar"><input id="q" type="search" placeholder="Find a case by id, category or tag" aria-label="Find a case">
<span class="meta" id="cnt"></span></div>
<div class="toc">{''.join(toc)}</div>
{''.join(blocks)}
</div>
<script>
if (/[?&]embed=1/.test(location.search)) document.querySelectorAll("h1, .lede").forEach(e => e.remove());   // inside the console
const q = document.getElementById("q"), cnt = document.getElementById("cnt");
function f() {{ const v = q.value.trim().toLowerCase(); let n = 0;
  document.querySelectorAll("section.case").forEach(s => {{ const on = !v || s.dataset.text.includes(v); s.classList.toggle("hidden", !on); n += on; }});
  cnt.textContent = v ? n + " shown" : ""; }}
q.addEventListener("input", f);
</script></html>"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / out_name).write_text(doc, encoding="utf-8")
    return {"file": out_name, "cases": len(bases), "rewordings": n_var, "accepted": n_ok}
