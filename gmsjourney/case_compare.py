"""Expected vs actual, case by case: one HTML page per campaign, in the answer sheet's format.

For every run of the bot under test (base cases, enriched cases, LLM-arm cases) the page puts what the oracle
expects next to what the bot did:

* the verdict and exactly what was compared to reach it (the outcome signature: how the call ended, the
  transfer reason, the responses actually recorded, the card action actually completed; for a policy-question
  case also the graded policy answer), field by field;
* the checks reported next to the verdict but not part of it (per-stage decisions, workflow admissibility,
  runtime invariance, budgets);
* for a failed run, the reasons in plain words (read off the graded row, nothing re-judged);
* turn by turn: the customer's words, then EXPECTED (oracle: tool calls and bot acts) beside ACTUAL (bot: tool
  calls, acts and its words), with the turns that differ marked.

Failed runs are shown by default; buttons show the passed ones, all of them, or one arm only.
Built from the campaign's own files - dataset/turn_by_turn.csv, dataset/turn_by_turn_summary.csv,
dataset/case_results.jsonl and report.json - so it can be rebuilt without re-running anything:

    python tools/case_comparison.py --campaign output/campaign_out
"""
from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

E = lambda s: html.escape(str(s), quote=True)

STYLE = """
:root{--bg:#eef1f4;--surface:#fff;--ink:#131c26;--muted:#5a6b7d;--line:#d3dbe3;--line-soft:#e3e9ee;
--accent:#a8500a;--accent-soft:#f7e9db;--pass:#1c7a4a;--pass-soft:#e2f2e9;--code-bg:#f2f5f8;
--fail:#b3261e;--fail-soft:#fbe9e7;
--display:"IBM Plex Sans Condensed",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10161d;--surface:#161e27;--ink:#e6edf3;
--muted:#93a3b3;--line:#2b3743;--line-soft:#222c36;--accent:#f0a35e;--accent-soft:#33261a;--pass:#5cc98d;
--pass-soft:#16281f;--code-bg:#131a22;--fail:#f2837a;--fail-soft:#2e1a19;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#10161d;--surface:#161e27;--ink:#e6edf3;--muted:#93a3b3;--line:#2b3743;
--line-soft:#222c36;--accent:#f0a35e;--accent-soft:#33261a;--pass:#5cc98d;--pass-soft:#16281f;--code-bg:#131a22;
--fail:#f2837a;--fail-soft:#2e1a19;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.6;padding-inline:16px}
.wrap{max-width:1360px;margin:0 auto;padding-block:clamp(26px,5vw,60px)}
h1,h2,h3{font-family:var(--display);line-height:1.15;margin:0;text-wrap:balance}
h1{font-size:clamp(1.9rem,5vw,2.9rem);font-weight:700}
h2{font-size:1.5rem;font-weight:700;margin-block:1.6rem .4rem}
h3{font-size:1.06rem;font-weight:600;display:flex;flex-wrap:wrap;gap:.5rem;align-items:baseline}
.lede{color:var(--muted);font-size:1.06rem;max-width:72ch}
code{font-family:var(--mono);font-size:.85em;background:var(--code-bg);border:1px solid var(--line-soft);border-radius:4px;padding:.06em .3em;word-break:break-word}
.kicker{display:inline-flex;align-items:center;gap:.5rem;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.16em;color:var(--accent)}
.kicker::before{content:"";width:22px;height:2px;background:var(--accent)}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:clamp(14px,2vw,20px);margin-block:14px}
.panel ol,.panel ul{margin:.3rem 0 .2rem;padding-left:1.3rem}.panel li{margin:.25rem 0}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:1rem 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.6rem .9rem;min-width:150px}
.tile b{display:block;font-family:var(--display);font-size:1.6rem}.tile span{font-size:.8rem;color:var(--muted)}
.case{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:clamp(14px,2vw,20px);margin-block:14px;scroll-margin-top:60px}
.case.fail{border-left:4px solid var(--fail)}.case.pass{border-left:4px solid var(--pass)}
.chead{display:flex;flex-direction:column;gap:.25rem;border-bottom:1px solid var(--line-soft);padding-bottom:.6rem}
.cat{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);background:var(--accent-soft);padding:.12rem .45rem;border-radius:4px}
.kind{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);border:1px solid var(--line);padding:.1rem .45rem;border-radius:4px}
.tag{border-radius:4px;padding:.05rem .45rem;font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.06em}
.tag.ok{background:var(--pass-soft);color:var(--pass)}.tag.no{background:var(--fail-soft);color:var(--fail)}
.tag.sev{border:1px solid var(--line);color:var(--muted)}
.cmeta{font-size:.8rem;color:var(--muted)}
.who{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);display:block}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;vertical-align:top;padding:.4rem .6rem;border-bottom:1px solid var(--line-soft)}
th{font-family:var(--display);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
td.m{white-space:nowrap;font-weight:600}td.m.ok{color:var(--pass)}td.m.no{color:var(--fail)}
tr.sub td{color:var(--muted);font-size:.8rem}tr.grp td{font-family:var(--display);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);background:var(--code-bg)}
.cmp{margin-top:.7rem;overflow-x:auto;border:1px solid var(--line-soft);border-radius:8px}
.why{margin:.7rem 0 .2rem;padding:.55rem .8rem;border-left:3px solid var(--fail);background:var(--fail-soft);border-radius:0 7px 7px 0;font-size:.92rem}
.why ul{margin:.2rem 0 0;padding-left:1.1rem}.why li{margin:.15rem 0}
.turns{display:grid;gap:10px;margin-top:.8rem}
.turn{border:1px solid var(--line-soft);border-radius:9px;padding:.7rem .85rem;min-width:0}
.turn.diff{border-color:var(--fail);background:linear-gradient(90deg,var(--fail-soft),transparent 30%)}
.turn .hd{display:flex;flex-wrap:wrap;gap:.5rem;align-items:baseline;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin-bottom:.4rem}
.turn .hd b{font-family:var(--display);font-size:.92rem;letter-spacing:0;text-transform:none;color:var(--ink)}
.turn .hd span{border-left:1px solid var(--line);padding-left:.5rem}
.say{display:grid;grid-template-columns:6.4rem 1fr;gap:.5rem;margin:.3rem 0}
.ea{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;margin-top:.4rem}
@media (max-width:820px){.ea{grid-template-columns:1fr}}
.col{border-radius:0 7px 7px 0;padding:.45rem .6rem;font-size:.88rem;min-width:0}
.col.exp{border-left:3px solid var(--pass);background:var(--pass-soft)}
.col.act{border-left:3px solid var(--line);background:var(--code-bg)}
.turn.diff .col.act{border-left-color:var(--fail)}
.col ol{margin:.15rem 0 0;padding-left:1.3rem}.col li{margin:.15rem 0}
.sk{display:inline-block;min-width:2.9rem;margin-right:.4rem;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
li.tool .sk{color:var(--accent)}
.words{margin-top:.35rem;font-size:.86rem;color:var(--muted)}
.mk{font-family:var(--mono);font-size:.7rem}.mk.ok{color:var(--pass)}.mk.no{color:var(--fail)}
.note{font-size:.8rem;color:var(--muted);margin-top:.3rem}
.viewbar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;padding:.5rem 0;background:var(--bg)}
.viewbar button{font:inherit;font-size:.82rem;padding:.3rem .7rem;border-radius:6px;border:1px solid var(--line);background:var(--surface);color:var(--ink);cursor:pointer}
.viewbar button[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
.viewbar .sep{width:1px;height:1.4rem;background:var(--line);margin:0 .3rem}
.viewbar .count{font-size:.8rem;color:var(--muted);margin-left:.4rem}
.toc{columns:3;column-gap:2rem;font-size:.86rem;margin-top:.6rem}
@media (max-width:900px){.toc{columns:1}}
.toc a{display:block;padding:.12rem 0;color:var(--ink);text-decoration:none;border-bottom:1px dashed var(--line-soft);font-family:var(--mono);font-size:.76rem}
.toc a.f::before{content:"✗ ";color:var(--fail)}.toc a.p::before{content:"✓ ";color:var(--pass)}
.small{font-size:.84rem;color:var(--muted)}
footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.85rem;color:var(--muted)}
"""

HOW_JUDGED = """
<div class="panel">
<h2 style="margin-top:0">How a case is judged</h2>
<p>Every run is graded by comparing what the bot <b>did</b> - read from the calls it made on the test backend,
not from what it said - with what the oracle (an executable reading of the policy) does in the same situation.</p>
<ol>
<li><b>Correct / incorrect</b> is decided by one comparison, the <b>outcome signature</b>, and all four fields must match:
<ul>
<li><b>how the call ended</b>: <code>TRANSFERRED</code> (transfer_to_human was called), <code>ENDED</code> (the bot closed
the call), or <code>INCOMPLETE</code> (neither);</li>
<li><b>the transfer reason</b> passed to transfer_to_human (e.g. <code>record_failed</code>, <code>customer_request</code>);</li>
<li><b>the responses actually recorded</b>: every record_charge_response call that returned SUCCESS for the same
charge, as charge → disposition (RECOGNIZED / UNRECOGNIZED / UNSURE);</li>
<li><b>the card action actually completed</b>: a submit_card_action that returned SUCCESS for the requested action,
with its delivery option.</li>
</ul>
For a <b>policy-question case</b> the graded policy answer must also be correct: its figure equals the exact value in the
policy store, or the bot abstains where the policy states nothing.</li>
<li><b>Reported next to the verdict, not part of it</b>: the per-stage decisions (authenticate, retrieve, present, kb,
capture, record, card action, transfer, summary), <b>workflow admissibility</b> (no policy violation such as claiming a
record that did not succeed, using an account tool before authentication, submitting without confirmation, skipping the
fee disclosure), runtime invariance of the customer turns, and the latency / length budgets. They set a run's severity
and feed the release gate.</li>
<li><b>Turn by turn</b> (below each case) is diagnostic only: the bot may take a different but allowed route and still
be correct, so a turn that differs from the oracle's does not by itself fail the case. After the first difference the
simulated customer reacts to what the bot actually asked, so later turns may legitimately differ.
A turn counts as <b>same as expected</b> when the bot made the same tool calls with the same key arguments, in order,
and took the same decisions (act, charge, reason, in order). Acknowledgements such as <code>OPENING</code> and
<code>CONFIRM_RECORDED</code> and a re-presentation of the same charge after a clarification are ignored, and so is the
content of a policy answer (<code>ANSWER_KB</code>), which is graded by the policy-answer check above.</li>
</ol>
</div>"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _j(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _b(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _fmt(v: Any) -> str:
    if v is None or v == [] or v == {} or v == "":
        return "none"
    if isinstance(v, list) and v and all(isinstance(x, list) and len(x) == 2 for x in v):
        return ", ".join(f"{a} = {b}" for a, b in v)
    if isinstance(v, (list, tuple)):
        return " / ".join("none" if x is None else str(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k} = {x}" for k, x in v.items()) or "none"
    return str(v)


def _items(s: str) -> list[str]:
    s = (s or "").strip()
    return [] if s in ("", "-", "(no turn)") else [x.strip() for x in s.split(" ; ") if x.strip()]


def _col(kind: str, tools: str, acts: str, words: str = "") -> str:
    li = [f'<li class="tool"><span class="sk">calls</span><code>{E(t)}</code></li>' for t in _items(tools)]
    li += [f'<li><span class="sk">says</span><code>{E(a)}</code></li>' for a in _items(acts)]
    if (tools or "").strip() == "(no turn)":
        body = "<p class='small'>(no such turn in this run)</p>"
    else:
        body = f"<ol>{''.join(li)}</ol>" if li else "<p class='small'>(nothing)</p>"
    head = "Expected (oracle)" if kind == "exp" else "Actual (bot)"
    return (f'<div class="col {kind}"><span class="who">{head}</span>{body}'
            + (f'<div class="words">“{E(words)}”</div>' if words else "") + "</div>")


def _speech_rows(row: dict) -> list[str]:
    """The speech check's verdict on the bot's words (gmsjourney/speech), as table rows: one for the words,
    one for the flow (part of the verdict only under --speech-gate strict)."""
    ok, flow = bool(int(row.get("c_speech", 1))), bool(int(row.get("c_flow", 1)))
    n, good = row.get("speech_claims", 0), row.get("speech_supported", 0)
    probs = [p for p in (row.get("speech_problems") or []) if p.get("kind") != "not_allowed_here"]
    detail = ("; ".join(f"{p.get('where') or 'turn ' + str(p.get('turn'))}, {p.get('kind')}: "
                        + (f"“{p.get('words')}” ({p.get('why')})" if p.get("words") else str(p.get("why")))
                        for p in probs[:3]) if probs
              else f"{good} of {n} claims supported, nothing missing")
    flow_probs = [p for p in (row.get("speech_problems") or []) if p.get("kind") == "not_allowed_here"]
    out = [f'<tr><td>What the customer heard (every claim grounded)</td><td colspan="2" class="small">{E(detail)}</td>'
           f'<td class="m {"ok" if ok else "no"}">{"✓ grounded" if ok else "✗ not grounded"}</td></tr>']
    out.append(f'<tr class="sub"><td>Speech flow (each act a graph transition)</td><td colspan="2" class="small">'
               f'{E("; ".join(str(p.get("why", "")) for p in flow_probs[:2]) or "every act follows the graph")}</td>'
               f'<td class="m {"ok" if flow else "no"}">{"✓" if flow else "✗"}</td></tr>')
    return out


def _verdict_table(row: dict, summ: dict) -> str:
    sig = row.get("signature") or row.get("why") or {}
    exp = sig.get("expected") if row else _j(summ.get("expected_outcome"))
    act = sig.get("actual") if row else _j(summ.get("actual_outcome"))
    exp, act = exp or {}, act or {}
    lines = ['<tr class="grp"><td colspan="4">Compared to decide correct / incorrect (all must match)</td></tr>']
    fields = [("How the call ended", "end"), ("Transfer reason", "transfer_reason"),
              ("Responses actually recorded", "recorded"), ("Card action actually completed", "action")]
    for label, k in fields:
        e, a = exp.get(k) if isinstance(exp, dict) else None, act.get(k) if isinstance(act, dict) else None
        ok = e == a
        lines.append(f'<tr><td>{E(label)}</td><td><code>{E(_fmt(e))}</code></td><td><code>{E(_fmt(a))}</code></td>'
                     f'<td class="m {"ok" if ok else "no"}">{"✓ match" if ok else "✗ differs"}</td></tr>')
    kb = row.get("kb_detail") or []
    if kb:
        for k in kb:
            ok = bool(k.get("correct"))
            lines.append(f'<tr><td>Policy answer to <code>{E(k.get("frame"))}</code></td>'
                         f'<td class="small">{E(k.get("detail"))}</td><td>“{E(k.get("answer"))}”</td>'
                         f'<td class="m {"ok" if ok else "no"}">{"✓ correct" if ok else "✗ wrong"}</td></tr>')
    gate = row.get("speech_gate")
    speech_rows = _speech_rows(row) if "c_speech" in row else []
    if gate in ("fail", "strict"):
        lines += speech_rows
    lines.append('<tr class="grp"><td colspan="4">Also checked, reported next to the verdict (not part of it)</td></tr>')
    if gate not in ("fail", "strict"):
        lines += speech_rows
    stages = (row.get("why") or {}).get("stages") or {}
    wrong = set(_j(summ.get("stages_wrong")) or []) | set(stages)
    graded = sorted(k[2:] for k in row if k.startswith("c_") and k not in ("c_speech", "c_flow"))
    for st in graded:
        if st in stages:
            d = stages[st]
            lines.append(f'<tr class="sub"><td>Stage {E(st)}</td><td><code>{E(_fmt(d.get("expected")))}</code></td>'
                         f'<td><code>{E(_fmt(d.get("actual")))}</code></td><td class="m no">✗ differs</td></tr>')
        else:
            ok = st not in wrong
            lines.append(f'<tr class="sub"><td>Stage {E(st)}</td><td colspan="2"></td>'
                         f'<td class="m {"ok" if ok else "no"}">{"✓ match" if ok else "✗ differs"}</td></tr>')
    viol = row.get("violations") or []
    lines.append(f'<tr class="sub"><td>Workflow admissibility</td><td colspan="2">'
                 + (", ".join(f"<code>{E(v)}</code>" for v in viol) if viol else "no policy violation")
                 + f'</td><td class="m {"no" if viol else "ok"}">{"✗ violations" if viol else "✓ admissible"}</td></tr>')
    if "invariance_ok" in row:
        ok = bool(row.get("invariance_ok"))
        lines.append(f'<tr class="sub"><td>Runtime invariance (customer turns read back)</td><td colspan="2"></td>'
                     f'<td class="m {"ok" if ok else "no"}">{"✓" if ok else "✗"}</td></tr>')
    if row.get("budget_violations"):
        lines.append(f'<tr class="sub"><td>Budgets</td><td colspan="2">{E("; ".join(map(str, row["budget_violations"])))}'
                     f'</td><td class="m no">✗ over</td></tr>')
    return ('<div class="cmp"><table><thead><tr><th>What is compared</th><th>Expected (oracle)</th>'
            '<th>Actual (bot)</th><th>Result</th></tr></thead><tbody>' + "".join(lines) + "</tbody></table></div>")


def _case(row: dict, summ: dict, turns: list[dict]) -> str:
    from .report import failure_reasons
    sid = row.get("sid") or summ.get("case_id")
    ok = bool(int(row.get("correct", 0))) if row else _b(summ.get("outcome_correct"))
    arm = row.get("arm") or summ.get("arm") or "base"
    sev = row.get("severity") or "none"
    factors = {k[2:]: v for k, v in row.items() if k.startswith("f_")}
    meta = [f"base case <code>{E(row.get('seed') or summ.get('base_id'))}</code>"]
    if summ:
        meta.append(f"customer turns expected <b>{E(summ.get('turns_expected'))}</b>, actual <b>{E(summ.get('turns_actual'))}</b>")
        fd = summ.get("first_divergence_turn")
        meta.append("turns all match" if fd in ("", None) else f"first different turn <b>{E(fd)}</b>")
    if factors:
        meta.append("factor levels <code>" + E(", ".join(f"{k}={v}" for k, v in factors.items())) + "</code>")
    why = ""
    if not ok or not bool(int(row.get("workflow_adherent", 1) or 1)):
        why = ('<div class="why"><span class="who">Why it ' + ("failed" if not ok else "was flagged") + "</span><ul>"
               + "".join(f"<li>{r}</li>" for r in failure_reasons(row)) + "</ul>"
               + (f"<div class='small'>failure codes: {E(', '.join(row.get('failure_codes') or []))}</div>"
                  if row.get("failure_codes") else "") + "</div>")
    opening = ""
    if summ and (summ.get("expected_opening") or summ.get("actual_opening")):
        same = summ.get("expected_opening") == summ.get("actual_opening")
        opening = (f'<div class="turn{"" if same else " diff"}"><div class="hd"><b>Call opens</b><span>before any customer turn</span>'
                   f'<span class="mk {"ok" if same else "no"}">{"same" if same else "differs"}</span></div><div class="ea">'
                   + _col("exp", "", summ.get("expected_opening", "")) + _col("act", "", summ.get("actual_opening", ""))
                   + "</div></div>")
    tb = []
    for t in turns:
        match = _b(t.get("turn_match"))
        after = _b(t.get("after_divergence"))
        flags = (f'<span class="mk {"ok" if match else "no"}">{"same as expected" if match else "differs"}</span>'
                 + ("<span>after the first difference</span>" if after else ""))
        tb.append(f'<div class="turn{"" if match else " diff"}"><div class="hd"><b>Turn {E(t.get("turn"))}</b>'
                  f'<span>{E(t.get("state"))}</span>{flags}</div>'
                  f'<div class="say"><span class="who">Customer</span><span>“{E(t.get("customer_text"))}”'
                  + (f' <span class="small"><code>{E(t.get("customer_card"))}</code></span>' if t.get("customer_card") else "")
                  + "</span></div><div class='ea'>"
                  + _col("exp", t.get("expected_tools", ""), t.get("expected_bot", ""))
                  + _col("act", t.get("actual_tools", ""), t.get("actual_bot_acts", ""), t.get("actual_bot_text", ""))
                  + "</div></div>")
    turns_html = (f'<div class="turns">{opening}{"".join(tb)}</div>' if tb or opening else
                  "<p class='small'>(no turn-by-turn record for this run)</p>")
    return (f'<section class="case {"pass" if ok else "fail"}" id="{E(sid)}" data-ok="{int(ok)}" data-arm="{E(arm)}">'
            f'<div class="chead"><h3>{E(sid)} <span class="cat">{E(row.get("base") or "")}</span>'
            f'<span class="kind">{E(arm)}</span><span class="tag {"ok" if ok else "no"}">{"correct" if ok else "incorrect"}</span>'
            + (f'<span class="tag sev">severity {E(sev)}</span>' if sev != "none" else "")
            + f'</h3><div class="cmeta">{" · ".join(meta)}</div></div>'
            + _verdict_table(row, summ) + why + turns_html + "</section>")


def write_case_comparison(out_dir: str | Path, out_name: str = "case_comparison.html") -> Path:
    out = Path(out_dir)
    ds = out / "dataset"
    rows = {}
    rpath = ds / "case_results.jsonl"
    if rpath.exists():
        for line in rpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["sid"]] = r
    summ = {s["case_id"]: s for s in _read_csv(ds / "turn_by_turn_summary.csv")}
    turns: dict[str, list[dict]] = {}
    for t in _read_csv(ds / "turn_by_turn.csv"):
        turns.setdefault(t["case_id"], []).append(t)
    R = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    bot = (R.get("results") or {}).get("bot", "the bot under test")
    order = list(rows) + [s for s in summ if s not in rows]
    arm_rank = {"base": 0, "template": 1, "llm": 2}
    order.sort(key=lambda s: (arm_rank.get((rows.get(s) or summ.get(s) or {}).get("arm") or "base", 9), s))
    blocks, toc = [], []
    n_ok = n_bad = 0
    arms: dict[str, int] = {}
    for sid in order:
        row, sm = rows.get(sid, {}), summ.get(sid, {})
        ok = bool(int(row.get("correct", 0))) if row else _b(sm.get("outcome_correct"))
        n_ok += ok
        n_bad += not ok
        arm = row.get("arm") or sm.get("arm") or "base"
        arms[arm] = arms.get(arm, 0) + 1
        blocks.append(_case(row, sm, turns.get(sid, [])))
        toc.append(f'<a class="{"p" if ok else "f"}" data-ok="{int(ok)}" data-arm="{E(arm)}" href="#{E(sid)}">{E(sid)}</a>')
    smp = (R.get("results") or {}).get("sample") or {}
    sample = (f" Sample: {smp['base_cases_tested']} of {smp['base_cases_total']} base cases (max_cases={smp['max_cases']})."
              if smp.get("max_cases") is not None else "")
    arm_buttons = "".join(f'<button type="button" data-arm="{E(a)}" aria-pressed="false">{E(a)}</button>' for a in sorted(arms))
    doc = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Expected vs Actual</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{STYLE}</style>
<div class="wrap">
<span class="kicker">Campaign · expected vs actual</span>
<h1>Expected vs actual, case by case</h1>
<p class="lede">System under test: <b>{E(bot)}</b>. Each run of the campaign, with what the oracle expects beside what the
bot did. {len(order)} runs: <b>{n_bad}</b> incorrect, <b>{n_ok}</b> correct.{E(sample)}</p>
<div class="tiles"><div class="tile"><b>{n_bad}</b><span>incorrect</span></div><div class="tile"><b>{n_ok}</b><span>correct</span></div>
{''.join(f'<div class="tile"><b>{n}</b><span>{E(a)} runs</span></div>' for a, n in sorted(arms.items()))}</div>
{HOW_JUDGED}
<h2>Index</h2>
<div class="toc" id="toc">{''.join(toc)}</div>
<h2>Cases</h2>
<div class="viewbar" role="group" aria-label="Show">
  <span class="who" style="display:inline">Show</span>
  <button type="button" data-ok="0" aria-pressed="true">Incorrect</button>
  <button type="button" data-ok="1" aria-pressed="false">Correct</button>
  <button type="button" data-ok="all" aria-pressed="false">All cases</button>
  <span class="sep"></span><span class="who" style="display:inline">Arm</span>
  <button type="button" data-arm="all" aria-pressed="true">all</button>{arm_buttons}
  <span class="count" id="cnt"></span>
</div>
<div id="cases">{''.join(blocks)}</div>
<footer>Generated by <code>gmsjourney/case_compare.py</code> from the campaign's own files
(<code>dataset/case_results.jsonl</code>, <code>dataset/turn_by_turn.csv</code>, <code>dataset/turn_by_turn_summary.csv</code>,
<code>report.json</code>); rebuild with <code>python tools/case_comparison.py --campaign {E(out)}</code>.
See also <code>report.html</code> (the campaign report).</footer>
</div>
<script>
(function () {{
  var okSel = "0", armSel = "all";
  try {{ okSel = localStorage.getItem("cmp-ok") || "0"; armSel = localStorage.getItem("cmp-arm") || "all"; }} catch (e) {{}}
  function apply() {{
    var n = 0;
    document.querySelectorAll("#cases .case, #toc a").forEach(function (el) {{
      var show = (okSel === "all" || el.dataset.ok === okSel) && (armSel === "all" || el.dataset.arm === armSel);
      el.style.display = show ? "" : "none";
      if (show && el.classList.contains("case")) n++;
    }});
    document.querySelectorAll(".viewbar button[data-ok]").forEach(function (b) {{ b.setAttribute("aria-pressed", String(b.dataset.ok === okSel)); }});
    document.querySelectorAll(".viewbar button[data-arm]").forEach(function (b) {{ b.setAttribute("aria-pressed", String(b.dataset.arm === armSel)); }});
    document.getElementById("cnt").textContent = n + " shown";
    try {{ localStorage.setItem("cmp-ok", okSel); localStorage.setItem("cmp-arm", armSel); }} catch (e) {{}}
  }}
  document.querySelectorAll(".viewbar button[data-ok]").forEach(function (b) {{ b.addEventListener("click", function () {{ okSel = b.dataset.ok; apply(); }}); }});
  document.querySelectorAll(".viewbar button[data-arm]").forEach(function (b) {{ b.addEventListener("click", function () {{ armSel = b.dataset.arm; apply(); }}); }});
  if (location.hash) {{ var t = document.getElementById(location.hash.slice(1)); if (t) {{ okSel = t.dataset.ok; armSel = "all"; }} }}
  apply();
}})();
</script>
"""
    path = out / out_name
    path.write_text(doc, encoding="utf-8")
    return path
