"""Before and after GEODE: only the learned work of GEODE itself, measured on held-out inputs.

    python tools/geode_before_after.py                                   # the campaign's store (built or reloaded)
    python tools/geode_before_after.py --store-dir ~/.cache/gmsjourney/stores/<hash>    # a store already built
    -> docs/GEODE_BEFORE_AFTER.html (+ .json with every number)

What counts as GEODE here: the two GEODE steps of the build (knowledge/kx_store.py), nothing after them.

1  The GEODE embed loop (build step 2, knowlytix GeodeEmbedLoop): it fine-tunes the meaning (v) encoder from the
   policy's own alias structure plus links found in the trained geometry, iterating until a pass adds nothing.
   Shown: held-out typo / speech-recognition variants of the policy's aliases (never trained on), each matched to
   the nearest entity by the raw encoder (before) and by GEODE's tuned encoder (after) - the build's own probes
   and rule (_noisy_probes, nearest prototype) - overall, per entity and per example; and the loop's own history
   (labels, geometry links added, held-out accuracy per pass), read from gms_build_report.json.
2  The GEODE self-correction loop (build step 1, knowlytix GeodeLoop): what its trained geometry questioned - the
   facts its duplicate resolution scored as cap outliers (with the distance), and what became of them - plus the
   loop's own counts (passes, convergence, geometric corrections). Read from gms_build_report.json.

Left out on purpose: the deterministic checks (type checks, prose-vs-register number checks, unbound sentences,
the exact-value anchor check, the injected "$150" misread), and everything after GEODE - the u-space contradiction
SFT (step 3), relevance calibration (4), the GMS store training (5) and the judge calibration (7).
knowlytix public APIs only (the calls the build itself makes).
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

E = lambda s: html.escape(str(s), quote=True)


# ============================================================================ measurements
def names_view(store, report: dict) -> dict:
    """Held-out noisy aliases: raw-encoder guess vs GEODE-tuned-encoder guess, per example."""
    import numpy as np
    from gmsjourney.knowledge.kx_store import _noisy_probes, _quiet, device
    with _quiet():
        from knowlytix.core.graph.encoders import encode_texts
    pairs = [(c, a) for c, als in store.aliases.items() for a in als]
    probes = _noisy_probes(pairs)
    source_alias = [a for c, a in pairs for _s in (101, 102, 103)]          # same order as _noisy_probes
    ft = store.v_ft
    labels = list(ft.label_order)
    texts, gold = [p for p, _ in probes], [c.lower() for _, c in probes]
    with _quiet():
        after, _ = ft.classify(texts)
        P = encode_texts([lab.replace("_", " ") for lab in labels], report["encoders"]["v"], device()).cpu().numpy()
        X = encode_texts(texts, report["encoders"]["v"], device()).cpu().numpy()
    before = [labels[int(np.argmax(P @ x))] for x in X]
    rows = [{"text": t, "alias": s, "gold": g, "before": b, "after": a}
            for t, s, g, b, a in zip(texts, source_alias, gold, before, after)]
    acc = lambda rs, k: float(np.mean([r[k] == r["gold"] for r in rs])) if rs else 0.0
    per = {g: {"n": len(rs), "before": acc(rs, "before"), "after": acc(rs, "after")}
           for g in sorted({r["gold"] for r in rows}) for rs in [[r for r in rows if r["gold"] == g]]}
    return {"rows": rows, "before": acc(rows, "before"), "after": acc(rows, "after"), "n": len(rows),
            "per_entity": per}


def loop_view(report: dict) -> dict:
    """What GEODE's self-correction loop questioned, from the build report (nothing recomputed)."""
    g = report.get("geode", {})
    rejected = {tuple(d["triple"]): d.get("decision") for d in g.get("duplicates_rejected", [])}
    mode = g.get("dedup_mode", "audit")
    props = []
    for d in g.get("duplicates_proposed", []):
        tr = tuple(d["triple"])
        outcome = ("kept: removal overruled, the source table states it verbatim" if tr in rejected else None) or ("removed (dedup_mode accept)" if mode == "accept" else
                                       "kept: dedup_mode off, proposals are reported only" if mode == "off" else "removed")
        props.append({"triple": list(tr), "reason": d.get("reason"), "cap_distance": d.get("cap_distance"),
                      "kept_tails": d.get("kept_tails", []), "outcome": outcome})
    if mode == "off" and isinstance(g.get("dedup_audit_only"), dict):
        for d in g["dedup_audit_only"].get("would_remove", []) or []:
            if isinstance(d, dict) and "triple" in d:
                props.append({"triple": list(d["triple"]), "reason": d.get("reason"), "cap_distance": d.get("cap_distance"),
                              "kept_tails": d.get("kept_tails", []), "outcome": "audit only on the trained store: not removed"})
    return {"converged": g.get("converged"), "iterations": g.get("iterations"), "max_iters": g.get("max_iters"),
            "n_proposed": g.get("n_proposed"), "n_kept": g.get("n_kept_fact_triples"),
            "n_corrections": len(g.get("corrections", []) or []), "corrections": g.get("corrections", []),
            "dedup_mode": mode, "proposals": props, "n_flagged": len(g.get("duplicates_flagged", []) or [])}


# ============================================================================ page
CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--surface-2:#f0efec;--line:#e1e0d9;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6f6d68;
--after:#2a78d6;--before:#9a988f;--ok:#006300;--bad:#c62f2f;--warnc:#eb6834;--font:"Segoe UI",system-ui,-apple-system,sans-serif;--mono:ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--surface-2:#242422;--line:#2c2c2a;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#9a988f;
--after:#3987e5;--before:#6f6d68;--ok:#0ca30c;--bad:#e66767;--warnc:#d95926}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--surface-2:#242422;--line:#2c2c2a;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#9a988f;
--after:#3987e5;--before:#6f6d68;--ok:#0ca30c;--bad:#e66767;--warnc:#d95926}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);line-height:1.45}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:clamp(1.6rem,3vw,2.3rem);margin:0 0 .3rem;letter-spacing:-.01em}
h2{font-size:1.35rem;margin:2.4rem 0 .3rem}
h3{font-size:1.05rem;margin:1.6rem 0 .2rem}
.sub{color:var(--ink-2);max-width:85ch;margin:0}
.meta{color:var(--muted);font-size:.85rem;margin-top:.5rem;font-family:var(--mono)}
.warn{border-left:3px solid var(--warnc);background:var(--surface);padding:.5rem .8rem;margin:.8rem 0;font-size:.9rem;color:var(--ink-2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:1rem 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.9rem 1rem}
.tile .k{color:var(--muted);font-size:.82rem}
.tile .v{font-size:1.9rem;font-weight:700;margin-top:.15rem}
.tile .v small{font-size:1rem;color:var(--muted);font-weight:500}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1rem;margin-top:.8rem;overflow-x:auto}
.legend{display:flex;gap:1.2rem;font-size:.85rem;color:var(--ink-2);margin-bottom:.6rem}
.legend i{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:.35rem;vertical-align:middle}
.ent{display:grid;grid-template-columns:minmax(160px,220px) 1fr;gap:.25rem 1rem;align-items:center}
.ent .name{font-family:var(--mono);font-size:.85rem}
.ent .name small{color:var(--muted);font-family:var(--font)}
.pair{display:grid;gap:3px}
.pair div{height:11px;border-radius:0 3px 3px 0;position:relative}
.pair .b{background:var(--before)} .pair .a{background:var(--after)}
.pair span{position:absolute;left:calc(100% + 6px);top:-3px;font-size:.75rem;color:var(--ink-2);font-family:var(--mono);white-space:nowrap}
.ent .row{padding:.3rem 0;border-bottom:1px dashed var(--line)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:.35rem .6rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:.8rem}
td.num{font-family:var(--mono);text-align:right}
td.mono{font-family:var(--mono);font-size:.82rem}
.yes{color:var(--ok);font-weight:600}.no{color:var(--bad);font-weight:600}
summary{cursor:pointer;color:var(--ink-2);margin:.3rem 0}
.note{color:var(--muted);font-size:.85rem;margin-top:.5rem}
footer{margin-top:3rem;color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:1rem}
button{font:inherit;font-size:.85rem;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:8px;padding:.3rem .7rem;cursor:pointer;float:right}
"""


def per_entity_html(per: dict) -> str:
    rows = []
    for g, v in sorted(per.items(), key=lambda kv: kv[1]["after"] - kv[1]["before"], reverse=True):
        rows.append(f'<div class="name row">{E(g)} <small>({v["n"]} variants)</small></div>'
                    f'<div class="pair row"><div class="b" style="width:{max(.5, 85 * v["before"]):.1f}%"><span>'
                    f'{100 * v["before"]:.0f}%</span></div><div class="a" style="width:{max(.5, 85 * v["after"]):.1f}%">'
                    f'<span>{100 * v["after"]:.0f}%</span></div></div>')
    return ('<div class="legend"><span><i style="background:var(--before)"></i>before: raw encoder</span>'
            '<span><i style="background:var(--after)"></i>after: GEODE-tuned encoder</span></div>'
            f'<div class="ent">{"".join(rows)}</div>')


def examples_html(rows: list[dict], n: int = 16) -> str:
    fixed = [r for r in rows if r["before"] != r["gold"] and r["after"] == r["gold"]]
    broke = [r for r in rows if r["before"] == r["gold"] and r["after"] != r["gold"]]
    still = [r for r in rows if r["before"] != r["gold"] and r["after"] != r["gold"]]
    both = len(rows) - len(fixed) - len(broke) - len(still)
    mark = lambda r, k: f'<span class="{"yes" if r[k] == r["gold"] else "no"}">{E(r[k])}</span>'
    line = lambda r: (f'<tr><td class="mono">{E(r["text"])}</td><td class="mono">{E(r.get("alias", ""))}</td>'
                      f'<td class="mono">{E(r["gold"])}</td><td class="mono">{mark(r, "before")}</td>'
                      f'<td class="mono">{mark(r, "after")}</td></tr>')
    head = ('<table><thead><tr><th>held-out input</th><th>alias it was made from</th><th>should match</th>'
            '<th>before</th><th>after</th></tr></thead><tbody>')
    return (f'<p class="note">{len(fixed)} fixed by GEODE, {len(broke)} made worse, {len(still)} wrong either way, '
            f'{both} right either way.</p>{head}{"".join(line(r) for r in fixed[:n])}</tbody></table>'
            f'<details><summary>made worse ({len(broke)}) and still wrong ({len(still)})</summary>{head}'
            f'{"".join(line(r) for r in broke + still)}</tbody></table></details>')


def render(data: dict) -> str:
    n = data["names"]
    hist = data.get("embed_history") or []
    hist_rows = "".join(f'<tr><td class="num">{h.get("iter")}</td><td class="num">{h.get("n_classes")}</td>'
                        f'<td class="num">{h.get("n_labels")}</td><td class="num">{h.get("geometry_links_added")}</td>'
                        f'<td class="num">{h.get("pseudo_labeled")}</td><td class="num">'
                        f'{100 * h["heldout_acc"]:.0f}%</td></tr>' if isinstance(h.get("heldout_acc"), (int, float)) else ""
                        for h in hist)
    sec1 = (f'<h2>1 · GEODE embed loop: matching messy names to the right entity</h2>'
            f'<p class="sub">GEODE fine-tunes the meaning encoder from the policy\'s own aliases plus links it finds in '
            f'the trained geometry. The test inputs below are typo and speech-recognition variants of those aliases that '
            f'were never trained on. Before: the raw encoder picks the nearest entity name. After: GEODE\'s tuned '
            f'encoder.</p>'
            f'<div class="tiles"><div class="tile"><div class="k">held-out variants matched to the right entity</div>'
            f'<div class="v">{100 * n["before"]:.0f}% → {100 * n["after"]:.0f}%</div></div>'
            f'<div class="tile"><div class="k">held-out variants tested</div><div class="v">{n["n"]}</div></div>'
            + (f'<div class="tile"><div class="k">links GEODE added from the trained geometry</div><div class="v">'
               f'{sum(h.get("geometry_links_added") or 0 for h in hist)}</div></div>' if hist else "")
            + '</div>'
            + (f'<h3>By entity</h3><div class="card">{per_entity_html(n["per_entity"])}</div>' if n.get("per_entity") else "")
            + (f'<h3>Examples</h3><div class="card">{examples_html(n["rows"])}</div>' if n.get("rows") else
               f'<p class="note">{E(n.get("note", ""))}</p>')
            + (f'<h3>The loop, pass by pass</h3><div class="card"><table><thead><tr><th>pass</th><th>entities</th>'
               f'<th>labelled surfaces</th><th>geometry links added</th><th>pseudo-labelled</th><th>held-out accuracy</th>'
               f'</tr></thead><tbody>{hist_rows}</tbody></table><p class="note">It stops when a pass adds nothing new.</p>'
               f'</div>' if hist_rows else ""))
    sec2 = ""
    lp = data.get("loop")
    if lp:
        props = lp["proposals"]
        prow = "".join(f'<tr><td class="mono">{E(" ".join(p["triple"]))}</td><td class="num">'
                       f'{p["cap_distance"]:.3f}</td><td class="mono">{E(", ".join(map(str, p["kept_tails"][:6])))}'
                       f'{" …" if len(p["kept_tails"]) > 6 else ""}</td><td>{E(p["outcome"])}</td></tr>'
                       for p in sorted(props, key=lambda p: -(p["cap_distance"] or 0)))
        sec2 = (f'<h2>2 · GEODE self-correction loop: what its trained geometry questioned</h2>'
                f'<p class="sub">The loop trains on the proposed facts, then asks the trained geometry which facts do '
                f'not fit. Its duplicate resolution flags a fact as a <i>cap outlier</i> when, among several facts sharing '
                f'the same head and relation, it sits far outside the relation\'s learned cap.</p>'
                f'<div class="tiles"><div class="tile"><div class="k">passes (until nothing changed)</div><div class="v">'
                f'{lp["iterations"]}<small>/{lp["max_iters"]}</small></div></div>'
                f'<div class="tile"><div class="k">facts proposed → kept</div><div class="v">{lp["n_proposed"]} → '
                f'{lp["n_kept"]}</div></div>'
                f'<div class="tile"><div class="k">geometric corrections applied</div><div class="v">{lp["n_corrections"]}'
                f'</div></div><div class="tile"><div class="k">facts questioned as cap outliers</div><div class="v">'
                f'{len(props)}</div></div></div>'
                + (f'<div class="card"><table><thead><tr><th>fact questioned</th><th>cap distance</th><th>the other tails '
                   f'it was compared with</th><th>outcome</th></tr></thead><tbody>{prow}</tbody></table>'
                   f'<p class="note">Duplicate resolution assumes one value per head and relation. Where the policy '
                   f'states several (a state applying several rules), the build keeps the questioned fact because the '
                   f'source states it verbatim (dedup mode {E(lp["dedup_mode"])}).</p></div>' if props else ""))
    warn = ""
    if data.get("encoders_kind") == "offline-stand-in":
        warn = ('<div class="warn">These numbers come from the offline stand-in encoders (the HuggingFace models were '
                'not reachable when this store was built). Re-run where the real MiniLM / mpnet encoders are available '
                'before presenting them.</div>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEODE before and after</title><style>{CSS}</style></head><body><main>
<button id="theme" type="button">Light / dark</button>
<h1>What GEODE learns</h1>
<p class="sub">Only the learned work of GEODE, measured on held-out inputs. Not shown: the deterministic checks of the
build, and the steps after GEODE (contradiction fine-tuning, store training, calibration).</p>
<div class="meta">{E(data["meta"])}</div>{warn}
{sec1}{sec2}
<footer>{E(data["method"])}</footer></main>
<script>
document.getElementById("theme").onclick = () => {{ const r = document.documentElement, dark = getComputedStyle(r).colorScheme.includes("dark");
  r.dataset.theme = dark ? "light" : "dark"; }};
</script></body></html>"""


# ============================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default=str(ROOT / "gmsjourney" / "policy" / "charge_review_kg.json"))
    ap.add_argument("--store-dir", default=None, help="a built knowlytix store folder (default: the campaign's store)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "docs" / "GEODE_BEFORE_AFTER.html"))
    a = ap.parse_args()
    import importlib.util
    missing = [m for m in ("numpy", "torch", "knowlytix") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"This Python ({sys.executable}) is missing: {', '.join(missing)}.\n"
              "Run the script with the same Python as the campaign / notebook kernel, e.g. activate its environment "
              "first, or from a notebook cell:  !{sys.executable} tools/geode_before_after.py")
        return 2
    from gmsjourney import backend
    policy = Path(a.policy)
    data: dict = {}
    if not backend.available():
        print(f"knowlytix is not available in this Python [{sys.executable}] (needs knowlytix, proofloop, forgeloop "
              "and a licence: KNOWLYTIX_LICENSE_KEY or ~/.knowlytix/license.key).")
        return 2
    from gmsjourney.knowledge.kx_store import load_kx_store
    from gmsjourney.pipeline import build_gms
    if a.store_dir:
        sdir = Path(a.store_dir).expanduser()
        print(f"loading the store from {sdir}")
        kb = load_kx_store(policy, sdir)
        store, report = kb.store, kb.report
    else:
        store, report, _info = build_gms(policy, seed=a.seed)
        sdir = Path(report["store_dir"])
    print("GEODE embed loop: held-out noisy names, raw vs tuned encoder")
    data["names"] = names_view(store, report)
    el = report.get("embed_loop", {})
    data["embed_history"] = el.get("history", [])
    rb, ra = el.get("before"), el.get("after")
    if rb is not None and (abs(rb - data["names"]["before"]) > 1e-9 or abs(ra - data["names"]["after"]) > 1e-9):
        print(f"note: this measurement ({data['names']['before']:.3f} -> {data['names']['after']:.3f}) differs from "
              f"the build report ({rb:.3f} -> {ra:.3f})")
    print("GEODE self-correction loop: from the build report")
    data["loop"] = loop_view(report)
    data["encoders_kind"] = report["encoders"]["kind"]
    mode = el.get("mode") or "rotation"
    data["meta"] = (f"knowlytix store {sdir.name} · encoders: {report['encoders']['kind']} · embed loop: "
                    f"{el.get('iterations')} passes, {mode} mode, {el.get('n_canonicals')} entities")
    data["method"] = ("Section 1: the build's own held-out probes (_noisy_probes: typo and ASR variants of every "
                      "alias, seeds 101-103) and rule (nearest entity prototype), raw v encoder vs the GEODE-tuned "
                      "encoder saved with the store (tuned_encoder/); the pass-by-pass table is the embed loop's own "
                      "history. Section 2: the GEODE loop's output as recorded in gms_build_report.json.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out} and {out.with_suffix('.json').name}: held-out names "
          f"{100 * data['names']['before']:.0f}% -> {100 * data['names']['after']:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
