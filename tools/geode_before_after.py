"""Before and after GEODE training: what the learned parts of the GMS can do that the untrained ones cannot.

    python tools/geode_before_after.py                                  # the campaign's store (built or reloaded)
    python tools/geode_before_after.py --store-dir ~/.cache/gmsjourney/stores/<hash>   # a store already built
    python tools/geode_before_after.py --curve 1,10,40                  # also train twins for a progression
    -> docs/GEODE_BEFORE_AFTER.html (+ .json with every number)

Only the learned work is shown; the deterministic clean-up (type checks, prose-vs-register number checks,
unbound sentences, the injected "$150" misread) is left out, since it needs no training.

1  Telling a true fact from a believable wrong one (the trained store, step 5 of the build).
   For every exact-number fact of the policy, the true value and a real value from elsewhere in the policy
   (another option's fee, another limit) are scored by the store - the geodesic distance from the relation's
   learned cap - and compared with that relation's cut (its cap radius). "Before" is a twin store built from
   the SAME facts, SAME encoders and SAME warm-start vectors, trained for --before-epochs (default 1) instead
   of the full run; "after" is the trained store. Per relation, the realised true-accept / false-accept rates
   are shown for both.
2  Recognising messy names (v-space SFT): held-out typo / ASR variants of the policy's aliases, nearest entity
   with the raw encoder vs the tuned encoder - the same probes and rule as the build's own evaluation.
3  Routing a question to its attribute (u-space contradiction SFT): two-fold held-out phrasings, nearest
   attribute with the raw encoder vs an encoder fine-tuned on the other fold (as the build's evaluation;
   --skip-u to leave this out, it retrains the encoder twice).

Every figure is measured on this run; if training does not help a fact, the page shows that.
knowlytix backend: public APIs only (the calls the build itself makes). Built-in backend: section 1 compares the
warm-start store with the trained one; sections 2-3 report the built-in training's own before/after figures.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

E = lambda s: html.escape(str(s), quote=True)


# ============================================================================ measurements
def facts_view(store) -> dict:
    """Section 1 for one store: the true-vs-substituted rows and the per-relation accept rates."""
    from gmsjourney.knowledge.geode import oracle_property
    rows = []
    for r in oracle_property(store):
        rows.append({k: r[k] for k in ("head", "relation", "true", "fabricated", "score_true", "score_fabricated",
                                        "tau", "calibrated_ok")})
    per_rel = {}
    for rel, c in (store.calibration.get("score_triple") or {}).items():
        per_rel[rel] = {k: c.get(k) for k in ("tau", "tar", "far", "n_pos", "n_neg")}
    return {"rows": rows, "per_relation": per_rel}


def kx_twin(store_dir: Path, report: dict, epochs: int, seed: int, policy: Path, log=print):
    """A knowlytix store built from the trained store's own facts, encoders and warm-start vectors, trained for
    `epochs` epochs - the calls build_kx_store makes in its step 5, with only the epoch count changed."""
    import tempfile
    import torch
    from gmsjourney.knowledge.kx_store import KXPolicyStore, _quiet
    from gmsjourney.knowledge.geode import build_store
    with _quiet():
        from knowlytix.core.config import EmbeddingConfig
        from knowlytix.knowledge.config import DocGMSConfig
        from knowlytix.knowledge.geode.rag import store_from_triples
    enc = report["encoders"]
    triples = [tuple(t) for t in json.loads((store_dir / "kx" / "triples.json").read_text())]
    tmp = Path(tempfile.mkdtemp(prefix=f"geode_twin_e{epochs}_"))
    cfg = DocGMSConfig(store_path=str(tmp / "kx"), ingest_mode="regex", loss_mode="cap")
    cfg.train.epochs, cfg.train.device, cfg.train.seed = epochs, "cpu", seed
    cfg.embedding = EmbeddingConfig(semantic_model=enc["v"], logical_model=enc["u"], warm_start=True,
                                    freeze_base=True, dim_policy="project", device="cpu")
    cfg.embedding.v_vectors_path = str(store_dir / "v_emb.pt")
    random.seed(seed)
    torch.manual_seed(seed)
    t = time.time()
    with _quiet():
        kx0 = store_from_triples(str(store_dir / "policy.gms.md"), triples, cfg, torch.device("cpu"),
                                 ingest_mode="regex")
    log(f"      twin store, {epochs} epoch(s): {len(triples)} facts ({time.time() - t:.0f}s)")
    compiled = build_store(policy, train=False).store
    return KXPolicyStore.from_compiled(compiled, kx0, None, None, None, enc)


def names_view(store, report: dict) -> dict:
    """Section 2 (knowlytix): every held-out noisy alias, raw-encoder guess vs tuned-encoder guess."""
    import numpy as np
    from gmsjourney.knowledge.kx_store import _noisy_probes, _quiet
    with _quiet():
        from knowlytix.core.graph.encoders import encode_texts
    pairs = [(c, a) for c, als in store.aliases.items() for a in als]
    probes = _noisy_probes(pairs)
    ft = store.v_ft
    labels = list(ft.label_order)
    texts, gold = [p for p, _ in probes], [c.lower() for _, c in probes]
    with _quiet():
        after, _ = ft.classify(texts)
        P = encode_texts([lab.replace("_", " ") for lab in labels], report["encoders"]["v"], "cpu").numpy()
        X = encode_texts(texts, report["encoders"]["v"], "cpu").numpy()
    before = [labels[int(np.argmax(P @ x))] for x in X]
    rows = [{"text": t, "gold": g, "before": b, "after": a} for t, g, b, a in zip(texts, gold, before, after)]
    acc = lambda k: float(np.mean([r[k] == r["gold"] for r in rows])) if rows else 0.0
    return {"rows": rows, "before": acc("before"), "after": acc("after"), "n": len(rows)}


def routing_view(report: dict, epochs: int) -> dict:
    """Section 3 (knowlytix): two-fold held-out phrasings, raw-encoder centroid vs fine-tuned encoder."""
    import numpy as np
    from gmsjourney.knowledge.kx_store import _attribute_phrasings, _quiet
    with _quiet():
        from knowlytix.core.graph.encoders import encode_texts
        from knowlytix.knowledge.geode.embed_loop import contradiction_sft
    base = report["encoders"]["u"]
    groups = {g: p for g, p in _attribute_phrasings().items() if len(p) >= 4}
    rows = []
    for fold in (0, 1):
        tr = {g: [x for i, x in enumerate(p) if i % 2 != fold] for g, p in groups.items()}
        te = [(x, g) for g, p in groups.items() for i, x in enumerate(p) if i % 2 == fold]
        with _quiet():
            ft = contradiction_sft(tr, base_model=base, epochs=epochs)
            after, _ = ft.classify([x for x, _ in te])
            cen = {g: encode_texts(p, base, "cpu").numpy().mean(0) for g, p in tr.items()}
            X = encode_texts([x for x, _ in te], base, "cpu").numpy()
        names = list(cen)
        C = np.stack([cen[g] / np.linalg.norm(cen[g]) for g in names])
        before = [names[int(np.argmax(C @ x))] for x in X]
        rows += [{"text": x, "gold": g, "before": b, "after": a, "fold": fold}
                 for (x, g), b, a in zip(te, before, after)]
    acc = lambda k: float(np.mean([r[k] == r["gold"] for r in rows])) if rows else 0.0
    return {"rows": rows, "before": acc("before"), "after": acc("after"), "n": len(rows), "epochs": epochs}


# ============================================================================ page
CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--surface-2:#f0efec;--line:#e1e0d9;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6f6d68;
--true:#2a78d6;--fake:#eb6834;--ok:#006300;--bad:#c62f2f;--zone:rgba(42,120,214,.10);--font:"Segoe UI",system-ui,-apple-system,sans-serif;--mono:ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--surface-2:#242422;--line:#2c2c2a;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#9a988f;
--true:#3987e5;--fake:#d95926;--ok:#0ca30c;--bad:#e66767;--zone:rgba(57,135,229,.14)}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--surface-2:#242422;--line:#2c2c2a;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#9a988f;
--true:#3987e5;--fake:#d95926;--ok:#0ca30c;--bad:#e66767;--zone:rgba(57,135,229,.14)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);line-height:1.45}
main{max-width:1240px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:clamp(1.6rem,3vw,2.3rem);margin:0 0 .3rem;letter-spacing:-.01em}
h2{font-size:1.35rem;margin:2.4rem 0 .3rem}
.sub{color:var(--ink-2);max-width:85ch;margin:0}
.meta{color:var(--muted);font-size:.85rem;margin-top:.5rem;font-family:var(--mono)}
.warn{border-left:3px solid var(--fake);background:var(--surface);padding:.5rem .8rem;margin:.8rem 0;font-size:.9rem;color:var(--ink-2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:1rem 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.9rem 1rem}
.tile .k{color:var(--muted);font-size:.82rem}
.tile .v{font-size:1.9rem;font-weight:700;margin-top:.15rem}
.tile .v small{font-size:1rem;color:var(--muted);font-weight:500}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1rem;margin-top:.8rem;overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;font-size:.85rem;color:var(--ink-2);margin:.4rem 0 .6rem}
.legend svg{vertical-align:middle;margin-right:.35rem}
svg text{fill:var(--ink-2);font-family:var(--font)}
svg .lab{fill:var(--ink);font-size:12.5px}
svg .lab2{fill:var(--muted);font-size:11px;font-family:var(--mono)}
svg .ptitle{fill:var(--ink);font-size:13px;font-weight:700}
svg .axis{stroke:var(--line)}
svg .tick text{font-size:10.5px;fill:var(--muted)}
svg .zone{fill:var(--zone)}
svg .cut{stroke:var(--ink-2);stroke-width:1.5;stroke-dasharray:3 2}
svg .bar{stroke:var(--line);stroke-width:2}
svg .t{fill:var(--true);stroke:var(--surface);stroke-width:2}
svg .f{fill:var(--fake);stroke:var(--surface);stroke-width:2}
svg .okv{fill:var(--ok);font-weight:700;font-size:13px}
svg .badv{fill:var(--bad);font-weight:700;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
summary{cursor:pointer;color:var(--ink-2);margin:.3rem 0}
th,td{padding:.35rem .6rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:.8rem}
td.num{font-family:var(--mono);text-align:right}
td.mono{font-family:var(--mono);font-size:.82rem}
.yes{color:var(--ok);font-weight:600}.no{color:var(--bad);font-weight:600}
.bars{display:grid;grid-template-columns:90px 1fr 60px;gap:.4rem .8rem;align-items:center;max-width:620px;margin:.6rem 0}
.bars .b{height:16px;border-radius:0 4px 4px 0;background:var(--true)}
.bars .b.before{background:var(--muted)}
.note{color:var(--muted);font-size:.85rem;margin-top:.5rem}
footer{margin-top:3rem;color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:1rem}
button{font:inherit;font-size:.85rem;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:8px;padding:.3rem .7rem;cursor:pointer;float:right}
"""


def dumbbells(before_rows: list[dict], after_rows: list[dict], before_label: str, after_label: str) -> str:
    """One row per fact; left panel before, right panel after, on one shared distance scale."""
    key = lambda r: (r["head"], r["relation"], r["true"])
    after_by = {key(r): r for r in after_rows}
    rows = [(b, after_by.get(key(b))) for b in before_rows if key(b) in after_by]
    vals = [v for b, a in rows for r in (b, a) for v in (r["score_true"], r["score_fabricated"], r["tau"])]
    xmax = max(vals + [0.1]) * 1.06
    lw, pw, gap, rh, top = 300, 380, 46, 34, 46
    W, H = lw + 2 * pw + gap + 40, top + rh * len(rows) + 30
    sx = lambda v, x0: x0 + 10 + (pw - 20) * v / xmax
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="True and substituted values, distance '
           f'to each relation\'s cut, before and after training">']
    for i, (title, x0) in enumerate(((before_label, lw), (after_label, lw + pw + gap))):
        out.append(f'<text class="ptitle" x="{x0 + 10}" y="18">{E(title)}</text>')
        y_axis = top + rh * len(rows) + 4
        out.append(f'<line class="axis" x1="{x0 + 10}" x2="{x0 + pw - 10}" y1="{y_axis}" y2="{y_axis}"/>')
        for k in range(5):
            v = xmax * k / 4
            out.append(f'<g class="tick"><text x="{sx(v, x0):.1f}" y="{y_axis + 15}" text-anchor="middle">{v:.2f}</text></g>')
        out.append(f'<text class="lab2" x="{x0 + pw - 10}" y="36" text-anchor="end">distance from the relation →</text>')
    for j, (b, a) in enumerate(rows):
        y = top + rh * j + rh / 2
        out.append(f'<text class="lab" x="0" y="{y - 2}">{E(b["head"])}</text>')
        out.append(f'<text class="lab2" x="0" y="{y + 12}">{E(b["relation"])}: {E(b["true"])} vs {E(b["fabricated"])}</text>')
        for r, x0 in ((b, lw), (a, lw + pw + gap)):
            tt, tf, tau = r["score_true"], r["score_fabricated"], r["tau"]
            out.append(f'<rect class="zone" x="{x0 + 10}" y="{y - rh / 2 + 3}" width="{max(0, sx(tau, x0) - x0 - 10):.1f}" '
                       f'height="{rh - 6}"/>')
            out.append(f'<line class="cut" x1="{sx(tau, x0):.1f}" x2="{sx(tau, x0):.1f}" y1="{y - rh / 2 + 3}" y2="{y + rh / 2 - 3}"/>')
            out.append(f'<line class="bar" x1="{sx(tt, x0):.1f}" x2="{sx(tf, x0):.1f}" y1="{y}" y2="{y}"/>')
            out.append(f'<circle class="t" cx="{sx(tt, x0):.1f}" cy="{y}" r="7"><title>true value {E(r["true"])}: '
                       f'{tt:.3f} (cut {tau:.3f})</title></circle>')
            d = 7.5
            fx = sx(tf, x0)
            out.append(f'<path class="f" d="M{fx:.1f},{y - d} L{fx + d:.1f},{y} L{fx:.1f},{y + d} L{fx - d:.1f},{y} Z">'
                       f'<title>substituted value {E(r["fabricated"])}: {tf:.3f} (cut {tau:.3f})</title></path>')
            ok = r["calibrated_ok"]
            out.append(f'<text class="{"okv" if ok else "badv"}" x="{x0 + pw - 4}" y="{y + 4}" text-anchor="end">'
                       f'{"✓" if ok else "✗"}</text>')
    out.append("</svg>")
    return "".join(out)


def legend() -> str:
    return ('<div class="legend">'
            '<span><svg width="16" height="16"><circle cx="8" cy="8" r="6" fill="var(--true)"/></svg>true value (from the policy)</span>'
            '<span><svg width="16" height="16"><path d="M8,1 L15,8 L8,15 L1,8Z" fill="var(--fake)"/></svg>substituted value '
            '(a real figure from elsewhere in the policy)</span>'
            '<span><svg width="26" height="16"><rect x="0" y="2" width="16" height="12" fill="var(--zone)"/>'
            '<line x1="16" x2="16" y1="0" y2="16" stroke="var(--ink-2)" stroke-dasharray="3 2"/></svg>accepted zone and the '
            'relation\'s cut</span>'
            '<span><b class="yes">✓</b> true inside, substituted outside · <b class="no">✗</b> not separated</span></div>')


def bars(before: float, after: float, unit: str = "") -> str:
    return (f'<div class="bars"><span>before</span><div class="b before" style="width:{100 * before:.1f}%"></div>'
            f'<span class="num">{100 * before:.0f}%</span><span>after</span><div class="b" style="width:{100 * after:.1f}%"></div>'
            f'<span class="num">{100 * after:.0f}%</span></div>')


def examples_table(rows: list[dict], n: int = 14) -> str:
    fixed = [r for r in rows if r["before"] != r["gold"] and r["after"] == r["gold"]]
    broke = [r for r in rows if r["before"] == r["gold"] and r["after"] != r["gold"]]
    still = [r for r in rows if r["before"] != r["gold"] and r["after"] != r["gold"]]
    show = fixed[:n] + broke[:4] + still[:4]
    mark = lambda r, k: f'<span class="{"yes" if r[k] == r["gold"] else "no"}">{E(r[k])}</span>'
    body = "".join(f'<tr><td class="mono">{E(r["text"])}</td><td class="mono">{E(r["gold"])}</td>'
                   f'<td class="mono">{mark(r, "before")}</td><td class="mono">{mark(r, "after")}</td></tr>' for r in show)
    return (f'<p class="note">{len(fixed)} fixed by training, {len(broke)} broken by it, {len(still)} wrong either way, '
            f'{len(rows) - len(fixed) - len(broke) - len(still)} right either way. Shown: fixed first, then any broken, '
            f'then some still wrong.</p><table><thead><tr><th>input</th><th>should be</th><th>before</th><th>after</th>'
            f'</tr></thead><tbody>{body}</tbody></table>')


def render(data: dict) -> str:
    f = data["facts"]
    bl, al = f["before_label"], f["after_label"]
    rows_b, rows_a = f["before"]["rows"], f["after"]["rows"]
    sep_b, sep_a = sum(r["calibrated_ok"] for r in rows_b), sum(r["calibrated_ok"] for r in rows_a)
    marg = lambda rows: sum(r["score_fabricated"] - r["score_true"] for r in rows) / max(1, len(rows))
    closer = lambda rows: sum(r["score_true"] < r["score_fabricated"] for r in rows)
    avg = lambda w, k: sum(f[w]["per_relation"][r][k] for r in rels) / max(1, len(rels))
    rels = sorted(set(f["before"]["per_relation"]) & set(f["after"]["per_relation"]))
    rel_rows = "".join(
        f'<tr><td class="mono">{E(r)}</td>'
        + "".join(f'<td class="num">{f[w]["per_relation"][r][k]:.2f}</td>' for w in ("before", "after")
                  for k in ("tar", "far"))
        + f'<td class="num">{f["after"]["per_relation"][r]["n_pos"]}</td><td class="num">{f["after"]["per_relation"][r]["n_neg"]}</td></tr>'
        for r in rels)
    curve = ""
    if data.get("curve"):
        pts = "".join(f'<tr><td class="num">{c["epochs"]}</td><td class="num">{c["separated"]}/{c["n"]}</td>'
                      f'<td class="num">{c["mean_margin"]:+.3f}</td></tr>' for c in data["curve"])
        curve = (f'<h3>Progress with training</h3><div class="card"><table><thead><tr><th>epochs</th><th>facts separated</th>'
                 f'<th>mean gap (substituted − true)</th></tr></thead><tbody>{pts}</tbody></table></div>')
    sec2 = sec3 = ""
    if data.get("names"):
        n = data["names"]
        sec2 = (f'<h2>2 · Recognising messy names</h2><p class="sub">Typo and speech-recognition variants of the policy\'s '
                f'own aliases (never seen in training), matched to the nearest entity: raw encoder before, tuned encoder '
                f'after.</p><div class="card">{bars(n["before"], n["after"])}'
                + (examples_table(n["rows"]) if n.get("rows") else f'<p class="note">{E(n.get("note", ""))}</p>') + '</div>')
    if data.get("routing"):
        u = data["routing"]
        sec3 = (f'<h2>3 · Routing a question to its attribute</h2><p class="sub">Phrasings of the same question (a fee, '
                f'a delivery time, a limit…), half held out each time: which attribute each held-out phrasing is routed '
                f'to, raw encoder before, fine-tuned encoder after.</p><div class="card">{bars(u["before"], u["after"])}'
                + (examples_table(u["rows"]) if u.get("rows") else f'<p class="note">{E(u.get("note", ""))}</p>') + '</div>')
    warn = ""
    if data.get("encoders_kind") == "offline-stand-in":
        warn = ('<div class="warn">These numbers come from the offline stand-in encoders (the HuggingFace models were '
                'not reachable when this store was built). Re-run where the real MiniLM / mpnet encoders are available '
                'before presenting them.</div>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEODE before and after</title><style>{CSS}</style></head><body><main>
<button id="theme" type="button">Light / dark</button>
<h1>What GEODE training changes</h1>
<p class="sub">The learned parts of the policy store, measured on the same questions before and after training. The
deterministic clean-up of the build (type checks, number cross-checks, dropped sentences) is not shown: it needs no training.</p>
<div class="meta">{E(data["meta"])}</div>{warn}
<h2>1 · Telling a true fact from a believable wrong one</h2>
<p class="sub">Each row is an exact figure from the policy. The store scores the true value and a real figure taken from
elsewhere in the policy (another option's fee, another limit) by their distance from the relation; a value is accepted
when it falls inside the relation's cut. <b>{E(bl)}</b>: the same facts, encoders and starting vectors, without the full
training. <b>{E(al)}</b>: the store the campaign uses.</p>
<div class="tiles">
<div class="tile"><div class="k">facts where the true value is accepted and the substituted one rejected</div>
<div class="v">{sep_b}<small>/{len(rows_b)}</small> → {sep_a}<small>/{len(rows_a)}</small></div></div>
<div class="tile"><div class="k">facts where the true value is closer than the substituted one</div>
<div class="v">{closer(rows_b)}<small>/{len(rows_b)}</small> → {closer(rows_a)}<small>/{len(rows_a)}</small></div></div>
<div class="tile"><div class="k">average gap between substituted and true (larger = clearer)</div>
<div class="v">{marg(rows_b):+.2f} → {marg(rows_a):+.2f}</div></div>
</div>
<div class="card">{legend()}{dumbbells(rows_b, rows_a, bl, al)}</div>
{curve}
<h3>Every relation of the store</h3>
<p class="sub">True-accept rate (its own facts inside the cut) and false-accept rate (counterfactual tails inside the cut),
per relation, before and after.</p>
<div class="card"><p><b>Average over {len(rels)} relations:</b> true-accept {avg("before", "tar"):.2f} → {avg("after", "tar"):.2f},
false-accept {avg("before", "far"):.2f} → {avg("after", "far"):.2f}.</p>
<details><summary>Every relation</summary><table><thead><tr><th>relation</th><th>true-accept before</th><th>false-accept before</th>
<th>true-accept after</th><th>false-accept after</th><th>facts</th><th>counterfactuals</th></tr></thead><tbody>{rel_rows}</tbody></table></details></div>
{sec2}{sec3}
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
    ap.add_argument("--backend", default="auto", choices=["auto", "knowlytix", "builtin"])
    ap.add_argument("--before-epochs", type=int, default=1, help="epochs of the 'before' twin store (knowlytix)")
    ap.add_argument("--curve", default="", help="extra twin epoch counts for a progression table, e.g. 1,10,40")
    ap.add_argument("--u-epochs", type=int, default=300, help="contradiction SFT epochs per fold (as the build)")
    ap.add_argument("--skip-u", action="store_true", help="leave out section 3 (it retrains the u encoder twice)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "docs" / "GEODE_BEFORE_AFTER.html"))
    a = ap.parse_args()
    from gmsjourney import backend
    policy = Path(a.policy)
    use_kx = a.backend != "builtin" and backend.available()
    data: dict = {}
    if use_kx:
        from gmsjourney.knowledge.kx_store import load_kx_store
        from gmsjourney.pipeline import build_gms
        if a.store_dir:
            sdir = Path(a.store_dir).expanduser()
            print(f"loading the trained store from {sdir}")
            kb = load_kx_store(policy, sdir)
            store, report = kb.store, kb.report
        else:
            store, report, _info = build_gms(policy, backend_choice="knowlytix", seed=a.seed)
            sdir = Path(report["store_dir"])
        epochs_after = report["store"]["epochs"]
        print("section 1: true vs substituted facts")
        after = facts_view(store)
        twin = kx_twin(sdir, report, a.before_epochs, a.seed, policy)
        before = facts_view(twin)
        curve = []
        for ep in [int(x) for x in a.curve.split(",") if x.strip()]:
            v = facts_view(kx_twin(sdir, report, ep, a.seed, policy))["rows"]
            curve.append({"epochs": ep, "n": len(v), "separated": sum(r["calibrated_ok"] for r in v),
                          "mean_margin": sum(r["score_fabricated"] - r["score_true"] for r in v) / max(1, len(v))})
        if curve:
            vb = before["rows"]
            curve.insert(0, {"epochs": a.before_epochs, "n": len(vb), "separated": sum(r["calibrated_ok"] for r in vb),
                             "mean_margin": sum(r["score_fabricated"] - r["score_true"] for r in vb) / max(1, len(vb))})
            curve.append({"epochs": epochs_after, "n": len(after["rows"]),
                          "separated": sum(r["calibrated_ok"] for r in after["rows"]),
                          "mean_margin": sum(r["score_fabricated"] - r["score_true"] for r in after["rows"]) / max(1, len(after["rows"]))})
        data["facts"] = {"before": before, "after": after,
                         "before_label": f"Before: {a.before_epochs} epoch{'s' if a.before_epochs != 1 else ''}",
                         "after_label": f"After: {epochs_after} epochs"}
        data["curve"] = curve
        print("section 2: noisy names")
        data["names"] = names_view(store, report)
        if not a.skip_u:
            print(f"section 3: routing (two folds x {a.u_epochs} epochs)")
            data["routing"] = routing_view(report, a.u_epochs)
        data["encoders_kind"] = report["encoders"]["kind"]
        data["meta"] = (f"knowlytix store {sdir.name} · {report['store']['n_entities']} entities, "
                        f"{report['store']['n_triples']} facts · encoders: {report['encoders']['kind']} · "
                        f"store trained {epochs_after} epochs (cap loss)")
        data["method"] = ("Before = a twin store built with the same facts (kx/triples.json), the same encoders and the "
                          "same warm-start vectors (v_emb.pt) as the trained store, trained for "
                          f"{a.before_epochs} epoch(s) instead of {epochs_after}. Substituted values come from "
                          "oracle_property (another entity's real figure on the same relation, else another limit of the "
                          "same head). Names: the build's own held-out probes and rule (_noisy_probes, nearest prototype). "
                          "Routing: the build's two-fold held-out evaluation, per phrasing.")
    else:
        from gmsjourney.knowledge.geode import build_store
        print("built-in backend: warm-start store vs trained store")
        b0 = build_store(policy, train=False)
        b1 = build_store(policy, train=True)
        data["facts"] = {"before": facts_view(b0.store), "after": facts_view(b1.store),
                         "before_label": "Before: warm start (encoders + Procrustes)",
                         "after_label": "After: trained (store + embed loop + u channel)"}
        tr = b1.training
        el, uc = tr.get("embed_loop", {}), tr.get("u_channel", {})
        ev = el.get("eval") or {}
        if "before" in ev and "after" in ev:
            data["names"] = {"before": ev["before"], "after": ev["after"],
                             "note": "built-in embed loop, its own held-out evaluation"}
        if "routing_acc_before" in uc:
            data["routing"] = {"before": uc["routing_acc_before"], "after": uc["routing_acc_after"],
                               "note": "built-in u channel, held-out paraphrase depth"}
        data["encoders_kind"] = "built-in hashing"
        data["meta"] = f"built-in store · {len(b1.store.triples)} facts"
        data["method"] = "Before = build_store(train=False); after = build_store(train=True), same policy and encoders."
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    fb, fa = data["facts"]["before"]["rows"], data["facts"]["after"]["rows"]
    print(f"wrote {out} and {out.with_suffix('.json').name}: separated {sum(r['calibrated_ok'] for r in fb)}/{len(fb)}"
          f" -> {sum(r['calibrated_ok'] for r in fa)}/{len(fa)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
