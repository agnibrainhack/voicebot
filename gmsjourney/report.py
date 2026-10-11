"""Self-contained HTML campaign report (one file, light and dark themes)."""
from __future__ import annotations

import html
import json
from typing import Any

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--good:#0ca30c;--goodtext:#006300;
--critical:#d03b3b;--warn:#b07600}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--good:#0ca30c;--goodtext:#0ca30c;--critical:#e66767;--warn:#fab219}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--good:#0ca30c;
--goodtext:#0ca30c;--critical:#e66767;--warn:#fab219}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}h3{font-size:15px;margin:20px 0 6px}
p.lede{color:var(--ink2);margin:0 0 16px}.muted{color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .v{font-size:26px;font-weight:600}.tile .l{color:var(--ink2);font-size:12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--goodtext);font-weight:600}.bad{color:var(--critical);font-weight:600}.warn{color:var(--warn);font-weight:600}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
details{margin:6px 0}summary{cursor:pointer;color:var(--ink2)}
.turn{border-left:3px solid var(--grid);padding:2px 10px;margin:4px 0}.turn.u{border-color:var(--s1)}
svg text{fill:var(--ink2);font-size:11px}svg .bar{fill:var(--s1)}svg .grid{stroke:var(--grid)}svg .ci{stroke:var(--ink)}
"""


def esc(x: Any) -> str:
    return html.escape(str(x))


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def tile(v: str, label: str) -> str:
    return f'<div class="tile"><div class="v">{esc(v)}</div><div class="l">{esc(label)}</div></div>'


def table(head: list[str], rows: list[list[Any]], num_cols: set[int] = frozenset()) -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="num">{c}</td>' if i in num_cols else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    import math
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


def hbar(items: list[tuple[str, float, str]], vmax: float | None = None, ci: list[tuple[float, float]] | None = None,
         fmt=lambda v: f"{v:.3f}", width: int = 680) -> str:
    """Horizontal single-series bar chart: per-bar hover titles, direct labels placed clear of any
    interval whisker, round tick values, and no upscaling of text on wide screens."""
    if not items:
        return ""
    raw = vmax or max([v for _, v, _ in items] + ([hi for _, hi in ci] if ci else []))
    vmax = _nice_max(raw)
    lw, bh, gap = 200, 16, 9
    plot = width - lw - 64
    h = len(items) * (bh + gap) + 26
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px" role="img">']
    for k in range(5):
        x = lw + plot * k / 4
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{h - 20}"/>'
                   f'<text x="{x:.1f}" y="{h - 6}" text-anchor="middle">{fmt(vmax * k / 4)}</text>')
    for i, (label, v, tip) in enumerate(items):
        y = i * (bh + gap) + 4
        w = max(2.0, plot * v / vmax)
        out.append(f'<text x="{lw - 8}" y="{y + 12}" text-anchor="end">{esc(label)}</text>')
        out.append(f'<path class="bar" d="M{lw},{y} h{w - 4:.1f} q4,0 4,4 v{bh - 8} q0,4 -4,4 h-{w - 4:.1f} z">'
                   f'<title>{esc(tip)}</title></path>')
        end = lw + w
        if ci:
            lo, hi = ci[i]
            x1, x2 = lw + plot * lo / vmax, lw + plot * hi / vmax
            out.append(f'<line class="ci" x1="{x1:.1f}" x2="{x2:.1f}" y1="{y + bh / 2}" y2="{y + bh / 2}" stroke-width="1.5"/>'
                       f'<line class="ci" x1="{x2:.1f}" x2="{x2:.1f}" y1="{y + 3}" y2="{y + bh - 3}" stroke-width="1.5"/>')
            end = max(end, x2)
        out.append(f'<text x="{end + 6:.1f}" y="{y + 12}">{esc(fmt(v))}</text>')
    out.append("</svg>")
    return "".join(out)


def pfmt(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.4f}"


def render_html(R: dict[str, Any]) -> str:
    st, cov, inv, des = R["store"], R["coverage"], R["invariance"], R["design"]
    res, dg = R["results"], R["diagnosis"]
    s = inv["summary"]
    nc = inv["negative_controls"]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>Journey Test Report</title><style>{CSS}</style>"
             f"</head><body><main>"]
    parts.append("<h1>Charge-review bot: GMS-grounded journey test campaign</h1>")
    parts.append(f"<p class='lede'>System under test: <b>{esc(res['bot'])}</b>. Ground truth is computed from the "
                 f"policy store, never hand-labelled; every enriched case is proved equivalent to its base case "
                 f"before it is run. Elapsed {R.get('elapsed_seconds', '–')} s.</p>")
    mut = R.get("mutation", {})
    parts.append("<div class='tiles'>" +
                 tile(str(len(R["bases"])), "base cases (journey + policy question)") +
                 tile(str(s["scenarios"]), "enriched scenarios") +
                 tile(pct(s["acceptance_rate"]), "scenarios proved invariant") +
                 tile(pct(nc["_overall"]["detection_rate"]), "meaning-changing edits rejected") +
                 tile(pct(res["base_level"]["accuracy"]), "bot accuracy, base cases") +
                 tile(pct(res["enriched"]["accuracy"]), "bot accuracy, enriched") +
                 (tile(pct(mut["mutation_score"]["all_suites_combined"]), "seeded defects caught") if mut else "") +
                 "</div>")

    # 1 store
    parts.append("<h2>1. Knowledge store (oracle)</h2>")
    pk = R.get("policy_kg")
    if pk:
        cons = pk.get("consistency", {})
        parts.append(f"<p>Source: the policy graph <code>policy_kg.json</code> — {pk['triples']} head/relation/tail "
                     f"triples from <i>{esc(pk['document'])}</i> ({esc(pk['by_kind'])}), {pk['states']} dialog states "
                     f"({pk['prompt_states']} answered by the customer), {pk['transitions']} state transitions, "
                     f"{pk['rules']} rules; every triple quotes its source section verbatim. The GMS is trained on "
                     f"{pk['gms_triples']} of them, state layer included. Trained store carries every fact of the graph: "
                     f"<span class='{'ok' if cons.get('all_agree') else 'bad'}'>{'yes' if cons.get('all_agree') else 'no'}"
                     f"</span>.</p>")
    ex = st["primitive_examples"]
    bk = R.get("backend", {})
    if bk.get("selected") == "knowlytix":
        bd = st["build"]
        g, el, uc = bd["geode"], bd["embed_loop"], bd["contradiction_sft"]
        vers = ", ".join(f"{k} {v}" for k, v in (bk.get("versions") or {}).items())
        parts.append(f"<p>Built, trained, calibrated and judged by <b>knowlytix</b> ({esc(vers)}). Text encoders: "
                     f"<b>{esc(bd['encoders']['kind'])}</b> — {esc(bd['encoders']['reason'])}. The policy was compiled "
                     f"once and rendered as GEODE input (<code>policy.gms.md</code>); every ENM figure agrees between the "
                     f"compiler and knowlytix's ingest: <span class='{'ok' if bd['enm_agreement']['all_agree'] else 'bad'}'>"
                     f"{bd['enm_agreement']['n']}/{bd['enm_agreement']['n'] if bd['enm_agreement']['all_agree'] else '?'}</span>.</p>")
        parts.append("<div class='card'>" + table(["step (knowlytix API)", "result"], [
            ["GEODE self-correction (GeodeLoop)", f"dedup mode <b>{esc(g.get('dedup_mode', 'audit'))}</b>; converged={g['converged']} "
             f"in {g['iterations']}/{g['max_iters']} iterations; {g['n_proposed']} proposed, {g['n_kept_fact_triples']} fact triples "
             f"kept; {len(g['corrections'])} corrections; anchor violations <b>{g.get('n_anchor_violations', len(g.get('anchor_violations') or []))}</b>; "
             f"dedup removals {len(g.get('duplicates_proposed') or [])}, reinstated {len(g.get('duplicates_rejected', []))}"
             + (f"; audit-only proposals on the trained store: {len((g.get('dedup_audit_only') or {}).get('would_remove') or [])} "
                f"remove, {len((g.get('dedup_audit_only') or {}).get('would_flag') or [])} flag"
                if g.get('dedup_audit_only') else "")],
            [f"v-space SFT (GeodeEmbedLoop, {esc(el.get('mode', 'rotation'))})", f"{el['iterations']} iterations, converged={el['converged']}; held-out noisy-alias "
             f"binding {pct(el['before'])} → <b>{pct(el['after'])}</b> (n={el['n_probes']})"
             + (" · A/B: " + ", ".join(f"{m} {pct(v['after'])}" for m, v in el['ab']['modes'].items())
                + f" → kept <b>{esc(el['ab']['chosen'])}</b> ({esc(el['ab']['rule'])})" if el.get('ab') else "")],
            ["u-space SFT (contradiction_sft, full)", f"held-out phrasing routing {pct(uc['before'])} → <b>{pct(uc['after'])}</b> "
             f"over {len(uc['groups'])} relation groups"],
            ["production store (store_from_triples, cap loss)", f"{bd['store']['n_entities']} entities, {bd['store']['n_triples']} triples, "
             f"{bd['store']['n_enm']} ENM entries, {bd['store']['epochs']} epochs, geometry {esc(bd['store']['geometry'])}"],
            ["calibration (GMSJudge.calibrate)", esc(bd["judge_thresholds"])],
            ["build time", esc(bd["timings_s"])]]) + "</div>")
    else:
        parts.append(f"<p>Built-in store: {st['entities']} entities, {st['triples']} triples, {st['relations']} relations, "
                     f"{st['enm_entries']} exact numeric entries. Self-correction over {st.get('candidates', '–')} candidates: "
                     f"{esc(st.get('corrections'))}.</p>")
    parts.append(f"<p>Compiler critics: injected OCR misread caught: <span class='{'ok' if st['injected_misread_caught'] else 'bad'}'>"
                 f"{'yes' if st['injected_misread_caught'] else 'no'}</span>.</p>")
    parts.append("<div class='card'>" + table(["primitive", "value"],
                 [[f"<code>{esc(k)}</code>", esc(round(v, 4) if isinstance(v, float) else v)] for k, v in ex.items()]) + "</div>")
    tb = st.get("tau_bootstrap") or {}
    if tb.get("relations"):
        f2 = lambda v: "–" if not v else f"{v[0]:.3f}–{v[1]:.3f}"
        parts.append(f"<details><summary>Cut stability: bootstrap intervals per relation (B={tb['B']})</summary>"
                     "<div class='card'>" + table(["relation", "deployed cut", "facts / counterfactuals", "refit cut 95%",
                                                   "true-accept 95%", "false-accept 95%"],
            [[f"<code>{esc(r)}</code>", f"{v['tau_deployed']:.3f}", f"{v['n_pos']} / {v['n_neg']}", f2(v["refit_tau_ci95"]),
              f2(v["tar_ci95"]), f2(v["far_ci95"])] for r, v in tb["relations"].items()], {1}) +
            f"<p class='muted'>{esc(tb.get('note', ''))}</p></div></details>")
    op = st.get("oracle_property") or []
    if op:
        parts.append("<h3>An oracle, not an index: true facts inside the calibrated band, cross-tail substitutions outside</h3>"
                     "<div class='card'>" + table(["fact", "true", "score", "substituted", "score", "cut", "ok"],
            [[esc(f"{o['head']} {o['relation']}"), esc(o["true"]), f"{o['score_true']:.3f}", esc(o["fabricated"]),
              f"{o['score_fabricated']:.3f}", f"{o['tau']:.3f}",
              f"<span class='{'ok' if o['calibrated_ok'] else 'bad'}'>{'yes' if o['calibrated_ok'] else 'no'}</span>"] for o in op],
            {2, 4, 5}) + "</div>")
    kb = R["kb"]
    parts.append(f"<p>Policy-question bases: {len(kb['items'])} mined; graph baseline accuracy on its own questions "
                 f"<b>{pct(kb['precondition']['graph_baseline_accuracy'])}</b>; store self-tests "
                 f"<span class='{'ok' if kb['self_tests']['passed'] else 'bad'}'>"
                 f"{'passed' if kb['self_tests']['passed'] else 'failed'}</span>.</p>")
    rec = kb.get("recovery", {})
    if rec:
        tot = {}
        for c, v in rec["stats"].items():
            for k2, n in v.items():
                tot[k2] = tot.get(k2, 0) + n
        fg = rec["forged"]
        parts.append(f"<p>Two-way admission ({esc(rec.get('gate'))}): {esc(tot)}. A forged question whose answer recomputes "
                     f"correctly but declares another option's fee as its fact triple: symbolic check "
                     f"{'admits' if fg['corrupted_triple']['symbolic'] else 'rejects'}, two-way check "
                     f"<b>{esc(fg['corrupted_triple']['two_way'])}</b>. multi_hop ground truth "
                     f"<code>{esc(rec.get('multi_hop_ground_truth'))}</code>.</p>")
        sub = rec.get("knowlytix_substrate")
        if sub:
            parts.append("<details><summary>knowlytix production generators on the store (substrate self-test)</summary>"
                         "<div class='card'>" + table(["generator", "admitted", "recovery split", "graph answers own questions", "cut read"],
                [[esc(k), v.get("admitted", "–"), esc(v.get("recovery_stats") or v.get("skipped") or v.get("error")),
                  pct(v.get("graph_answers_own_questions")) if v.get("graph_answers_own_questions") is not None else "–",
                  esc(v.get("calibrated_cut") or "")] for k, v in sub.items()]) + "</div></details>")
    hp = kb.get("hallucination_probes")
    if hp and hp.get("n"):
        parts.append(f"<h3>Hallucination probes</h3><p>{hp['n']} probes whose truth is absence ({esc(hp['by_kind'])}): "
                     "near-miss option names a caller could use and attributes a sparsely described entity does not have. "
                     "The correct bot abstains; a quoted figure is graded <i>fabricated</i> (severity critical).</p>"
                     "<div class='card'>" + table(["probe", "kind", "meaning card", "admission", "GMS binding of the fake name"],
            [[esc(x["qid"]), esc(x["kind"]), f"<code>{esc(x['frame'])}</code>", esc(x["admission"]),
              esc({k: v for k, v in x["diagnostic"].items() if k in ("gms_binds_to", "confidence", "binds_to_real_option",
                                                                       "facts_about_subject")})]
             for x in hp["items"]]) + "</div>")
    parts.append("<details><summary>Mined policy questions</summary><div class='card'>" + table(
        ["qid", "category", "frame", "answer", "primitive", "recovery"],
        [[esc(i["qid"]), esc(i["category"]), f"<code>{esc(i['frame'])}</code>", esc(i["answer"]), esc(i["primitive"]),
          esc(i.get("recovery", i["admission"]))] for i in kb["items"]]) + "</div></details>")

    # 2 bases
    parts.append("<h2>2. Base cases and coverage</h2>")
    parts.append(f"<p>The covering array exercises every feasible pair of content levels "
                 f"(<b>{cov['pairs_covered']}/{cov['pairs_required']}</b>; {cov['pairs_infeasible']} pairs are "
                 f"infeasible by construction) with {cov['n_covering']} journeys; closure added {cov['n_closure']} "
                 f"to reach every rule, edge, trigger and boundary.</p>")
    parts.append("<div class='card'>" + table(["goal", "covered", "missing"], [
        ["policy rules", len(cov["rules_covered"]), esc(", ".join(cov["rules_missing"]) or "none")],
        ["workflow edges", len(cov["edges_covered"]), esc(", ".join(cov["edges_missing"]) or "none")],
        ["escalation triggers", len(cov["triggers_covered"]), esc(", ".join(cov["triggers_missing"]) or "none")],
        ["boundaries", esc(", ".join(cov["boundaries_covered"])), ""]]) + "</div>")
    parts.append("<details><summary>All base cases</summary><div class='card'>" + table(
        ["qid", "category", "content (relevant dimensions)", "expected outcome", "turns"],
        [[esc(b["qid"]), esc(b["category"]), f"<span class='mono'>{esc(b['relevant'])}</span>",
          f"<span class='mono'>{esc(b['expected'])}</span>", b["turns"]] for b in R["bases"]]) + "</div></details>")

    # 3 design
    parts.append("<h2>3. Enrichment design</h2>")
    parts.append(f"<p>{len(des['factors'])} of the catalog's 42 factors selected for this bot: {esc(', '.join(des['factors']))}. "
                 f"{des['n_runs']} runs, base as a balanced blocking factor, rows dealt along a farthest-point order "
                 f"(mean within-base minimum distance {des['per_base_dispersion']['dispersed']:.3f} vs "
                 f"{des['per_base_dispersion']['shuffle']:.3f} with a random shuffle).</p>")
    q = des["quality"]
    parts.append("<div class='card'>" + table(["design", "log φp (lower = better fill)", "min distance",
                                               "pairwise level coverage", "worst marginal χ² p"],
        [[k, f"{v['log_phi_p']:.3f}", f"{v['min_distance']:.3f}", pct(v["pairwise_level_coverage"]),
          f"{v['worst_marginal_chi2_p']:.3f}"] for k, v in q.items()], {1, 2, 3, 4}) + "</div>")
    if des.get("power"):
        parts.append("<p class='muted'>Power to detect one level raising failure odds 3× (base failure 15%, "
                     "Bonferroni over factors): " + ", ".join(f"n={p['n_runs']}: {pct(p['power'])}" for p in des["power"]) + "</p>")
    if des.get("level_provenance"):
        parts.append(f"<details><summary>Level provenance: every application level traced to the official catalog "
                     f"({esc(des.get('official_catalogs'))})</summary><div class='card'>" + table(
            ["factor", "application level → catalog level", "catalog levels not realised"],
            [[esc(r["factor"]), esc(r["levels"]), esc({**r["unrealised"], **{k: f"moved to base: {v}" for k, v in r["moved_to_base"].items()}} or "")]
             for r in des["level_provenance"]]) + "</div></details>")
    parts.append(f"<details><summary>The other {len(des['dropped'])} catalog factors: excluded or moved to the base, with reasons</summary><div class='card'>" +
                 table(["factor", "reason"], [[esc(f), esc(r)] for f, r in des["dropped"]]) + "</div></details>")

    # 4 invariance
    parts.append("<h2>4. Proof that enriched cases equal their base cases</h2>")
    parts.append(f"<p>Every customer turn on each scenario's ground-truth path was realised under the scenario's "
                 f"factor levels and read back into its ontological frame. "
                 f"L1 ground truth identical: <b>{s['L1_gt_identical']}/{s['scenarios']}</b>; "
                 f"L2 frames equivalent: <b>{s['L2_frames_equivalent']}/{s['scenarios']}</b> "
                 f"({s['turns_checked']} turns); L3 oracle re-derived the same answer from the enriched transcript: "
                 f"<b>{s['L3_oracle_rederived']}/{s['scenarios']}</b>; ontological signatures equal: "
                 f"<b>{s['signatures_equal']}/{s['scenarios']}</b>. Maximum ontological distance among accepted "
                 f"scenarios: <b>{s['max_frame_distance_accepted']:.2f}</b>. Mean surface cosine to the base wording "
                 f"{s['surface_cosine_mean']:.2f} (min {s['surface_cosine_min']:.2f}): the wording moved, the meaning did not.</p>")
    sm = R.get("state_mapping", {})
    if sm.get("policy_graph"):
        parts.append("<p>Each turn is read in the dialog state it answers (state-aware ontology mapping: a bare yes/no "
                     "is read the way the state reads it; a card must be admissible in its state).</p><div class='card'>"
                     + table(["state", "accepts", "turns", "same meaning", "admissible"],
                             [[esc(k), esc(", ".join(sm["accepted_acts"].get(k, []))), v["turns"],
                               pct(v["equivalent"] / max(1, v["turns"])), pct(v["admissible"] / max(1, v["turns"]))]
                              for k, v in sm["states"].items()], {2, 3, 4}) + "</div>")
    parts.append("<h3>Negative controls: meaning-changing edits must be rejected</h3><div class='card'>" + table(
        ["edit", "n", "rejected", "mean ontological distance", "changes the expected bot behaviour"],
        [[esc(k), v["n"], f"<span class='{'ok' if v['detection_rate'] == 1 else 'warn'}'>{pct(v['detection_rate'])}</span>",
          "–" if v.get("mean_ontological_distance") is None else f"{v['mean_ontological_distance']:.2f}",
          pct(v.get("behaviour_changed_rate"))] for k, v in nc.items() if not k.startswith("_")], {1, 3}) + "</div>")
    by = s["by_level"]
    parts.append("<details><summary>Acceptance and surface distance by factor level</summary><div class='card'>" + table(
        ["factor=level", "scenarios", "accepted", "mean surface cosine"],
        [[esc(k), v["scenarios"], pct(v["acceptance"]), f"{v['mean_surface_cosine']:.2f}"] for k, v in by.items()],
        {1, 2, 3}) + "</div></details>")
    for e in inv["examples"][:4]:
        from .enrich.realizers import BASE_LEVELS
        changed = ", ".join(f"{k}={v}" for k, v in e["levels"].items() if BASE_LEVELS.get(k) != v) or "all base levels"
        parts.append(f"<details><summary>Example {esc(e['sid'])}: {esc(changed)}</summary><div class='card'>")
        for t in e["turns"]:
            parts.append(f"<div class='turn u'><div class='mono'>{esc(t['frame'])}</div>"
                         f"<div>base: {esc(t['base_text'])}</div><div>enriched: {esc(t['enriched_text'])}</div>"
                         f"<div class='muted'>read back: {esc(t['read_back'])} · distance {t['frame_distance']:.2f} · "
                         f"surface cos {t['surface_cosine']:.2f}</div></div>")
        parts.append("</div></details>")

    parts.append(_llm_section(R))

    # 5 results
    parts.append("<h2>5. Results</h2>")
    b, e = res["base_level"], res["enriched"]
    el = res.get("enriched_llm")
    keys = [k for k in e if k not in ("n",)]
    head = ["metric", "base cases", "enriched (template arm)"] + (["enriched (LLM arm)"] if el else [])
    parts.append("<div class='card'>" + table(head,
                 [[esc(k), pct(b.get(k)), pct(e.get(k))] + ([pct(el.get(k))] if el else []) for k in keys],
                 {1, 2, 3}) + f"<p class='muted'>Runs: base {b['n']}, template {e['n']}" +
                 (f", LLM {el['n']}" if el else "") + f". Budgets per bot turn: {esc(res.get('budgets'))}.</p></div>")
    parts.append(_arms_section(R))
    parts.append(_release_section(R))
    att = dg["outcome_attribution"]
    parts.append(f"<h3>Which presentation factors drive failure (logistic deviance, BH-corrected)</h3>"
                 f"<p class='muted'>{dg['failures']} failures in {dg['n']} runs"
                 f"{'' if dg['logit_reliable'] else ' — too few for stable logistic fits; read the empirical rates'}.</p>")
    parts.append("<div class='card'>" + hbar([(t["factor"] + (" *" if t["significant_adj"] else ""), t["pseudo_r2"],
                                              f"G²={t['deviance']:.2f}, p_adj={t['p_value_adj']:.4f}") for t in att],
                                             fmt=lambda v: f"{v:.3f}") +
                 "<p class='muted'>McFadden pseudo-R²; * = significant after Benjamini–Hochberg at α=0.05.</p>" +
                 table(["factor", "G²", "df", "p", "p (BH)", "pseudo-R²", "significant"],
                       [[esc(t["factor"]), f"{t['deviance']:.2f}", t["df"], pfmt(t["p_value"]), pfmt(t["p_value_adj"]),
                         f"{t['pseudo_r2']:.3f}", "<span class='bad'>yes</span>" if t["significant_adj"] else "no"]
                        for t in att], {1, 2, 3, 4, 5}) + "</div>")
    wl = dg["worst_levels"]
    parts.append("<h3>Most damaging conditions</h3><div class='card'>" +
                 hbar([(f"{w['factor']}={w['level']}", w["failure_rate"], f"n={w['n']}, Wilson {w['wilson'][0]:.2f}–{w['wilson'][1]:.2f}")
                       for w in wl], ci=[w["wilson"] for w in wl], fmt=lambda v: f"{100 * v:.0f}%") +
                 "<p class='muted'>Failure rate per level with its 95% Wilson interval.</p></div>")
    if dg.get("joint_model"):
        jm = dg["joint_model"]
        parts.append(f"<p class='muted'>Joint model: pseudo-R² {jm['pseudo_r2']:.3f}, AIC {jm['aic']:.1f}, BIC {jm['bic']:.1f}, "
                     f"AUC {jm['roc_auc']:.3f}, Hosmer–Lemeshow p {jm['hl_p_value']:.3f}.</p>")
    if dg["interactions"]:
        parts.append("<div class='card'>" + table(["interaction", "G²", "df", "p"],
                     [[esc(" × ".join(i["pair"])), f"{i['deviance']:.2f}", i["df"], pfmt(i["p_value"])] for i in dg["interactions"]],
                     {1, 2, 3}) + "</div>")
    kxa = dg.get("knowlytix")
    if kxa and kxa.get("logistic"):
        parts.append("<h3>knowlytix DOEAnalyzer (GraphDOE): logistic analysis of deviance, BH-corrected</h3><div class='card'>" + table(
            ["factor", "deviance", "p", "p (BH)", "pseudo-R²", "significant"],
            [[esc(t.get("factor")), f"{t.get('deviance', float('nan')):.2f}", pfmt(t.get("p_value", 1.0)),
              pfmt(t.get("p_value_adj", t.get("p_adjusted", t.get("p_value", 1.0)))), f"{t.get('pseudo_r2', float('nan')):.3f}",
              "<span class='bad'>yes</span>" if t.get("significant_adj", t.get("significant")) else "no"]
             for t in sorted(kxa["logistic"], key=lambda t: t.get("p_value", 1.0))], {1, 2, 3, 4}) +
            (f"<p class='muted'>Joint model: {esc(kxa.get('joint_model'))}</p>" if kxa.get("joint_model") else "") + "</div>")
    elif kxa and kxa.get("note"):
        parts.append(f"<p class='muted'>knowlytix DOEAnalyzer: {esc(kxa['note'])}</p>")
    pc = dg["per_component"]
    parts.append("<h3>Per component</h3><div class='card'>" + table(
        ["component", "graded", "failures", "accuracy", "factors significant for this component"],
        [[esc(c), v["graded"], v["failures"], pct(v["accuracy"]),
          esc(", ".join(f"{f} (p={'<0.001' if pv < 0.001 else f'{pv:.3f}'})" for f, pv in v.get("significant") or []) or "none")]
         for c, v in pc.items()], {1, 2, 3}) +
        f"<p class='muted'>Weak link (first erring stage per failed run): {esc(dg['weak_link'])}</p></div>")
    parts.append("<h3>Per base category</h3><div class='card'>" + table(
        ["category", "runs", "accuracy", "failure-rate 95% CI"],
        [[esc(c), v["n"], pct(v["accuracy"]), f"{pct(v['wilson_failure'][0])}–{pct(v['wilson_failure'][1])}"]
         for c, v in dg["per_category"].items()], {1, 2}) + "</div>")

    # 6 resilience
    if R.get("resilience"):
        parts.append("<h2>6. Resilience under tool faults</h2><div class='card'>" + table(
            ["tool", "fault", "reached", "oracle escalates", "bot failed loud", "silent", "undetectable", "detection", "matches oracle"],
            [[esc(r["tool"]), esc(r["kind"]), r["reached"], r["expected_loud"], r["detected"],
              f"<span class='{'bad' if r['silent'] else 'ok'}'>{r['silent']}</span>", r["undetectable"],
              pct(r["detection_rate"]), pct(r["oracle_agreement"])] for r in R["resilience"]], {2, 3, 4, 5, 6, 7, 8}) +
            "<p class='muted'>'Undetectable' faults are well formed and carry nothing the bot can verify (a forced "
            "AUTHENTICATED, an inflated options list); the oracle proceeds too. They need verification on the backend side.</p></div>")

    # 7 mutation
    if mut:
        names = list(mut["mutation_score"].keys())
        parts.append("<h2>7. Mutation analysis: does the suite catch real bugs?</h2>")
        parts.append("<div class='card'>" + table(["suite", "defects caught"],
                     [[esc(k), pct(v)] for k, v in mut["mutation_score"].items()], {1}) + "</div>")
        cols = [n for n in names if n != "all_suites_combined"]
        rows = []
        for m, v in mut["mutants"].items():
            cells = []
            for n in cols:
                k = v["killed_by"].get(n, {})
                cells.append("<span class='ok'>✓ caught</span>" if k.get("killed") else "<span class='muted'>– missed</span>")
            rows.append([f"<b>{esc(m)}</b><div class='muted'>{esc(v['description'])}</div>"] + cells)
        if "runtime_monitor_catch_rate" in mut:
            cols_rt = True
            rows = [r + [("<span class='ok'>✓ flagged</span>" if mut["mutants"][m].get("flagged_at_runtime") else "<span class='muted'>–</span>")]
                    for r, m in zip(rows, mut["mutants"])]
            parts.append(f"<p>Runtime monitor (no oracle, gates only) flags <b>{pct(mut['runtime_monitor_catch_rate'])}</b> of the "
                         "seeded defects as they happen.</p>")
            cols = cols + ["runtime monitor"]
        parts.append("<div class='card'>" + table(["seeded defect"] + cols, rows) + "</div>")

    gv = R.get("governance")
    if gv:
        pcal, plan, sw = gv["plausibility_calibration"], gv["plan_calibration"], gv["probe_sweep"]
        parts.append("<h2>8. Runtime governance: calibrated gates, live monitoring, audit</h2>")
        parts.append(f"<p>Engine: {esc(gv['engine'])}. Plausibility gate on <code>enables</code> calibrated on a labelled "
                     f"cohort from the policy graph ({pcal['n_admit']} admit, {pcal['n_block']} block; AUC {pcal['auc']:.3f}). "
                     f"Regulated operating point (zero false admits, fewest false rejects): θ = <b>{pcal['theta']:.3f}</b>, "
                     f"false rejects {pcal['regulated']['false_rejects']} "
                     f"({esc(', '.join(x['edge'] for x in pcal['false_rejects_at_theta']) or 'none')}); escalation actions are "
                     f"safe-harboured (escalated, never denied).</p>")
        parts.append("<div class='card'>" + table(["θ (book grid)", "false admits", "false rejects", "accuracy"],
                     [[r["theta"], r["false_admits"], r["false_rejects"], pct(r["accuracy"])] for r in pcal["book_grid"]], {0, 1, 2, 3}) + "</div>")
        parts.append(_operating_points_section("Plausibility gate", pcal))
        parts.append(_operating_points_section("Plan gate", plan))
        parts.append(f"<p>Plan gate: {plan['n_admissible_plans']} admissible oracle plans vs {plan['n_inadmissible_plans']} skipped/"
                     f"swapped/premature ones. Book criterion (mean drift) at zero false admits rejects "
                     f"{plan['mean_drift']['regulated']['false_rejects']}; per-step max rejects {plan['max_step']['regulated']['false_rejects']} "
                     f"→ deployed criterion <b>{esc(plan['criterion'])}</b>, budget {plan['drift_budget']:.3f}. Gate probe sweep "
                     f"(every previous step × proposed tool): {esc(sw['confusion'])}.</p>")
        mt = gv["multi_threshold_design"]
        parts.append(f"<p class='muted'>Multi-threshold balanced design (proofloop balanced_design, balanced={mt['balanced']}): "
                     f"best row {esc({k: v for k, v in mt['best'].items() if not k.endswith('errors') or k == 'total_errors'})}.</p>")
        rt = gv.get("runtime")
        if rt:
            parts.append("<div class='tiles'>" + tile(pct(rt["false_alarm_rate"]), "false alarms on correct runs") +
                         tile(f"{rt['failed_runs_flagged']}/{rt['failed_runs']}", "failed runs flagged live") +
                         tile("yes" if rt["audit_chains_verified"] else "no", "audit hash chains verified") +
                         tile("yes" if rt["recalibration"]["recalibrate"] else "no", "recalibration triggered") + "</div>")
            parts.append("<div class='card'>" + table(["gate:rule", "alarms"], [[esc(k), v] for k, v in sorted(
                rt["alarms_by_gate_rule"].items(), key=lambda kv: -kv[1])], {1}) +
                f"<p class='muted'>Prompt-injection regex baseline (forgeloop policies) on the instruction_conflict levels: "
                f"{esc(rt['injection_guard_by_level'])} — the paraphrased overrides pass the regex; the bot's behaviour under "
                f"them is what the invariance proof and outcome scoring test. Recalibration triggers: "
                f"{esc(rt['recalibration']['triggers'])} (PSI {rt['recalibration']['psi']:.3f}).</p></div>")
            qa = rt.get("quorum_and_attestor")
            if qa:
                parts.append(f"<p>Quorum rule (gmsh check_with_quorum, shadow-computed for every call): "
                             f"{qa['calls_routed_to_review']} calls would be routed to review instead of denied; strict and "
                             f"quorum disagree on {qa['calls_where_strict_and_quorum_disagree']} calls. Numeric range attestor: "
                             f"{qa['runs_with_numeric_review']} runs with a REVIEW figure, {qa['runs_with_numeric_fail']} with an "
                             f"invented figure; correct runs with a numeric alarm: {qa['correct_runs_with_numeric_alarm']}.</p>")
    sft = R.get("sft")
    if sft and "classifier" in sft:
        c = sft["classifier"]
        parts.append("<h2>9. Training corpora from the design (proofloop emitters)</h2>")
        parts.append(f"<p>Classifier corpus (<code>sft_classifier.jsonl</code>, schema {esc(c['schema'])}): {c['n']} records from "
                     f"{c['seeds']} seed turns, {c['distinct_messages']} distinct messages; one label per seed across all its phrasings: "
                     f"<b>{'yes' if c['one_label_per_seed'] else 'no'}</b>; dropped by the grounding-preserving filter: {c['dropped_by_grounding_filter']}.</p>")
        parts.append("<div class='card'>" + table(["draft responder", "kept", "dropped by golden contract", "example dropped"],
            [[esc(k.replace("draft_", "")), v["kept"], v["dropped"], esc(v["dropped_examples"][:1])]
             for k, v in sft.items() if k.startswith("draft_")], {1, 2}) + "</div>")
    fm = R.get("failure_modes")
    if fm:
        parts.append("<h2>10. Failure-mode coverage (proofloop taxonomy)</h2><div class='card'>" + table(
            ["failure mode", "probed by", "defects caught by the suite", "flagged at runtime"],
            [[esc(r["failure_mode"]), esc(", ".join(r.get("mutants", []) + r.get("faults", []) + r.get("factors", []) + r.get("gates", []))),
              esc(", ".join(r["mutants_killed"]) or "–"), esc(", ".join(r["mutants_flagged_at_runtime"]) or "–")] for r in fm]) + "</div>")

    # 8 failures
    ds, cn = R.get("dataset"), R.get("canonicals")
    if ds or cn:
        parts.append("<h2>11. Exports</h2><div class='card'>" + table(["file", "content"], [
            ["<code>policy_kg.json</code>", "the policy graph this run was built from"],
            ["<code>gms_canonicals.json</code>", esc(f"{cn['entities']} entities, {cn['value_sets']} value sets, "
                                                    f"{cn['numbers']} exact numbers, {cn['states']} states, {cn['facts']} "
                                                    f"trained facts ({pct(cn['facts_inside_cut'])} inside their cut); "
                                                    f"alias bindings agreeing with the graph {pct(cn['alias_agreement'])}")
             if cn else "–"],
            ["<code>dataset/test_cases.csv</code>", esc(f"{ds['test_cases']} cases ({ds['base_cases']} base, "
                                                       f"{ds['enriched_cases']} enriched)") if ds else "–"],
            ["<code>dataset/test_turns.csv</code>", esc(f"{ds['turn_rows']} customer turns") if ds else "–"],
            ["<code>dataset/policy_questions.csv</code>", esc(f"{ds['policy_questions']} policy questions") if ds else "–"],
        ] + ([["<code>dataset/llm_enrichment_turns.csv</code>", esc(f"{R['llm_exports'].get('llm_turns', 0)} LLM-written "
                                                                    "turns with every gate, both readers and the factor check")],
              ["<code>dataset/llm_negative_controls_audit.csv</code>", esc(f"{R['llm_exports'].get('llm_controls_audit', 0)} "
                                                                           "controls the LLM gate accepted (for human audit)")],
              ["<code>dataset/arm_pairs.csv</code>", esc(f"{R['llm_exports'].get('arm_pairs', 0)} template/LLM pairs")]]
             if R.get("llm_exports") else [])) + "</div>")
    parts.append("<h2>12. Reading the failures</h2><p>Every failed run is written to <code>failures.jsonl</code> "
                 "with its transcript; <code>results_enriched.csv</code> holds one row per scenario.</p>")
    parts.append("</main></body></html>")
    return "".join(parts)


# ----------------------------------------------------------------------------- gmsh-parity sections
def _operating_points_section(name: str, cal: dict[str, Any]) -> str:
    p50, ho, bs = cal.get("p50"), cal.get("holdout"), cal.get("bootstrap")
    if not p50:
        return ""
    reg = cal.get("regulated") or (cal.get("mean_drift") or {}).get("regulated") or {}
    theta = cal.get("theta", cal.get("drift_budget"))
    rows = []
    for rule, th, fa, fr in (("regulated (zero false admits)", theta, reg.get("false_admits"), reg.get("false_rejects")),
                             ("gmsh P=0.5 (logistic)", p50.get("theta"), p50.get("false_admits"), p50.get("false_rejects"))):
        key = "regulated" if rule.startswith("regulated") else "p50"
        h = (ho or {}).get(key, {})
        b = (bs or {}).get(key, {})
        rows.append([esc(rule), "–" if th is None else f"{th:.3f}", esc(fa), esc(fr),
                     pct(h.get("heldout_accuracy")) if h else "–",
                     f"{b['ci95'][0]:.3f}–{b['ci95'][1]:.3f}" if b.get("ci95") else "–"])
    return (f"<h3>{esc(name)}: operating points side by side</h3><div class='card'>" +
            table(["rule", "threshold", "false admits", "false rejects", f"held-out accuracy ({(ho or {}).get('k', '–')}-fold)",
                   f"bootstrap 95% interval (B={(bs or {}).get('B', '–')})"], rows, {1, 2, 3, 4}) + "</div>")


def _llm_section(R: dict[str, Any]) -> str:
    L = R.get("invariance_llm")
    if not L:
        return ""
    s, nc = L["summary"], L["negative_controls"]
    adv = nc.get("_adversarial", {})
    out = ["<h2>4b. LLM enrichment arm: separate invariance validation</h2>",
           f"<p>Customer turns written by <b>{esc(L['settings']['model'])}</b> through LangChain "
           f"(mode {esc(L['settings']['mode'])}, temperature {L['settings']['temperature']}), blind to the expected outcome. "
           "Every turn passed G0 form, G1 no new facts, G3 one clear decision and G2 the rule reader's meaning card in its "
           f"dialog state{' and the independent LLM reader' if s['dual_read']['mode'] == 'gate' else ''}; then the same "
           "scenario-level L1/L3 test as the template arm.</p>"]
    if L.get("unavailable_reason"):
        out.append(f"<p class='warn'>LLM not callable during this run: {esc(L['unavailable_reason'])}. Only cached "
                   "generations were usable; scenarios without one are reported as not materialised.</p>")
    out.append("<div class='tiles'>" + tile(f"{s['accepted']}/{s['scenarios']}", "LLM scenarios proved invariant") +
               tile(pct(nc["_overall"]["detection_rate"]), "meaning-changing edits of LLM text rejected") +
               tile(pct(adv.get("rejection_rate")) if adv.get("n") else "–", "adversarial LLM rewrites rejected") +
               tile(pct(s["dual_read"]["agreement"]) if s["dual_read"]["agreement"] is not None else "–",
                    "independent reader agrees") + "</div>")
    out.append("<div class='card'>" + table(["check", "value"], [
        ["L1 ground truth identical", f"{s['L1_gt_identical']}/{s['scenarios']}"],
        ["L2 every turn passed every gate", f"{s['L2_frames_equivalent']}/{s['scenarios']} ({s['turns_checked']} turns)"],
        ["L3 oracle re-derived the answer", f"{s['L3_oracle_rederived']}/{s['scenarios']}"],
        ["signatures equal", f"{s['signatures_equal']}/{s['scenarios']}"],
        ["mean attempts per turn", "–" if s["mean_attempts_per_turn"] is None else f"{s['mean_attempts_per_turn']:.2f}"],
        ["gate failures over all attempts", esc(s["gate_failures_by_attempt"])],
        ["not materialised (excluded, reported)", esc(s["not_materialised"])],
        ["surface cosine to base / to the template text", f"{s['surface_cosine_mean']:.2f} / "
         + ("–" if s["cosine_to_template_text_mean"] is None else f"{s['cosine_to_template_text_mean']:.2f}")],
        ["scenarios with an unrealised factor (attributed at the realised level)", esc(s["scenarios_with_unrealised_factors"])],
    ]) + "</div>")
    if s.get("top_rejection_reasons"):
        out.append("<details><summary>Why LLM turns were rejected</summary><div class='card'>" +
                   table(["reason", "turns"], [[esc(r), n] for r, n in s["top_rejection_reasons"]], {1}) + "</div></details>")
    out.append("<h3>Negative controls on LLM text</h3><div class='card'>" + table(["edit", "n", "rejected"],
               [[esc(k), v["n"], f"<span class='{'ok' if v['detection_rate'] == 1 else 'warn'}'>{pct(v['detection_rate'])}</span>"]
                for k, v in nc.items() if not k.startswith("_")], {1, 2}) +
               (f"<p class='muted'>Adversarial rewrites: {adv['n']} generated, {adv['rejected']} rejected; of the "
                f"{adv['effective']} the rule reader reads as the target meaning, {pct(adv['rejection_rate_when_meaning_changed'])} "
                f"rejected. Accepted controls are listed in <code>dataset/llm_negative_controls_audit.csv</code> "
                f"({L.get('controls_audit_n', 0)}).</p>" if adv.get("n") else "") + "</div>")
    fr = s.get("factor_realisation") or {}
    if fr:
        out.append("<details><summary>Manipulation check: did each requested factor level reach the text?</summary>"
                   "<div class='card'>" + table(["factor=level", "turns", "realised"],
                   [[esc(k), v["turns"], pct(v["rate"])] for k, v in fr.items()], {1, 2}) + "</div></details>")
    for e in L.get("examples", [])[:3]:
        out.append(f"<details><summary>LLM example {esc(e['sid'])} (accepted: {e['accepted']})</summary><div class='card'>")
        for t in e["turns"]:
            out.append(f"<div class='turn u'><div class='mono'>{esc(t['frame'])} · state {esc(t['state'])}</div>"
                       f"<div>base: {esc(t.get('base_text'))}</div><div>template arm: {esc(t.get('template_text'))}</div>"
                       f"<div>LLM arm: {esc(t.get('enriched_text'))}</div>"
                       f"<div class='muted'>rule reader: {esc(t.get('read_back'))} · independent reader: {esc(t.get('dual_read'))}"
                       f" · gates {esc(t.get('gates'))}</div></div>")
        out.append("</div></details>")
    return "".join(out)


def _arms_section(R: dict[str, Any]) -> str:
    A = R.get("arms")
    if not A or not A.get("pairs_n"):
        return ""
    out = [f"<h3>Template arm vs LLM arm on the same design rows</h3><p>{A['pairs_n']} pairs (same base, same assigned "
           f"levels, both proved): template accuracy {pct(A['template_accuracy_on_pairs'])}, LLM accuracy "
           f"{pct(A['llm_accuracy_on_pairs'])}; discordant pairs: template-only correct {A['template_only_correct']}, "
           f"LLM-only correct {A['llm_only_correct']} (exact McNemar p = {pfmt(A['mcnemar_exact_p'])}).</p>"]
    dl = R.get("diagnosis_llm")
    if dl and dl.get("outcome_attribution"):
        out.append("<div class='card'>" + table(["factor (LLM arm, realised levels)", "G²", "p (BH)", "significant"],
                   [[esc(t["factor"]), f"{t['deviance']:.2f}", pfmt(t["p_value_adj"]),
                     "<span class='bad'>yes</span>" if t["significant_adj"] else "no"] for t in dl["outcome_attribution"][:8]],
                   {1, 2}) + "</div>")
    return "".join(out)


def _release_section(R: dict[str, Any]) -> str:
    G = R.get("release_gate")
    if not G:
        return ""
    rows = []
    for suite, g in G.items():
        for tier, t in (g.get("tiers") or {}).items():
            rows.append([esc(suite), esc(tier), t["n"], pct(t["accuracy"]), esc(t["severity"]), pct(t["within_budget"]),
                         f"<span class='{'ok' if t['passed'] else 'bad'}'>{'pass' if t['passed'] else 'fail'}</span>",
                         esc("; ".join(t["reasons"]) or "–")])
    return ("<h3>Severity and the tiered release gate (gmsh severity + risk tiers)</h3><div class='card'>" +
            table(["suite", "risk tier", "runs", "accuracy", "runs by worst severity", "within budget", "gate", "why"],
                  rows, {2, 3, 5}) +
            "<p class='muted'>Action-taking journeys (the correct path writes to the account) promote moderate and high "
            "failures by one level, as gmsh's tier escalation does; each tier has an accuracy floor and failure budgets "
            "(evaluation/severity.py).</p></div>")
