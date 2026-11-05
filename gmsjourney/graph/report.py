"""The HTML report of a campaign on a compact policy graph (one self-contained file)."""
from __future__ import annotations

from typing import Any

from ..report import CSS, CSS_FAIL, _failed_section, _release_section, esc, hbar, pct, table, tile


def render_graph_html(R: dict[str, Any]) -> str:
    st, cov, inv, des = R["store"], R["coverage"], R["invariance"], R["design"]
    res = R["results"]
    s = inv["summary"]
    nc = inv["negative_controls"]
    mut = R.get("mutation")
    title = esc(R["domain"]["name"].replace("_", " "))
    P = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Journey Test Report</title><style>{CSS}{CSS_FAIL}</style>"
         f"</head><body><main>",
         f"<h1>{title}: journey test campaign mined from the policy graph</h1>",
         f"<p class='lede'>System under test: <b>{esc(res['bot'])}</b>. Everything below - the store, the policy "
         f"questions, the journeys, the expected behaviour, the runtime checks and the seeded defects - is mined from "
         f"<code>{esc(R['domain']['graph'])}</code>; the pack's <code>domain.yaml</code> only adds customer wording.</p>"]
    P.append("<div class='tiles'>" +
             tile(str(cov["bases"]), "base cases mined") +
             tile(f"{cov['states']['covered']}/{cov['states']['total']}", "dialog states reached") +
             tile(f"{cov['workflow_edges']['covered']}/{cov['workflow_edges']['total']}", "workflow edges traversed") +
             tile(f"{cov['rules']['covered']}/{cov['rules']['total']}", "rules exercised") +
             tile(pct(s["acceptance_rate"]), "enriched cases proved equivalent") +
             tile(pct(nc["_overall"]["detection_rate"]), "meaning changes caught") +
             tile(pct(res["base_level"]["accuracy"]), "bot accuracy · base") +
             tile(pct(res["enriched"]["accuracy"]), "bot accuracy · enriched") +
             (tile(pct(mut["mutation_score"].get("non_equivalent", mut["mutation_score"]["all_suites_combined"])),
                   f"seeded defects caught ({len(mut.get('equivalent') or [])} equivalent excluded)") if mut else "") +
             "</div>")
    # 1 store
    pk = R.get("policy_kg") or {}
    cons = pk.get("consistency", {})
    bd = st.get("build") or {}
    P.append("<h2>1. Knowledge store (oracle)</h2>")
    P.append(f"<p>{pk.get('triples')} triples expanded from the compact graph ({pk.get('states')} dialog states, "
             f"{pk.get('prompt_states')} answered by the customer, {pk.get('transitions')} transitions, {pk.get('rules')} rules). "
             f"Trained store carries every fact of the graph: <span class='{'ok' if cons.get('all_agree') else 'bad'}'>"
             f"{'yes' if cons.get('all_agree') else 'no'}</span>. Compiler critics: planted misread caught: "
             f"<span class='{'ok' if st.get('injected_misread_caught') else 'bad'}'>{'yes' if st.get('injected_misread_caught') else 'no'}</span>.</p>")
    rows = [[esc(c["action"]), esc(" ".join(map(str, c["fact"]))), esc(c["reason"])] for c in st.get("compiler_corrections", [])]
    if rows:
        P.append("<details><summary>Knowledge-base prose reconciled against the exact register</summary><div class='card'>" +
                 table(["critic", "candidate fact", "why"], rows) + "</div></details>")
    if bd.get("geode"):
        gd, el = bd["geode"], bd.get("embed_loop", {})
        P.append("<div class='card'>" + table(["step (knowlytix)", "result"], [
            ["GEODE correction loop", f"converged={gd.get('converged')} iterations={gd.get('iterations')} kept "
                                      f"{gd.get('n_kept_fact_triples')} fact triples, {gd.get('n_anchor_violations')} anchor violations"],
            ["embed loop (v)", f"noisy-alias binding {el.get('before', 0):.2f} -> {el.get('after', 0):.2f}"],
            ["production store", f"{(bd.get('store') or {}).get('n_entities')} entities, {(bd.get('store') or {}).get('n_triples')} "
                                 f"triples, {(bd.get('store') or {}).get('n_enm')} exact registers"],
            ["exact register agreement", str((bd.get('enm_agreement') or {}).get('all_agree'))],
            ["build time", f"{bd.get('elapsed_s')} s"]]) + "</div>")
    P.append("<div class='card'>" + table(["primitive", "value"], [[esc(k), esc(v)] for k, v in st["primitive_examples"].items()]) + "</div>")
    kb = R["kb"]
    P.append(f"<p>Policy questions mined from the graph's figures: {len(kb['items'])}; admitted "
             f"{sum(i['admission'] != 'rejected' for i in kb['items'])} (two-way: symbolic recompute + knowlytix geometric "
             f"recovery). Graph baseline accuracy on its own questions: {pct(kb['precondition']['graph_baseline_accuracy'])}.</p>")
    P.append("<details><summary>Mined policy questions</summary><div class='card'>" + table(
        ["qid", "question", "answer", "admission"],
        [[esc(i["qid"]), f"<code>{esc(i['frame'])}</code>", esc(i["answer"]), esc(i["admission"])] for i in kb["items"]]) +
        "</div></details>")
    # 2 journeys
    P.append("<h2>2. Journeys mined from the state machine</h2>")
    P.append(f"<p>{cov['paths']} paths through the machine (at most two deviations from the default path, plus "
             f"{cov.get('closure_paths', 0)} deeper paths kept because they reach something new) and {cov['perturbations']} "
             f"failure perturbations laid over every prompt (a request for a person, an unclear answer recovered and "
             f"persisting, a policy question mid-journey, the turn limit at and over).</p>")
    P.append("<div class='card'>" + table(["coverage goal", "covered", "missing"], [
        ["dialog states", f"{cov['states']['covered']}/{cov['states']['total']}", esc(", ".join(cov["states"]["missing"]) or "–")],
        ["workflow edges", f"{cov['workflow_edges']['covered']}/{cov['workflow_edges']['total']}",
         esc(", ".join(cov["workflow_edges"]["missing"]) or "–")],
        ["rules", f"{cov['rules']['covered']}/{cov['rules']['total']}", esc(", ".join(cov["rules"]["missing"]) or "–")],
        ["transfer reasons", str(len(cov["transfer_reasons"])), esc(", ".join(cov["transfer_reasons"]))]]) + "</div>")
    fi = cov.get("factors")
    if fi:
        P.append(f"<h3>Base factors (base_factors.yaml): strength-{fi['strength']} covering array</h3>")
        P.append(f"<p>{len(fi['factors'])} factors, {sum(fi['factors'].values())} levels. Of every pair of levels, "
                 f"{fi['feasible']} can occur together in a conversation the policy allows (the oracle checked each, "
                 f"with the other levels that make both reachable); <b>{fi['covered']}</b> are covered by "
                 f"{fi['array_cases']} cases. {fi['infeasible']} pairs never occur together (one ends the call before "
                 f"the other is reached, or they script the same answer). The closure added "
                 f"{fi['closure']['added']} cases for graph items the array missed; graph items covered "
                 f"{fi['closure']['covered_items']}/{fi['closure']['reachable_items']}.</p>")
        P.append("<div class='card'>" + table(["factor", "levels"], [[esc(f), n] for f, n in fi["factors"].items()], {1}) + "</div>")
        if fi.get("uncovered"):
            P.append("<details open><summary>Feasible pairs not covered</summary><div class='card'>" +
                     table(["pair"], [[esc(" + ".join(u))] for u in fi["uncovered"]]) + "</div></details>")
        if fi.get("never_relevant_levels"):
            P.append("<p class='bad'>Levels no case can reach: " + esc(", ".join(fi["never_relevant_levels"])) + "</p>")
        P.append("<details><summary>Examples of pairs the policy never allows together</summary><div class='card'>" +
                 table(["pair"], [[esc(" + ".join(u))] for u in fi.get("infeasible_examples", [])]) + "</div></details>")
    cats = cov["categories"]
    P.append("<div class='card'>" + hbar([(k, v, str(v)) for k, v in sorted(cats.items(), key=lambda x: -x[1])]) + "</div>")
    P.append("<details><summary>All base cases</summary><div class='card'>" + table(
        ["qid", "category", "decisions", "expected outcome"],
        [[esc(b["qid"]), esc(b["category"]), esc("; ".join(str(v) for v in b["relevant"].values())),
          f"<code>{esc(b['expected'])}</code>"] for b in R["bases"]]) + "</div></details>")
    # 3 design + 4 proof
    P.append("<h2>3. Enrichment design and the invariance proof</h2>")
    P.append(f"<p>{len(des['factors'])} presentation factors: {esc(', '.join(des['factors']))}. {des['n_runs']} enriched "
             f"scenarios; {s['accepted']} of {s['scenarios']} proved equal to their base case "
             f"({s['turns_checked']} turns read back in their dialog state; oracle re-derived from the read-back cards).</p>")
    P.append("<div class='card'>" + table(["negative control", "n", "rejected", "behaviour changed"],
             [[esc(k), v["n"], pct(v["detection_rate"]), pct(v["behaviour_changed_rate"]) if v["behaviour_changed_rate"] is not None else "–"]
              for k, v in nc.items() if k != "_overall"], {1, 2, 3}) + "</div>")
    # 5 results
    P.append("<h2>4. Results</h2>")
    head = ["suite", "runs", "accuracy", "workflow adherence", "runtime invariance", "policy answers"]
    P.append("<div class='card'>" + table(head, [
        [k, r["n"], pct(r["accuracy"]), pct(r["workflow_adherence"]), pct(r["runtime_invariance"]), pct(r.get("kb_accuracy"))]
        for k, r in (("base level", res["base_level"]), ("enriched", res["enriched"]))], {1, 2, 3, 4, 5}) + "</div>")
    comps = sorted({k for r in (res["base_level"], res["enriched"]) for k in r if k.startswith("component_")})
    P.append("<h3>Per stage</h3><div class='card'>" + table(["stage", "base", "enriched"],
             [[esc(c[len("component_"):]), pct(res["base_level"].get(c)), pct(res["enriched"].get(c))] for c in comps], {1, 2}) + "</div>")
    P.append(_release_section(R))
    dg = R.get("diagnosis") or {}
    tests = dg.get("factor_tests") or []
    if tests:
        P.append("<h3>Which presentation factors drive failure (logistic deviance, BH-corrected)</h3><div class='card'>" +
                 hbar([(t["factor"] + (" *" if t.get("significant_adj") else ""), t["pseudo_r2"], f"{t['pseudo_r2']:.3f}")
                       for t in tests[:12]]) + "</div>")
    P.append(_failed_section(R))
    # resilience + mutation
    if R.get("resilience"):
        P.append("<h2>5. Resilience under tool faults</h2><div class='card'>" + table(
            ["tool", "fault", "reached", "expected loud", "detected", "silent", "agrees with oracle"],
            [[esc(r["tool"]), esc(r["kind"]), r["reached"], r["expected_loud"], r["detected"], r["silent"],
              pct(r["oracle_agreement"])] for r in R["resilience"]], {2, 3, 4, 5, 6}) + "</div>")
    if mut:
        P.append("<h2>6. Seeded defects mined from the graph</h2>")
        P.append("<div class='card'>" + table(["suite", "defects caught"],
                 [[esc(k), pct(v)] for k, v in mut["mutation_score"].items()], {1}) + "</div>")
        if mut.get("equivalent"):
            P.append(f"<p>{len(mut['equivalent'])} defects are <b>equivalent</b>: they behave exactly like the reference bot "
                     f"on every mined journey, so no conversation the policy allows can tell them apart "
                     f"({esc(', '.join(mut['equivalent']))}). Score without them: {pct(mut['mutation_score'].get('non_equivalent'))}.</p>")
        P.append("<div class='card'>" + table(["defect", "caught", "by", "runtime monitor"],
                 [[f"<code>{esc(m)}</code>", f"<span class='{'ok' if e['killed'] else ('muted' if e.get('equivalent') else 'bad')}'>"
                   f"{'yes' if e['killed'] else ('equivalent' if e.get('equivalent') else 'no')}</span>",
                   esc(", ".join(k for k, v in e["killed_by"].items() if v["killed"]) or "–"),
                   "yes" if e.get("flagged_at_runtime") else "–"] for m, e in mut["mutants"].items()]) + "</div>")
    if R.get("skipped"):
        P.append("<h2>Not run for this pack</h2><div class='card'>" + table(["step", "why"],
                 [[esc(k), esc(str(v)[:300])] for k, v in R["skipped"].items()]) + "</div>")
    P.append("</main></body></html>")
    return "".join(P)
