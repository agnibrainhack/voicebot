"""The one-page verdict of a finished run: is the bot ready, what are the risks, can anyone reproduce it, and, case
by case, what the policy graph expected next to what the bot under test did.

``build(run_dir)`` returns:

* ``gates``: the release gate on the original and on the reworded wording, with the reasons a gate failed;
* ``numbers``: cases tested, correct on original and reworded wording, reworded cases proved to mean the same,
  planted defects caught, runs whose words were all grounded;
* ``risks``: the failure codes ranked by severity and by how many runs they hit, each in plain words with an
  example case;
* ``fingerprint``: what the run was (journey, graph file and its hash, code commit, bot, model and prompt hash,
  seed, cases, device, store and package versions) so anyone can run it again and compare;
* ``cases``: every run of a case (original wording and each reworded version): the expected outcome (end, transfer
  reason, writes) beside the bot's, the failure codes, and per customer turn the graph's expected tools and acts
  beside the bot's tools, acts and words, whether they match, and what the speech check found.

Read from the run's own files (job.json, events.jsonl, out/report.json, out/dataset/*, out/failures.jsonl); nothing
is recomputed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# what each failure code means, in words a reader outside the team follows
PLAIN = {
    "CLAIM_WITHOUT_SUCCESS": "told the customer something was done when the system did not confirm it",
    "ACCOUNT_TOOL_BEFORE_AUTH": "opened account data before the caller was verified",
    "SUBMIT_WITHOUT_CONFIRMATION": "acted without the customer's confirmation",
    "FEE_NOT_DISCLOSED": "did not state a fee before the customer agreed to it",
    "TOOL_AFTER_TRANSFER": "kept acting after handing the call to a person",
    "WRONG_RECORD": "recorded something the customer did not say",
    "HALLUCINATION_DETECTED": "made up an answer the policy does not give",
    "OUTCOME_INCORRECT": "the call ended differently from what the policy requires",
    "ESCALATION_MISSED": "did not hand over to a person when the policy requires it",
    "AUTH_ATTEMPTS_EXCEEDED": "allowed more verification attempts than the policy permits",
    "TRANSACTION_LIMIT_EXCEEDED": "went past a limit the policy sets",
    "FACT_INCORRECT": "answered a policy question wrongly",
    "WRONG_CARD_ACTION": "took the wrong action, or none, on the customer's request",
    "PROCESS_DEVIATION": "took a step the policy's process does not allow",
    "TURN_LIMIT_EXCEEDED": "ran past the conversation's turn limit",
    "ESCALATION_INAPPROPRIATE": "handed over to a person when it did not need to",
    "UNCLEAN_TERMINATION": "did not end the call properly",
    "JOURNEY_STOPPED": "left the policy's journey (two wrong steps in a row, or ended the call early), so the test stopped",
    "STAGE_INCORRECT": "got a stage of the process wrong",
    "BUDGET_EXCEEDED": "too slow, too long-winded or too many tool calls",
    "TOOL_LOOP": "called the same tool over and over",
    "RUNTIME_INVARIANCE": "the customer's words were misread at run time",
    "SPEECH_CONTRADICTED": "said something false (a wrong figure, the opposite of what was recorded)",
    "SPEECH_UNSUPPORTED": "said something nothing backs (a promise or fact the system never gave)",
    "SPEECH_UNFAITHFUL": "its words say something other than what it did",
    "SPEECH_DISCLOSED": "told the customer an internal figure",
    "SPEECH_MISSING": "left out something the policy requires it to say",
    "SPEECH_FLOW": "said something the dialog does not allow at that point",
    "SPEECH_VOLUNTEERED": "volunteered a figure the customer did not ask for",
    "POLICY_RULE_BROKEN": "broke a rule of the policy",
    "LIMIT_EXCEEDED": "went past a limit the policy sets",
    "WRONG_STATEMENT": "made a wrong statement",
}


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None


def _writes(ws) -> list[str]:
    out = []
    for w in ws or []:
        if isinstance(w, (list, tuple)) and len(w) == 2:
            args = ", ".join(f"{k}={v}" for k, v in (w[1] or {}).items())
            out.append(f"{w[0]}({args})")
        else:
            out.append(str(w))
    return out


def _outcome(o: dict | None) -> dict:
    o = o or {}
    return {"end": o.get("end"), "transfer_reason": o.get("transfer_reason"), "writes": _writes(o.get("writes")),
            "told": o.get("told") or []}


def _plain_reason(r: str) -> str:
    """A release-gate reason (evaluation/severity.release_gate) in words: 'accuracy 0.000 < 0.95' -> 'correct 0% of
    runs, needs 95%'."""
    import re
    pc = lambda x: f"{100 * float(x):.0f}%"              # noqa: E731
    for pat, fmt in ((r"accuracy ([\d.]+) < ([\d.]+)", lambda m: f"correct {pc(m[1])} of runs, needs {pc(m[2])}"),
                     (r"(\d+) critical failures > (\d+)", lambda m: f"{m[1]} critical failures, {m[2]} allowed"),
                     (r"high-severity rate ([\d.]+) > ([\d.]+)", lambda m: f"{pc(m[1])} of runs with a high-severity failure, {pc(m[2])} allowed"),
                     (r"within-budget rate ([\d.]+) < ([\d.]+)", lambda m: f"{pc(m[1])} of runs within the time, length and tool-call budget, needs {pc(m[2])}"),
                     (r"(\w+): (\d+) > budget (\d+)", lambda m: f"{m[2]} runs with {m[1]}, {m[3]} allowed")):
        m = re.fullmatch(pat, r.strip())
        if m:
            return fmt(m)
    return r


def _speech(text: str | None) -> str:
    """The speech check's findings for a turn without the advisory notes (unread, volunteered): what failed."""
    parts = [p.strip() for chunk in (text or "").split(" | ") for p in chunk.split(" ; ")]
    keep = [p for p in parts if p and p != "ok" and not p.startswith(("unread", "volunteered"))]
    return " ; ".join(keep)


def provenance(pack: str) -> dict[str, Any]:
    """What a run can be reproduced from, taken when it starts: the code commit (and whether the working tree had
    uncommitted edits) and the hash of the journey's graph file."""
    import subprocess
    root = Path(__file__).resolve().parents[2]
    out: dict[str, Any] = {}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                               text=True, timeout=10)
        if commit.returncode == 0:
            out["code"] = {"commit": commit.stdout.strip(), "uncommitted_changes": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from ..s01_packs.loading import load
        out["graph_sha"] = hashlib.sha256(load(pack).graph_path.read_bytes()).hexdigest()[:12]
    except Exception:                              # noqa: BLE001 - provenance is best effort
        pass
    return out


def build(path: str | Path) -> dict[str, Any]:
    """The verdict of a finished run: ``path`` is a console run folder (with job.json and out/) or a campaign's output
    folder (out_dir, with report.json)."""
    p = Path(path)
    run, out = (p, p / "out") if (p / "out" / "report.json").exists() else (p.parent, p)
    if not (out / "report.json").exists():
        raise KeyError("this run has not finished: the verdict is built from its report")
    R = json.loads((out / "report.json").read_text(encoding="utf-8"))
    job = json.loads((run / "job.json").read_text(encoding="utf-8")) if (run / "job.json").exists() else {}
    job = {**(R.get("provenance") or {}), **job}
    events = _jsonl(run / "events.jsonl")
    rows = _jsonl(out / "dataset" / "case_results.jsonl")
    fails = {r["sid"]: r for r in _jsonl(out / "failures.jsonl")}
    tc = {r["case_id"]: r for r in _jsonl(out / "dataset" / "test_cases.jsonl")}
    tbt: dict[str, list[dict]] = defaultdict(list)
    if (out / "dataset" / "turn_by_turn.csv").exists():
        with open(out / "dataset" / "turn_by_turn.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                tbt[r["case_id"]].append(r)
    from ..s07_checks.scoring import DEFAULT_SEVERITY, RANK
    from .cases import read_judges
    judged = read_judges(out / "dataset")

    # ---- gates and numbers
    gates = []
    for key, label in (("base_level", "original wording"), ("enriched_template", "reworded wording"),
                       ("enriched_llm", "LLM-reworded wording")):
        g = (R.get("release_gate") or {}).get(key)
        if not g:
            continue
        reasons = [f"{_plain_reason(r)} ({tier.replace('_', '-')} cases)" for tier, t in (g.get("tiers") or {}).items()
                   for r in t.get("reasons") or []]
        gates.append({"wording": label, "passed": bool(g.get("passed")), "reasons": reasons})
    res = R.get("results") or {}
    b, e = res.get("base_level") or {}, res.get("enriched") or {}
    numbers = {"cases_generated": len(R.get("bases") or []),
               "cases_tested": (res.get("sample") or {}).get("base_cases_tested", b.get("n")),
               "runs": len(rows), "accuracy_original": b.get("accuracy"), "accuracy_reworded": e.get("accuracy"),
               "proved": ((R.get("invariance") or {}).get("summary") or {}).get("acceptance_rate"),
               "defects_caught": ((R.get("mutation") or {}).get("mutation_score") or {}).get("all_suites_combined"),
               "speech_grounded_original": b.get("speech_grounded"), "outcome_original": b.get("outcome_accuracy"),
               "outcome_reworded": e.get("outcome_accuracy"),
               "judges": {k: v for k, v in ((R.get("turn_by_turn") or {}).get("llm_judges") or {}).items() if k != "to_review"} or None}

    # ---- risks: codes by severity, then by runs hit
    hit: Counter = Counter()
    example: dict[str, str] = {}
    for r in sorted(rows, key=lambda r: (r.get("arm") != "base", r["sid"])):
        for c in r.get("failure_codes") or []:
            hit[c] += 1
            example.setdefault(c, r["sid"])
    risks = sorted(({"code": c, "runs": n, "severity": DEFAULT_SEVERITY.get(c, "moderate"), "what": PLAIN.get(c, c.lower().replace("_", " ")),
                     "example": example[c]} for c, n in hit.items()),
                   key=lambda x: (-RANK.get(x["severity"], 0), -x["runs"], x["code"]))

    # ---- fingerprint
    sut = job.get("sut") or {}
    sut_full = json.loads((run / "sut.json").read_text(encoding="utf-8")) if (run / "sut.json").exists() else {}
    dev = next((x for x in events if x.get("type") == "device"), {})
    start = next((x for x in events if x.get("type") == "start"), {})
    done = next((x for x in events if x.get("type") == "done"), {})
    import time as _t
    fp = {"run": job.get("id") or (run.name if (run / "job.json").exists() else out.name),
          "started": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(start["t"])) if start.get("t") else None,
          "minutes": round((done.get("t", 0) - start.get("t", 0)) / 60, 1) if done.get("t") and start.get("t") else None,
          "journey": (R.get("domain") or {}).get("name") or job.get("pack"),
          "graph": (R.get("domain") or {}).get("graph"), "graph_sha": job.get("graph_sha"),
          "code": job.get("code"), "bot": res.get("bot") or (R.get("config") or {}).get("bot_name"),
          "model": (sut.get("model") or "gpt-41") if sut.get("kind") == "llm" else None,
          "prompt_sha": _sha(out / "sut_prompt.txt"), "custom_prompt": sut.get("custom_prompt"),
          "voice": (sut_full.get("voice") or sut.get("voice")) if sut.get("kind") == "llm" else None,
          "references": [r.get("name") for r in (sut_full.get("references") or sut.get("references") or [])],
          "seed": (R.get("config") or {}).get("seed"), "max_cases": (R.get("config") or {}).get("max_cases"),
          "runs_per_case": (R.get("config") or {}).get("runs_per_base"),
          "device": dev.get("device") or job.get("device"), "device_name": dev.get("name"),
          "store": Path(str((R.get("store") or {}).get("store_dir") or "")).name or None,
          "packages": (R.get("backend") or {}).get("versions")}

    # ---- case by case
    cases = []
    for r in rows:
        sid = r["sid"]
        c = tc.get(sid) or {}
        why = (fails.get(sid) or {}).get("why") or {}
        exp = _outcome(why.get("expected") or c.get("expected_outcome"))
        act = _outcome(why.get("actual")) if why.get("actual") else (dict(exp) if r.get("correct") else
                                                                     {"end": r.get("status"), "transfer_reason": r.get("transfer_reason"), "writes": [], "told": []})
        stages = [{"stage": k, "expected": v.get("expected"), "actual": v.get("actual")}
                  for k, v in (why.get("stages") or {}).items() if v.get("expected") != v.get("actual")]
        turns = [{"turn": t.get("turn"), "state": t.get("state"), "customer": t.get("customer_text"),
                  "exp_tools": "" if t.get("expected_tools") in (None, "-") else t["expected_tools"],
                  "exp_acts": t.get("expected_bot") or "",
                  "bot_tools": "" if t.get("actual_tools") in (None, "-") else t["actual_tools"],
                  "bot_acts": t.get("actual_bot_acts") or "", "bot_text": t.get("actual_bot_text") or "",
                  "speech": _speech(t.get("speech_check")), "match": t.get("turn_match") == "True",
                  "after_end": t.get("after_bot_ended") == "True",
                  "after": t.get("after_divergence") == "True"} for t in tbt.get(sid, [])]
        cases.append({"sid": sid, "base": r.get("seed") or sid.split("~")[0], "arm": r.get("arm") or "base",
                      "wording": "original" if (r.get("arm") or "base") == "base" else f"reworded ({r.get('arm')})",
                      "category": c.get("category") or r.get("base"), "correct": bool(r.get("correct")),
                      "severity": r.get("severity"), "codes": r.get("failure_codes") or [],
                      "question": c.get("policy_question"), "changed": c.get("enrichment_changed") or "",
                      "expected": exp, "actual": act, "stages": stages, "turns": turns,
                      "first_difference": next((t["turn"] for t in turns if not t["match"]), None),
                      "judges": _judges(judged.get(sid))})
    cases.sort(key=lambda x: (x["correct"], -RANK.get(x["severity"] or "none", 0), x["base"], x["arm"] != "base", x["sid"]))
    ready = bool(gates) and all(g["passed"] for g in gates if g["wording"] != "LLM-reworded wording")
    return {"run": fp["run"], "ready": ready, "gates": gates, "numbers": numbers, "risks": risks,
            "fingerprint": fp, "cases": cases}


# ----------------------------------------------------------------------------- the verdict as one standalone page
_CSS = """
:root { --ink:#141716; --ink2:#4B524E; --ink3:#737A76; --line:#E2E4DF; --sunk:#F0F1EE; --graph:#E3F2EE; --ok:#15703A; --okbg:#E4F4EA;
  --bad:#B42318; --badbg:#FDECEA; --warn:#8A5A00; --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
* { box-sizing:border-box; } body { margin:0; padding:28px; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif; color:var(--ink); background:#fff; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:26px 0 6px; } h3 { font-size:14px; margin:18px 0 6px; }
.muted { color:var(--ink3); } .small { font-size:12.5px; } .mono { font-family:var(--mono); font-size:12px; }
.banner { border-radius:10px; padding:12px 14px; margin:12px 0; } .banner.ok { background:var(--okbg); color:var(--ok); } .banner.bad { background:var(--badbg); color:var(--bad); }
.banner .big { font-size:22px; font-weight:700; }
.tiles { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:8px; margin:10px 0; }
.tile { border:1px solid var(--line); border-radius:8px; padding:8px 10px; } .tile .v { font-size:18px; font-weight:650; } .tile .l { font-size:11.5px; color:var(--ink3); }
table { border-collapse:collapse; width:100%; font-size:12.5px; } th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { background:var(--sunk); font-size:11.5px; color:var(--ink2); }
.chip { display:inline-block; padding:1px 7px; border-radius:6px; font:11.5px var(--mono); background:var(--sunk); margin:1px 2px; }
.chip.bad { background:var(--badbg); color:var(--bad); } .pill { display:inline-block; padding:1px 9px; border-radius:999px; font-size:12px; font-weight:600; }
.pill.ok { background:var(--okbg); color:var(--ok); } .pill.bad { background:var(--badbg); color:var(--bad); }
dl.fp { display:grid; grid-template-columns:150px minmax(0,1fr) 150px minmax(0,1fr); gap:3px 12px; font-size:12px; margin:0; }
dl.fp dt { color:var(--ink3); } dl.fp dd { margin:0; font-family:var(--mono); font-size:11.5px; overflow-wrap:anywhere; }
.case { border:1px solid var(--line); border-radius:10px; margin:14px 0; padding:12px; } .case.fail { border-left:3px solid var(--bad); }
.case header { display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
.grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:10px; }
.side { border-radius:8px; padding:7px 9px; overflow-wrap:anywhere; } .side.exp { background:var(--graph); } .side.bot { background:var(--sunk); }
.side.diff { box-shadow:inset 3px 0 0 var(--bad); } .lbl { font:600 10.5px ui-sans-serif,system-ui,sans-serif; text-transform:uppercase; letter-spacing:.05em; color:var(--ink3); }
.side .mono { font-size:11.5px; display:block; } .said { margin-top:3px; } .speech { color:var(--bad); font-size:12px; }
.turn { border:1px solid var(--line); border-radius:8px; padding:7px; margin-top:8px; } .turn.bad { border-color:var(--bad); } .turn.after { opacity:.75; }
.cust { display:flex; gap:10px; align-items:baseline; font-size:13px; margin-bottom:6px; } .cust .who { font:600 11px var(--mono); color:var(--ink3); }
.cust .mark { margin-left:auto; font-size:12px; font-weight:650; } .ok { color:var(--ok); } .bad { color:var(--bad); } .warn { color:var(--warn); }
details > summary { cursor:pointer; color:var(--ink2); font-size:12.5px; margin-top:8px; }
.page1 { break-after:page; }
@media print { body { padding:0; font-size:11.5px; } .case header, .turn, .grid.head { break-inside:avoid; } details { display:block; } details > * { display:block; } }
"""


def _judges(j: dict | None) -> dict | None:
    """One run's two LLM judges, as the verdict shows them beside the rules."""
    if not j:
        return None
    c = j.get("combined") or {}
    return {"status": c.get("status"), "rules": c.get("rules"),
            "a": {k: (j.get("a") or {}).get(k) for k in ("verdict", "reason", "first_wrong_turn")},
            "b": {k: (j.get("b") or {}).get(k) for k in ("verdict", "reason", "first_wrong_turn")}}


def render_html(V: dict[str, Any]) -> str:
    """The verdict as a standalone page (out/verdict.html): page one is the one-pager, every case follows."""
    from html import escape as e
    pc = lambda x: "–" if x is None else f"{100 * x:.1f}%"           # noqa: E731
    N, F = V["numbers"], V["fingerprint"]
    gates = "".join(f"<div><b>{'✓ Passed' if g['passed'] else '✗ Not passed'} on {e(g['wording'])}</b>"
                    f"{'' if g['passed'] else ': ' + e('; '.join(g['reasons']))}</div>" for g in V["gates"])
    tested = N["cases_tested"] if N["cases_tested"] in (None, N["cases_generated"]) else f"{N['cases_tested']} of {N['cases_generated']}"
    J = N.get("judges") or {}
    tile_list = [
        (tested, "cases tested"), (pc(N["accuracy_original"]), "correct, original wording"), (pc(N["accuracy_reworded"]), "correct, reworded"),
        (pc(N["proved"]), "reworded cases proved to mean the same"), ("not run" if N["defects_caught"] is None else pc(N["defects_caught"]), "planted defects caught"),
        (pc(N["speech_grounded_original"]), "runs where everything said was true")]
    if J:
        tile_list += [(pc(J.get("judges_agree")), "the two LLM judges agree"),
                      (pc(J.get("a_agrees_with_rules")), "judge A agrees with the rules"),
                      (pc(J.get("b_agrees_with_rules")), "judge B agrees with the rules")]
    tiles = "".join(f"<div class='tile'><div class='v'>{e(str(v))}</div><div class='l'>{e(l)}</div></div>" for v, l in tile_list)
    risks = ("<table><tr><th>Severity</th><th>What the bot did</th><th>Code</th><th>Runs</th><th>Example</th></tr>" + "".join(
        f"<tr><td><span class='chip {'bad' if r['severity'] in ('critical', 'high') else ''}'>{e(r['severity'])}</span></td><td>{e(r['what'])}</td>"
        f"<td class='mono'>{e(r['code'])}</td><td>{r['runs']}</td><td class='mono'><a href='#case-{e(r['example'])}'>{e(r['example'])}</a></td></tr>"
        for r in V["risks"][:6]) + "</table>") if V["risks"] else "<p class='banner ok'>No failure in any run.</p>"
    code = F.get("code")
    fp_rows = [("Run", F["run"]), ("Started", f"{F['started']} · {F['minutes']} min" if F.get("started") else None), ("Journey", F["journey"]),
               ("Policy graph", f"{F['graph']}" + (f" · sha {F['graph_sha']}" if F.get("graph_sha") else "") if F.get("graph") else None),
               ("Code", f"commit {code['commit']}" + (" + uncommitted changes" if code.get("uncommitted_changes") else "") if code else "not recorded"),
               ("Bot under test", F.get("bot")), ("Model", F.get("model")),
               ("Model prompts", f"sha {F['prompt_sha']}" + (" (edited)" if F.get("custom_prompt") else "") if F.get("prompt_sha") else None),
               ("Wording voice", F.get("voice")),
               ("Reference files", ", ".join(F.get("references") or []) or None),
               ("Seed · cases · runs", f"{F.get('seed')} · {F.get('max_cases') or 'all'} · {F.get('runs_per_case') or 1} per case"),
               ("Device", F.get("device")), ("Knowledge store", F.get("store")),
               ("Packages", ", ".join(f"{k} {v}" for k, v in (F.get("packages") or {}).items()) or None)]
    fp = "".join(f"<dt>{e(k)}</dt><dd>{e(str(v))}</dd>" for k, v in fp_rows if v)

    def side(cls, label, lines, extra=""):
        return f"<div class='side {cls}'><div class='lbl'>{e(label)}</div>" + "".join(f"<span class='mono'>{e(x)}</span>" for x in lines) + extra + "</div>"

    def turn(t):
        mark = ("ok", "✓ same") if t["match"] else ("bad", "✗ the bot had already ended the call") if t.get("after_end") else \
            ("warn", "after a difference") if t["after"] else ("bad", "✗ first difference")
        bot_extra = (f"<div class='said'>“{e(t['bot_text'])}”</div>" if t["bot_text"] else "") + \
            (f"<div class='speech'>Speech check: {e(t['speech'])}</div>" if t["speech"] else "")
        return (f"<div class='turn {'ok' if t['match'] else 'after' if t['after'] else 'bad'}'><div class='cust'><span class='who'>Turn {e(str(t['turn']))}"
                f"{' · ' + e(t['state']) if t.get('state') else ''}</span><span>{e(t['customer'] or '')}</span><span class='mark {mark[0]}'>{mark[1]}</span></div>"
                f"<div class='grid'>{side('exp', '', [x for x in (t['exp_tools'], t['exp_acts'] or 'nothing (the call has ended)') if x])}"
                f"{side('bot', '', [x for x in (t['bot_tools'], t['bot_acts'] or '–') if x], bot_extra)}</div></div>")

    def judges_line(j):
        if not j:
            return ""
        cls = "ok" if j["status"] == "agree" else "bad" if j["status"] in ("judges disagree with the rules", "judges split") else ""
        one = lambda n, v: (f"<span class='mono'>{n}: <b class='{'ok' if v.get('verdict') == 'correct' else 'bad' if v.get('verdict') == 'incorrect' else 'warn'}'>"  # noqa: E731
                            f"{e(str(v.get('verdict') or '–'))}</b></span> <span class='small muted'>{e(v.get('reason') or '')}</span>")
        return (f"<div class='side' style='margin-bottom:8px'><div class='lbl'>LLM judges, beside the rules (the rules decide) · "
                f"<span class='{cls}'>{e(j['status'] or '')}</span></div><div>{one('A, auditor', j['a'])}</div><div>{one('B, reviewer', j['b'])}</div></div>")

    def case(c):
        out_lines = lambda o: [f"ends {o['end'] or '?'}" + (f" (transfer: {o['transfer_reason']})" if o.get("transfer_reason") else "")] + (o["writes"] or ["no write"])  # noqa: E731
        same = c["expected"] == c["actual"]
        cut = len(c["turns"]) if c["first_difference"] is None else next(i for i, t in enumerate(c["turns"]) if str(t["turn"]) == str(c["first_difference"])) + 2
        head, rest = c["turns"][:cut], c["turns"][cut:]
        sub = " · ".join(x for x in (c["wording"], c["category"], f"asks: {c['question']}" if c.get("question") else "",
                                      f"reworded: {c['changed']}" if c.get("changed") else "") if x)
        return (f"<section class='case {'' if c['correct'] else 'fail'}' id='case-{e(c['sid'])}'><header><div><b class='mono'>{e(c['sid'])}</b><div class='small muted'>{e(sub)}</div></div>"
                f"<div><span class='pill {'ok' if c['correct'] else 'bad'}'>{'passed' if c['correct'] else 'failed · ' + e(str(c['severity']))}</span> "
                + "".join(f"<span class='chip bad'>{e(x)}</span>" for x in c["codes"]) + "</div></header>"
                + judges_line(c.get("judges"))
                + f"<div class='grid head'>{side('exp', 'Expected outcome (policy graph)', out_lines(c['expected']))}"
                f"{side('bot' + ('' if same else ' diff'), 'The bot’s outcome · ' + ('same' if same else 'different'), out_lines(c['actual']))}</div>"
                + (f"<p class='small'><b>Stages that differ:</b> {e(', '.join(s['stage'] for s in c['stages']))}</p>" if c["stages"] else "")
                + (f"<div class='grid small muted' style='margin-top:8px'><div class='lbl'>The policy graph expects</div><div class='lbl'>The bot did and said</div></div>" if c["turns"] else "")
                + "".join(turn(t) for t in head)
                + (f"<details><summary>{len(rest)} more turn{'s' if len(rest) > 1 else ''} after the first difference</summary>{''.join(turn(t) for t in rest)}</details>" if rest else "")
                + ("" if c["turns"] else "<p class='small muted'>No turn-by-turn record for this run.</p>") + "</section>")
    fails = sum(not c["correct"] for c in V["cases"])
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Verdict · {e(F['run'])}</title><style>{_CSS}</style></head><body>"
            f"<div class='page1'><div class='small muted'>Verdict</div><h1>{e(str(F['journey']))} · {e(str(F.get('bot') or ''))}</h1><div class='mono muted'>{e(F['run'])}</div>"
            f"<div class='banner {'ok' if V['ready'] else 'bad'}'><div class='big'>{'Ready to release' if V['ready'] else 'Not ready to release'}</div>{gates}</div>"
            f"<div class='tiles'>{tiles}</div><h3>Top Risks</h3><p class='small muted'>What went wrong most seriously and most often, each with a case to look at.</p>{risks}"
            f"<h3>How to Reproduce It</h3><p class='small muted'>The same journey, graph, code, bot and settings give the same cases and the same verdict.</p><dl class='fp'>{fp}</dl></div>"
            f"<h2>Case by Case: What the Policy Expects, What the Bot Did</h2><p class='small muted'>{len(V['cases'])} runs, {fails} failed (shown first). "
            "Left: the graph's answer key for each turn (the tools to call and the decisions to make). Right: what the bot under test called, decided and said. "
            "The first turn that differs is marked; later turns are folded.</p>"
            + "".join(case(c) for c in V["cases"]) + "</body></html>")


def write(out_dir: str | Path) -> dict[str, Any]:
    """Build the verdict of a finished campaign and write out/verdict.json and out/verdict.html; returns a summary."""
    out = Path(out_dir)
    V = build(out)
    (out / "verdict.json").write_text(json.dumps(V, indent=1, default=str), encoding="utf-8")
    (out / "verdict.html").write_text(render_html(V), encoding="utf-8")
    return {"ready": V["ready"], "gates": V["gates"], "risks": V["risks"][:6], "runs": len(V["cases"]),
            "failed": sum(not c["correct"] for c in V["cases"]), "files": ["verdict.html", "verdict.json"]}
