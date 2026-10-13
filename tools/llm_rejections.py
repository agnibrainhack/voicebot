"""Why are LLM-written turns rejected? A breakdown of an LLM enrichment export.

    python tools/llm_rejections.py output/llm_cell9d/llm_enrichment_turns.csv
    python tools/llm_rejections.py output/campaign_out/dataset/llm_enrichment_turns.csv --examples 8

Counts, over every attempt the model made: which check failed (G0 form, G1 facts, G3 one clear
decision, G2 rule reader, G2 independent reader), broken down by the meaning card's act and dialog
state and by factor level, the most common reasons, and example attempts with both readers' cards.
It also reports how many turns needed the template fallback and how many scenarios survived.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict

CHECKS = ("G0_form", "G1_facts", "G3_one_decision", "G2_reader", "G2_dual_read")


def _j(v):
    try:
        return json.loads(v) if v else None
    except ValueError:
        return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--examples", type=int, default=5)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    if not rows:
        print("empty file")
        return
    scen = {}
    attempts, failed, by_act, by_level, reasons, errors = 0, Counter(), defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    src = Counter()
    old_format = [False]
    no_verdict = [0]
    examples = defaultdict(list)
    for r in rows:
        scen[r["sid"]] = r.get("scenario_accepted") == "True"
        src[r.get("turn_source") or "llm"] += 1
        act = re.match(r"([A-Z_]+)", r.get("meaning_card") or "")
        key = f"{act.group(1) if act else '?'} @ {r.get('state')}"
        levels = _j(r.get("levels")) or {}
        log = _j(r.get("attempt_log"))
        if log is None:      # an export written before attempt_log existed: the final attempt of each turn only
            if r.get("turn_accepted") != "True" and not any(r.get(f"gate_{c}") for c in CHECKS):
                no_verdict[0] += 1           # no check ran on this turn: a failed or skipped gateway call
                continue
            log = [{"text": r.get("llm_final"), "rule_read": r.get("rule_reader_card"),
                    "independent_read": r.get("independent_reader_card"), "reasons": _j(r.get("reasons")),
                    "failed": [c for c in CHECKS if r.get(f"gate_{c}") == "False"]}]
            old_format[0] = True
        for at in log:
            if at.get("error"):
                errors[re.sub(r"\d+", "#", at["error"])[:80]] += 1
                continue
            attempts += 1
            fl = at.get("failed") or []
            for c in fl:
                failed[c] += 1
                by_act[key][c] += 1
                for k, v in levels.items():
                    by_level[f"{k}={v}"][c] += 1
            by_act[key]["attempts"] += 1
            for k, v in levels.items():
                by_level[f"{k}={v}"]["attempts"] += 1
            for x in (at.get("reasons") or [])[:3]:
                reasons[re.sub(r"\d+", "#", x)[:100]] += 1
            if fl and len(examples[fl[0]]) < a.examples:
                examples[fl[0]].append((key, r.get("meaning_card"), at.get("text"), at.get("rule_read"),
                                        at.get("independent_read"), at.get("reasons")))
    ok = sum(scen.values())
    print(f"scenarios: {len(scen)}  accepted: {ok} ({ok / max(1, len(scen)):.0%})")
    print(f"turns: {len(rows)}  by source: {dict(src)}")
    print(f"model attempts: {attempts}  transport errors: {sum(errors.values())}")
    if old_format[0]:
        print("(older export without attempt_log: only each turn's final attempt is counted)")
    if no_verdict[0]:
        print(f"turns with NO verdict (a gateway call failed, or the turn was never reached): {no_verdict[0]} "
              f"- these reject a conversation without any check failing")
    rej = [sid for sid, ok in scen.items() if not ok]
    if rej:
        good = sum(1 for r in rows if r["sid"] in rej and r.get("turn_accepted") == "True")
        tot = sum(1 for r in rows if r["sid"] in rej)
        print(f"rejected scenarios: {len(rej)}; {good} of their {tot} turns passed on their own "
              f"(a scenario is rejected when any one turn fails)")
    print("\nfailed checks over all attempts (an attempt can fail several):")
    for c in CHECKS:
        print(f"  {c:18s} {failed[c]:5d}  ({failed[c] / max(1, attempts):.0%} of attempts)")
    print("\nby meaning card @ dialog state (attempts / failures per check):")
    for k, c in sorted(by_act.items(), key=lambda kv: -sum(kv[1][x] for x in CHECKS)):
        print(f"  {k:40s} attempts {c['attempts']:4d}  " + "  ".join(f"{x.split('_', 1)[1]}={c[x]}" for x in CHECKS if c[x]))
    print("\nfactor levels with the highest failure rate (>= 5 attempts):")
    rate = [(sum(c[x] for x in CHECKS) / c["attempts"], k, c) for k, c in by_level.items() if c["attempts"] >= 5]
    for rt, k, c in sorted(rate, reverse=True)[:15]:
        print(f"  {k:40s} {rt:5.0%} of {c['attempts']} attempts  " + "  ".join(f"{x.split('_', 1)[1]}={c[x]}" for x in CHECKS if c[x]))
    print("\nmost common reasons:")
    for x, n in reasons.most_common(15):
        print(f"  {n:4d}  {x}")
    if errors:
        print("\ntransport errors:")
        for x, n in errors.most_common(5):
            print(f"  {n:4d}  {x}")
    for c, ex in examples.items():
        print(f"\n--- examples failing {c}")
        for key, card, text, rr, ir, rs in ex:
            print(f"  [{key}] want {card}\n    text: {text}\n    rule reader: {rr} | independent reader: {ir}\n    why: {rs}")


if __name__ == "__main__":
    main()
