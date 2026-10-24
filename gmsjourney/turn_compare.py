"""Turn-by-turn: what the oracle says the bot should do, next to what the bot did.

For one scenario (base or enriched) the oracle's expected run and the bot's actual run
are cut into customer turns. For every turn the table shows:

* the dialog state the customer is answering (the bot act that prompted the turn) and
  the customer's words (the text the bot received) with their meaning card;
* EXPECTED (oracle): the tool calls the bot should make in response, with arguments and
  results, and the bot acts it should produce (e.g. ``PRESENT_TXN(T2: Starbucks $6.25)``,
  ``ANSWER_KB(value -> 0)``, ``TRANSFER(card_options_unavailable)``). The oracle is an
  executable reading of the policy: it decides *what* to do, not the wording;
* ACTUAL (bot): the tool calls the bot made, the acts it produced (structured acts from the
  bot, or ``BotActClassifier`` when the bot returns text only) and its text;
* tools_match (same tools, same key arguments, in order), acts_match (same decision acts:
  optional acknowledgements such as CONFIRM_RECORDED and a re-presentation of the same
  charge after a clarification are ignored, see ``normalize_acts``), turn_match (both), and
  the first turn where the two runs diverge.

The customer is simulated reactively (journey.UserSimulator): after the first divergence
the bot has asked something the oracle did not, so the conversations may legitimately
differ; those rows are marked ``after_divergence`` and are diagnostic only.

How this relates to grading (evaluation/runner.score): the campaign does not require the
bot's reply at turn i to equal the oracle's at turn i. It grades the whole run - the
outcome, the per-stage decisions, and whether every step was admissible - because a bot
may legitimately take a different but allowed route. This module is the turn-level view of
the same two runs, to see where and how a failed run went wrong.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .sut.tools import key_args

ACT_FIELDS = ("txn", "intent", "answer", "reason", "recorded", "action", "delivery", "fee", "disposition")


# ----------------------------------------------------------------------------- rendering
def describe_act(a: dict, merchants: dict[str, tuple[str, str]] | None = None) -> str:
    """A bot act as a short readable string: PRESENT_TXN(T2: Starbucks $6.25)."""
    name = a.get("act", "?")
    if name == "OPENING":
        return "OPENING"
    parts = []
    if a.get("txn"):
        t = a["txn"]
        m = (merchants or {}).get(t)
        parts.append(f"{t}: {m[0]} ${m[1]}" if m else t)
    if name == "ANSWER_KB" and (a.get("intent") or a.get("answer") is not None):
        parts.append(f"{a.get('intent')} -> {json.dumps(a.get('answer'), default=str)}")
    for k in ("reason", "action", "delivery", "disposition"):
        if a.get(k):
            parts.append(str(a[k]))
    if a.get("fee") is not None and name == "DISCLOSE_FEE":
        parts.append(f"${a['fee']}")
    if name == "SUMMARIZE" and a.get("recorded") is not None:
        rec = a["recorded"]
        parts.append(", ".join(f"{k}={v}" for k, v in (rec.items() if isinstance(rec, dict) else rec)) or "nothing")
    return f"{name}({'; '.join(parts)})" if parts else name


def describe_tool(e: dict) -> str:
    args = e.get("args") or {}
    keep = {k: v for k, v in args.items() if k in key_args().get(e["tool"], tuple(args))}
    res = e.get("result")
    status = (res.get("status") or res.get("error")) if isinstance(res, dict) else ("error" if e.get("ok") is False else res)
    a = ", ".join(f"{k}={v}" for k, v in keep.items())
    return f"{e['tool']}({a}) -> {status}"


# Acknowledgements the bot may add or omit without changing what it does; the record they
# acknowledge is checked on the tool call itself.
OPTIONAL_ACTS = {"OPENING", "CONFIRM_RECORDED"}
REASK_AFTER = {"ASK_CLARIFY", "PROVIDE_CONTEXT", "ANSWER_KB"}


def normalize_acts(acts: list[dict]) -> list[tuple]:
    """The decision content of a bot reply: (act, txn, reason) in order, without optional
    acknowledgements, without a re-presentation of the same charge right after a
    clarification / context / policy answer, and without immediate repeats."""
    out: list[tuple] = []
    prev = None
    for a in acts:
        name = a.get("act")
        if name in OPTIONAL_ACTS:
            continue
        if name == "PRESENT_TXN" and prev is not None and prev.get("act") in REASK_AFTER \
                and (prev.get("txn") in (None, a.get("txn"))) and prev.get("act") != "ANSWER_KB":
            prev = a
            continue
        # a policy answer is about the question, not the charge on screen; its correctness is
        # graded separately (kb_correct: the figure must equal the exact register)
        key = (name, None if name == "ANSWER_KB" else a.get("txn"), a.get("reason"))
        if not out or out[-1] != key:
            out.append(key)
        prev = a
    return out


def _tool_key(e: dict) -> tuple:
    args = e.get("args") or {}
    return (e["tool"], tuple(sorted((k, str(v)) for k, v in args.items() if k in key_args().get(e["tool"], ()))))


# ----------------------------------------------------------------------------- segmentation
def segment(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Cut a run into (opening bot acts, per customer turn: prompt state, customer, tools, bot acts/text)."""
    opening, turns, cur, last_act = [], [], None, None
    for ev in events:
        role = ev.get("role")
        if role == "user":
            cur = {"state": last_act, "frame": ev.get("frame"), "text": ev.get("text"), "tools": [], "acts": [],
                   "bot_text": []}
            turns.append(cur)
        elif role == "bot":
            acts = list(ev.get("acts") or [])
            if acts:
                last_act = acts[-1].get("act")
            if cur is None:
                opening += acts
            else:
                cur["acts"] += acts
                if ev.get("text"):
                    cur["bot_text"].append(ev["text"])
        elif role == "tool" and cur is not None:
            cur["tools"].append(ev)
        elif role == "tool":
            opening.append({"act": f"(tool {ev.get('tool')})"})
    return opening, turns


# ----------------------------------------------------------------------------- one scenario
def turn_by_turn(runner, oracle, scenario, faults=None, validator=None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expected (oracle) vs actual (bot) for one scenario, one row per customer turn."""
    item = scenario.base
    spec, fx = item.metadata["spec"], item.metadata["fixture"]
    merchants = {t["transaction_id"]: (t["merchant"], t["amount"]) for t in fx["transactions"]}
    expected = oracle.run(spec, fx)
    path = validator.base_path(item) if validator is not None else None
    res = runner.run(scenario, faults=faults, oracle_path=path)
    actual = res.extras["trace"]
    _, exp_turns = segment(expected.events)
    _, act_turns = segment(actual.events)
    ont = runner.ont
    rows, diverged_at = [], None
    for i in range(max(len(exp_turns), len(act_turns))):
        e = exp_turns[i] if i < len(exp_turns) else None
        a = act_turns[i] if i < len(act_turns) else None
        st = (e or a)["state"]
        frame = (a or e)["frame"]
        e_tools = [_tool_key(x) for x in (e["tools"] if e else [])]
        a_tools = [_tool_key(x) for x in (a["tools"] if a else [])]
        e_acts = normalize_acts(e["acts"]) if e else []
        a_acts = normalize_acts(a["acts"]) if a else []
        tools_ok, acts_ok = e_tools == a_tools, e_acts == a_acts
        ok = tools_ok and acts_ok and e is not None and a is not None
        if not ok and diverged_at is None:
            diverged_at = i
        card = ont.card(frame, st) if frame is not None else {}
        text = (a or {}).get("text") or ""
        if not text and e is not None and path is not None and i < len(path):
            f0, ctx = path[i]                         # what the customer would have said on the expected path
            att = scenario.metadata.get("seed_overrides", {}).get(i, 0)
            seed = (f"{scenario.sid}|{i}" + (f"#{att}" if att else "")) if scenario.factor_levels else f"{item.qid}|{i}|base"
            text = "(expected path) " + runner.realizer.realize(f0, scenario.factor_levels, seed, ctx)
        rows.append({
            "turn": i, "state": st,
            "customer_text": text,
            "customer_card": json.dumps({"act": card.get("act"), "slots": card.get("slots")}) if card else "",
            "expected_tools": " ; ".join(describe_tool(x) for x in (e["tools"] if e else [])) or ("-" if e else "(no turn)"),
            "actual_tools": " ; ".join(describe_tool(x) for x in (a["tools"] if a else [])) or ("-" if a else "(no turn)"),
            "expected_bot": " ; ".join(describe_act(x, merchants) for x in (e["acts"] if e else [])) or "-",
            "actual_bot_acts": " ; ".join(describe_act(x, merchants) for x in (a["acts"] if a else [])) or "-",
            "actual_bot_text": " ".join(a["bot_text"]) if a else "",
            "expected_next_state": e_acts[-1][0] if e_acts else "-",
            "actual_next_state": a_acts[-1][0] if a_acts else "-",
            "tools_match": tools_ok, "acts_match": acts_ok, "turn_match": ok,
            "after_divergence": diverged_at is not None and i > diverged_at,
        })
    from .evaluation.runner import score
    row = score(scenario, res, oracle.limits, oracle.store.workflow)
    summary = {
        "case_id": scenario.sid, "base_id": item.qid, "turns_expected": len(exp_turns), "turns_actual": len(act_turns),
        "first_divergence_turn": diverged_at, "turns_matching": sum(r["turn_match"] for r in rows),
        "expected_outcome": json.loads(item.answer) if isinstance(item.answer, str) else item.answer,
        "actual_outcome": json.loads(res.answer) if isinstance(res.answer, str) else res.answer,
        "outcome_correct": bool(row["correct"]), "workflow_adherent": bool(row["workflow_adherent"]),
        "violations": row["violations"],
        "stages_wrong": [k[2:] for k, v in row.items() if k.startswith("c_") and v == 0],
        "expected_opening": " ; ".join(describe_act(x) for x in segment(expected.events)[0]),
        "actual_opening": " ; ".join(describe_act(x) for x in segment(actual.events)[0]),
    }
    return pd.DataFrame(rows), summary


def show_turns(runner, oracle, scenario, validator=None, display=None):
    df, s = turn_by_turn(runner, oracle, scenario, validator=validator)
    show = display or (lambda d: print(d.to_string(index=False)))
    print(f"{s['case_id']}: outcome {'CORRECT' if s['outcome_correct'] else 'WRONG'} | "
          f"turns matching {s['turns_matching']}/{max(s['turns_expected'], s['turns_actual'])} | "
          f"first divergence: {s['first_divergence_turn']} | stages wrong: {s['stages_wrong'] or '-'}")
    print("expected outcome:", s["expected_outcome"])
    print("actual outcome:  ", s["actual_outcome"])
    show(df[["turn", "state", "customer_text", "expected_tools", "actual_tools", "expected_bot", "actual_bot_acts",
             "turn_match"]])
    return df, s


# ----------------------------------------------------------------------------- many scenarios
def export_turn_by_turn(out_dir: str | Path, runner, oracle, scenarios, validator=None,
                        show: bool = True) -> dict[str, Any]:
    """dataset/turn_by_turn.csv (one row per turn per scenario) + turn_by_turn_summary.csv."""
    from .evaluation.runner import progress_bar
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trows, srows = [], []
    bar = progress_bar(len(scenarios), "turn-by-turn export", show)
    for scn in scenarios:
        if bar is not None:
            bar.update(1)
        df, s = turn_by_turn(runner, oracle, scn, validator=validator)
        for r in df.to_dict(orient="records"):
            trows.append({"case_id": scn.sid, "base_id": scn.base.qid,
                          "case_type": "base" if not scn.factor_levels else "enriched",
                          "arm": scn.metadata.get("arm", "base" if not scn.factor_levels else "template"), **r})
        srows.append({**s, "case_type": "base" if not scn.factor_levels else "enriched",
                      "arm": scn.metadata.get("arm", "base" if not scn.factor_levels else "template")})

    if bar is not None:
        bar.close()

    def write(path, rows):
        if not rows:
            return
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v for k, v in r.items()})

    write(out / "turn_by_turn.csv", trows)
    write(out / "turn_by_turn_summary.csv", srows)
    n = len(srows) or 1
    return {"scenarios": len(srows), "turn_rows": len(trows),
            "fully_matching_runs": sum(s["turns_matching"] == max(s["turns_expected"], s["turns_actual"]) for s in srows),
            "outcome_correct": sum(s["outcome_correct"] for s in srows),
            "first_divergence_by_state": _divergence_states(trows), "share_outcome_correct": sum(
                s["outcome_correct"] for s in srows) / n}


def _divergence_states(trows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    seen = set()
    for r in trows:
        if r["case_id"] in seen or r["turn_match"]:
            continue
        seen.add(r["case_id"])
        out[r["state"] or "-"] = out.get(r["state"] or "-", 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))
