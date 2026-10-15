"""Generate an answer sheet (HTML) for every base case of a campaign.

    python tools/answer_sheet.py --campaign output/campaign_llm --out docs/ANSWER_SHEET.html

The cases are rebuilt deterministically from the policy graph (same seed and the same
construction as ``pipeline.run_campaign``), and the required bot behaviour is read from
the oracle's own trace — so the tool calls, statuses and knowledge-base answers in the
sheet are what the system actually computes, not a transcription.

For each case the sheet shows:
  * the customer's line (the base realisation) and the canonical meaning card it must
    read back to;
  * the acts a correct bot must perform, in plain words, and the tool calls in order
    with the results the mock bank returns;
  * the dialog state's accepted acts, rules and limits, from the policy graph;
  * the expected outcome (end state, transfer reason, recorded dispositions) and the
    per-stage expectations;
  * collapsed, the same turn in every enriched variant of the case - both arms (template
    and LLM), each labelled with its changed factors and its proof verdict. An accepted
    variant carries the identical meaning card and therefore the identical answer key (see
    ``gmsjourney/enrich/invariance.py`` and ``gmsjourney/enrich/llm/validate.py``); a rejected
    one is shown for audit only.

The enriched wording and the base/enriched pairing are read from the campaign's
``dataset/test_cases.csv`` and ``dataset/test_turns.csv`` when they are present; the
sheet renders without them. Datasets written before the ``arm`` column existed are read
as template-only.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

# Runnable from any directory, with or without PYTHONPATH: Python puts a script's own directory on
# sys.path, not the working directory, so the package root has to be added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsjourney.bases.journey import JourneyBaseGenerator, to_item      # noqa: E402
from gmsjourney.bases.kb import mine_kb_items
from gmsjourney.enrich.invariance import InvarianceValidator
from gmsjourney.enrich.reader import FrameReader
from gmsjourney.enrich.realizers import Realizer
from gmsjourney.journey import JourneySpec
from gmsjourney.ontology import SemanticFrame, sequence_signature
from gmsjourney.oracle import PolicyOracle
from gmsjourney.pipeline import build_gms, tags_for

ROOT = Path(__file__).resolve().parents[1]
E = html.escape

NICE = {"standard_delivery": "standard delivery", "expedited_delivery": "expedited delivery",
        "SECURE_CARD": "securing the card", "REPLACE_CARD": "replacing the card"}


def say(*p):
    print(*p, flush=True)


def nice(v) -> str:
    if isinstance(v, list):
        lo, hi = v[0], v[-1]
        return f"{lo}" if lo == hi else f"{lo}–{hi}"
    return NICE.get(str(v), str(v).replace("_", " "))


def read_limits(store) -> dict:
    """The figures the answer phrasings need, if the store's register exposes them."""
    out: dict = {}
    for category, key, name in (("delivery_fee_usd", "expedited_delivery", "expedited_fee"),
                                ("delivery_fee_usd", "standard_delivery", "standard_fee"),
                                ("limit", "lookback_days", "lookback_days")):
        try:
            out[name] = int(store.lookup_enm(category, key))
        except Exception:                                        # noqa: BLE001 - optional
            pass
    return out


def kb_phrase(frame: SemanticFrame | None, answer, limits: dict) -> str:
    """The answer to a knowledge-base question, phrased from the question's own frame."""
    i = frame.get("intent") if frame is not None else None
    ent = frame.get("entity") if frame is not None else None
    if answer in (None, "absent"):
        return ("<b>there is no such option or fact in the policy</b> — the bot must say so and must not "
                "invent a figure")
    ent_s = nice(ent)
    if i == "value" and frame.get("attr") == "fee" and not isinstance(answer, list):
        return f"<b>{ent_s} costs ${answer}</b>" + (" (free)" if str(answer) == "0" else "")
    if i == "value" and frame.get("attr") == "days":
        return f"<b>{ent_s} takes {nice(answer)} business days</b>"
    if i == "limit":
        return {
            "lookback_days": f"<b>{answer} days</b> — charges older than that cannot be reviewed",
            "max_transactions_per_session": f"<b>{answer} transactions</b> can be reviewed in one call",
            "max_auth_attempts": f"<b>{answer} attempts</b> are allowed to verify the card",
        }.get(str(frame.get("key")), f"<b>{nice(answer)}</b>")
    if i == "plausibility":
        want = int(frame.get("value") or 0)
        actual = limits.get("expedited_fee" if ent == "expedited_delivery" else "standard_fee")
        if actual is None:
            return f"<b>{nice(answer)}</b> — is {ent_s} ${want}?"
        return (f"<b>{'yes' if want == actual else 'no'}</b> — {ent_s} "
                f"{'does' if want == actual else 'does not'} cost ${want}")
    special = {
        "faster": lambda: f"<b>{ent_s or nice(answer)} is faster</b>",
        "cheaper": lambda: f"<b>{ent_s or nice(answer)} is cheaper</b>",
        "fee_difference": lambda: f"<b>${answer} more</b> for expedited delivery than standard",
        "count_options": lambda: f"<b>{nice(answer)} delivery options</b>",
        "refund": lambda: ("<b>no refund is provided</b> — an unrecognised charge is recorded as a "
                           "customer-reported concern, not refunded automatically"),
        "fraud": lambda: ("<b>no</b> — an unrecognised charge is not by itself proof of fraud (R27); it is "
                          "recorded as a customer-reported concern"),
        "skip_auth": lambda: "<b>no</b> — identity verification cannot be skipped",
        "window_check": lambda: (f"<b>{'yes' if int(frame.get('days') or 0) <= limits.get('lookback_days', 30) else 'no'}</b>"
                                 f" — the review window is {limits.get('lookback_days', 30)} days"),
        "needs_confirmation": lambda: f"<b>{nice(answer)} needs the customer’s clear confirmation</b>",
        "faster_option_fee": lambda: f"<b>${answer}</b> for the fastest option",
    }
    if i in special:
        return special[i]()
    if isinstance(answer, list):
        return "<b>" + ", ".join(nice(x) for x in answer) + "</b>"
    return f"<b>{E(nice(answer))}</b>"


def render_act(act: dict, frame: SemanticFrame | None, fx: dict, outcome: dict, limits: dict) -> str:
    a = act.get("act")
    txn = act.get("txn")
    txns = {t["transaction_id"]: t for t in (fx.get("transactions") or [])}
    t = txns.get(txn, {})
    where = str(txn) + (f" ({t.get('merchant')} ${t.get('amount')})" if t else "")
    if a == "OPENING":
        return ("Greets and explains the call: “" + E(str(act.get("text")
                or "Before I review account-specific charges, I need to authenticate you.")) + "”")
    if a == "ASK_AUTH":
        return "Asks for the last four digits of the customer’s card."
    if a == "ASK_AUTH_RETRY":
        return "Says the digits did not match and asks again (attempt 2 of the 2 allowed)."
    if a == "PRESENT_TXN":
        return f"Reads out a transaction — <b>{E(where)}</b> — and asks whether the customer recognises it."
    if a == "PROVIDE_CONTEXT":
        return f"Repeats which charge is being discussed — <b>{E(where)}</b> — and asks again."
    if a == "ASK_CLARIFY":
        return "Says it did not catch that and asks the customer to repeat (one clarification is allowed)."
    if a == "CONFIRM_RECORDED":
        d = act.get("disposition")
        return (f"Confirms it recorded <b>{E(str(d))}</b> for <b>{E(str(txn))}</b>."
                if d else f"Confirms the response for <b>{E(str(txn))}</b> was recorded.")
    if a == "INFORM_NO_CARD_ACTION":
        return "Says no card action options are available on this account."
    if a == "CONFIRM_ACTION":
        act_name = str(act.get("action", "")).replace("_", " ").lower()
        dl = str(act.get("delivery", "")).replace("_", " ")
        return (f"Asks the customer to confirm the card action (<b>{E(act_name)}</b>"
                f"{', ' + E(dl) if dl else ''}) before submitting.")
    if a == "DISCLOSE_FEE":
        return (f"Discloses the <b>${E(str(act.get('fee', 15)))}</b> "
                f"{E(str(act.get('delivery', 'expedited delivery')).replace('_', ' '))} fee and asks for a "
                f"clear confirmation.")
    if a == "ACTION_DONE":
        return "Confirms the card action was submitted successfully."
    if a == "ASK_MORE_HELP":
        return "Asks whether there is anything else it can help with."
    if a == "EXPLAIN_LIMIT":
        return "Explains that a session limit has been reached."
    if a == "EXPLAIN_NO_TXNS":
        return "Explains that there are no charges available to review."
    if a == "SUMMARIZE":
        rec = act.get("recorded")
        return f"Summarises the call{f' (recorded so far: {E(json.dumps(rec))})' if rec else ''}."
    if a == "TRANSFER":
        return ("Tells the customer it is <b>transferring them to a human</b> — required reason: <code>"
                + E(str(act.get("reason") or outcome.get("transfer_reason") or "—")) + "</code>.")
    if a == "ANSWER_KB":
        return "Answers the policy question: " + kb_phrase(frame, act.get("answer"), limits) + "."
    if a == "END":
        return "Ends the call."
    return f"Emits <code>{E(str(a))}</code>."


# ---------------------------------------------------------------- plain-English narrative
STATE_EN = {
    "OPENING": "the greeting", "ASK_AUTH": "asking for the card digits", "ASK_AUTH_RETRY": "asking for the digits again",
    "PRESENT_TXN": "asking about a charge", "ANSWER_KB": "answering a policy question",
    "ASK_CLARIFY": "asking the caller to repeat", "PROVIDE_CONTEXT": "repeating the charge details",
    "CONFIRM_RECORDED": "confirming the recorded answer", "DISCLOSE_FEE": "disclosing the expedited fee",
    "CONFIRM_ACTION": "asking to confirm the card action", "ACTION_DONE": "confirming the card action is done",
    "ACK_DECLINED": "acknowledging the declined card action", "INFORM_NO_CARD_ACTION": "saying no card action is available",
    "EXPLAIN_LIMIT": "explaining a session limit", "EXPLAIN_NO_TXNS": "explaining there are no charges to review",
    "SUMMARIZE": "summarising the call", "ASK_MORE_HELP": "asking whether anything else is needed",
    "TRANSFER": "handing over to a human", "END": "the end of the call",
}
DIM_EN = {
    "auth": {"ok": "the caller verifies on the first try", "retry_ok": "the first card digits are wrong and the second try succeeds",
             "fail_twice": "both verification attempts fail", "unavailable": "the verification system is down",
             "human_at_auth": "the caller asks for a human before verifying"},
    "retrieve": {"unavailable": "the system that lists the charges is down"},
    "n_txn": {0: "there are no charges to review", 1: "one charge to review", 2: "two charges to review",
              4: "four charges (exactly the per-call limit)", 5: "five charges (one more than the per-call limit)"},
    "window": {"day30": "a charge exactly at the 30-day review limit", "day31": "a charge one day past the review limit"},
    "disposition": {"all_recognized": "the caller recognises the charges", "mixed": "the caller recognises some charges and not others",
                    "unsure_resolved": "the caller is unsure at first, then answers", "unsure_twice": "the caller stays unsure",
                    "ambiguous_resolved": "an unclear answer is cleared up after one clarification",
                    "ambiguous_twice": "the answer stays unclear after a clarification",
                    "correction": "the caller corrects an earlier answer"},
    "record": {"fail_first": "recording the first answer fails"},
    "card_intent": {"secure": "the caller asks to secure the card", "replace_standard": "the caller asks for a replacement card, standard delivery",
                    "replace_expedited": "the caller asks for a replacement card, expedited delivery"},
    "card_options": {"none": "no card action is available on the account", "unavailable": "the card-options system is down"},
    "expedited_confirm": {"switch_standard": "the caller switches to standard delivery when told the fee"},
    "submit": {"fail": "submitting the card action fails"},
    "human_mid": {"yes": "the caller asks for a human part-way through"},
    "kb_mid": {"ask": "the caller asks a policy question before answering"},
    "turn_pressure": {"at_limit": "the call reaches exactly the 20-turn limit", "over_limit": "the call would run past the 20-turn limit"},
}


def tool_en(ev: dict) -> str:
    res = ev.get("result") or {}
    st = res.get("status") or ("ERROR" if res.get("error") else "")
    tool = ev.get("tool", "")
    if res.get("transactions") is not None and tool == "get_review_transactions":
        n = len(res.get("transactions") or [])
        return f"<code>{E(tool)}</code> returns {E(st)} ({n} charge{'s' if n != 1 else ''})"
    if res.get("options"):
        return f"<code>{E(tool)}</code> returns {E(st)} ({E(', '.join(res['options']))})"
    return f"<code>{E(tool)}</code> returns <b>{E(st)}</b>"


def state_en(s: str | None) -> str:
    return f"{E(STATE_EN.get(s or '', str(s)))} (<code>{E(str(s))}</code>)"


def render_tool(ev: dict) -> str:
    args = ", ".join(f"{k}={v}" for k, v in (ev.get("args") or {}).items()) or "—"
    res = ev.get("result") or {}
    status = str(res.get("status", ""))
    extra = ""
    if res.get("transactions"):
        extra = " → " + ", ".join(f"{t['transaction_id']} {t['merchant']} ${t['amount']} ({t.get('age_days')}d)"
                                  for t in res["transactions"])
    elif res.get("options"):
        extra = f" → {', '.join(res['options'])}"
    elif res.get("reason"):
        extra = f" → reason {res['reason']}"
    return (f"<code>{E(ev.get('tool', ''))}({E(args)})</code> "
            f"<span class='res'>→ <b>{E(status)}</b>{E(extra)}</span>")


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
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.6;padding-inline:16px}
.wrap{max-width:1360px;margin:0 auto;padding-block:clamp(26px,5vw,60px)}
h1,h2,h3{font-family:var(--display);line-height:1.15;margin:0;text-wrap:balance}
h1{font-size:clamp(1.9rem,5vw,2.9rem);font-weight:700}
h2{font-size:1.5rem;font-weight:700;margin-block:1.6rem .4rem}
h3{font-size:1.06rem;font-weight:600;display:flex;flex-wrap:wrap;gap:.5rem;align-items:baseline}
.lede{color:var(--muted);font-size:1.06rem;max-width:68ch}
code{font-family:var(--mono);font-size:.85em;background:var(--code-bg);border:1px solid var(--line-soft);border-radius:4px;padding:.06em .3em;word-break:break-word}
.kicker{display:inline-flex;align-items:center;gap:.5rem;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.16em;color:var(--accent)}
.kicker::before{content:"";width:22px;height:2px;background:var(--accent)}
.case{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:clamp(14px,2vw,20px);margin-block:14px;scroll-margin-top:14px}
.chead{display:flex;flex-direction:column;gap:.25rem;border-bottom:1px solid var(--line-soft);padding-bottom:.6rem}
.cat{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);background:var(--accent-soft);padding:.12rem .45rem;border-radius:4px}
.kind{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);border:1px solid var(--line);padding:.1rem .45rem;border-radius:4px}
.cmeta{font-size:.8rem;color:var(--muted)}
.kbmeta{font-size:.88rem;margin-top:.5rem}
.verdict{display:grid;gap:.2rem;margin:.7rem 0 .3rem;font-size:.92rem}
.who{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);display:block}
.turns{display:grid;gap:10px;margin-top:.7rem}
.turn{border:1px solid var(--line-soft);border-radius:9px;padding:.7rem .85rem;min-width:0}
.turn .hd{display:flex;gap:.5rem;align-items:baseline;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin-bottom:.4rem}
.turn .hd b{font-family:var(--display);font-size:.92rem;letter-spacing:0;text-transform:none;color:var(--ink)}
.turn .hd span{border-left:1px solid var(--line);padding-left:.5rem}
.say,.mean,.must,.ctx{display:grid;grid-template-columns:6.4rem 1fr;gap:.5rem;margin:.3rem 0}
.must{border-left:3px solid var(--pass);background:var(--pass-soft);border-radius:0 7px 7px 0;padding:.45rem .6rem}
.must ul{margin:0;padding-left:1.05rem}
.must li{margin:.15rem 0}
.steps{margin:.15rem 0 0;padding-left:1.4rem}
.steps li{margin:.2rem 0;padding-left:.15rem}
.steps li::marker{font-family:var(--mono);font-size:.78rem;color:var(--muted)}
.sk{display:inline-block;min-width:2.9rem;margin-right:.4rem;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.steps li.tool .sk{color:var(--accent)}
.hops{margin-top:.25rem}
.hops table{font-size:.8rem}
.hops td,.hops th{padding:.3rem .45rem}
tr.unused td{opacity:.55}
tr:target td{background:var(--accent-soft)}
.story{display:grid;gap:.45rem;margin:.7rem 0;padding:.65rem .8rem;border:1px solid var(--line-soft);border-radius:9px;background:var(--code-bg);font-size:.92rem}
.story ol{margin:.15rem 0 0;padding-left:1.3rem}
.story li{margin:.2rem 0}
.plain{font-size:.9rem;margin:.35rem 0;color:var(--ink)}
.pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.7fr);gap:10px;align-items:start}
.plaincol{position:sticky;top:3rem;border:1px solid var(--line-soft);border-radius:9px;padding:.65rem .8rem;background:var(--code-bg);font-size:.9rem}
.plaincol .plain{margin:0}
@media (max-width:900px){.pair{grid-template-columns:1fr}.plaincol{position:static}}
.viewbar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;padding:.5rem 0;background:var(--bg)}
.viewbar button{font:inherit;font-size:.82rem;padding:.3rem .7rem;border-radius:6px;border:1px solid var(--line);background:var(--surface);color:var(--ink);cursor:pointer}
.viewbar button[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
body.view-side .storylist{display:none}
body.view-details .story,body.view-details .plaincol{display:none}
body.view-details .pair{grid-template-columns:1fr}
body.view-story .turns{display:none}
.brief{font-size:.95rem;line-height:1.55}
.brief .small{margin-top:.25rem}
.trans{display:grid;grid-template-columns:6.4rem 1fr;gap:.5rem;margin:.3rem 0;font-size:.86rem;align-items:baseline}
.trans code{font-weight:600}
.arr{font-family:var(--mono);font-size:.72rem;color:var(--muted);white-space:nowrap;cursor:help}
.res{color:var(--muted)}
details{margin-top:.5rem}
summary{cursor:pointer;font-size:.82rem;color:var(--accent)}
.enr{font-size:.88rem;background:var(--code-bg);border-radius:6px;padding:.5rem .6rem;margin-top:.4rem;overflow-wrap:anywhere}
.enr.rej{opacity:.75;border-left:3px solid var(--fail)}
.vmeta{display:flex;flex-wrap:wrap;gap:.35rem;align-items:baseline;font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-bottom:.25rem}
.tag{border-radius:4px;padding:0 .35rem;font-size:.68rem;text-transform:uppercase;letter-spacing:.06em}
.tag.template{background:var(--code-bg);border:1px solid var(--line);color:var(--ink)}
.tag.llm{background:var(--accent-soft);color:var(--accent)}
.tag.ok{background:var(--pass-soft);color:var(--pass)}
.tag.no{background:var(--fail-soft);color:var(--fail)}
.tag.gen{background:var(--code-bg);border:1px dashed var(--line);color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;vertical-align:top;padding:.45rem .6rem;border-bottom:1px solid var(--line-soft)}
th{font-family:var(--display);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.tscroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
.small{font-size:.84rem;color:var(--muted)}
.toc{columns:2;column-gap:2rem;font-size:.86rem;margin-top:.6rem}
@media (max-width:640px){.toc{columns:1}}
@media (max-width:520px){.say,.mean,.must,.ctx,.trans{grid-template-columns:1fr}}
.toc a{display:block;padding:.16rem 0;color:var(--ink);text-decoration:none;border-bottom:1px dashed var(--line-soft);font-family:var(--mono);font-size:.78rem}
.toc a:hover{color:var(--accent)}
footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.85rem;color:var(--muted)}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer sheet for every base case of a campaign.")
    ap.add_argument("--policy", default=str(ROOT / "gmsjourney" / "policy" / "charge_review_kg.json"))
    ap.add_argument("--campaign", default="output/campaign_llm",
                    help="campaign output dir (dataset/test_cases.csv, dataset/test_turns.csv)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "ANSWER_SHEET.html"))
    ap.add_argument("--backend", default="auto", choices=["auto", "knowlytix", "builtin"])
    ap.add_argument("--store-dir", default=None, help="knowlytix store dir (default: the build cache)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summaries", default="template", choices=["template", "llm"],
                    help="llm: add a short model-written summary to each case, written from the case's fact sheet "
                         "and checked against it (the deterministic story is always shown)")
    ap.add_argument("--llm-model", default="", help="gateway deployment for --summaries llm (default gpt-41)")
    ap.add_argument("--llm-workers", type=int, default=8, help="concurrent gateway calls for --summaries llm")
    ap.add_argument("--max-turn-limit-cases", type=int, default=None,
                    help="as in the campaign config (default: read from <campaign>/report.json, else 2; -1 = no cap)")
    a = ap.parse_args()

    say("[1/4] the knowledge store (reloaded from the cache when one matches)")
    store, _build, info = build_gms(a.policy, backend_choice=a.backend, kx_store_dir=a.store_dir,
                                    seed=a.seed, log=say)
    say("      backend:", info.get("selected"))
    oracle, reader, realizer = PolicyOracle(store), FrameReader(store), Realizer()
    kg = getattr(store, "kg", None)
    validator = InvarianceValidator(store, oracle, reader, realizer)
    limits = read_limits(store)

    say("[2/4] rebuilding the base cases (covering array + closure, then the policy questions)")
    try:
        _kbq = json.loads((Path(a.campaign) / "report.json").read_text())["config"].get("kb_questions", "standard")
    except (OSError, ValueError, KeyError):
        _kbq = "standard"
    items, _info = mine_kb_items(store, a.seed, extended=_kbq == "extended")   # as the campaign mined them
    admitted = [k for k in items if k.admission != "rejected"]
    mtl = a.max_turn_limit_cases
    if mtl is None:                         # the campaign's own setting, so the case ids match its dataset
        try:
            mtl = json.loads((Path(a.campaign) / "report.json").read_text())["config"].get("max_turn_limit_cases", None)
        except (OSError, ValueError, KeyError):
            mtl = 2
    elif mtl < 0:
        mtl = None
    gen = JourneyBaseGenerator(oracle, kb_pool=[k.frame for k in admitted], seed=a.seed, max_turn_limit_cases=mtl)
    cover, rep = gen.covering()
    kb_traces = [oracle.run(JourneySpec(n_txn=1, kb_mid="ask", kb_frames=(k.frame,))) for k in admitted]
    extra = gen.closure(cover, store, rep, extra_traces=kb_traces)
    specs = list({s.key(): s for s in cover + extra}.values())
    bases = []
    for i, sp in enumerate(specs):
        it = to_item(sp, oracle, f"J{i:03d}")
        it.metadata["tags"] = tags_for(sp)
        bases.append(it)
    for k in admitted:
        it = to_item(JourneySpec(n_txn=1, kb_mid="ask", kb_frames=(k.frame,)), oracle, k.qid,
                     category=k.category, kb_item=k)
        it.metadata["tags"] = [k.category]
        bases.append(it)
    kb_by_qid = {k.qid: k for k in admitted}
    say(f"      {len(specs)} journey bases + {len(admitted)} policy-question bases")

    # optional: the campaign's enriched wording and its base -> enriched pairing
    camp = Path(a.campaign)
    variants: dict[str, list[dict]] = {}          # base_id -> every enriched variant (both arms)
    enr_text, enr_src = {}, {}
    if (camp / "dataset" / "test_cases.csv").exists():
        with open(camp / "dataset" / "test_cases.csv", encoding="utf-8") as f:
            for x in csv.DictReader(f):
                if x["case_type"] != "enriched":
                    continue
                arm = x.get("arm") or ("llm" if x["case_id"].endswith("~llm") else "template")
                variants.setdefault(x["base_id"], []).append({
                    "case_id": x["case_id"], "arm": arm, "changed": x.get("enrichment_changed", ""),
                    "accepted": str(x.get("proof_accepted", "")).lower() == "true",
                    "status": x.get("proof_status") or ("accepted" if str(x.get("proof_accepted", "")).lower()
                                                        == "true" else "rejected")})
        for v in variants.values():                # accepted first, template before LLM, then by id
            v.sort(key=lambda r: (not r["accepted"], r["arm"] != "template", r["case_id"]))
    if (camp / "dataset" / "test_turns.csv").exists():
        with open(camp / "dataset" / "test_turns.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                enr_text[(r["case_id"], r["turn"])] = r["enriched_text"]
                enr_src[(r["case_id"], r["turn"])] = r.get("turn_source") or ""
    allv = [r for v in variants.values() for r in v]
    say(f"      enriched variants: {len(allv)} for {len(variants)} base cases "
        f"({sum(r['arm'] == 'template' for r in allv)} template, {sum(r['arm'] == 'llm' for r in allv)} LLM; "
        f"{sum(r['status'] == 'rejected' for r in allv)} rejected, "
        f"{sum(r['status'] == 'not_generated' for r in allv)} not generated)"
        if variants else "      no campaign dataset found: rendering without the enriched lines")

    def variant_block(base_id: str, turn: int) -> str:
        rows = []
        for v in variants.get(base_id, []):
            t = enr_text.get((v["case_id"], str(turn)), "")
            if not t:
                continue
            ok = {"accepted": '<span class="tag ok">accepted</span>',
                  "not_generated": '<span class="tag gen">not generated</span>'}.get(
                v["status"], '<span class="tag no">rejected</span>')
            rows.append(f'<div class="enr{"" if v["accepted"] else " rej"}"><div class="vmeta">'
                        f'<span class="tag {E(v["arm"])}">{E(v["arm"])}</span>{ok}'
                        + ('<span class="tag gen">template turn</span>'
                           if enr_src.get((v["case_id"], str(turn))) in ("template_fallback", "template_by_design")
                           else "")
                        + f'<span>{E(v["case_id"])}</span>'
                        f'<span>{E(v["changed"] or "base levels")}</span></div>{E(t)}</div>')
        if not rows:
            return ""
        n_ok = sum(v["accepted"] for v in variants.get(base_id, []))
        n_llm = sum(v["arm"] == "llm" for v in variants.get(base_id, []))
        return (f'<details><summary>same test, other wordings — {len(rows)} enriched variants '
                f'({n_llm} LLM, {n_ok} accepted)</summary>{"".join(rows)}</details>')

    say("[3/4] rendering each case from the oracle's own trace")
    blocks = []
    from collections import Counter as _Counter
    used_edges: _Counter = _Counter()
    inexact_hops: list = []                         # hops whose states match a graph transition but not its trigger               # policy-graph transition triple id -> hops in this sheet
    wf_cond = {w.wid: f"{w.src} -> {w.dst} : {w.condition}" for w in getattr(store, "workflow", [])}
    from gmsjourney.narrate import case_facts
    facts_all: list[dict] = []                      # one fact sheet per case (for --summaries llm)
    for it in bases:
        spec, fx = it.metadata["spec"], it.metadata["fixture"]
        tr = oracle.run(spec, fx)
        outcome = json.loads(it.answer) if isinstance(it.answer, str) else it.answer
        path = validator.base_path(it)
        states = validator._states.get(it.qid, [None] * len(path))
        sig = sequence_signature([validator.ont.canonicalize(f, s) for (f, _), s in zip(path, states)], states)

        # every bot act and tool call after a customer turn, in the exact order of the oracle's trace
        def render_steps(steps, frame):
            if not steps:
                return '<ol class="steps"><li class="small">(the bot says nothing more: the call has ended)</li></ol>'
            li = []
            for kind, x in steps:
                if kind == "tool":
                    li.append(f'<li class="tool"><span class="sk">calls</span>{render_tool(x)}</li>')
                else:
                    li.append(f'<li class="act"><span class="sk">says</span>{render_act(x, frame, fx, outcome, limits)}</li>')
            return '<ol class="steps">' + "".join(li) + "</ol>"

        def hops_of(st, frame, steps):
            """Split a turn's ordered steps into dialog-state hops: (before, trigger, tool calls, after, says).
            The customer's meaning triggers the first hop; the bot's own moves after it are 'auto'."""
            known = set(kg.states) if kg is not None else set()
            hops, cur, trig, tools, pending_says = [], st, (frame.act if frame is not None else "call start"), [], []
            for kind, x in steps:
                if kind == "tool":
                    tools.append(x)
                elif x.get("act") in known:
                    hops.append({"src": cur, "on": trig, "tools": tools, "dst": x.get("act"), "act": x})
                    cur, trig, tools = x.get("act"), "auto", []
            ended = (tools and tools[-1].get("tool") == "transfer_to_human") or (cur in ("TRANSFER", "END"))
            if tools or (ended and cur != "END"):
                hops.append({"src": cur, "on": trig, "tools": tools, "dst": "END" if ended else None, "act": None})
            return hops

        def graph_edge(src, dst, on):
            """The policy graph's leads_to triple for this hop (exact trigger first), or None."""
            if kg is None or not src or not dst:
                return None
            cand = [t for t in kg.where("leads_to", head=src) if t["tail"] == dst]
            exact = next((t for t in cand if (t.get("qualifiers") or {}).get("on") == on), None)
            if exact is None and cand:                 # same states, different trigger: report it, never pass it off
                inexact_hops.append((it.qid, src, on, dst, [(c["id"], (c.get("qualifiers") or {}).get("on")) for c in cand]))
            return exact

        def transitions(st, frame, steps):
            hops = hops_of(st, frame, steps)
            if not hops:
                return ""
            rows_ = []
            for n, h in enumerate(hops, 1):
                t = graph_edge(h["src"], h["dst"], h["on"])
                q = (t or {}).get("qualifiers") or {}
                used_edges.update([t["id"]] if t else [])
                tools_ = "<br>".join(render_tool(x) for x in h["tools"]) or '<span class="small">—</span>'
                rules = " ".join(f'<code title="{E(kg.rule_text(r))}">{E(r)}</code>' for r in q.get("rules") or [])
                edges = " ".join(f'<code title="{E(wf_cond.get(w, ""))}">{E(w)}</code>' for w in q.get("edges") or [])
                if t:
                    ref = (f'<a href="#{E(t["id"])}"><code>{E(t["id"])}</code></a> '
                           f'<span class="small">{E(str(t.get("source")))}: “{E(str(t.get("evidence") or ""))}”</span>')
                    cond = E(str(q.get("when") or "always"))
                elif h["src"] is None:                  # the call itself starting: the graph begins at OPENING
                    ref, cond = '<span class="small">entry state of the policy graph</span>', "call begins"
                else:
                    ref = '<span class="tag no">not in the policy graph</span>' if h["dst"] else ""
                    cond = "—"
                rows_.append(f"<tr><td>{n}</td><td><code>{E(str(h['src'] or 'start'))}</code></td>"
                             f"<td><code>{E(str(h['on']))}</code></td><td>{tools_}</td><td>{cond}</td>"
                             f"<td><code>{E(str(h['dst'] or '(stays)'))}</code></td>"
                             f"<td>{rules}{' · ' if rules and edges else ''}{edges}</td><td>{ref}</td></tr>")
            return ('<div class="tscroll hops"><table><thead><tr><th>#</th><th>State before</th><th>Trigger</th>'
                    '<th>Tool calls → result</th><th>Transition condition</th><th>State after</th>'
                    '<th>Rules · workflow</th><th>Policy graph</th></tr></thead><tbody>' + "".join(rows_)
                    + "</tbody></table></div>")

        groups, opening, turn_i = [], [], -1
        for ev in tr.events:
            if ev["role"] == "user":
                turn_i += 1
                groups.append({"turn": turn_i, "steps": []})
            elif ev["role"] == "tool":
                (groups[-1]["steps"] if groups else opening).append(("tool", ev))
            else:
                (groups[-1]["steps"] if groups else opening).extend(("act", a) for a in ev.get("acts") or [])

        rows = []
        if opening:
            rows.append('<div class="pair"><div class="plaincol">@@PLAIN_O@@</div>'
                        '<div class="turn opening"><div class="hd"><b>Call opens</b>'
                        '<span>before any customer turn</span></div>'
                        '<div class="who" style="margin-top:.4rem">State transitions</div>' + transitions(None, None, opening)
                        + '<div class="must"><div class="who">Correct bot behaviour, in order</div>'
                        + render_steps(opening, None) + "</div></div></div>")
        for g in groups:
            i = g["turn"]
            frame, ctx = path[i] if i < len(path) else (None, {})
            st = states[i] if i < len(states) else None
            card = validator.ont.card(frame, st)
            base_text = realizer.realize(frame, {}, f"{it.qid}|{i}|base", ctx) if frame else ""
            steps = render_steps(g["steps"], frame)
            cardj = json.dumps({"act": card.get("act"), "slots": card.get("slots")}, ensure_ascii=False)
            if kg is not None and st in getattr(kg, "prompt_states", []):
                ctxline = (f"accepts <code>{E(', '.join(kg.accepted_acts(st)))}</code> · "
                           f"rules <code>{E(' '.join(kg.rules_for(st)))}</code> · "
                           f"limits <code>{E(' '.join(kg.limits_for(st)))}</code>")
            else:
                ctxline = ""
            rows.append(f"""
      <div class="pair"><div class="plaincol">@@PLAIN_{i}@@</div>
      <div class="turn">
        <div class="hd"><b>Turn {i}</b><span>{E(str(st))}</span><span>{E(str(card.get('stage') or ''))}</span></div>
        <div class="say"><span class="who">Customer</span><span>“{E(base_text)}”</span></div>
        <div class="mean"><span class="who">means</span><span><code>{E(cardj)}</code></span></div>
        <div class="who" style="margin-top:.4rem">State transitions</div>
        {transitions(st, frame, g["steps"])}
        <div class="must">
          <div class="who">Correct bot behaviour, in order</div>
          {steps}
        </div>
        <div class="ctx"><span class="who">state rules</span><span class="small">{ctxline}</span></div>
        {variant_block(it.qid, i)}
      </div></div>""")

        # ---- plain-English story of the case, from the same hops and policy-graph transitions
        from gmsjourney.enrich.llm.prompts import describe_meaning

        def rule_en(rules):
            rules = [r for r in rules or [] if kg is not None and kg.rule_text(r)]
            return "; ".join(f"<code>{E(r)}</code> “{E(kg.rule_text(r).rstrip('.'))}”" for r in rules[:2])

        def hop_sentence(h, first):
            t = graph_edge(h["src"], h["dst"], h["on"])
            q = (t or {}).get("qualifiers") or {}
            why = E(str(q.get("when") or "always"))
            tools_ = ", ".join(tool_en(x) for x in h["tools"])
            lead = ("" if not tools_ else f"{tools_}; ")
            if h["src"] is None:
                return f"The call starts at {state_en(h['dst'])}."
            subj = "the bot moves" if first else "it then moves"
            rr = rule_en(q.get("rules"))
            cond = "" if why == "always" else f"because <i>{why}</i>, "
            return (f"{lead}{cond}{subj} from {state_en(h['src'])} to {state_en(h['dst'] or h['src'])}"
                    + (f" — {rr}" if rr else "") + ".")

        def turn_story(st, frame, steps, who):
            hops = hops_of(st, frame, steps)
            m_ = describe_meaning(frame).split(" - BOTH parts must be said")[0] if frame is not None else ""
            said = (f"The customer {E(m_)}. " if frame is not None else "")
            parts = [hop_sentence(h, n == 0) for n, h in enumerate(hops)]
            text = " ".join(p[0].upper() + p[1:] if p else p for p in parts)
            return f"<b>{E(who)}.</b> {said}{text}" if (said or text) else ""

        story = [turn_story(None, None, opening, "Opening")] if opening else []
        for g in groups:
            i = g["turn"]
            fr_ = path[i][0] if i < len(path) else None
            st_ = states[i] if i < len(states) else None
            story.append(turn_story(st_, fr_, g["steps"], f"Turn {i}"))
        rel_ = {d: v for d, v in (it.metadata.get("relevant") or {}).items() if v not in (None, "N/A")}
        tests = [DIM_EN.get(d, {}).get(v) for d, v in rel_.items()]
        tests = [x for x in tests if x]
        _k = kb_by_qid.get(it.qid)
        if _k is not None:                          # a single-question policy case: the question is what is tested
            tests = [f"one policy question — the customer {describe_meaning(_k.frame)}; the answer must be computed "
                     f"from the policy ({_k.answer!s})"
                     + ("; the policy says nothing about it, so the bot must say it does not know" if _k.probe else "")]
        all_hops = []
        for g in groups:
            i = g["turn"]
            all_hops += hops_of(states[i] if i < len(states) else None, path[i][0] if i < len(path) else None, g["steps"])
        # the decisive step: walk back from the end past the wrap-up states to the step that started the ending
        final = None
        for h in reversed(all_hops):
            final = h
            if h["src"] not in ("SUMMARIZE", "TRANSFER", "END"):
                break
        endw = str(outcome.get("end"))
        rec = outcome.get("recorded") or []
        rec_en = ", ".join(f"{E(str(a))} {E(str(b))}" for a, b in rec) if isinstance(rec, list) else E(json.dumps(rec))
        ending = f"The call ends <b>{E(endw)}</b>"
        if outcome.get("transfer_reason"):
            ending += f" (reason <code>{E(str(outcome.get('transfer_reason')))}</code>)"
        if final is not None:
            t = graph_edge(final["src"], final["dst"], final["on"])
            q = (t or {}).get("qualifiers") or {}
            ending += (f". It is decided at the step from {state_en(final['src'])} to {state_en(final['dst'])}"
                       + (f", triggered by the customer's <code>{E(str(final['on']))}</code>" if final["on"] not in ("auto", "call start") else "")
                       + f", taken because <i>{E(str(q.get('when') or 'always'))}</i>")
            rr = rule_en(q.get("rules"))
            ending += (f" — {rr}" if rr else "")
        ending += "."
        ending += (f" Recorded: {rec_en}." if rec else " Nothing is recorded.")
        _turns = ([{"turn": "opening", "state": None, "trigger": "call start", "customer_says": None,
                    "customer_means": None, "steps": opening,
                    "bot_says": [render_act(x, None, fx, outcome, limits) for k_, x in opening if k_ == "act"]}]
                  if opening else [])
        for g in groups:
            i = g["turn"]
            fr_ = path[i][0] if i < len(path) else None
            _turns.append({"turn": i, "state": states[i] if i < len(states) else None,
                           "trigger": fr_.act if fr_ is not None else None,
                           "customer_says": realizer.realize(fr_, {}, f"{it.qid}|{i}|base", path[i][1]) if fr_ else None,
                           "customer_means": (describe_meaning(fr_).split(" - BOTH parts must be said")[0]
                                              if fr_ is not None else None),
                           "steps": g["steps"],
                           "bot_says": [render_act(x, fr_, fx, outcome, limits) for k_, x in g["steps"] if k_ == "act"]})
        _q = None
        if _k is not None:
            _q = {"question": realizer.realize(_k.frame, {}, f"{_k.qid}|kb|base"), "answer": _k.answer,
                  "policy_is_silent": bool(_k.probe)}
        facts_all.append(case_facts(it.qid, "policy question" if _k is not None else "journey", outcome,
                                    _turns, kg, _q))
        story_html = ('<div class="story">' f'@@BRIEF:{it.qid}@@' '<div><span class="who">What this case tests</span>'
                      + (E("; ".join(tests)).capitalize() + "." if tests else "The standard path with no complications.")
                      + '</div><div class="storylist"><span class="who">What happens</span><ol>'
                      + "".join(f"<li>{x}</li>" for x in story if x)
                      + f'</ol></div><div><span class="who">Why it ends this way</span>{ending}</div></div>')
        # the plain-English account of each turn, beside the turn's detailed view
        plain_by = {str(g["turn"]): s for g, s in zip(groups, story[1:] if opening else story)}
        if opening:
            plain_by["O"] = story[0]
        rows = [re.sub(r"@@PLAIN_(\w+)@@", lambda m: (f'<div class="plain"><span class="who">In plain words</span>'
                                                      f'{plain_by.get(m.group(1), "")}</div>'), r) for r in rows]

        k = kb_by_qid.get(it.qid)
        kbmeta = ""
        if k is not None:
            q = realizer.realize(k.frame, {}, f"{k.qid}|kb|base")
            ans = k.answer if isinstance(k.answer, str) else json.dumps(k.answer, default=str)
            kbmeta = (f"<div class='kbmeta'>Question under test: <b>{E(q)}</b> · expected answer "
                      f"<b>{E(ans)}</b> · computed by "
                      f"<code>{E(str(k.primitive))}</code> · admission <code>{E(str(k.admission))}</code>"
                      + (f" · hallucination probe (<code>{E(str(k.probe))}</code>)" if k.probe else "") + "</div>")
        blocks.append(f"""
  <section class="case" id="{E(it.qid)}">
    <div class="chead">
      <h3>{E(it.qid)} <span class="cat">{E(str(it.base))}</span>
        <span class="kind">{"policy question" if k is not None else "journey"}</span></h3>
      <div class="cmeta">tags <code>{E(', '.join(it.metadata.get('tags') or []))}</code> · customer turns <b>{len(groups)}</b>
        · conversation signature <code>{E(sig)}</code></div>
    </div>
    {kbmeta}
    {story_html}
    <div class="verdict"><span class="who">Correct outcome</span>
      <span>end <b>{E(str(outcome.get('end')))}</b>{f" · transfer reason <b>{E(str(outcome.get('transfer_reason')))}</b>" if outcome.get('transfer_reason') else ''}
      · recorded <code>{E(json.dumps(outcome.get('recorded')))}</code> ·
      per stage <code>{E(json.dumps(it.components))}</code></span></div>
    <div class="turns">{''.join(rows)}</div>
  </section>""")

    graph_table = ""
    if kg is not None:                              # every transition of the policy graph, with how often it is used
        rws = []
        for t in sorted(kg.where("leads_to"), key=lambda t: (t["head"], t["tail"])):
            q = t.get("qualifiers") or {}
            n = used_edges.get(t["id"], 0)
            rules_ = " ".join('<code title="%s">%s</code>' % (E(kg.rule_text(r)), E(r)) for r in q.get("rules") or [])
            edges_ = " ".join('<code title="%s">%s</code>' % (E(wf_cond.get(w, "")), E(w)) for w in q.get("edges") or [])
            sep = " · " if rules_ and edges_ else ""
            rws.append(f'<tr id="{E(t["id"])}" class="{"unused" if not n else ""}"><td><code>{E(t["id"])}</code></td>'
                       f'<td><code>{E(t["head"])}</code></td><td><code>{E(str(q.get("on")))}</code></td>'
                       f'<td>{E(str(q.get("when") or "always"))}</td><td><code>{E(t["tail"])}</code></td>'
                       f'<td>{rules_}{sep}{edges_}</td>'
                       f'<td class="small">{E(str(t.get("source")))}: “{E(str(t.get("evidence") or ""))}”</td>'
                       f'<td>{n}</td></tr>')
        graph_table = ('<div class="tscroll"><table><thead><tr><th>Triple</th><th>State before</th><th>Trigger</th>'
                       '<th>Condition</th><th>State after</th><th>Rules · workflow</th><th>Source (policy text)</th>'
                       '<th>Hops in this sheet</th></tr></thead><tbody>' + "".join(rws) + "</tbody></table></div>")
    state_table = ""
    if kg is not None:
        rws = []
        for s in kg.prompt_states:
            yes, no = kg.interprets(s)
            rws.append(f"<tr><td><code>{E(s)}</code></td><td>{E(', '.join(kg.accepted_acts(s)))}</td>"
                       f"<td class='small'>yes → {E(str(yes))} · no → {E(str(no))}</td>"
                       f"<td class='small'><code>{E(' '.join(kg.rules_for(s)))}</code></td>"
                       f"<td class='small'><code>{E(' '.join(kg.limits_for(s)))}</code></td></tr>")
        state_table = ("<table><thead><tr><th>Dialog state</th><th>Accepts (customer acts)</th>"
                       "<th>Bare yes / no means</th><th>Rules</th><th>Limits</th></tr></thead><tbody>"
                       + "".join(rws) + "</tbody></table>")

    say("[4/4] writing", a.out)
    doc = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answer Sheet</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{STYLE}</style>
<div class="wrap">
<span class="kicker">The answer key · every base case of the campaign</span>
<h1>Answer Sheet</h1>
<p class="lede">For each of the {len(bases)} cases: what the customer says, the meaning it carries, the dialog
state it answers and what that state allows, the tool calls a correct bot must make with the results the mock
bank returns, and the response it must give — followed by the expected outcome.</p>

<p><b>How to read a case.</b> <i>Customer</i> is the exact line the bot under test receives at the base levels.
<i>means</i> is the canonical meaning card the reader must recover — the ground truth, independent of wording.
<i>Correct bot behaviour</i> lists the acts the bot must perform and the tool calls in order, with the status
the mock bank returns. The collapsed list in a turn shows the same turn in every enriched variant of the case,
from both arms: <i>template</i> (rule-based rewording) and <i>LLM</i> (written by the model on the gateway), each with
the factors it changes. An <i>accepted</i> variant passed its invariance proof (<code>gmsjourney/enrich/invariance.py</code>;
for the LLM arm also the gates and the independent reader in <code>gmsjourney/enrich/llm/validate.py</code>), so it carries
the identical meaning card and therefore the identical answer key. A <i>rejected</i> variant is listed for audit only
and is not used in the run.</p>

<div class="tscroll">{state_table}</div>

<h2>Policy graph: dialog-state transitions</h2>
<p class="small">Every <code>leads_to</code> triple of the policy graph (<code>gmsjourney/policy/charge_review_kg.json</code>),
with the condition, the rules and workflow edges it carries and the policy text it was built from. Each transition in a case
links here; the last column counts how often the answer key uses it (greyed: no case in this sheet reaches it).</p>
{graph_table}

<h2>Index</h2>
<div class="toc">{''.join(f'<a href="#{E(b.qid)}">{E(b.qid)} · {E(str(b.base))}</a>' for b in bases)}</div>

<h2>Cases</h2>
<div class="viewbar" role="group" aria-label="View">
  <span class="who" style="display:inline">View</span>
  <button type="button" data-view="side" aria-pressed="true">Side by side</button>
  <button type="button" data-view="details" aria-pressed="false">Details only (previous view)</button>
  <button type="button" data-view="story" aria-pressed="false">Story only</button>
</div>
{''.join(blocks)}

<footer>Generated by <code>tools/answer_sheet.py</code> from the trained GMS store and the oracle
(backend: {E(str(info.get("selected")))}). Machine-readable twins in the campaign output:
<code>dataset/test_cases.csv</code>, <code>dataset/test_turns.csv</code>,
<code>dataset/policy_questions.csv</code>.<br>
See also: <a href="ANNOTATED.html">Frames, Reader, Oracle</a> (the three core files, line by line) ·
<a href="EXPLAINER.html">Inside gmsjourney</a> (the twelve campaign steps) ·
<a href="INVARIANCE.html">How Invariance Is Proven</a>.</footer>
</div>
"""
    briefs: dict = {}
    if a.summaries == "llm":                        # model-written, checked against each case's own fact sheet
        from gmsjourney.enrich.llm.client import LLMSettings
        from gmsjourney.narrate import LLMNarrator
        narr = LLMNarrator(LLMSettings(model=a.llm_model or ""), workers=a.llm_workers, log=say)
        why = narr.llm.unavailable_reason()
        if why:
            say(f"      LLM summaries skipped: {why}")
        else:
            briefs = {cid: txt for cid, (txt, src) in narr.summarize_all(facts_all).items() if src == "llm"}
    for cid in [f["case"] for f in facts_all]:
        tok = f"@@BRIEF:{cid}@@"
        doc = doc.replace(tok, (f'<div class="brief"><span class="who">In brief <span class="tag llm">LLM</span></span>'
                                f'{E(briefs[cid])}<div class="small">Written by the model from this case\'s facts '
                                f'(turns, transitions, tool results, rules) and checked against them; the step-by-step '
                                f'account below is computed directly from the policy graph.</div></div>')
                          if cid in briefs else "")
    doc += """<script>
(function () {
  var KEY = "answer-sheet-view";
  function set(v) {
    document.body.classList.remove("view-side", "view-details", "view-story");
    document.body.classList.add("view-" + v);
    document.querySelectorAll(".viewbar button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.view === v));
    });
    try { localStorage.setItem(KEY, v); } catch (e) {}
  }
  var v = "side";
  try { v = localStorage.getItem(KEY) || "side"; } catch (e) {}
  set(["side", "details", "story"].indexOf(v) >= 0 ? v : "side");
  document.querySelectorAll(".viewbar button").forEach(function (b) {
    b.addEventListener("click", function () { set(b.dataset.view); });
  });
})();
</script>
"""
    Path(a.out).write_text(doc, encoding="utf-8")
    if inexact_hops:
        say(f"      {len(inexact_hops)} hops match a graph transition only by states (trigger differs):", sorted({(x[1], x[2], x[3]) for x in inexact_hops}))
    say(f"wrote {a.out} ({len(doc) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
