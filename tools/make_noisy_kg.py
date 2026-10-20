"""A noisy copy of the policy graph, for training / demonstrating GEODE's duplicate and contradiction cleaning.

    python tools/make_noisy_kg.py
        -> gmsjourney/policy/charge_review_kg_noisy.json           (feed this to CELL 2 / CELL 3)
           gmsjourney/policy/charge_review_kg_noisy.answers.json   (answer key: never read by the cleaner)

Every original triple is kept unchanged; planted triples are inserted at seeded positions with new ids
(T0725 ...), the way a second extraction pass would add them. Two families:

duplicates      the same fact again: an exact copy, a number written as text, a head or tail written with
                other case / spacing, a customer synonym instead of the canonical value, the same fact quoted
                from another section;
contradictions  a single-valued relation given a second, wrong tail: a misread figure with the original quote,
                a figure with a quote that is not in the document, an unknown rule or dialog state, a wrong
                stage or tool stage whose quote is copied from the true triple.

The cleaned graph (``PolicyKG.load`` on the noisy file) must equal the original: tests/test_selfchecks.py checks it.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmsjourney.knowledge.policy_kg import KG_PATH, _content_hash  # noqa: E402


def _find(triples, head, relation, tail=None, kind=None, **qual):
    for t in triples:
        if t["head"] == head and t["relation"] == relation and (tail is None or t["tail"] == tail) and \
                (kind is None or t["kind"] == kind) and \
                all((t.get("qualifiers") or {}).get(k) == v for k, v in qual.items()):
            return t
    raise KeyError((head, relation, tail, qual))


def plant(triples: list[dict]) -> list[tuple[dict, dict]]:
    """(planted triple, answer) pairs. Each planted triple starts from the original it duplicates / contradicts."""
    out = []

    def add(orig, family, how, expected_action, **change):
        t = copy.deepcopy(orig)
        t.update(change)
        out.append((t, {"family": family, "how": how, "expected_action": expected_action, "original_id": orig["id"],
                        "original": [orig["head"], orig["relation"], orig["tail"]]}))

    dup, con = "duplicate", "contradiction"
    D, C, Q = "drop_duplicate", "drop_contradiction", "quarantine_for_review"
    # ---- duplicates
    add(_find(triples, "standard_delivery", "has_fee_usd", 0), dup, "exact copy", D)
    add(_find(triples, "PRESENT_TXN", "leads_to", "SUMMARIZE", on="INFORM_DISPOSITION"), dup,
        "exact copy of a transition (same customer act and condition)", D)
    add(_find(triples, "expedited_delivery", "has_fee_usd", 15), dup, "number written as text", D, tail="15")
    add(_find(triples, "lookback_days", "has_value", 30), dup, "number written as text with a decimal point", D,
        tail="30.0")
    add(_find(triples, "PRESENT_TXN", "calls_tool", "record_charge_response"), dup,
        "head written with other case", D, head="Present_Txn")
    add(_find(triples, "replace_card", "delivered_by", "standard_delivery", kind="policy"), dup, "tail written with a space instead of _", D, tail="standard delivery")
    add(_find(triples, "ASK_CLARIFY", "interprets_yes_as", "RECOGNIZED"), dup,
        "customer synonym instead of the canonical value", D, tail="mine")
    add(_find(triples, "PROVIDE_CONTEXT", "interprets_no_as", "UNRECOGNIZED"), dup,
        "customer synonym instead of the canonical value", D, tail="not mine")
    orig = _find(triples, "max_auth_attempts", "has_value")
    fire = _find(triples, "auth_failed", "fires_in_state", "ASK_AUTH_RETRY")
    add(orig, dup, "same fact quoted from another section", D, source=fire["source"], evidence=fire["evidence"])
    # ---- contradictions
    add(_find(triples, "expedited_delivery", "has_fee_usd", 15), con, "misread figure, original quote", C, tail=150)
    add(orig, con, "misread figure, original quote", C, tail=3)
    add(_find(triples, "standard_delivery", "has_business_days_max", 7), con, "misread figure, original quote", C,
        tail=10)
    lb = _find(triples, "lookback_days", "has_value", 30)
    add(lb, con, "figure with a quote that is not in the document", C, tail=31,
        evidence=lb["evidence"].replace("30", "31"))
    add(_find(triples, "auth_failed", "justified_by"), con, "rule id that does not exist", C, tail="R99")
    lt = _find(triples, "PRESENT_TXN", "leads_to", "SUMMARIZE", on="INFORM_DISPOSITION")
    add(lt, con, "transition to a dialog state that does not exist", C, tail="ASK_PIN")
    add(_find(triples, "ASK_AUTH", "in_stage", "authenticate"), con, "wrong stage, quote copied", Q, tail="capture")
    add(_find(triples, "authenticate_customer", "has_stage"), con, "wrong tool stage, quote copied", Q,
        tail="capture")
    add(_find(triples, "CONFIRM_RECORDED", "in_stage", "record"), con, "wrong stage, quote copied", Q,
        tail="transfer")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=str(KG_PATH))
    ap.add_argument("--out", default=str(KG_PATH.with_name(KG_PATH.stem + "_noisy.json")))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    data = json.loads(Path(a.policy).read_text(encoding="utf-8"))
    triples = data["triples"]
    planted = plant(triples)
    n = max(int(t["id"][1:]) for t in triples)
    rng = random.Random(a.seed)
    noisy = list(triples)
    answers = []
    for i, (t, ans) in enumerate(planted, 1):
        t["id"] = f"T{n + i:04d}"
        pos = rng.randrange(0, len(noisy) + 1)
        noisy.insert(pos, t)
        answers.append({"id": t["id"], **ans, "planted": [t["head"], t["relation"], t["tail"]]})
    out = {k: v for k, v in data.items() if k != "sha256"}
    out["triples"] = noisy
    out["sha256"] = _content_hash(out)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    key = Path(a.out).with_suffix(".answers.json")
    key.write_text(json.dumps({"source": Path(a.policy).name, "noisy": Path(a.out).name, "seed": a.seed,
                               "original_triples": len(triples), "planted": answers}, indent=1) + "\n")
    print(f"{a.out}: {len(triples)} original + {len(planted)} planted triples "
          f"({sum(p['family'] == 'duplicate' for p in answers)} duplicates, "
          f"{sum(p['family'] == 'contradiction' for p in answers)} contradictions); answer key {key}")


if __name__ == "__main__":
    main()
