"""Block 6 - build the dataset: design -> conversations -> rows.

  python -m turnval.generate --config turnval/config/default.yaml [--renderer template] [--limit 20]

Steps per design row (in parallel over rows):
  1. simulate the correct conversation (Block 3) and script the injected error (Block 4);
  2. write the words turn by turn (Block 5): the customer reacts to what the bot actually said (including an error),
     the bot's text follows its spec; a step summary of everything before the customer's turn;
  3. emit rows: the dialogue response, plus counterfactual responses for the same context (the correct response at
     the error turn, side negatives, an out-of-place response) - rows sharing (summary, customer) form one
     contrastive group.

Outputs in <output.dir>: design.csv (+report), conversations.jsonl, rows.csv, generation_report.json.
Block 7 (split.py) then writes train.csv / val.csv / test.csv.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import random
import time
import traceback
import zlib
from pathlib import Path
from typing import Any

import yaml

from .errors import BotSpec, build_script
from .factors import Factors, write_csv
from .graph import Flow, HRTGraph
from .scenario import HANDOFF_STATE, OPENING, simulate
from .textgen import facts_for, make_renderer, mask_digits

ROOT = Path(__file__).resolve().parent.parent
STYLE_FACTORS = ("cust_style", "cust_register", "cust_noise", "cust_emotion", "digit_format")


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    for k in ("hrt_graph", "aop_flow", "factors"):
        p = Path(cfg["data"][k])
        cfg["data"][k] = str(p if p.is_absolute() else ROOT / p)
    out = Path(cfg["output"]["dir"])
    cfg["output"]["dir"] = str(out if out.is_absolute() else ROOT / out)
    return cfg


def cv_prefix(profile: dict[str, Any], kind: str) -> list[tuple[str, str]]:
    """What came before a Charge Verification conversation: identification, short or in full."""
    if kind != "full_ui":
        return [("bot", "[User Identification completed: the caller was identified and verified for a fraud alert.]"),
                ("bot", profile["question"])]
    return [("bot", OPENING), ("customer", "Yes."),
            ("bot", "Before I can help, I'll need to verify your identity. What are the last four digits of your social security number?"),
            ("customer", "[REDACTED]"),
            ("bot", f"Are you asking about the card ending in {' '.join(profile['cm5'])}?"), ("customer", "Yes, that's the one."),
            ("bot", profile["question"])]


def _position(i: int, n: int) -> str:
    if n <= 1:
        return "early"
    f = i / (n - 1)
    return "early" if f < 1 / 3 else ("middle" if f < 2 / 3 else "late")


def _length_band(n: int) -> str:
    return "short" if n <= 5 else ("medium" if n <= 12 else "long")


class Generator:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.flow = Flow(cfg["data"]["aop_flow"])
        self.hrt = HRTGraph(cfg["data"]["hrt_graph"])
        self.factors = Factors.load(cfg["data"]["factors"])
        self.renderer = make_renderer(cfg["text"])
        self.mask = bool(cfg["text"].get("mask_customer_digits", True))

    def conversation(self, row: dict[str, str], idx: int) -> dict[str, Any]:
        seed = zlib.crc32(f"{row['design_id']}|{row.get('replicate', '0')}|{self.cfg['design']['seed']}".encode())
        conv_id = f"{row['design_id']}-r{row.get('replicate', '0')}"
        sim_cfg = self.cfg["simulation"]
        conv, sim = simulate(self.flow, self.hrt, row, conv_id, seed, max_turns=sim_cfg["max_turns"],
                             include_handoff_turns=sim_cfg["include_handoff_turns"])
        script = build_script(sim, conv, row, seed, sim_cfg.get("side_negatives_per_conversation", 2))
        profile = conv.profile
        style = [row.get(k, "") for k in STYLE_FACTORS]
        history: list[tuple[str, str]] = [("bot", OPENING)] if conv.aop == "User Identification" else \
            cv_prefix(profile, row.get("cv_prefix", "short"))
        turns_out = []
        n = len(script.turns)
        for i, st in enumerate(script.turns):
            summary = self.renderer.summary(history)
            cust = self.renderer.customer(st.customer, profile, history, style)
            if self.mask:
                cust = mask_digits(cust)
            facts = facts_for(st.gold, profile)
            responses = []
            for kind, spec in [("dialogue", st.bot)] + [("counterfactual" if v.kind == "gold" else "side", v)
                                                         for v in st.variants]:
                text, accepted = self._respond(spec, facts, history, cust)
                responses.append({"variant": kind, "text": text, "label": spec.label, "spec": _spec_meta(spec),
                                  "accepted": accepted})
            history += [("customer", cust), ("bot", responses[0]["text"])]
            turns_out.append({"turn": st.index, "seq": i, "phase": st.phase, "position": _position(i, n),
                              "summary": summary, "customer": cust, "customer_intent": _jsonable(st.customer),
                              "gold_state": st.gold.state, "context": st.gold.context, "responses": responses})
        return {"conv_id": conv_id, "aop": conv.aop, "row": row, "end": conv.end, "error": _jsonable(script.error),
                "n_turns": n, "length_band": _length_band(n), "profile": {k: v for k, v in profile.items()
                                                                           if k not in ("ssn9", "ssn4", "cm15", "otp")},
                "turns": turns_out}

    def _respond(self, spec: BotSpec, facts, history, cust) -> tuple[str, bool]:
        boxes = [(b, self.hrt.box_text(b)) for b in spec.boxes if b != HANDOFF_STATE]
        text, ok = "", False
        for _ in range(int(self.cfg["text"].get("max_tries", 3))):
            text = self.renderer.bot(spec, facts, history, cust, boxes)
            ok = bool(self.renderer.check(spec, facts, text).get("accepted", True))
            if ok:
                break
        return text, ok


def _spec_meta(s: BotSpec) -> dict[str, Any]:
    return {"kind": s.kind, "state": s.state, "gold_state": s.gold_state, "family": s.family, "operator": s.operator,
            "violations": s.violations, "instruction": s.instruction}


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


# ------------------------------------------------------------------------------------------------------- rows
ROW_FIELDS = ["row_id", "conv_id", "design_id", "group_id", "aop", "turn", "seq", "phase", "variant", "position",
              "n_turns", "length_band", "summary", "customer", "response", "label", "gold_state", "response_state",
              "error_family", "error_family_row", "error_operator", "violations", "accepted", "conv_error_family"]


def rows_of(conv: dict[str, Any], factor_names: list[str]) -> list[dict[str, Any]]:
    out = []
    eff = conv["error"].get("family_effective", "none")
    for t in conv["turns"]:
        for j, r in enumerate(t["responses"]):
            sp = r["spec"]
            out.append({
                "row_id": f"{conv['conv_id']}:{t['seq']}:{j}", "conv_id": conv["conv_id"],
                "design_id": conv["row"]["design_id"], "group_id": f"{conv['conv_id']}:{t['seq']}",
                "aop": conv["row"]["aop"], "turn": t["turn"], "seq": t["seq"], "phase": t["phase"],
                "variant": r["variant"], "position": t["position"], "n_turns": conv["n_turns"],
                "length_band": conv["length_band"], "summary": t["summary"], "customer": t["customer"],
                "response": r["text"], "label": int(r["label"]), "gold_state": t["gold_state"],
                "response_state": sp["state"], "error_family": sp["family"] if r["label"] == 0 else "none",
                "error_family_row": sp["family"], "error_operator": sp["operator"],
                "violations": "|".join(sp["violations"]), "accepted": int(bool(r["accepted"])),
                "conv_error_family": eff,
                **{f"f_{k}": conv["row"].get(k, "n/a") for k in factor_names}})
    return out


def out_of_place(rows: list[dict[str, Any]], per_conv: int, seed: int) -> list[dict[str, Any]]:
    """A correct response from another state, placed in this context: right words, wrong moment (label 0)."""
    rng = random.Random(seed)
    gold = [r for r in rows if r["label"] == 1 and r["variant"] == "dialogue" and r["gold_state"] != HANDOFF_STATE]
    by_aop: dict[str, list[dict[str, Any]]] = {}
    for r in gold:
        by_aop.setdefault(r["aop"], []).append(r)
    out = []
    by_conv: dict[str, list[dict[str, Any]]] = {}
    for r in gold:
        by_conv.setdefault(r["conv_id"], []).append(r)
    for cid, rs in by_conv.items():
        for r in rng.sample(rs, min(per_conv, len(rs))):
            pool = [x for x in by_aop[r["aop"]] if x["gold_state"] != r["gold_state"] and x["conv_id"] != cid]
            if not pool:
                continue
            donor = rng.choice(pool)
            out.append({**r, "row_id": r["row_id"] + ":oop", "variant": "out_of_place", "response": donor["response"],
                        "label": 0, "response_state": donor["gold_state"], "error_family": "out_of_place",
                        "error_family_row": "out_of_place", "error_operator": f"from {donor['gold_state']}",
                        "violations": f"transition {r['gold_state']} -> {donor['gold_state']}"})
    return out


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    cols = ROW_FIELDS + sorted({k for r in rows for k in r if k.startswith("f_")})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(cfg: dict[str, Any], limit: int = 0, out_dir: str | Path | None = None, do_split: bool = True,
        log=print) -> dict[str, Any]:
    """The whole dataset step: design -> conversations -> rows -> (split). Returns the output paths and the report."""
    out = Path(out_dir or cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    gen = Generator(cfg)
    design, rep = gen.factors.design(cfg["design"]["seed"], cfg["design"]["replicates"], cfg["design"]["extra_random"],
                                     log=log)
    write_csv(design, out / "design.csv")
    (out / "design.report.json").write_text(json.dumps({**rep, "coverage": gen.factors.coverage(design)}, indent=1))
    if limit:
        design = random.Random(1).sample(design, min(limit, len(design)))
    t0 = time.time()
    convs, failures = [], []
    workers = int(cfg["text"].get("workers", 8)) if cfg["text"]["renderer"] == "llm" else 1
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(gen.conversation, row, i): row for i, row in enumerate(design)}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            row = futs[fut]
            try:
                convs.append(fut.result())
            except Exception as e:                      # noqa: BLE001 - one bad conversation does not stop the run
                failures.append({"design_id": row["design_id"], "replicate": row.get("replicate"),
                                 "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)})
            if n % 25 == 0 or n == len(futs):
                log(f"[generate] {n}/{len(futs)} conversations ({len(failures)} failed) {time.time() - t0:.0f}s")
    convs.sort(key=lambda c: c["conv_id"])
    with open(out / "conversations.jsonl", "w") as fh:
        for c in convs:
            fh.write(json.dumps(c, default=str) + "\n")
    rows = [r for c in convs for r in rows_of(c, gen.factors.names)]
    rows += out_of_place(rows, int(cfg["simulation"].get("out_of_place_per_conversation", 1)), cfg["design"]["seed"])
    if not rows:
        raise RuntimeError(f"no rows were generated ({len(failures)} conversations failed; first: "
                           f"{failures[0]['error'] if failures else '-'})")
    write_rows(rows, out / "rows.csv")
    report = {"conversations": len(convs), "failed": len(failures), "rows": len(rows),
              "positives": sum(r["label"] for r in rows), "negatives": sum(1 - r["label"] for r in rows),
              "rejected_by_quality_gate": sum(1 for r in rows if not int(r["accepted"])),
              "unknown_branch_conditions": sorted(gen.hrt.unknown_conditions), "failures": failures[:50],
              "seconds": round(time.time() - t0, 1), "renderer": cfg["text"]["renderer"],
              "backend": cfg["text"].get("backend")}
    (out / "generation_report.json").write_text(json.dumps(report, indent=1))
    log(f"[generate] {len(convs)} conversations, {len(rows)} rows -> {out}")
    result = {"out_dir": str(out), "rows": str(out / "rows.csv"), "conversations": str(out / "conversations.jsonl"),
              "report": report}
    if do_split:
        from .split import split_files
        result["split"] = split_files(out / "rows.csv", out, cfg["split"], gen.factors.names)
        result.update({s: str(out / f"{s}.csv") for s in cfg["split"]["ratios"]})
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turnval.generate")
    ap.add_argument("--config", default=str(ROOT / "turnval/config/default.yaml"))
    ap.add_argument("--renderer", choices=["llm", "template"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N conversations (pilot)")
    ap.add_argument("--out")
    ap.add_argument("--no-split", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.renderer:
        cfg["text"]["renderer"] = args.renderer
    run(cfg, args.limit, args.out, not args.no_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
