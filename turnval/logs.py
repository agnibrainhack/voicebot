"""Block 9b - real bot logs -> rows for hand labelling (the real test set).

Reads either export the team has shared:
  * OpenTelemetry JSON (resourceSpans -> spans: invoke_agent with gen_ai.input/output.messages, tool spans with
    gen_ai.tool.name / gen_ai.tool.call.result, aop_selection) - a truncated file is read span by span;
  * the tab-separated span table (Turn ID, Category, Span Name, Full Input Ref, Full Output Ref, ...).

Writes one row per agent turn, in the training rows' format (summary, customer, response, aop, ...) with an empty
``label`` for the annotator, plus what the log says about the turn (AOP selected, tools and results) and
``overlap`` = 1 when the turn started before the previous one ended (barge-in: check the turn boundary).

  python -m turnval.logs --in log1.json log2.txt --out turnval/out/real_to_label.csv [--renderer template|llm]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

csv.field_size_limit(10 ** 9)


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for a in span.get("attributes", []):
        v = a.get("value", {})
        out[a["key"]] = next(iter(v.values()), None) if v else None
    return out


def _content(msgs: str | None) -> str:
    if not msgs:
        return ""
    try:
        m = json.loads(msgs)
        return " ".join(p.get("content", "") for x in m for p in x.get("parts", [])).strip()
    except (ValueError, TypeError, AttributeError):
        return str(msgs)


def _spans_from_otel(text: str) -> list[dict[str, Any]]:
    try:
        doc = json.loads(text)
        return [s for rs in doc.get("resourceSpans", []) for ss in rs.get("scopeSpans", []) for s in ss.get("spans", [])]
    except ValueError:
        dec, spans = json.JSONDecoder(), []
        for m in re.finditer(r'\{\s*"traceId"', text):
            try:
                obj, _ = dec.raw_decode(text, m.start())
                spans.append(obj)
            except ValueError:
                continue                                  # the last, cut-off span
        return spans


def read_otel(path: str) -> list[dict[str, Any]]:
    spans = _spans_from_otel(Path(path).read_text(encoding="utf-8", errors="ignore"))
    kids: dict[str, list[dict[str, Any]]] = {}
    for s in spans:
        if s.get("parentSpanId"):
            kids.setdefault(s["parentSpanId"], []).append(s)
    turns = []
    for s in sorted(spans, key=lambda x: int(x.get("startTimeUnixNano", 0))):
        if not s.get("name", "").startswith("invoke_agent"):
            continue
        a = _attrs(s)
        tools, aop = [], ""
        for c in sorted(kids.get(s["spanId"], []), key=lambda x: int(x.get("startTimeUnixNano", 0))):
            ca = _attrs(c)
            if ca.get("gen_ai.tool.name"):
                tools.append({"tool": ca["gen_ai.tool.name"], "result": str(ca.get("gen_ai.tool.call.result", ""))[:300]})
            if c.get("name") == "aop_selection":
                aop = _content(ca.get("gen_ai.output.messages"))
        turns.append({"conv": a.get("gen_ai.conversation.id", Path(path).stem), "customer": _content(a.get("gen_ai.input.messages")),
                      "bot": _content(a.get("gen_ai.output.messages")), "aop": aop, "tools": tools,
                      "start": int(s.get("startTimeUnixNano", 0)), "end": int(s.get("endTimeUnixNano", 0))})
    return turns


def read_table(path: str) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="ignore"), delimiter="\t"))
    turns: dict[str, dict[str, Any]] = {}
    order = []
    for r in rows:
        t = r.get("Turn ID") or ""
        if t not in turns:
            turns[t] = {"conv": Path(path).stem, "customer": "", "bot": "", "aop": "", "tools": [], "start": 0, "end": 0}
            order.append(t)
        d = turns[t]
        cat = r.get("Category", "")
        if cat == "Agent Invocation":
            d["customer"], d["bot"] = r.get("Full Input Ref", ""), r.get("Full Output Ref", "")
            d["start_s"], d["end_s"] = r.get("Start UTC", ""), r.get("End UTC", "")
        elif cat == "AOP Selection":
            d["aop"] = r.get("Output Summary", "")
        elif cat == "Tool":
            d["tools"].append({"tool": r.get("Span Name", ""), "result": r.get("Output Summary", "")[:300]})
    out = [turns[t] for t in order if turns[t]["bot"] or turns[t]["customer"]]
    for d in out:
        d["start"], d["end"] = d.pop("start_s", ""), d.pop("end_s", "")
    return out


def to_rows(turns: list[dict[str, Any]], renderer) -> list[dict[str, Any]]:
    rows, hist, prev_end, conv = [], [], None, None
    for i, t in enumerate(turns):
        if t["conv"] != conv:
            hist, prev_end, conv = [], None, t["conv"]
        cust = "" if t["customer"] == "[inbound_call]" else t["customer"]
        overlap = int(prev_end is not None and t["start"] and t["start"] < prev_end)
        if cust or hist:
            rows.append({"row_id": f"{conv}:{len(rows)}", "conv_id": conv, "group_id": f"{conv}:{len(rows)}",
                         "turn": i, "summary": renderer.summary(hist), "customer": cust, "response": t["bot"],
                         "label": "", "aop": t["aop"], "overlap": overlap,
                         "tools": json.dumps(t["tools"])[:2000], "error_family": "", "notes": ""})
        if cust:
            hist.append(("customer", cust))
        hist.append(("bot", t["bot"]))
        prev_end = t["end"] or prev_end
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turnval.logs")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--renderer", default="template")
    ap.add_argument("--config", default="turnval/config/default.yaml")
    a = ap.parse_args(argv)
    from .textgen import TemplateRenderer
    if a.renderer == "llm":
        from .generate import load_config
        from .textgen import LLMRenderer
        renderer = LLMRenderer(load_config(a.config)["text"])
    else:
        renderer = TemplateRenderer()
    rows = []
    for p in a.inputs:
        head = Path(p).read_text(encoding="utf-8", errors="ignore")[:200]
        turns = read_otel(p) if head.lstrip().startswith("{") else read_table(p)
        rows += to_rows(turns, renderer)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["row_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"[logs] {len(rows)} turns to label -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
