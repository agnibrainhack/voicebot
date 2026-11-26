"""turnval as Python calls, one per step, for the Jupyter notebook (notebooks/turnval_end_to_end.ipynb).

    from turnval import api
    cfg = api.config(vllm_url="http://gpu-host:8000/v1", vllm_model="openai/gpt-oss-20b")
    api.check_data(cfg); api.check_llm(cfg)
    api.audit(cfg); design, rep = api.design(cfg)
    pilot = api.generate(cfg, limit=20, out_dir="turnval/out/pilot"); api.show_conversation(pilot, 0)
    full = api.generate(cfg)
    plan = api.gpu_plan(num_gpus=1, gpu_name="H100 80GB")
    run = api.train(full, plan)
    api.show_report(run, "test")
"""
from __future__ import annotations

import copy
import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from .generate import ROOT, load_config

DEFAULT_CONFIG = ROOT / "turnval/config/default.yaml"
TRAIN_CONFIG = ROOT / "turnval/train/config.yaml"


def _set(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# ---------------------------------------------------------------------------------------------------- configuration
def config(path: str | Path = DEFAULT_CONFIG, *, backend: str = "local", vllm_url: str = "", vllm_model: str = "",
           gateway_model: str = "", renderer: str = "llm", out_dir: str | None = None, replicates: int | None = None,
           workers: int | None = None, verify: bool | None = None, cache_dir: str | None = None,
           **overrides: Any) -> dict[str, Any]:
    """The generation config with the notebook's inputs applied.
    backend "local": gpt-oss on vLLM at ``vllm_url`` (its /v1 base) serving ``vllm_model``;
    backend "gateway": the Microsoft GPT gateway, deployment ``gateway_model`` (e.g. gpt-41).
    ``overrides``: dotted keys, e.g. **{"simulation.side_negatives_per_conversation": 3}."""
    cfg = load_config(path)
    t = cfg["text"]
    t["renderer"], t["backend"] = renderer, backend
    if backend == "local":
        if renderer == "llm" and not vllm_url:
            raise ValueError("backend 'local' needs vllm_url (the server's /v1 base, e.g. http://host:8000/v1)")
        t["vllm_url"] = vllm_url.rstrip("/")
        t["local_path"] = vllm_model or t.get("local_path") or "openai/gpt-oss-20b"
    elif backend == "gateway":
        if renderer == "llm" and not gateway_model:
            raise ValueError("backend 'gateway' needs gateway_model (the deployment name, e.g. gpt-41)")
        t["model"] = gateway_model
    else:
        raise ValueError("backend is 'local' (vLLM) or 'gateway' (Microsoft GPT)")
    if out_dir:
        p = Path(out_dir)
        cfg["output"]["dir"] = str(p if p.is_absolute() else ROOT / p)
    if replicates is not None:
        cfg["design"]["replicates"] = replicates
    if workers is not None:
        t["workers"] = workers
    if verify is not None:
        t["verify"] = verify
    if cache_dir:
        t["cache_dir"] = cache_dir
    for k, v in overrides.items():
        _set(cfg, k, v)
    return cfg


def check_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """The AOP flow and the HRT graph load, and the walker replays the reference call."""
    from .graph import DictTurn, Flow, HRTGraph, Walker
    flow, hrt = Flow(cfg["data"]["aop_flow"]), HRTGraph(cfg["data"]["hrt_graph"])
    w = Walker(flow, "User Identification", {"checkpoint": "check4SSN", "channel_type": "voice", "journey": "Fraud"})
    states = [w.run(DictTurn({"set_4ssn_metadata": {"match": "NO_USER_RESPONSE"}})).state,
              w.run(DictTurn({"set_4ssn_metadata": {"match": "PROFILE_SINGLE_MATCH"}})).state]
    out = {"sections": len(flow.sections), "speaking_boxes": len(hrt.boxes), "replay": states,
           "replay_ok": states == ["UI_CLAUSE_108", "UI_CLAUSE_119"]}
    if not out["replay_ok"]:
        raise RuntimeError(f"the flow does not replay the reference call: {states}")
    return out


def check_llm(cfg: dict[str, Any]) -> dict[str, Any]:
    """The text model answers: the vLLM server lists the model and returns JSON, or the gateway answers."""
    t = cfg["text"]
    out: dict[str, Any] = {"backend": t["backend"]}
    if t["renderer"] != "llm":
        return {**out, "ok": True, "detail": "template renderer: no model needed"}
    if t["backend"] == "local":
        with urllib.request.urlopen(f"{t['vllm_url']}/models", timeout=20) as r:
            models = [m["id"] for m in json.loads(r.read()).get("data", [])]
        out["served_models"] = models
        if t["local_path"] not in models:
            raise RuntimeError(f"the server at {t['vllm_url']} serves {models}, not '{t['local_path']}' (set vllm_model)")
    from .textgen import SUMMARY_SYSTEM, LLMRenderer
    r = LLMRenderer(t)
    txt = r._ask(r.summ, SUMMARY_SYSTEM, "CALL SO FAR:\nAgent: Hello, are you calling about a fraud alert?\nCustomer: Yes.",
                 "summary", "health")
    return {**out, "ok": True, "sample": txt[:120]}


# ---------------------------------------------------------------------------------------------------- design + data
def audit(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flow outcomes no factor level drives (empty = every decision is covered)."""
    from .factors import Factors
    from .graph import Flow
    return Factors.load(cfg["data"]["factors"]).audit(Flow(cfg["data"]["aop_flow"]))["not_driven_by_a_factor"]


def design(cfg: dict[str, Any]):
    """The covering array as a DataFrame, and its report."""
    import pandas as pd
    from .factors import Factors
    fx = Factors.load(cfg["data"]["factors"])
    rows, rep = fx.design(cfg["design"]["seed"], cfg["design"]["replicates"], cfg["design"]["extra_random"])
    return pd.DataFrame(rows), rep


def generate(cfg: dict[str, Any], limit: int = 0, out_dir: str | Path | None = None, split: bool = True) -> dict[str, Any]:
    """Conversations -> rows.csv (-> train / val / test.csv). ``limit``: a pilot of N conversations."""
    from .generate import run
    if out_dir:
        p = Path(out_dir)
        out_dir = p if p.is_absolute() else ROOT / p
    return run(cfg, limit, out_dir, split)


def conversations(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(l) for l in open(result["conversations"])]


def show_conversation(result: dict[str, Any], i: int = 0, variants: bool = True) -> str:
    """One generated conversation as text: each turn's customer, bot reply, label and error."""
    c = conversations(result)[i]
    lines = [f"{c['conv_id']}  {c['aop']}  error: {c['error'].get('family_effective')} at turn "
             f"{c['error'].get('turn')}  ({c['n_turns']} turns, ended: {c['end']})"]
    for t in c["turns"]:
        lines.append(f"\n[{t['seq']}] {t['phase']}  state {t['gold_state'][:70]}")
        lines.append(f"  summary : {t['summary'][:160]}")
        lines.append(f"  customer: {t['customer']}")
        for r in t["responses"] if variants else t["responses"][:1]:
            sp = r["spec"]
            tag = "" if r["variant"] == "dialogue" else f" ({r['variant']})"
            err = "" if r["label"] else f"  <- {sp['family']}: {sp['operator']}"
            lines.append(f"  bot{tag} [{r['label']}]: {r['text'][:220]}{err}")
    text = "\n".join(lines)
    print(text)
    return text


def split_report(result: dict[str, Any]):
    """Per factor level: share of conversations in train / val / test (and the largest deviation)."""
    import pandas as pd
    rep = result.get("split") or json.loads((Path(result["out_dir"]) / "split_report.json").read_text())
    rows = [{"factor": f, "level": lv, **v} for f, lvs in rep["factors"].items() for lv, v in lvs.items()]
    return pd.DataFrame(rows).sort_values("max_dev", ascending=False), rep


# ---------------------------------------------------------------------------------------------------- training
def gpu_plan(num_gpus: int, gpu_name: str, memory_gb: float | None = None, max_length: int = 1536) -> dict[str, Any]:
    """Restrict the kernel to ``num_gpus`` GPUs and size the gpt-oss-20b training to them (see train/hardware.py)."""
    from .train.hardware import plan, select_gpus
    select_gpus(num_gpus)
    return plan(num_gpus, gpu_name, memory_gb, max_length=max_length)


def train_config(data: dict[str, Any], plan: dict[str, Any] | None = None, path: str | Path = TRAIN_CONFIG,
                 out_dir: str | None = None, **overrides: Any) -> dict[str, Any]:
    """The training config: the dataset paths from ``generate``, the GPU plan, then any dotted overrides."""
    cfg = yaml.safe_load(Path(path).read_text())
    for s in ("train", "val", "test"):
        if s in data:
            cfg["data"][s] = data[s]
    for k, v in ((plan or {}).get("overrides") or {}).items():
        _set(cfg, k, v)
    if out_dir:
        cfg["train"]["out_dir"] = out_dir
    for k, v in overrides.items():
        _set(cfg, k, v)
    return cfg


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit, select, calibrate and report (train/trainer.py). Returns the summary and the run folder."""
    from .train.trainer import Trainer
    tr = Trainer(copy.deepcopy(cfg))
    fit = tr.fit()
    summary = tr.finish()
    summary["fit"] = fit
    (tr.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return {"run_dir": str(tr.out), "summary": summary}


def show_report(run: dict[str, Any], split: str = "test") -> str:
    p = Path(run["run_dir"]) / f"{split}_report.md"
    text = p.read_text()
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        print(text)
    return text


def categories(run: dict[str, Any], split: str = "test"):
    import pandas as pd
    return pd.read_csv(Path(run["run_dir"]) / f"{split}_report_categories.csv")


# ---------------------------------------------------------------------------------------------------- real logs
def logs_to_label(inputs: list[str], out_csv: str, cfg: dict[str, Any] | None = None) -> str:
    """Real bot logs (OTel JSON or the span table) -> rows to hand-label; summaries by the same summariser."""
    from .logs import main
    args = ["--in", *inputs, "--out", out_csv]
    if cfg is not None and cfg["text"]["renderer"] == "llm":
        tmp = Path(out_csv).with_suffix(".config.yaml")
        tmp.write_text(yaml.safe_dump({**cfg, "data": cfg["data"]}))
        args += ["--renderer", "llm", "--config", str(tmp)]
    main(args)
    return out_csv


def predict(run: dict[str, Any], csv_path: str, out_csv: str | None = None):
    """Score any rows file (summary, customer, response[, label]) with the trained validator: adds p = P(admissible)
    and, with the val thresholds, the decision at each. Returns the DataFrame (and writes ``out_csv``)."""
    from .train.trainer import Predictor
    pr = Predictor(run["run_dir"])
    df = pr.score_csv(csv_path)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def evaluate_labelled(run: dict[str, Any], labelled_csv: str, name: str = "real_test") -> dict[str, Any]:
    """Hand-labelled real turns (label 1 / 0): scored, then the full report in the run folder (<name>_report.*)."""
    from .evaluate import full_report, write
    df = predict(run, labelled_csv)
    df = df[df["label"].astype(str).isin(["0", "1"])].copy()
    df["label"] = df["label"].astype(int)
    thr = json.loads((Path(run["run_dir"]) / "thresholds.json").read_text())
    tc = json.loads((Path(run["run_dir"]) / "train_config.json").read_text())
    rep = full_report(df, thr, tc["train"].get("precision_target", 0.95))
    write(rep, Path(run["run_dir"]) / f"{name}_report", f"Turn validator - {name}")
    df.to_csv(Path(run["run_dir"]) / f"{name}_predictions.csv", index=False)
    return rep
