"""Block 8d - train, select, calibrate, report.

  python -m turnval.train.trainer --config turnval/train/config.yaml [--set train.epochs=1 model.backbone=tiny-random]

1. load train / val / test (rows.csv format), tokenise (summary trimmed from the front to fit max_length);
2. LoRA on the backbone + score head + projection; AdamW (LoRA lr, head lr), warm-up then cosine decay;
3. each step: a batch of whole context groups; loss = bce * BCE + pairwise * margin loss + supcon * SupCon;
4. evaluate on val every epoch (or every N steps), keep the best by ``select_metric``, stop early on no gain;
5. fit a temperature on val (calibration), pick thresholds on val (0.5, best F1, precision targets);
6. score val / test / real_test, write predictions and reports (Block 9).

Outputs in train.out_dir: adapter/ (LoRA + heads), calibration.json, thresholds.json, *_predictions.csv,
*_report.{json,md}, *_report_categories.csv, train_log.jsonl, summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _set(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def _path(p: str) -> str:
    return p if not p or Path(p).is_absolute() else str(ROOT / p)


class Trainer:
    def __init__(self, cfg: dict[str, Any]):
        import torch
        from .data import encode, load_rows
        from .model import TurnValidator, load_backbone
        self.cfg, self.torch = cfg, torch
        tc = cfg["train"]
        random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"])
        d = cfg["data"]
        self.frames = {k: load_rows(_path(d[k]), d.get("drop_rejected", True)) for k in ("train", "val", "test")}
        if d.get("real_test"):
            self.frames["real_test"] = load_rows(_path(d["real_test"]), False)
        texts = (self.frames["train"]["summary"] + " " + self.frames["train"]["customer"] + " " +
                 self.frames["train"]["response"]).tolist()
        backbone, tok = load_backbone(cfg["model"], texts)
        self.model = TurnValidator(backbone, tok, cfg["model"])
        self.device = self.model.lm.get_input_embeddings().weight.device
        self.model.head.to(self.device); self.model.proj.to(self.device)
        sl = cfg["loss"].get("supcon_label", "label")
        self.items = {k: encode(f, tok, d["max_length"], sl) for k, f in self.frames.items()}
        self.pad = tok.pad_token_id
        y = self.frames["train"]["label"]
        pw = cfg["loss"].get("pos_weight", "auto")
        self.pos_weight = float((y == 0).sum() / max(1, (y == 1).sum())) if pw == "auto" else (float(pw) if pw else None)
        self.out = Path(_path(tc["out_dir"]))
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = open(self.out / "train_log.jsonl", "a")

    # ------------------------------------------------------------------ scoring
    def scores(self, split: str, batch_rows: int | None = None) -> np.ndarray:
        from .data import collate
        torch = self.torch
        self.model.eval()
        items = self.items[split]
        out = np.zeros(len(items))
        bs = batch_rows or self.cfg["train"]["batch_rows"]
        with torch.no_grad():
            for i in range(0, len(items), bs):
                chunk = items[i:i + bs]
                b = collate(chunk, self.pad)
                with self._autocast():
                    s = self.model(b["input_ids"].to(self.device), b["attention_mask"].to(self.device))["score"]
                out[i:i + len(chunk)] = s.float().cpu().numpy()
        self.model.train()
        return out

    def _autocast(self):
        torch = self.torch
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def _val_metrics(self) -> dict[str, Any]:
        from ..evaluate import binary
        s = self.scores("val")
        y = self.frames["val"]["label"].to_numpy()
        p = 1 / (1 + np.exp(-s))
        m = binary(y, p, 0.5)
        return {"pr_auc": m["pr_auc_error"] or 0.0, "roc_auc": m["roc_auc"] or 0.0, "error_f1": m["error_f1"],
                "accuracy": m["accuracy"]}

    # ------------------------------------------------------------------ training
    def fit(self) -> dict[str, Any]:
        from .data import GroupBatches, collate
        from .losses import total
        torch = self.torch
        tc, lc = self.cfg["train"], self.cfg["loss"]
        lora, head = self.model.trainable()
        opt = torch.optim.AdamW([{"params": lora, "lr": tc["lr"]}, {"params": head, "lr": tc["head_lr"]}],
                                weight_decay=tc["weight_decay"])
        batches = GroupBatches(self.items["train"], tc["batch_rows"], True, tc["seed"])
        steps = max(1, len(batches) * tc["epochs"] // tc["grad_accum"])
        warm = int(steps * tc["warmup_ratio"])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
        best, best_metric, bad, step, t0 = None, -1.0, 0, 0, time.time()
        every = tc.get("eval_every_steps") or 0
        self.model.train()
        for ep in range(tc["epochs"]):
            for bi, idx in enumerate(batches):
                b = collate([self.items["train"][i] for i in idx], self.pad)
                with self._autocast():
                    out = self.model(b["input_ids"].to(self.device), b["attention_mask"].to(self.device))
                loss, parts = total(out, {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in b.items()},
                                    lc, self.pos_weight)
                (loss / tc["grad_accum"]).backward()
                if (bi + 1) % tc["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(lora + head, tc["max_grad_norm"])
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 10 == 0:
                        self._log({"epoch": ep, "step": step, "loss": float(loss.detach()), **parts,
                                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)})
                    if every and step % every == 0:
                        best, best_metric, bad = self._checkpoint(ep, step, best, best_metric, bad)
            if not every:
                best, best_metric, bad = self._checkpoint(ep, step, best, best_metric, bad)
            if bad >= tc["early_stopping_patience"]:
                self._log({"early_stop": True, "epoch": ep})
                break
        return {"best_val": best, "steps": step, "seconds": round(time.time() - t0, 1)}

    def _checkpoint(self, ep, step, best, best_metric, bad):
        m = self._val_metrics()
        key = self.cfg["train"].get("select_metric", "pr_auc")
        self._log({"eval": "val", "epoch": ep, "step": step, **m})
        if m[key] > best_metric:
            self.model.save(str(self.out / "adapter"))
            return m, m[key], 0
        return best, best_metric, bad + 1

    def _log(self, d: dict[str, Any]) -> None:
        self.log.write(json.dumps(d) + "\n"); self.log.flush()
        print("[train]", json.dumps(d))

    # ------------------------------------------------------------------ calibrate and report
    def finish(self) -> dict[str, Any]:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from ..evaluate import best_f1_threshold, full_report, target_thresholds, write
        ad = self.out / "adapter"
        if (ad / "adapter_model.safetensors").exists():
            set_peft_model_state_dict(self.model.lm, load_file(str(ad / "adapter_model.safetensors")))
            self.model.load_heads(str(ad))
        s_val = self.scores("val")
        y_val = self.frames["val"]["label"].to_numpy()
        T = fit_temperature(s_val, y_val)
        target = self.cfg["train"].get("precision_target", 0.95)
        p_val = 1 / (1 + np.exp(-s_val / T))
        tt = target_thresholds(y_val, p_val, target)
        thr = {"default_0.5": 0.5, "best_f1": best_f1_threshold(y_val, p_val)}
        if "flag" in tt:
            thr[f"flag_precision_{target}"] = tt["flag"]["threshold"]
        if "pass" in tt:
            thr[f"pass_precision_{target}"] = tt["pass"]["threshold"]
        (self.out / "calibration.json").write_text(json.dumps({"temperature": T}))
        (self.out / "thresholds.json").write_text(json.dumps(thr, indent=1))
        summary: dict[str, Any] = {"temperature": T, "thresholds": thr}
        for split in self.frames:
            s = s_val if split == "val" else self.scores(split)
            df = self.frames[split].copy()
            df["score"] = s
            df["p"] = 1 / (1 + np.exp(-s / T))
            df.to_csv(self.out / f"{split}_predictions.csv", index=False)
            rep = full_report(df, thr, target)
            write(rep, self.out / f"{split}_report", f"Turn validator - {split}")
            summary[split] = rep["overall@best_f1"]
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary


def fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: the T minimising the val negative log-likelihood of sigmoid(score / T)."""
    best, bt = float("inf"), 1.0
    for T in np.exp(np.linspace(np.log(0.05), np.log(20), 200)):
        p = np.clip(1 / (1 + np.exp(-scores / T)), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best:
            best, bt = nll, float(T)
    return bt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turnval.train.trainer")
    ap.add_argument("--config", default=str(ROOT / "turnval/train/config.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=1 model.backbone=tiny-random")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        _set(cfg, k, v)
    tr = Trainer(cfg)
    fit = tr.fit()
    summary = tr.finish()
    summary["fit"] = fit
    (tr.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("thresholds", "fit")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
