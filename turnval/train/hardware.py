"""Training on the GPUs you have: number of GPUs + GPU name -> a training plan for gpt-oss-20b.

``plan(num_gpus, gpu_name)`` checks that the model fits and returns the overrides for train/config.yaml:
  model.device_map / model.max_memory   the layers spread over the GPUs (model parallel), never offloaded to CPU
  data.max_length, train.batch_rows, train.grad_accum   sized to the memory left after the weights
It raises ``ValueError`` with the reason when the GPUs cannot train the model (too little memory, no bf16).

Memory model (gpt-oss-20b, LoRA on attention, bf16, gradient checkpointing):
  weights 21B parameters x 2 bytes = ~42 GB (the MXFP4 checkpoint is dequantised for training)
  overhead ~3 GB per GPU (CUDA context, LoRA weights and optimiser states, buffers)
  activations ~1.0 GB per row of 1,536 tokens (the LM head is skipped: only the last hidden state is scored)
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

# name pattern -> (memory GB, bf16 support)
GPUS: list[tuple[str, int, bool]] = [
    (r"b200", 180, True), (r"h200", 141, True), (r"h100.*nvl", 94, True), (r"h100|h800", 80, True),
    (r"a100.*40|a100-40", 40, True), (r"a100|a800", 80, True), (r"l40s?\b|l40s", 48, True),
    (r"rtx\s*6000\s*ada|6000 ada", 48, True), (r"a6000|rtx a6000", 48, True), (r"a40\b", 48, True),
    (r"a10g", 24, True), (r"a10\b", 24, True), (r"l4\b", 24, True), (r"4090", 24, True), (r"3090", 24, True),
    (r"v100.*16", 16, False), (r"v100", 32, False), (r"t4\b", 16, False), (r"p100", 16, False),
]
MODEL_GB = {"gpt-oss-20b": 42.0, "tiny-random": 0.1}
OVERHEAD_GB = 3.0
ROW_GB_AT_1536 = 1.0


def gpu_spec(name: str, memory_gb: float | None = None) -> tuple[float, bool]:
    n = name.lower()
    for rx, mem, bf16 in GPUS:
        if re.search(rx, n):
            return float(memory_gb or mem), bf16
    if memory_gb:
        return float(memory_gb), True
    raise ValueError(f"unknown GPU '{name}': pass memory_gb=<GB per GPU> (and check it supports bf16)")


def detect() -> dict[str, Any]:
    """The GPUs this kernel actually sees (None when torch / CUDA is not available)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"count": 0, "names": []}
        return {"count": torch.cuda.device_count(),
                "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "memory_gb": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
                              for i in range(torch.cuda.device_count())]}
    except Exception:                                    # noqa: BLE001
        return {"count": None, "names": []}


def select_gpus(num_gpus: int) -> str:
    """Restrict the kernel to the first ``num_gpus`` GPUs. Call it before torch touches CUDA."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    return os.environ["CUDA_VISIBLE_DEVICES"]


def plan(num_gpus: int, gpu_name: str, memory_gb: float | None = None, model: str = "gpt-oss-20b",
         max_length: int = 1536, effective_rows: int = 32) -> dict[str, Any]:
    if num_gpus < 1:
        raise ValueError("training gpt-oss-20b needs at least one GPU")
    mem, bf16 = gpu_spec(gpu_name, memory_gb)
    if not bf16:
        raise ValueError(f"{gpu_name} has no bf16 support: gpt-oss-20b training needs Ampere or newer "
                         "(A100, A10G, L4, L40S, H100, ...)")
    weights = MODEL_GB.get(model, 42.0)
    usable = num_gpus * (mem * 0.92 - OVERHEAD_GB)
    free = usable - weights
    length = max_length
    row_gb = ROW_GB_AT_1536 * length / 1536
    while free < 4 * row_gb and length > 512:        # at least one context group (about 4 rows) per step
        length = int(length * 0.75)
        row_gb = ROW_GB_AT_1536 * length / 1536
    if free < 4 * row_gb:
        need = weights + 4 * row_gb + num_gpus * OVERHEAD_GB
        raise ValueError(f"{num_gpus} x {gpu_name} ({mem:.0f} GB each) is too small for {model}: about "
                         f"{need:.0f} GB are needed in total, {num_gpus * mem:.0f} GB are there. Use more GPUs or "
                         "larger ones (e.g. 1 x 80 GB, or 2 x 48 GB).")
    batch_rows = int(max(4, min(32, math.floor(free / row_gb))))
    grad_accum = max(1, round(effective_rows / batch_rows))
    per_gpu = f"{int(mem * 0.92 - OVERHEAD_GB)}GiB"
    seen = detect()
    notes = []
    if seen.get("count") not in (None, 0) and seen["count"] < num_gpus:
        notes.append(f"asked for {num_gpus} GPUs, this kernel sees {seen['count']}: {seen['names']}")
    if seen.get("names") and not any(re.search(gpu_name.lower().split()[-1], n.lower()) for n in seen["names"]):
        notes.append(f"the GPUs seen are {sorted(set(seen['names']))}, not '{gpu_name}'")
    if length < max_length:
        notes.append(f"max_length lowered from {max_length} to {length} tokens to fit the memory")
    if num_gpus > 1:
        notes.append("layers are spread over the GPUs (model parallel): more memory, not more speed")
    return {"num_gpus": num_gpus, "gpu": gpu_name, "memory_gb_per_gpu": mem, "model_gb": weights,
            "free_for_activations_gb": round(free, 1), "seen": seen, "notes": notes,
            "overrides": {"model.device_map": "auto",
                          "model.max_memory": {i: per_gpu for i in range(num_gpus)},
                          "model.dtype": "bfloat16", "model.gradient_checkpointing": True,
                          "data.max_length": length, "train.batch_rows": batch_rows, "train.grad_accum": grad_accum}}
