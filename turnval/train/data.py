"""Block 8a - rows to model inputs, and batches that keep contrastive groups together."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

PROMPT = ("You check one turn of an American Express Account Protection call against its operating procedure.\n"
          "### Call so far (summary)\n{summary}\n### Customer\n{customer}\n### Agent response\n{response}\n"
          "### Is the agent response admissible? Answer yes or no.\nAnswer:")


def load_rows(path: str, drop_rejected: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    df["label"] = df["label"].astype(int)
    if drop_rejected and "accepted" in df.columns:
        df = df[df["accepted"].astype(str) != "0"]
    return df.reset_index(drop=True)


def class_ids(df: pd.DataFrame, mode: str) -> list[int]:
    """Classes for the supervised contrastive loss: admissible vs not, or admissible vs each error family."""
    if mode == "label":
        return df["label"].tolist()
    keys = ["OK" if l == 1 else f"ERR:{f}" for l, f in zip(df["label"], df["error_family"])]
    vocab = {k: i for i, k in enumerate(sorted(set(keys)))}
    return [vocab[k] for k in keys]


@dataclass
class Encoded:
    input_ids: list[int]
    label: int
    group: str
    cls: int
    idx: int


def encode(df: pd.DataFrame, tokenizer, max_length: int, supcon_label: str = "label") -> list[Encoded]:
    """Tokenise each row; when too long, the summary is cut from its start (the latest steps matter most)."""
    out = []
    classes = class_ids(df, supcon_label)
    for i, r in enumerate(df.itertuples(index=False)):
        tail = PROMPT.format(summary="", customer=r.customer, response=r.response)
        t_ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
        s_ids = tokenizer(r.summary, add_special_tokens=False)["input_ids"]
        room = max(0, max_length - len(t_ids) - 4)
        summary = r.summary if len(s_ids) <= room else tokenizer.decode(s_ids[-room:]) if room else ""
        ids = tokenizer(PROMPT.format(summary=summary, customer=r.customer, response=r.response),
                        add_special_tokens=True)["input_ids"][-max_length:]
        out.append(Encoded(ids, int(r.label), r.group_id, classes[i], i))
    return out


class GroupBatches:
    """Batches of about ``batch_rows`` rows made of whole context groups (so a correct response and its wrong
    alternatives for the same context are always compared in the same step)."""

    def __init__(self, items: list[Encoded], batch_rows: int, shuffle: bool = True, seed: int = 0):
        self.batch_rows, self.shuffle = batch_rows, shuffle
        self.rng = random.Random(seed)
        groups: dict[str, list[int]] = {}
        for k, it in enumerate(items):
            groups.setdefault(it.group, []).append(k)
        self.groups = list(groups.values())

    def __iter__(self):
        gs = list(self.groups)
        if self.shuffle:
            self.rng.shuffle(gs)
        batch: list[int] = []
        for g in gs:
            if batch and len(batch) + len(g) > self.batch_rows:
                yield batch
                batch = []
            batch.extend(g)
        if batch:
            yield batch

    def __len__(self):
        n, cur = 0, 0
        for g in self.groups:
            if cur and cur + len(g) > self.batch_rows:
                n, cur = n + 1, 0
            cur += len(g)
        return n + (1 if cur else 0)


def collate(items: list[Encoded], pad_id: int) -> dict[str, Any]:
    import torch
    L = max(len(x.input_ids) for x in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for i, x in enumerate(items):                      # left padding: the last position is the last real token
        ids[i, L - len(x.input_ids):] = torch.tensor(x.input_ids)
        att[i, L - len(x.input_ids):] = 1
    groups = {g: k for k, g in enumerate(dict.fromkeys(x.group for x in items))}
    return {"input_ids": ids, "attention_mask": att,
            "labels": torch.tensor([x.label for x in items], dtype=torch.float),
            "groups": torch.tensor([groups[x.group] for x in items]),
            "classes": torch.tensor([x.cls for x in items]), "idx": [x.idx for x in items]}
