"""Block 8c - the losses.

  BCE        every row on its own: admissible (1) or not (0)
  pairwise   inside each context group (same summary + customer turn), every correct response must score at least
             ``margin`` above every wrong one: mean softplus(margin - (s_pos - s_neg))
  supcon     supervised contrastive (Khosla et al., 2020) on the projections: rows of the same class pull together,
             others push apart (class = admissible, or admissible / each error family)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce(score: torch.Tensor, y: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    pw = torch.tensor(pos_weight, device=score.device) if pos_weight else None
    return F.binary_cross_entropy_with_logits(score, y, pos_weight=pw)


def pairwise(score: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    same = groups[:, None] == groups[None, :]
    pos_neg = (y[:, None] == 1) & (y[None, :] == 0) & same
    if not pos_neg.any():
        return score.sum() * 0.0
    diff = score[:, None] - score[None, :]
    return F.softplus(margin - diff[pos_neg]).mean()


def supcon(z: torch.Tensor, classes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    n = z.size(0)
    if n < 2:
        return z.sum() * 0.0
    sim = z @ z.T / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (classes[:, None] == classes[None, :]) & ~eye
    has = pos.any(dim=1)
    if not has.any():
        return z.sum() * 0.0
    return -(logp.masked_fill(~pos, 0).sum(1)[has] / pos.sum(1)[has]).mean()


def total(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict, pos_weight: float | None):
    y, g, c = batch["labels"], batch["groups"], batch["classes"]
    parts = {"bce": bce(out["score"], y, pos_weight),
             "pairwise": pairwise(out["score"], y, g, cfg.get("pairwise_margin", 1.0)),
             "supcon": supcon(out["z"], c, cfg.get("supcon_temperature", 0.1))}
    loss = sum(cfg.get(k, 0.0) * v for k, v in parts.items())
    return loss, {k: float(v.detach()) for k, v in parts.items()}
