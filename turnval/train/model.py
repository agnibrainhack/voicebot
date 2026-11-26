"""Block 8b - the classifier: a decoder (gpt-oss-20b) with LoRA, a score head and a projection for contrastive loss.

score(x) = w . h_last(x) + b  (head: linear), or logit("yes") - logit("no") at the last position (head: lm_token);
z(x) = normalise(P h_last(x)) for the supervised contrastive loss. The prompt asks "Is the agent response
admissible?", so the last token's state carries the judgement.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def tiny_backbone(texts: list[str]):
    """A small random decoder and a word-level tokenizer built from ``texts`` (tests and dry runs; no download)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(vocab_size=4000, special_tokens=["[PAD]", "[UNK]", "[BOS]"]))
    t = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]")
    cfg = LlamaConfig(vocab_size=t.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg), t


def load_backbone(mcfg: dict[str, Any], texts: list[str] | None = None):
    if mcfg["backbone"] == "tiny-random":
        return tiny_backbone(texts or ["hello world"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw: dict[str, Any] = {"torch_dtype": getattr(torch, mcfg.get("dtype", "bfloat16")),
                          "device_map": mcfg.get("device_map", "auto")}
    if mcfg.get("max_memory"):                          # per-GPU caps (train/hardware.plan); no CPU offload
        kw["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in mcfg["max_memory"].items()}
    if mcfg.get("dequantize_mxfp4", True) and "gpt-oss" in mcfg["backbone"]:
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(mcfg["backbone"], **kw)
    tok = AutoTokenizer.from_pretrained(mcfg["backbone"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


class TurnValidator(nn.Module):
    def __init__(self, backbone, tokenizer, mcfg: dict[str, Any]):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        lc = mcfg.get("lora", {})
        kw = dict(r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32), lora_dropout=lc.get("dropout", 0.05),
                  target_modules=lc.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]), bias="none")
        if lc.get("target_parameters"):                 # MoE experts (fused parameters): peft needs lora_dropout 0
            kw["target_parameters"] = lc["target_parameters"]
            kw["lora_dropout"] = 0.0
        if mcfg.get("gradient_checkpointing") and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()
        self.lm = get_peft_model(backbone, LoraConfig(**kw))
        self.tok = tokenizer
        hid = backbone.config.hidden_size
        self.head_kind = mcfg.get("head", "linear")
        self.drop = nn.Dropout(mcfg.get("dropout", 0.1))
        self.head = nn.Linear(hid, 1)
        self.proj = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, mcfg.get("proj_dim", 256)))
        if self.head_kind == "lm_token":
            enc = lambda w: tokenizer(w, add_special_tokens=False)["input_ids"][-1]
            self.yes_id, self.no_id = enc(" yes"), enc(" no")

    def _decoder(self):
        base = self.lm.get_base_model()
        return base.get_decoder() if hasattr(base, "get_decoder") else base.model

    def forward(self, input_ids, attention_mask) -> dict[str, torch.Tensor]:
        # the decoder only (LoRA is inside it): no full-vocabulary logits for every position
        pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)    # left padding: positions start at the first token
        out = self._decoder()(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos, use_cache=False)
        h_last = out.last_hidden_state[:, -1, :]                 # left padded: the last position is the last token
        if self.head_kind == "lm_token":
            base = self.lm.get_base_model()
            logits = base.lm_head(h_last.to(base.lm_head.weight.device))
            score = (logits[:, self.yes_id] - logits[:, self.no_id]).float().to(self.head.weight.device)
        h = h_last.to(self.head.weight.device, self.head.weight.dtype)   # the heads live on the first GPU
        if self.head_kind != "lm_token":
            score = self.head(self.drop(h)).squeeze(-1).float()
        z = F.normalize(self.proj(h).float(), dim=-1)
        return {"score": score, "z": z}

    def trainable(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        lora = [p for n, p in self.lm.named_parameters() if p.requires_grad]
        head = list(self.head.parameters()) + list(self.proj.parameters())
        return lora, head

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        self.lm.save_pretrained(path)
        torch.save({"head": self.head.state_dict(), "proj": self.proj.state_dict(), "head_kind": self.head_kind},
                   f"{path}/head.pt")

    def load_heads(self, path: str) -> None:
        st = torch.load(f"{path}/head.pt", map_location="cpu")
        self.head.load_state_dict(st["head"])
        self.proj.load_state_dict(st["proj"])
