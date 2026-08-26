"""Tiny causal LM assembled from DualMoEBlocks (demo scale)."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import BlockConfig, DenseBlock, DualMoEBlock, RMSNorm


@dataclass
class ModelConfig:
    vocab_size: int = 256
    max_seq_len: int = 512
    num_layers: int = 4
    block: BlockConfig = field(default_factory=BlockConfig)
    tie_embeddings: bool = True
    layouts: Tuple[str, ...] = ("sequential",)   # per-block layout: 'sequential' | 'mixed'


class TinyCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.block.dim
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, d)
        layouts = [cfg.layouts[i % len(cfg.layouts)] for i in range(cfg.num_layers)]
        self.blocks = nn.ModuleList(
            DenseBlock(cfg.block) if lay == "dense"
            else DualMoEBlock(cfg.block, layout=lay)
            for lay in layouts
        )
        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[dict]]:
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]

        for blk in self.blocks:
            x = blk(x)

        logits = self.head(self.ln_f(x))
        metrics = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                 targets.reshape(-1))
            zero = torch.zeros((), device=x.device)
            aux = sum((sum(blk.last_aux, zero) for blk in self.blocks), zero)
            z = sum((sum(blk.last_z, zero) for blk in self.blocks), zero)
            total = ce + self.cfg.block.aux_coef * aux + self.cfg.block.z_loss_coef * z
            metrics = {"ce": ce.detach(), "aux": aux.detach(), "z": z.detach(),
                       "total": total}
            return logits, metrics
        return logits, None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return idx

    # ------------------------------------------------------------------
    @property
    def device(self):
        return self.tok_emb.weight.device

    def num_params(self, active: bool = False) -> int:
        """Total params; `active` estimates params touched per token
        (embeddings excluded, k experts per type instead of the full pool)."""
        from .experts import AttentionExpert, ProjectionMoEAttention
        from .moe import SparseMoE, TYPE_ATTN, TYPE_FFN

        n = sum(p.numel() for p in self.tok_emb.parameters())
        n += sum(p.numel() for p in self.pos_emb.parameters())
        n += sum(p.numel() for p in self.ln_f.parameters())
        n += 0 if self.cfg.tie_embeddings else sum(p.numel() for p in self.head.parameters())

        for blk in self.blocks:
            cfg = blk.cfg
            if getattr(blk, "layout", None) == "dense":
                n += sum(p.numel() for p in blk.parameters())
                continue
            moes: list = []
            if blk.layout == "sequential":
                if isinstance(blk.attn, ProjectionMoEAttention):
                    ka = cfg.attn_top_k
                    n += blk.attn.router.gate.weight.numel() + ka * blk.attn.experts[0].weight.numel()
                    n += blk.attn.wo.weight.numel()
                elif not hasattr(blk.attn, "router"):      # напр. HierRefineAttention
                    n += sum(p.numel() for p in blk.attn.parameters())
                else:
                    moes.append((blk.attn, {TYPE_ATTN: cfg.attn_top_k}))
                moes.append((blk.ffn, {TYPE_FFN: cfg.ffn_top_k}))
            else:
                moes.append((blk.mixed, dict(cfg.mixed_ks or
                                             {"ffn": cfg.ffn_top_k, "attn": 1})))
            for moe, ks in moes:
                n += moe.router.gate.weight.numel()
                if moe.shared_expert is not None:
                    n += sum(p.numel() for p in moe.shared_expert.parameters())
                if not active:
                    for e in moe.experts:
                        n += sum(p.numel() for p in e.parameters())
                else:
                    # k experts of each type are touched per token; the pool is
                    # homogeneous within a type, so take one representative.
                    for t, k in ks.items():
                        reps = [e for e, et in zip(moe.experts, moe.types) if et == t]
                        if reps and k > 0:
                            n += k * sum(p.numel() for p in reps[0].parameters())
        return n
