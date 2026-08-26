"""Transformer block with MoE everywhere.

Two block layouts ("how to place the two expert types"):

  layout='sequential'  -- JetMoE-style: two sublayers, each its own router:
        x -> LN -> [MoE attention experts] -> +res -> LN -> [MoE FFN experts] -> +res

  layout='mixed'       -- one shared pool with typed quotas and ONE router:
        x -> LN -> MixedMoE( pick {attn: ka, ffn: kf} per token ) -> +res
     i.e. per token the layer assembles its own team of a sparse-attention
     expert(s) plus FFN expert(s).

Attention flavour ('attn_kind'):
  * 'independent' -- our AttentionExpert pool (true sparse patterns per expert);
  * 'projection'  -- JetMoE-style ProjectionMoEAttention (shared kernel).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import FFNExpert, ProjectionMoEAttention
from .moe import TYPE_ATTN, TYPE_FFN, SparseMoE, make_attn_pool, make_ffn_pool


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# --------------------------------------------------------------------------
# Обычный (плотный) трансформерный блок -- базлайн для сравнения.
@dataclass
class BlockConfig:
    dim: int = 128
    # FFN experts
    num_ffn_experts: int = 8
    ffn_top_k: int = 2
    ffn_mult: float = 4.0
    ffn_glu: bool = True
    # attention experts / projection attention
    attn_kind: str = "independent"      # 'independent' | 'projection'
    num_attn_experts: int = 4
    attn_top_k: int = 2                 # experts per token for attention sublayer/pool
    attn_heads_per_expert: int = 2
    attn_pattern: str = "window"        # 'dense' | 'window' | 'topk'
    window: int = 64
    attn_topk_keys: int = 8
    proj_attn_heads: int = 4            # heads of the shared kernel (projection kind)
    attn_expert_kinds: Optional[tuple] = None   # явный состав пула, напр. ('window','linear')
    hier_block_size: int = 32           # блок иерархического внимания
    hier_top_blocks: int = 3            # top-k прошлых блоков в гейте
    hier_loop_iters: int = 1            # итерации уточнения с фиксированным V
    proj_router: bool = False           # роутер внимания видит дешёвую проекцию контекста
    # mixed-pool quotas (layout='mixed'): {type: k}
    mixed_ks: Optional[Dict[str, int]] = None   # default {'ffn': ffn_top_k, 'attn': 1}
    # routing / extras
    norm_topk_prob: bool = True
    renorm_typed: bool = False   # mixed pool: renormalise softmax over selected union
    aux_coef: float = 1e-2
    z_loss_coef: float = 1e-3
    shared_expert: bool = False
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Обычное каузальное multi-head внимание без всяких экспертов."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                    # [B,T,D]
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def sp(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = F.scaled_dot_product_attention(sp(q), sp(k), sp(v), is_causal=True)
        att = att.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.proj(att))


class DenseBlock(nn.Module):
    """Классический блок: внимание + FFN, всё плотное, без роутинга."""

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.layout = "dense"
        self.cfg = cfg
        heads = next(h for h in (16, 12, 10, 8, 6, 4, 2) if cfg.dim % h == 0)
        self.attn = CausalSelfAttention(cfg.dim, n_heads=heads,
                                        dropout=cfg.dropout)
        self.ffn = FFNExpert(cfg.dim, mult=cfg.ffn_mult, glu=cfg.ffn_glu,
                             dropout=cfg.dropout)
        self.ln1, self.ln2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.last_aux: List[torch.Tensor] = []
        self.last_z: List[torch.Tensor] = []

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    @torch.no_grad()
    def usage(self):
        return {}



class DualMoEBlock(nn.Module):
    def __init__(self, cfg: BlockConfig, layout: str = "sequential"):
        super().__init__()
        assert layout in ("sequential", "mixed")
        self.cfg = cfg
        self.layout = layout
        shared = (
            FFNExpert(cfg.dim, mult=cfg.ffn_mult // 2, glu=cfg.ffn_glu)
            if cfg.shared_expert else None
        )

        if self.layout == "sequential":
            if cfg.attn_kind == "hier":
                from .hier import HierConfig, HierRefineAttention
                self.attn = HierRefineAttention(
                    HierConfig(dim=cfg.dim,
                               n_heads=max(4, cfg.attn_heads_per_expert),
                               block_size=cfg.hier_block_size,
                               top_blocks=cfg.hier_top_blocks,
                               loop_iters=cfg.hier_loop_iters),
                    dropout=cfg.dropout)
            elif cfg.attn_kind == "projection":
                self.attn = ProjectionMoEAttention(
                    cfg.dim, n_heads=cfg.proj_attn_heads,
                    num_experts=cfg.num_attn_experts, top_k=cfg.attn_top_k,
                    aux_coef=cfg.aux_coef, z_loss_coef=cfg.z_loss_coef,
                    dropout=cfg.dropout)
            else:
                self.attn = SparseMoE(
                    cfg.dim,
                    make_attn_pool(cfg.dim, cfg.num_attn_experts,
                                   n_heads=cfg.attn_heads_per_expert,
                                   kinds=cfg.attn_expert_kinds,
                                   pattern=cfg.attn_pattern, window=cfg.window,
                                   topk_keys=cfg.attn_topk_keys, dropout=cfg.dropout),
                    [TYPE_ATTN] * cfg.num_attn_experts,
                    top_k=cfg.attn_top_k, norm_topk_prob=cfg.norm_topk_prob,
                    aux_coef=cfg.aux_coef, z_loss_coef=cfg.z_loss_coef,
                    proj_router=cfg.proj_router,
                )
            self.ffn = SparseMoE(
                cfg.dim,
                make_ffn_pool(cfg.dim, cfg.num_ffn_experts,
                              mult=cfg.ffn_mult, glu=cfg.ffn_glu, dropout=cfg.dropout),
                [TYPE_FFN] * cfg.num_ffn_experts,
                top_k=cfg.ffn_top_k, norm_topk_prob=cfg.norm_topk_prob,
                aux_coef=cfg.aux_coef, z_loss_coef=cfg.z_loss_coef,
                shared_expert=shared,
            )
            self.ln1, self.ln2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)

        else:  # mixed: one pool, typed quotas
            ks = cfg.mixed_ks or {"ffn": cfg.ffn_top_k, "attn": 1}
            assert cfg.attn_kind == "independent", \
                "mixed pool requires independent attention experts"
            assert ks.get(TYPE_ATTN, 0) >= 1 and ks.get(TYPE_FFN, 0) >= 1
            pool = (make_ffn_pool(cfg.dim, cfg.num_ffn_experts,
                                  mult=cfg.ffn_mult, glu=cfg.ffn_glu, dropout=cfg.dropout)
                    + make_attn_pool(cfg.dim, cfg.num_attn_experts,
                                     n_heads=cfg.attn_heads_per_expert,
                                     kinds=cfg.attn_expert_kinds,
                                     pattern=cfg.attn_pattern, window=cfg.window,
                                     topk_keys=cfg.attn_topk_keys, dropout=cfg.dropout))
            types = [TYPE_FFN] * cfg.num_ffn_experts + [TYPE_ATTN] * cfg.num_attn_experts
            self.mixed = SparseMoE(
                cfg.dim, pool, types, typed_ks=ks,
                norm_topk_prob=cfg.norm_topk_prob,
                renorm_typed=cfg.renorm_typed,
                aux_coef=cfg.aux_coef, z_loss_coef=cfg.z_loss_coef,
                shared_expert=shared,
            )
            self.ln1 = RMSNorm(cfg.dim)

        self.last_aux: List[torch.Tensor] = []
        self.last_z: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_aux, self.last_z = [], []
        if self.layout == "sequential":
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            for m in (self.attn, self.ffn):
                if not hasattr(m, "last_route"):        # напр. HierRefineAttention
                    continue
                r = m.router.last if isinstance(m, ProjectionMoEAttention) else m.last_route
                if r is not None:
                    self.last_aux.append(r.aux_loss)
                    self.last_z.append(r.z_loss)
        else:
            x = x + self.mixed(self.ln1(x))
            r = self.mixed.last_route
            self.last_aux.append(r.aux_loss)
            self.last_z.append(r.z_loss)
        return x

    @torch.no_grad()
    def usage(self) -> Dict[str, List]:
        out = {}
        if self.layout == "sequential":
            for name in ("attn", "ffn"):
                m = getattr(self, name, None)
                if isinstance(m, SparseMoE):
                    out[name] = m.usage().tolist()
        else:
            out["mixed"] = self.mixed.usage().tolist()
        return out
