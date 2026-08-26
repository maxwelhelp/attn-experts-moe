"""SparseMoE: one router over a pool of experts, token-level top-k dispatch.

Dispatch (production-style): all (token, slot) assignments are sorted by
expert id once, every expert then processes ONE contiguous chunk with plain
GEMMs, and a single scatter-add collects each token's k contributions.
One small host sync per layer supplies the chunk boundaries.

The pool may contain experts of different types:
  * FFNExpert            -- token-wise;
  * AttentionExpert      -- sequence-aware sparse attention (see its docs);
selection is uniform top-k or typed quotas (router handles both).

Optionally a dense `shared_expert` (DeepSeek-V2 style) serves every token.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .experts import (AttentionExpert, FFNExpert, LinearAttentionExpert,
                      ProjectionMoEAttention)
from .router import TopKRouter

TYPE_FFN, TYPE_ATTN = "ffn", "attn"


class SparseMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        experts: Sequence[nn.Module],
        expert_types: Optional[Sequence[str]] = None,
        *,
        top_k: Optional[int] = None,
        typed_ks: Optional[Dict[str, int]] = None,
        norm_topk_prob: bool = True,
        renorm_typed: bool = False,
        aux_coef: float = 1e-2,
        z_loss_coef: float = 1e-3,
        shared_expert: Optional[nn.Module] = None,
    ):
        super().__init__()
        types = list(expert_types) if expert_types is not None else [TYPE_FFN] * len(experts)
        assert len(types) == len(experts)
        self.experts = nn.ModuleList(experts)
        self.types = types
        self.router = TopKRouter(
            dim, len(experts),
            top_k=top_k,
            expert_types=types if typed_ks is not None else None,
            typed_ks=typed_ks,
            norm_topk_prob=norm_topk_prob,
            renorm_typed=renorm_typed,
            aux_coef=aux_coef,
            z_loss_coef=z_loss_coef,
        )
        self.shared_expert = shared_expert
        self.last_route = None

    # ------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:               # [B,T,D]
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        rr = self.router(flat)
        self.last_route = rr

        K = rr.idx.shape[-1]
        flat_idx = rr.idx.reshape(-1)                                 # [N*K]
        order = torch.argsort(flat_idx, stable=True)                  # group by expert
        pos_sorted = torch.div(order, K, rounding_mode="floor")       # token per slot
        w_sorted = rr.weights.reshape(-1)[order].to(flat.dtype)

        counts = torch.bincount(flat_idx, minlength=len(self.experts))
        offsets = torch.zeros(len(self.experts) + 1,
                              dtype=torch.long, device=flat.device)
        torch.cumsum(counts, 0, out=offsets[1:])
        bounds = offsets.tolist()          # the ONLY host sync in this layer

        xp = flat.index_select(0, pos_sorted)                         # [N*K, D]
        yp = torch.empty_like(xp)

        attn_full: dict = {}
        for e, expert in enumerate(self.experts):
            s0, s1 = bounds[e], bounds[e + 1]
            if s1 <= s0:
                continue
            if isinstance(expert, AttentionExpert):
                # batched fast path: dense compute once per used expert, take
                # the routed query rows (mathematically identical -- see docs)
                if e not in attn_full:
                    attn_full[e] = expert.forward_batch(x).reshape(B * T, D)
                yp[s0:s1] = attn_full[e].index_select(0, pos_sorted[s0:s1]) \
                    * w_sorted[s0:s1, None]
            else:
                yp[s0:s1] = expert(xp[s0:s1]) * w_sorted[s0:s1, None]

        y = torch.zeros_like(flat)
        y.index_add_(0, pos_sorted, yp)     # each token collects its k slots
        if self.shared_expert is not None:
            y = y + self.shared_expert(flat)
        return y.view(B, T, D)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def usage(self) -> torch.Tensor:
        """Fraction of assignments routed to each expert (last pass)."""
        if self.last_route is None:
            return torch.zeros(len(self.experts))
        counts = torch.bincount(self.last_route.idx.reshape(-1),
                                minlength=len(self.experts)).float()
        return counts / counts.sum()


def make_ffn_pool(dim: int, num_experts: int, **kw) -> List[FFNExpert]:
    return [FFNExpert(dim, **kw) for _ in range(num_experts)]


def make_attn_pool(dim: int, num_experts: int, n_heads: int = 2, **kw) -> List[AttentionExpert]:
    """Пул attention-экспертов. kinds (опционально) задаёт состав явно,
    например ('window','window','linear','topk') -- смесь РАЗНЫХ механизмов
    смешивания токенов; иначе все эксперты одного pattern с разными окнами."""
    experts = []
    kinds = kw.pop("kinds", None)
    base_window = kw.pop("window", 64)
    if kinds is None:
        kinds = [kw.get("pattern", "window")] * num_experts
    assert len(kinds) == num_experts
    for i, kind in enumerate(kinds):
        if kind == "linear":
            experts.append(LinearAttentionExpert(dim, n_heads=max(4, n_heads),
                                                 dropout=kw.get("dropout", 0.0)))
            continue
        experts.append(AttentionExpert(
            dim, n_heads=n_heads, pattern=kind,
            window=max(4, base_window // (2 ** min(i, 3))),
            topk_keys=kw.get("topk_keys", 8),
            dropout=kw.get("dropout", 0.0)))
    return experts
