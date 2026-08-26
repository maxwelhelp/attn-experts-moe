"""Top-k routing for sparse MoE layers.

Supports two selection regimes:
  * uniform top-k  -- pick the best `top_k` experts regardless of type;
  * typed quotas   -- pool contains experts of different types ('ffn', 'attn'),
    and per token we pick exactly `typed_ks[type]` best experts of each type.
    This is the "two kinds of experts in one pool" design: for every token the
    layer assembles a small team, e.g. {attn: 1, ffn: 1}.

Auxiliary losses:
  * Switch-style load balancing loss (fed into the total training loss);
  * optional z-loss on router logits (JetMoE / ST-MoE style) for stability.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouteResult:
    idx: torch.Tensor        # [N, K] selected expert indices per token
    weights: torch.Tensor    # [N, K] gate weights (softmax over selected)
    aux_loss: torch.Tensor   # scalar load-balancing loss
    z_loss: torch.Tensor     # scalar z-loss


def typed_topk(
    logits: torch.Tensor,          # [N, E]
    expert_types: List[str],       # length E
    ks: Dict[str, int],            # type -> how many to select
) -> torch.Tensor:
    """Select top-k experts of each type from joint logits; returns [N, sum(k)]."""
    parts = []
    for t, k in ks.items():
        cols = torch.tensor(
            [e for e, et in enumerate(expert_types) if et == t],
            device=logits.device, dtype=torch.long,
        )
        if cols.numel() == 0 or k <= 0:
            continue
        local = logits.index_select(1, cols).topk(min(k, int(cols.numel())), dim=-1).indices
        parts.append(cols[local])                       # [N, k] global expert ids
    assert parts, "no selectable expert types"
    return torch.cat(parts, dim=-1)


class TopKRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: Optional[int] = None,
        *,
        expert_types: Optional[List[str]] = None,
        typed_ks: Optional[Dict[str, int]] = None,
        norm_topk_prob: bool = True,
        renorm_typed: bool = False,
        aux_coef: float = 1e-2,
        z_loss_coef: float = 1e-3,
        jitter_eps: float = 0.0,
    ):
        super().__init__()
        if typed_ks is not None:
            assert expert_types is not None and len(expert_types) == num_experts
            assert set(typed_ks) <= set(expert_types)
            top_k = sum(v for v in typed_ks.values() if v > 0)
        assert top_k is not None and 1 <= top_k <= num_experts

        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_types = list(expert_types) if expert_types is not None else None
        self.typed_ks = dict(typed_ks) if typed_ks is not None else None
        self.norm_topk_prob = norm_topk_prob
        self.renorm_typed = renorm_typed
        self.aux_coef = aux_coef
        self.z_loss_coef = z_loss_coef
        self.jitter_eps = jitter_eps
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RouteResult:      # x: [N, D]
        n = x.shape[0]
        logits = self.gate(x).float()                       # [N, E]
        if self.training and self.jitter_eps > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_eps

        if self.typed_ks is not None:
            idx = typed_topk(logits, self.expert_types, self.typed_ks)
        else:
            idx = logits.topk(self.top_k, dim=-1).indices

        # gates: softmax restricted to the selected experts (renormalised),
        # which is Mixtral's norm_topk_prob=True behaviour.
        sel_logits = logits.gather(-1, idx)                 # [N, K]
        if self.norm_topk_prob and idx.shape[-1] > 1 and \
                (self.typed_ks is None or self.renorm_typed):
            weights = F.softmax(sel_logits, dim=-1)
        elif self.norm_topk_prob and idx.shape[-1] > 1:
            weights = torch.softmax(logits, dim=-1).gather(-1, idx)
        else:
            weights = torch.softmax(logits, dim=-1).gather(-1, idx)
        weights = weights.to(x.dtype)

        aux_loss = x.new_zeros((), dtype=torch.float32)
        z_loss = x.new_zeros((), dtype=torch.float32)
        if self.training:
            probs = torch.softmax(logits, dim=-1)           # [N, E]
            # f_i = fraction of tokens that route to expert i (counting all k slots).
            assign = F.one_hot(idx, self.num_experts).sum(1).float()   # [N, E]
            f = assign.sum(0) / max(n, 1)
            p = probs.mean(0)
            aux_loss = self.num_experts * (f * p).sum()
            z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        return RouteResult(idx=idx, weights=weights, aux_loss=aux_loss, z_loss=z_loss)

    def extra_repr(self) -> str:
        mode = f"typed{self.typed_ks}" if self.typed_ks is not None else f"top-{self.top_k}"
        return f"{mode}, E={self.num_experts}"
