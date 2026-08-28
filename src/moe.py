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
        proj_router: bool = False,
    ):
        super().__init__()
        types = list(expert_types) if expert_types is not None else [TYPE_FFN] * len(experts)
        assert len(types) == len(experts)
        self.experts = nn.ModuleList(experts)
        self.types = types
        # «Роутер видит проекцию»: к признакам токена добавляется каузальное
        # бегущее среднее спроецированного прошлого ctx_t = mean_{s<=t}(W x_s).
        # Дёшево (O(T*d)), строго каузально, даёт гейту глобальный контекст --
        # идея «роутер читает дешёвую проекцию» из иерархической схемы.
        self.ctx_proj = nn.Linear(dim, dim, bias=False) if proj_router else None
        router_dim = dim * 2 if proj_router else dim
        self.router = TopKRouter(
            router_dim, len(experts),
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
    @staticmethod
    def _causal_ctx(x: torch.Tensor, proj: Optional[nn.Linear]) -> torch.Tensor:
        """Каузальное бегущее среднее спроецированного прошлого [B,T,D]."""
        if proj is None:
            return None
        p = proj(x)
        return p.cumsum(dim=1) / torch.arange(
            1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)

    def forward_rows(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Вклад пула только в выбранных запросах: x [B,T,D], keep [B,T] bool.

        Незабранные позиции получают ноль (residual их проводит без изменений).
        Оконные/top-k эксперты считают ТОЛЬКО выбранные строки против полных
        K/V (точный путь forward_selected) -- здесь настоящая экономия FLOPs;
        линейные эксперты всё равно сканируют всю последовательность, им
        дешевле плотный путь.
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)

        r_feat = torch.cat([x, self._causal_ctx(x, self.ctx_proj)], dim=-1) \
            if self.ctx_proj is not None else x

        rr = self.router(r_feat.reshape(B * T, -1)[keep.reshape(B * T)])
        self.last_route = rr

        K = rr.idx.shape[-1]
        flat_idx = rr.idx.reshape(-1)
        order = torch.argsort(flat_idx, stable=True)
        w_sorted = rr.weights.reshape(-1)[order].to(x.dtype)

        counts = torch.bincount(flat_idx, minlength=len(self.experts))
        offsets = torch.zeros(len(self.experts) + 1, dtype=torch.long,
                              device=x.device)
        torch.cumsum(counts, 0, out=offsets[1:])
        bounds = offsets.tolist()

        # глобальные id выбранных токенов в порядке сортировки по экспертам
        # (каждый выбранный токен занимает K слотов роутера)
        gidx_all = keep.reshape(B * T).nonzero(as_tuple=False).squeeze(-1)
        tok_sorted = gidx_all.repeat_interleave(K)[order]

        for e, expert in enumerate(self.experts):
            s0, s1 = bounds[e], bounds[e + 1]
            if s1 <= s0:
                continue
            rows = tok_sorted[s0:s1]
            wts = w_sorted[s0:s1]
            if isinstance(expert, AttentionExpert) \
                    and getattr(expert, "pattern", None) in ("window", "topk"):
                b_of = torch.div(rows, T, rounding_mode="floor")
                t_of = rows % T
                for b in range(B):                    # точный путь по строкам
                    m = b_of == b
                    if m.any():
                        rb = t_of[m]
                        val = expert.forward_selected(x[b], rb)
                        out[b].index_put_((rb,), val * wts[m][:, None],
                                          accumulate=True)
            else:                                     # дешёвые: плотно один раз
                full = expert(x.reshape(B, T, D)).reshape(B * T, D)
                out.view(B * T, D).index_put_((rows,), full.index_select(0, rows)
                                              * wts[:, None], accumulate=True)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:               # [B,T,D]
        B, T, D = x.shape
        flat = x.reshape(B * T, D)                    # вход экспертов: D
        r_in = flat
        ctx = self._causal_ctx(x, self.ctx_proj)
        if ctx is not None:
            r_in = torch.cat([x, ctx], dim=-1).reshape(B * T, 2 * D)
        rr = self.router(r_in)
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
        if kind in ("linear", "lin9", "lin99", "linL"):
            decay = {"linear": "none", "lin9": "fixed", "lin99": "fixed",
                     "linL": "learn"}[kind]
            gamma = {"lin9": 0.9, "lin99": 0.99}.get(kind, 0.99)
            experts.append(LinearAttentionExpert(
                dim, n_heads=max(4, n_heads), dropout=kw.get("dropout", 0.0),
                decay=decay, gamma_init=gamma))
            continue
        experts.append(AttentionExpert(
            dim, n_heads=n_heads, pattern=kind,
            window=max(4, base_window // (2 ** min(i, 3))),
            topk_keys=kw.get("topk_keys", 8),
            dropout=kw.get("dropout", 0.0)))
    return experts
