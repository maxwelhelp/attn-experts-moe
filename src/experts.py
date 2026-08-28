"""Experts: plain FFN experts and (sparse) attention experts.

Two flavours of attention MoE are supported in this project:

1. `AttentionExpert` -- *independent* attention expert ("mini head-group").
   It owns its own Wq/Wk/Wv/Wo and its own sparse attention pattern
   (dense causal | sliding window | top-k keys). Per token the router picks
   top-k such experts, so every query position is answered by a *mixture of
   different sparse attentions* -- exactly the idea of assembling a
   "composition of attentions" per token.

   Subtlety (why JetMoE did not do this): a query routed to expert e still
   needs K/V *of all positions* to attend over. We resolve it the honest way:
   each selected expert computes K/V densely for the whole sequence but Q and
   the output projection only for the tokens actually routed to it. Savings:
   ~2x on Q/O compute vs a dense head-group, plus true per-token sparsity of
   the mixture; cost: K/V computed once per selected expert.

2. `ProjectionMoEAttention` -- JetMoE-style: experts are projection chunks.
   Each expert emits 1/top_k of Q,K,V; slices are concatenated and ONE shared
   causal attention kernel runs on top. Sparsity lives only in projections,
   but it is cheap and batch-friendly (this is the published baseline).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ACT2FN = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}


class FFNExpert(nn.Module):
    """SwiGLU-style FFN expert (token-wise, no cross-token mixing)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, mult: float = 4.0,
                 glu: bool = True, act: str = "silu", dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(round(mult * dim))
        self.glu = glu
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) if glu else None
        self.act = ACT2FN[act]
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # [N, D] -> [N, D]
        h = self.w1(x) * self.act(self.w3(x)) if self.glu else self.act(self.w1(x))
        return self.w2(self.drop(h))


class AttentionExpert(nn.Module):
    """Independent sparse-causal attention expert.

    Two equivalent entry points:
      * `forward_selected(x_seq, rows)` -- computes attention ONLY at query
        positions `rows` of one sequence (the literal "routed queries" path);
      * `forward_batch(x)` -- computes the expert's sparse attention densely
        for the whole batch and lets the MoE layer gather the routed rows.
        Same math for the selected rows (attention at row t never depends on
        which other rows are queried), far fewer kernel launches -> used by
        default in SparseMoE.

    Patterns:
      * 'dense'  -- full causal attention;
      * 'window' -- causal sliding-window attention (`window` keys back);
      * 'topk'   -- keep only top-`topk_keys` keys per (query row, head).
    """

    def __init__(self, dim: int, n_heads: int = 2, pattern: str = "window",
                 window: int = 64, topk_keys: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim, self.n_heads, self.dh = dim, n_heads, dim // n_heads
        assert pattern in ("dense", "window", "topk")
        self.pattern = pattern
        self.window = window
        self.topk_keys = topk_keys
        self.scale = 1.0 / math.sqrt(self.dh)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self._mask_cache: dict = {}

    # ------------------------------------------------------------------ utils
    def _band_mask(self, T: int, device) -> torch.Tensor:
        """Bool [T,T]: True = allowed key for that query (causal + window)."""
        key = (T, str(device))
        if key not in self._mask_cache:
            pos = torch.arange(T, device=device)
            m = pos[None, :] <= pos[:, None]
            if self.pattern == "window":
                m &= (pos[:, None] - pos[None, :]) < self.window
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                q_rows: torch.Tensor, T: int) -> torch.Tensor:
        """Single-sequence path. q: [S,H,dh] rows q_rows; k,v: [T,H,dh]."""
        S, H, dh = q.shape
        scores = torch.einsum("shd,thd->hst", q, k) * self.scale      # [H,S,T]

        pos = torch.arange(T, device=q.device)
        allow = pos[None, :] <= q_rows[:, None]                        # causal [S,T]
        if self.pattern == "window":
            allow &= (q_rows[:, None] - pos[None, :]) < self.window
        scores = scores.masked_fill(~allow[None], float("-inf"))

        if self.pattern == "topk":
            kk = min(self.topk_keys, T)
            keep = torch.zeros_like(scores, dtype=torch.bool)          # [H,S,T]
            keep.scatter_(-1, scores.topk(kk, dim=-1).indices, True)
            scores = scores.masked_fill(~keep, float("-inf"))

        attn = torch.softmax(scores, dim=-1)                           # diagonal always allowed
        out = torch.einsum("hst,thd->shd", attn, v).reshape(S, H * dh)
        return self.drop(out)

    # ---------------------------------------------------------------- forwards
    def _split(self, t: torch.Tensor, B: int, T: int):
        return t.view(B, T, self.n_heads, self.dh).transpose(1, 2)

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Batched dense-query path. x: [B,T,D] -> [B,T,D]."""
        B, T, D = x.shape
        q = self._split(self.wq(x), B, T)
        k = self._split(self.wk(x), B, T)
        v = self._split(self.wv(x), B, T)

        if self.pattern in ("dense", "window"):
            mask = None if self.pattern == "dense" else self._band_mask(T, x.device)
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                 is_causal=mask is None)
        else:  # 'topk', batched manual kernel
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
            # causal restriction FIRST, exactly like the exact path: top-k keys
            # are picked among past positions only
            allow = self._band_mask(T, x.device)                       # causal [T,T]
            scores = scores.masked_fill(~allow[None, None], float("-inf"))
            kk = min(self.topk_keys, T)
            keep = torch.zeros_like(scores, dtype=torch.bool)
            keep.scatter_(-1, scores.topk(kk, dim=-1).indices, True)
            scores = scores.masked_fill(~keep, float("-inf"))
            att = torch.matmul(torch.softmax(scores, dim=-1), v)

        att = att.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.wo(att))

    def forward_selected(self, x_seq: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """x_seq: [T,D] single sequence; rows: [S] long tensor of query positions."""
        T = x_seq.shape[0]
        H, dh = self.n_heads, self.dh
        q = self.wq(x_seq.index_select(0, rows)).view(-1, H, dh)       # [S,H,dh]
        k = self.wk(x_seq).view(T, H, dh)                              # dense K/V ...
        v = self.wv(x_seq).view(T, H, dh)                              # ... see docstring
        return self.wo(self._attend(q, k, v, rows, T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts [T,D] or [B,T,D]."""
        if x.dim() == 3:
            return self.forward_batch(x)
        return self.forward_selected(x, torch.arange(x.shape[0], device=x.device))


class LinearAttentionExpert(AttentionExpert):
    """Эксперт на линейном внимании (упрощённый DeltaNet / родня Mamba-семейства).

    Состояние:  S_t = Σ_{s<=t} φ(k_s) v_sᵀ      (накопительная ассоциативная память)
    Выход:      y_t  = φ(q_t) · S_t             φ(x) = elu(x)+1 > 0

    Стоимость O(T·d²) НЕ зависит от длины контекста и от того, сколько токенов
    маршрутизировано: состояние считается для всех позиций (дёшево), выход
    берётся только для выбранных строк -- тот же трюк, что с K/V у внимания.

    Режимы затухания (decay):
      * "none"  -- gamma=1, вечная память (исходный вариант);
      * "fixed" -- gamma=gamma_init на голову: конкретный горизонт
        (0.9 ~ десять шагов, 0.99 ~ сотня) -- «спектр горизонтов»;
      * "learn" -- gamma обучаемый на голову (через сигмоиду).
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0,
                 decay: str = "none", gamma_init: float = 0.99):
        super().__init__(dim, n_heads=n_heads, pattern="dense")   # маска не нужна
        self.pattern = "linear"
        self.decay_mode = decay
        assert decay in ("none", "fixed", "learn")
        if decay == "fixed":
            self.register_buffer("gamma_logit",
                                 torch.full((n_heads,), self._logit(gamma_init)))
        elif decay == "learn":
            self.gamma_logit = nn.Parameter(
                torch.full((n_heads,), self._logit(gamma_init)))

    @staticmethod
    def _logit(p):
        import math
        return math.log(p / (1.0 - p))

    def _gamma(self):
        return torch.sigmoid(self.gamma_logit).view(1, self.n_heads, 1, 1)

    def gamma_values(self):
        if self.decay_mode == "none":
            return [1.0] * self.n_heads
        return [round(float(v), 4) for v in torch.sigmoid(self.gamma_logit)]

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:     # [B,T,D]
        B, T, D = x.shape
        q = self._split(self.wq(x), B, T)                         # [B,H,T,dh]
        k = self._split(self.wk(x), B, T)
        v = self._split(self.wv(x), B, T)
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0

        if self.decay_mode == "none":
            # состояние S_t = cumsum_{s<=t}(φk_s ⊗ v_s) -> [B,H,T,dh,dv]
            outer = phi_k.unsqueeze(-1) * v.unsqueeze(-2)         # [B,H,T,dh,dv]
            state = outer.cumsum(dim=2)
            num = torch.einsum("bhtd,bhtde->bhte", phi_q, state)
            # нормировка на сумму весов ключей -- иначе выход не ограничен и
            # разрастается на тысячи, заглушая остальных экспертов в смеси
            den = torch.einsum("bhtd,bhtd->bht", phi_q,
                               phi_k.cumsum(dim=2)).clamp_min(1e-3)
            out = num / den.unsqueeze(-1)                         # [B,H,T,dv]
        else:
            # каузальная рекуррентность с затуханием: S_t = γ·S_{t-1} + φk⊗v
            g = self._gamma()                                     # [1,H,1,1]
            gh = g.view(1, self.n_heads, 1)                       # [1,H,1] для z
            state = x.new_zeros(B, self.n_heads, self.dh, self.dh)
            zsum = x.new_zeros(B, self.n_heads, self.dh)
            outs = []
            for t in range(T):
                state = g * state + torch.einsum(
                    "bhd,bhe->bhde", phi_k[:, :, t], v[:, :, t])
                zsum = gh * zsum + phi_k[:, :, t]
                num = torch.einsum("bhd,bhde->bhe", phi_q[:, :, t], state)
                den = torch.einsum("bhd,bhd->bh",
                                   phi_q[:, :, t], zsum).clamp_min(1e-3)
                outs.append(num / den.unsqueeze(-1))
            out = torch.stack(outs, dim=2)                        # [B,H,T,dv]
        return self.drop(self.wo(out.transpose(1, 2).reshape(B, T, D)))


class ProjectionMoEAttention(nn.Module):
    """JetMoE-style sparse MoE attention: experts are Q/K/V projection chunks.

    Router picks top_k experts per token; each expert maps D -> 3*(D/top_k)
    (its slice of Q, K and V); the selected slices are summed with gate weights
    into full-size Q, K, V, then ONE shared causal attention kernel runs.
    Output projection Wo is dense here (JetMoE uses a second small MoE for O).
    """

    def __init__(self, dim: int, n_heads: int, num_experts: int, top_k: int,
                 norm_topk_prob: bool = True, aux_coef: float = 1e-2,
                 z_loss_coef: float = 1e-3, dropout: float = 0.0):
        from .router import TopKRouter
        super().__init__()
        assert dim % n_heads == 0 and dim % top_k == 0
        self.dim, self.n_heads, self.top_k = dim, n_heads, top_k
        slice_d = dim // top_k
        self.experts = nn.ModuleList(
            [nn.Linear(dim, 3 * slice_d, bias=False) for _ in range(num_experts)]
        )
        self.router = TopKRouter(dim, num_experts, top_k,
                                 norm_topk_prob=norm_topk_prob,
                                 aux_coef=aux_coef, z_loss_coef=z_loss_coef)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(dim // n_heads)

    def route_result(self):
        return getattr(self.router, "last", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:               # [B,T,D]
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        rr = self.router(flat)
        self.router.last = rr

        N, slice_d = flat.shape[0], D // self.top_k
        q = flat.new_zeros(N, D)
        k, v = q.clone(), q.clone()
        K = rr.idx.shape[-1]

        # zero-sync dispatch: per-expert boolean masks over the flat assignment
        # list; slices are placed by routing rank (JetMoE concatenation order)
        flat_idx = rr.idx.reshape(-1)
        assign = torch.arange(flat_idx.numel(), device=flat.device)
        pos_all = torch.div(assign, K, rounding_mode="floor")
        slot_all = assign.remainder(K)
        cols_base = torch.arange(slice_d, device=flat.device)

        for e, proj in enumerate(self.experts):
            m = flat_idx == e
            rows = pos_all[m]
            seg_cols = (slot_all[m, None] * slice_d + cols_base[None, :])
            w = rr.weights.reshape(-1)[assign[m]].to(flat.dtype)
            c = proj(flat[rows]) * w[:, None]                     # [S, 3*slice]
            rows_g = rows.repeat_interleave(slice_d)
            cols_g = seg_cols.reshape(-1)
            for part, buf in enumerate((q, k, v)):                # 0=q, 1=k, 2=v
                piece = c[:, part * slice_d:(part + 1) * slice_d]
                buf.index_put_((rows_g, cols_g), piece.reshape(-1))

        def split(t):
            return t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)

        att = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
        out = att.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.wo(out))
