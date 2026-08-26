"""Иерархическое «сначала дёшево, потом уточнение» -- прототип идеи.

Ступень 1 (всегда, дёшево): контекст сжимается в лендмарки -- по одному
вектору на блок позиций (пулинг + линейная проекция). Стоимость ~O(T·d).

Гейт: для каждого БЛОКА запросов билинейным скорингом сравнивается его
резюме с лендмарками всех блоков ключей; берутся top-k блоков (+ свой блок
и все предыдущие выбранные уважают каузальность). Это «роутер читает
дешёвую проекцию», а не скрытое состояние.

Ступень 2 (дорогая, выборочно): точное каузальное внимание, но каждое
подмножество запросов смотрит только на выбранные блоки ключей.

Опционально -- петля уточнения с фиксированным V: K,V считаются один раз,
а Q-проекция пересчитывается несколько раз; каждая итерация уточняет
карту внимания при замороженных значениях (родня Universal Transformer/ACT).

Честное ограничение прототипа: маска реализована через плотный SDPA, т.е.
FLOPs экономит только структура выбора (метрика selected_fraction), но не
стеночное время -- для настоящей экономии нужен блочно-разреженный кернел
(см. README, «Куда развивать»).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HierConfig:
    dim: int = 128
    n_heads: int = 4
    block_size: int = 32          # размер блока позиций (лендмарк на блок)
    top_blocks: int = 3           # сколько блоков ключей выбирает гейт
    loop_iters: int = 2           # итерации уточнения с фиксированным V


class HierRefineAttention(nn.Module):
    def __init__(self, cfg: HierConfig, dropout: float = 0.0):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads, self.head_dim = cfg.n_heads, cfg.dim // cfg.n_heads

        # ступень 1: дешёвая проекция контекста в лендмарки
        self.landmark = nn.Linear(cfg.dim, cfg.dim, bias=False)

        # гейт: билинейный скоринг «блок запросов <-> блок ключей»
        self.gate_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.gate_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.gate_scale = 1.0 / math.sqrt(cfg.dim)

        # ступень 2: точное внимание; Q своя на каждую итерацию петли,
        # K/V общие (фиксированный V внутри петли)
        self.wq = nn.ModuleList([nn.Linear(cfg.dim, cfg.dim, bias=False)
                                 for _ in range(cfg.loop_iters)])
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _pool(self, x):                                   # [B,T,D] -> [B,nb,D]
        B, T, D = x.shape
        bs = self.cfg.block_size
        pad = (bs - T % bs) % bs
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        nb = x.shape[1] // bs
        return x.view(B, nb, bs, D).mean(dim=2), nb       # среднее по блоку

    def _gate_mask(self, x):
        """(additive float mask [B,T,T], доля выбранных блоков).

        Гейт ПОТОКЕННЫЙ и строго каузальный: токен t скорит лендмарки только
        тех блоков, что целиком позади него (свой блок разрешён всегда и
        внутри режется потокенной каузальностью). Мягкая поправка
        beta*score внутри разрешённой зоны проводит градиент в параметры
        гейта (straight-through стиль).
        """
        B, T, _ = x.shape
        ksum, nk = self._pool(self.landmark(x))            # [B,nk,D]
        scores = torch.einsum("btd,bmd->btm",
                              self.gate_q(x),
                              self.gate_k(ksum)) * self.gate_scale   # [B,T,nk]

        bs = self.cfg.block_size
        starts = torch.arange(nk, device=x.device) * bs     # начало блока j
        ti = torch.arange(T, device=x.device)
        fully_past = (starts[None, :] + bs) <= ti[:, None]  # блок целиком до t
        own = (ti[:, None] // bs) == torch.arange(nk, device=x.device)[None, :]
        scores_m = scores.masked_fill(~fully_past[None], float("-inf"))

        k = min(self.cfg.top_blocks, nk)
        keep = torch.zeros_like(scores_m, dtype=torch.bool)
        keep.scatter_(-1, scores_m.topk(k, dim=-1).indices, True)
        keep &= fully_past[None]
        keep |= own                                        # свой блок всегда
        frac = float(keep.float().mean())

        # разворачиваем в потакенную маску по ключам [B,T,T]
        tok_keep = keep.repeat_interleave(bs, dim=2)[:, :, :T]
        pos = torch.arange(T, device=x.device)
        tok_keep &= pos[None, :] <= pos[:, None]

        blk_scores = scores_m.repeat_interleave(bs, dim=2)[:, :, :T]
        bias = torch.zeros(B, T, T, device=x.device, dtype=x.dtype)
        bias = bias.masked_fill(~tok_keep, float("-inf"))
        soft = torch.nan_to_num(blk_scores, neginf=0.0) * tok_keep.to(x.dtype)
        return bias + 0.05 * soft, frac

    def _attend(self, q, k, v, bias):
        B, T, D = q.shape
        s = lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(
            s(q), s(k), s(v), attn_mask=bias[:, None])
        return att.transpose(1, 2).reshape(B, T, D)

    # ------------------------------------------------------------------
    def forward(self, x):                                 # [B,T,D]
        B, T, D = x.shape
        bias, frac = self._gate_mask(x)
        self.last_selected_fraction = frac

        k = self.wk(x)                                    # фиксированные K/V
        v = self.wv(x)
        out = 0
        for it in range(self.cfg.loop_iters):
            q = self.wq[it](x)                            # Q уточняется
            out = out + self.drop(self._attend(q, k, v, bias))
        return self.wo(out)


class DenseAttention(nn.Module):
    """Плотный базлайн того же интерфейса -- для сравнения."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        from .block import CausalSelfAttention
        self.inner = CausalSelfAttention(dim, n_heads=n_heads, dropout=dropout)

    def forward(self, x):
        return self.inner(x)
