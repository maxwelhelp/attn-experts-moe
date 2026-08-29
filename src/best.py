# -*- coding: utf-8 -*-
"""Лучшая архитектура проекта: селективный гибридный слой внимания (hybrid100).

Состав, подтверждённый экспериментами (см. README, «Развитие 7»):

    attention-подслой = пул механизмов + роутер top-2:
        окно 72  -- точный локальный поиск (дорогой, точечный)
        lin9     -- линейное внимание, γ=0.9  (~10 шагов памяти)
        lin99    -- линейное внимание, γ=0.99 (~100 шагов)
        linL     -- линейное внимание, γ обучаемый (сходится к ~0.985)
        роутер видит [токен ; каузальное бегущее среднее проекций прошлого]
    FFN-подслой = обычный MoE (top-2 из 8)

Итог на разнородной нагрузке: CE 1.874 против 2.344 у плотных окон (-20%),
лучший худший случай по подзадачам. Пример:

    from src.best import make_hybrid100
    cfg, kinds = make_hybrid100(dim=128, num_layers=4)
    model = TinyCausalLM(cfg)
"""
from typing import Tuple

from .block import BlockConfig
from .model import ModelConfig

HYBRID_KINDS = ("window", "lin9", "lin99", "linL")


def make_hybrid100(dim: int = 128, num_layers: int = 4, window: int = 72,
                   num_ffn_experts: int = 8, ffn_top_k: int = 2,
                   attn_top_k: int = 2, vocab_size: int = 256,
                   max_seq_len: int = 512) -> Tuple[ModelConfig, tuple]:
    """Конфиг лучшей сборки: гетерогенный пул внимания + проекционный роутер.

    Возвращает (ModelConfig, kinds). Все решения зафиксированы
    экспериментами; произвольные -- только размерности и длины.
    """
    blk = BlockConfig(
        dim=dim,
        num_ffn_experts=num_ffn_experts,
        ffn_top_k=ffn_top_k,
        num_attn_experts=len(HYBRID_KINDS),
        attn_top_k=attn_top_k,
        attn_expert_kinds=HYBRID_KINDS,
        window=window,
        proj_router=True,        # роутер видит дешёвую проекцию прошлого
        attn_select_frac=1.0,    # БЕЗ бюджета: на этом пуле он вредит
    )
    cfg = ModelConfig(vocab_size=vocab_size, max_seq_len=max_seq_len,
                      num_layers=num_layers, block=blk,
                      layouts=("sequential",))
    return cfg, HYBRID_KINDS
