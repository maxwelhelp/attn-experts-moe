"""attn-experts-moe: MoE transformer where experts come in two types --
sparse attention experts and FFN experts -- assembled per token by a router."""

from .block import BlockConfig, DualMoEBlock
from .experts import AttentionExpert, FFNExpert, ProjectionMoEAttention
from .model import ModelConfig, TinyCausalLM
from .moe import SparseMoE, make_attn_pool, make_ffn_pool
from .router import RouteResult, TopKRouter, typed_topk

__version__ = "0.1.0"
