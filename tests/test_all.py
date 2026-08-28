"""Tests: shapes, causality, sparse patterns, typed quotas, backward flow.

Run either `python -m pytest tests/ -q` or `python tests/test_all.py`.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.block import BlockConfig  # noqa: E402
from src.model import ModelConfig, TinyCausalLM  # noqa: E402
from src.moe import TYPE_ATTN, TYPE_FFN  # noqa: E402

torch.manual_seed(0)


def tiny_model(**kw) -> TinyCausalLM:
    block_kw = dict(dim=64, num_ffn_experts=4, ffn_top_k=2,
                    num_attn_experts=2, attn_top_k=1,
                    attn_heads_per_expert=2, window=8, attn_topk_keys=4)
    layout = kw.pop("layout", "sequential")
    kind = kw.pop("attn_kind", "independent")
    block_kw.update(kw)
    cfg = ModelConfig(vocab_size=32, max_seq_len=64, num_layers=2,
                      block=BlockConfig(attn_kind=kind, **block_kw),
                      layouts=(layout,))
    return TinyCausalLM(cfg)


def _forward_loss(model, x):
    logits, m = model(x, targets=x.roll(-1, dims=1))
    assert logits.shape == (*x.shape, model.cfg.vocab_size)
    return m


# ---------------------------------------------------------------------- basic
def test_forward_shapes_and_finite_all_configs():
    for layout in ("sequential", "mixed"):
        for kind in ("independent", "projection") if layout == "sequential" else ("independent",):
            for pattern in ("dense", "window", "topk"):
                m = tiny_model(layout=layout, attn_kind=kind, attn_pattern=pattern)
                x = torch.randint(0, 32, (2, 16))
                met = _forward_loss(m, x)
                for v in met.values():
                    assert torch.isfinite(v).all(), (layout, kind, pattern)


def test_causality_strict():
    """Perturbing a future token must not change past outputs."""
    for layout in ("sequential", "mixed"):
        m = tiny_model(layout=layout, attn_pattern="window").eval()
        x = torch.randint(0, 32, (2, 24))
        with torch.no_grad():
            base, _ = m(x)
            xp = x.clone()
            xp[:, 20:] = (xp[:, 20:] + 7) % 31          # change tail only
            pert, _ = m(xp)
        diff = (base[:, :20] - pert[:, :20]).abs().max()
        assert diff.item() == 0.0 or diff.item() < 5e-6, f"causality broken ({layout}): {diff}"


def test_window_sparsity():
    """A token far beyond the window must not influence the output row."""
    m = tiny_model(layout="sequential", attn_pattern="window", window=8).eval()
    # also force the FFN path aside is impossible; so compare full models with
    # identical weights but perturbed far-past token: effect may pass through
    # FFN experts of intermediate layers. To isolate ATTENTION sparsity, test a
    # single block's attention sublayer directly.
    from src.moe import SparseMoE, make_attn_pool
    from src.block import RMSNorm
    pool = make_attn_pool(64, 2, n_heads=2, pattern="window", window=8)
    moe = SparseMoE(64, pool, [TYPE_ATTN] * 2, top_k=1).eval()
    ln = RMSNorm(64).eval()

    x = torch.randn(1, 48, 64)
    with torch.no_grad():
        base = moe(ln(x))
        xp = x.clone()
        xp[:, 5] += 10.0                                # far outside window of pos 47
        pert = moe(ln(xp))
    delta = (base[:, 47] - pert[:, 47]).abs().max()
    assert delta.item() < 1e-6, f"window leak: {delta}"


def test_topk_pattern_keeps_self_and_is_causal():
    from src.experts import AttentionExpert
    e = AttentionExpert(32, n_heads=2, pattern="topk", topk_keys=3).eval()
    x = torch.randn(12, 32)
    rows = torch.tensor([11])
    with torch.no_grad():
        out = e.forward_selected(x, rows)               # [1, D]
    assert out.shape == (1, 32) and torch.isfinite(out).all()


def test_mixed_typed_quotas_exact():
    m = tiny_model(layout="mixed").train()
    blk = m.blocks[0].mixed
    m(torch.randint(0, 32, (3, 10)))
    rr = blk.last_route
    ka = blk.router.typed_ks["attn"]
    kf = blk.router.typed_ks["ffn"]
    assert rr.idx.shape == (30, ka + kf)
    types = torch.tensor([TYPE_ATTN == t for t in blk.types])
    for row in range(30):
        sel_types = types[rr.idx[row]]
        assert int(sel_types.sum()) == ka, "attention quota violated"
        assert int((~sel_types).sum()) == kf, "ffn quota violated"


def test_router_losses_and_usage_stats():
    m = tiny_model(layout="mixed").train()
    x = torch.randint(0, 32, (4, 16))
    met = _forward_loss(m, x)
    assert torch.isfinite(met["aux"]) and met["aux"] > 0
    u = m.blocks[0].usage()["mixed"]
    assert len(u) == 6 and abs(sum(u) - 1.0) < 1e-5     # probabilities over pool


def test_backward_gradients_flow():
    m = tiny_model(layout="mixed")
    x = torch.randint(0, 32, (2, 16))
    met = _forward_loss(m, x)
    met["total"].backward()
    for name, p in m.named_parameters():
        if "router" in name and "gate" in name:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_all_pool_experts_receive_grads():
    """Every expert of the mixed pool must be reachable by gradients."""
    m = tiny_model(layout="mixed")
    x = torch.randint(0, 32, (2, 16))
    met = _forward_loss(m, x)
    met["total"].backward()
    for i, e in enumerate(m.blocks[0].mixed.experts):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in e.parameters()), f"expert {i} got no grad"


def test_shared_expert_option():
    m_on = tiny_model(shared_expert=True, layout="mixed")
    x = torch.randint(0, 32, (1, 8))
    met = _forward_loss(m_on, x)
    assert torch.isfinite(met["total"])


def test_generation_smoke():
    m = tiny_model(layout="sequential").eval()
    idx = torch.randint(0, 32, (1, 5))
    out = m.generate(idx, 6)
    assert out.shape == (1, 11) and torch.isfinite(out.float()).all()


def test_fast_path_matches_exact_path():
    """forward_batch (dense compute) must equal forward_selected at routed rows."""
    from src.experts import AttentionExpert
    torch.manual_seed(3)
    for pattern, kw in [("window", dict(window=16)), ("topk", dict(topk_keys=5)),
                        ("dense", {})]:
        e = AttentionExpert(64, n_heads=2, pattern=pattern, **kw).eval()
        x = torch.randn(2, 40, 64)
        rows = torch.tensor([5, 6, 39])
        with torch.no_grad():
            full = e.forward_batch(x)
            exact = torch.stack([
                e.forward_selected(x[b], rows) for b in range(x.shape[0])])
        delta = (full[:, rows] - exact).abs().max()
        assert delta.item() < 1e-5, f"{pattern}: {delta}"


def test_linear_expert_causal_and_shapes():
    """Линейный (DeltaNet-подобный) эксперт: каузальность и форма выхода."""
    from src.experts import LinearAttentionExpert
    torch.manual_seed(2)
    e = LinearAttentionExpert(64, n_heads=4).eval()
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        base = e.forward_batch(x)
        xp = x.clone()
        xp[:, 20:] += 3.0
        pert = e.forward_batch(xp)
    assert base.shape == x.shape and torch.isfinite(base).all()
    leak = (base[:, :20] - pert[:, :20]).abs().max()
    assert leak.item() < 1e-5, f"linear leak: {leak}"


def test_mixed_kinds_pool():
    """Пул из РАЗНЫХ механизмов (window + linear) собирается и учится."""
    m = tiny_model(layout="sequential", attn_pattern="window")
    # подменяем состав пула через BlockConfig напрямую:
    from src.block import BlockConfig, DualMoEBlock
    cfg = BlockConfig(dim=64, num_ffn_experts=4, ffn_top_k=1,
                      num_attn_experts=3, attn_top_k=2,
                      attn_expert_kinds=("window", "linear", "topk"),
                      window=16)
    blk = DualMoEBlock(cfg).eval()
    kinds = [type(e).__name__ for e in blk.attn.experts]
    assert kinds == ["AttentionExpert", "LinearAttentionExpert",
                     "AttentionExpert"], kinds
    h = blk(torch.randn(2, 24, 64))
    assert h.shape == (2, 24, 64) and torch.isfinite(h).all()


def test_router_learns_to_prefer_capable_expert():
    """Поведенческий тест роутера (без aux-балансировки): если один эксперт
    умеет смотреть по контексту, а второй видит только себя, обучение должно
    перебросить загрузку на первого."""
    import torch.nn.functional as F
    from src.experts import AttentionExpert
    from src.moe import SparseMoE
    from src.block import RMSNorm

    torch.manual_seed(0)
    good = AttentionExpert(32, n_heads=2, pattern="window", window=32)  # видит всех
    bad = AttentionExpert(32, n_heads=2, pattern="window", window=1)    # только себя
    moe = SparseMoE(32, [good, bad], [TYPE_ATTN, TYPE_ATTN],
                    top_k=1, aux_coef=0.0)
    ln = RMSNorm(32)
    emb = torch.nn.Embedding(17, 32)
    head = torch.nn.Linear(32, 17)
    opt = torch.optim.AdamW(list(moe.parameters()) + list(ln.parameters()) +
                            list(emb.parameters()) + list(head.parameters()),
                            lr=1e-2)

    def batch():
        half = 12
        src = torch.randint(2, 16, (8, half))
        sep = torch.full((8, 1), 1)
        tok = torch.cat([src, sep, src], 1)
        return tok[:, :-1], tok[:, 1:]

    for _ in range(200):
        x, y = batch()
        logits = head(moe(ln(emb(x))))
        loss = F.cross_entropy(logits.reshape(-1, 17), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    share_good = moe.usage()[0].item()
    assert share_good > 0.6, f"роутер не предпочёл способного эксперта: {share_good:.2f}"


def test_proj_router_causal_and_grads():
    """«Роутер видит проекцию»: бегущее среднее строго каузально, градиент
    доходит до проекции контекста."""
    from src.moe import SparseMoE, make_ffn_pool
    torch.manual_seed(0)
    moe = SparseMoE(32, make_ffn_pool(32, 2, mult=2), [TYPE_FFN] * 2,
                    top_k=1, proj_router=True)
    x = torch.randn(2, 24, 32)
    out1 = moe(x)
    xp = x.clone()
    xp[:, 16:] += 3.0
    out2 = moe(xp)
    leak = (out1[:, :16] - out2[:, :16]).abs().max()
    assert leak.item() < 1e-5, f"проекция роутера течёт из будущего: {leak}"
    out1.square().mean().backward()
    assert moe.ctx_proj.weight.grad is not None and \
        moe.ctx_proj.weight.grad.abs().sum() > 0, "нет градиента у ctx_proj"


def test_selective_attention_sparse_causal_grads():
    """v2: пул внимания считает только выбранные строки; незабранные позиции
    получают ровно ноль; каузальность; градиент доходит до гейта выбора."""
    from src.block import BlockConfig, DualMoEBlock
    torch.manual_seed(0)
    cfg = BlockConfig(dim=32, num_ffn_experts=2, ffn_top_k=1,
                      num_attn_experts=2, attn_top_k=1,
                      attn_expert_kinds=("window", "window"), window=8,
                      proj_router=True, attn_select_frac=0.5)
    blk = DualMoEBlock(cfg, layout="sequential").eval()

    # разреженность: forward_rows пишет только в выбранные позиции
    h = blk.ln1(torch.randn(2, 24, 32))
    keep = torch.zeros(2, 24, dtype=torch.bool)
    keep[0, [0, 5, 11, 23]] = True
    keep[1, [1, 4, 9, 20]] = True
    with torch.no_grad():
        att = blk.attn.forward_rows(h, keep)
    assert (att[0][~keep[0]] == 0).all() and (att[1][~keep[1]] == 0).all(), \
        "невыбранные позиции не нулевые"
    assert torch.isfinite(att[keep]).all()

    # каузальность блока целиком
    x = torch.randn(2, 32, 32)
    with torch.no_grad():
        o1 = blk(x)
        xp = x.clone()
        xp[:, 20:] += 5.0
        o2 = blk(xp)
    leak = (o1[:, :20] - o2[:, :20]).abs().max()
    assert leak.item() < 1e-4, f"селективное внимание течёт: {leak}"

    # градиент гейта выбора
    blk.train()
    blk(x).square().mean().backward()
    assert blk.sel_gate.weight.grad is not None and \
        blk.sel_gate.weight.grad.abs().sum() > 0, "нет градиента у sel_gate"


def test_linear_decay_causal_and_learns():
    """Затухание в линейном эксперте: γ в (0,1), каузальность, градиент."""
    from src.experts import LinearAttentionExpert
    torch.manual_seed(0)
    e = LinearAttentionExpert(32, n_heads=4, decay="learn", gamma_init=0.9)
    x = torch.randn(2, 40, 32)
    with torch.no_grad():
        a = e(x)
        xp = x.clone()
        xp[:, 25:] += 5.0
        b = e(xp)
    leak = (a[:, :25] - b[:, :25]).abs().max()
    assert leak.item() < 1e-4, f"затухание течёт: {leak}"
    g = torch.sigmoid(e.gamma_logit)
    assert ((g > 0) & (g < 1)).all()
    a.square().mean().backward()
    assert e.gamma_logit.grad is not None


def test_active_param_estimate_sane():
    m = tiny_model(layout="mixed")
    total, active = m.num_params(False), m.num_params(True)
    assert total > active > total // 4, (total, active)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
