"""Тесты иерархического «дёшево -> уточнение» внимания."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hier import HierConfig, HierRefineAttention  # noqa: E402


def make(T=128, **kw):
    torch.manual_seed(0)
    cfg = HierConfig(dim=64, n_heads=4, block_size=16, top_blocks=3,
                     loop_iters=kw.pop("loop_iters", 2))
    return HierRefineAttention(cfg).eval(), kw


def test_shapes_and_finite():
    m, _ = make()
    x = torch.randn(2, 100, 64)          # T не кратен блоку -- паддинг внутри
    out = m(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_selected_fraction_respects_budget():
    torch.manual_seed(0)
    cfg = HierConfig(dim=64, n_heads=4, block_size=16, top_blocks=3, loop_iters=1)
    mm = HierRefineAttention(cfg).eval()
    nb = 8
    budget = (3 + 1) / nb                # top_blocks + собственный блок
    x = torch.randn(2, 8 * 16, 64)
    mm(x)
    frac = mm.last_selected_fraction
    assert frac <= budget + 1e-9, f"{frac} > {budget}"


def test_causality_strict():
    m, _ = make(loop_iters=2)
    x = torch.randn(2, 96, 64)
    with torch.no_grad():
        base = m(x)
        xp = x.clone()
        xp[:, 70:] += 5.0
        pert = m(xp)
    leak = (base[:, :70] - pert[:, :70]).abs().max()
    assert leak.item() < 1e-5, f"каузальность нарушена: {leak}"


def test_fixed_v_loop_changes_output():
    """Петля с фиксированным V: итерации уточнения реально меняют выход."""
    torch.manual_seed(0)
    cfg = HierConfig(dim=64, n_heads=4, block_size=16, top_blocks=3, loop_iters=3)
    m = HierRefineAttention(cfg)
    x = torch.randn(2, 64, 64)

    outs = []
    k = m.wk(x)
    v = m.wv(x)
    mask, _ = m._gate_mask(x)
    with torch.no_grad():
        acc = 0
        for it in range(3):
            acc = acc + m._attend(m.wq[it](x), k, v, mask)
            outs.append(acc.clone())
    d12 = (outs[1] - outs[0]).abs().max().item()
    d23 = (outs[2] - outs[1]).abs().max().item()
    assert d12 > 1e-4 and d23 > 1e-4, "итерации уточнения не меняют выход"
    assert torch.isfinite(outs[-1]).all()


def test_backward_flows():
    m, _ = make()
    x = torch.randn(2, 64, 64, requires_grad=True)
    loss = m(x).square().mean()
    loss.backward()
    g = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(gi is not None for gi in g), "не у всех параметров есть градиент"


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
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
