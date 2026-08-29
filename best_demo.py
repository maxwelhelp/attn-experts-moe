# -*- coding: utf-8 -*-
"""Демо лучшей архитектуры (src/best.py, hybrid100): короткий прогон на
разнородной задаче (копирование + слежение за модой).

    python3 best_demo.py            # ~2-3 мин на GPU, ~5 на CPU
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from src.best import make_hybrid100                      # noqa: E402
from src.model import TinyCausalLM                       # noqa: E402
from exp_mixedtask import make_batch                     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--seq", type=int, default=96)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    vocab_hi = 31
    cfg, kinds = make_hybrid100(dim=args.dim, num_layers=args.layers,
                                vocab_size=vocab_hi + 2,
                                max_seq_len=args.seq)
    model = TinyCausalLM(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    total, active = model.num_params(False), model.num_params(True)
    print(f"гибрид100: механизмы={kinds} слоёв={args.layers} dim={args.dim}")
    print(f"параметров: {total/1e6:.2f}M всего, {active/1e6:.2f}M активных "
          f"на токен ({total/active:.1f}x экономия)")
    for step in range(1, args.steps + 1):
        x, y, _ = make_batch(args.bsz, args.seq, vocab_hi, args.device)
        _, met = model(x, y)
        (met["total"] / x.numel()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % (args.steps // 5 or 1) == 0 or step == 1:
            print(f"  шаг {step:4d}  ce {met['ce'].item():.3f}")

    # что выучилось: гаммы линейных экспертов и загрузка механизмов
    for i, blk in enumerate(model.blocks):
        for e in blk.attn.experts:
            if getattr(e, "decay_mode", "none") == "learn":
                print(f"слой {i}: выученные γ = {e.gamma_values()}")
    use = {}
    for blk in model.blocks:
        for m in blk.modules():
            lr = getattr(m, "last_route", None)
            if lr is not None:
                flat = lr.idx.reshape(-1)                       # [N*K]
                for j, name in enumerate(kinds):
                    share = (flat == j).float().mean().item()
                    use[name] = use.get(name, 0.0) + share
    n_blk = args.layers
    print("загрузка механизмов роутером (доля на токен):")
    for name, v in use.items():
        print(f"  {name:6s} {v/n_blk:.0%}")


if __name__ == "__main__":
    main()
