"""Иерархическое «дёшево -> выбор -> уточнение» против плотного внимания.

Задача: копирование через разделитель на T=256 (дистанция ~129 -- за пределами
любого разумного окна). FFN в обеих моделях одинаковый одиночный, отличается
только механизм внимания:

  * dense -- обычное каузальное внимание, все ко всем;
  * hier  -- лендмарки (дешёвая проекция контекста) -> поточный гейт выбирает
             top_blocks прошлых блоков -> точное внимание только к ним
             (+ петля уточнения с фиксированным V).

Метрики: качество (CE) и selected_fraction -- средняя доля блоков ключей,
которые гейт реально открыл (прокси экономии вычислений; стеночное время
на этом прототипе НЕ экономится -- маска подаётся в плотный SDPA).
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.block import BlockConfig                       # noqa: E402
from src.model import ModelConfig, TinyCausalLM         # noqa: E402


def make_copy(bsz, seq, vocab_hi, device):
    half = seq // 2 - 1
    src = torch.randint(2, vocab_hi, (bsz, half))
    sep = torch.full((bsz, 1), 1)
    tok = torch.cat([src, sep, src], 1)
    x, y = tok[:, :-1].to(device), tok[:, 1:].clone().to(device)
    return x, y


def run(tag, kind, args):
    torch.manual_seed(args.seed)
    blk_kwargs = dict(dim=args.d_model, num_ffn_experts=1, ffn_top_k=1,
                      dropout=0.0)
    if kind == "dense":
        cfg = ModelConfig(vocab_size=args.vocab_hi + 2,
                          max_seq_len=args.seq_len * 2, num_layers=args.layers,
                          block=BlockConfig(**blk_kwargs), layouts=("dense",))
        model = TinyCausalLM(cfg).to(args.device)
        frac_fn = lambda: 1.0
    else:
        blk = BlockConfig(attn_kind="hier",
                          hier_block_size=args.hier_block_size,
                          hier_top_blocks=args.hier_top_blocks,
                          hier_loop_iters=args.hier_loop_iters, **blk_kwargs)
        cfg = ModelConfig(vocab_size=args.vocab_hi + 2,
                          max_seq_len=args.seq_len * 2, num_layers=args.layers,
                          block=blk, layouts=("sequential",))
        model = TinyCausalLM(cfg).to(args.device)

        fracs = []
        def frac_fn():
            fs = [b.attn.last_selected_fraction for b in model.blocks]
            return sum(fs) / len(fs)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_par = model.num_params()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = make_copy(args.batch, args.seq_len, args.vocab_hi, args.device)
        _, met = model(x, y)
        opt.zero_grad(set_to_none=True)
        met["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"  [{tag}] шаг {step:4d}  ce {met['ce'].item():.3f} "
                  f" открытых блоков {frac_fn():.0%}", flush=True)
    dt = time.time() - t0
    res = {"tag": tag, "ce": round(met["ce"].item(), 3),
           "frac": round(frac_fn(), 3), "params": n_par, "sec": round(dt)}
    print(f"  [{tag}] ГОТОВО ce={res['ce']} открытых={res['frac']:.0%} "
          f"параметров={n_par/1e6:.2f}M {dt:.0f}с\n", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--vocab-hi", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--hier-block-size", type=int, default=32)
    p.add_argument("--hier-top-blocks", type=int, default=3)
    p.add_argument("--hier-loop-iters", type=int, default=2)
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    out = []
    out.append(run("плотное", "dense", args))
    out.append(run("иерархическое", "hier", args))

    import json
    Path("results").mkdir(exist_ok=True)
    Path("results/hier_vs_dense.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
