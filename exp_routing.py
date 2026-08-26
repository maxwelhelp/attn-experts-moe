"""Диагностика роутера: мешает ли balancing-loss правильной специализации?

Гипотеза: aux-потеря (Switch-style) тянет загрузку к равномерной, поэтому на
задаче, где почти все запросы должны идти к оконным экспертам, роутер
«неправильно» держит линейные эксперты занятыми.

Эксперимент: копирование через разделитель, пул внимания
[окно, окно, linear, linear], top-2 из 4. Два запуска: aux=0.01 и aux=0.
Метрика: доля назначений, уходящих к линейным экспертам + качество.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.block import BlockConfig                       # noqa: E402
from src.experts import LinearAttentionExpert           # noqa: E402
from src.model import ModelConfig, TinyCausalLM         # noqa: E402
from train_demo import make_batch                       # noqa: E402


def linear_share(model) -> float:
    """Доля назначений внимания, уходящих к линейным экспертам."""
    tot = lin = 0
    for blk in model.blocks:
        rr = blk.attn.last_route
        for i, e in enumerate(blk.attn.experts):
            c = int((rr.idx == i).sum())
            tot += c
            if isinstance(e, LinearAttentionExpert):
                lin += c
    return lin / max(tot, 1)


def run(tag: str, args) -> dict:
    torch.manual_seed(args.seed)
    blk = BlockConfig(
        dim=args.d_model, num_ffn_experts=8, ffn_top_k=2,
        num_attn_experts=4, attn_top_k=2, attn_kind="independent",
        attn_heads_per_expert=4,
        attn_expert_kinds=("window", "window", "linear", "linear"),
        window=args.seq_len // 2 + 8,
        aux_coef=args.aux)
    cfg = ModelConfig(vocab_size=args.vocab_hi + 2, max_seq_len=args.seq_len * 2,
                      num_layers=args.layers, block=blk, layouts=("sequential",))
    model = TinyCausalLM(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    hist, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x, y = make_batch(args.batch, args.seq_len, args.vocab_hi, model.device)
        _, met = model(x, y)
        opt.zero_grad(set_to_none=True)
        met["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            rec = {"step": step, "ce": round(met["ce"].item(), 3),
                   "linear_share": round(linear_share(model), 3)}
            hist.append(rec)
            print(f"  [{tag}] шаг {step:4d}  ce {rec['ce']:.3f}  "
                  f"доля линейных {rec['linear_share']:.2f}", flush=True)

    res = {"tag": tag, "aux": args.aux, "final_ce": round(met["ce"].item(), 3),
           "final_linear_share": round(linear_share(model), 3),
           "seconds": round(time.time() - t0), "history": hist}
    print(f"  [{tag}] ГОТОВО ce={res['final_ce']} "
          f"линейных={res['final_linear_share']:.0%}\n", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--vocab-hi", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    out = []
    for tag, aux in [("с балансировкой (aux=0.01)", 0.01),
                     ("без балансировки (aux=0)", 0.0)]:
        print(f"=== {tag} ===", flush=True)
        ns = argparse.Namespace(**vars(args), aux=aux)
        out.append(run(tag, ns))

    (Path(__file__).parent / "results" / "routing_diag.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("сохранено -> results/routing_diag.json")


if __name__ == "__main__":
    main()
