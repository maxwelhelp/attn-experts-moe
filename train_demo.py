"""Smoke training on a copy task: compare MoE block designs.

Task (needs attention, not just n-gram stats): the model sees
    [src | SEP | src]
and must reproduce `src` in the second half -> has to attend back to the
first occurrence of every token. Unigram floor is ~ln(V_src); a working
attention mechanism drives CE towards 0.

Configs compared:
  sequential/independent -- JetMoE layout + our sparse AttentionExperts (your idea)
  mixed/independent      -- ONE pool with typed quotas {ffn: kf, attn: ka} per token
  sequential/projection  -- JetMoE-style projection-expert attention (baseline)

Usage: python train_demo.py [--steps 500] [--device auto] [--configs all]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.block import BlockConfig                      # noqa: E402
from src.model import ModelConfig, TinyCausalLM        # noqa: E402

SEP_ID = 0


def make_batch(bsz: int, seq_len: int, vocab_hi: int, device):
    half = seq_len // 2
    src = torch.randint(1, vocab_hi, (bsz, half), device=device)
    sep = torch.full((bsz, 1), SEP_ID, dtype=torch.long, device=device)
    tokens = torch.cat([src, sep, src], dim=1)         # [B, 2*half+1]
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def build_model(name: str, args) -> TinyCausalLM:
    kind = "projection" if name.endswith("projection") else "independent"
    layout = "mixed" if name.startswith("mixed") else "sequential"
    # window must reach across the separator (copy distance ~= seq_len//2)
    window = args.window if args.window else args.seq_len // 2 + 8
    blk = BlockConfig(
        dim=args.d_model,
        num_ffn_experts=args.ffn_experts, ffn_top_k=args.k_ffn,
        num_attn_experts=args.attn_experts, attn_top_k=args.k_attn if layout == "sequential" else 1,
        attn_kind=kind,
        attn_heads_per_expert=args.heads_per_expert,
        attn_expert_kinds=tuple(args.kinds.split(",")) if args.kinds else None,
        attn_pattern=args.pattern, window=window,
        attn_topk_keys=args.topk_keys,
        proj_attn_heads=args.proj_heads,
        mixed_ks={"ffn": args.k_ffn, "attn": args.k_attn} if layout == "mixed" else None,
        renorm_typed=args.renorm_typed,
    )
    cfg = ModelConfig(vocab_size=args.vocab_hi + 2, max_seq_len=args.seq_len * 4,
                      num_layers=args.layers, block=blk, layouts=(layout,))
    torch.manual_seed(args.seed)                       # same init across configs
    return TinyCausalLM(cfg).to(args.device)


@torch.no_grad()
def eval_loss(model, args, iters: int = 20):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = make_batch(args.batch, args.seq_len, args.vocab_hi, model.device)
        _, m = model(x, y)
        tot += m["ce"].item()
    model.train()
    return tot / iters


@torch.no_grad()
def usage_str(model):
    parts = []
    for i, blk in enumerate(model.blocks):
        for key, u in blk.usage().items():
            u = ",".join(f"{x:.2f}" for x in u)
            parts.append(f"L{i}/{key}:[{u}]")
    return " ".join(parts)


def train_config(name: str, args) -> dict:
    model = build_model(name, args)
    total, active = model.num_params(False), model.num_params(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(args.batch, args.seq_len, args.vocab_hi, model.device)
        _, m = model(x, y)
        opt.zero_grad(set_to_none=True)
        m["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            history.append({"step": step, "ce": round(m["ce"].item(), 4)})
            print(f"  [{name}] step {step:4d}  ce {m['ce'].item():7.4f}  "
                  f"aux {m['aux'].item():.3f}")

    final = eval_loss(model, args)
    dt = time.time() - t0
    floor = math.log(args.vocab_hi)          # unigram: all src tokens unpredictable
    copy_floor = floor / 2                   # if copying is solved, only the noise half remains
    result = {
        "config": name, "final_ce": round(final, 4),
        "unigram_floor": round(floor, 3),
        "copy_floor": round(copy_floor, 3),
        "params_total_M": round(total / 1e6, 3),
        "params_active_per_token_M": round(active / 1e6, 3),
        "train_seconds": round(dt, 1),
        "history": history,
    }
    print(f"  [{name}] DONE  final_ce={final:.4f}  (unigram≈{floor:.3f}, "
          f"copy-solved≈{copy_floor:.2f})  params {total/1e6:.2f}M / active {active/1e6:.2f}M  {dt:.1f}s")
    if args.verbose:
        print("   usage:", usage_str(model))
    return result


CONFIGS = ["sequential_independent", "mixed_independent", "sequential_projection"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab-hi", type=int, default=16, help="copy alphabet size")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn-experts", type=int, default=8)
    p.add_argument("--k-ffn", type=int, default=2)
    p.add_argument("--attn-experts", type=int, default=4)
    p.add_argument("--k-attn", type=int, default=2)
    p.add_argument("--heads-per-expert", type=int, default=2)
    p.add_argument("--pattern", default="window", choices=["dense", "window", "topk"])
    p.add_argument("--kinds", default=None,
                   help="состав пула внимания через запятую: window,topk,linear")
    p.add_argument("--window", type=int, default=None,
                   help="sliding window; default: seq_len//2 + 8 (copy-reachable)")
    p.add_argument("--topk-keys", type=int, default=16)
    p.add_argument("--proj-heads", type=int, default=4)
    p.add_argument("--renorm-typed", action="store_true",
                   help="mixed pool: renormalise gate softmax over selected union")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.add_argument("--configs", default="all", help="comma list or 'all'")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    args.device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    names = CONFIGS if args.configs == "all" else args.configs.split(",")

    print(f"device={args.device}  steps={args.steps}  seq={args.seq_len}  "
          f"d={args.d_model}  layers={args.layers}")
    results = [train_config(n, args) for n in names]

    out = Path(__file__).parent / "results" / "demo_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")
    print("\n=== summary ===")
    for r in results:
        print(f"{r['config']:24s} ce={r['final_ce']:8.4f}  "
              f"total={r['params_total_M']:.2f}M  active/token={r['params_active_per_token_M']:.2f}M")


if __name__ == "__main__":
    main()
