"""Смешанный бенчмарк: проверка «разным работам — свои механизмы».

Два типа примеров в одном потоке (50/50):

* КОПИРОВАНИЕ через разделитель -- точный поиск на дистанции ~seq/2,
  выигрывает у точного локального внимания (окна);
* СЛЕЖЕНИЕ ЗА МОДОЙ -- y_t = самый частый токен в префиксе 1..t,
  чистая агрегация счётчиков, идеальна для накопительной памяти
  (линейное внимание), недоступна окнам.

Вопрос эксперимента: гетерогенный пул внимания (окна+линейные) и
«роутер видит дешёвую проекцию контекста» (--proj-router) -- помогают ли
они на разнородной нагрузке против однородного пула окон?

Конфигурации:
  A win4        -- 4 оконных эксперта (однородный базлайн)
  B mixed       -- 2 окна + 2 линейных
  C mixed_proj  -- то же + роутер читает проекцию контекста
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.block import BlockConfig                       # noqa: E402
from src.experts import LinearAttentionExpert           # noqa: E402
from src.model import ModelConfig, TinyCausalLM         # noqa: E402


def make_batch(bsz, seq, vocab_hi, device):
    """Половина батча -- копирование, половина -- слежение за модой."""
    n_copy = bsz // 2
    n_mode = bsz - n_copy
    xs, ys, kinds = [], [], []

    if n_copy:
        half_len = seq // 2 - 1
        src = torch.randint(2, vocab_hi, (n_copy, half_len))
        sep = torch.full((n_copy, 1), 1)
        tok = torch.cat([src, sep, src,
                         torch.zeros(n_copy, seq - (2 * half_len + 1), dtype=torch.long)],
                        dim=1)                            # длиной ровно seq
        xs.append(tok[:, :-1])
        ys.append(tok[:, 1:])
        kinds += ["copy"] * n_copy

    if n_mode:
        m = torch.randint(2, vocab_hi, (n_mode, seq))
        counts = F.one_hot(m, num_classes=vocab_hi + 2).cumsum(dim=1)
        mode = counts.argmax(dim=-1)                      # ничья -> меньший id
        xs.append(m[:, :-1])
        ys.append(mode[:, 1:])
        kinds += ["mode"] * n_mode

    x = torch.cat(xs).to(device)
    y = torch.cat(ys).to(device)
    kind = torch.tensor([0 if k == "copy" else 1 for k in kinds], device=device)
    return x, y, kind


def linear_share(model):
    tot = lin = 0
    for blk in model.blocks:
        rr = getattr(blk.attn, "last_route", None)
        if rr is None:
            continue
        for i, e in enumerate(blk.attn.experts):
            c = int((rr.idx == i).sum())
            tot += c
            if isinstance(e, LinearAttentionExpert):
                lin += c
    return lin / max(tot, 1)


def run(tag, kinds, proj, args, schedule=False):
    torch.manual_seed(args.seed)
    blk = BlockConfig(
        dim=args.d_model, num_ffn_experts=8, ffn_top_k=2,
        num_attn_experts=len(kinds), attn_top_k=2,
        attn_expert_kinds=kinds, window=args.seq_len // 2 + 8,
        proj_router=proj, attn_select_frac=getattr(args, "select_frac", 1.0))
    blk_cfg = ModelConfig(vocab_size=args.vocab_hi + 2,
                          max_seq_len=args.seq_len * 2, num_layers=args.layers,
                          block=blk, layouts=("sequential",))
    if schedule:
        # КОНТРОЛЬ: жёсткое расписание вместо роутера -- чётные слои только
        # окна, нечётные только линейные, top-2 из 2 (оба всегда активны),
        # та же проекция в роутере и тот же бюджет на слой
        w = BlockConfig(**{**vars(blk).copy(), "num_attn_experts": 2,
                           "attn_expert_kinds": ("window", "window")})
        l = BlockConfig(**{**vars(blk).copy(), "num_attn_experts": 2,
                           "attn_expert_kinds": ("linear", "linear")})
        blk_cfg = ModelConfig(vocab_size=args.vocab_hi + 2,
                              max_seq_len=args.seq_len * 2, num_layers=args.layers,
                              block=w, block_schedule=(w, l, w, l),
                              layouts=("sequential",))
    else:
        blk_cfg = ModelConfig(vocab_size=args.vocab_hi + 2,
                              max_seq_len=args.seq_len * 2, num_layers=args.layers,
                              block=blk, layouts=("sequential",))
    model = TinyCausalLM(blk_cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def subset_ce(kind_id, n_batches=4):
        """CE отдельно по копированию (0) и моде (1)."""
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for _ in range(n_batches):
                x, y, kd = make_batch(args.batch, args.seq_len, args.vocab_hi,
                                      args.device)
                logits, _ = model(x)
                sel = kd == kind_id
                if sel.any():
                    lg = logits[sel]
                    yy = y[sel]
                    tot += F.cross_entropy(
                        lg.reshape(-1, lg.shape[-1]).float(), yy.reshape(-1),
                        reduction="sum").item()
                    cnt += yy.numel()
        model.train()
        return tot / max(cnt, 1)

    hist, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x, y, _ = make_batch(args.batch, args.seq_len, args.vocab_hi, args.device)
        _, met = model(x, y)
        opt.zero_grad(set_to_none=True)
        met["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "ce": round(met["ce"].item(), 3),
                   "lin": round(linear_share(model), 2)}
            hist.append(rec)
            print(f"  [{tag}] шаг {step:4d} ce {rec['ce']:.3f} "
                  f"линейные {rec['lin']:.0%}", flush=True)

    gammas = {}
    for blk in model.blocks:
        for e in blk.attn.experts:
            if hasattr(e, "decay_mode") and e.decay_mode == "learn":
                gammas = e.gamma_values()
    res = {"tag": tag, "ce": round(met["ce"].item(), 3),
           "learned_gamma": gammas,
           "linear_share": round(linear_share(model), 3),
           "ce_copy": round(subset_ce(0), 3), "ce_mode": round(subset_ce(1), 3),
           "params": model.num_params(), "sec": round(time.time() - t0),
           "select_frac": getattr(args, "select_frac", 1.0),
           "history": hist}
    print(f"  [{tag}] ГОТОВО ce={res['ce']} копия={res['ce_copy']} "
          f"мода={res['ce_mode']} линейные={res['linear_share']:.0%}\n",
          flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--vocab-hi", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="auto")
    p.add_argument("--select-frac", type=float, default=1.0,
                   help="v2: доля токенов в дорогом уточнении внимания")
    p.add_argument("--only", nargs="*", default=None,
                   choices=["win4", "mixed", "mixed_proj", "schedule_proj",
                            "horizons", "decay_mix"],
                   help="запустить только эти конфиги")
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    plan = [("win4", ("window",) * 4, False),
            ("mixed", ("window", "window", "linear", "linear"), False),
            ("mixed_proj", ("window", "window", "linear", "linear"), True),
            ("schedule_proj", ("window", "window", "linear", "linear"), True),
            ("horizons", ("window", "lin9", "lin99", "linL"), True),
            ("decay_mix", ("window", "window", "linL", "linL"), True)]
    if args.only:
        plan = [pl for pl in plan if pl[0] in set(args.only)]

    out = []
    for tag, kinds, proj in plan:
        out.append(run(tag, kinds, proj, args, schedule=(tag == "schedule_proj")))

    Path("results").mkdir(exist_ok=True)
    Path("results/mixedtask.json").write_text(json.dumps(out, indent=2,
                                                         ensure_ascii=False))
    print("=== сводка ===")
    for r in out:
        print(f"{r['tag']:12s} ce={r['ce']:.3f} копия={r['ce_copy']:.3f} "
              f"мода={r['ce_mode']:.3f} линейные={r['linear_share']:.0%} "
              f"γ={r.get('learned_gamma', '-')}")


if __name__ == "__main__":
    main()
