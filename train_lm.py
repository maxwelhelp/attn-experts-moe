"""Честное сравнение на реальном тексте: tiny Shakespeare, посимвольно.

Сравниваем при одинаковом числе шагов/токенов:
  dense        -- обычный трансформер (плотное внимание + плотный FFN)
  moe_indep    -- наша идея: эксперты внимания (скользящее окно) + FFN-эксперты
  moe_proj     -- JetMoE-стиль: эксперты-проекции + общее внимание

Для каждой модели: качество на отложенной части текста, скорость обучения,
размер и сколько параметров реально работает на каждом токене, плюс пример
сгенерированного текста.

Запуск: python3 train_lm.py [--steps 800] [--device auto]
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.block import BlockConfig                       # noqa: E402
from src.model import ModelConfig, TinyCausalLM         # noqa: E402

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")
DATA_PATH = Path(__file__).parent / "results" / "data" / "shakespeare.txt"
PROMPT = "First Citizen:\n"


def load_text() -> str:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print("скачиваю tiny Shakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text()


class CharData:
    def __init__(self, text: str, train_frac: float = 0.9):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        cut = int(len(ids) * train_frac)
        self.train, self.val = ids[:cut], ids[cut:]
        self.vocab_size = len(chars)

    def batch(self, split, bsz: int, seq_len: int, device):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - seq_len - 1, (bsz,))
        x = torch.stack([data[i:i + seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)


def build_model(name: str, args, vocab_size: int) -> TinyCausalLM:
    window = args.seq_len // 2 + 8          # чтобы окно достаёт через полтекста
    if name == "dense":
        blk = BlockConfig(dim=args.d_model)
        layouts = ("dense",)
    elif name == "dense_big":
        # плотный трансформер, уравненный по числу РАБОТАЮЩИХ параметров
        # с MoE-версией (~2.2M на токен) -- честное сравнение "за те же деньги"
        blk = BlockConfig(dim=args.dense_big_dim)
        layouts = ("dense",)
    elif name == "moe_indep":
        blk = BlockConfig(
            dim=args.d_model, num_ffn_experts=8, ffn_top_k=2,
            num_attn_experts=4, attn_top_k=2, attn_kind="independent",
            attn_heads_per_expert=2, attn_pattern="window", window=window,
            proj_attn_heads=4)
        layouts = ("sequential",)
    elif name == "moe_proj":
        blk = BlockConfig(
            dim=args.d_model, num_ffn_experts=8, ffn_top_k=2,
            num_attn_experts=4, attn_top_k=2, attn_kind="projection",
            proj_attn_heads=8, window=window)
        layouts = ("sequential",)
    else:
        raise ValueError(name)
    cfg = ModelConfig(vocab_size=vocab_size, max_seq_len=args.seq_len * 2,
                      num_layers=args.layers, block=blk, layouts=layouts)
    torch.manual_seed(args.seed)
    return TinyCausalLM(cfg).to(args.device)


def model_dim(args, name: str) -> int:
    return args.dense_big_dim if name == "dense_big" else args.d_model


@torch.no_grad()
def eval_val(model, data, args, iters: int = 30) -> float:
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = data.batch("val", args.batch, args.seq_len, args.device)
        _, m = model(x, y)
        tot += m["ce"].item()
    model.train()
    return tot / iters


@torch.no_grad()
def sample(model, data, n_new: int = 300, temp: float = 0.8) -> str:
    model.eval()
    idx = torch.tensor([[data.stoi[c] for c in PROMPT]], device=model.device)
    for _ in range(n_new):
        ctx = idx[:, -model.cfg.max_seq_len:]
        logits, _ = model(ctx)
        probs = torch.softmax(logits[:, -1] / temp, -1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], 1)
    model.train()
    return "".join(data.itos[i] for i in idx[0].tolist())


def train_one(name: str, data: CharData, args) -> dict:
    model = build_model(name, args, data.vocab_size)
    total = model.num_params(False)
    active = model.num_params(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    curve, t0 = [], time.time()
    tokens = 0
    for step in range(1, args.steps + 1):
        x, y = data.batch("train", args.batch, args.seq_len, args.device)
        _, m = model(x, y)
        opt.zero_grad(set_to_none=True)
        m["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tokens += x.numel()
        if step % args.eval_every == 0 or step == 1:
            vloss = eval_val(model, data, args)
            curve.append({"step": step, "val": round(vloss, 4)})
            dt = time.time() - t0
            print(f"  [{name}] шаг {step:5d}  ошибка на отложке {vloss:.4f}  "
                  f"{tokens / max(dt, 1e-9):,.0f} символов/сек", flush=True)

    vfinal = eval_val(model, data, args)
    secs = time.time() - t0
    text = sample(model, data)
    (Path(__file__).parent / "results" / f"sample_{name}.txt").write_text(text)
    res = {
        "config": name,
        "val_loss_final": round(vfinal, 4),
        "val_curve": curve,
        "params_total_M": round(total / 1e6, 2),
        "params_active_M": round(active / 1e6, 2),
        "train_seconds": round(secs, 1),
        "chars_per_second": round(tokens / secs),
        "perplexity": round(math.exp(vfinal), 1),
    }
    print(f"  [{name}] ГОТОВО: ошибка {vfinal:.4f} (перплексия {res['perplexity']}), "
          f"параметры {res['params_total_M']}M "
          f"(работает на токен {res['params_active_M']}M), {secs:.0f} сек\n", flush=True)
    return res


CONFIGS = ["dense", "dense_big", "moe_indep", "moe_proj"]
NAMES_RU = {
    "dense": "обычный трансформер (малый)",
    "dense_big": "обычный трансформер (равен MoE по вычислениям)",
    "moe_indep": "эксперты внимания+FFN (наша идея)",
    "moe_proj": "эксперты-проекции (JetMoE-стиль)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--dense-big-dim", type=int, default=180,
                   help="размер плотного базлайна, уравненного по активным параметрам")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--configs", default="all")
    args = p.parse_args()

    args.device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    names = CONFIGS if args.configs == "all" else args.configs.split(",")

    text = load_text()
    data = CharData(text)
    print(f"текст: {len(text):,} символов, алфавит {data.vocab_size}, "
          f"device={args.device}, шагов={args.steps}\n")

    results = []
    for name in names:
        print(f"=== обучаю: {NAMES_RU.get(name, name)} ===")
        results.append(train_one(name, data, args))

    out = Path(__file__).parent / "results" / "lm_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n================ ИТОГИ ================")
    for r in sorted(results, key=lambda r: r["val_loss_final"]):
        print(f"{NAMES_RU.get(r['config'], r['config']):38s} "
              f"ошибка {r['val_loss_final']:.3f} | перплексия {r['perplexity']:6.1f} | "
              f"{r['params_total_M']:5.2f}M пар. ({r['params_active_M']:.2f}M актив.) | "
              f"{r['chars_per_second']:>7,.0f} симв/сек")

    print("\n================ ПРИМЕРЫ ГЕНЕРАЦИИ (после подсказки 'First Citizen:') ================")
    for r in results:
        s = (Path(__file__).parent / "results" / f"sample_{r['config']}.txt").read_text()
        print(f"\n----- {NAMES_RU.get(r['config'], r['config'])} -----")
        print(s[:280])


if __name__ == "__main__":
    main()
