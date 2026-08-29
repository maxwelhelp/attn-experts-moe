# Attn-Experts MoE: routing attention mechanisms per token

*(русская версия: [README.md](README.md))*

The project's core idea (by the project author): **for every token,
assemble its own mixture of attentions from a pool of different sparse
mechanisms** — the router decides what this token needs: a precise
window, fast memory, long memory, or jumps. The project is the path from
a prototype to a working architecture, with every decision validated by
a control experiment.

## Final architecture (hybrid100) — the headline

The project's best result: **CE 1.874 vs 2.344 for dense windows — a 20%
error reduction** on a mixed workload (copy-through-separator + mode
tracking), with the best worst case across subtasks. One constructor —
`src/best.py::make_hybrid100`; demo — `best_demo.py`.

```
x ──► LayerNorm ──► ATTENTION MECHANISM POOL (top-2 of 4, routed) ──► +x
                      window 72 — precise local lookup
                      lin9      — linear attention, γ=0.9  (~10 steps)
                      lin99     — linear attention, γ=0.99 (~100 steps)
                      linL      — learnable γ (converges to ~0.985–0.99)
          the router sees [token ; causal running mean of projected
          past] — a cheap snapshot of the whole context
 then: FFN-MoE (top-2 of 8) ──► +residual     (regular block part)
```

Every decision was paid for by an experiment:

| decision | why | without it |
|---|---|---|
| top-2 of a heterogeneous pool | different tasks need different mechanisms | windows: 2.344 |
| past projection in the router | the router sees the task type, not just the token | 2.136 and a 50/50 equilibrium |
| memory horizons + learnable decay | equally strong mechanisms: "per-step / per-paragraph" memory | 1.92–1.94 |
| normalized expert outputs | unnormalized, the linear expert outputs \|7310\| vs 0.4 | the mixture dies |
| NO selection budget (100% tokens) | on this pool the budget hurts | 2.064 |

Learned γ≈0.985–0.99 in every run — the model itself prefers to forget
rather than remember forever. Active params per token ~2.1M of 7.4M
(**3.5× fewer** than the full pool). In the demo the router splits load
window 29% / lin9 49% / lin99 37% / linL 38% — visible specialization.

## Results stage by stage (same mixed workload, 400 steps)

| version | CE | copy | mode | worst |
|---|---|---|---|---|
| dense windows (start) | 2.344 | 2.168 | 2.556 | 2.556 |
| v1: old-pool router + projection | 2.136 | 2.107 | 2.126 | 2.126 |
| control: fixed win/lin schedule | 2.103 | 1.758 | 2.452 | 2.452 |
| v2: v1 + 60% budget | 2.050 | 2.262 | 1.876 | 2.262 |
| + decay (linL) | 1.924 | 1.915 | 1.825 | 1.915 |
| **full assembly hybrid100** | **1.874** | 2.034 | **1.772** | **2.034** |

## How it works (briefly)

* **Experts** (`src/experts.py`): `AttentionExpert` (window/top-k/dense
  patterns, own Wq/Wk/Wv/Wo, exact row path `forward_selected`),
  `LinearAttentionExpert` — linear attention (DeltaNet relative) with
  decay: a fast cumsum path equivalent to the recurrence to 1e-7, plus an
  honest loop; `ProjectionMoEAttention` (JetMoE style).
* **Router** (`src/router.py::TopKRouter`): top-k with typed quotas,
  Switch aux and z-loss; `proj_router` mode appends a causal running mean
  of projected past to token features (`SparseMoE._causal_ctx`).
* **Dispatcher** (`src/moe.py::SparseMoE`): sort assignments → one GEMM
  per expert, one host sync per layer; `forward_rows` runs expensive
  experts only on selected rows.
* **Block** (`src/block.py::DualMoEBlock`): attention pool + FFN pool,
  `sequential` / `mixed` layouts.

## What was tested and did NOT hold (honest)

1. **Router vs fixed schedule.** With a known task profile the schedule
   is not worse (better on copy). The router wins via robustness to an
   unknown profile (worst subtask 2.03 vs 2.45) and wins in the full
   assembly with equally strong mechanisms.
2. **Selection budget (v2)**: on the weak pool it gave quality AND
   savings, but it does **not transfer** to the strong pool — budget
   regularization turned out to be a property of the weak pool, not a
   universal law.
3. **Routing equilibrium**: without context the router sticks at 50/50
   regardless of the aux loss; the past projection removes the problem.
4. **Hierarchical prototype** (`src/hier.py`): 2.52 at 33% keys vs 2.59
   for dense — the mechanics work, but wall-clock savings need
   block-sparse kernels.

## Limitations

Toy scale (d=128, 4 layers, single seeds); savings are structural (FLOPs),
not wall-clock — a block-sparse kernel is required; the linear expert is a
simplified DeltaNet without decay gates on values; runs on a shared GPU,
so wall-clock is not meaningful.

## Related work

MoA / SwitchHead / JetMoE (MoE inside attention); NSA / MoBA / Routing
Transformer / Landmark (hierarchical block selection); Jamba / Zamba /
Samba / Griffin / Hymba (attention+SSM hybrids with fixed schedules);
Mixture-of-Depths (depth routing). The free niche this project moves
towards: **heterogeneous hierarchical attention with per-token routing
BETWEEN mechanisms driven by a cheap context projection**.

## Reproduction

```bash
python3 best_demo.py                 # the best architecture: short demo
python3 tests/test_all.py            # 19 mechanics & behavior tests
python3 tests/test_hier.py           # 5 hierarchical attention tests
python3 exp_mixedtask.py --only hybrid100                # record assembly
python3 exp_mixedtask.py --only hybrid60 --select-frac 0.6
python3 exp_mixedtask.py             # all mixed-workload configs
python3 train_demo.py --steps 400    # copy: homogeneous vs mixed pool
python3 train_lm.py                  # Shakespeare char-LM
python3 exp_routing.py               # router equilibrium diagnostics
python3 exp_hier.py                  # hierarchical vs dense
```

The best assembly is `src/best.py::make_hybrid100(dim=128, num_layers=4)`.
Main development branch — `hier-refine-attention`; base — `master`.
