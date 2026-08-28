# Attn-Experts MoE: routing attention mechanisms per token

*(русская версия: [README.md](README.md))*

A research scaffold built around one idea (due to the project author):
**for every token, assemble its own mixture of attentions from a pool of
different sparse mechanisms** — the router decides which mixing type this
token needs: a precise window, top-k jumps, linear memory, or budget
projections. Three developments are built on top of this core: a
"cheap → select → refine" hierarchy, a router that reads a cheap context
projection, and selective expensive refinement.

## TL;DR headline results

| experiment | result |
|---|---|
| Copy task, pool of 2 windows + 2 linears, projection-aware router | **1.90** vs 2.00 for the homogeneous pool |
| Mixed workload (copy + mode tracking), projection-aware router | **2.136** — best on BOTH subtasks at once |
| Hierarchical attention on copy T=128 | **2.52** with only **33%** of keys opened (dense: 2.59 / 100%) |
| v2: 60% refinement budget | **2.05** — better quality with 40% of attention saved |
| Training start vs JetMoE-style | noticeably faster at equal budget |
| Active params per token | ~3.4× fewer than the full pool |

Honest verdict: as a drop-in replacement for plain attention today — no
(dense attention is more accurate at equal budget and faster at toy scale).
The value is in the confirmed specialization mechanics and in compute
savings that convert into wins at scale with block-sparse kernels.

## How the core "author's attention" works

Not a single module but a way of assembling a layer. Data flow:

1. **Expert pool** (`src/experts.py`, factories `make_attn_pool` /
   `make_ffn_pool`): several independent attention blocks of different
   kinds — `AttentionExpert` with `window` / `topk` / `dense` patterns
   (own Wq/Wk/Wv/Wo), `LinearAttentionExpert` (linear attention, DeltaNet
   relative: state S_t = Σ φk⊗v, output φq·S_t, denominator-normalized),
   and `ProjectionMoEAttention` (JetMoE-style shared kernel).
2. **Router** (`src/router.py`, `TopKRouter`): scores all experts per
   token and picks top-k; two modes — plain uniform top-k and **typed
   quotas** (`typed_ks`: exactly k_ffn + k_attn), plus Switch-style
   balancing aux loss and z-loss.
3. **Dispatcher** (`src/moe.py`, `SparseMoE`): assignments are sorted by
   expert id forming contiguous segments — one GEMM per expert, one
   scatter back, one host sync per layer.
4. **Assembly**: selected expert outputs are weighted by gate values and
   added to the residual. Blocks are `DualMoEBlock` (`src/block.py`):
   `sequential` (attention pool + FFN pool) or `mixed` (one pool with
   typed quotas).

## Development 1: mixture of mechanisms and its main pitfall

The first `LinearAttentionExpert` had unnormalized state: outputs blew up
to ~7310 vs ~0.4 for window experts and drowned the whole mixture
(diagnostics in `exp_routing.py`). Fixed with denominator normalization
(Katharopoulos/RetNet style). After the fix the heterogeneous pool beats
the homogeneous one (1.90 vs 2.00).

Router diagnostics also revealed a **local routing equilibrium**: without
external hints it keeps ~50% of load on each type even when one type is
objectively better (and it is not the aux loss — verified by disabling).
A behavioral test pins the boundary: with a large immediate capability gap
the router does learn preference.

## Development 2 (v1): the router sees a cheap context projection

Implementation of the discussion idea: give the gate a coarse snapshot of
the whole past. A causal running mean of projected tokens is concatenated
to token features at the gate input (`proj_router`).

Mixed-workload benchmark `exp_mixedtask.py` — a 50/50 stream of copying
(window work) and prefix-mode tracking (linear-memory work):

| config | CE total | copy | mode | linear share |
|---|---|---|---|---|
| 4 windows | 2.344 | 2.168 | 2.556 | — |
| windows+linears | 2.391 | 2.485 | 2.380 | 56% |
| **windows+linears + projection router** | **2.136** | **2.107** | **2.126** | 47% |

The projection removes the specialization trade-off: best on both
subtasks at once. First positive answer to the "router equilibrium" risk.

## Development 3: hierarchical "cheap → select → refine"

Prototype `src/hier.py`: block landmarks (cheap projection) → per-token
strictly causal gate picks top-k past blocks → exact attention to them
only + a fixed-V refinement loop (K/V once, Q recomputed). Tests catch
three future-leak channels through gates.

Copy T=128 (distance 65), 600 steps: hierarchical **2.52 at 33% keys**
vs dense 2.59/100%. Caveat: wall-clock is not saved yet — the mask goes
through dense SDPA; a block-sparse kernel is required.

## Development 4 (v2): expensive refinement only where selected

A block gate (`attn_select_frac`) keeps a fraction of tokens by a
"growing budget" rule (a token is kept if it is in the top fraction of
ITS OWN prefix — a global top-k would depend on future scores). The pool
computes only the selected rows: window/top-k experts take the exact
`forward_selected` path (real FLOPs saving ∝ fraction), linears scan
everything anyway (they are cheap). A soft sigmoid weight carries
gradient into the selection gate.

| config | refinement | CE | copy | mode |
|---|---|---|---|---|
| windows | 100% | 2.344 | 2.168 | 2.556 |
| mixed | 100% | 2.391 | 2.485 | 2.380 |
| **v1: + projection router** | 100% | 2.136 | 2.107 | 2.126 |
| **v2: + 60% budget** | **60%** | **2.050** | 2.262 | **1.876** |
| v2: 35% budget | 35% | 2.390 | 2.651 | 2.167 |

Key finding: **a 60% budget improves overall quality while saving 40% of
attention** — constrained refinement acts as regularization and forces
the router to be pickier. A too-tight budget (35%) starts cutting into
the flesh.

## Honest limitations

* Toy scale (d≤128, ≤4 layers, one-two seeds); wall-clock on a shared GPU
  is not indicative.
* Compute savings are structural/analytic so far: without block-sparse
  kernels wall-clock does not win.
* Router equilibrium is softened by the projection, not solved in general.
* The linear expert is a simplified DeltaNet without decay/gating.

## Related work and where the novelty is

Occupied niches: MoA/SwitchHead/JetMoE (MoE inside attention);
NSA/MoBA/Routing Transformer/Landmark/Informer (hierarchical block
selection); Jamba/Zamba/Samba/Griffin/Hymba (attention+SSM hybrids with
fixed schedules); Mixture-of-Depths (depth routing).

The free intersection this code is heading towards: **heterogeneous
hierarchical attention — per-token routing BETWEEN mechanisms
(window/top-k/DeltaNet/projections), driven by a cheap context
projection**. The pieces are known individually; their combination was
not found in the literature.

## Reproduction

```bash
python3 tests/test_all.py            # 17 mechanics & behavior tests
python3 tests/test_hier.py           # 5 hierarchical attention tests
python3 train_demo.py --steps 400    # copy: homogeneous vs mixed pool
python3 train_lm.py                  # Shakespeare char-LM comparison
python3 exp_routing.py               # router equilibrium diagnostics
python3 exp_mixedtask.py             # v1: mixed workload (+ --select-frac)
python3 exp_hier.py                  # hierarchical vs dense
```

Branch `hier-refine-attention`; the base scaffold commit is on `master`.
