# 08-simcot-pondernet-gamma-frontier — Runs

See [experiment.md](experiment.md) for what's being tested.

> **Baseline (threshold-only frontier, no retrain)** — exp-07 ep5 `checkpoint-3890`,
> faithful bs=1, validation n=500, greedy:
> thr0.3 → 39.2% @ 3.27 · thr0.4 → 40.4% @ 3.78 · thr0.5 → 40.8% @ 4.34 · thr0.8 → 41.0% @ 6.80.
> Knee at ~3.8 steps (thr0.4). Artifacts: `results/07-…/ep5-bs1/thr{0.3,0.4,0.5}/`.
> This experiment's training runs aim to beat that frontier (≥40% below ~3.5 steps).

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| `fullscope-adaptive-g0.10-b1.0-k12-ep5` | γ=0.10, α=1.0, β=1.0 (floor −0.5) | — | — | ❌ invalid (β=1.0) | — |
| `fullscope-adaptive-g0.10-b1.5-k12-ep5` | γ=0.10, α=1.0, β=1.5 (tighten only) | **41.2% thr0.8 · 40.6% thr0.4** | 3.10 @ thr0.4 | ✅ eval done | ep5 best; thr0.4 beats baseline (+0.2pp, −18% steps) |
| `fullscope-adaptive-g0.10-a0.6-b1.5-k12-ep5` | γ=0.10, α=0.6, β=1.5 (tail-cap) | **40.6% thr0.5 · 40.2% thr0.4** | 2.47 @ thr0.4 | ✅ eval done | ep5 best; −20% steps vs Run B, −0.4pp acc |

## Run C results — `fullscope-adaptive-g0.10-a0.6-b1.5-k12-ep5`

Faithful bs=1 eval on validation n=500, greedy. Artifacts: `results/08-.../fullscope-adaptive-g0.10-a0.6-b1.5-k12-ep5/<ep>-bs1/thr<T>/`.

| epoch | thr0.3 | thr0.4 | thr0.5 | thr0.8 |
|-------|--------|--------|--------|--------|
| ep3 | 38.8% @ 2.04 | 39.6% @ 2.46 | 40.6% @ 2.92 | 41.0% @ 4.80 |
| ep4 | 39.2% @ 2.04 | 40.0% @ 2.47 | 40.4% @ 2.94 | 40.6% @ 4.86 |
| ep5 | **39.6% @ 2.03** | **40.2% @ 2.47** | **40.6% @ 2.93** | **40.4% @ 4.85** |

**Best checkpoint: ep5.** vs Run B ep5: ~20% fewer steps at every threshold, −0.4–0.8pp accuracy.
vs exp-07 baseline: −38% steps at thr0.4 (2.47 vs 3.78), −0.2pp accuracy.

> **Run A dropped (β=1.0 mis-specified).** `β=1.0` puts the n_i=0 examples (16% of train) on
> `geom_mean_i=1.0` — the degenerate g=1 point-mass prior the code forbids (`prior_offset` "must
> be >1"). It crashed at epoch 2 with `RuntimeError: masked_scatter_: BFloat16 vs Float` in
> backward, at BOTH bs=24 and bs=16 (so it's the β, not the batch size). The "shift the floor
> earlier" idea isn't safely reachable via β; use α<1 (run C) or higher γ instead. Runs B/C
> (β=1.5) are unaffected. See [[project-masked-scatter-batch-bug]].

## Run B results — `fullscope-adaptive-g0.10-b1.5-k12-ep5`

Faithful bs=1 eval on validation n=500, greedy. Artifacts: `results/08-.../fullscope-adaptive-g0.10-b1.5-k12-ep5/<ep>-bs1/thr<T>/`.

| epoch | thr0.3 | thr0.4 | thr0.5 | thr0.8 |
|-------|--------|--------|--------|--------|
| ep3 | 39.2% @ 2.55 | 40.0% @ 3.06 | 40.2% @ 3.62 | 41.0% @ 6.17 |
| ep4 | 39.4% @ 2.56 | 40.2% @ 3.10 | 40.6% @ 3.66 | 40.8% @ 6.25 |
| ep5 | **40.0% @ 2.55** | **40.6% @ 3.10** | **41.0% @ 3.64** | **41.2% @ 6.23** |

**Best checkpoint: ep5.** Compared to exp-07 ep5 baseline (thr0.4: 40.4% @ 3.78):
- thr0.4: **+0.2pp accuracy, −18% steps** (3.10 vs 3.78) ✅
- thr0.5: **+0.2pp accuracy, −16% steps** (3.64 vs 4.34 exp-07-thr0.5)
- thr0.3: 40.0% @ 2.55 — aggressive but near-baseline accuracy at half the steps.
