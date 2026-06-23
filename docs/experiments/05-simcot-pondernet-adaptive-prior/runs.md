# 05-simcot-pondernet-adaptive-prior — Runs

See [experiment.md](experiment.md) for what's being tested.

> Re-validated 2026-06-23 (validation split, 500 ex, greedy). Only ep4/ep5 survived (ep1–ep3
> deleted). Accuracy is flat vs the re-validated baseline; the robust gain is difficulty tracking.

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| `04/g0.05-gm3.0-ep5` (baseline) | global `geom_mean=3.0` | 40.80% @ thr0.5 (val, n=500) | 4.060 | reference | reused from [exp 04](../04-simcot-pondernet-gammasweep/runs.md); Spearman(steps,#expr)=+0.553 (val) |
| `perinstance-g0.05-b1.5-ep5` | affine prior α=1, β=1.5, γ=0.05 | **40.80% @ thr0.5 (ep4, val)** | **4.276** | complete | no accuracy win vs baseline on validation; robust gain is difficulty tracking: **Spearman +0.662** (ep4) vs +0.553 baseline. ep4 (`_best_epoch`)/ep5 survived; prior test "+0.29pp acc" was a leakage artifact |

`perinstance` re-validated: ep4 40.80% / 4.276 @ thr0.5, 39.80% / 5.232 @ thr0.8; ep5 41.00% / 4.282 @ thr0.5, 40.60% / 5.244 @ thr0.8 (val, n=500). Spearman(steps,#expr) ep4/ep5 = +0.662 / +0.663 @ thr0.5. Prior test numbers (biased): ep4 thr0.8 40.49%, Spearman thr0.5 0.650.

_Smoke runs (`smoke-adaptive`, `smoke-affine`) were throwaway wiring checks on the 3060, not logged here._
