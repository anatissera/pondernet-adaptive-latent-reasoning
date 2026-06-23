# 05-simcot-pondernet-adaptive-prior — Runs

See [experiment.md](experiment.md) for what's being tested.

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| `04/g0.05-gm3.0-ep5` (baseline) | global `geom_mean=3.0` | 40.2% @ 5.38 (thr0.8) | 5.38 | reference | reused from [exp 04](../04-simcot-pondernet-gammasweep/runs.md); Spearman(steps,#expr)=+0.58 |
| `perinstance-g0.05-b1.5-ep5` | affine prior α=1, β=1.5, γ=0.05 | **40.49% @ thr0.8 (ep4)** | **5.262** | complete | beats baseline on all 3 metrics: +0.29pp acc, −0.12 avg_steps, Spearman +0.650 (baseline +0.58); ep4/step 3112 wins; ep5 drops to 40.11% |

_Smoke runs (`smoke-adaptive`, `smoke-affine`) were throwaway wiring checks on the 3060, not logged here._
