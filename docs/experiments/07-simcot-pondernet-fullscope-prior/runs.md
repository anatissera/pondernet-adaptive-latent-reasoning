# 07-simcot-pondernet-fullscope-prior — Runs

See [experiment.md](experiment.md) for what's being tested.

> Re-validated 2026-06-23 (validation split, 500 ex, greedy). Only ep5 survived (ep1–ep4
> deleted). Accuracy sits at the 40.80% greedy validation baseline; the result is the project's
> strongest difficulty tracking. The old record ep3 Spearman = 0.684 (test) cannot be re-validated.

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| `fullscope-adaptive-g0.05-b1.5-k12-ep5` | full scope + adaptive prior α=1 β=1.5 γ=0.05 K_max=12 | 41.00% @ thr0.5/thr0.8 ep5 (val, n=500) | 4.336 (thr0.5) | ✅ done | [detail](fullscope-adaptive-g0.05-b1.5-k12-ep5.md) |

ep5 re-validated: 41.00% / 4.336 @ thr0.5, 41.00% / 6.804 @ thr0.8, 40.00% / 8.192 @ thr0.9 (val, n=500). Spearman(steps,#expr) ep5 = +0.675 / +0.671 / +0.662 @ thr0.5/0.8/0.9. Prior test (biased): ep5 thr0.8 40.26%, thr0.5 40.11%, Spearman thr0.5 0.676; ep3 thr0.5 Spearman 0.684 (record, UNREVALIDATABLE — ckpt deleted).
