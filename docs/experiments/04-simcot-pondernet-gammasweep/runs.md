# 04-simcot-pondernet-gammasweep - Runs

See [experiment.md](experiment.md) for the frontier and the sweet-spot conclusion.
Accuracy = best operating point at thr=0.8; "min steps" = thr=0.5 efficiency point.

> Re-validated 2026-06-23 (partial): only `g0.05-gm3.0-ep5` (ep3) survived and was re-evaluated
> on the validation split (500 ex, greedy). The other 7 γ configs were deleted - their numbers
> are on the GSM8K test set (biased) and the sweep's winner-selection cannot be reproduced.

| run | key variable | best accuracy (thr=0.8) | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| g0.0-gm3.0-ep5 | γ=0.0 (control) | 40.56% (test, biased - ckpt deleted) | 6.00 | ✅ done | [detail](g0.0-gm3.0-ep5.md) |
| g0.05-gm3.0-ep5 | γ=0.05 | **41.40% (val, n=500)** | **5.340** ✅ sweet spot | ✅ done | [detail](g0.05-gm3.0-ep5.md) |
| g0.07-gm3.0-ep5 | γ=0.07 | 39.73% (test, biased - ckpt deleted) | 5.33 | ✅ done | [detail](g0.07-gm3.0-ep5.md) |
| g0.08-gm3.0-ep5 | γ=0.08 | 39.80% (test, biased - ckpt deleted) | 5.30 | ✅ done | [detail](g0.08-gm3.0-ep5.md) |
| g0.1-gm3.0-ep5 | γ=0.1 | 39.58% (test, biased - ckpt deleted) | 5.25 | ✅ done | [detail](g0.1-gm3.0-ep5.md) |
| g0.12-gm3.0-ep5 | γ=0.12 | 39.35% (test, biased - ckpt deleted) | 5.20 | ✅ done | [detail](g0.12-gm3.0-ep5.md) |
| g0.15-gm3.0-ep5 | γ=0.15 | 39.73% (test, biased - ckpt deleted) | 5.10 | ✅ done | [detail](g0.15-gm3.0-ep5.md) |
| g0.3-gm3.0-ep5 | γ=0.3 | 39.12% (test, biased - ckpt deleted) | 4.84 | ✅ done | [detail](g0.3-gm3.0-ep5.md) |
| fine-g0.02-gm3.0-ep5 | γ=0.02 (fine point) | not collated | - | ⚠ trained, eval not tabulated | [detail](fine-g0.02-gm3.0-ep5.md) |

`g0.05` re-validated: 41.40% / 5.340 @ thr0.8, 40.80% / 4.060 @ thr0.5 (val, n=500); prior test (biased) was 40.18% / 38.97%. Validation Spearman(n_expr, steps) = +0.553 @ thr0.5.
