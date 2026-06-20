# perinstance-g0.05-b1.5-ep5

**Experiment:** [05-simcot-pondernet-adaptive-prior](runs.md)  **Date:** 2026-06-19 → 2026-06-20  **Status:** ✅ done

## Summary

Per-instance affine halting prior (α=1, β=1.5, γ=0.05) beats the global-prior baseline on
all three metrics: **+0.29pp accuracy**, **−0.12 avg_steps**, **Spearman +0.650** (vs +0.58).
Easy problems (0–1 expr) halt at 2.5–3.0 avg steps; hard problems saturate near K=6.
Epoch 4 wins; ep5 drops slightly (typical early-peak pattern).

## Hyperparameters

| param | value |
|-------|-------|
| method | PonderNet adaptive halting, per-instance prior |
| epochs · lr · eff. batch | 5 · 2e-5 · 128 |
| γ · K_max | 0.05 · 6 |
| prior | affine: geom_mean_i = 1·n_i + 1.5  (n_i = #expr − 1, clamped to [1.5, 6]) |
| warm-start · seed · data | full-model SIM-CoT CODI · 42 · train100k.jsonl |

## Results — epoch sweep (thr=0.8)

| epoch | step | acc | avg_steps |
|-------|------|-----|-----------|
| 1 | 778 | 39.50% | 5.407 |
| 2 | 1556 | 39.80% | 5.303 |
| 3 | 2334 | 39.80% | 5.254 |
| **4** | **3112** | **40.49%** | **5.262** |
| 5 | 3890 | 40.11% | 5.264 |

## Results — winner (ep4 / checkpoint-3112)

| threshold | accuracy | avg_steps | n_samples |
|-----------|----------|-----------|-----------|
| 0.5 | 40.18% | 4.365 | 1319 |
| **0.8** | **40.49%** | **5.262** | 1319 |
| 0.9 | 40.49% | 5.585 | 1319 |

**vs baseline (`04/g0.05-gm3.0-ep5`):**
- acc: 40.49% vs 40.20% (+0.29pp) ✓
- avg_steps: 5.262 vs 5.38 (−0.118) ✓
- Spearman(steps, #expr) @ thr0.5: **+0.650** vs +0.58 (+0.070) ✓

## Steps vs difficulty (ep4, thr=0.5)

| #expr | n | avg_steps | acc% |
|-------|---|-----------|------|
| 0 | 83 | 2.518 | 44.6 |
| 1 | 357 | 2.980 | 61.3 |
| 2 | 364 | 4.558 | 45.9 |
| 3 | 290 | 5.407 | 27.6 |
| 4 | 138 | 5.493 | 15.2 |
| 5 | 57 | 5.684 | 5.3 |
| 6 | 21 | 5.810 | 4.8 |
| 7 | 9 | 5.889 | 22.2 |

Monotone across all bins; Spearman r=+0.650, p=4.7e-159.

## Artifacts

- best checkpoint: `models/checkpoints/05-simcot-pondernet-adaptive-prior/perinstance-g0.05-b1.5-ep5/_best_epoch/` (= step 3112)
- train log: `outputs/05-simcot-pondernet-adaptive-prior/perinstance-g0.05-b1.5-ep5/train.log`
- eval results: `results/05-simcot-pondernet-adaptive-prior/perinstance-g0.05-b1.5-ep5/ep{1-5}/thr{0.5,0.8,0.9}/`
- epoch table: `results/05-simcot-pondernet-adaptive-prior/perinstance-g0.05-b1.5-ep5/epoch_eval.tsv`

## Notes

The average-target confound applies: `mean(n_i) ≈ 1.6` over the training set vs global 3.0,
so the per-instance prior also lowers the average step target. A matched-mean control
(global `geom_mean ≈ 1.6`) would isolate per-instance *shape* from *lower mean*; deferred.
A γ × β sweep is the other natural follow-up (γ=0.05 may be slightly strong given sharper
per-instance KL).
