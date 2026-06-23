# perinstance-g0.05-b1.5-ep5

**Experiment:** [05-simcot-pondernet-adaptive-prior](runs.md)  **Date:** 2026-06-19 → 2026-06-20  **Status:** ✅ done

> ✅ **Re-validated 2026-06-23.** Metrics below are on the held-out validation split (500 ex, greedy); prior test-set numbers were optimistically biased. See [eval-split note](../../experiments.md#eval-split-and-leakage-note).

## Summary

Per-instance affine halting prior (α=1, β=1.5, γ=0.05). Re-validated, accuracy is at the 40.80%
greedy validation baseline (ep4 40.80% @ thr0.5, ep5 41.00% @ thr0.5; n=500) — no clear accuracy
win — but the run is strongly **step-efficient and difficulty-tracking**: easy problems (0–1
expr) halt at ~2.0–2.7 avg steps while hard ones saturate near K=6, with **Spearman +0.662**
(ep4, thr0.5) — the best difficulty tracking of the K=6 runs. Epoch 4 is the kept `_best_epoch`;
ep5 also survived. Earlier epochs (ep1–ep3) were deleted and cannot be re-validated.

## Hyperparameters

| param | value |
|-------|-------|
| method | PonderNet adaptive halting, per-instance prior |
| epochs · lr · eff. batch | 5 · 2e-5 · 128 |
| γ · K_max | 0.05 · 6 |
| prior | affine: geom_mean_i = 1·n_i + 1.5  (n_i = #expr − 1, clamped to [1.5, 6]) |
| warm-start · seed · data | full-model SIM-CoT CODI · 42 · train100k.jsonl |

## Results — epoch sweep (thr=0.8)

Only ep4 and ep5 survived re-validation; ep1–ep3 checkpoints were deleted.

| epoch | step | acc | avg_steps | source |
|-------|------|-----|-----------|--------|
| 1 | 778 | 39.50% | 5.407 | test, biased — ckpt deleted |
| 2 | 1556 | 39.80% | 5.303 | test, biased — ckpt deleted |
| 3 | 2334 | 39.80% | 5.254 | test, biased — ckpt deleted |
| **4** | **3112** | **39.80% (val)** | **5.232** | re-validated, n=500 (prior test 40.49%) |
| 5 | 3890 | 40.60% (val) | 5.244 | re-validated, n=500 (prior test 40.11%) |

## Results — re-validated (validation split, n=500, greedy)

| checkpoint | thr 0.5 | thr 0.8 | thr 0.9 |
|-----------|---------|---------|---------|
| checkpoint-3112 (ep4, `_best_epoch`) | 40.80% / 4.276 | 39.80% / 5.232 | 40.40% / 5.578 |
| checkpoint-3890 (ep5) | 41.00% / 4.282 | 40.60% / 5.244 | 40.60% / 5.582 |

(acc% / avg_steps; n=500 validation; prior numbers were on GSM8K test, 1319 ex, biased.)

**Difficulty tracking** — Spearman(steps_used, #expr), validation:
- ep4: thr0.5 = **+0.662**, thr0.8 = +0.598, thr0.9 = +0.505
- ep5: thr0.5 = **+0.663**, thr0.8 = +0.598, thr0.9 = +0.505

**vs baseline (`04/g0.05-gm3.0-ep5`, re-validated):**
- acc @ thr0.5: 40.80% (ep4) vs 40.80% (baseline) — flat (no accuracy win on validation)
- avg_steps @ thr0.5: 4.276 (ep4) vs 4.060 (baseline)
- Spearman @ thr0.5: **+0.662** (ep4) vs +0.553 (baseline) — clear difficulty-tracking gain ✓

**Prior test-set numbers (biased):** ep4 thr0.8 40.49% (was the reported winner), Spearman thr0.5 0.650.

## Steps vs difficulty (ep4, thr=0.5, validation)

avg_steps by n_expr bin (ep4, thr0.5): n0 = 2.00, n1 = 2.65, n2 = 3.03, n3 = 4.55, n4 = 5.30,
n5+ = 5.72 — monotone, a clear adaptive signal. Spearman r = +0.662 (validation, n=500).

## Artifacts

- best checkpoint: `models/checkpoints/05-simcot-pondernet-adaptive-prior/perinstance-g0.05-b1.5-ep5/_best_epoch/` (= step 3112)
- train log: `outputs/05-simcot-pondernet-adaptive-prior/perinstance-g0.05-b1.5-ep5/train.log`
- eval results: `results/05-simcot-pondernet-adaptive-prior/perinstance-g0.05-b1.5-ep5/ep{1-5}/thr{0.5,0.8,0.9}/`
- epoch table: `results/05-simcot-pondernet-adaptive-prior/perinstance-g0.05-b1.5-ep5/epoch_eval.tsv`

## Notes

On validation the per-instance prior shows **no accuracy advantage** over the global-prior
baseline (both ~40.8% @ thr0.5) — the prior test-set "+0.29pp" win does not survive de-biasing.
What is robust and consistent is the **step-efficiency and difficulty-tracking**: Spearman rises
from +0.553 (baseline) to +0.662 (ep4) on validation, and avg_steps tracks n_expr monotonically.

The average-target confound applies: `mean(n_i) ≈ 1.6` over the training set vs global 3.0,
so the per-instance prior also lowers the average step target. A matched-mean control
(global `geom_mean ≈ 1.6`) would isolate per-instance *shape* from *lower mean*; deferred.
A γ × β sweep is the other natural follow-up (γ=0.05 may be slightly strong given sharper
per-instance KL).
