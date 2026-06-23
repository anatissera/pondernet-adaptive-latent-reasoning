# fullscope-adaptive-g0.05-b1.5-k12-ep5

**Experiment:** [07-simcot-pondernet-fullscope-prior](runs.md)  **Date:** 2026-06-23  **Status:** ✅ done

> ✅ **Re-validated 2026-06-23.** Metrics below are on the held-out validation split (500 ex, greedy); prior test-set numbers were optimistically biased. Only ep5 survived; ep1–ep4 were deleted. See [eval-split note](../../experiments.md#eval-split-and-leakage-note).

## Summary

Combined the teammate's Recipe C (full backbone unfreeze, `scope=full`) with exp-05's per-instance
adaptive prior (`geom_mean_i = n_i + 1.5`) and K_max=12 — the first run to test all three
together. Re-validated (ep5, the only surviving checkpoint): **41.00% @ thr0.5 / 4.336 avg_steps**
and **41.00% @ thr0.8 / 6.804 avg_steps** (validation, n=500) — at the 40.80% greedy validation
baseline, so no accuracy win, but the run delivers the **strongest difficulty tracking in the
project**: **Spearman +0.675 @ thr0.5** (ep5, validation). The old project-record Spearman
(ep3 = 0.684, test) **cannot be re-validated** — ep1–ep4 checkpoints were deleted.

## Hyperparameters _(from `command.sh`)_

| param | value |
|-------|-------|
| method | PonderNet adaptive, full scope |
| epochs · lr · eff. batch | 5 · 2e-5 · 128 (bs=16 × accum=8) |
| γ · adaptive prior · K_max | 0.05 · α=1.0 β=1.5 (geom_mean_i = n_i + 1.5) · 12 |
| warm-start · seed · data | SIM-CoT CODI full · 42 · train100k.jsonl |
| train scope | `full` (backbone LoRA + non-LoRA + halt_head; decoder frozen) |
| TRUNC_K | False |

## Results — re-validated (validation split, n=500, greedy)

Only ep5 (`checkpoint-3890`) survived re-validation; ep1–ep4 checkpoints were deleted.

| checkpoint | thr 0.5 | thr 0.8 | thr 0.9 |
|-----------|---------|---------|---------|
| checkpoint-3890 (ep5) | 41.00% / 4.336 | 41.00% / 6.804 | 40.00% / 8.192 |

(acc% / avg_steps; n=500 validation; prior numbers were on GSM8K test, 1319 ex, biased.)

**Difficulty tracking** — Spearman(steps_used, #expr), validation, ep5: thr0.5 = **+0.675**, thr0.8 = +0.671, thr0.9 = +0.662 — the strongest difficulty tracking in the project (on validation).

### Prior test-set numbers (biased; ep1–ep4 checkpoints deleted, not reconcilable)

The full per-epoch×threshold grid below was on the GSM8K **test** set (leakage). Only **ep5**
was re-validated (above); **ep1–ep4 cannot be re-validated** — their checkpoints were deleted,
**including the old project-record Spearman ep3 = 0.684**.

| epoch | threshold | accuracy | avg steps | Spearman r | source |
|-------|-----------|----------|-----------|------------|--------|
| ep1 | 0.5 | 39.65% | 4.544 | 0.683 | test, biased — ckpt deleted |
| ep1 | 0.8 | 40.03% | 7.083 | 0.663 | test, biased — ckpt deleted |
| ep1 | 0.9 | 39.42% | 8.502 | 0.640 | test, biased — ckpt deleted |
| ep2 | 0.5 | 39.42% | 4.482 | 0.682 | test, biased — ckpt deleted |
| ep2 | 0.8 | 40.11% | 7.014 | 0.668 | test, biased — ckpt deleted |
| ep2 | 0.9 | 39.88% | 8.448 | 0.651 | test, biased — ckpt deleted |
| ep3 | 0.5 | 39.73% | 4.462 | **0.684** | test, biased — ckpt deleted (UNREVALIDATABLE record) |
| ep3 | 0.8 | 39.88% | 6.949 | 0.670 | test, biased — ckpt deleted |
| ep3 | 0.9 | 39.88% | 8.367 | 0.654 | test, biased — ckpt deleted |
| ep4 | 0.5 | 39.65% | 4.471 | 0.677 | test, biased — ckpt deleted |
| ep4 | 0.8 | 40.03% | 7.001 | 0.668 | test, biased — ckpt deleted |
| ep4 | 0.9 | 40.11% | 8.431 | 0.653 | test, biased — ckpt deleted |
| ep5 | 0.5 | 40.11% | 4.460 | 0.676 | test, biased (re-validated above: 41.00% / 4.336, +0.675) |
| ep5 | 0.8 | 40.26% | 6.959 | 0.672 | test, biased (re-validated above: 41.00% / 6.804, +0.671) |
| ep5 | 0.9 | 40.03% | 8.375 | 0.655 | test, biased (re-validated above: 40.00% / 8.192, +0.662) |

**avg_steps by n_expr bin (ep5, thr0.5, validation):** n0 = 1.83, n1 = 2.50, n2 = 2.86,
n3 = 4.49, n4 = 5.43, n5+ = 6.41 — monotone, the project's clearest adaptive signal.

## Artifacts

- checkpoint: `models/checkpoints/07-simcot-pondernet-fullscope-prior/fullscope-adaptive-g0.05-b1.5-k12-ep5/` (ep5 kept; ep1–4 deleted)
- command + log: `outputs/07-simcot-pondernet-fullscope-prior/fullscope-adaptive-g0.05-b1.5-k12-ep5/`
- eval results: `results/07-simcot-pondernet-fullscope-prior/fullscope-adaptive-g0.05-b1.5-k12-ep5/`

## Notes

**Comparison to baselines (validation, n=500, greedy):**

| model | acc (thr0.5) | acc (thr0.8) | avg_steps (thr0.5) | Spearman (thr0.5) |
|-------|------------|------------|-----------------|---------|
| baseline-k6 (fixed K=6) | 40.80% | — | 6.00 | — |
| exp-05: adaptive prior, frozen backbone (ep4) | 40.80% | 39.80% | 4.276 | 0.662 |
| **exp-07 ep5 (this run)** | 41.00% | 41.00% | 4.336 | **0.675** |

(teammate C-k12 was on the test set and is not re-validated here.)

**Interpretation:** Full-scope unfreeze + adaptive prior + K_max=12 gives the **strongest
difficulty tracking in the project** on validation (Spearman +0.675 @ thr0.5, vs +0.662 for
exp-05) but sits **at the 40.80% greedy validation baseline on accuracy** — no accuracy win
once the test-set bias is removed. The adaptive prior targets `geom_mean_i = n_i + 1.5`, which
for typical GSM8K problems (n_expr ≈ 2–4) means a target halting mean of 3.5–5.5 steps; this
trades compute efficiency for per-difficulty calibration. The old project-record Spearman
(ep3 = 0.684, test set) **cannot be re-validated** — that checkpoint was deleted.

**Accuracy is essentially flat across epochs** (39.42%–40.26%), suggesting full-scope training
converged in step-count calibration quickly (ep1 Spearman already 0.683) while accuracy
continued to improve slowly through ep5. This mirrors the exp-06 pattern where the halting
head learns fast but the backbone needs more epochs for answer quality.

**Loss trend (epoch averages):**

| epoch | avg train loss | avg ce loss |
|-------|---------------|------------|
| ep1 | 0.575 | 0.172 |
| ep2 | 0.550 | 0.153 |
| ep3 | 0.558 | 0.148 |
| ep4 | 0.559 | 0.151 |
| ep5 | 0.533 | 0.140 |

Training ran clean: no OOM, no loss spikes, 8h 54m on RTX 3090.
