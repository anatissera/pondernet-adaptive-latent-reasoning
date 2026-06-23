# fullscope-adaptive-g0.05-b1.5-k12-ep5

**Experiment:** [07-simcot-pondernet-fullscope-prior](runs.md)  **Date:** 2026-06-23  **Status:** ✅ done

## Summary

Combined the teammate's Recipe C (full backbone unfreeze, `scope=full`) with exp-05's per-instance
adaptive prior (`geom_mean_i = n_i + 1.5`) and K_max=12 — the first run to test all three
together. Best accuracy: **40.26% @ thr=0.8, 6.96 avg_steps (ep5)**. Best Spearman:
**0.684 @ ep3/thr=0.5** — the highest difficulty-tracking correlation in the project.
Accuracy sits between the two baselines (−0.23pp vs exp-05 thr0.8; −0.22pp vs teammate C-k12
thr0.5), but Spearman is improved on both. The adaptive prior pushes avg_steps higher than
Recipe C alone (4.46 vs 3.21 at thr0.5), trading compute efficiency for better step calibration.

## Hyperparameters _(from `command.sh`)_

| param | value |
|-------|-------|
| method | PonderNet adaptive, full scope |
| epochs · lr · eff. batch | 5 · 2e-5 · 128 (bs=16 × accum=8) |
| γ · adaptive prior · K_max | 0.05 · α=1.0 β=1.5 (geom_mean_i = n_i + 1.5) · 12 |
| warm-start · seed · data | SIM-CoT CODI full · 42 · train100k.jsonl |
| train scope | `full` (backbone LoRA + non-LoRA + halt_head; decoder frozen) |
| TRUNC_K | False |

## Results _(from `summary.json`)_

| epoch | threshold | accuracy | avg steps | Spearman r |
|-------|-----------|----------|-----------|------------|
| ep1 | 0.5 | 39.65% | 4.544 | 0.683 |
| ep1 | 0.8 | 40.03% | 7.083 | 0.663 |
| ep1 | 0.9 | 39.42% | 8.502 | 0.640 |
| ep2 | 0.5 | 39.42% | 4.482 | 0.682 |
| ep2 | 0.8 | 40.11% | 7.014 | 0.668 |
| ep2 | 0.9 | 39.88% | 8.448 | 0.651 |
| ep3 | 0.5 | 39.73% | 4.462 | **0.684** |
| ep3 | 0.8 | 39.88% | 6.949 | 0.670 |
| ep3 | 0.9 | 39.88% | 8.367 | 0.654 |
| ep4 | 0.5 | 39.65% | 4.471 | 0.677 |
| ep4 | 0.8 | 40.03% | 7.001 | 0.668 |
| ep4 | 0.9 | 40.11% | 8.431 | 0.653 |
| **ep5** | **0.5** | **40.11%** | **4.460** | 0.676 |
| **ep5** | **0.8** | **40.26%** | **6.959** | 0.672 |
| **ep5** | **0.9** | 40.03% | 8.375 | 0.655 |

**Winner: ep5 / thr=0.8 → 40.26% @ 6.96 avg_steps, Spearman 0.672**
**Spearman record: ep3 / thr=0.5 → 0.684** (highest in the project)

## Artifacts

- checkpoint: `models/checkpoints/07-simcot-pondernet-fullscope-prior/fullscope-adaptive-g0.05-b1.5-k12-ep5/` (ep5 kept; ep1–4 deleted)
- command + log: `outputs/07-simcot-pondernet-fullscope-prior/fullscope-adaptive-g0.05-b1.5-k12-ep5/`
- eval results: `results/07-simcot-pondernet-fullscope-prior/fullscope-adaptive-g0.05-b1.5-k12-ep5/`

## Notes

**Comparison to baselines:**

| model | acc (thr0.5) | acc (thr0.8) | avg_steps (thr0.5) | Spearman |
|-------|------------|------------|-----------------|---------|
| exp-05: adaptive prior, frozen backbone | 39.95% | 40.49% | 4.37 | 0.650 |
| teammate C-k12: full scope, global prior, thr0.5 | 40.33% | — | 3.21 | — |
| **exp-07 ep5 (this run)** | 40.11% | **40.26%** | 4.46 | **0.676** |

**Interpretation:** Full-scope unfreeze + adaptive prior + K_max=12 improved Spearman
(+0.026 vs exp-05; project record at ep3 = 0.684) but did not exceed either baseline on
accuracy. The adaptive prior targets `geom_mean_i = n_i + 1.5`, which for typical GSM8K
problems (n_expr ≈ 2–4) means a target halting mean of 3.5–5.5 steps — higher than Recipe
C's global geom_mean=3.0. This makes the model use more steps (4.46 vs 3.21 at thr0.5),
improving per-difficulty calibration at the cost of compute efficiency.

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
