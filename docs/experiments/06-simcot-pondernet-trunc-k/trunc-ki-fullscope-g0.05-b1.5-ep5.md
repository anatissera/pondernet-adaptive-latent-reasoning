# trunc-ki-fullscope-g0.05-b1.5-ep5

> ⚠️ **Biased metric — test set, not reconcilable.** The numbers on this page were computed on the GSM8K **test** split (not the held-out validation set), and the checkpoint no longer exists, so they cannot be re-evaluated on validation. Treat them as historical/biased. See the [eval-split & leakage note](../../experiments.md#eval-split-and-leakage-note).

**Experiment:** [06-simcot-pondernet-trunc-k](runs.md)  **Date:** 2026-06-20 → 2026-06-21  **Status:** ✅ done

## Summary

Per-instance truncated-K training (K_i = max(1, n_i)) with full-scope training (backbone LoRA + prj unfrozen) and exp-05 adaptive prior (α=1, β=1.5, γ=0.05). Accuracy fell ~4pp below exp-05 (36.32% vs 40.49% @ thr0.8) but avg_steps at thr0.5 dropped to 3.66 (from 4.37 in exp-05), and the per-n_expr step distribution is the most linear we have measured. Model had not converged at ep5 (accuracy still rising); further epochs or a relaxed truncation are the natural follow-ups.

## Hyperparameters

| param | value |
|-------|-------|
| method | PonderNet adaptive halting, trunc-K + per-instance prior |
| epochs · lr · eff. batch | 5 · 2e-5 · 96 (bs=24 ga=4; OOM prevented intended bs=32 ga=4 eff=128) |
| γ · K_max | 0.05 · 8 |
| prior | adaptive: geom_mean_i = 1·n_i + 1.5, clamped to [1.5, 8] |
| trunc-K | K_i = max(1, min(n_i, 8)); loop runs to max(K_i) in batch |
| training scope | full (backbone LoRA + prj + halt_head; decoder frozen) |
| warm-start · seed · data | full-model SIM-CoT CODI · 42 · train100k.jsonl |

## Results — epoch sweep

| epoch | step | acc (thr0.5) | acc (thr0.8) | avg_steps (thr0.5) |
|-------|------|-------------|-------------|-------------------|
| 1 | 1037 | 34.65% | 34.19% | 4.531 |
| 2 | 2074 | 34.80% | 34.19% | 3.887 |
| 3 | 3111 | 35.78% | 35.78% | 3.792 |
| 4 | 4148 | 36.24% | 36.24% | 3.669 |
| **5** | **5185** | **36.39%** | **36.32%** | **3.660** |

## Results — winner (ep5 / checkpoint-5185)

| threshold | accuracy | avg_steps | n_samples |
|-----------|----------|-----------|-----------|
| **0.5** | **36.39%** | **3.660** | 1319 |
| 0.8 | 36.32% | 5.066 | 1319 |
| 0.9 | 36.47% | 5.532 | 1319 |

**vs exp-05 (`05/perinstance-g0.05-b1.5-ep5` ep4):**
- acc: 36.32% vs 40.49% (−4.17pp) ✗
- avg_steps @ thr0.5: 3.660 vs 4.365 (−0.705) ✓
- Spearman @ thr0.5: +0.596 vs +0.650 (−0.054)

## Steps vs difficulty (ep5, thr=0.5)

| #expr | n | avg_steps |
|-------|---|-----------|
| 0 | 18 | 2.333 |
| 1 | 65 | 2.077 |
| 2 | 357 | 2.588 |
| 3 | 364 | 3.412 |
| 4 | 290 | 4.697 |
| 5 | 138 | 4.877 |
| 6 | 57 | 4.930 |
| 7 | 21 | 5.524 |
| 8 | 9 | 5.778 |

Spearman r=+0.596, p=8.2e-128 (thr0.5); r=+0.501, p=9.6e-85 (thr0.8).

## Artifacts

- best checkpoint: `models/checkpoints/06-simcot-pondernet-trunc-k/trunc-ki-fullscope-g0.05-b1.5-ep5/_best_epoch/` (= step 5185, ep5)
- train log: `outputs/06-simcot-pondernet-trunc-k/trunc-ki-fullscope-g0.05-b1.5-ep5/train.log`
- eval results: `results/06-simcot-pondernet-trunc-k/trunc-ki-fullscope-g0.05-b1.5-ep5/ep{1-5}/thr{0.5,0.8,0.9}/`
- epoch table: `results/06-simcot-pondernet-trunc-k/trunc-ki-fullscope-g0.05-b1.5-ep5/epoch_eval.tsv`

## Notes

OOM history: bs=32 cold-start OOMed at ep3/step460 (22.96 GB allocated); bs=24 resumed from ep2 checkpoint also OOMed at the same point (resume overhead inflated steady-state from 18 → 22 GB). Cold-start at bs=24 cleared the OOM (stable at 19.7 GB). The problematic batch at ep3/step460 consistently requests a 1.16 GiB allocation spike; the cold-start has enough headroom to absorb it.

Effective batch was 96 (bs=24 × ga=4) instead of the intended 128. This changes the convergence speed; a fair comparison against exp-05 would use eff-batch=128.

Accuracy still monotonically rising at ep5 — the model had not converged. Planned follow-ups: (a) continue from ep5 checkpoint for more epochs; (b) relax truncation to K_i = n_i + 2.
