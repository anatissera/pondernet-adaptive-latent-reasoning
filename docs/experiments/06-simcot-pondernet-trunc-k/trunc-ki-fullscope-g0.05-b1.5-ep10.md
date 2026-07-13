# trunc-ki-fullscope-g0.05-b1.5-ep10

> ⚠️ **Biased metric - test set, not reconcilable.** The numbers on this page were computed on the GSM8K **test** split (not the held-out validation set), and the checkpoint no longer exists, so they cannot be re-evaluated on validation. Treat them as historical/biased. See the [eval-split & leakage note](../../experiments.md#eval-split-and-leakage-note).

**Experiment:** [06-simcot-pondernet-trunc-k](runs.md)  **Date:** 2026-06-22 → 2026-06-23  **Status:** ✅ done

## Summary

10-epoch extension of the trunc-K investigation (cf. `trunc-ki-fullscope-g0.05-b1.5-ep5`), targeting convergence with correct eff-batch 128 on the RTX 3090. Training was interrupted by an OOM at epoch 3 step 460 (batch=32 too large for K_max=8 activations on 24 GB) and resumed from the epoch-2 checkpoint with per_device=16, accum=8 (same eff-batch 128, smaller per-step activations). Due to HF Trainer recounting steps with the new batch size, the resumed run ran 6 additional real training epochs (ep3–ep8 total). Best checkpoint is ep8 (global step 3890): **36.54% @ thr0.5**, **35.94% @ thr0.8**, Spearman 0.592. This is −4.55pp below exp-05 on accuracy but avg_steps @ thr0.8 (5.194) undercuts exp-05 (5.262) and the adaptive compute signal is strongly monotonic across difficulty bins.

## Hyperparameters _(from `command.sh`)_

| param | value |
|-------|-------|
| method | PonderNet adaptive (trunc-K + adaptive prior) |
| epochs · lr · eff. batch | 8 real epochs (OOM+resume; see Notes) · 2e-5 · 128 (16×8 after resume) |
| γ · geom_mean · K_max | 0.05 · 3.0 · 8 |
| adaptive prior | geom_mean_i = 1.0·n_i + 1.5 (clamped to [1.5, 8]) |
| trunc-K | K_i = clamp(n_i, 1, 8) per example |
| warm-start · seed · data | full SIM-CoT CODI · 42 · train100k.jsonl |

## Results _(from `summary.json`, best checkpoint = ep8 / step 3890)_

| checkpoint | threshold | accuracy | avg steps | n_samples |
|-----------|-----------|----------|-----------|-----------|
| checkpoint-3890 | 0.5 | 36.54% | 3.766 | 1 (greedy) |
| checkpoint-3890 | 0.8 | 35.94% | 5.194 | 1 (greedy) |
| checkpoint-3890 | 0.9 | 35.56% | 5.630 | 1 (greedy) |

### vs baselines

| metric | exp-04 (γ=0.05) | exp-05 (ep4) | ep5 run | **this run (ep8)** |
|--------|----------------|-------------|---------|------------------|
| acc @ thr0.8 | 40.20% | 40.49% | 36.32% | **35.94%** |
| avg_steps @ thr0.5 | ~4.4 | 4.365 | 3.660 | **3.766** |
| avg_steps @ thr0.8 | 5.380 | 5.262 | 5.066 | **5.194** |
| Spearman @ thr0.5 | 0.580 | 0.650 | 0.596 | **0.592** |
| Spearman @ thr0.8 | - | 0.586 | 0.501 | **0.492** |

### Steps vs difficulty (thr=0.5, ep8)

| n_expr | n | avg_steps |
|--------|---|-----------|
| 0 | 18 | 2.50 |
| 1 | 65 | 2.15 |
| 2 | 357 | 2.70 |
| 3 | 364 | 3.54 |
| 4 | 290 | 4.77 |
| 5 | 138 | 4.97 |
| 6 | 57 | 5.00 |
| 7 | 21 | 5.62 |
| 8 | 9 | 6.00 |

Spearman r=0.592 (thr0.5), p=1.93e-125. Easy problems (n_expr ≤ 2) average 2.15–2.70 steps; hard problems (n_expr ≥ 7) reach 5.6–6.0. The monotonic trend confirms the trunc-K training signal was learned and generalizes to inference.

### Epoch sweep (thr=0.5 / thr=0.8)

| actual epoch | global step | acc (thr0.5) | acc (thr0.8) | avg_steps (thr0.5) |
|-------------|-------------|-------------|-------------|-------------------|
| 1 | 778 | 35.10% | 34.65% | 4.806 |
| 2 | 1556 | 35.18% | 35.10% | 4.077 |
| 3 | 1945 | 36.16% | 35.03% | 3.892 |
| 4 | 2334 | 35.78% | 35.25% | 3.780 |
| 5 | 2723 | 35.48% | 35.78% | 3.765 |
| 6 | 3112 | 36.09% | 35.71% | 3.764 |
| 7 | 3501 | 36.01% | 35.56% | 3.762 |
| **8** | **3890** | **36.54%** | **35.94%** | **3.766** |

Accuracy is still slowly improving at ep8; avg_steps plateaued around ep5 (3.76) and barely moves thereafter.

## Artifacts

- checkpoint: `models/checkpoints/06-simcot-pondernet-trunc-k/trunc-ki-fullscope-g0.05-b1.5-ep10/` _(cleaned up post-training; best epoch preserved in `_best_epoch/` until deletion)_
- command + log: `outputs/06-simcot-pondernet-trunc-k/trunc-ki-fullscope-g0.05-b1.5-ep10/{command.sh,train.log}`
- eval results: `results/06-simcot-pondernet-trunc-k/trunc-ki-fullscope-g0.05-b1.5-ep10/`

## Notes

**Training timeline:** Launched 2026-06-22 with per_device=32, accum=4 (eff-batch 128). OOM'd at epoch 3 step 460 - GPU 1 (RTX 3090, 24 GB) was using 22.78 GB and needed 596 MB more. Resumed from checkpoint-1556 (end of ep2) with per_device=16, accum=8 (same eff-batch 128, ~7 GB of VRAM headroom freed). The HF Trainer recomputed steps-per-epoch as 389 (vs 778 at batch 32 - reason unclear, possibly a fast-forward artifact in the progress bar). The epoch numbering in the watcher TSV is mislabeled for ep3+; actual epoch-to-step mapping is in the table above.

**Accuracy did not recover.** After 8 real epochs, accuracy plateaued in the mid-36% range at thr0.5 and mid-35% at thr0.8 - substantially below the exp-05 best of 40.49% and essentially where the ep5 run ended. Two remaining confounds: (1) the OOM + resume introduced a batch-size change mid-run, which may have disrupted the LR schedule (cosine with warmup, baked in at launch) and (2) the effective epoch count from the resumed trainer's perspective may have been wrong (it appears to have run only 6 more epochs, not 8 from scratch).

**Avg_steps saturated quickly.** The halting head learned its target avg_steps by ep3 (~3.77 at thr0.5) and did not change thereafter. This decoupling of avg_steps (stable) from accuracy (still slowly improving) suggests the halt head converged but the backbone is still learning to use the allocated steps effectively - more epochs would likely help accuracy without changing the adaptive compute behavior.

**Adaptive signal is strong and preserved.** Despite the accuracy regression, the per-n_expr step distribution is wide and monotonic: 2.1→6.0 steps across n_expr 1→8. Spearman 0.592 is between the baseline (0.580) and exp-05 (0.650), confirming trunc-K contributes adaptive signal beyond the geometric prior alone.

**Suggested follow-up:** (a) relaxed truncation K_i = min(n_i + 2, K_max) to give the backbone more breathing room; (b) start fresh (no resume) at per_device=16, accum=8 for 12–15 epochs; (c) consider whether a lower γ (0.01 instead of 0.05) would allow the backbone more freedom to use steps before the KL penalty forces early halting.
