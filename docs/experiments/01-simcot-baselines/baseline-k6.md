# baseline-k6

**Experiment:** [01-simcot-baselines](runs.md)  **Date:** 2026-06-08  **Status:** ✅ reference

> ✅ **Re-validated 2026-06-23.** Metrics below are on the held-out validation split (500 ex, greedy); prior test-set numbers were optimistically biased. See [eval-split note](../../experiments.md#eval-split-and-leakage-note).

## Summary
Upstream SIM-CoT fixed-K (K=6) eval of the CODI checkpoint. This is the canonical accuracy bar that every PonderNet run is compared against. Re-validated greedy single-pass: **40.80%** (validation, n=500, K=6 fixed). The new 40.80% is greedy single-pass on validation — directly comparable to all our PonderNet runs (which are greedy). (Prior: 39.50% test, avg-5-sample.)

## Hyperparameters   _(from `command.sh`)_
| param | value |
|-------|-------|
| method | SIM-CoT fixed-K (baseline, no halt head) |
| epochs · lr · eff. batch | — · 1e-3 (upstream default) · — |
| γ · geom_mean · K_max | — · — · 6 (fixed) |
| warm-start · seed · data | upstream CODI checkpoint · — · validation (500 ex) |

## Results
| checkpoint | threshold | accuracy | avg steps | n_samples |
|-----------|-----------|----------|-----------|-----------|
| CODI checkpoint | fixed K=6 | **40.80%** (val) | 6.00 (fixed) | 1 (greedy), n=500 |

Prior test-set number (biased): 39.50% on GSM8K test (1319 ex), averaged over 5 sampled passes — not directly comparable to the greedy PonderNet runs.

## Artifacts
- checkpoint: `models/checkpoints/01-simcot-baselines/baseline-k6/`
- command + log: `outputs/01-simcot-baselines/baseline-k6/{command.sh,train.log}`
- eval results: `results/01-simcot-baselines/baseline-k6/`

## Notes
Reference point, not one of our trained runs. The headline **40.80%** is greedy single-pass on the held-out validation split (n=500) — directly comparable to all PonderNet runs, which are greedy. The prior 39.50% was on the GSM8K **test** set (biased; data leakage across exp 01–07) and was an average of 5 sampled passes, so it was never apples-to-apples with the greedy runs.
