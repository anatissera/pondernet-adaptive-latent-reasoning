# fixedk-k6-lr1e4

> ⚠️ **Biased metric — test set, not reconcilable.** The numbers on this page were computed on the GSM8K **test** split (not the held-out validation set), and the checkpoint no longer exists, so they cannot be re-evaluated on validation. Treat them as historical/biased. See the [eval-split & leakage note](../../experiments.md#eval-split-and-leakage-note).

**Experiment:** [01-simcot-baselines](runs.md)  **Date:** 2026-06-08  **Status:** ⚠ known-bad (KV-cache bug)

## Summary
Low-LR re-train of fixed-K SIM-CoT (lr=1e-4). Collapsed to 17.82% — an early symptom of the gradient-checkpointing / KV-cache training bug root-caused in experiment 03.

## Hyperparameters   _(from `command.sh`)_
| param | value |
|-------|-------|
| method | SIM-CoT fixed-K retrain |
| epochs · lr · eff. batch | — · 1e-4 · — |
| γ · geom_mean · K_max | — · — · 6 (fixed) |
| warm-start · seed · data | — · — · GSM8K test |

## Results
| checkpoint | threshold | accuracy | avg steps | n_samples |
|-----------|-----------|----------|-----------|-----------|
| retrain | fixed K=6 | 17.82% | 6.00 (fixed) | avg 5 |

## Artifacts
- checkpoint: `models/checkpoints/01-simcot-baselines/fixedk-k6-lr1e4/`
- command + log: `outputs/01-simcot-baselines/fixedk-k6-lr1e4/{command.sh,train.log}`
- eval results: `results/01-simcot-baselines/fixedk-k6-lr1e4/`

## Notes
Trained with --gradient_checkpointing True (KV cache silently off). Not a usable baseline; superseded by experiment 03.
