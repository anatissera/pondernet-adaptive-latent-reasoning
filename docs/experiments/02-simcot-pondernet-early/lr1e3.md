# lr1e3

> ⚠️ **Biased metric — test set, not reconcilable.** The numbers on this page were computed on the GSM8K **test** split (not the held-out validation set), and the checkpoint no longer exists, so they cannot be re-evaluated on validation. Treat them as historical/biased. See the [eval-split & leakage note](../../experiments.md#eval-split-and-leakage-note).

**Experiment:** [02-simcot-pondernet-early](runs.md)  **Date:** 2026-06-09  **Status:** ⚠ known-bad (KV-cache bug)

## Summary
Early PonderNet run at the higher lr=1e-3. Trained but not evaluated; predates the gradient-checkpointing fix so it is not worth evaluating.

## Hyperparameters   _(from `command.sh`)_
| param | value |
|-------|-------|
| method | PonderNet adaptive halting |
| epochs · lr · eff. batch | 40 · 1e-3 · — |
| γ · geom_mean · K_max | 0.01 · 3.0 · 6 |
| warm-start · seed · data | full-model · 42 · GSM8K-Aug |

## Results
| checkpoint | threshold | accuracy | avg steps | n_samples |
|-----------|-----------|----------|-----------|-----------|
| — | — | not evaluated | — | — |

## Artifacts
- checkpoint: `models/checkpoints/02-simcot-pondernet-early/lr1e3/`
- command + log: `outputs/02-simcot-pondernet-early/lr1e3/{command.sh,train.log}`
- eval results: `results/02-simcot-pondernet-early/lr1e3/`

## Notes
Trained with --gradient_checkpointing True (KV cache off). No eval results.
