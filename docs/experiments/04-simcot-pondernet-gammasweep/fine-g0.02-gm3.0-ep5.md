# fine-g0.02-gm3.0-ep5

> ⚠️ **Biased metric — test set, not reconcilable.** The numbers on this page were computed on the GSM8K **test** split (not the held-out validation set), and the checkpoint no longer exists, so they cannot be re-evaluated on validation. Treat them as historical/biased. See the [eval-split & leakage note](../../experiments.md#eval-split-and-leakage-note).

**Experiment:** [04-simcot-pondernet-gammasweep](runs.md)  **Date:** 2026-06-16  **Status:** ⚠ trained, eval not tabulated

## Summary
A finer γ=0.02 point between the control (γ=0) and the sweet spot (γ=0.05). The checkpoint exists but its eval was never collated into the sweep frontier TSV.

## Hyperparameters   _(from `command.sh`)_
| param | value |
|-------|-------|
| method | PonderNet adaptive halting |
| epochs · lr · eff. batch | 5 · 2e-5 · 128 |
| γ · geom_mean · K_max | 0.02 · 3.0 · 6 |
| warm-start · seed · data | full-model · 42 · train100k.jsonl |

## Results
| checkpoint | threshold | accuracy | avg steps | n_samples |
|-----------|-----------|----------|-----------|-----------|
| — | — | not collated | — | — |

## Artifacts
- checkpoint: `models/checkpoints/04-simcot-pondernet-gammasweep/fine-g0.02-gm3.0-ep5/`
- command + log: `outputs/04-simcot-pondernet-gammasweep/fine-g0.02-gm3.0-ep5/{command.sh,train.log}`
- eval results: `results/04-simcot-pondernet-gammasweep/fine-g0.02-gm3.0-ep5/`

## Notes
Checkpoint present under models/checkpoints/04-simcot-pondernet-gammasweep/fine-g0.02-gm3.0-ep5/ but no aggregated metrics; re-record this row after evaluating if this point is needed.
