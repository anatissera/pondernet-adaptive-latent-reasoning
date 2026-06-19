# lr1e4

**Experiment:** [02-simcot-pondernet-early](runs.md)  **Date:** 2026-06-09  **Status:** ⚠ known-bad (KV-cache bug)

## Summary
Best of the early (pre-gcfix) PonderNet runs at 19.26% (thr=0.8) — still roughly half the 39.5% baseline because of the KV-cache training bug.

## Hyperparameters   _(from `command.sh`)_
| param | value |
|-------|-------|
| method | PonderNet adaptive halting |
| epochs · lr · eff. batch | 40 · 1e-4 · — |
| γ · geom_mean · K_max | 0.01 · 3.0 · 6 |
| warm-start · seed · data | full-model · 42 · GSM8K-Aug |

## Results
| checkpoint | threshold | accuracy | avg steps | n_samples |
|-----------|-----------|----------|-----------|-----------|
| best | 0.8 | 19.26% | — | avg 5 |
| best | 0.9 | 17.82% | — | avg 5 |

## Artifacts
- checkpoint: `models/checkpoints/02-simcot-pondernet-early/lr1e4/`
- command + log: `outputs/02-simcot-pondernet-early/lr1e4/{command.sh,train.log}`
- eval results: `results/02-simcot-pondernet-early/lr1e4/`

## Notes
Trained with --gradient_checkpointing True (KV cache off). Numbers reflect the bug, not the method. See experiment 03 Root Cause.
