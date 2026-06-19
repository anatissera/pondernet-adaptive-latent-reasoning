# warmstart-lr1e4

**Experiment:** [02-simcot-pondernet-early](runs.md)  **Date:** 2026-06-09  **Status:** ⚠ known-bad (KV-cache bug)

## Summary
PonderNet warm-started from SIM-CoT (400 tensors), lr=1e-4. Reached 15.24% — warm-starting did not help because the KV-cache bug dominated.

## Hyperparameters   _(from `command.sh`)_
| param | value |
|-------|-------|
| method | PonderNet adaptive halting (warm-started) |
| epochs · lr · eff. batch | 40 · 1e-4 · — |
| γ · geom_mean · K_max | 0.01 · 3.0 · 6 |
| warm-start · seed · data | full-model (400 tensors) · 42 · GSM8K-Aug |

## Results
| checkpoint | threshold | accuracy | avg steps | n_samples |
|-----------|-----------|----------|-----------|-----------|
| best | 0.8 | 15.24% | — | avg 5 |

## Artifacts
- checkpoint: `models/checkpoints/02-simcot-pondernet-early/warmstart-lr1e4/`
- command + log: `outputs/02-simcot-pondernet-early/warmstart-lr1e4/{command.sh,train.log}`
- eval results: `results/02-simcot-pondernet-early/warmstart-lr1e4/`

## Notes
Trained with --gradient_checkpointing True (KV cache off). Lower than the non-warmstart lr1e4; the bug masks any warm-start benefit.
