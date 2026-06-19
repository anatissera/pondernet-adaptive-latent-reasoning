# baseline-k6

**Experiment:** [01-simcot-baselines](runs.md)  **Date:** 2026-06-08  **Status:** ✅ reference

## Summary
Upstream SIM-CoT fixed-K (K=6) eval of the CODI checkpoint. This is the canonical accuracy bar (39.50%) that every PonderNet run is compared against.

## Hyperparameters   _(from `command.sh`)_
| param | value |
|-------|-------|
| method | SIM-CoT fixed-K (baseline, no halt head) |
| epochs · lr · eff. batch | — · 1e-3 (upstream default) · — |
| γ · geom_mean · K_max | — · — · 6 (fixed) |
| warm-start · seed · data | upstream CODI checkpoint · — · GSM8K test |

## Results
| checkpoint | threshold | accuracy | avg steps | n_samples |
|-----------|-----------|----------|-----------|-----------|
| CODI checkpoint | fixed K=6 | 39.50% | 6.00 (fixed) | avg 5 |

## Artifacts
- checkpoint: `models/checkpoints/01-simcot-baselines/baseline-k6/`
- command + log: `outputs/01-simcot-baselines/baseline-k6/{command.sh,train.log}`
- eval results: `results/01-simcot-baselines/baseline-k6/`

## Notes
Reference point, not one of our trained runs. Reported as the average of 5 samples; later PonderNet runs are single greedy passes, so compare with that caveat.
