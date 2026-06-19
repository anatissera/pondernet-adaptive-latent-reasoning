# 01: SIM-CoT fixed-K baselines

**Status:** reference   **Dates:** 2026-06-08 → 2026-06-08

## What's being tested
Reference points for the adaptive-halting work: vanilla SIM-CoT with a **fixed** number
of latent steps K (no PonderNet halt head). Establishes the accuracy bar every later
PonderNet run is compared against. Varied: learning rate (upstream default vs a low-LR
re-train). Held fixed: K=6, greedy decoding, GSM8K test.

## Setup
- Method: SIM-CoT fixed-K decoding (CODI checkpoint), K=6.
- `baseline-k6` is the upstream SIM-CoT eval of the CODI checkpoint (the canonical bar).
- `fixedk-k6-lr1e4` re-trains fixed-K at lr=1e-4 to sanity-check our training path.
- Eval: GSM8K test; `baseline-k6` reported as avg of 5 samples.

## Findings
- **39.50%** (`baseline-k6`, avg 5 samples) is the reference accuracy at a fixed 6 steps.
- The low-LR re-train (`fixedk-k6-lr1e4`) collapses to 17.82% — an early symptom of the
  KV-cache training bug later root-caused in [03](../03-simcot-pondernet-gcfix/experiment.md).

See [runs.md](runs.md) for the run table · artifacts under `<dir>/01-simcot-baselines/`.
