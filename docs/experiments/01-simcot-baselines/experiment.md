# 01: SIM-CoT fixed-K baselines

**Status:** reference   **Dates:** 2026-06-08 → 2026-06-08

## What's being tested
Reference points for the adaptive-halting work: vanilla SIM-CoT with a **fixed** number
of latent steps K (no PonderNet halt head). Establishes the accuracy bar every later
PonderNet run is compared against. Varied: learning rate (upstream default vs a low-LR
re-train). Held fixed: K=6, greedy decoding.

> The canonical bar (`baseline-k6`) was **re-validated 2026-06-23** on the held-out validation
> split (500 ex, greedy single-pass): **40.80%**. The prior 39.50% was on the GSM8K **test** set
> (biased) and averaged over 5 sampled passes. `fixedk-k6-lr1e4` carries a separate bias banner
> and was **not** re-evaluated.

## Setup
- Method: SIM-CoT fixed-K decoding (CODI checkpoint), K=6.
- `baseline-k6` is the upstream SIM-CoT eval of the CODI checkpoint (the canonical bar).
- `fixedk-k6-lr1e4` re-trains fixed-K at lr=1e-4 to sanity-check our training path.
- Eval: `baseline-k6` re-validated on the validation split (500 ex, greedy single-pass); the
  prior 39.50% was on GSM8K test, avg of 5 samples (biased).

## Findings
- **40.80%** (`baseline-k6`, validation, n=500, greedy single-pass) is the reference accuracy at
  a fixed 6 steps - directly comparable to the greedy PonderNet runs. (Prior: 39.50% test,
  avg-5-sample; biased by test-set leakage.)
- The low-LR re-train (`fixedk-k6-lr1e4`) collapses to 17.82% - an early symptom of the
  KV-cache training bug later root-caused in [03](../03-simcot-pondernet-gcfix/experiment.md).

See [runs.md](runs.md) for the run table · artifacts under `<dir>/01-simcot-baselines/`.
