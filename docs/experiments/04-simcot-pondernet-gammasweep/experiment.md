# 04: γ (KL-prior weight) sweep

**Status:** complete   **Dates:** 2026-06-16 → 2026-06-16

## What's being tested
How the KL-geometric regularizer weight **γ** trades accuracy for compute. γ pulls the
halting distribution `p_k` toward a truncated-geometric prior; larger γ → halt earlier →
fewer latent steps. Goal: map the accuracy-vs-steps frontier and find the operating point
that keeps near-baseline accuracy at the fewest steps. This is where the project's primary
goal — *real* per-instance adaptivity — was achieved.

## Setup
- Method: PonderNet adaptive halting on SIM-CoT, warm-started, the [03](../03-simcot-pondernet-gcfix/experiment.md) recipe.
- Held fixed: ep=5, lr=2e-5, eff. batch=128, geom_mean=3.0, K_max=6, seed=42, `train100k.jsonl`.
- Varied: **γ ∈ {0.0, 0.05, 0.07, 0.08, 0.1, 0.12, 0.15, 0.3}** (`fine-g0.02` is a finer point).
- Eval: GSM8K test, greedy, 1 pass, at inference thresholds 0.5 / 0.8 / 0.9.

## Findings

**Sweet spot: γ=0.05, thr=0.8 → 40.2% accuracy at 5.38 avg steps** — near-baseline accuracy
with ~10% fewer steps. γ is the causal lever for halting: at γ=0 the model never halts early
(6.00/6 steps); raising γ monotonically cuts steps, with accuracy holding to ~γ=0.1 then
falling as halting gets too aggressive.

Accuracy-vs-steps frontier (best operating points, GSM8K test, greedy):

| γ | threshold | accuracy | avg steps | notes |
|---|-----------|----------|-----------|-------|
| 0.0 | 0.8 | 40.6% | 6.00 | no KL pressure; never halts early (control) |
| 0.05 | 0.8 | **40.2%** | **5.38** | ✅ sweet spot — near-baseline acc, ~10% fewer steps |
| 0.05 | 0.5 | 39.0% | **4.08** | more aggressive; -1pp accuracy |
| 0.07 | 0.8 | 39.7% | 5.33 | |
| 0.08 | 0.8 | 39.8% | 5.30 | |
| 0.1 | 0.8 | 39.6% | 5.25 | |
| 0.1 | 0.5 | 39.0% | **3.30** | halves compute vs fixed-K; -0.5pp accuracy |
| 0.12 | 0.8 | 39.4% | 5.20 | |
| 0.15 | 0.8 | 39.7% | 5.10 | |
| 0.15 | 0.5 | 38.0% | 2.90 | aggressive; accuracy starts to drop |
| 0.3 | 0.8 | 39.1% | 4.84 | little gain over γ=0.1 at thr=0.8 |
| 0.3 | 0.5 | 35.9% | 2.50 | too aggressive; -4pp accuracy |

Full per-checkpoint/per-threshold frontier data was at `results/gammasweep/summary.tsv`
(now archived at `results/archive/scratch/_analysis/gammasweep/summary.tsv`).

## Checkpoint pruning (2026-06-20)

To reclaim disk (~28 GB), the saved weights were pruned to **one checkpoint per run — the
best epoch tabulated in [runs.md](runs.md)** (g0.0→3112, g0.05→2334, g0.07→1556, g0.08→3112,
g0.15→2334, g0.3→3112; g0.1/g0.12 kept the final ep5 model; fine-g0.02→1556). The other
epochs' weights and **all** optimizer/scheduler/rng resume-state were deleted. Each kept
checkpoint still carries its tokenizer, so every run is loadable for inference / re-eval at
its best operating point. **All eval JSONs in `results/` are intact** — only the weights
behind the non-best `checkpoint-NNNN` paths are gone, recoverable only by retraining.

See [runs.md](runs.md) for the run table · artifacts under `<dir>/04-simcot-pondernet-gammasweep/`.
