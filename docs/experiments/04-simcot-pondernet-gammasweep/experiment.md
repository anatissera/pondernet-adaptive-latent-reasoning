# 04: γ (KL-prior weight) sweep

**Status:** complete   **Dates:** 2026-06-16 → 2026-06-16

> ✅ **Re-validated 2026-06-23 (partial).** Only the sweet-spot run `g0.05` (ep3,
> `checkpoint-2334`) survived and was re-evaluated on the held-out validation split (500 ex,
> greedy). The other **7 γ configs were deleted** — their numbers below stay on the test set
> (biased) and the **sweep's winner-selection cannot be reproduced on validation**. See
> [eval-split note](../../experiments.md#eval-split-and-leakage-note).

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
- Eval: greedy, 1 pass, thresholds 0.5 / 0.8 / 0.9. Only `g0.05` re-validated (validation,
  500 ex); the rest remain on GSM8K test (biased).

## Findings

**Winner-selection caveat (2026-06-23).** The γ-sweep was conducted entirely on the **test**
set (data leakage). Only the selected `g0.05` checkpoint survived; the other 7 γ configs were
deleted, so **the sweep cannot be re-run on validation and its winner cannot be re-confirmed**.
The frontier table below is kept on the test set (biased) for the 7 deleted configs; only the
`g0.05` rows are re-validated.

**Sweet spot (re-validated): γ=0.05, thr0.8 → 41.40% (val) / 5.340 avg_steps; thr0.5 → 40.80%
(val) / 4.060 avg_steps** — at/near the 40.80% greedy validation baseline while spending fewer
steps. The accuracy is within noise of the baseline (n=500); the durable result is
step-efficiency at fixed accuracy plus positive difficulty tracking (Spearman +0.553 @ thr0.5).
Qualitatively γ remains the causal lever for halting: at γ=0 the model never halts early
(6.00/6 steps) and raising γ cuts steps — but this trend was characterized on the (biased) test
set and only the γ=0.05 point survives on validation.

Accuracy-vs-steps frontier (best operating points):

| γ | threshold | accuracy | avg steps | notes |
|---|-----------|----------|-----------|-------|
| 0.0 | 0.8 | 40.6% | 6.00 | test, biased — ckpt deleted; never halts early (control) |
| 0.05 | 0.8 | **41.40% (val)** | **5.340** | ✅ sweet spot, re-validated (n=500); prior test 40.18% |
| 0.05 | 0.5 | 40.80% (val) | **4.060** | re-validated (n=500); prior test 38.97% |
| 0.07 | 0.8 | 39.7% | 5.33 | test, biased — ckpt deleted |
| 0.08 | 0.8 | 39.8% | 5.30 | test, biased — ckpt deleted |
| 0.1 | 0.8 | 39.6% | 5.25 | test, biased — ckpt deleted |
| 0.1 | 0.5 | 39.0% | **3.30** | test, biased — ckpt deleted |
| 0.12 | 0.8 | 39.4% | 5.20 | test, biased — ckpt deleted |
| 0.15 | 0.8 | 39.7% | 5.10 | test, biased — ckpt deleted |
| 0.15 | 0.5 | 38.0% | 2.90 | test, biased — ckpt deleted |
| 0.3 | 0.8 | 39.1% | 4.84 | test, biased — ckpt deleted |
| 0.3 | 0.5 | 35.9% | 2.50 | test, biased — ckpt deleted |

The 7 non-`g0.05` configs above are **on the GSM8K test set (biased; checkpoints deleted, not
reconcilable)** and cannot be re-validated. Full per-checkpoint/per-threshold frontier data was
at `results/gammasweep/summary.tsv` (now archived at
`results/archive/scratch/_analysis/gammasweep/summary.tsv`).

## Checkpoint pruning (2026-06-20)

To reclaim disk (~28 GB), the saved weights were pruned to **one checkpoint per run — the
best epoch tabulated in [runs.md](runs.md)** (g0.0→3112, g0.05→2334, g0.07→1556, g0.08→3112,
g0.15→2334, g0.3→3112; g0.1/g0.12 kept the final ep5 model; fine-g0.02→1556). The other
epochs' weights and **all** optimizer/scheduler/rng resume-state were deleted. Each kept
checkpoint still carries its tokenizer, so every run is loadable for inference / re-eval at
its best operating point. **All eval JSONs in `results/` are intact** — only the weights
behind the non-best `checkpoint-NNNN` paths are gone, recoverable only by retraining.

See [runs.md](runs.md) for the run table · artifacts under `<dir>/04-simcot-pondernet-gammasweep/`.
