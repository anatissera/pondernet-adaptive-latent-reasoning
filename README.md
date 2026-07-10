# Option A: an upfront predictor of the latent reasoning budget

This branch is the final state of **Option A** of the adaptive latent reasoning
project: instead of always running a fixed number of latent reasoning steps, a
lightweight auxiliary classifier looks at the prompt and predicts, before the main
model runs, how many latent steps `k` it should use:

```
task prompt -> classifier -> estimated k -> model reasons with that k -> answer
```

For the project overview and the other two approaches (Option B: adaptive vectors per
step; Option C: PonderNet adaptive halting), see the `main` branch.

## What was done and found

All the work lives in [`k-classifier/`](k-classifier/README.md):

1. **k sweeps** over GSM8K with the SIM-CoT CODI and Coconut checkpoints established
   that the latent budget matters: outputs change with k in ~97% of examples, and
   accuracy on the full train sweep (n=7473) climbs from 40.8% at k=1 to ~64% at
   k=5-6, then saturates.
2. **Oracle labels** `k_star` (the smallest k reaching each example's best score) are
   heavily imbalanced: 66% of examples peak at k=1, with a thin tail at high k. This
   drove the classifier design (regression over ordinal k rather than multiclass; see
   [`k-classifier/docs/classifier-design.md`](k-classifier/docs/classifier-design.md)).
3. **Two classifier variants** were built: a regression head over frozen
   all-MiniLM-L6-v2 sentence embeddings, and a multi-output DistilBERT classifier
   trained with `BCEWithLogitsLoss` that predicts which budgets are likely to solve a
   prompt.

Full details, commands, and results: [`k-classifier/README.md`](k-classifier/README.md).

## Structure

```
k-classifier/            # The Option A pipeline: sweeps, labels, classifiers
baselines/           # Reference implementations (Coconut, CODI, SIM-CoT), read-only
docs/papers/         # Paper summaries for context
```

## Setup and environment

Option A runs on its own environment (`.venv-option-a`, `k-classifier/requirements.txt`),
not the root `pyproject.toml`. See [`k-classifier/README.md`](k-classifier/README.md) for the
commands and `k-classifier/AGENTS.md` for the VM environment notes (GCP T4, system torch,
venv recipe, persistence between sessions).
