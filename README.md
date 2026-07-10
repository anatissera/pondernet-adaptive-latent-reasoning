# Option C: task-optimized adaptive halting (PonderNet)

This branch is the final state of **Option C** of the adaptive latent reasoning project:
learning *when to stop* the latent chain-of-thought directly from the task signal,
following the PonderNet formulation (Banino et al., 2021). Each latent step emits a
conditional halting probability, and at inference the model stops as soon as the
accumulated halting mass crosses a threshold, so easy problems use fewer latent steps
than hard ones.

The base system is [SIM-CoT](https://arxiv.org/pdf/2509.20317) (ICLR 2026), which runs a
*fixed* number K of implicit reasoning steps. Option C makes that number variable per
instance at inference time. For the project overview and the other two approaches
(Option A: an upfront k* classifier; Option B: adaptive vectors per step), see the
`main` branch.

## Results in one paragraph

On GSM8K with a held-out 500-example validation split, the greedy fixed-K baseline is
**40.80%** at K=6. The best adaptive runs match it (41.40% at 5.34 steps, 41.20% at
ep5/thr0.5), so the accuracy gain is within noise. The durable result is **compute
adaptivity at matched accuracy**: the per-instance geometric prior and full-scope
training make halting track difficulty strongly (Spearman between steps used and
expression count rises from +0.46 to +0.675), and the gamma-frontier runs cut average
steps by up to 32% (40.6% at 2.93 steps) without losing accuracy. The full story,
including the negative results and a validation-split correction that retired earlier
headlines, is in [`docs/experiments.md`](docs/experiments.md).

## Repository layout

```
pondernet/           # The Option C codebase: train.py, test.py, src/model.py, scripts/
baselines/           # Reference implementations (Coconut, CODI, SIM-CoT), read-only
docs/experiments.md  # Experiment index (01-11) with per-experiment and per-run docs
docs/pipeline.md     # End-to-end workflow: artifacts -> train -> eval -> record
docs/parameters.md   # CLI flag reference and warm-start recipes
docs/papers/         # Paper summaries for context
models/, outputs/, results/, data/   # Run artifacts (gitignored, shared on disk)
```

## Setup

```bash
uv sync
```

Pretrained artifacts (GPT-2, the SIM-CoT CODI checkpoint, and the extracted decoder)
are gitignored; see step 1 of [`docs/pipeline.md`](docs/pipeline.md) for where each one
comes from.

## Running an experiment

Every run is identified by an experiment folder and a run id, which the scripts use to
derive checkpoint, log, and result paths:

```bash
cd pondernet

# Train (example: the gamma sweep point gamma=0.05, geometric mean 3.0, 5 epochs)
EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/train_gpt2_gsm8k_pondernet.sh

# Evaluate on the held-out validation split
EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5 \
  bash scripts/eval_gpt2_gsm8k_pondernet.sh
```

Key hyperparameters (all documented in [`pondernet/README.md`](pondernet/README.md) and
[`docs/parameters.md`](docs/parameters.md)):

| Flag | Meaning |
|---|---|
| `--max_latent_steps` | Upper bound K_max on latent steps |
| `--pondernet_gamma` | Weight of the KL term pulling halting toward the geometric prior; higher gamma means earlier halting |
| `--pondernet_geom_mean` | Mean of the geometric prior (expected number of steps) |
| `--pondernet_inf_threshold` | Inference-time halting threshold |
| `--pondernet_beta` | Weight of the per-step auxiliary decoder loss |

## Experiment index

See [`docs/experiments.md`](docs/experiments.md) for the full table with numbers. Summary:

| # | Experiment | One-line takeaway |
|---|---|---|
| 01 | simcot-baselines | Fixed-K reference: 40.80% greedy at K=6 |
| 02 | simcot-pondernet-early | First runs, broken KV cache, superseded |
| 03 | simcot-pondernet-gcfix | Gradient-checkpointing fix recovers baseline accuracy |
| 04 | simcot-pondernet-gammasweep | Gamma trains halting, threshold tunes it: a clean accuracy/steps frontier |
| 05 | simcot-pondernet-adaptive-prior | Per-instance geometric prior lifts difficulty tracking |
| 06 | simcot-pondernet-trunc-k | Per-instance truncated-K training, negative result |
| 07 | simcot-pondernet-fullscope-prior | Full unfreeze + adaptive prior: best difficulty tracking (Spearman +0.675) |
| 08 | simcot-pondernet-gamma-frontier | Pushes the frontier left: 40.6% at 2.93 steps (-32% compute) |
| 09 | simcot-pondernet-gamma-push | Planned: does the frontier keep moving left past gamma 0.10? |
| 10 | simcot-pondernet-fromscratch | Planned fair comparison from vanilla GPT-2 (no warm start) |
| 11 | simcot-pondernet-k-recipe | Recipe x K_max factorial; the C > A headline retired after validation re-eval |

## Context for contributors

See `AGENTS.md` for the agent briefing and `docs/pipeline.md` for the full workflow,
including the experiment-scaffolding and run-recording conventions used to keep this
index consistent.
