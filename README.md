# adaptive-latent-reasoning

NLP final project — adaptive latent chain-of-thought reasoning.

We extend [SIM-CoT](https://arxiv.org/pdf/2509.20317) (ICLR 2026) to make the number of implicit reasoning steps **variable at inference time**, rather than fixed.


## Structure

```
baselines/           # Reference implementations (Coconut, CODI, SIM-CoT) — read-only
pondernet/           # PonderNet adaptive halting — Subgroup 1's active working area
models/checkpoints/  # Trained runs, grouped as <NN-exp>/<run-id>/ (gitignored)
outputs/             # Training logs + TensorBoard events, <NN-exp>/<run-id>/ (gitignored)
results/             # Eval outputs, <NN-exp>/<run-id>/ (gitignored)
data/                # Evaluation datasets
docs/experiments.md  # Experiment index → per-experiment → per-run docs (git-tracked)
docs/papers/         # Paper summaries for context
```

Runs are launched with `EXP=<NN-exp> RUN=<run-id>` and recorded via the `/log-run` skill;
see [`docs/pipeline.md`](docs/pipeline.md) and [`docs/experiments.md`](docs/experiments.md).

## Setup

```bash
uv sync
```

## Context for contributors

See `AGENTS.md` for a full briefing on repo layout, subgroup strategies, and how to run the baselines.
