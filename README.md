# adaptive-latent-reasoning

NLP final project — adaptive latent chain-of-thought reasoning.

We extend [SIM-CoT](https://arxiv.org/pdf/2509.20317) (ICLR 2026) to make the number of implicit reasoning steps **variable at inference time**, rather than fixed.


## Structure

```
baselines/           # Reference implementations (Coconut, CODI, SIM-CoT) — read-only
pondernet/           # PonderNet adaptive halting — Subgroup 1's active working area
models/              # Pre-trained / fine-tuned model weights (gitignored)
outputs/             # Training logs and TensorBoard events (gitignored)
data/                # Evaluation datasets
docs/papers/         # Paper summaries for context
```

## Setup

```bash
uv sync
# or
pip install -r pondernet/requirements.txt
```

## Context for contributors

See `AGENTS.md` for a full briefing on repo layout, subgroup strategies, and how to run the baselines.
