# adaptive-latent-reasoning

NLP final project (5-person team, 2 subgroups). Goal: make the number of latent reasoning steps **adaptive** in a latent CoT model, building on SIM-CoT (ICLR 2026).

## Repo Structure

- `baselines/` — upstream reference implementations; treat as read-only
  - `Coconut/` — Coconut latent CoT (GSM8K, ProntoQA tasks)
  - `CODI/` — CODI implicit CoT baseline (upstream reference)
- `pondernet/` — PonderNet adaptive halting built on CODI; Subgroup 1's active working area
- `models/` — model weights (gitignored), split by provenance:
  - `pretrained/` — downloaded backbones + decoder: `gpt2`, `simcot-gpt2-codi`, `simcot-gpt2-coconut`, `simcot-gpt2-decoder` (fetch the decoder with `pondernet/scripts/fetch_simcot_decoder.py`)
  - `checkpoints/<run-id>/` — our trained runs
- `outputs/` — training logs and TensorBoard events (gitignored); one subdir per run-id
- `results/` — evaluation outputs (gitignored); one subdir per run-id
- `data/` — datasets (gitignored); `gsm8k_aug/` holds the GSM8k-Aug jsonl, training is pinned to `train15k.jsonl` by default
- A single **run-id** names the same experiment across `models/checkpoints/`, `outputs/`, and `results/` (see `docs/runs.md`)
- `docs/pipeline.md` — end-to-end training/eval workflow + diagram; **read this first to train a model**
- `docs/parameters.md` — CLI flag reference, warm-start recipes, and the kept-name glossary
- `docs/runs.md` — run manifest (run-id, date, hparams, accuracy)
- `docs/methods-comparison.md` — cross-paper comparison table and chain-of-influence narrative; read this for method context
- `docs/papers/` — full paper content (raw); only read if you need deeper detail beyond the comparison doc

## Two Subgroups

| Branch | Strategy | Active Working Area |
|--------|----------|---------------------|
| `pondernet` | PonderNet-style adaptive halting (phases 1–6 complete) | `pondernet/` |

## Running Baselines

```bash
# Coconut (from baselines/Coconut/)
python run.py args/gsm_coconut.yaml

# PonderNet — adaptive halting variant (from pondernet/)
# Trains on the pinned data/gsm8k_aug/train15k.jsonl by default.
SAVE_DIR=../models/checkpoints/<run-id> LOG_DIR=../outputs/<run-id> \
  bash scripts/train_gpt2_gsm8k_pondernet.sh
```

See `docs/pipeline.md` for the full train → eval → record workflow and `docs/parameters.md` for every flag.

## Setup

```bash
uv sync
```
