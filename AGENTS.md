# adaptive-latent-reasoning

NLP final project (5-person team, 2 subgroups). Goal: make the number of latent reasoning steps **adaptive** in a latent CoT model, building on SIM-CoT (ICLR 2026).

## Repo Structure

- `baselines/` — upstream reference implementations; treat as read-only
  - `Coconut/` — Coconut latent CoT (GSM8K, ProntoQA tasks)
  - `CODI/` — CODI implicit CoT baseline (upstream reference)
- `pondernet/` — PonderNet adaptive halting built on CODI; Subgroup 1's active working area
- `models/` — pre-trained and fine-tuned weights (gitignored); load with `GPT2_PATH=../models/<name>`
- `outputs/` — training logs and TensorBoard events (gitignored); one subdir per run
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
bash scripts/train_gpt2_gsm8k_pondernet.sh
```

## Setup

```bash
uv sync
```
