# adaptive-latent-reasoning

NLP final project (5-person team, 2 subgroups). Goal: make the number of latent reasoning steps **adaptive** in a latent CoT model, building on SIM-CoT (ICLR 2026).

## Repo Structure

- `baselines/` — upstream reference implementations; treat as read-only unless actively extending
  - `Coconut/` — Coconut latent CoT (GSM8K, ProntoQA tasks)
  - `CODI/` — CODI implicit CoT; Subgroup 1's active working area (PonderNet extensions)
- `docs/methods-comparison.md` — cross-paper comparison table and chain-of-influence narrative; read this for method context
- `docs/papers/` — full paper content (raw); only read if you need deeper detail beyond the comparison doc

## Two Subgroups

| Branch | Strategy | Active Working Area |
|--------|----------|---------------------|
| `develop-c` | PonderNet-style adaptive halting (phases 1–6 complete) | `baselines/CODI/` |
| `option-a` | Fixed-k sweep, multi-backend evaluation | `k-classifier/` |

## Running Baselines

```bash
# Coconut (from baselines/Coconut/)
python run.py args/gsm_coconut.yaml

# CODI — PonderNet variant (from baselines/CODI/)
bash scripts/train_gpt2_gsm8k_pondernet.sh
```

## Setup

```bash
uv sync   # or: pip install -r baselines/CODI/requirements.txt
```
