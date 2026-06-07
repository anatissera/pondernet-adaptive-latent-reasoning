# adaptive-latent-reasoning

NLP final project (5-person team, 2 subgroups). Goal: make the number of latent reasoning steps **adaptive** in a latent CoT model, building on SIM-CoT (ICLR 2026).

## Repo Structure

- `baselines/` — upstream reference implementations; treat as read-only unless actively extending
  - `Coconut/` — Coconut latent CoT (GSM8K, ProntoQA tasks)
  - `CODI/` — CODI implicit CoT; Subgroup 1's active working area (PonderNet extensions)
- `k-classifier/` — Subgroup 2's k-sweep multi-backend pipeline (Coconut + CODI backends)
- `docs/papers/` — paper summaries; read the relevant one before touching related code
- `docs/exploration.md` — team research notes, hypotheses, and experiment log

## Two Subgroups

| Branch | Strategy | Active Working Area |
|--------|----------|---------------------|
| `develop-c` | PonderNet-style adaptive halting (phases 1–6 complete) | `baselines/CODI/` |
| `option-a` | Fixed-k sweep, multi-backend evaluation | `k-classifier/` |

## Key Papers

Read the summary in `docs/papers/` before touching related code:

- `coconut.md` — Coconut latent CoT (basis for Option-A backend)
- `codi.md` — CODI implicit CoT (basis for Subgroup 1's work)
- `sim-cot.md` — SIM-CoT, the upstream this project forks; understand this first
- `pondernet.md` — PonderNet adaptive halting (Subgroup 1's core extension)

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
