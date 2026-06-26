# 09: γ push — does higher KL weight keep moving the frontier left?

**Status:** planned   **Dates:** 2026-06-26 →

## What's being tested

**Goal:** determine whether γ=0.10 (exp-08's best) is near the optimum or whether
increasing γ further (0.15, 0.20) continues to reduce average latent steps without
accuracy loss.

**Motivating finding — exp-08 γ↑ result.** Raising γ from 0.05 to 0.10 uniformly
improved the accuracy–steps frontier: Run B ep5 achieved 41.0% @ 3.64 steps (thr0.5),
+0.2pp and −16% steps vs the exp-07 threshold-only baseline knee. The KL-geometric
term finally "binds" at γ=0.10. The question is whether the trend continues.

| exp | γ | best op. point (thr0.5 ep5) | avg steps | Δ steps vs baseline |
|-----|---|----------------------------|-----------|---------------------|
| 07 (threshold-only knee) | 0.05 | 40.8% @ 4.34 | 4.34 | −36% |
| 08 Run B | 0.10 | 41.0% @ 3.64 | 3.64 | −47% |
| **09 Run A** | **0.15** | TBD | TBD | TBD |

**Hypothesis:** γ=0.15 will pull avg_steps below ~3.0 at thr0.5 while holding
≥40.5% accuracy. If accuracy drops materially (<40%), the model is over-regularized
and 0.10 is the sweet spot. If accuracy holds and steps keep falling, run 0.20.

**Secondary hypothesis:** the Spearman correlation between n_i (ground-truth steps)
and avg-halt-steps should increase further with tighter γ, since the per-instance
prior more sharply enforces difficulty-proportional compute.

## Setup

- **Base recipe:** identical to exp-08 Run B — GPT-2, full warm-start from SIM-CoT
  CODI, `scope=full` (backbone + halt_head trainable; decoder + adapters frozen),
  adaptive per-instance prior, K_max=12, train100k.jsonl, lr 2e-5, 5 epochs, seed 42.
- **Batch:** eff. batch 128 (bs=16, accum=8) — same proven-clean sequence.
- **α=1.0, β=1.5** — same prior shape as exp-08 Run B; only γ changes.
- **Varied:**

  | run | γ | intent |
  |-----|---|--------|
  | `fullscope-adaptive-g0.15-b1.5-k12-ep5` | **0.15** | primary: does the frontier keep moving left? |
  | `fullscope-adaptive-g0.20-b1.5-k12-ep5` | **0.20** | secondary: add only if Run A holds accuracy |

- **GPU:** RTX 5070 (GPU0), `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`.
  Requires torch+cu128 (`uv sync` after applying pytorch-cu128 index — see
  `feat/adaptive-k-from-scratch` commit 2dc97884 for the pyproject.toml change).
- **Eval:** validation split (n=500, greedy, bs=1), thresholds 0.3/0.4/0.5/0.8.

## Launch

```bash
cd /home/tpnlp/adaptive-latent-reasoning/pondernet

# Run A (γ=0.15)
EXP=09-simcot-pondernet-gamma-push RUN=fullscope-adaptive-g0.15-b1.5-k12-ep5 \
SAVE_DIR=../models/checkpoints/09-simcot-pondernet-gamma-push/fullscope-adaptive-g0.15-b1.5-k12-ep5 \
LOG_DIR=../outputs/09-simcot-pondernet-gamma-push/fullscope-adaptive-g0.15-b1.5-k12-ep5 \
ADAPTIVE_PRIOR=True GAMMA=0.15 PRIOR_OFFSET=1.5 PRIOR_SCALE=1.0 \
TRAIN_SCOPE=full DATA_PATH=../data/gsm8k_aug/subsamples/train100k.jsonl \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_gpt2_gsm8k_pondernet.sh \
  --max_latent_steps 12 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 8 \
  --save_total_limit 3 &

# Run B (γ=0.20) — launch only after Run A ep3 eval looks promising
# EXP=09-simcot-pondernet-gamma-push RUN=fullscope-adaptive-g0.20-b1.5-k12-ep5 \
# SAVE_DIR=../models/checkpoints/09-simcot-pondernet-gamma-push/fullscope-adaptive-g0.20-b1.5-k12-ep5 \
# LOG_DIR=../outputs/09-simcot-pondernet-gamma-push/fullscope-adaptive-g0.20-b1.5-k12-ep5 \
# ADAPTIVE_PRIOR=True GAMMA=0.20 PRIOR_OFFSET=1.5 PRIOR_SCALE=1.0 \
# TRAIN_SCOPE=full DATA_PATH=../data/gsm8k_aug/subsamples/train100k.jsonl \
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
# bash scripts/train_gpt2_gsm8k_pondernet.sh ...
```

**Prerequisites before launch:**
1. Apply pytorch-cu128 to pondernet pyproject.toml (cherry-pick from
   `feat/adaptive-k-from-scratch` commit `2dc97884` or apply manually)
2. `uv sync` in `/home/tpnlp/adaptive-latent-reasoning/` — only after the exp-08
   re-run on GPU1 (3090) finishes (ETA ~2026-06-26 05:00 UTC), to avoid overwriting
   the running venv mid-training

## What to record

After each epoch eval:
- accuracy, avg_steps at thr 0.3/0.4/0.5/0.8 (faithful bs=1, validation n=500)
- Spearman(n_i, halt_steps_i) per epoch (from per-instance detail json)
- Compare to exp-08 Run B ep5 baseline at same thresholds

Fill results into [runs.md](runs.md).
