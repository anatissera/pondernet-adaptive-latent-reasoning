# 07: full-scope training + per-instance adaptive prior

**Status:** complete   **Dates:** 2026-06-23

## What's being tested

**Hypothesis:** combining the teammate's Recipe C (full backbone unfreeze, `scope=full`)
with our per-instance adaptive prior (exp-05) addresses both failure modes simultaneously:

- Exp-05's adaptive prior improved accuracy (+0.29pp) and difficulty tracking (Spearman
  +0.07) but used a **frozen backbone** (LoRA-only). The per-instance KL targets were
  nudging the halting head while the backbone still carried the K=6 warm-start bias,
  limiting how answer-ready early latent states (z₂–z₃) could become.
- The teammate's Recipe C unfrozen backbone (full scope) gave +0.83pp at 3.21 avg steps,
  because z₃ accuracy rose from ~30% (frozen) to ~37.5% (unfrozen). But Recipe C used a
  global geometric prior, not per-instance.

Combining them should let the backbone reorganize z₂–z₃ representations (full scope) while
the halting head receives a per-instance difficulty target (adaptive prior) — instead of one
or the other in isolation.

K_max=12 is used instead of 6: the teammate's sweep showed C-k12 (40.33%) > C-k6 (39.88%)
despite both halting at ~3.2 steps, because the larger budget sharpens KL pressure toward
early halting and covers the hard-problem tail.

## Setup

- **Backbone:** GPT-2, full warm-start from SIM-CoT CODI checkpoint.
- **Training scope:** `full` — entire backbone (LoRA + non-LoRA weights) + halt_head; auxiliary
  decoder and its projection adapters (`decoder.*`, `pj_in*`, `pj_out*`) stay frozen. Same as
  teammate's Recipe C.
- **Adaptive prior:** `geom_mean_i = α·n_i + β` with α=1.0, β=1.5, clamped to [β, K_max].
  Same as exp-05.
- **K_max:** 12 (from teammate's C-k12 finding).
- **Held fixed:** train100k.jsonl, eff. batch 128 (bs=16, accum=8), lr 2e-5, ep 5, γ=0.05,
  seed 42, no trunc-K.
- **Baselines:**
  - `05/perinstance-g0.05-b1.5-ep5` — adaptive prior, frozen backbone: 40.49% @ 5.26 avg_steps (thr=0.8)
  - teammate's `recipeC-k12` — full scope, global prior: 40.33% @ 3.21 avg_steps (thr=0.5)
- **GPU:** RTX 3090 (CUDA_VISIBLE_DEVICES=1).
- **Eval:** GSM8K test, greedy, bs=1 (faithful), thresholds 0.5/0.8/0.9.

## Launch

```bash
cd /home/tpnlp/alr-valentino/pondernet
EXP=07-simcot-pondernet-fullscope-prior RUN=fullscope-adaptive-g0.05-b1.5-k12-ep5 \
ADAPTIVE_PRIOR=True GAMMA=0.05 PRIOR_OFFSET=1.5 PRIOR_SCALE=1.0 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/train_gpt2_gsm8k_pondernet.sh \
  --pondernet_train_scope full \
  --max_latent_steps 12 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --save_total_limit 6
```

## Findings

**Run:** `fullscope-adaptive-g0.05-b1.5-k12-ep5` — completed 2026-06-23, trained 8h 54m on
RTX 3090 (5 epochs / 3890 steps, bs=16 ga=8 eff-batch=128). Eval on RTX 3060, bs=16,
thresholds 0.5/0.8/0.9, all 5 epochs evaluated.

**Best accuracy: ep5 / thr=0.8 → 40.26% @ 6.96 avg_steps**
**Best Spearman: ep3 / thr=0.5 → 0.684** (project record)

See [fullscope-adaptive-g0.05-b1.5-k12-ep5.md](fullscope-adaptive-g0.05-b1.5-k12-ep5.md) for the full epoch×threshold table, loss trend, and interpretation.

**Verdict:** The combination improved step-difficulty calibration (Spearman +0.026 vs exp-05)
but did not surpass either baseline on accuracy. The adaptive prior's higher target step count
(geom_mean_i ≈ 3.5–5.5 for typical problems) trades compute efficiency for calibration quality
relative to Recipe C's global geom_mean=3.0. Full-scope convergence appears to need more than
5 epochs for accuracy to exceed exp-05's 40.49%.
