#!/usr/bin/env bash
# Train PonderNet adaptive halting FROM SCRATCH (vanilla GPT-2), no SIM-CoT warm-start.
# Run from pondernet/:  bash scripts/train_gpt2_gsm8k_pondernet_fromscratch.sh
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The normal recipe (train_gpt2_gsm8k_pondernet.sh) warm-starts the whole CODI model
# (backbone + LoRA + prj + aux-decoder) from the downloaded SIM-CoT CODI checkpoint and
# only fine-tunes 5 epochs at lr 2e-5. That means the *latent-reasoning ability* is
# inherited for free from ~40 epochs of someone else's CODI training -- so our accuracy
# is not attributable to our method alone.
#
# This script trains the SAME model + method (exp-08 Run C: γ=0.10, α=0.6, β=1.5 adaptive
# prior, K_max=12) but from vanilla GPT-2, paying the full from-scratch cost ourselves, so
# the comparison against SIM-CoT is fair. The ONLY differences vs the warm-start recipe are
# the things a from-scratch run requires:
#   * SIMCOT_CKPT=""        -> no full-model warm-start (cold backbone)
#   * DECODER_PATH=gpt2     -> aux decoder is vanilla GPT-2 (not SIM-CoT's trained decoder)
#   * --pondernet_train_scope full_dec  -> ALSO train the decoder (SIM-CoT learns it; the
#                              other scopes freeze it because they reused SIM-CoT's)
#   * LR=1e-3 + warmup 0.05 -> CORRECTED stable LR (see below), NOT the warm-start 2e-5
#                              (which would barely move a cold model)
#   * EPOCHS=40            -> CODI GPT-2 budget (SIM-CoT paper Table E.3), NOT 5
# Everything else (data, eff. batch 128, seed, geom prior shape) matches exp-08 Run C.
#
# ⚠️ CORRECTED LR (2026-06-28). The paper's from-scratch LR is 3e-3 (Table E.3), and this
# script used to default to it. EMPIRICALLY, 3e-3 DIVERGES in our bf16 setup: a real run on
# an L4 ran clean ~2200 steps then blew up at ~step 5820 (grad_norm 20 -> 2.5e11 in 10 steps,
# NaN/Inf, ce stuck ~11). A probe at LR=1e-3 + warmup_ratio=0.05 stayed stable through the
# same region (grad_norm ~0.5 flat) and trained cleanly. So the validated from-scratch recipe
# is LR=1e-3, warmup 0.05. This is the only deviation from the faithful SIM-CoT base and it is
# forced by numerical stability, not a method change. See docs/experiments/10-.../experiment.md.
#
# Launch with EXP/RUN so the run is named consistently and artifacts are grouped:
#   EXP=10-simcot-pondernet-fromscratch RUN=fromscratch-runC-lr1e-3-g0.10-a0.6-b1.5-k12-ep40 \
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle-gpu> \
#   bash scripts/train_gpt2_gsm8k_pondernet_fromscratch.sh
#
# Tunables (env overrides): LR, WARMUP, EPOCHS, BS, ACCUM, SAVE_LIMIT, DATA_PATH, GAMMA,
#                           PRIOR_OFFSET, PRIOR_SCALE, CKPT_KEEP_FULL, CKPT_KEEP_WEIGHTS.
# A100 note: on a big card override BS/ACCUM to keep eff.batch 128 but go faster, e.g.
#   BS=32 ACCUM=4   (A100-40GB)   or   BS=64 ACCUM=2   (A100-80GB).
# Disk: a CheckpointPruneCallback (train.py) keeps the newest CKPT_KEEP_FULL (default 2)
# checkpoints fully resumable, and from older ones strips only the resume-only optimizer/
# scheduler/rng state while keeping weights + jsons (so every epoch stays evaluable for the
# accuracy-vs-steps graphs). ~0.58 GB/epoch instead of 1.2. Set CKPT_KEEP_WEIGHTS=N to also
# drop weights from epochs older than the newest N (keep only jsons) if you need minimal disk.
set -euo pipefail

# --- from-scratch recipe (override via env) ---------------------------------------------
export GPT2_PATH="${GPT2_PATH:-gpt2}"          # cold backbone
export SIMCOT_CKPT=""                          # MUST be empty: no full-model warm-start
export DECODER_PATH="${DECODER_PATH:-gpt2}"    # aux decoder starts from vanilla GPT-2
export LR="${LR:-1e-3}"                         # CORRECTED stable LR (3e-3 from the paper diverges in bf16; see header)
WARMUP="${WARMUP:-0.05}"                        # corrected warmup (paper/base is 0.03; 0.05 damps the cold-start spike)
export ADAPTIVE_PRIOR="${ADAPTIVE_PRIOR:-True}"
export GAMMA="${GAMMA:-0.10}"                   # Run C
export PRIOR_OFFSET="${PRIOR_OFFSET:-1.5}"      # Run C: β=1.5 (also avoids the masked_scatter crash at β<=1)
export PRIOR_SCALE="${PRIOR_SCALE:-0.6}"        # Run C: α=0.6

# Full GSM8k-Aug (385,620 ex) -- the faithful SIM-CoT data budget, NOT the 100k subsample.
# Create it first with:  python scripts/prep_gsm8k_aug.py   (downloads from the HF hub).
export DATA_PATH="${DATA_PATH:-../data/gsm8k_aug/train.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-400000}"           # >= dataset size -> use every example
EPOCHS="${EPOCHS:-40}"                          # CODI GPT-2 budget (paper Table E.3)
BS="${BS:-16}"                                  # exp-08 proven-clean batch sequence (L4-safe) ...
ACCUM="${ACCUM:-8}"                             # ... eff. batch = BS*ACCUM = 128
SAVE_LIMIT="${SAVE_LIMIT:-40}"                  # keep ALL 40 epoch checkpoints for model selection (~48GB)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Hyperparameters aligned to the upstream CODI GPT-2 recipe so the base is a faithful
# SIM-CoT (baselines/CODI/scripts/train_gpt2_gsm8k-aug-decoder-2.sh): model_max_length 512
# and max_grad_norm 2.0 (our main script defaults 384 / 1.0). LoRA (r128/α32/dropout0.1/
# c_attn,c_proj,c_fc/init), lr 3e-3, 40 ep, eff. batch 128, wd 0.1, warmup 0.03, cosine,
# prj_dim 768, distill_loss_div_std all already match. Deliberate deviations from the CODI
# base: K_max=12 (vs fixed num_latent=6 - the whole point is adaptive halting) and seed 42
# (our experiment convention vs upstream 11).
exec bash "$SCRIPT_DIR/train_gpt2_gsm8k_pondernet.sh" \
  --pondernet_train_scope full_dec \
  --max_latent_steps 12 \
  --model_max_length 512 \
  --max_grad_norm 2.0 \
  --warmup_ratio "$WARMUP" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BS" \
  --gradient_accumulation_steps "$ACCUM" \
  --max_train_samples "$MAX_SAMPLES" \
  --save_total_limit "$SAVE_LIMIT" \
  "$@"
