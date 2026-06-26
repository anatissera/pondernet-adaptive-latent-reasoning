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
#   * LR=3e-3               -> CODI GPT-2 from-scratch LR (SIM-CoT paper Table E.3), NOT the
#                              warm-start 2e-5 (which would barely move a cold model)
#   * EPOCHS=40            -> CODI GPT-2 budget (SIM-CoT paper Table E.3), NOT 5
# Everything else (data, eff. batch 128, seed, geom prior shape) matches exp-08 Run C.
#
# Launch with EXP/RUN so the run is named consistently and artifacts are grouped:
#   EXP=10-simcot-pondernet-fromscratch RUN=fromscratch-runC-g0.10-a0.6-b1.5-k12-ep40 \
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle-gpu> \
#   bash scripts/train_gpt2_gsm8k_pondernet_fromscratch.sh
#
# Tunables (env overrides): LR, EPOCHS, BS, ACCUM, DATA_PATH, GAMMA, PRIOR_OFFSET, PRIOR_SCALE.
set -euo pipefail

# --- from-scratch recipe (override via env) ---------------------------------------------
export GPT2_PATH="${GPT2_PATH:-gpt2}"          # cold backbone
export SIMCOT_CKPT=""                          # MUST be empty: no full-model warm-start
export DECODER_PATH="${DECODER_PATH:-gpt2}"    # aux decoder starts from vanilla GPT-2
export LR="${LR:-3e-3}"                         # CODI GPT-2 from-scratch LR (paper Table E.3)
export ADAPTIVE_PRIOR="${ADAPTIVE_PRIOR:-True}"
export GAMMA="${GAMMA:-0.10}"                   # Run C
export PRIOR_OFFSET="${PRIOR_OFFSET:-1.5}"      # Run C: β=1.5 (also avoids the masked_scatter crash at β<=1)
export PRIOR_SCALE="${PRIOR_SCALE:-0.6}"        # Run C: α=0.6

# Full GSM8k-Aug (385,620 ex) -- the faithful SIM-CoT data budget, NOT the 100k subsample.
export DATA_PATH="${DATA_PATH:-../data/gsm8k_aug/train.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-400000}"           # >= dataset size -> use every example
EPOCHS="${EPOCHS:-40}"                          # CODI GPT-2 budget (paper Table E.3)
BS="${BS:-16}"                                  # exp-08 proven-clean batch sequence ...
ACCUM="${ACCUM:-8}"                             # ... eff. batch = BS*ACCUM = 128

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_gpt2_gsm8k_pondernet.sh" \
  --pondernet_train_scope full_dec \
  --max_latent_steps 12 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BS" \
  --gradient_accumulation_steps "$ACCUM" \
  --max_train_samples "$MAX_SAMPLES" \
  --save_total_limit 3 \
  "$@"
