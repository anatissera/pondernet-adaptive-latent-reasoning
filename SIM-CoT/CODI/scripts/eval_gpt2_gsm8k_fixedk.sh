#!/usr/bin/env bash
# Evaluate a standard SIM-CoT (fixed-K) GPT-2 checkpoint on GSM8K.
# Use different --num_latent and --inf_latent_iterations values for
# the fixed-K baseline comparison in Phase 6.
#
# Run from SIM-CoT/CODI/:
#   CKPT=/path/to/checkpoint NUM_LATENT=6 bash scripts/eval_gpt2_gsm8k_fixedk.sh

set -euo pipefail

CKPT="${CKPT:?Set CKPT=/path/to/checkpoint}"
GPT2_PATH="${GPT2_PATH:-gpt2}"
RESULTS_DIR="${RESULTS_DIR:-./results/fixedk}"
NUM_LATENT="${NUM_LATENT:-6}"

mkdir -p "$RESULTS_DIR"

python test.py \
    --model_name_or_path "$GPT2_PATH" \
    --ckpt_dir "$CKPT" \
    --data_name gsm8k \
    --results_dir "$RESULTS_DIR" \
    --batch_size 1 \
    --num_latent "$NUM_LATENT" \
    --inf_latent_iterations "$NUM_LATENT" \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --bf16 \
    --use_prj True \
    --prj_dim 768 \
    --remove_eos True \
    --greedy True \
    --train False \
    "$@"
