#!/usr/bin/env bash
# Evaluate a PonderNet-trained GPT-2 checkpoint on GSM8K test set.
# Run from pondernet/:
#   CKPT=/path/to/checkpoint bash scripts/eval_gpt2_gsm8k_pondernet.sh
#
# Prints: accuracy, average latent steps used, accuracy-vs-budget table.
# Saves per-instance detail to results/ for offline plotting.

set -euo pipefail

CKPT="${CKPT:?Set CKPT=/path/to/checkpoint}"
GPT2_PATH="${GPT2_PATH:-gpt2}"
RESULTS_DIR="${RESULTS_DIR:-./results/pondernet}"
THRESHOLD="${THRESHOLD:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "$RESULTS_DIR"

python test.py \
    --model_name_or_path "$GPT2_PATH" \
    --ckpt_dir "$CKPT" \
    --data_name gsm8k \
    --results_dir "$RESULTS_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_latent 6 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --bf16 \
    --use_prj True \
    --prj_dim 768 \
    --remove_eos True \
    --greedy True \
    --pondernet True \
    --pondernet_inf_threshold "$THRESHOLD" \
    --pondernet_halt_bias_init -2.0 \
    --train False \
    "$@"
