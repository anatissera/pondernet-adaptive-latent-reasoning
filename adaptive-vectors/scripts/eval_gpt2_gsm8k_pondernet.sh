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
RESULTS_DIR="${RESULTS_DIR:-../results/simcot-pondernet-default}"
THRESHOLD="${THRESHOLD:-0.5}"
# Per-example halting is now faithful for batch>1 (see _slice_past_key_values in
# test.py), so BATCH_SIZE>1 speeds eval without changing results. Default stays 1.
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "$RESULTS_DIR"

uv run python test.py \
    --model_name_or_path "$GPT2_PATH" \
    --ckpt_dir "$CKPT" \
    --data_name gsm8k \
    --results_dir "$RESULTS_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_latent_steps 6 \
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
