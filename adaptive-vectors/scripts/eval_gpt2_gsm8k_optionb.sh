#!/usr/bin/env bash
# Evaluate an Option-B-trained GPT-2 checkpoint on the GSM8K test set.
# Run from adaptive-vectors/:  CKPT=/path/to/checkpoint bash scripts/eval_gpt2_gsm8k_optionb.sh
#
# Prints: accuracy, avg vectors/step, n_k distribution, accuracy-vs-budget table.
# BATCH_SIZE must be 1 for faithful per-step halting (shared-cache caveat, see test.py).
# Runs on the RTX 3060 (PCI idx 2).
set -euo pipefail

CKPT="${CKPT:?Set CKPT=/path/to/checkpoint}"
GPT2_PATH="${GPT2_PATH:-gpt2}"
RESULTS_DIR="${RESULTS_DIR:-../results/optionb-default}"
K="${K:-4}"; M="${M:-3}"; EPS="${EPS:-0.01}"; MMAX="${MMAX:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "$RESULTS_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python test.py \
    --model_name_or_path "$GPT2_PATH" \
    --ckpt_dir "$CKPT" \
    --data_name gsm8k \
    --results_dir "$RESULTS_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_latent_steps 6 \
    --use_lora True --lora_r 128 --lora_alpha 32 --lora_init \
    --bf16 --use_prj True --prj_dim 768 --remove_eos True --greedy True \
    --option_b True --ob_num_steps "$K" --ob_subvectors_per_step "$M" \
    --ob_mlp_hidden 256 --ob_max_subvectors "$MMAX" --ob_eps "$EPS" \
    --train False \
    "$@"
