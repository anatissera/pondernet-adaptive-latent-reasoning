#!/usr/bin/env bash
# Option-B Phase-1 feasibility probe (GO/NO-GO gate).
# Runs the model with --ob_probe on a handful of batches and logs, for each
# reasoning step, the decoder reconstruction loss L_step from each of M
# sub-vectors. We want L_step to DECREASE across sub-vectors within a step.
#
# Read-only diagnostic: returns a zero loss, trains nothing.
# Runs on the RTX 3060 (PCI index 2) per project convention.
#   3060 = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
#
# Run from adaptive-vectors/:  bash scripts/ob_probe.sh
set -euo pipefail

GPT2_PATH="${GPT2_PATH:-gpt2}"
SIMCOT_CKPT="${SIMCOT_CKPT-../models/pretrained/simcot-gpt2-codi/model-00001-of-00001.safetensors}"
DECODER_PATH="${DECODER_PATH:-../models/pretrained/simcot-gpt2-decoder}"
DATA_DIR="${DATA_DIR:-../data}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/gsm8k_aug/train15k.jsonl}"
LOG_DIR="${LOG_DIR:-../outputs/optionb-probe}"
M="${M:-4}"

mkdir -p "$LOG_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # 3060
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[probe] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (expect RTX 3060)"
uv run python -c "import torch; print('[probe] device:', torch.cuda.get_device_name(0))"

uv run python train.py \
    --output_dir "$LOG_DIR" \
    --logging_dir "$LOG_DIR" \
    --logging_steps 1 \
    --model_name_or_path "$GPT2_PATH" \
    --data_name icot \
    --data_path "$DATA_PATH" \
    --seed 42 \
    --model_max_length 384 \
    --max_token_num 700 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing False \
    --dataloader_num_workers 2 \
    --bf16 \
    --max_steps 10 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy no \
    --save_safetensors False \
    --do_train \
    --report_to none \
    --max_latent_steps 6 \
    --use_prj True --prj_dim 768 --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --remove_eos True \
    --use_decoder True \
    --decoder_path "$DECODER_PATH" \
    --simcot_ckpt "$SIMCOT_CKPT" \
    --print_loss False \
    --max_train_samples 320 \
    --option_b True \
    --ob_probe True \
    --ob_subvectors_per_step "$M" \
    2>&1 | tee "$LOG_DIR/probe.log"

echo "[probe] done. Summary of L_step-by-subvector lines:"
grep "sub-vector position" "$LOG_DIR/probe.log" || true
