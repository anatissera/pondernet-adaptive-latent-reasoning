#!/usr/bin/env bash
# Option-B overfit smoke test. Trains on a tiny fixed batch and prints the per-step
# [OB] loss line so we can confirm the answer CE and the MLP distillation loss L_dist
# both DECREASE. Runs on the RTX 3060 (PCI idx 2).
#
# Usage:  bash scripts/ob_smoke.sh            # Phase-2 check: penalty off (LAMBDA_HALT=0)
#         LAMBDA_HALT=0.05 bash scripts/ob_smoke.sh   # Phase-3 check: penalty on
set -euo pipefail

GPT2_PATH="${GPT2_PATH:-gpt2}"
SIMCOT_CKPT="${SIMCOT_CKPT-../models/pretrained/simcot-gpt2-codi/model-00001-of-00001.safetensors}"
DECODER_PATH="${DECODER_PATH:-../models/pretrained/simcot-gpt2-decoder}"
DATA_PATH="${DATA_PATH:-../data/gsm8k_aug/train15k.jsonl}"
LOG_DIR="${LOG_DIR:-../outputs/optionb-smoke}"
K="${K:-4}"; M="${M:-3}"; LAMBDA_HALT="${LAMBDA_HALT:-0.0}"

mkdir -p "$LOG_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[smoke] device check:"; uv run python -c "import torch;print(' ', torch.cuda.get_device_name(0))"

uv run python train.py \
    --output_dir "$LOG_DIR" --logging_dir "$LOG_DIR" --logging_steps 1 \
    --model_name_or_path "$GPT2_PATH" --data_name icot --data_path "$DATA_PATH" \
    --seed 42 --model_max_length 384 --max_token_num 700 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
    --gradient_checkpointing False --dataloader_num_workers 2 --bf16 \
    --max_steps 40 --num_train_epochs 50 --learning_rate 1e-4 --max_grad_norm 1.0 \
    --use_lora True --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy no --save_safetensors False --do_train --report_to none \
    --use_prj True --prj_dim 768 --prj_dropout 0.0 \
    --distill_loss_div_std True --remove_eos True \
    --use_decoder True --decoder_path "$DECODER_PATH" --simcot_ckpt "$SIMCOT_CKPT" \
    --print_loss True --max_train_samples 8 \
    --option_b True --ob_num_steps "$K" --ob_subvectors_per_step "$M" \
    --ob_lambda_ans 1.0 --ob_lambda_step 1.0 --ob_lambda_dist 1.0 --ob_lambda_halt "$LAMBDA_HALT" \
    2>&1 | tee "$LOG_DIR/smoke.log"

echo "[smoke] first vs last [OB] lines:"
grep '\[OB\]' "$LOG_DIR/smoke.log" | head -1
grep '\[OB\]' "$LOG_DIR/smoke.log" | tail -1
