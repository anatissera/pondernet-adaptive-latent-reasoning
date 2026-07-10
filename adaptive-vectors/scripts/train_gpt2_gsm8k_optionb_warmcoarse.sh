#!/usr/bin/env bash
# Option-B WARM + COARSE: the missing cell of the 2x2 (init x granularity).
# Keeps the SIM-CoT/CODI warm-start (=> a model whose latents actually compute, unlike the
# cold collapse) but switches to COARSE step segmentation (group ops into K=3 buckets =>
# variable per-step difficulty). Isolates the granularity variable on a WORKING model:
#   if the fixed-c curve steepens past 2 here -> per-step density IS the lever, a harder
#   dataset is worth it; if still flat -> grouping trivial ops isn't enough.
# Recipe mirrors the warm baseline (LR 2e-5, 3 epochs, LoRA r128) + K=3 coarse, penalty off.
# Runs on a free GPU (NOT the 3060). gradient_checkpointing MUST stay False.
#
# Run from adaptive-vectors/:  bash scripts/train_gpt2_gsm8k_optionb_warmcoarse.sh
set -euo pipefail

SAVE_DIR="${SAVE_DIR:-../models/checkpoints/optionb-warm-coarse}"
LOG_DIR="${LOG_DIR:-../outputs/optionb-warm-coarse}"
GPT2_PATH="${GPT2_PATH:-gpt2}"
SIMCOT_CKPT="${SIMCOT_CKPT-../models/pretrained/simcot-gpt2-codi/model-00001-of-00001.safetensors}"
DECODER_PATH="${DECODER_PATH:-../models/pretrained/simcot-gpt2-decoder}"
DATA_PATH="${DATA_PATH:-../data/gsm8k_aug/train15k.jsonl}"

K="${K:-3}"; M="${M:-3}"                 # K=3 coarse buckets, up to M=3 sub-vectors each
BS="${BS:-8}"; ACCUM="${ACCUM:-4}"
EPOCHS="${EPOCHS:-3}"; MAXSAMPLES="${MAXSAMPLES:-8000}"
LR="${LR:-2e-5}"                         # warm baseline LR
LAMBDA_HALT="${LAMBDA_HALT:-0.0}"        # penalty OFF: clean c-curve headroom test first
export HALT_HEAD_LR="${HALT_HEAD_LR:-1e-3}"
export UV_NO_SYNC=1

mkdir -p "$SAVE_DIR" "$LOG_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # default 3090; NEVER the 3060 (idx 2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[warm-coarse] device:"; uv run python -c "import torch;print(' ', torch.cuda.get_device_name(0))"
echo "[warm-coarse] WARM start + COARSE steps | K=$K M=$M BS=$BS ACCUM=$ACCUM EPOCHS=$EPOCHS N=$MAXSAMPLES LR=$LR LAMBDA_HALT=$LAMBDA_HALT"

uv run python train.py \
    --output_dir "$SAVE_DIR" --logging_dir "$LOG_DIR" --logging_steps 20 \
    --model_name_or_path "$GPT2_PATH" --data_name icot --data_path "$DATA_PATH" \
    --seed 42 --model_max_length 384 --max_token_num 700 \
    --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$ACCUM" \
    --gradient_checkpointing False --dataloader_num_workers 4 --bf16 \
    --num_train_epochs "$EPOCHS" --learning_rate "$LR" --max_grad_norm 1.0 \
    --use_lora True --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy epoch --save_total_limit 2 --save_safetensors False \
    --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --do_train --report_to tensorboard \
    --use_prj True --prj_dim 768 --prj_dropout 0.0 \
    --distill_loss_div_std True --remove_eos True \
    --use_decoder True --decoder_path "$DECODER_PATH" --simcot_ckpt "$SIMCOT_CKPT" \
    --print_loss False --max_train_samples "$MAXSAMPLES" \
    --option_b True --ob_num_steps "$K" --ob_subvectors_per_step "$M" \
    --ob_coarse_steps True --ob_mlp_hidden 256 --ob_detach_hk True \
    --ob_lambda_ans 1.0 --ob_lambda_step 1.0 --ob_lambda_dist 1.0 --ob_lambda_halt "$LAMBDA_HALT" \
    2>&1 | tee "$LOG_DIR/train.log"
