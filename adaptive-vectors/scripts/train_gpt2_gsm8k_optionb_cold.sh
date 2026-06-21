#!/usr/bin/env bash
# Option-B UNBIASED cold-start training: plain GPT-2, NO SIM-CoT warm-start, fresh
# decoder, coarse step segmentation. Removes the two controllable biases (c=1 anchor +
# atomic granularity) so we can ask whether the c-axis has real headroom.
# Recipe mirrors the original CODI cold run (LR 3e-3, LoRA r128, many epochs) + block-of-c.
# Runs on the RTX 3060 (PCI idx 2). gradient_checkpointing MUST stay False.
#
# Run from adaptive-vectors/:  bash scripts/train_gpt2_gsm8k_optionb_cold.sh
set -euo pipefail

SAVE_DIR="${SAVE_DIR:-../models/checkpoints/optionb-cold-coarse}"
LOG_DIR="${LOG_DIR:-../outputs/optionb-cold-coarse}"
GPT2_PATH="${GPT2_PATH:-gpt2}"          # plain GPT-2 — NO SIM-CoT checkpoint
DATA_PATH="${DATA_PATH:-../data/gsm8k_aug/train15k.jsonl}"

K="${K:-3}"; M="${M:-3}"                 # K coarse steps, up to M sub-vectors each
BS="${BS:-8}"; ACCUM="${ACCUM:-8}"       # eff batch 64 (memory-safe on 12GB)
EPOCHS="${EPOCHS:-30}"; MAXSAMPLES="${MAXSAMPLES:-15000}"
LR="${LR:-1e-3}"                         # cold LR (3e-3 diverged at ep7 w/ block-of-c; lowered)
LAMBDA_HALT="${LAMBDA_HALT:-0.0}"        # penalty off for the first cold run (headroom first)
export HALT_HEAD_LR="${HALT_HEAD_LR:-1e-3}"   # MLP/halt fast group; lowered in lockstep w/ LR
GRADNORM="${GRADNORM:-0.5}"              # tighter clip than 1.0 to catch divergence spikes
WARMUP="${WARMUP:-0.05}"                 # a bit more warmup for cold stability
export UV_NO_SYNC=1                      # never let uv sync/wipe the shared venv

mkdir -p "$SAVE_DIR" "$LOG_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[cold] device:"; uv run python -c "import torch;print(' ', torch.cuda.get_device_name(0))"
echo "[cold] COLD start (no warm-start), COARSE steps | K=$K M=$M BS=$BS ACCUM=$ACCUM EPOCHS=$EPOCHS N=$MAXSAMPLES"
echo "[cold] LR=$LR HALT_HEAD_LR=$HALT_HEAD_LR GRADNORM=$GRADNORM WARMUP=$WARMUP LAMBDA_HALT=$LAMBDA_HALT"

# NOTE: no --simcot_ckpt and no --decoder_path  => plain GPT-2 backbone + fresh decoder.
uv run python train.py \
    --output_dir "$SAVE_DIR" --logging_dir "$LOG_DIR" --logging_steps 20 \
    --model_name_or_path "$GPT2_PATH" --data_name icot --data_path "$DATA_PATH" \
    --seed 42 --model_max_length 384 --max_token_num 700 \
    --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$ACCUM" \
    --gradient_checkpointing False --dataloader_num_workers 4 --bf16 \
    --num_train_epochs "$EPOCHS" --learning_rate "$LR" --max_grad_norm "$GRADNORM" \
    --use_lora True --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy epoch --save_total_limit 4 --save_safetensors False \
    --weight_decay 0.1 --warmup_ratio "$WARMUP" --lr_scheduler_type cosine \
    --do_train --report_to tensorboard \
    --use_prj True --prj_dim 768 --prj_dropout 0.0 \
    --distill_loss_div_std True --remove_eos True \
    --use_decoder True \
    --simcot_ckpt "" \
    --print_loss False --max_train_samples "$MAXSAMPLES" \
    --option_b True --ob_num_steps "$K" --ob_subvectors_per_step "$M" \
    --ob_coarse_steps True --ob_mlp_hidden 256 --ob_detach_hk True \
    --ob_lambda_ans 1.0 --ob_lambda_step 1.0 --ob_lambda_dist 1.0 --ob_lambda_halt "$LAMBDA_HALT" \
    2>&1 | tee "$LOG_DIR/train.log"
