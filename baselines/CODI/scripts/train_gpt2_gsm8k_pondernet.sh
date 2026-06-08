#!/usr/bin/env bash
# Train GPT-2 with PonderNet adaptive halting on GSM8K-Aug.
# Run from baselines/CODI/:  bash scripts/train_gpt2_gsm8k_pondernet.sh
#
# Key PonderNet flags:
#   --pondernet True               enable adaptive halting
#   --pondernet_beta 1.0           weight on aux-decoder L_step
#   --pondernet_gamma 0.01         weight on KL-geometric regularizer
#   --pondernet_geom_mean 3.0      geometric prior mean steps (tune this)
#   --pondernet_inf_threshold 0.5  inference early-stop threshold
#
# Data is loaded from HuggingFace (zen-E/GSM8k-Aug) by default.
# Set --data_path /path/to/local.json to use a local file instead.

set -euo pipefail

SAVE_DIR="${SAVE_DIR:-./outputs/pondernet}"
GPT2_PATH="${GPT2_PATH:-gpt2}"   # HF model ID or local path

mkdir -p "$SAVE_DIR"

# Avoids CUDA allocator fragmentation (important with K separate answer-decode forwards)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name gsm8k_gpt2_pondernet \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path "$GPT2_PATH" \
    --data_name icot \
    --seed 42 \
    --model_max_length 384 \
    --max_token_num 700 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --bf16 \
    --num_train_epochs 40 \
    --learning_rate 3e-3 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --do_train \
    --report_to tensorboard \
    --num_latent 6 \
    --logging_strategy steps \
    --use_prj True \
    --prj_dim 768 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --remove_eos True \
    --print_ref_model_stats False \
    --use_decoder True \
    --print_loss False \
    --pondernet True \
    --pondernet_beta 1.0 \
    --pondernet_gamma 0.01 \
    --pondernet_geom_mean 3.0 \
    --pondernet_halt_bias_init -2.0 \
    --pondernet_inf_threshold 0.5 \
    "$@"
