#!/usr/bin/env bash
# One-off: resume the simcot-pondernet-gcfix-100k run from its epoch-2 checkpoint
# (checkpoint-1556) after the OOM crash. Calls train.py directly with the exact
# gcfix recipe so HF Trainer auto-resumes; does NOT use the (reverted) train .sh.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/tpnlp/adaptive-latent-reasoning
python train.py \
    --output_dir "$REPO/models/checkpoints/simcot-pondernet-gcfix-100k" \
    --expt_name default \
    --logging_dir "$REPO/outputs/simcot-pondernet-gcfix-100k" \
    --logging_steps 10 \
    --model_name_or_path gpt2 \
    --data_name icot \
    --data_path "$REPO/data/gsm8k_aug/train100k.jsonl" \
    --seed 42 \
    --model_max_length 384 \
    --max_token_num 700 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --bf16 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy epoch \
    --save_total_limit 2 \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --do_train \
    --report_to tensorboard \
    --max_latent_steps 6 \
    --logging_strategy steps \
    --use_prj True \
    --prj_dim 768 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --remove_eos True \
    --print_ref_model_stats False \
    --use_decoder True \
    --decoder_path "$REPO/models/pretrained/simcot-gpt2-decoder" \
    --simcot_ckpt "$REPO/models/pretrained/simcot-gpt2-codi/model-00001-of-00001.safetensors" \
    --print_loss False \
    --pondernet True \
    --pondernet_beta 1.0 \
    --pondernet_gamma 0.01 \
    --pondernet_geom_mean 3.0 \
    --pondernet_halt_bias_init -2.0 \
    --pondernet_inf_threshold 0.5
