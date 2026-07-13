#!/usr/bin/env bash
# Recorded by eval_gpt2_gsm8k_fixedk.sh on 2026-07-10T20:42:07+00:00
# host=tpnlp  git=33604147
#   CKPT=/tmp/claude-1000/-home-tpnlp-alr-valentino/cdb652e1-e1f4-4294-9f22-d55aa9bd6731/scratchpad/codi_ckpt_prepared  NUM_LATENT=6
#   CUDA_VISIBLE_DEVICES=1
python test.py --model_name_or_path gpt2 --ckpt_dir /tmp/claude-1000/-home-tpnlp-alr-valentino/cdb652e1-e1f4-4294-9f22-d55aa9bd6731/scratchpad/codi_ckpt_prepared --data_name gsm8k --data_path ../data/gsm8k_aug/test.jsonl --results_dir ../results/01-simcot-baselines/baseline-k6/bs1-test --batch_size 1 --max_latent_steps 6 --inf_latent_iterations 6 --use_lora True --lora_r 128 --lora_alpha 32 --lora_init --bf16 --use_prj True --prj_dim 768 --remove_eos True --greedy True --train False 
