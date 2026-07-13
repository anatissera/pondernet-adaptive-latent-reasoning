#!/usr/bin/env bash
# Recorded by eval_gpt2_gsm8k_pondernet.sh on 2026-07-10T19:45:05+00:00
# host=tpnlp  git=33604147
#   CKPT=/home/tpnlp/alr-valentino/models/checkpoints/08-simcot-pondernet-gamma-frontier/fullscope-adaptive-g0.10-a0.6-b1.5-k12-ep5/default/gpt2/ep_5/lr_2e-05/seed_42/checkpoint-3890  THRESHOLD=0.3  BATCH_SIZE=1
#   CUDA_VISIBLE_DEVICES=1
python test.py --model_name_or_path gpt2 --ckpt_dir /home/tpnlp/alr-valentino/models/checkpoints/08-simcot-pondernet-gamma-frontier/fullscope-adaptive-g0.10-a0.6-b1.5-k12-ep5/default/gpt2/ep_5/lr_2e-05/seed_42/checkpoint-3890 --data_name gsm8k --data_path ../data/gsm8k_aug/test.jsonl --results_dir ../results/08-simcot-pondernet-gamma-frontier/fullscope-adaptive-g0.10-a0.6-b1.5-k12-ep5/ep5-bs1-test/thr0.3 --batch_size 1 --max_latent_steps 6 --use_lora True --lora_r 128 --lora_alpha 32 --lora_init --bf16 --use_prj True --prj_dim 768 --remove_eos True --greedy True --pondernet True --pondernet_inf_threshold 0.3 --pondernet_halt_bias_init -2.0 --train False --max_latent_steps 12 
