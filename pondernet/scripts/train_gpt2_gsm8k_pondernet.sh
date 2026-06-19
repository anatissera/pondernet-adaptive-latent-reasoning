#!/usr/bin/env bash
# Train GPT-2 with PonderNet adaptive halting on GSM8K-Aug.
# Run from pondernet/:  bash scripts/train_gpt2_gsm8k_pondernet.sh
#
# Key PonderNet flags:
#   --pondernet True               enable adaptive halting
#   --pondernet_beta 1.0           weight on aux-decoder L_step
#   --pondernet_gamma 0.01         weight on KL-geometric regularizer
#   --pondernet_geom_mean 3.0      geometric prior mean steps (tune this)
#   --pondernet_inf_threshold 0.5  inference early-stop threshold
#
# Training reads the local pinned subset data/gsm8k_aug/train15k.jsonl by default.
# Override with DATA_PATH=/path/to/other.jsonl or pass --data_path on the command line.
# The HF hub (zen-E/GSM8k-Aug) is the fallback when no --data_path is given.

set -euo pipefail

# Experiment-scoped layout: EXP=<NN-name> RUN=<run-id> derive the artifact dirs.
# Explicit SAVE_DIR/LOG_DIR still override (back-compat).
EXP="${EXP:-}"
RUN="${RUN:-}"
if [[ -n "$EXP" && -n "$RUN" ]]; then
    SAVE_DIR="${SAVE_DIR:-../models/checkpoints/$EXP/$RUN}"
    LOG_DIR="${LOG_DIR:-../outputs/$EXP/$RUN}"
else
    SAVE_DIR="${SAVE_DIR:-../models/checkpoints/simcot-pondernet-default}"
    LOG_DIR="${LOG_DIR:-../outputs/simcot-pondernet-default}"
fi
# GPT2_PATH MUST be a plain GPT-2 checkpoint. Do NOT set it to the SIM-CoT CODI
# checkpoint -- that is a CODI wrapper (keys under codi.base_model.model.*), and loading
# it here as a bare GPT2LMHeadModel silently random-inits the whole backbone. Warm-start
# from SIM-CoT via SIMCOT_CKPT (full-model) or DECODER_PATH (decoder-only) below.
GPT2_PATH="${GPT2_PATH:-gpt2}"   # HF model ID or local path

# Default recipe: full-model warm-start. Load the FULL CODI model (backbone + LoRA
# adapters + decoder + prj) from the SIM-CoT CODI checkpoint, so the model starts already
# "thinking in latent space"; --pondernet then trains the backbone (via LoRA) + halt head
# while the decoder stays warm-but-frozen. Loaded via load_state_dict(strict=False) after
# assembly; only the halt head is fresh.
#   - The checkpoint lives at repo-root models/ (gitignored; repo owners share it on the FS).
#   - Use `-` (not `:-`) so an explicit empty value is respected: `SIMCOT_CKPT="" bash ...`
#     falls back to the decoder-only recipe (cold backbone, warm decoder via DECODER_PATH).
SIMCOT_CKPT="${SIMCOT_CKPT-../models/pretrained/simcot-gpt2-codi/model-00001-of-00001.safetensors}"

# Initialize the auxiliary decoder from a SIM-CoT-trained checkpoint instead of
# vanilla GPT-2, so L_step/L_pondernet provide real signal from epoch 0.
# Fetch with: python scripts/fetch_simcot_decoder.py --out ../models/pretrained/simcot-gpt2-decoder
DECODER_PATH="${DECODER_PATH:-../models/pretrained/simcot-gpt2-decoder}"
DATA_DIR="${DATA_DIR:-../data}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/gsm8k_aug/train100k.jsonl}"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

# Avoids CUDA allocator fragmentation (important with K separate answer-decode forwards)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Assemble the training command into an array so we can both record it (for
# reproducibility -- LR and every other resolved flag) and run it verbatim.
TRAIN_CMD=(python train.py
    --output_dir "$SAVE_DIR" \
    --logging_dir "$LOG_DIR" \
    --logging_steps 10 \
    --model_name_or_path "$GPT2_PATH" \
    --data_name icot \
    --data_path "$DATA_PATH" \
    --seed 42 \
    --model_max_length 384 \
    --max_token_num 700 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --bf16 \
    --num_train_epochs 5 \
    --learning_rate "${LR:-2e-5}" \
    --max_grad_norm 1.0 \
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
    --max_latent_steps 6 \
    --logging_strategy steps \
    --use_prj True \
    --prj_dim 768 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --remove_eos True \
    --print_ref_model_stats False \
    --use_decoder True \
    --decoder_path "$DECODER_PATH" \
    --simcot_ckpt "$SIMCOT_CKPT" \
    --print_loss False \
    --pondernet True \
    --pondernet_beta 1.0 \
    --pondernet_gamma 0.01 \
    --pondernet_geom_mean 3.0 \
    --pondernet_halt_bias_init -2.0 \
    --pondernet_inf_threshold 0.5 \
    --max_train_samples 100000 \
    "$@"
)

# Record the exact invocation into the run folder: resolved env overrides + the
# full flag list (printf %q makes it copy-paste re-runnable) + provenance.
{
    echo "#!/usr/bin/env bash"
    echo "# Recorded by train_gpt2_gsm8k_pondernet.sh on $(date -Iseconds)"
    echo "# host=$(hostname)  user=$(whoami)  cwd=$(pwd)"
    echo "# git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo '-dirty')"
    echo "#"
    echo "# Key env overrides (so the run can be reproduced):"
    echo "#   SAVE_DIR=$SAVE_DIR"
    echo "#   LOG_DIR=$LOG_DIR"
    echo "#   LR=${LR:-2e-5}"
    echo "#   GPT2_PATH=$GPT2_PATH"
    echo "#   SIMCOT_CKPT=$SIMCOT_CKPT"
    echo "#   DECODER_PATH=$DECODER_PATH"
    echo "#   DATA_PATH=$DATA_PATH"
    echo "#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "#   PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
    echo
    printf '%q ' "${TRAIN_CMD[@]}"
    echo
} > "$LOG_DIR/command.sh"

echo "[train] command recorded to $LOG_DIR/command.sh"
echo "[train] teeing stdout+stderr to $LOG_DIR/train.log"

# Run it, mirroring all output into the run folder so logs live with the run
# (pipefail, set above, makes the exit status reflect train.py, not tee).
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/train.log"
