#!/usr/bin/env bash
# Evaluate a standard SIM-CoT (fixed-K) GPT-2 checkpoint on GSM8K.
# Use different --max_latent_steps and --inf_latent_iterations values for
# the fixed-K baseline comparison in Phase 6.
#
# Run from pondernet/:
#   CKPT=/path/to/checkpoint NUM_LATENT=6 bash scripts/eval_gpt2_gsm8k_fixedk.sh

set -euo pipefail

CKPT="${CKPT:?Set CKPT=/path/to/checkpoint}"
GPT2_PATH="${GPT2_PATH:-gpt2}"
EXP="${EXP:-}"
RUN="${RUN:-}"
if [[ -n "$EXP" && -n "$RUN" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-../results/$EXP/$RUN}"
else
    RESULTS_DIR="${RESULTS_DIR:-../results/simcot-fixedk-default}"
fi
NUM_LATENT="${NUM_LATENT:-6}"
# Default to the augmented VALIDATION set. Previously no --data_path was passed, so
# test.py silently fell back to the HuggingFace gsm8k *test* split (leakage). See
# docs/experiments.md "Eval split / leakage note" (2026-06-23).
DATA_PATH="${DATA_PATH:-../data/gsm8k_aug/validation.jsonl}"

mkdir -p "$RESULTS_DIR"

# Record the exact resolved invocation for reproducibility.
{
    echo "#!/usr/bin/env bash"
    echo "# Recorded by eval_gpt2_gsm8k_fixedk.sh on $(date -Iseconds)"
    echo "# host=$(hostname)  git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo -dirty)"
    echo "#   CKPT=$CKPT  NUM_LATENT=$NUM_LATENT"
    echo "#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
} > "$RESULTS_DIR/command.sh"

EVAL_CMD=(python test.py
    --model_name_or_path "$GPT2_PATH"
    --ckpt_dir "$CKPT"
    --data_name gsm8k
    --data_path "$DATA_PATH"
    --results_dir "$RESULTS_DIR"
    --batch_size 1
    --max_latent_steps "$NUM_LATENT"
    --inf_latent_iterations "$NUM_LATENT"
    --use_lora True
    --lora_r 128 --lora_alpha 32 --lora_init
    --bf16
    --use_prj True
    --prj_dim 768
    --remove_eos True
    --greedy True
    --train False
    "$@"
)
printf '%q ' "${EVAL_CMD[@]}" >> "$RESULTS_DIR/command.sh"; echo >> "$RESULTS_DIR/command.sh"

echo "[eval] results → $RESULTS_DIR"
"${EVAL_CMD[@]}" 2>&1 | tee "$RESULTS_DIR/eval.log"
