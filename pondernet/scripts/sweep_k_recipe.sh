#!/usr/bin/env bash
# Adaptive-K / warm-start factorial: test the "SIM-CoT overfits to K=6" hypothesis.
#
# The current PonderNet recipe warm-starts the full SIM-CoT/CODI checkpoint and trains
# only LoRA + the halt head, so the backbone keeps a baked-in "the answer lives at step
# K" prior (gamma=0 never halts early -- docs/runs.md "Gamma Sweep"). This sweep crosses
# the warm-start RECIPE with K_max to see whether removing/overwriting that prior lets
# the model learn a halting distribution matching the true 2-3-step data distribution
# (data is mean 2.59 steps, 80% need <=3, only 0.8% need >6).
#
# Recipes (TRAIN_SCOPE controls which params are unfrozen; see train.py freeze block):
#   A  full SIM-CoT warm-start, scope=lora      -> baseline (K-fixed prior, frozen backbone)
#   B  cold GPT-2 backbone (SIMCOT_CKPT=""),     -> no answer-position prior at all;
#      decoder-only warm-start, scope=lora_prj      prj is random so it MUST be trained
#   C  full SIM-CoT warm-start, scope=full       -> keep the good init, overwrite the prior
#
# Constants held across the grid so the only axes are (recipe, K_max):
#   gamma=0.1 (sweet spot from the gamma sweep), geom_mean=3.0 (~ data mean), lr 2e-5,
#   eff batch ~128, 5 epochs, seed 42, gradient_checkpointing False (KV-cache bug).
#
# Run from pondernet/:   bash scripts/sweep_k_recipe.sh
#
# Override via env:
#   RECIPES="A B C"                recipes to run
#   KMAXES="4 6 8 10 12"           K_max grid (--max_latent_steps)
#   GAMMA=0.1  GEOM_MEAN=3.0       halting pressure / prior (held constant)
#   EPOCHS=5                       epochs per run
#   THRESHOLDS="0.5 0.8 0.9"       inference halt thresholds to eval each checkpoint at
#   EVAL_BATCH_SIZE=16             eval batch size (faithful via the test.py fix)
#   MAX_TRAIN_SAMPLES=             training-set cap (unset => full 100k)
#   TRAIN_GPU=0  EVAL_GPU=0        GPU ids (single-GPU GCP: same card, eval runs after train)
#   SWEEP_TAG=k_recipe_sweep       names results/<tag>/summary.tsv
#   SIMCOT_DEFAULT=<path>          SIM-CoT CODI checkpoint for recipes A/C
#
# Frontier -> results/<TAG>/summary.tsv (recipe, k_max, checkpoint, threshold, accuracy, avg_steps, run_id)

set -euo pipefail

if [[ ! -f test.py || ! -f scripts/train_gpt2_gsm8k_pondernet.sh ]]; then
    echo "[ksweep] ERROR: run this from the pondernet/ directory." >&2
    exit 1
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
TRAIN_GPU="${TRAIN_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-0}"     # single-GPU default: eval after training on the same card

RECIPES="${RECIPES:-A B C}"
KMAXES="${KMAXES:-4 6 8 10 12}"
GAMMA="${GAMMA:-0.1}"
GEOM_MEAN="${GEOM_MEAN:-3.0}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
THRESHOLDS="${THRESHOLDS:-0.5 0.8 0.9}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SWEEP_TAG="${SWEEP_TAG:-k_recipe_sweep}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-$((EPOCHS + 1))}"
SIMCOT_DEFAULT="${SIMCOT_DEFAULT:-../models/pretrained/simcot-gpt2-codi/model-00001-of-00001.safetensors}"

SWEEP_DIR="../results/${SWEEP_TAG}"
# SUMMARY_FILE lets two parallel streams (e.g. one per GPU) write disjoint summary files
# under the same SWEEP_TAG (so run-ids/checkpoint dirs stay consistent); merge later.
SUMMARY="${SUMMARY_FILE:-${SWEEP_DIR}/summary.tsv}"
mkdir -p "$SWEEP_DIR" "$(dirname "$SUMMARY")"
# Only write the header if the file is new - avoids one parallel stream wiping another's rows.
[[ -f "$SUMMARY" ]] || printf 'recipe\tk_max\tcheckpoint\tthreshold\taccuracy_pct\tavg_steps\trun_id\n' > "$SUMMARY"

echo "[ksweep] recipes=[$RECIPES] kmaxes=[$KMAXES] gamma=$GAMMA geom_mean=$GEOM_MEAN epochs=$EPOCHS lr=$LR"
echo "[ksweep] train_gpu=$TRAIN_GPU eval_gpu=$EVAL_GPU thresholds=[$THRESHOLDS]"
echo "[ksweep] frontier -> $SUMMARY"

# eff batch ~128: shrink per-device batch as K_max grows (more latent + per-step answer
# forwards held in memory). bs*accum stays ~128.
pick_batch() {  # $1=K_max -> echoes "BS ACCUM"
    local k="$1"
    # FORCE_BS/FORCE_ACCUM override the auto-sizing (e.g. the 12GB 3060 can't fit bs=32);
    # keep eff batch ~128 by setting FORCE_ACCUM accordingly.
    if [[ -n "${FORCE_BS:-}" ]]; then echo "${FORCE_BS} ${FORCE_ACCUM:-$((128 / FORCE_BS))}"; return; fi
    if   (( k <= 8 ));  then echo "32 4"
    elif (( k <= 10 )); then echo "24 5"
    else                     echo "16 8"
    fi
}

for R in $RECIPES; do
    case "$R" in
        A) R_SIMCOT="$SIMCOT_DEFAULT"; R_SCOPE="lora" ;;
        B) R_SIMCOT="";                R_SCOPE="lora_prj" ;;   # cold backbone, decoder-only warm-start
        C) R_SIMCOT="$SIMCOT_DEFAULT"; R_SCOPE="full" ;;
        *) echo "[ksweep] ERROR: unknown recipe '$R' (want A|B|C)" >&2; exit 1 ;;
    esac

    for K in $KMAXES; do
        read -r BS ACCUM < <(pick_batch "$K")
        RUN_ID="simcot-pondernet-${SWEEP_TAG}-recipe${R}-k${K}-ep${EPOCHS}"
        SAVE_DIR="../models/checkpoints/${RUN_ID}"
        LOG_DIR="../outputs/${RUN_ID}"

        echo "==================================================================="
        echo "[ksweep] TRAIN recipe=$R (scope=$R_SCOPE simcot='${R_SIMCOT:-<cold>}') K_max=$K bs=$BS accum=$ACCUM -> $RUN_ID"
        echo "==================================================================="

        # Trailing flags override the train script's hardcoded values.
        TRAIN_EXTRA=(--pondernet_gamma "$GAMMA"
                     --pondernet_geom_mean "$GEOM_MEAN"
                     --pondernet_train_scope "$R_SCOPE"
                     --max_latent_steps "$K"
                     --per_device_train_batch_size "$BS"
                     --gradient_accumulation_steps "$ACCUM"
                     --num_train_epochs "$EPOCHS"
                     --save_total_limit "$SAVE_TOTAL_LIMIT")
        # DL_WORKERS overrides the train script's hardcoded --dataloader_num_workers 4
        # (each worker holds ~1.6GB of tokenized data → cut RAM when sharing the box).
        if [[ -n "${DL_WORKERS:-}" ]]; then
            TRAIN_EXTRA+=(--dataloader_num_workers "$DL_WORKERS")
        fi
        # SEED overrides the train script's hardcoded --seed 42 (for multi-seed error bars).
        if [[ -n "${SEED:-}" ]]; then
            TRAIN_EXTRA+=(--seed "$SEED")
        fi
        if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
            TRAIN_EXTRA+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
        fi

        # SIMCOT_CKPT="" (recipe B) is respected by the train script (`-` not `:-`),
        # giving a cold backbone + decoder-only warm-start. TRAIN_SCOPE drives the freeze.
        if ! CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
             SAVE_DIR="$SAVE_DIR" LOG_DIR="$LOG_DIR" \
             SIMCOT_CKPT="$R_SIMCOT" TRAIN_SCOPE="$R_SCOPE" LR="$LR" \
             bash scripts/train_gpt2_gsm8k_pondernet.sh "${TRAIN_EXTRA[@]}"; then
            echo "[ksweep] WARN: training failed for recipe=$R K=$K; skipping." >&2
            continue
        fi

        mapfile -t CKPTS < <(find "$SAVE_DIR" -name 'pytorch_model.bin' -printf '%h\n' 2>/dev/null | sort -u)
        if [[ ${#CKPTS[@]} -eq 0 ]]; then
            echo "[ksweep] WARN: no checkpoints under $SAVE_DIR; skipping eval." >&2
            continue
        fi

        for CKPT in "${CKPTS[@]}"; do
            CKPT_LABEL="$(basename "$CKPT")"
            for TH in $THRESHOLDS; do
                RESULTS_DIR="../results/${RUN_ID}/${CKPT_LABEL}/thr${TH}"
                mkdir -p "$RESULTS_DIR"
                echo "---- [ksweep] EVAL recipe=$R K=$K ckpt=$CKPT_LABEL thr=$TH ----"
                # eval at the same K_max the run was trained with (eval script hardcodes 6).
                if ! CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
                     CKPT="$CKPT" RESULTS_DIR="$RESULTS_DIR" THRESHOLD="$TH" BATCH_SIZE="$EVAL_BATCH_SIZE" \
                     bash scripts/eval_gpt2_gsm8k_pondernet.sh --max_latent_steps "$K" 2>&1 | tee "$RESULTS_DIR/eval.log"; then
                    echo "[ksweep] WARN: eval failed for recipe=$R K=$K ckpt=$CKPT_LABEL thr=$TH." >&2
                fi
                ACC="$(grep -oE 'GSM8K test accuracy: [0-9.]+' "$RESULTS_DIR/eval.log" | tail -1 | grep -oE '[0-9.]+' | tail -1 || true)"
                STEPS="$(grep -oE 'average latent steps used: [0-9.]+' "$RESULTS_DIR/eval.log" | tail -1 | grep -oE '[0-9.]+' | tail -1 || true)"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$R" "$K" "$CKPT_LABEL" "$TH" "${ACC:-NA}" "${STEPS:-NA}" "$RUN_ID" >> "$SUMMARY"
            done
        done
    done
done

echo
echo "[ksweep] done. Accuracy-vs-steps frontier:"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY"
else
    cat "$SUMMARY"
fi
echo "[ksweep] summary saved to $SUMMARY"
