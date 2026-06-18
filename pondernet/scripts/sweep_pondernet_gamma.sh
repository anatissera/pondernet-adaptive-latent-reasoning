#!/usr/bin/env bash
# Complete PonderNet adaptivity sweep: trade accuracy for fewer latent steps by
# sweeping the KL-geometric weight (--pondernet_gamma), then map the full
# accuracy-vs-steps frontier (every epoch checkpoint x every halt threshold).
#
# Why: the gcfix-100k run recovered accuracy (42.23%) but barely halted early
# (avg 5.88/6 steps) because gamma=0.01 makes gamma*KL ~0.003 of the loss
# (docs/runs.md -> Next Steps). Raising gamma is the primary lever for real
# adaptivity. We also keep every epoch (accuracy peaked at ep2 then drifted to
# baseline by ep5 in gcfix-100k) and eval at several thresholds.
#
# Run from pondernet/:
#   bash scripts/sweep_pondernet_gamma.sh
#
# Each run trains on TRAIN_GPU and is evaluated on EVAL_GPU *after* training
# finishes (never concurrently -- sharing the 3090 is what OOM-crashed gcfix).
#
# Override via env:
#   GAMMAS="0.0 0.05 0.1 0.3 1.0"   gamma grid
#   GEOM_MEAN=3.0                   geometric-prior mean steps (lower => push to halt sooner)
#   EPOCHS=5                        epochs per run
#   THRESHOLDS="0.5 0.8 0.9"        inference halt thresholds to eval each checkpoint at
#   MAX_TRAIN_SAMPLES=              training-set cap (unset => full 100k; set lower for a quick pass)
#   EVAL_BATCH_SIZE=1               eval batch size (>1 is faithful via the test.py fix; validate first)
#   SAVE_TOTAL_LIMIT=EPOCHS+1       checkpoints to keep (default keeps every epoch)
#   HALT_HEAD_LR=                   optional hot LR for the halt head (passes to train.py)
#   TRAIN_GPU=1  EVAL_GPU=0         GPU ids in PCI order (1=3090 train, 0=3060 eval)
#   SWEEP_TAG=gammasweep            names results/<tag>/summary.tsv
#
# Artifacts per gamma g use run-id  simcot-pondernet-<TAG>-g<g>-gm<GEOM_MEAN>-ep<EPOCHS>
# under models/checkpoints/, outputs/, results/. The combined frontier is written to
# results/<TAG>/summary.tsv  (gamma, geom_mean, checkpoint, threshold, accuracy, avg_steps).

set -euo pipefail

if [[ ! -f test.py || ! -f scripts/train_gpt2_gsm8k_pondernet.sh ]]; then
    echo "[sweep] ERROR: run this from the pondernet/ directory." >&2
    exit 1
fi

# PCI ordering so CUDA indices match nvidia-smi (docs/runs.md): 1=3090, 0=3060.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-0}"

GAMMAS="${GAMMAS:-0.0 0.05 0.1 0.3}"
GEOM_MEAN="${GEOM_MEAN:-3.0}"
EPOCHS="${EPOCHS:-5}"
THRESHOLDS="${THRESHOLDS:-0.5 0.8 0.9}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SWEEP_TAG="${SWEEP_TAG:-gammasweep}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-$((EPOCHS + 1))}"
# Batch size / accumulation overrides (default: train script values: bs=32, accum=4).
# Set TRAIN_BS=16 TRAIN_ACCUM=8 when training on the 3060 (12 GB) instead of the 3090.
TRAIN_BS="${TRAIN_BS:-}"
TRAIN_ACCUM="${TRAIN_ACCUM:-}"

SWEEP_DIR="../results/${SWEEP_TAG}"
SUMMARY="${SWEEP_DIR}/summary.tsv"
mkdir -p "$SWEEP_DIR"
printf 'gamma\tgeom_mean\tcheckpoint\tthreshold\taccuracy_pct\tavg_steps\trun_id\n' > "$SUMMARY"

echo "[sweep] gammas=[$GAMMAS] geom_mean=$GEOM_MEAN epochs=$EPOCHS thresholds=[$THRESHOLDS]"
echo "[sweep] train_gpu=$TRAIN_GPU eval_gpu=$EVAL_GPU (CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER)"
echo "[sweep] batch overrides: TRAIN_BS=${TRAIN_BS:-<script default 32>} TRAIN_ACCUM=${TRAIN_ACCUM:-<script default 4>}"
echo "[sweep] frontier -> $SUMMARY"

for G in $GAMMAS; do
    RUN_ID="simcot-pondernet-${SWEEP_TAG}-g${G}-gm${GEOM_MEAN}-ep${EPOCHS}"
    SAVE_DIR="../models/checkpoints/${RUN_ID}"
    LOG_DIR="../outputs/${RUN_ID}"

    echo "==================================================================="
    echo "[sweep] TRAIN gamma=$G  ->  $RUN_ID"
    echo "==================================================================="

    # Trailing flags override the hardcoded values in the train script
    # (argparse keeps the last occurrence).
    TRAIN_EXTRA=(--pondernet_gamma "$G"
                 --pondernet_geom_mean "$GEOM_MEAN"
                 --num_train_epochs "$EPOCHS"
                 --save_total_limit "$SAVE_TOTAL_LIMIT")
    if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
        TRAIN_EXTRA+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
    fi
    if [[ -n "${TRAIN_BS:-}" ]]; then
        TRAIN_EXTRA+=(--per_device_train_batch_size "$TRAIN_BS")
    fi
    if [[ -n "${TRAIN_ACCUM:-}" ]]; then
        TRAIN_EXTRA+=(--gradient_accumulation_steps "$TRAIN_ACCUM")
    fi

    if ! CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
         SAVE_DIR="$SAVE_DIR" LOG_DIR="$LOG_DIR" HALT_HEAD_LR="${HALT_HEAD_LR:-}" \
         bash scripts/train_gpt2_gsm8k_pondernet.sh "${TRAIN_EXTRA[@]}"; then
        echo "[sweep] WARN: training failed for gamma=$G; skipping to next gamma." >&2
        continue
    fi

    # Every dir holding a saved model: each kept epoch's checkpoint-N + the final root.
    mapfile -t CKPTS < <(find "$SAVE_DIR" -name 'pytorch_model.bin' -printf '%h\n' 2>/dev/null | sort -u)
    if [[ ${#CKPTS[@]} -eq 0 ]]; then
        echo "[sweep] WARN: no checkpoints under $SAVE_DIR; skipping eval for gamma=$G." >&2
        continue
    fi

    for CKPT in "${CKPTS[@]}"; do
        CKPT_LABEL="$(basename "$CKPT")"     # checkpoint-NNN, or seed_NN for the final model
        for TH in $THRESHOLDS; do
            RESULTS_DIR="../results/${RUN_ID}/${CKPT_LABEL}/thr${TH}"
            mkdir -p "$RESULTS_DIR"
            echo "---- [sweep] EVAL gamma=$G ckpt=$CKPT_LABEL thr=$TH ----"
            if ! CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
                 CKPT="$CKPT" RESULTS_DIR="$RESULTS_DIR" THRESHOLD="$TH" BATCH_SIZE="$EVAL_BATCH_SIZE" \
                 bash scripts/eval_gpt2_gsm8k_pondernet.sh 2>&1 | tee "$RESULTS_DIR/eval.log"; then
                echo "[sweep] WARN: eval failed for g=$G ckpt=$CKPT_LABEL thr=$TH." >&2
            fi
            ACC="$(grep -oE 'GSM8K test accuracy: [0-9.]+' "$RESULTS_DIR/eval.log" | tail -1 | grep -oE '[0-9.]+' | tail -1 || true)"
            STEPS="$(grep -oE 'average latent steps used: [0-9.]+' "$RESULTS_DIR/eval.log" | tail -1 | grep -oE '[0-9.]+' | tail -1 || true)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$G" "$GEOM_MEAN" "$CKPT_LABEL" "$TH" "${ACC:-NA}" "${STEPS:-NA}" "$RUN_ID" >> "$SUMMARY"
        done
    done
done

echo
echo "[sweep] done. Accuracy-vs-steps frontier:"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY"
else
    cat "$SUMMARY"
fi
echo "[sweep] summary saved to $SUMMARY"
