#!/usr/bin/env bash
# Option-B synthetic eval sweep: for each test level L0..L3, run the fixed-c curve (c=1..4),
# an adaptive eps sweep, random, and the oracle. bs=1 (faithful halting). NEVER the 3060.
set -euo pipefail
CKPT="${CKPT:?Set CKPT=.../checkpoint-NNN (the synth-main checkpoint)}"
OUTROOT="${OUTROOT:-../results/optionb-synth}"
DATADIR="${DATADIR:-../data/synth_density}"
K=4; MMAX=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # NEVER 2 (3060 prohibited)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_NO_SYNC=1
run() {  # level tag extra-args...
  local lvl="$1" tag="$2"; shift 2
  local results_dir="$OUTROOT/$lvl/$tag"
  mkdir -p "$results_dir"
  CKPT="$CKPT" RESULTS_DIR="$results_dir" K=$K MMAX=$MMAX \
    bash scripts/eval_gpt2_gsm8k_optionb.sh \
      --data_path "$DATADIR/$lvl.json" \
      --max_latent_steps 16 \
      "$@" 2>&1 \
    | tee "$OUTROOT/$lvl/${tag}.log" \
    | grep -aE "GSM8K test accuracy|avg TOTAL" || true
}
for lvl in L0 L1 L2 L3; do
  mkdir -p "$OUTROOT/$lvl"
  echo "############ $lvl ############"
  for c in 1 2 3 4; do run "$lvl" "fixed_c${c}" --ob_max_subvectors $c --ob_eps 0.0; done
  for e in 0.05 0.15 0.30; do run "$lvl" "eps${e/./}" --ob_eps "$e"; done
  run "$lvl" "random" --ob_random True
  run "$lvl" "oracle" --ob_oracle True
done
echo DONE_SYNTH_EVAL > /tmp/synth_eval_done
echo "[eval_synth_density] DONE"
