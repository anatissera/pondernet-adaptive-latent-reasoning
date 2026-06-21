#!/usr/bin/env bash
# Option-B Step 3 — re-measure the UNBIASED (cold + coarse) model and compare to the
# warm+atomic baseline (c1=27.52% c2=39.42% c3=39.88%, adaptive~=random).
#
# Two parts:
#   A. Fixed-c curve on the FULL GSM8K test (1319 ex): c=1,2,3 via --ob_max_subvectors,
#      --ob_eps 0.0 (never halts early -> exactly c vectors/step). Headline comparison.
#   B. Adaptive eps sweep + random baseline on the 300-ex subset (fast): does learned
#      halting beat random at matched budget? does the MLP spend more on harder items?
#
# Runs on the RTX 3060 (PCI idx 2) so it can overlap other GPUs. bs=1 (faithful halting).
# Auto-detects the highest-numbered checkpoint under the cold-coarse run.
#
# Run from adaptive-vectors/:  bash scripts/ob_step3_coldcoarse.sh
set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-../models/checkpoints/optionb-cold-coarse/default/gpt2/ep_30/lr_0.001/seed_42}"
# pick the checkpoint with the largest step number
CKPT="${CKPT:-$(ls -d "$CKPT_ROOT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 | sed "s|^|$CKPT_ROOT/checkpoint-|")}"
[ -d "$CKPT" ] || { echo "[step3] no checkpoint found under $CKPT_ROOT" >&2; exit 1; }

K="${K:-3}"; M="${M:-3}"
SUB="${SUB:-../results/gsm8k_test_sub300.json}"
OUT="${OUT:-../results/optionb-cold-coarse}"
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_NO_SYNC=1
echo "[step3] CKPT=$CKPT  K=$K M=$M  device:"; uv run python -c "import torch;print(' ', torch.cuda.get_device_name(0))"

# --- Part A: fixed-c curve on the FULL test set ---
for C in 1 2 3; do
  echo ""
  echo "==================== FULL-TEST fixed c=$C ===================="
  CKPT="$CKPT" RESULTS_DIR="$OUT/fixed_c$C" K="$K" MMAX="$C" \
    bash scripts/eval_gpt2_gsm8k_optionb.sh --ob_eps 0.0 2>&1 \
    | tee "$OUT/fixed_c$C.log" \
    | grep -E "GSM8K test accuracy|\[Option-B\]|Vecs \||^ *[0-9]+ \|" || true
done

# --- Part B: adaptive eps sweep + random, on the 300-ex subset ---
sweep() {
  local tag="$1"; shift
  echo ""
  echo "==================== SUB300 $tag ===================="
  CKPT="$CKPT" RESULTS_DIR="$OUT/$tag" K="$K" MMAX="$M" \
    bash scripts/eval_gpt2_gsm8k_optionb.sh --data_path "$SUB" "$@" 2>&1 \
    | tee "$OUT/$tag.log" \
    | grep -E "GSM8K test accuracy|\[Option-B\]|Vecs \||correct|^ *[0-9]+ \|" || true
}
sweep "fixed_M"  --ob_eps 0.0
sweep "eps0p01"  --ob_eps 0.01
sweep "eps0p05"  --ob_eps 0.05
sweep "eps0p15"  --ob_eps 0.15
sweep "random"   --ob_random True

echo ""
echo "[step3] done. logs + per-config dirs under $OUT"
