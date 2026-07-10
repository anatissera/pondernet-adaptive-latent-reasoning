#!/usr/bin/env bash
# Fast Option-B eval sweep on a 300-example GSM8K subset (3060). For each config it
# prints accuracy + the [Option-B] vectors/step + accuracy-vs-budget summary, so we can
# see (a) whether n_k VARIES across steps/instances (headroom) and (b) whether the MLP
# halting beats the random baseline at matched budget.
#
#   CKPT=/path/to/checkpoint-NNN bash scripts/ob_eval_sweep.sh
set -euo pipefail
CKPT="${CKPT:?Set CKPT=.../checkpoint-NNN}"
SUB="${SUB:-../results/gsm8k_test_sub300.json}"
K="${K:-4}"; MMAX="${MMAX:-3}"

run() {
  local tag="$1"; shift
  echo ""
  echo "==================== $tag ===================="
  CKPT="$CKPT" RESULTS_DIR="../results/optionb-sweep/$tag" K="$K" MMAX="$MMAX" \
    bash scripts/eval_gpt2_gsm8k_optionb.sh --data_path "$SUB" "$@" 2>&1 \
    | grep -E "GSM8K test accuracy|\[Option-B\]|Vecs \||^ *[0-9]+ \|" || true
}

run "fixed_M"  --ob_eps 0.0          # never halts early -> M vectors/step (fixed-c=M ceiling)
run "eps0p01"  --ob_eps 0.01
run "eps0p05"  --ob_eps 0.05
run "eps0p15"  --ob_eps 0.15
run "random"   --ob_random True      # matched-ish budget control
echo ""
echo "[sweep] done."
