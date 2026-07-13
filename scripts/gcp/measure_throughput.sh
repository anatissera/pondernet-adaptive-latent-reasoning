#!/usr/bin/env bash
# Measure real training throughput for a BS/ACCUM candidate. Run ON the VM, from /opt/alr.
#
# Do NOT trust scripts/profile_batch_size.py for this decision: it profiles SEQ_LEN=256,
# K=6, WITH gradient checkpointing - none of which match this run (512, K=12, grad-ckpt
# False). This script instead launches the actual wrapper into a throwaway dir, lets it
# run STEPS optimizer steps, reads the pace off the tqdm line, and kills it.
#
# Usage:
#   bash scripts/gcp/measure_throughput.sh            # default BS=64 ACCUM=2
#   BS=128 ACCUM=1 bash scripts/gcp/measure_throughput.sh
#
# Read the result as:  hours for N epochs = N * 2999 * s_per_step / 3600
set -euo pipefail

BS="${BS:-64}" ACCUM="${ACCUM:-2}" STEPS="${STEPS:-60}"
[ $((BS * ACCUM)) -eq 128 ] || { echo "BS*ACCUM must be 128 (got $((BS*ACCUM)))" >&2; exit 1; }

PROBE_DIR=$(mktemp -d /tmp/thruput.XXXX)
LOG="$PROBE_DIR/train.log"

# train.py is invoked as a bare `python train.py` and DATA_PATH defaults to ../data/...,
# so the wrapper only works with cwd=pondernet/ and with the venv on PATH.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO/pondernet"

EXP=10-simcot-pondernet-fromscratch RUN="throughput-bs$BS" \
SAVE_DIR="$PROBE_DIR/ckpt" LOG_DIR="$PROBE_DIR" BS="$BS" ACCUM="$ACCUM" \
  bash scripts/train_gpt2_gsm8k_pondernet_fromscratch.sh \
    --dataloader_num_workers 16 &
PID=$!
trap 'kill -9 $PID 2>/dev/null; rm -rf "$PROBE_DIR"' EXIT

# tqdm writes "N/M [MM:SS<..., X.XXs/it]" (or it/s). Wait for STEPS steps, then report.
# The first ~10 steps are warmup (allocator growth, cudnn autotune) - pace is read from
# the tqdm average at step STEPS, which amortizes that.
echo "==> BS=$BS ACCUM=$ACCUM - waiting for $STEPS optimizer steps..."
for _ in $(seq 1 240); do
  sleep 15
  kill -0 $PID 2>/dev/null || { echo "training died - last log lines:"; tail -20 "$LOG"; exit 1; }
  line=$(grep -oE "$STEPS/[0-9]+ \[[^]]*\]" "$LOG" | tail -1) || true
  if [ -n "${line:-}" ]; then
    echo "==> tqdm at step $STEPS:  $line"
    rate=$(grep -oE '[0-9.]+(s/it|it/s)' <<<"$line" | tail -1)
    case "$rate" in
      *s/it) s=${rate%s/it} ;;
      *it/s) s=$(python3 -c "print(1/${rate%it/s})") ;;
      *) echo "could not parse rate"; exit 1 ;;
    esac
    python3 - "$s" <<'PY'
import sys
s = float(sys.argv[1]); spe = 2999
print(f"\n  {s:.2f} s/step  ->  {s*spe/3600:.2f} h/epoch")
for ep in (28, 36):
    print(f"  {ep} epochs: {s*spe*ep/3600:.1f} h")
print(f"  epochs in 24 h: {24*3600/(s*spe):.1f}")
PY
    exit 0
  fi
done
echo "timed out waiting for step $STEPS"; exit 1
