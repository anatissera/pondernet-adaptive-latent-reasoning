#!/usr/bin/env bash
# Runs ON the training VM as a systemd unit. Keeps the eval L4 powered off unless there is a
# batch of work for it.
#
# The eval VM is on-demand (~$0.71/h) and bills for every second it is RUNNING, so leaving it
# up to poll for checkpoints would cost more than the evaluations themselves. Instead it stays
# TERMINATED; when BATCH checkpoints have piled up in the bucket unevaluated, this starts it.
# Its startup-script drains every pending checkpoint and calls `poweroff`, which returns the
# instance to TERMINATED. So the VM is billed only while it is actually scoring.
#
# The last partial batch (fewer than BATCH epochs, e.g. epochs 37-40) would otherwise never
# trigger, so once training exits we drain whatever is left.
#
# This lives on the training VM rather than a laptop because the training VM is up for the
# whole run by definition, and systemd restarts this unit after a Spot preemption reboot.
set -uo pipefail

BUCKET="${BUCKET:-gs://alr-exp10-ckpts-244544686610}"
EVAL_PROJECT="${EVAL_PROJECT:-tp-final-rl-kv-eviction}"
EVAL_ZONE="${EVAL_ZONE:-us-central1-a}"
EVAL_VM="${EVAL_VM:-alr-eval-l4}"
BATCH="${BATCH:-6}"

ev() { gcloud compute instances "$1" "$EVAL_VM" --project="$EVAL_PROJECT" --zone="$EVAL_ZONE" "${@:2}"; }
status() { ev describe --format='value(status)' 2>/dev/null; }

pending_count() {
  local ready done_
  ready=$(gcloud storage ls "$BUCKET/ckpt/*/.ready" 2>/dev/null | wc -l)
  done_=$(gcloud storage ls "$BUCKET/results/*/summary.json" 2>/dev/null | wc -l)
  echo $(( ready - done_ ))
}

while true; do
  n=$(pending_count)
  train_up=$(systemctl is-active alr-train.service)
  st=$(status)

  # Drain when a full batch has accumulated, or when training is over and stragglers remain.
  if [ "$n" -ge "$BATCH" ] || { [ "$train_up" != active ] && [ "$n" -gt 0 ]; }; then
    if [ "$st" = TERMINATED ]; then
      echo "[$(date -Is)] $n checkpoint(s) pending (batch=$BATCH, train=$train_up) - starting $EVAL_VM"
      if ev start --quiet 2>&1; then
        echo "[$(date -Is)] $EVAL_VM started; it drains and powers itself off"
      else
        echo "[$(date -Is)] start failed (capacity?); retrying next cycle"
      fi
    else
      echo "[$(date -Is)] $n pending, $EVAL_VM already $st - letting it finish"
    fi
  fi

  # Everything scored and the trainer is gone: nothing left to orchestrate.
  if [ "$train_up" != active ] && [ "$n" -le 0 ] && [ "$st" = TERMINATED ]; then
    echo "[$(date -Is)] training finished and all checkpoints evaluated - exiting"
    exit 0
  fi
  sleep 300
done
