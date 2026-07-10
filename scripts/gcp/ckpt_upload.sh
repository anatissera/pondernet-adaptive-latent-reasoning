#!/usr/bin/env bash
# Runs ON the training VM (project adaptative-latent-reasoning), detached.
# Publishes each new epoch checkpoint to GCS so a separate eval VM — on a different GPU, in a
# different project — can score it without stealing SM time from training.
#
# Only pytorch_model.bin is shipped: test.py loads weights from ckpt_dir but takes the
# tokenizer from --model_name_or_path (gpt2, off the hub). ~578 MB/epoch, ~23 GB for all 40.
#
# The .ready marker is written AFTER the weights land, so the eval worker never picks up a
# half-uploaded file. Ordering matters: GCS object writes are atomic, multi-object ones are not.
set -uo pipefail

REPO=/opt/alr
EXP=10-simcot-pondernet-fromscratch
RUN_NAME=fromscratch-runC-lr1e-3-g0.10-a0.6-b1.5-k12-ep40
CKPT_ROOT="$REPO/ckpt/models/checkpoints/$EXP/$RUN_NAME/default/gpt2/ep_40/lr_0.001/seed_42"
BUCKET="${BUCKET:-gs://alr-exp10-ckpts-244544686610}"
STATE=/opt/alr/ckpt/.uploaded
mkdir -p "$STATE"

while true; do
  # sort by step number, not by path: `sort -t- -k2` keys on a hyphen field of the *directory*
  # path (10-simcot-…), which ties on every line and degrades to lexicographic order.
  for step in $(ls -d "$CKPT_ROOT"/checkpoint-* 2>/dev/null \
                  | sed -nE 's#.*/checkpoint-([0-9]+)$#\1#p' | sort -n); do
    ck="$CKPT_ROOT/checkpoint-$step"
    [ -f "$STATE/$step" ] && continue
    bin="$ck/pytorch_model.bin"
    [ -f "$bin" ] || continue
    # the dir appears before the weights finish writing — wait for the size to settle
    s1=$(stat -c%s "$bin"); sleep 20; s2=$(stat -c%s "$bin")
    [ "$s1" = "$s2" ] && [ "$s1" -gt 100000000 ] || continue

    echo "[$(date -Is)] uploading step $step ($(numfmt --to=iec "$s1"))"
    if gcloud storage cp "$bin" "$BUCKET/ckpt/checkpoint-$step/pytorch_model.bin" --quiet; then
      echo "$step" | gcloud storage cp - "$BUCKET/ckpt/checkpoint-$step/.ready" --quiet
      touch "$STATE/$step"
      echo "[$(date -Is)] step $step published"
    else
      echo "[$(date -Is)] upload failed for step $step, will retry"
    fi
  done
  systemctl is-active --quiet alr-train.service || echo "[$(date -Is)] training not active"
  sleep 120
done
