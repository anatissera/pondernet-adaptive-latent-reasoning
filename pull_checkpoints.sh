#!/usr/bin/env bash
# Pull per-epoch checkpoints from the L4 VM to the local repo (incremental).
# Safe to run anytime, repeatedly. Only downloads checkpoints not already present
# locally, and only ones the VM has finished writing (trainer_state.json exists).
#
# Usage:  bash pull_checkpoints.sh
set -euo pipefail

# Pin project/account per-invocation (the user switches gcloud projects for other work).
export CLOUDSDK_CORE_PROJECT="${CLOUDSDK_CORE_PROJECT:-adaptative-latent-reasoning}"
export CLOUDSDK_CORE_ACCOUNT="${CLOUDSDK_CORE_ACCOUNT:-aptissera04@gmail.com}"

NAME="${NAME:-exp10-l4}"
ZONE="${ZONE:-us-central1-a}"
RUN="probe-lr1e-3-wu0.05-ep4"
# train.py nests output_dir as default/gpt2/ep_<N>/lr_<lr>/seed_<seed>/ (see train.py:574-579),
# so the HF checkpoint-<step>/ dirs live here, NOT at the top run dir.
# Probe run: LR=1e-3, 4 epochs → ep_4/lr_0.001.
REMOTE_DIR="adaptive-latent-reasoning/models/checkpoints/10-simcot-pondernet-fromscratch/$RUN/default/gpt2/ep_4/lr_0.001/seed_42"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/checkpoints/10-simcot-pondernet-fromscratch/$RUN"
mkdir -p "$LOCAL_DIR"

ssh_vm() { gcloud compute ssh "$NAME" --zone "$ZONE" --tunnel-through-iap --command "$1" 2>/dev/null; }

echo "[pull] listing complete checkpoints on VM..."
# A checkpoint is 'complete' once trainer_state.json is present.
# || true: the for-loop exits 1 when no checkpoint-* dirs exist (glob no-match), which
# would abort the script under set -euo pipefail. The empty result is handled below.
remote_ckpts=$(ssh_vm "for d in ~/$REMOTE_DIR/checkpoint-*/; do [ -f \"\${d}trainer_state.json\" ] && basename \"\$d\"; done; true" 2>/dev/null | tr -d '\r') || true

if [ -z "$remote_ckpts" ]; then echo "[pull] no complete checkpoints yet."; exit 0; fi

for ck in $remote_ckpts; do
  if [ -f "$LOCAL_DIR/$ck/trainer_state.json" ]; then
    echo "[pull] $ck already local, skip."
  else
    echo "[pull] downloading $ck ..."
    gcloud compute scp --recurse --zone "$ZONE" --tunnel-through-iap "$NAME:~/$REMOTE_DIR/$ck" "$LOCAL_DIR/" \
      && echo "[pull] $ck done." \
      || echo "[pull] WARN: $ck failed (will retry next run)."
  fi
done
echo "[pull] up to date. Local: $LOCAL_DIR"
ls -1 "$LOCAL_DIR" 2>/dev/null | grep checkpoint || true
